"""
Work out what level the candidate is actually at, once per run.

WHY THIS EXISTS. Every stage of this pipeline was making its own implicit
guess about seniority from the same free-text background, and they did
not agree. The planner read "8 years of experience" and planned angles
for senior roles; the screener read the same text and rated a staff role
8/10; the judge then rejected it on seniority three cycles later. Nothing
was wrong with any one of those calls — they just had no shared answer to
"what jobs would actually shortlist this person".

This produces that answer once, as structured data, and every stage is
given it. One judgement, made deliberately, instead of four made in
passing.

WHAT IT OPTIMIZES FOR. Getting hired soon, not getting hired at the
highest possible title. That means it is as interested in OVER-
qualification as under: a candidate applying below their band is screened
out as a flight risk, and the application is as wasted as an overreach.
It also names the hurdles — a gap, a pivot, a non-standard degree, a
recency problem — because those shape which roles are winnable, and a
pipeline that doesn't know about them keeps targeting roles where they
are disqualifying.

HONESTY. The bands come from the candidate's documents and nothing else.
This does not flatter and does not manufacture a hurdle to look rigorous.
An empty hurdles list is a legitimate output.

Runs on the planner's model: a structured reading task over text that is
already in hand, not an evaluation.
"""

import json
import re

import anthropic

from config import ANTHROPIC_API_KEY, MODEL_PLANNER
from logger_setup import get_logger, note_model

log = get_logger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)

BANDS = ["intern", "junior", "mid", "senior", "lead", "principal"]

SYSTEM_PROMPT = """You are calibrating a job search. Given a candidate's \
background, decide what level of role they would actually be SHORTLISTED \
for right now, and what would get in the way.

The goal is a job soon, not an impressive title. Two failure modes cost \
the candidate equally, and you are guarding against both:
- Reaching too high: applications to roles they'd be screened out of.
- Aiming too low: employers reject candidates who are clearly over-\
qualified, because they read as a flight risk or a salary mismatch. A \
candidate with 8 years of delivery is not a credible junior hire.

Judge the band on DEMONSTRATED scope, not on years elapsed. Years set a \
ceiling, not the level. What moves someone up a band: owning a system end \
to end, deciding the approach rather than executing one, work others \
depended on, scale, and formal leadership. Someone with eight years of \
executing well-specified tasks is mid, not senior. Someone with four years \
who owned production systems may be senior.

Bands, lowest to highest: intern, junior, mid, senior, lead, principal.

HURDLES are real, checkable facts in the background that narrow which \
roles are winnable. Only list what is actually there. Typical ones: an \
employment gap and its length; a career change into this field; a degree \
that is not the one postings ask for; experience concentrated in one \
industry or one employer; a long stretch in a title below their demonstrated \
level; skills evidenced years ago but not recently; work authorization \
limits; being newly arrived in a country where employers want local \
experience. For each, say plainly what it rules out or makes harder, and \
the most credible way to work around it in a search. Do not invent a hurdle \
to seem thorough, and do not soften a real one.

TITLES. Give the titles EMPLOYERS use for roles this person would be \
shortlisted for — not the candidate's own past titles, and not aspirational \
ones. Also name titles to avoid: roles their background superficially \
matches but that would reject them, in either direction.

Respond with ONLY a JSON object, no other text:
{
  "band": "<one of: intern, junior, mid, senior, lead, principal>",
  "target_bands": ["<the 1-3 bands worth applying to, best first>"],
  "years_relevant": <number — years in the field they are targeting, not total>,
  "evidence": "<one sentence: what puts them at this band>",
  "titles_to_target": ["<4-8 employer-side titles>"],
  "titles_to_avoid": ["<titles that would reject them, over or under>"],
  "hurdles": [
    {"issue": "<the fact>", "impact": "<what it rules out or makes harder>",
     "approach": "<how the search should handle it>"}
  ],
  "search_guidance": "<2-3 sentences: how these facts should shape which \
postings are targeted>"
}"""


def _parse_json(raw_text: str) -> dict | None:
    candidates = [raw_text]
    fence = _CODE_FENCE_RE.match(raw_text.strip())
    if fence:
        candidates.append(fence.group(1).strip())
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw_text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def assess_level(candidate_context: str, target_role_description: str = "") -> dict | None:
    """
    Produce the candidate's level profile, or None if it can't be built.

    None is a supported outcome, not an error. Every consumer treats a
    missing profile as "carry on as before" — the pipeline worked without
    this for its whole life, and a failed calibration call must not stop a
    run or silently narrow a search.

    Args:
        target_role_description: What the user typed. Passed as context so
            the profile can speak to it, but it does NOT constrain the
            answer — the whole point is to tell the user when what they
            typed is off their actual level.
    """
    try:
        response = client.messages.create(
            model=MODEL_PLANNER,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Candidate background:\n{candidate_context}\n\n"
                        f"What they typed as a target role: "
                        f"{target_role_description or '(nothing specific)'}\n\n"
                        "Calibrate their level now."
                    ),
                }
            ],
        )
    except anthropic.APIError as exc:
        log.error("level_agent.assess_level: Anthropic API call failed: %s", exc, exc_info=True)
        log.info("  Warning: could not calibrate the candidate's level (%s). Continuing without it.", exc)
        return None

    note_model(log, "level", response)
    raw_text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    profile = _parse_json(raw_text)

    if not profile or not profile.get("band"):
        log.error(
            "level_agent.assess_level: could not parse the level profile "
            "(stop_reason=%s). Raw: %s",
            getattr(response, "stop_reason", "unknown"), raw_text[:400],
        )
        log.info("  Warning: level calibration returned nothing usable. Continuing without it.")
        return None

    band = str(profile.get("band", "")).strip().lower()
    if band not in BANDS:
        # A band outside the vocabulary would silently mean nothing to
        # every consumer downstream; better to drop the field than to
        # pass "very senior" around as if it were comparable.
        log.error("level_agent: unrecognized band %r — dropping it.", band)
        profile["band"] = None
    else:
        profile["band"] = band

    return profile


def level_block(profile: dict | None) -> str:
    """
    Render the profile for a prompt, or "" when there is none.

    One shared renderer so the planner, screener, judge and writer are all
    reading the SAME words. When each stage formatted its own view of this,
    they drifted — which is the problem this module exists to end.
    """
    if not profile:
        return ""

    lines = ["=== CANDIDATE LEVEL (calibrated for this run) ==="]
    band = profile.get("band")
    years = profile.get("years_relevant")
    if band:
        lines.append(f"Level: {band}" + (f" — about {years} years of relevant experience" if years else ""))
    if profile.get("evidence"):
        lines.append(f"Why: {profile['evidence']}")

    targets = profile.get("target_bands")
    if isinstance(targets, list) and targets:
        lines.append(f"Apply at these levels: {', '.join(str(t) for t in targets)}")

    titles = profile.get("titles_to_target")
    if isinstance(titles, list) and titles:
        lines.append(f"Titles that would shortlist them: {', '.join(str(t) for t in titles)}")

    avoid = profile.get("titles_to_avoid")
    if isinstance(avoid, list) and avoid:
        lines.append(
            f"Titles to avoid (rejected as over- or under-qualified): "
            f"{', '.join(str(t) for t in avoid)}"
        )

    hurdles = profile.get("hurdles")
    if isinstance(hurdles, list) and hurdles:
        lines.append("Hurdles in this candidate's background:")
        for hurdle in hurdles:
            if not isinstance(hurdle, dict):
                continue
            issue = hurdle.get("issue")
            if not issue:
                continue
            lines.append(
                f"  - {issue} \u2192 {hurdle.get('impact', 'affects which roles are winnable')}"
                + (f" | handle by: {hurdle['approach']}" if hurdle.get("approach") else "")
            )

    if profile.get("search_guidance"):
        lines.append(f"How this should shape the search: {profile['search_guidance']}")

    lines.append(
        "Treat this as the calibration for every decision below. Aiming above "
        "it wastes the application on a screening rejection; aiming below it "
        "wastes it on an over-qualification rejection. Both cost the same."
    )
    lines.append("=== END CANDIDATE LEVEL ===\n")
    return "\n".join(lines) + "\n"


def log_profile(profile: dict | None) -> None:
    """Narrate the profile in the run log, where the user will see it."""
    if not profile:
        return
    log.info(
        "  Calibrated level: %s%s — %s",
        profile.get("band") or "unknown",
        f" (~{profile['years_relevant']}y relevant)" if profile.get("years_relevant") else "",
        profile.get("evidence") or "",
    )
    targets = profile.get("titles_to_target")
    if isinstance(targets, list) and targets:
        log.info("    Targeting titles: %s", ", ".join(str(t) for t in targets))
    avoid = profile.get("titles_to_avoid")
    if isinstance(avoid, list) and avoid:
        log.info("    Avoiding: %s", ", ".join(str(t) for t in avoid))
    hurdles = profile.get("hurdles")
    if isinstance(hurdles, list):
        for hurdle in hurdles:
            if isinstance(hurdle, dict) and hurdle.get("issue"):
                log.info("    Hurdle: %s — %s", hurdle["issue"], hurdle.get("impact", ""))
