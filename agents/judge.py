"""
Judge agent (formerly "reflector").

Runs as TWO independent evaluations, used in two separate sequential
loops (see main.py):

  1. review_job() — evaluates ONLY the job the search agent picked:
     is this genuinely the single best match for the candidate, does
     the posting look live and current, does the application URL look
     like a real, working, direct application link. Drives the
     search-agent <-> judge loop (up to MAX_JOB_SEARCH_CYCLES cycles,
     or early approval) that runs BEFORE any resume is written.

  2. review_resume() — evaluates ONLY the drafted resume against the
     (by then already-approved-or-cycle-exhausted) job: fitness score,
     strengths, gaps, concrete improvement suggestions. Drives the
     writer <-> judge loop (up to MAX_RESUME_REVISE_CYCLES cycles, or
     early approval) that runs AFTER the job stage is done.

These used to be one combined call; they're now split because the job
pick should be locked in (or exhausted) before the writer ever starts
drafting — there's no point revising a resume for a job that might
still change.

Both use adaptive thinking at high effort so the judge reasons
carefully rather than pattern-matching a quick verdict — these are
judgment calls, not generation tasks.

Note on "reasoning model": Claude doesn't have a separate reasoning
model line. Instead, Sonnet 5 (and other current models) support
*adaptive thinking* — a mode where the model reasons through a hidden
scratchpad before answering. That's what's used here via
`thinking={"type": "adaptive"}` + `output_config={"effort": "high"}`.
"""

import json
import re
import anthropic

from config import ANTHROPIC_API_KEY, MODEL_JUDGE, JUDGE_APPROVAL_SCORE, RESUME_APPROVAL_SCORE, RESUME_MAX_PAGES, RESUME_PREFERRED_PAGES
from agents.market_context import MARKET_CALIBRATION
from logger_setup import get_logger, note_model

log = get_logger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Matches a fenced code block, optionally tagged ```json ... ``` — models
# occasionally wrap JSON in one of these despite being told not to.
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _parse_review_json(raw_text: str) -> dict | None:
    """
    Try to parse the model's final answer as the review JSON, tolerating
    a couple of common non-conformant shapes (fenced code blocks, or
    leading/trailing prose around the JSON object) before giving up.

    Returns the parsed dict, or None if nothing usable could be parsed.
    """
    candidates = [raw_text]

    fence_match = _CODE_FENCE_RE.match(raw_text.strip())
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    brace_start = raw_text.find("{")
    brace_end = raw_text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(raw_text[brace_start : brace_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_text(response, where: str) -> str:
    """
    Pull the answer text out of a response that may also contain thinking
    blocks, and log loudly if it came back empty or truncated.

    With adaptive thinking, response.content holds a "thinking" block (the
    reasoning) followed by the "text" block with the actual answer. Thinking
    tokens count against max_tokens, so too small a budget means the model
    is cut off mid-reasoning and NEVER emits the text block at all — the
    parse then fails on an empty string, which looks like a parser bug but
    isn't. Diagnose that explicitly rather than silently returning "".
    """
    text_blocks = [b.text for b in response.content if b.type == "text"]
    raw_text = "\n".join(text_blocks).strip()

    if not raw_text:
        thinking_blocks = [b for b in response.content if b.type == "thinking"]
        log.error(
            "%s: model returned no text block (stop_reason=%s, %d thinking block(s), "
            "usage=%s). If stop_reason is 'max_tokens', the thinking budget consumed "
            "the entire max_tokens allowance before the JSON answer could be written "
            "— raise max_tokens for this call.",
            where, response.stop_reason, len(thinking_blocks), getattr(response, "usage", None),
        )
    elif response.stop_reason == "max_tokens":
        log.error(
            "%s: response was truncated mid-answer (stop_reason=max_tokens, usage=%s). "
            "The JSON is likely incomplete and will fail to parse — raise max_tokens.",
            where, getattr(response, "usage", None),
        )

    return raw_text


# --- Stage 1: job match review ---

JOB_SYSTEM_PROMPT = f"""You are a skeptical, experienced hiring manager \
acting as a judge over a job-search agent's work. You will be shown a \
candidate's background and the SINGLE job posting a search agent picked as \
its best match. A resume has NOT been written yet — your job right now is \
only to judge whether this was genuinely the right job to target, not to \
evaluate any resume.

{MARKET_CALIBRATION}

Assess:
1. Job match quality on a 1-10 scale. Anchor to these bands and do not \
drift upward — a generous score here is not kindness, it sends the candidate \
after a role they will not be shortlisted for and commits the whole pipeline \
to writing a resume for it:
   10  — the candidate would be a shortlist name: core discipline, primary \
methods, and seniority all align, and their demonstrated work is the work \
the role describes.
   8-9 — strong: the core work aligns and seniority fits; only soft items \
are missing.
   6-7 — plausible but compromised: the core only partly aligns, OR \
seniority is a full band off, OR the day-to-day is meaningfully different \
from what the candidate has actually done.
   4-5 — weak: same field and real keyword overlap, but the actual work \
differs.
   1-3 — wrong job, or a hard blocker applies.
   A hard blocker caps the score at 3 no matter what else lines up. Ask \
explicitly whether one applies before scoring.
2. Job posting trustworthiness: does this look like a live, current posting \
(not stale, filled, or a generic evergreen listing), and does the URL look \
like a real, direct, working application link rather than a broken, \
placeholder, or overly generic careers-homepage URL? Flag anything suspicious.
3. Concerns: any specific reasons to doubt this was the right pick — a \
better angle the search agent likely missed, a seniority or location \
mismatch, a posting that reads as closed/expired despite the recency filter, \
or a link that doesn't look like it would actually let someone apply. Be \
specific enough that the search agent could act on this feedback and find a \
genuinely different, better posting next time. Do not raise soft items as \
concerns — the search agent cannot fix the candidate's background, and \
sending it after a posting with five fewer nice-to-haves wastes a cycle.
4. An explicit approval decision. Set "approved" to true ONLY when all three \
hold: job_match_score is at least {JUDGE_APPROVAL_SCORE}, the application \
link looks real and working, and no hard blocker applies. There is no \
latitude below that bar. In particular, do NOT approve because cycles are \
running out, because the pool looks thin, because the candidate could stretch \
into the role, or because this pick is better than the last one you rejected \
— the pool holds dozens of postings and the loop exists to work down it. If \
nothing clears the bar, leave approved false and say so plainly in \
job_match_summary: the pipeline keeps the best-scoring pick regardless, and \
an honest low score is far more useful to everything downstream than a \
generous approval.

Be honest even when the news isn't good. If you're reviewing a re-pick after \
rejecting an earlier one, say plainly whether this new pick actually resolves \
the concerns you raised before, or whether the search agent missed the point.

Respond ONLY with valid JSON in this exact shape, no other text:
{{
  "job_match_score": <1-10 integer>,
  "job_match_summary": "<1-2 sentence verdict>",
  "job_link_trustworthy": <true/false>,
  "job_concerns": ["<specific, actionable point>", ...],
  "approved": <true/false>
}}"""


def _enforce_job_approval_bar(result: dict) -> dict:
    """
    Hold the approval bar in code, not only in the prompt.

    The prompt used to allow approving "slightly below" the threshold when
    concerns looked market-inherent, and that clause did all the work: the
    judge approved mediocre picks and the search loop exited after one
    cycle without ever working down the pool. A score below the threshold,
    or a link the judge itself called untrustworthy, is not approvable —
    whatever the model wrote in the "approved" field.

    Only downgrades. A model that declines to approve a 9/10 pick had a
    reason, and overriding that would be the same mistake in reverse.
    """
    score = result.get("job_match_score")
    score_value = score if isinstance(score, (int, float)) else 0
    trustworthy = result.get("job_link_trustworthy")

    result.setdefault("approved", score_value >= JUDGE_APPROVAL_SCORE)
    if not result.get("approved"):
        return result

    reason = None
    if score_value < JUDGE_APPROVAL_SCORE:
        reason = (
            f"approval withdrawn: score {score} is below the "
            f"{JUDGE_APPROVAL_SCORE} bar"
        )
    elif trustworthy is False:
        reason = "approval withdrawn: the judge flagged the application link as untrustworthy"

    if reason:
        log.info("    %s", reason)
        result["approved"] = False
        concerns = result.get("job_concerns")
        result["job_concerns"] = (concerns if isinstance(concerns, list) else []) + [reason]

    return result


def review_job(job: dict, candidate_context: str) -> dict:
    """
    Critically review the job the search agent picked — no resume
    involved yet. Drives the search-agent <-> judge loop that runs
    BEFORE the writer starts drafting.

    Args:
        job: The job dict from search_agent.find_best_job().
        candidate_context: The context agent's gathered candidate summary.

    Returns:
        Dict with keys: job_match_score, job_match_summary,
        job_link_trustworthy, job_concerns, approved.
    """
    job_block = (
        f"Job picked by the search agent:\n"
        f"  Title: {job.get('job_title')}\n"
        f"  Company: {job.get('company')}\n"
        f"  Location: {job.get('location')}\n"
        f"  Posting URL: {job.get('url')}\n"
        f"  Posted date: {job.get('posted_date')}\n"
        f"  Search agent's own rationale for this pick: {job.get('match_rationale')}\n"
        f"  Search agent's notes on angles tried: {job.get('search_notes')}\n"
        f"Requirements/responsibilities:\n{job.get('full_requirements')}\n"
    )
    try:
        response = client.messages.create(
            model=MODEL_JUDGE,
            # Generous budget on purpose: adaptive thinking at high effort
            # spends tokens on reasoning BEFORE writing the JSON answer, and
            # thinking counts against max_tokens. The candidate context plus a
            # full job posting easily consumes a small budget entirely, leaving
            # nothing for the answer and producing an empty response that looks
            # like a parse failure. Leave plenty of headroom.
            max_tokens=8192,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=JOB_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Candidate background:\n{candidate_context}\n\n"
                        f"{job_block}\n"
                        "Evaluate whether this was the right job to target now."
                    ),
                }
            ],
        )
    except anthropic.APIError as exc:
        log.error("judge.review_job: Anthropic API call failed: %s", exc, exc_info=True)
        raise RuntimeError(
            f"judge.review_job: failed to reach the Anthropic API ({exc}). "
            "Check your ANTHROPIC_API_KEY and network connection."
        ) from exc

    note_model(log, "job judge", response)
    raw_text = _extract_text(response, "judge.review_job")

    result = _parse_review_json(raw_text)
    if result is not None:
        return _enforce_job_approval_bar(result)

    summary = (
        "Judge returned no answer — the response was cut off before the review "
        "was written (see log for stop_reason/usage)."
        if not raw_text
        else "Could not parse structured review."
    )

    return {
        "job_match_score": None,
        "job_match_summary": summary,
        "job_link_trustworthy": None,
        "job_concerns": [],
        "approved": False,
        "raw_response": raw_text,
    }


# --- Stage 2: resume review ---

RESUME_SYSTEM_PROMPT = f"""You are a skeptical, experienced hiring manager and \
resume reviewer, acting as the final judge before a resume goes out the door. \
You will be shown a tailored resume and the specific job it was written for \
(that job has already been separately vetted and approved — you do not need \
to re-judge the job itself here, only the resume). Your job is to critically \
evaluate whether this resume would actually help the candidate get an \
interview — not to be encouraging for its own sake.

{MARKET_CALIBRATION}

Apply that calibration here too. The job is already locked in, and \
whether this candidate should apply is a settled question you are not \
reopening. Your subject is the DRAFT.

SCORE THE DRAFT, NOT THE CANDIDATE. The only thing another revision \
cycle can change is the writing, so score against the ceiling this \
candidate's real background sets: of everything their background could \
honestly show for this job, how much has this draft actually surfaced, \
and how well is it framed in the posting's terms? A draft that extracts \
everything available scores 9-10 EVEN IF the candidate covers only part \
of what the posting asks for. Do NOT deduct for requirements the \
background cannot evidence — that was priced in when the job was \
approved, the candidate cannot acquire the experience between cycles, \
and the only way a writer can respond to such a deduction is by \
overclaiming, which is the thing you exist to catch.

You may be shown the earlier stage's job review, listing limitations \
already known and accepted when this job was chosen. Those are settled. \
Do not re-raise them as gaps, do not score against them, and do not \
withhold approval over them.

THE TRAILING "NOTES" SECTION IS EXPECTED. The writer is instructed to \
end its output with a section headed "Notes (not part of the resume)", \
listing real gaps between the candidate and the posting. That section is \
DELIBERATE and is stripped automatically before the PDF is rendered — it \
never reaches an employer, and it is displayed separately for the \
candidate as interview preparation. Do NOT treat its presence as \
contamination, a formatting defect, or a template-fidelity failure; do \
NOT deduct for it; do NOT tell the writer to remove it; and do NOT \
withhold approval over it. A draft is submission-ready WITH that section \
attached. Read it — it tells you whether the writer flagged gaps honestly \
rather than burying them, which is a point in the draft's favour — but \
judge the resume body above it. The one thing worth raising about Notes \
is if the section is truncated mid-sentence, which means the response was \
cut off and the resume body may be incomplete too.

Assess:
1. Fitness of the draft, 1-10, on the ceiling-relative basis described \
above — how completely and how well this draft presents the candidate's \
real background for this job.
2. Strengths: what genuinely works well and should stay.
3. Gaps or weaknesses: missing keywords, vague or unquantified claims, \
misaligned emphasis, anything an ATS or a skimming recruiter would flag. \
Weight these by how CENTRAL they are to the role, per the calibration above: \
reserve real weight for the role's core requirements — the discipline, the \
methods, the seniority, the primary technical skills the job is actually \
about.

The "gaps" list drives another revision cycle, so put in it ONLY what the \
writer can actually act on using the candidate's real material: content the \
background supports but the draft buried, omitted, left vague, or framed in \
the wrong terms. A gap the candidate's background genuinely cannot close is \
not a writing problem — state it once in fitness_summary and leave it out of \
"gaps", because listing it there sends the writer back to rewrite something \
no rewrite can fix, and the only way it can comply is by overclaiming, which \
is exactly what you are here to catch. Never raise the same soft item across \
multiple cycles, and never let one hold back approval on its own.

Do NOT comment on visual formatting — spacing, margins, alignment, fonts, \
line breaks, or how the page looks. You are shown Markdown, and all of that \
is decided later by a stylesheet the writer does not control. Judge \
STRUCTURE (which sections exist, their order and naming, what content sits \
under each, entry consistency) and CONTENT. Flagging "tighten the spacing" \
or "align the dates" sends the writer chasing something it cannot change.
4. Length and template fidelity: estimate whether this resume would run \
longer than {RESUME_MAX_PAGES} pages once rendered (roughly 900-1100 words \
is the {RESUME_MAX_PAGES}-page ceiling, ~600-800 words is the \
{RESUME_PREFERRED_PAGES}-page target — use word count as a rough proxy, not \
a precise measure). If it's likely over {RESUME_MAX_PAGES} pages, this is a \
concrete gap the writer must fix by cutting lower-value content, not a minor \
note. If a TEMPLATE RESUME is supplied, compare the draft against it and \
flag STRUCTURAL deviations as gaps: different section headers or wording, \
different section order, a different date format, inconsistent entry shapes \
between sections, a much longer or shorter summary, or a different Skills \
organization. Also flag internal sloppiness: a section label repeated as the \
first words of its own content, or items filed under a heading they don't \
belong to. Structure and wording only — not how it will look on the page.
5. Concrete improvement suggestions: specific, actionable edits — not generic \
advice like "add more detail."
6. An explicit approval decision: set "approved" to true if this draft is \
the best honest resume this candidate's background can produce for this \
job — within the page cap, faithful to any style template given, with no \
further meaningful improvement available from another revision cycle. A \
draft scoring {RESUME_APPROVAL_SCORE} or higher should usually be \
approved. If your ONLY remaining reservations are things the candidate's \
background cannot evidence, APPROVE and state the limitation in \
fitness_summary — withholding approval there costs cycles and changes \
nothing, because the next draft will have exactly the same limitation. \
Withhold approval only for something a rewrite can actually fix: content \
the background supports but the draft omits or buries, overclaiming, \
running over the page cap, or a structural break from the template.

Be honest even when the news isn't good. A resume that overclaims or is \
poorly matched to the role does the candidate no favors if you sugarcoat it. \
If you've reviewed a previous version of this resume, say plainly whether \
this revision actually improved on the specific issues you raised before, or \
whether the writer missed the point.

Respond ONLY with valid JSON in this exact shape, no other text:
{{
  "fitness_score": <1-10 integer>,
  "fitness_summary": "<1-2 sentence verdict>",
  "strengths": ["<point>", ...],
  "gaps": ["<point>", ...],
  "suggestions": ["<specific, actionable edit>", ...],
  "approved": <true/false>
}}"""


def _known_limitations_block(job_review: dict | None) -> str:
    """
    Hand the resume judge what the JOB judge already found and accepted.

    Stage 1 scores candidate-against-job and only lets a pick through at a
    high bar, so by the time a draft exists, any remaining shortfall has
    been seen and deemed acceptable. Without this block the resume judge
    rediscovers those same shortfalls from scratch and treats them as
    fresh findings against the draft — rejecting a resume for a gap that
    was priced in before the writer was ever invoked, and that no revision
    can close.
    """
    if not job_review:
        return ""
    concerns = job_review.get("job_concerns")
    summary = job_review.get("job_match_summary")
    if not concerns and not summary:
        return ""

    lines = [
        "=== ALREADY SETTLED AT JOB-SELECTION STAGE ===",
        "These were weighed when this job was approved. They are known, "
        "accepted limitations of the candidate's background — not findings "
        "against the draft. Do not re-raise them as gaps or score against them.",
    ]
    if summary:
        lines.append(f"Job-match verdict ({job_review.get('job_match_score')}/10): {summary}")
    if isinstance(concerns, list) and concerns:
        lines.extend(f"- {concern}" for concern in concerns)
    lines.append("=== END SETTLED ITEMS ===\n")
    return "\n".join(lines) + "\n"


def review_resume(
    resume_draft: str,
    candidate_context: str,
    job: dict,
    style_template: str | None = None,
    job_review: dict | None = None,
) -> dict:
    """
    Critically review a drafted resume against the (already-approved)
    target job. Drives the writer <-> judge loop that runs AFTER the
    job-search stage is done.

    Args:
        resume_draft: The writer agent's output (Markdown resume) for
            this cycle.
        candidate_context: The context agent's gathered candidate summary
            — gives the judge the same grounding the writer had.
        job: The job dict from search_agent.find_best_job() (already
            vetted by review_job() in the prior pipeline stage).
        style_template: Raw text of the template resume, when one exists.
            Given to the judge for the same reason it's given to the
            writer — template fidelity can only be checked against the
            actual template, not a description of it.
        job_review: The stage-1 verdict on this job, if available. Its
            concerns are passed through as settled items so this stage
            doesn't re-litigate limitations already accepted.

    Returns:
        Dict with keys: fitness_score, fitness_summary, strengths,
        gaps, suggestions, approved.
    """
    job_block = (
        f"Target job: {job.get('job_title')} at {job.get('company')} "
        f"({job.get('location')})\n"
        f"Requirements/responsibilities:\n{job.get('full_requirements')}\n"
    )
    template_block = ""
    if style_template and style_template.strip():
        template_block = (
            "=== TEMPLATE RESUME the draft was supposed to follow "
            "(structure only — its facts belong to someone else) ===\n"
            f"{style_template}\n"
            "=== END TEMPLATE RESUME ===\n\n"
        )
    try:
        response = client.messages.create(
            model=MODEL_JUDGE,
            # See the note in review_job: thinking tokens count against
            # max_tokens. This call carries an entire resume draft on top of
            # the context and job, so it needs even more headroom.
            max_tokens=12000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=RESUME_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{template_block}"
                        f"{_known_limitations_block(job_review)}"
                        f"Candidate background:\n{candidate_context}\n\n"
                        f"{job_block}\n"
                        f"Drafted resume to review:\n{resume_draft}\n\n"
                        "Evaluate this resume now."
                    ),
                }
            ],
        )
    except anthropic.APIError as exc:
        log.error("judge.review_resume: Anthropic API call failed: %s", exc, exc_info=True)
        raise RuntimeError(
            f"judge.review_resume: failed to reach the Anthropic API ({exc}). "
            "Check your ANTHROPIC_API_KEY and network connection."
        ) from exc

    note_model(log, "resume judge", response)
    raw_text = _extract_text(response, "judge.review_resume")

    result = _parse_review_json(raw_text)
    if result is not None:
        result.setdefault("approved", result.get("fitness_score", 0) >= RESUME_APPROVAL_SCORE)
        return result

    # Model occasionally wraps JSON in prose despite instructions, or the
    # response gets cut off mid-object — fall back to returning the raw
    # text so nothing is silently lost. approved=False so the loop treats
    # this cycle as needing another pass rather than silently exiting.
    summary = (
        "Judge returned no answer — the response was cut off before the review "
        "was written (see log for stop_reason/usage)."
        if not raw_text
        else "Could not parse structured review."
    )

    return {
        "fitness_score": None,
        "fitness_summary": summary,
        "strengths": [],
        "gaps": [],
        "suggestions": [],
        "approved": False,
        "raw_response": raw_text,
    }
