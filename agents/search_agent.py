"""
Search agent.

Two phases, and only the first one calls a model to make judgements:

  1. build_job_pool() — ONE scraping pass per run. An LLM plans several
     complementary search angles, those angles are executed
     deterministically against JobSpy, and the deduplicated results are
     narrowed in three stages:

         scrape  -> up to JOB_SCRAPE_CEILING postings
         BM25    -> lexical rank against the resume, keep BM25_KEEP
         screen  -> a model rates each survivor 1-10, drops the duds

     What survives is sorted by rating, with BM25 rank breaking ties, and
     ALL of it is kept. There is no target pool size: nothing reads the
     pool wholesale, so truncating a ranked list only costs depth on the
     last judge cycle — precisely when depth is needed.

  2. next_candidate() — runs once per judge cycle. No model, no scraping.
     It walks the ranking and returns the next posting not already
     rejected, with its full description attached.

Why phase 2 is not an agent any more. It used to be a tool-use loop that
queried the pool, read a handful of postings and picked one. Given a pool
that is already rated and ranked, that loop was re-deriving — three times
per run, with no memory between cycles — an ordering the screener had
already computed once. Worse, its two most important behaviours ("read
the full description before committing", "never re-recommend a rejected
posting") were prompt instructions a model could silently skip. Both are
now code: the description is attached unconditionally and rejected URLs
are filtered by set membership.

What that removed, honestly: the agent could respond to a judge concern
by jumping to a DIFFERENT KIND of posting rather than merely the next one
down, and it had a one-per-run escape hatch to scrape again when the pool
genuinely could not satisfy a concern. Both are gone. The escape hatch is
the bigger loss — a pool that was wrong from the start now stays wrong —
but it fired rarely and cost a full scrape when it did. If rejections
turn out to cluster on one reason (always seniority, always location),
the right fix is a reason-driven skip in next_candidate(), not restoring
the agent.

Planning and screening run on Haiku 4.5 via their own config constants
(MODEL_PLANNER, MODEL_SCREENER) — different jobs with different stakes,
and the screener in particular makes permanent drops. Both are mechanical
next to the judge's evaluation work, which stays on Sonnet 5.

The screener shares agents/market_context.py with the judge, so the two
weigh a posting's requirements the same way. When they diverged, the loop
burned cycles: the pool would surface a posting the judge then rejected
on a requirement the screener never weighted.
"""

import json
import re
import anthropic

from config import (
    ANTHROPIC_API_KEY,
    MODEL_PLANNER,
    MODEL_SCREENER,
    JOB_SEARCH_SITES,
    JOB_SEARCH_HOURS_OLD,
    JOB_SCRAPE_CEILING,
    BM25_KEEP,
    POOL_SCREEN_MIN_FIT,
    POOL_SCREEN_BATCH_SIZE,
    POOL_SCREEN_SNIPPET_CHARS,
    JOBSPY_FETCH_LINKEDIN_DESCRIPTIONS,
    SKIP_PREVIOUSLY_SELECTED,
)
from agents.market_context import MARKET_CALIBRATION
from job_history import filter_seen
from tools.jobspy_tool import search as jobspy_search_fn
from tools.firecrawl_tool import scrape as firecrawl_scrape_fn
from tools.bm25_tool import rank_pool
from logger_setup import get_logger, note_model

log = get_logger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _parse_json(raw_text: str) -> dict | list | None:
    """
    Parse the model's answer as JSON, tolerating fenced code blocks and
    stray prose around the object/array before giving up.
    """
    candidates = [raw_text]

    fence_match = _CODE_FENCE_RE.match(raw_text.strip())
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = raw_text.find(opener), raw_text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append(raw_text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------
# Phase 1a: plan the angles
# ---------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = f"""You are planning a job search. You get ONE \
scraping pass for this entire run, so the angles you choose determine \
everything the rest of the pipeline will have to work with. There is no \
second chance to broaden later.

{MARKET_CALIBRATION}

Plan only for roles the candidate could realistically be SHORTLISTED for: \
the core discipline they have actually done, at a seniority within one band \
of where they sit, and consistent with their hard constraints (work \
authorization, where they can be, whether they said they can relocate or \
work remotely). An angle that returns roles they'd be screened out of costs \
the same as a good one and yields nothing. Do not build an angle around a \
skill they have touched once, or a title two levels above their experience.

Design 2-6 COMPLEMENTARY search angles from the candidate's background and \
the target role hint. Complementary means each angle should surface postings \
the others would miss:
- Vary the job TITLE across angles. Employers title the same work differently \
("Machine Learning Engineer", "Data Scientist", "Applied Scientist", "NLP \
Engineer", "MLOps Engineer"). Use the titles EMPLOYERS use, not the \
candidate's own job title.
- Vary seniority where the candidate's experience genuinely spans it (e.g. \
both "Senior Data Scientist" and "Data Scientist" if they'd be credible for \
both). Do not reach for levels their background doesn't support.
- Vary LOCATION deliberately. If the candidate can relocate or work remotely, \
include both their home city and the country's main job markets as separate \
angles. Omit location entirely on at least one angle for nationwide coverage.
- Consider one angle led by a distinctive, in-demand SKILL rather than a \
title, when the candidate has one that employers search on.
Do NOT submit near-duplicate angles (e.g. "Data Scientist" and "Data \
Scientist role") — they return the same postings and waste the pass.

search_term matches against the POSTING, not against the candidate. A term \
they would never put on their own resume is still right if that's how \
employers title the role.

Set results_wanted per angle generously. Everything scraped is ranked and \
screened afterwards, so a wide net costs little and a narrow one permanently \
limits what can be chosen from — expect heavy overlap between angles in the \
same city.

Respond with ONLY a JSON array, no other text. Each element:
{{
  "search_term": "<string, required>",
  "location": "<string or null for a nationwide search>",
  "is_remote": <true/false>,
  "job_type": "<fulltime|parttime|contract|internship, or null>",
  "results_wanted": <integer, 10-30>,
  "rationale": "<short note on what this angle is meant to catch>"
}}"""


def plan_search_angles(candidate_context: str, target_role_description: str) -> list[dict]:
    """
    Ask the model for a set of complementary JobSpy search angles.

    Returns a list of angle dicts. Falls back to a single angle built
    from the role hint if planning fails — a degraded pool beats no pool,
    and the failure is logged either way.
    """
    try:
        response = client.messages.create(
            model=MODEL_PLANNER,
            max_tokens=4096,
            system=PLANNER_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Candidate background:\n{candidate_context}\n\n"
                        f"Target role hint: {target_role_description}\n\n"
                        "Plan the search angles now."
                    ),
                }
            ],
        )
    except anthropic.APIError as exc:
        log.error("search_agent.plan_search_angles: Anthropic API call failed: %s", exc, exc_info=True)
        raise RuntimeError(
            f"search_agent: failed to reach the Anthropic API while planning angles ({exc}). "
            "Check your ANTHROPIC_API_KEY and network connection."
        ) from exc

    note_model(log, "planner", response)
    raw_text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    angles = _parse_json(raw_text)

    if not isinstance(angles, list) or not angles:
        log.error(
            "search_agent.plan_search_angles: could not parse angles (stop_reason=%s). "
            "Falling back to a single angle from the role hint. Raw: %s",
            response.stop_reason, raw_text[:500],
        )
        return [{"search_term": target_role_description, "location": None, "results_wanted": 30}]

    return [a for a in angles if isinstance(a, dict) and a.get("search_term")]


# ---------------------------------------------------------------------
# Phase 1b: screen the BM25 survivors
# ---------------------------------------------------------------------

SCREENER_SYSTEM_PROMPT = f"""You are screening scraped job postings for one \
candidate. You will be given a batch of postings — title, company, location, \
and the opening of the description — each with an index. Rate how well each \
ONE fits this specific candidate.

{MARKET_CALIBRATION}

WHAT TO WEIGHT INSIDE A POSTING. Postings are not uniform, and the section a \
statement appears in changes how much it matters. Headings vary and are often \
absent, so go by what the text is DOING, not by what it is labelled:
- Weight most heavily the employer's stated requirements — whatever reads as \
"what we're looking for", "qualifications", "who you are", "the ideal \
candidate", or simply a list of demands. This is the filter the candidate \
would actually be screened against.
- Weight moderately what the role actually involves day to day — the \
responsibilities or "what you'll be doing". Do not skip this: it is how you \
catch a posting whose requirements list the candidate's whole stack while \
the actual work is a different discipline.
- Weight at or near zero: benefits, salary ranges, company mission and \
culture blurbs, EEO statements, application instructions, and anything \
explicitly marked "nice to have", "bonus", or "preferred". The employer has \
already told you the last group is optional.

Rate 1-10:
  10  — the candidate would be a shortlist name: core discipline, methods \
and seniority all align, and their demonstrated work is the work described.
  8-9 — strong: core work aligns and seniority fits; only soft items missing.
  6-7 — good but imperfect: core work aligns, but seniority sits at the edge \
of a band, or part of the day-to-day is unfamiliar.
  5   — plausible: same core discipline, but seniority is a band off or the \
day-to-day is substantially different.
  3-4 — weak: same broad field, real keyword overlap, different actual work.
  1-2 — wrong role, or a hard blocker applies.

Use the full range. Do not cluster everything on one or two values — the \
ratings are sorted afterwards, and a batch rated all-8 carries no more \
information than a batch rated all-5.

You are working from the opening of a posting, not all of it, so judge what \
the role IS — the discipline and the level — rather than counting keywords. \
When the text is too thin to tell, rate 5 and say so; a posting wrongly \
dropped here is gone for the whole run, while a wrongly kept one just gets \
examined more closely later. Be decisive about the clear cases: a different \
discipline, or several seniority bands off, should get 1-3 rather than a \
cautious 5.

Rate every posting in the batch. Respond with ONLY a JSON array, no other \
text:
[{{"i": <index>, "fit": <1-10>, "why": "<max 12 words>"}}, ...]"""


def _screen_batch(
    batch: list[tuple[int, dict]],
    candidate_context: str,
    target_role_description: str,
) -> dict[int, dict]:
    """
    Rate one batch of postings. Returns {pool_index: {"fit": int, "why": str}}.

    Returns {} on any failure — API error, unparseable response. The caller
    treats an unrated posting as "keep, unrated" rather than dropping it:
    losing the screening signal degrades pool quality, but dropping real
    postings because a screening call failed would be worse.
    """
    listing = [
        {
            "i": index,
            "title": record.get("title"),
            "company": record.get("company"),
            "location": record.get("location"),
            "is_remote": record.get("is_remote"),
            "snippet": str(
                record.get("description_full") or record.get("description_snippet") or ""
            )[:POOL_SCREEN_SNIPPET_CHARS],
        }
        for index, record in batch
    ]

    try:
        response = client.messages.create(
            model=MODEL_SCREENER,
            max_tokens=4096,
            system=SCREENER_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Candidate background:\n{candidate_context}\n\n"
                        f"Target role hint: {target_role_description}\n\n"
                        f"Postings to rate:\n{json.dumps(listing, default=str)}\n\n"
                        "Rate every posting now."
                    ),
                }
            ],
        )
    except anthropic.APIError as exc:
        log.error("search_agent._screen_batch: Anthropic API call failed: %s", exc, exc_info=True)
        return {}

    note_model(log, "screener", response)
    raw_text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    parsed = _parse_json(raw_text)
    if not isinstance(parsed, list):
        log.error(
            "search_agent._screen_batch: could not parse ratings (stop_reason=%s). Raw: %s",
            response.stop_reason, raw_text[:300],
        )
        return {}

    ratings = {}
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry["i"])
            fit = int(entry["fit"])
        except (KeyError, TypeError, ValueError):
            continue
        # Nothing structurally constrains the model's output to 1-10, so clamp.
        ratings[index] = {"fit": max(1, min(10, fit)), "why": str(entry.get("why") or "")}
    return ratings


def screen_pool(
    pool: list[dict],
    candidate_context: str,
    target_role_description: str,
) -> list[dict]:
    """
    Rate every posting against the candidate, drop everything below
    POOL_SCREEN_MIN_FIT, and return the survivors in selection order.

    This is the stage that makes the pool's ordering mean something. BM25
    got the postings into roughly the right field; this reads them and
    catches what lexical scoring structurally cannot — seniority band,
    work authorization, licence requirements, employment type, and roles
    that share the candidate's vocabulary while being a different job.

    Each posting is rated INDEPENDENTLY against the rubric, never against
    the others in its batch. That keeps ratings comparable across batches
    and across cycles: a strong posting surrounded by duds doesn't score
    inflated, and the "N rated 8+" log line says something about the
    market rather than about batch composition.

    Sort order is (rating desc, BM25 rank asc). The rating dominates, so
    a posting BM25 ranked 40th and the screener rated 9 sits above one
    BM25 ranked 1st and rated 6. BM25 only breaks ties — which it will do
    often, since an integer rating clusters.

    Degrades gracefully: postings the screener didn't rate are kept and
    ranked below every rated one. A failed screening pass costs pool
    quality, never postings.
    """
    if not pool:
        return pool

    log.info("  Screening %d posting(s) for fit...", len(pool))

    ratings: dict[int, dict] = {}
    indexed = list(enumerate(pool))
    for start in range(0, len(indexed), POOL_SCREEN_BATCH_SIZE):
        batch = indexed[start : start + POOL_SCREEN_BATCH_SIZE]
        ratings.update(_screen_batch(batch, candidate_context, target_role_description))

    if not ratings:
        log.error("search_agent.screen_pool: no postings could be rated; keeping the pool unscreened.")
        log.info("  Warning: screening produced no ratings — using the BM25 order instead.")
        return pool

    kept, dropped, unrated = [], 0, 0
    for index, record in indexed:
        rating = ratings.get(index)
        if rating is None:
            record["fit_rating"] = None
            record["fit_note"] = "not screened"
            unrated += 1
            kept.append(record)
            continue
        if rating["fit"] < POOL_SCREEN_MIN_FIT:
            dropped += 1
            continue
        record["fit_rating"] = rating["fit"]
        record["fit_note"] = rating["why"]
        kept.append(record)

    # Rating first, BM25 rank as tie-break. Unrated records sort last
    # (rating 0) but keep their BM25 position relative to each other.
    kept.sort(key=lambda r: (-(r.get("fit_rating") or 0), r.get("bm25_rank") or 10**6))

    log.info(
        "  Screened: %d dropped below fit %d, %d unrated, %d kept.",
        dropped, POOL_SCREEN_MIN_FIT, unrated, len(kept),
    )
    strong = sum(1 for r in kept if (r.get("fit_rating") or 0) >= 8)
    log.info("  %d posting(s) rated 8+ for fit.", strong)
    if strong == 0:
        log.info(
            "  Note: nothing in the pool rates as a strong match. The judge's bar "
            "is unlikely to be met this run — consider a different target role hint."
        )
    if kept:
        log.info(
            "    top of the pool: %s at %s (fit %s, BM25 rank %s)",
            kept[0].get("title"), kept[0].get("company"),
            kept[0].get("fit_rating"), kept[0].get("bm25_rank"),
        )
    return kept


# ---------------------------------------------------------------------
# Phase 1c: build the pool
# ---------------------------------------------------------------------

def build_job_pool(
    candidate_context: str,
    target_role_description: str,
) -> list[dict]:
    """
    Run the single scraping pass for this run and return a ranked,
    screened, deduplicated pool.

    Angles are planned by a model but executed here deterministically, so
    the pass can't be cut short by a model deciding it has enough.
    Deduplication is on job_url.

    Scraping deliberately over-collects to JOB_SCRAPE_CEILING so that
    EVERY planned angle runs. Stopping at a target size, as this once
    did, meant pool membership was decided by angle ordering rather than
    by fit.
    """
    angles = plan_search_angles(candidate_context, target_role_description)
    log.info("  Planned %d search angle(s):", len(angles))
    for angle in angles:
        log.info(
            "    - %r%s%s",
            angle.get("search_term"),
            f" near {angle['location']}" if angle.get("location") else " (nationwide)",
            " [remote]" if angle.get("is_remote") else "",
        )

    pool: list[dict] = []
    seen_urls: set[str] = set()

    for angle in angles:
        if len(pool) >= JOB_SCRAPE_CEILING:
            log.info(
                "  Scrape ceiling of %d reached — skipping remaining angles.",
                JOB_SCRAPE_CEILING,
            )
            break

        results = jobspy_search_fn(
            search_term=angle["search_term"],
            location=angle.get("location"),
            site_name=JOB_SEARCH_SITES,
            is_remote=bool(angle.get("is_remote", False)),
            hours_old=JOB_SEARCH_HOURS_OLD,
            results_wanted=int(angle.get("results_wanted") or 20),
            job_type=angle.get("job_type"),
            linkedin_fetch_description=JOBSPY_FETCH_LINKEDIN_DESCRIPTIONS,
            keep_full_description=True,
        )

        added = 0
        for record in results:
            if record.get("error") or record.get("note"):
                continue
            url = record.get("job_url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            pool.append(record)
            added += 1
        log.info("    -> %d new posting(s) added (pool now %d)", added, len(pool))

    log.info("  Scraped %d unique posting(s) across all angles.", len(pool))

    # Before ranking and screening, not after: a posting already targeted
    # in an earlier run is dead weight either way, and dropping it here
    # means neither stage spends anything on it.
    if SKIP_PREVIOUSLY_SELECTED:
        pool = filter_seen(pool)

    if not pool:
        return pool

    # The candidate summary stands in for the resume as the BM25 query. It
    # is the same text every other agent reasons over, so a posting that
    # ranks well here is one that matches what the pipeline believes about
    # the candidate — including anything the context agent dropped.
    pool = rank_pool(pool, candidate_context, BM25_KEEP)

    return screen_pool(pool, candidate_context, target_role_description)


# ---------------------------------------------------------------------
# Phase 2: walk the ranking (no model, no scraping)
# ---------------------------------------------------------------------

def _full_requirements(record: dict) -> str | None:
    """
    The posting's real text, for the judge and the writer.

    Attached unconditionally in code. This used to be a prompt instruction
    to the selection agent ("you MUST call get_full_description"), which
    meant a skipped call silently handed both downstream stages a snippet
    to reason over, with nothing flagging it.
    """
    full = record.get("description_full") or record.get("description_snippet")
    if full:
        return full

    # JobSpy occasionally returns a posting with no description at all;
    # Firecrawl can sometimes still reach the page.
    url = record.get("job_url")
    log.info("    no stored description for %s — trying Firecrawl", url)
    scraped = firecrawl_scrape_fn(url) if url else None
    if not scraped or str(scraped).strip().startswith("[scrape"):
        log.error("search_agent: no description available for %s", url)
        return None
    return scraped


def next_candidate(
    pool: list[dict],
    rejected_urls: set[str] | list[str] | None = None,
) -> dict | None:
    """
    Return the next posting to put in front of the judge, or None when the
    pool is exhausted.

    The pool arrives sorted by (screener rating, BM25 rank), so "next"
    means the best remaining posting the judge has not already rejected.
    No model call: the ordering was computed once during the pool build
    and every cycle reads it rather than re-deriving it.

    Returns a dict in the shape the rest of the pipeline expects from the
    job stage — same keys the old selection agent produced, so judge,
    writer and renderers are unchanged.
    """
    rejected = set(rejected_urls or ())

    for position, record in enumerate(pool, start=1):
        url = record.get("job_url")
        if not url or url in rejected:
            continue

        rating = record.get("fit_rating")
        bm25_rank = record.get("bm25_rank")
        note = record.get("fit_note") or "no screening note"

        log.info(
            "    Pool position %d of %d: %s at %s (fit %s, BM25 rank %s)",
            position, len(pool), record.get("title"), record.get("company"),
            rating, bm25_rank,
        )

        return {
            "job_title": record.get("title"),
            "company": record.get("company"),
            "location": record.get("location"),
            "url": url,
            "posted_date": record.get("date_posted") or "unknown",
            "full_requirements": _full_requirements(record),
            "match_rationale": (
                f"Screener rated this {rating}/10 for fit: {note}. "
                f"BM25 lexical rank {bm25_rank} of the scraped set."
            ),
            "search_notes": (
                f"Selected deterministically at position {position} of {len(pool)} "
                f"in the screened, fit-ranked pool. "
                f"{len(rejected)} posting(s) skipped as already rejected this run."
            ),
        }

    log.info("    Pool exhausted — every posting has been put to the judge and rejected.")
    return None
