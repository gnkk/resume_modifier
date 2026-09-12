"""
Search agent.

Runs in two distinct phases, which used to be one function:

  1. build_job_pool() — ONE scraping pass per run. An LLM plans several
     complementary search angles from the candidate's background, those
     angles are executed deterministically against JobSpy, and the
     deduplicated results are over-collected past the target, screened
     for fit against the candidate, and cut down to the best
     ~JOB_POOL_TARGET_SIZE postings.

  2. select_best_job() — runs once per judge cycle. It does NOT scrape.
     It queries the existing pool and recommends one posting. When the
     judge rejects a pick, the next cycle picks a DIFFERENT posting from
     the same pool.

Why the split: previously every judge cycle called a fresh search agent
with no memory, so it re-ran near-identical queries and re-scraped
near-identical postings just to rebuild the candidate set it already
had. Nothing novel came back — the second and third cycles paid full
scrape cost to rediscover cycle one's results. Worse, "second best" was
incoherent across cycles, because each cycle drew from a freshly built
set rather than moving down one ranked list.

Building the pool once also makes it affordable to fetch full LinkedIn
descriptions up front (see JOBSPY_FETCH_LINKEDIN_DESCRIPTIONS), so every
posting in the pool carries real requirements text. The selecting agent
never sees those full descriptions in context — they stay in the pool
and are handed over one at a time via get_full_description. Those
descriptions are also what the screening pass reads, so turning the
fetch off degrades screening to near-useless on LinkedIn results.

Screening and selection share the market calibration in
agents/market_context.py with the judge, so all three weigh a posting's
requirements the same way. When they diverge, the loop burns cycles:
the search agent hands up a pick it rates highly and the judge rejects
it on a requirement the search agent never weighted.

Escape hatch: if the judge raises a concern the pool genuinely cannot
satisfy (wants remote, pool is all on-site), the agent may call
search_more ONCE per run, stating which concern forces it. Without it, a
pool that was wrong from the start guarantees two more bad picks;
ungated, it degenerates back into re-searching every cycle.

Firecrawl stays a support tool: broadening beyond JobSpy's boards during
the pool build, or scraping a specific posting whose description came
back empty.

Planning, screening, and selection each run on Haiku 4.5 via their own
config constant (MODEL_PLANNER, MODEL_SCREENER, MODEL_SELECTOR) rather
than one shared setting — they are different jobs with different stakes,
and the screener in particular makes permanent drops. All three are
mechanical next to the judge's evaluation work, which stays on Sonnet 5.
"""

import json
import re
import anthropic

from config import (
    ANTHROPIC_API_KEY,
    MODEL_PLANNER,
    MODEL_SCREENER,
    MODEL_SELECTOR,
    JOB_SEARCH_SITES,
    JOB_SEARCH_HOURS_OLD,
    JOB_POOL_TARGET_SIZE,
    JOB_POOL_OVERSCAN_FACTOR,
    POOL_SCREEN_MIN_FIT,
    POOL_SCREEN_BATCH_SIZE,
    JOBSPY_FETCH_LINKEDIN_DESCRIPTIONS,
    MAX_EXTRA_SEARCHES,
    SKIP_PREVIOUSLY_SELECTED,
)
from agents.market_context import MARKET_CALIBRATION
from job_history import filter_seen, load_history
from tools.jobspy_tool import search as jobspy_search_fn
from tools.firecrawl_tool import scrape as firecrawl_scrape_fn
from logger_setup import get_logger

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
# Phase 1: plan the angles, then build the pool
# ---------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = f"""You are planning a job search. You get ONE \
scraping pass for this entire run, so the angles you choose determine \
everything the downstream selection agent will have to work with. There is no \
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

Set results_wanted per angle generously. The pool is over-collected and then \
screened down to the best {JOB_POOL_TARGET_SIZE} postings, so a wide net \
costs little and a narrow one permanently limits what the selection agent \
can choose from — expect heavy overlap between angles in the same city.

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


SCREENER_SYSTEM_PROMPT = f"""You are screening scraped job postings for one \
candidate. You will be given a batch of postings — title, company, location, \
and the opening of the description — each with an index. Rate how well each \
ONE fits this specific candidate.

{MARKET_CALIBRATION}

Rate 1-5:
  5 — the candidate would be a shortlist name: core discipline, methods and \
seniority all align.
  4 — strong: core work aligns, seniority fits, only soft items missing.
  3 — plausible: same core discipline, but seniority is a band off or the \
day-to-day is partly different.
  2 — weak: same broad field, different actual work.
  1 — wrong role, or a hard blocker applies.

You are working from a snippet, not the full posting, so judge what the role \
IS — the discipline and the level — rather than counting keywords. When a \
snippet is too thin to tell, rate 3 and say so; a posting wrongly dropped \
here is gone for the whole run, while a wrongly kept one just gets examined \
more closely later. Be decisive about the clear cases: postings for a \
different discipline, or several seniority bands off, should get 1 or 2 \
rather than a cautious 3.

Rate every posting in the batch. Respond with ONLY a JSON array, no other \
text:
[{{"i": <index>, "fit": <1-5>, "why": "<max 12 words>"}}, ...]"""


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
            "snippet": str(record.get("description_snippet") or "")[:700],
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
        ratings[index] = {"fit": max(1, min(5, fit)), "why": str(entry.get("why") or "")}
    return ratings


def screen_pool(
    pool: list[dict],
    candidate_context: str,
    target_role_description: str,
    target_size: int = JOB_POOL_TARGET_SIZE,
) -> list[dict]:
    """
    Rate every scraped posting against the candidate, drop the clear
    mismatches, and keep the best `target_size`.

    This is the step that makes the pool's contents mean something. Before
    it, the pool was simply the first N postings the earliest angles
    returned — scraping stopped the moment the count was hit, so the later
    (often better-targeted) angles frequently never ran, and every posting
    in the pool looked equally plausible to the selection agent. All the
    filtering then happened downstream, one posting per judge cycle, at
    three cycles per run.

    Ratings are attached to each record as `fit_rating`/`fit_note` and
    travel with it into the selection agent's view of the pool, so the
    selector inherits this pass's context instead of re-deriving it from
    snippets.

    Degrades gracefully: postings the screener didn't rate are kept,
    ranked below rated ones. A failed screening pass costs pool quality,
    never postings.
    """
    if not pool:
        return pool

    log.info("  Screening %d scraped posting(s) for fit...", len(pool))

    ratings: dict[int, dict] = {}
    indexed = list(enumerate(pool))
    for start in range(0, len(indexed), POOL_SCREEN_BATCH_SIZE):
        batch = indexed[start : start + POOL_SCREEN_BATCH_SIZE]
        ratings.update(_screen_batch(batch, candidate_context, target_role_description))

    if not ratings:
        log.error("search_agent.screen_pool: no postings could be rated; keeping the pool unscreened.")
        log.info("  Warning: screening produced no ratings — using the unscreened pool.")
        return pool[:target_size]

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

    # Best first, unrated last (None sorts below any real rating).
    kept.sort(key=lambda r: r.get("fit_rating") or 0, reverse=True)
    screened = kept[:target_size]

    log.info(
        "  Screened: %d dropped below fit %d, %d unrated, %d kept (pool now %d).",
        dropped, POOL_SCREEN_MIN_FIT, unrated, len(kept), len(screened),
    )
    strong = sum(1 for r in screened if (r.get("fit_rating") or 0) >= 4)
    log.info("  %d posting(s) in the pool rated 4+ for fit.", strong)
    if strong == 0:
        log.info(
            "  Note: nothing in the pool rates as a strong match. The judge's bar "
            "is unlikely to be met this run — consider a different target role hint."
        )
    return screened


def build_job_pool(
    candidate_context: str,
    target_role_description: str,
    target_size: int = JOB_POOL_TARGET_SIZE,
) -> list[dict]:
    """
    Run the single scraping pass for this run and return a screened,
    deduplicated pool of postings.

    Angles are planned by the model but executed here deterministically,
    so the pass can't be cut short by the model deciding it has enough.
    Deduplication is on job_url.

    Scraping deliberately over-collects (up to target_size *
    JOB_POOL_OVERSCAN_FACTOR) so that EVERY planned angle runs, then
    screen_pool() rates the lot and keeps the best target_size. Stopping
    at target_size during scraping, as this used to, meant the pool was
    decided by angle ordering rather than by fit.
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

    scrape_ceiling = target_size * JOB_POOL_OVERSCAN_FACTOR
    pool: list[dict] = []
    seen_urls: set[str] = set()

    for angle in angles:
        if len(pool) >= scrape_ceiling:
            log.info(
                "  Scrape ceiling of %d reached — skipping remaining angles.",
                scrape_ceiling,
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

    # Before screening, not after: a posting already targeted in an earlier
    # run is dead weight either way, and dropping it here means the
    # screening pass doesn't spend tokens rating it.
    if SKIP_PREVIOUSLY_SELECTED:
        pool = filter_seen(pool)

    if len(pool) < target_size:
        log.info(
            "  Note: fewer than the %d-posting target remains — the market "
            "may be thin for these angles this fortnight.",
            target_size,
        )
    return screen_pool(pool, candidate_context, target_role_description, target_size)


# ---------------------------------------------------------------------
# Phase 2: select from the pool (no scraping)
# ---------------------------------------------------------------------

QUERY_POOL_TOOL_SCHEMA = {
    "name": "query_pool",
    "description": (
        "Search the job pool gathered and screened for this run. This is your "
        "primary tool — the pool already exists and costs nothing to query, so "
        "explore it thoroughly before concluding it lacks something. Returns "
        "matching postings with title, company, location, job_url, date_posted, "
        "salary when listed, description_snippet, and the screening pass's "
        "fit_rating (1-5) and fit_note. Results keep the pool's order, "
        "best-rated first. Call with no filters to see everything."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title_contains": {
                "type": "string",
                "description": "Case-insensitive substring match on the job title, e.g. 'engineer'.",
            },
            "location_contains": {
                "type": "string",
                "description": "Case-insensitive substring match on location, e.g. 'Toronto'.",
            },
            "company_contains": {
                "type": "string",
                "description": "Case-insensitive substring match on company name.",
            },
            "remote_only": {
                "type": "boolean",
                "description": "True to return only postings flagged remote.",
            },
            "limit": {
                "type": "integer",
                "description": "Max postings to return (default 25).",
            },
        },
    },
}

GET_FULL_DESCRIPTION_TOOL_SCHEMA = {
    "name": "get_full_description",
    "description": (
        "Retrieve the complete posting text for one job in the pool, by its "
        "job_url. Already stored locally, so this costs nothing and needs no "
        "network call. You MUST call this for the posting you finally "
        "recommend — full_requirements has to contain real requirements text, "
        "not a snippet, because the judge and resume writer both depend on it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "job_url": {
                "type": "string",
                "description": "The exact job_url of a posting from the pool.",
            }
        },
        "required": ["job_url"],
    },
}

SEARCH_MORE_TOOL_SCHEMA = {
    "name": "search_more",
    "description": (
        "ESCAPE HATCH — one use per run, total. Runs a fresh scrape and adds "
        "results to the pool. Only permitted when the judge has raised a "
        "concern that nothing in the pool can satisfy (e.g. the judge requires "
        "remote and the pool is entirely on-site). Do NOT use it to look for "
        "something marginally better, or because the pool feels small: the "
        "same queries return the same postings. Query the pool thoroughly "
        "first."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "Which specific judge concern the existing pool cannot satisfy, "
                    "and what you already checked in the pool to confirm that."
                ),
            },
            "search_term": {"type": "string", "description": "Job title / keywords."},
            "location": {"type": "string", "description": "City/region, or omit for nationwide."},
            "is_remote": {"type": "boolean", "description": "True for remote-only."},
            "results_wanted": {"type": "integer", "description": "Results per site (default 20)."},
        },
        "required": ["reason", "search_term"],
    },
}

SELECTOR_SYSTEM_PROMPT = f"""You are a job-selection agent. A pool of job \
postings has already been gathered and screened for this candidate in a \
single scraping pass. Your job is to recommend the SINGLE best-matching \
posting from that pool, as structured JSON.

{MARKET_CALIBRATION}

The pool is fixed. There is no benefit to re-searching — the same queries \
return the same postings, which is exactly why the pool exists. Use \
query_pool to explore it: start broad (no filters) to see the whole set, then \
narrow with filters to compare candidates.

Every posting carries `fit_rating` (1-5) and `fit_note` from the screening \
pass, and the pool is ordered best-rated first. Treat that as a first-pass \
signal read from a snippet, not a verdict: start with the 4s and 5s, but \
confirm against the full description before committing, and prefer a 4 that \
genuinely matches the core work over a 5 whose full text turns out to be a \
different job. A null rating means the screener could not rate it, not that \
it is weak.

Process:
1. Query the pool broadly first so you know what is actually available.
2. Judge postings on substance. description_snippet is the opening of the \
posting \u2014 enough to shortlist on, not enough to choose on.
3. Before committing, call get_full_description on your chosen posting and put \
its real text in full_requirements. A snippet there is not acceptable; the \
judge and the resume writer both depend on this.
4. Check the full description for hard blockers before you commit — required \
citizenship or clearance, a licence the candidate lacks, a location they \
cannot be in, or a role that is actually a different discipline. These are \
what the judge rejects picks over, and a snippet usually does not show them. \
If your leading candidate has one, move to the next.
5. Recommend exactly ONE posting.

"Best match" means the candidate's real, demonstrated experience aligns with \
the posting's stated requirements — substantive overlap on the CORE work, \
not keyword coincidence — the seniority fits, and nothing suggests the role \
is closed. A downstream judge scores this pick against the same calibration \
above and rejects anything below a high bar, so a posting you have talked \
yourself into will come straight back as a wasted cycle.

If you are told a previous pick was rejected, you MUST recommend a DIFFERENT \
posting (different job_url) from the pool, chosen to address the judge's \
stated concerns. This is the normal case: work down the pool, don't re-search.

search_more exists only for when the judge's concern is genuinely \
unsatisfiable from the pool ({MAX_EXTRA_SEARCHES} use per run). Explore the \
pool properly before even considering it, and state what you checked.

Respond with ONLY valid JSON in this exact shape, no other text:
{{
  "job_title": "<string>",
  "company": "<string>",
  "location": "<string>",
  "url": "<string, the posting URL>",
  "posted_date": "<string or 'unknown'>",
  "full_requirements": "<the posting's full requirements text from get_full_description>",
  "match_rationale": "<2-4 sentences on why this is the best match in the pool>",
  "search_notes": "<what you compared it against in the pool, and what you ruled out>"
}}"""


def _public_record(record: dict) -> dict:
    """Pool record as the model should see it — full description withheld."""
    return {k: v for k, v in record.items() if k != "description_full"}


def _query_pool(pool: list[dict], tool_input: dict) -> list[dict]:
    title = (tool_input.get("title_contains") or "").lower()
    location = (tool_input.get("location_contains") or "").lower()
    company = (tool_input.get("company_contains") or "").lower()
    remote_only = tool_input.get("remote_only", False)
    limit = int(tool_input.get("limit") or 25)

    matches = []
    for record in pool:
        if title and title not in str(record.get("title", "")).lower():
            continue
        if location and location not in str(record.get("location", "")).lower():
            continue
        if company and company not in str(record.get("company", "")).lower():
            continue
        if remote_only and not record.get("is_remote"):
            continue
        matches.append(_public_record(record))
    return matches[:limit]


def select_best_job(
    pool: list[dict],
    candidate_context: str,
    target_role_description: str,
    rejected_jobs: list[dict] | None = None,
    extra_searches_used: int = 0,
) -> tuple[dict, int]:
    """
    Recommend the single best posting from the existing pool. No scraping
    unless the escape hatch fires.

    Args:
        pool: Postings from build_job_pool(). Mutated in place if
            search_more is used, so later cycles see the added postings.
        candidate_context: The context agent's candidate summary.
        target_role_description: Free-text role hint from the CLI.
        rejected_jobs: [{"job": ..., "review": ...}] from earlier cycles.
        extra_searches_used: How many escape-hatch searches this run has
            already spent.

    Returns:
        (job_dict, extra_searches_used) — the count is returned so the
        caller can carry the budget across cycles.
    """
    user_prompt = (
        f"Candidate background:\n{candidate_context}\n\n"
        f"Target role hint: {target_role_description}\n\n"
        f"The pool holds {len(pool)} posting(s).\n\n"
    )

    if rejected_jobs:
        user_prompt += "Already rejected — do NOT recommend these URLs again:\n"
        for i, entry in enumerate(rejected_jobs, start=1):
            prior_job = entry.get("job", {})
            prior_review = entry.get("review", {})
            user_prompt += (
                f"{i}. {prior_job.get('job_title')} at {prior_job.get('company')} "
                f"({prior_job.get('url')})\n"
                f"   Judge's score: {prior_review.get('job_match_score')}/10\n"
                f"   Judge's verdict: {prior_review.get('job_match_summary')}\n"
                f"   Judge's concerns: {prior_review.get('job_concerns')}\n"
            )
        user_prompt += (
            "\nRecommend a DIFFERENT posting from the pool that addresses these "
            "concerns.\n\n"
        )

    user_prompt += "Recommend the single best posting and return the specified JSON."

    messages = [{"role": "user", "content": user_prompt}]
    tools = [QUERY_POOL_TOOL_SCHEMA, GET_FULL_DESCRIPTION_TOOL_SCHEMA]
    if extra_searches_used < MAX_EXTRA_SEARCHES:
        tools.append(SEARCH_MORE_TOOL_SCHEMA)

    max_turns = 12
    turn = 0

    while True:
        turn += 1
        if turn > max_turns:
            log.error(
                "search_agent.select_best_job: exceeded %d turns without finishing; "
                "aborting this cycle.",
                max_turns,
            )
            return (
                {
                    "job_title": None, "company": None, "location": None, "url": None,
                    "posted_date": None, "full_requirements": None, "match_rationale": None,
                    "search_notes": f"Selection did not finish within {max_turns} turns.",
                },
                extra_searches_used,
            )

        try:
            response = client.messages.create(
                model=MODEL_SELECTOR,
                max_tokens=8192,
                system=SELECTOR_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
        except anthropic.APIError as exc:
            log.error("search_agent.select_best_job: Anthropic API call failed: %s", exc, exc_info=True)
            raise RuntimeError(
                f"search_agent: failed to reach the Anthropic API ({exc}). "
                "Check your ANTHROPIC_API_KEY and network connection."
            ) from exc

        if response.stop_reason != "tool_use":
            raw_text = "\n".join(b.text for b in response.content if b.type == "text").strip()
            parsed = _parse_json(raw_text)
            if isinstance(parsed, dict):
                return parsed, extra_searches_used

            truncated = response.stop_reason == "max_tokens"
            log.error(
                "search_agent.select_best_job: could not parse job JSON "
                "(stop_reason=%s). Raw: %s",
                response.stop_reason, raw_text[:500],
            )
            return (
                {
                    "job_title": None, "company": None, "location": None, "url": None,
                    "posted_date": None, "full_requirements": None, "match_rationale": None,
                    "search_notes": (
                        "Response was cut off before the job JSON was complete."
                        if truncated
                        else "Could not parse structured job result."
                    ),
                    "raw_response": raw_text,
                },
                extra_searches_used,
            )

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "query_pool":
                matches = _query_pool(pool, block.input)
                log.info("    pool query %s -> %d match(es)", block.input or "{}", len(matches))
                result = json.dumps(matches, default=str)[:30000]

            elif block.name == "get_full_description":
                url = block.input.get("job_url")
                record = next((r for r in pool if r.get("job_url") == url), None)
                if record is None:
                    result = json.dumps(
                        {"error": f"No posting with job_url {url!r} is in the pool."}
                    )
                else:
                    full = record.get("description_full") or record.get("description_snippet")
                    if not full:
                        # JobSpy occasionally returns a posting with no description
                        # at all; Firecrawl can sometimes still reach the page.
                        log.info("    no stored description for %s — trying Firecrawl", url)
                        full = firecrawl_scrape_fn(url)
                    result = json.dumps({"job_url": url, "description": full}, default=str)[:30000]

            elif block.name == "search_more":
                if extra_searches_used >= MAX_EXTRA_SEARCHES:
                    result = json.dumps(
                        {"error": "The extra-search budget for this run is already spent. Pick from the pool."}
                    )
                else:
                    extra_searches_used += 1
                    log.info(
                        "    ESCAPE HATCH used (%d/%d) — reason: %s",
                        extra_searches_used, MAX_EXTRA_SEARCHES, block.input.get("reason"),
                    )
                    new_results = jobspy_search_fn(
                        search_term=block.input["search_term"],
                        location=block.input.get("location"),
                        site_name=JOB_SEARCH_SITES,
                        is_remote=bool(block.input.get("is_remote", False)),
                        hours_old=JOB_SEARCH_HOURS_OLD,
                        results_wanted=int(block.input.get("results_wanted") or 20),
                        linkedin_fetch_description=JOBSPY_FETCH_LINKEDIN_DESCRIPTIONS,
                        keep_full_description=True,
                    )
                    known = {r.get("job_url") for r in pool}
                    added = [
                        r for r in new_results
                        if not r.get("error") and not r.get("note")
                        and r.get("job_url") and r["job_url"] not in known
                    ]
                    # Same cross-run filter the pool build applies — without it
                    # the escape hatch is a back door for postings earlier runs
                    # already targeted.
                    if SKIP_PREVIOUSLY_SELECTED:
                        added = filter_seen(added, load_history())
                    pool.extend(added)
                    log.info("    escape-hatch search added %d new posting(s)", len(added))
                    result = json.dumps(
                        {
                            "added": len(added),
                            "pool_size": len(pool),
                            "postings": [_public_record(r) for r in added],
                        },
                        default=str,
                    )[:30000]
                    # Tool is spent — drop it so it can't be called again.
                    tools = [t for t in tools if t["name"] != "search_more"]

            else:
                result = json.dumps({"error": f"Unknown tool: {block.name}"})

            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )

        messages.append({"role": "user", "content": tool_results})
