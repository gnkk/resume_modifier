"""
Search agent.

Given a summary of the candidate's background (from context_agent.py),
finds the SINGLE best-matching job posting currently available, with a
high realistic chance of a positive response — not just the first
plausible-looking listing. Restricted to postings newer than one week.

Uses Firecrawl's real-time web search (v2 /search), which returns
ranked results with full page markdown scraped inline via
scrapeOptions — so most of the time a single web_search call gives
enough to judge a posting without a second scrape_url call. scrape_url
stays available for the agent to pull deeper content on a specific
URL when the inline content wasn't enough (e.g. a paginated listing
page, or truncated content).

Runs on Sonnet 5 — picking the single best option among several
plausible ones, and judging "high hiring chance" rather than just
"technically matches," is a judgment call worth a capable model.

Output is structured JSON so the judge/writer downstream get a clean,
unambiguous target rather than having to re-parse prose.

This agent is driven by a search <-> judge loop (see main.py,
run_job_search_loop): if the judge doesn't approve a pick, this agent
is called again with the rejected job(s) and the judge's concerns, and
must find a DIFFERENT, better job rather than re-submitting the same one.
"""

import json
import anthropic

from config import ANTHROPIC_API_KEY, MODEL_SEARCH, JOB_SEARCH_TBS
from tools.firecrawl_tool import (
    search as firecrawl_search_fn,
    scrape as firecrawl_scrape_fn,
    FIRECRAWL_SEARCH_TOOL_SCHEMA,
    FIRECRAWL_SCRAPE_TOOL_SCHEMA,
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

TOOLS = [FIRECRAWL_SEARCH_TOOL_SCHEMA, FIRECRAWL_SCRAPE_TOOL_SCHEMA]

SYSTEM_PROMPT = f"""You are a job-search agent. Your ONLY job is to find the \
SINGLE best-matching job posting for this candidate, and return it as \
structured JSON. You are not writing a resume and not evaluating a draft \
— a different agent does each of those.

Hard constraints:
- Postings must be newer than one week old. Always call web_search with \
tbs="{JOB_SEARCH_TBS}" (this is also the default) — never widen this window \
even if results seem thin. If genuinely nothing suitable turns up within a \
week, say so honestly rather than reaching further back.
- You must pick exactly ONE job at the end — the single best match, not a \
shortlist. Use "high chance" to mean: the candidate's real, demonstrated \
skills and experience align closely with the posting's stated requirements \
(not just keyword overlap), the seniority level fits, and there's nothing in \
the posting suggesting the role is effectively closed (e.g. "position filled" \
language, a posting that's clearly a repost/evergreen listing with no real \
recency signal despite the date filter).
- If you are told a previous pick was rejected by the reviewing judge, you \
MUST return a DIFFERENT job posting (a different URL) — do not resubmit the \
same posting, and directly address the judge's stated concerns in how you \
search and what you pick this time.

Process:
1. Read the candidate summary you're given. Identify 2-4 distinct search \
angles (e.g. exact title variants, adjacent titles, key skill combinations, \
target location or remote status) rather than one generic query. If a prior \
pick was rejected, adjust your angles to specifically address the judge's \
concerns (e.g. search more precisely on seniority, location, or a keyword \
the judge flagged as missing).
2. Run web_search for each angle. web_search scrapes full page content \
inline by default, so read the returned markdown directly — only call \
scrape_url separately if a result's inline content is missing or truncated.
3. Shortlist the 2-4 most promising results based on their full content, not \
just titles/snippets. Exclude any URL already rejected in a previous cycle.
4. Compare the shortlist against the candidate's actual background. Pick the \
single best match.
5. If your job description context already fully specifies a job (i.e. the \
candidate already told you exactly which posting they want), you may still \
do a light web_search pass to confirm it's live and current, but do not \
override the user's explicit choice with a "better" one you found.

Respond with ONLY valid JSON in this exact shape, no other text:
{{
  "job_title": "<string>",
  "company": "<string>",
  "location": "<string>",
  "url": "<string, the specific posting URL>",
  "posted_date": "<string or 'unknown'>",
  "full_requirements": "<the full text of the posting's requirements/responsibilities you found>",
  "match_rationale": "<2-4 sentences on why this is the single best match and why you believe it's a high-chance fit>",
  "search_notes": "<brief note on angles tried and what was ruled out, for transparency>"
}}"""


_LOCAL_TOOL_NAMES = {"web_search", "scrape_url"}


def _execute_tool(name: str, tool_input: dict) -> str:
    if name == "web_search":
        results = firecrawl_search_fn(
            tool_input["query"],
            limit=tool_input.get("limit", 5),
            tbs=tool_input.get("tbs", JOB_SEARCH_TBS),
            scrape_content=True,
        )
        return json.dumps(results)
    if name == "scrape_url":
        return firecrawl_scrape_fn(tool_input["url"])
    raise ValueError(f"Unknown tool: {name}")


def find_best_job(
    candidate_context: str,
    target_role_description: str,
    rejected_jobs: list[dict] | None = None,
) -> dict:
    """
    Find the single best-matching, recently-posted job for this candidate.

    Args:
        candidate_context: Output from context_agent.gather_candidate_context().
        target_role_description: Free-text hint of what to look for, e.g.
            "Data Scientist at Shopify, remote Canada" or just
            "Data Scientist roles, remote Canada" if no specific company.
        rejected_jobs: Optional list of {"job": <prior job dict>, "review":
            <judge review dict>} entries from earlier cycles in the job-search
            <-> judge loop. When given, this call must return a different
            posting and should address the judge's stated concerns.

    Returns:
        Dict with keys: job_title, company, location, url, posted_date,
        full_requirements, match_rationale, search_notes.
    """
    user_prompt = (
        f"Candidate background:\n{candidate_context}\n\n"
        f"Target role hint: {target_role_description}\n\n"
    )

    if rejected_jobs:
        user_prompt += "Previously rejected picks — do NOT return any of these URLs again:\n"
        for i, entry in enumerate(rejected_jobs, start=1):
            prior_job = entry.get("job", {})
            prior_review = entry.get("review", {})
            user_prompt += (
                f"{i}. {prior_job.get('job_title')} at {prior_job.get('company')} "
                f"({prior_job.get('url')})\n"
                f"   Judge's job match score: {prior_review.get('job_match_score')}/10\n"
                f"   Judge's verdict: {prior_review.get('job_match_summary')}\n"
                f"   Judge's concerns: {prior_review.get('job_concerns')}\n"
            )
        user_prompt += (
            "\nFind a DIFFERENT, better job posting that addresses the concerns above.\n\n"
        )

    user_prompt += (
        "Find the single best-matching job posting from the last week and "
        "return it as the specified JSON."
    )

    messages = [{"role": "user", "content": user_prompt}]

    while True:
        response = client.messages.create(
            model=MODEL_SEARCH,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            text_blocks = [b.text for b in response.content if b.type == "text"]
            raw_text = "\n".join(text_blocks).strip()
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                return {
                    "job_title": None,
                    "company": None,
                    "location": None,
                    "url": None,
                    "posted_date": None,
                    "full_requirements": None,
                    "match_rationale": None,
                    "search_notes": "Could not parse structured job result.",
                    "raw_response": raw_text,
                }

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = _execute_tool(block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result[:8000],
                }
            )
        messages.append({"role": "user", "content": tool_results})
