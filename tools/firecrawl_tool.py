"""
Firecrawl tool wrapper.

Two capabilities, matching Firecrawl's v2 API:
  - search():  find job postings across the web, restricted to a
    recency window via the `tbs` (time-based search) parameter, with
    full page markdown returned inline via scrapeOptions — no separate
    scrape call needed for most results.
  - scrape():  pull clean markdown/text content from one specific URL,
    for when the agent wants deeper content than the inline search
    scrape gave it (or wants to re-check a URL it already knows).

Uses `requests` directly against Firecrawl's REST API (v2) so this
project has minimal dependencies. Swap in the official `firecrawl-py`
SDK later if you want more features (crawl, map, agent, etc.).

Error handling: individual site failures (a job board blocking
scrapers, a dead link, a 403/429 from the target site passed through
by Firecrawl) are common and expected — they should not crash the
whole pipeline. scrape() therefore does NOT raise on HTTP errors; it
returns a short human-readable error string instead, which the calling
agent sees as normal tool output and can react to (e.g. try a
different result, or note the link couldn't be verified). Only
genuinely fatal problems (missing/invalid API key, network failure)
raise.

Blocked domains: sites like LinkedIn disallow automated scraping in
their robots.txt/ToS and reliably reject Firecrawl's requests, so
attempting them is a guaranteed wasted API call. search() filters
these out of returned results before the agent ever sees them, and
scrape() refuses outright (no network call at all) if asked to hit one
directly. See BLOCKED_DOMAINS below to extend the list.

Docs: https://docs.firecrawl.dev/api-reference/endpoint/search
"""

import requests
from urllib.parse import urlparse

from config import FIRECRAWL_API_KEY, FIRECRAWL_BASE_URL
from logger_setup import get_logger

log = get_logger(__name__)

HEADERS = {
    "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
    "Content-Type": "application/json",
}

# Domains that disallow scraping/automated access in their robots.txt or
# Terms of Service (LinkedIn is the clearest, best-known example — its
# robots.txt blocks nearly all automated crawling and it actively
# fingerprints/blocks scraper traffic). Firecrawl either refuses these
# outright or returns a 403 almost every time, so searching or scraping
# them is a guaranteed wasted API call and a wasted agent turn. Filtered
# out of search() results before the agent ever sees them, and scrape()
# refuses outright without hitting the network at all.
# Extend this list as you discover other sites that reliably block Firecrawl.
BLOCKED_DOMAINS = {
    "linkedin.com",
    "www.linkedin.com",
    "glassdoor.com",
    "www.glassdoor.com",
    "facebook.com",
    "www.facebook.com",
}


def _domain_of(url: str) -> str:
    """Extract the lowercase hostname from a URL, e.g. 'www.linkedin.com'."""
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def is_blocked_domain(url: str) -> bool:
    """
    True if url's host is in BLOCKED_DOMAINS or a subdomain of one
    (e.g. 'jobs.linkedin.com' matches 'linkedin.com').
    """
    host = _domain_of(url)
    if not host:
        return False
    return any(host == d or host.endswith(f".{d}") for d in BLOCKED_DOMAINS)

# Firecrawl's Google-style time-based search codes.
# See config.JOB_SEARCH_TBS for the default used by the search agent.
TBS_PAST_HOUR = "qdr:h"
TBS_PAST_DAY = "qdr:d"
TBS_PAST_WEEK = "qdr:w"
TBS_PAST_MONTH = "qdr:m"
TBS_PAST_YEAR = "qdr:y"


def _extract_error_detail(resp: requests.Response) -> str:
    """Pull a human-readable message out of a Firecrawl error response."""
    try:
        body = resp.json()
        return body.get("error") or body.get("message") or resp.text[:300]
    except ValueError:
        return resp.text[:300] or "(no response body)"


def search(
    query: str,
    limit: int = 5,
    tbs: str | None = None,
    scrape_content: bool = True,
) -> list[dict]:
    """
    Search the web via Firecrawl, optionally restricted to a recency
    window, with full page content scraped inline.

    Args:
        query: Search query, e.g. "senior data scientist job posting Halifax".
        limit: Max number of results.
        tbs: Time-based search filter, e.g. "qdr:w" for the past week.
            One of the TBS_* constants above, or a custom range like
            "cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY". None = no filter.
        scrape_content: If True (default), ask Firecrawl to return full
            page markdown for each result inline, avoiding a second
            scrape() call for the common case.

    Returns:
        List of result dicts with keys like 'title', 'url', 'description',
        and (when scrape_content=True) 'markdown' with the full page text.
        Raises requests.HTTPError for genuine search-endpoint failures
        (bad API key, malformed request) — these are not per-URL issues,
        so surfacing them loudly is correct.
    """
    payload = {"query": query, "limit": limit}
    if tbs:
        payload["tbs"] = tbs
    if scrape_content:
        payload["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}

    try:
        resp = requests.post(
            f"{FIRECRAWL_BASE_URL}/v2/search",
            headers=HEADERS,
            json=payload,
            timeout=45,
        )
    except requests.RequestException as exc:
        log.error("firecrawl search: network error for query '%s': %s", query, exc, exc_info=True)
        raise requests.HTTPError(
            f"Firecrawl search failed: network error reaching Firecrawl — {exc}"
        ) from exc

    if not resp.ok:
        detail = _extract_error_detail(resp)
        log.error("firecrawl search: %s for query '%s' — %s", resp.status_code, query, detail)
        raise requests.HTTPError(
            f"Firecrawl search failed ({resp.status_code}): {detail}",
            response=resp,
        )

    try:
        data = resp.json()
    except ValueError as exc:
        log.error("firecrawl search: non-JSON response for query '%s': %s", query, exc, exc_info=True)
        raise requests.HTTPError(f"Firecrawl search returned a non-JSON response: {exc}") from exc

    # v2 nests results by source type; default source is "web".
    results = data.get("data", {}).get("web", [])

    kept = []
    dropped = 0
    for r in results:
        r_url = r.get("url", "")
        if is_blocked_domain(r_url):
            dropped += 1
            continue
        kept.append(r)

    if dropped:
        log.info(
            "firecrawl search: dropped %d result(s) from blocked domains for query '%s' "
            "(sites that disallow scraping, e.g. LinkedIn) — no scrape attempted, no wasted call.",
            dropped, query,
        )

    return kept


def scrape(url: str) -> str:
    """
    Scrape a single URL and return clean markdown content.

    Refuses immediately, with no network call, for a URL on a domain
    known to block scraping/automated access (see BLOCKED_DOMAINS) —
    these reliably fail anyway, so attempting them just burns an API
    call and an agent turn for a predictable outcome.

    Otherwise does NOT raise on a non-2xx response from Firecrawl — a
    blocked scrape (403), dead link (404), or rate limit (429) on one
    specific URL is a normal, recoverable outcome for a job-search
    agent trying several candidate postings, not a reason to crash the
    pipeline. The agent gets a short error string back as if it were
    page content, and can decide how to proceed (try another result,
    note the link couldn't be verified, etc.).

    Args:
        url: The page to scrape (e.g. a specific job posting URL).

    Returns:
        Markdown text content of the page, or (on failure, or a
        blocked domain) a short "[scrape failed: ...]" /
        "[scrape skipped: ...]" string describing what happened.
    """
    if is_blocked_domain(url):
        log.info("firecrawl scrape: skipped blocked-domain URL (no network call): %s", url)
        return (
            f"[scrape skipped: {_domain_of(url)} disallows scraping/automated access "
            "(e.g. LinkedIn's robots.txt and Terms of Service block this). Do not use "
            "this URL as the final job pick, and don't retry scraping it — pick a "
            "different result instead.]"
        )

    try:
        resp = requests.post(
            f"{FIRECRAWL_BASE_URL}/v2/scrape",
            headers=HEADERS,
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            timeout=60,
        )
    except requests.RequestException as exc:
        log.error("firecrawl scrape: network error for %s: %s", url, exc, exc_info=True)
        return f"[scrape failed: network error reaching Firecrawl — {exc}]"

    if not resp.ok:
        detail = _extract_error_detail(resp)
        log.error("firecrawl scrape: %s for %s — %s", resp.status_code, url, detail)
        return (
            f"[scrape failed: {resp.status_code} for {url} — {detail}. "
            "This usually means the target site is blocking scrapers or the "
            "link is no longer live; treat this URL with caution or try a "
            "different result.]"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        log.error("firecrawl scrape: non-JSON response for %s: %s", url, exc, exc_info=True)
        return f"[scrape failed: Firecrawl returned a non-JSON response for {url} — {exc}]"

    return data.get("data", {}).get("markdown", "")


# --- Anthropic tool schemas ---
FIRECRAWL_SEARCH_TOOL_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web in real time for job postings, restricted to a "
        "recency window via the tbs parameter. Returns titles, URLs, "
        "short descriptions, and — since scraping is on by default — the "
        "full page markdown content for each result, so you usually don't "
        "need a separate scrape_url call for results this returns."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 5).",
            },
            "tbs": {
                "type": "string",
                "enum": ["qdr:h", "qdr:d", "qdr:w", "qdr:m", "qdr:y"],
                "description": (
                    "Recency filter using Google-style time codes: "
                    "qdr:h=past hour, qdr:d=past day, qdr:w=past week, "
                    "qdr:m=past month, qdr:y=past year. Use qdr:w (the "
                    "default) unless told otherwise — job postings must be "
                    "newer than one week old for this project."
                ),
            },
        },
        "required": ["query"],
    },
}

FIRECRAWL_SCRAPE_TOOL_SCHEMA = {
    "name": "scrape_url",
    "description": (
        "Scrape one specific URL (e.g. a job posting page found via "
        "web_search) and return its content as clean markdown text. "
        "Only needed when web_search's inline content wasn't enough, "
        "or to re-check a URL you already have. Do not call this on LinkedIn, "
        "Glassdoor, or Facebook URLs — those sites block automated scraping "
        "and this will always fail; web_search already filters them out of "
        "results, so avoid picking one as your final job even if you "
        "encounter one some other way. If the target site blocks scraping "
        "or the link is dead, this returns a short '[scrape failed: ...]' or "
        "'[scrape skipped: ...]' message instead of raising — treat that as "
        "a signal to be cautious about the URL or try a different result, "
        "not as a fatal error."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to scrape."}
        },
        "required": ["url"],
    },
}


if __name__ == "__main__":
    results = search(
        "junior data scientist job posting remote Canada",
        limit=3,
        tbs=TBS_PAST_WEEK,
    )
    for r in results:
        print(r.get("title"), "-", r.get("url"))
