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

Docs: https://docs.firecrawl.dev/api-reference/endpoint/search
"""

import requests
from config import FIRECRAWL_API_KEY, FIRECRAWL_BASE_URL

HEADERS = {
    "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
    "Content-Type": "application/json",
}

# Firecrawl's Google-style time-based search codes.
# See config.JOB_SEARCH_TBS for the default used by the search agent.
TBS_PAST_HOUR = "qdr:h"
TBS_PAST_DAY = "qdr:d"
TBS_PAST_WEEK = "qdr:w"
TBS_PAST_MONTH = "qdr:m"
TBS_PAST_YEAR = "qdr:y"


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
    """
    payload = {"query": query, "limit": limit}
    if tbs:
        payload["tbs"] = tbs
    if scrape_content:
        payload["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}

    resp = requests.post(
        f"{FIRECRAWL_BASE_URL}/v2/search",
        headers=HEADERS,
        json=payload,
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    # v2 nests results by source type; default source is "web".
    return data.get("data", {}).get("web", [])


def scrape(url: str) -> str:
    """
    Scrape a single URL and return clean markdown content.

    Args:
        url: The page to scrape (e.g. a specific job posting URL).

    Returns:
        Markdown text content of the page.
    """
    resp = requests.post(
        f"{FIRECRAWL_BASE_URL}/v2/scrape",
        headers=HEADERS,
        json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
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
        "or to re-check a URL you already have."
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
