"""
Firecrawl tool wrapper.

Two capabilities, matching Firecrawl's API:
  - search():  find job postings / company info across the web
  - scrape():  pull clean markdown/text content from a specific URL

Uses `requests` directly against Firecrawl's REST API so this project
has minimal dependencies. Swap in the official `firecrawl-py` SDK
later if you want more features (crawl, map, etc.).
"""

import requests
from config import FIRECRAWL_API_KEY, FIRECRAWL_BASE_URL

HEADERS = {
    "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
    "Content-Type": "application/json",
}


def search(query: str, limit: int = 5) -> list[dict]:
    """
    Search the web via Firecrawl.

    Args:
        query: Search query, e.g. "senior data scientist job posting Halifax".
        limit: Max number of results.

    Returns:
        List of result dicts with keys like 'title', 'url', 'description'.
    """
    resp = requests.post(
        f"{FIRECRAWL_BASE_URL}/v1/search",
        headers=HEADERS,
        json={"query": query, "limit": limit},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def scrape(url: str) -> str:
    """
    Scrape a single URL and return clean markdown content.

    Args:
        url: The page to scrape (e.g. a specific job posting URL).

    Returns:
        Markdown text content of the page.
    """
    resp = requests.post(
        f"{FIRECRAWL_BASE_URL}/v1/scrape",
        headers=HEADERS,
        json={"url": url, "formats": ["markdown"]},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {}).get("markdown", "")


# --- Anthropic tool schemas ---
FIRECRAWL_SEARCH_TOOL_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web for job postings, company information, or "
        "role requirements. Returns a list of results with titles, "
        "URLs, and short descriptions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 5).",
            },
        },
        "required": ["query"],
    },
}

FIRECRAWL_SCRAPE_TOOL_SCHEMA = {
    "name": "scrape_url",
    "description": (
        "Scrape a specific URL (e.g. a job posting page found via "
        "web_search) and return its content as clean markdown text."
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
    results = search("junior data scientist job posting remote Canada", limit=3)
    for r in results:
        print(r.get("title"), "-", r.get("url"))
