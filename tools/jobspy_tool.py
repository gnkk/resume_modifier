"""
JobSpy tool wrapper.

PRIMARY job-search tool for this project (see agents/search_agent.py).
python-jobspy scrapes job postings directly from LinkedIn, Indeed,
ZipRecruiter, and Glassdoor using purpose-built scrapers for each
site — the sites job seekers actually use, and specifically the sites
tools/firecrawl_tool.py's BLOCKED_DOMAINS list *cannot* reach, because
Firecrawl's generic request-based scraping gets rejected there
outright. JobSpy exists precisely to get past that.

No API key required — JobSpy scrapes directly rather than calling a
hosted search API. LinkedIn in particular rate-limits aggressively
after a handful of pages; if you see repeated empty results or 429s,
pass fewer results_wanted, narrow site_name, or configure proxies
(see python-jobspy's `proxies` parameter — not wired in here yet,
add if needed).

In practice LinkedIn and Indeed are the reliable two of the four:
ZipRecruiter and Glassdoor sit behind Cloudflare bot protection and
frequently refuse JobSpy's requests outright. See the blocked-site
circuit breaker below, which drops such sites automatically rather
than retrying them on every search.

Firecrawl remains available as a SUPPORT tool alongside this one: for
widening a search beyond these four boards (company career pages,
niche/regional boards), or for pulling deeper page content on a
specific URL when JobSpy's own `description` field is thin or missing.
See agents/search_agent.py for how the two are combined.

Docs: https://github.com/speedyapply/JobSpy
"""

import logging
import re
from urllib.parse import urlparse

from jobspy import scrape_jobs

from config import JOB_SEARCH_SITES, JOB_SEARCH_HOURS_OLD, JOBSPY_COUNTRY_INDEED
from logger_setup import get_logger

log = get_logger(__name__)


# --- Single LinkedIn posting by URL -------------------------------------
# Firecrawl cannot touch LinkedIn (see tools/firecrawl_tool.py
# BLOCKED_DOMAINS), but JobSpy can — it is the whole reason JobSpy is the
# primary search tool here. The same fetch that backs
# `linkedin_fetch_description=True` works for one known posting, so a user
# who pastes a LinkedIn job link can be served without asking them to
# copy-paste the description.
#
# CAVEAT, deliberately loud: this calls LinkedIn._get_job_details(), a
# PRIVATE method of python-jobspy. It is not part of the package's public
# API and a future release may rename or restructure it without warning.
# The call is wrapped so that a break degrades to "couldn't fetch, paste
# the text instead" rather than crashing a run. If it stops working after
# a `pip install -U python-jobspy`, this is the first place to look.
_LINKEDIN_JOB_ID_RE = re.compile(r"/jobs/view/(?:[^/?#]*-)?(\d{6,})")
_LINKEDIN_CURRENT_JOB_ID_RE = re.compile(r"[?&]currentJobId=(\d{6,})")


def linkedin_job_id(url: str) -> str | None:
    """
    Pull the numeric job id out of a LinkedIn posting URL.

    Handles the shapes LinkedIn hands out:
      /jobs/view/4285719301/?refId=...
      /jobs/view/data-scientist-at-acme-4285719301
      /jobs/search/?currentJobId=4285719301

    Returns None for any non-LinkedIn host. The host check is not
    decoration: `/jobs/view/<digits>` is a common path shape on ATS and
    company career sites too, and without it a Greenhouse or Workday URL
    would be handed to LinkedIn's fetcher, which would cheerfully request
    an unrelated LinkedIn job of the same id.
    """
    if not url:
        return None
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return None
    if not (host == "linkedin.com" or host.endswith(".linkedin.com")):
        return None

    for pattern in (_LINKEDIN_JOB_ID_RE, _LINKEDIN_CURRENT_JOB_ID_RE):
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def fetch_linkedin_description(url: str) -> str | None:
    """
    Fetch one LinkedIn posting's description text by its URL.

    Returns the description (Markdown), or None when the id can't be
    parsed, the fetch fails, or LinkedIn bounces the request to its
    signup wall — which it does unpredictably for guest traffic, so None
    is a routine outcome, not an exception. Callers should fall back to
    asking the user for the text.
    """
    job_id = linkedin_job_id(url)
    if not job_id:
        log.error("jobspy_tool: no LinkedIn job id found in %s", url)
        return None

    try:
        from jobspy.linkedin import LinkedIn
        from jobspy.model import ScraperInput, DescriptionFormat, Site

        scraper = LinkedIn()
        # _get_job_details reads description_format off scraper_input,
        # which is normally populated by scrape(). Set it by hand since we
        # are calling the detail fetch on its own.
        scraper.scraper_input = ScraperInput(
            site_type=[Site.LINKEDIN],
            search_term="",
            description_format=DescriptionFormat.MARKDOWN,
        )
        details = scraper._get_job_details(job_id)
    except Exception as exc:  # noqa: BLE001 — private API; any break must degrade, not crash
        log.error(
            "jobspy_tool: LinkedIn single-posting fetch failed for %s: %s. "
            "This path uses a private python-jobspy method, so a package "
            "upgrade may have changed it.",
            url, exc, exc_info=True,
        )
        return None

    description = (details or {}).get("description")
    if not description:
        log.info(
            "    LinkedIn returned no description for job %s (usually its signup "
            "wall intercepting guest traffic).",
            job_id,
        )
        return None
    return description


# --- Blocked-site circuit breaker ---------------------------------------
#
# ZipRecruiter and Glassdoor sit behind Cloudflare bot protection and
# routinely return 403 for JobSpy's scrapers. JobSpy handles these
# per-site failures internally — it logs the error and returns whatever
# the other sites produced — so scrape_jobs() still succeeds and this
# wrapper cannot tell from the return value that a site was blocked.
# Without the tracking below, every search angle and every job-search
# cycle re-requests the blocked sites from scratch, burning seconds of
# retries and flooding the console each time.
#
# So: attach a handler to JobSpy's own per-site loggers, watch for
# hard-block signatures, and once a site trips the threshold, drop it
# from subsequent requests for the rest of the process. A 403 is a
# decision about our traffic, not a transient hiccup — retrying it
# within one run will not produce a different answer.
#
# Deliberately NOT treated as blocking: Glassdoor's 400 'location not
# parsed', which is a per-query geocoding failure for that specific
# location string rather than a site-wide block — a different location
# may work fine, so the site stays in play.

_BLOCKED_SITES: set[str] = set()

# Consecutive hard-block signals before a site is dropped. 2 rather than 1
# so a genuinely transient blip doesn't disable a working site for the run.
_BLOCK_THRESHOLD = 2
_block_counts: dict[str, int] = {}

# JobSpy names its per-site loggers 'JobSpy:Glassdoor', 'JobSpy:ZipRecruiter',
# and so on. Map those suffixes to the site keys scrape_jobs() accepts.
_LOGGER_SUFFIX_TO_SITE = {
    "linkedin": "linkedin",
    "indeed": "indeed",
    "ziprecruiter": "zip_recruiter",
    "glassdoor": "glassdoor",
    "google": "google",
}

# Signatures of a hard block (bot protection / auth refusal), as opposed
# to an empty result set or a single malformed query.
_HARD_BLOCK_RE = re.compile(
    r"(status code:?\s*(401|403|429))|forbidden|blocked|captcha",
    re.IGNORECASE,
)


class _BlockedSiteDetector(logging.Handler):
    """
    Listens to JobSpy's per-site loggers and counts hard-block signals,
    marking a site blocked for the rest of the process once it trips
    _BLOCK_THRESHOLD. A read-only observer — it never suppresses or
    alters JobSpy's own logging.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno < logging.ERROR:
                return
            suffix = record.name.split(":")[-1].strip().lower()
            site = _LOGGER_SUFFIX_TO_SITE.get(suffix)
            if site is None or site in _BLOCKED_SITES:
                return
            if not _HARD_BLOCK_RE.search(record.getMessage()):
                return
            _block_counts[site] = _block_counts.get(site, 0) + 1
            if _block_counts[site] >= _BLOCK_THRESHOLD:
                _BLOCKED_SITES.add(site)
                log.warning(
                    "jobspy: '%s' returned repeated hard-block responses "
                    "(403/429/forbidden) and is being dropped for the rest of "
                    "this run. Remaining searches will use the other sites; "
                    "Firecrawl is available as a fallback if coverage gets thin.",
                    site,
                )
        except Exception:  # noqa: BLE001 — a logging observer must never raise into the caller
            pass


def _install_block_detector() -> None:
    """Attach the detector once to JobSpy's logger tree."""
    jobspy_logger = logging.getLogger("JobSpy")
    if not any(isinstance(h, _BlockedSiteDetector) for h in jobspy_logger.handlers):
        jobspy_logger.addHandler(_BlockedSiteDetector())


_install_block_detector()


def reset_blocked_sites() -> None:
    """
    Clear the blocked-site memory. Blocks are tracked per process, which
    is the right scope for a single pipeline run; this exists for tests
    and for long-lived processes that want a fresh start.
    """
    _BLOCKED_SITES.clear()
    _block_counts.clear()


# How much of each posting's description to include in the slimmed record.
# Enough to judge relevance (the opening of a posting is almost always the
# role summary and headline requirements); the agent pulls the full text for
# its shortlist only — see _slim_record.
_DESCRIPTION_SNIPPET_CHARS = 400

# The only fields the search agent needs to shortlist a posting. JobSpy's
# DataFrame carries ~30 columns per row (company_industry, company_logo,
# emails, job_level, listing_type, and so on) plus a full markdown
# description that frequently runs several thousand characters. Passing
# raw records back meant a handful of postings consumed the entire
# tool-result budget and the rest were silently truncated away — the agent
# would scrape 30 jobs and effectively choose from 3. Keeping only these
# fields fits the whole result set in the same budget, so nothing good gets
# missed.
_KEEP_FIELDS = (
    "title",
    "company",
    "location",
    "job_url",
    "date_posted",
    "job_type",
    "is_remote",
    "site",
)


def _format_salary(record: dict) -> str | None:
    """Collapse JobSpy's five separate salary columns into one short string."""
    low, high = record.get("min_amount"), record.get("max_amount")
    if low is None and high is None:
        return None
    currency = record.get("currency") or ""
    interval = record.get("interval") or ""
    if low is not None and high is not None:
        amount = f"{low:g}-{high:g}"
    else:
        amount = f"{(low if low is not None else high):g}"
    return " ".join(part for part in (currency, amount, interval) if part).strip()


def _slim_record(record: dict, keep_full_description: bool = False) -> dict:
    """
    Reduce one JobSpy record to the fields needed to shortlist it, with a
    truncated description.

    When keep_full_description is set, the untruncated description is also
    stored under "description_full". That field is for the job POOL (see
    agents/search_agent.py): it stays on disk and in memory, and is never
    sent to the model until it asks for one specific posting's full text.
    Keeping it locally means the requirements text costs nothing extra to
    retrieve later — no second scrape, and it works even for sites
    Firecrawl cannot reach.
    """
    slim = {key: record.get(key) for key in _KEEP_FIELDS if record.get(key) is not None}

    # Prefer the employer's own posting URL when JobSpy resolved one — it's a
    # more durable application link than the aggregator's redirect.
    direct = record.get("job_url_direct")
    if direct:
        slim["job_url_direct"] = direct

    salary = _format_salary(record)
    if salary:
        slim["salary"] = salary

    description = record.get("description")
    if description:
        text = str(description)
        if keep_full_description:
            slim["description_full"] = text
        if len(text) > _DESCRIPTION_SNIPPET_CHARS:
            slim["description_snippet"] = text[:_DESCRIPTION_SNIPPET_CHARS]
            slim["description_truncated"] = True
        else:
            slim["description_snippet"] = text

    return slim


def _clean_records(df, keep_full_description: bool = False) -> list[dict]:
    """
    Convert a JobSpy results DataFrame into slim, JSON-safe dicts:
    NaN/NaT become None, any Timestamp/date-like values become ISO
    strings, and each row is reduced to the shortlisting fields (see
    _slim_record) so the whole result set fits in the agent's
    tool-result budget instead of a few rows crowding out the rest.
    """
    if df is None or df.empty:
        return []
    safe = df.astype(object).where(df.notna(), None)
    records = safe.to_dict(orient="records")
    cleaned = []
    for record in records:
        for key, value in list(record.items()):
            if hasattr(value, "isoformat"):
                record[key] = value.isoformat()
        cleaned.append(_slim_record(record, keep_full_description=keep_full_description))
    return cleaned


def search(
    search_term: str,
    location: str | None = None,
    site_name: list[str] | None = None,
    is_remote: bool = False,
    hours_old: int | None = None,
    results_wanted: int = 15,
    job_type: str | None = None,
    distance: int = 50,
    country_indeed: str | None = None,
    linkedin_fetch_description: bool = False,
    keep_full_description: bool = False,
) -> list[dict]:
    """
    Search LinkedIn, Indeed, ZipRecruiter, and/or Glassdoor via JobSpy
    and return a list of plain-dict job postings (title, company,
    location, job_url, description, date_posted, job_type, salary
    fields when listed, site, etc. — whatever JobSpy's DataFrame
    columns hold for that site).

    Any site already proven blocked earlier in this run is filtered out
    before the request (see the circuit breaker above), so a Cloudflare
    403 on ZipRecruiter or Glassdoor costs a couple of retries once
    rather than on every angle of every cycle.

    JobSpy handles individual per-site scrape failures internally
    (e.g. one board rate-limiting this run gets logged and skipped
    there, not raised). Only a genuinely unexpected failure reaches
    the except below — caught and turned into a one-item list with an
    "error" key so a single bad call can't crash the search-agent
    loop; the agent sees it as normal tool output and can react (e.g.
    retry narrower, or fall back to Firecrawl for that angle).

    Args mirror python-jobspy's scrape_jobs() directly; see
    JOBSPY_SEARCH_TOOL_SCHEMA below for the subset exposed to the
    search agent as a tool call.
    """
    requested = site_name or JOB_SEARCH_SITES
    sites = [s for s in requested if s not in _BLOCKED_SITES]
    skipped = [s for s in requested if s in _BLOCKED_SITES]

    if skipped:
        log.info(
            "jobspy search: skipping %s (blocked earlier this run); searching %s",
            skipped, sites or "nothing",
        )

    if not sites:
        return [
            {
                "error": (
                    f"All requested sites ({', '.join(requested)}) are blocked for "
                    "this run (repeated 403/forbidden responses from their bot "
                    "protection). Retrying them will not help. Search 'linkedin' "
                    "and 'indeed' instead — they are the reliable sites here — or "
                    "use Firecrawl's web_search for this angle."
                )
            }
        ]

    try:
        df = scrape_jobs(
            site_name=sites,
            search_term=search_term,
            location=location,
            distance=distance,
            job_type=job_type,
            is_remote=is_remote,
            results_wanted=results_wanted,
            hours_old=hours_old if hours_old is not None else JOB_SEARCH_HOURS_OLD,
            country_indeed=country_indeed or JOBSPY_COUNTRY_INDEED,
            description_format="markdown",
            linkedin_fetch_description=linkedin_fetch_description,
        )
    except Exception as exc:  # noqa: BLE001 — 3rd-party scraper, guard broadly like firecrawl_tool does
        log.error(
            "jobspy search: failed for search_term=%r location=%r sites=%r: %s",
            search_term, location, sites, exc, exc_info=True,
        )
        return [
            {
                "error": (
                    f"JobSpy search failed unexpectedly ({exc}). This is usually "
                    "rate limiting (especially LinkedIn) or a transient site change, "
                    "not a real code problem. Try again with fewer results_wanted, "
                    "narrower site_name, or fall back to Firecrawl's web_search for "
                    "this particular angle."
                )
            }
        ]

    records = _clean_records(df, keep_full_description=keep_full_description)

    # Per-site counts and titles at INFO: makes it visible in the console
    # which board actually produced results for each angle, and what the
    # agent is choosing between — rather than a silent block of HTTP lines.
    if records:
        by_site: dict[str, list[str]] = {}
        for record in records:
            by_site.setdefault(record.get("site") or "unknown", []).append(
                record.get("title") or "(untitled)"
            )
        total = len(records)
        breakdown = ", ".join(f"{site}: {len(titles)}" for site, titles in sorted(by_site.items()))
        log.info(
            "    jobspy: %d job(s) for %r%s — %s",
            total, search_term, f" near {location}" if location else "", breakdown,
        )
        for site, titles in sorted(by_site.items()):
            for title in titles:
                log.info("      [%s] %s", site, title)

    if not records:
        log.info(
            "jobspy search: 0 results for search_term=%r location=%r sites=%r "
            "(hours_old=%s) — try a different search_term/location/site_name "
            "before falling back to Firecrawl.",
            search_term, location, sites, hours_old or JOB_SEARCH_HOURS_OLD,
        )
        note = (
            f"No results from {', '.join(sites)} for this query."
            if not skipped
            else (
                f"No results from {', '.join(sites)} for this query. "
                f"Note: {', '.join(skipped)} were skipped as blocked for this run — "
                "do not request them again."
            )
        )
        return [{"error": note}]

    # Tell the agent which sites were dropped, so it stops asking for them
    # even if it doesn't read the log output.
    if skipped:
        records.insert(
            0,
            {
                "note": (
                    f"{', '.join(skipped)} are blocked for this run and were skipped. "
                    "Do not request them again — use linkedin and indeed, or "
                    "Firecrawl's web_search, for further angles."
                )
            },
        )

    return records


# --- Anthropic tool schema ---
JOBSPY_SEARCH_TOOL_SCHEMA = {
    "name": "search_jobs",
    "description": (
        "PRIMARY job search tool — try this first for every search angle, before "
        "Firecrawl's web_search. Searches LinkedIn, Indeed, ZipRecruiter, and "
        "Glassdoor directly via purpose-built scrapers (not a generic web search), "
        "so it reaches these main job sites even though Firecrawl itself is blocked "
        "there. Each result carries title, company, location, job_url, date_posted, "
        "job_type, salary when listed, and description_snippet — the opening of the "
        "posting, enough to shortlist on. When description_truncated is true and you "
        "need the full text to judge a shortlisted posting, fetch it for that posting "
        "only: scrape_url for non-LinkedIn/Glassdoor URLs, or re-call this tool with "
        "linkedin_fetch_description=true for LinkedIn ones. "
        "LinkedIn and Indeed are the most reliable sites; ZipRecruiter and "
        "Glassdoor are often blocked by bot protection and are dropped "
        "automatically once they fail."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "search_term": {
                "type": "string",
                "description": "Job title / keywords to search for, e.g. 'Senior Data Scientist'.",
            },
            "location": {
                "type": "string",
                "description": (
                    "City/region to search near, e.g. 'Halifax, NS' or 'Toronto, ON'. "
                    "Omit for a nationwide/remote search."
                ),
            },
            "site_name": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["linkedin", "indeed", "zip_recruiter", "glassdoor"],
                },
                "description": (
                    "Which boards to search. Defaults to all four if omitted. Prefer "
                    "['linkedin', 'indeed'] — they are the reliable ones. If a result "
                    "tells you a site is blocked for this run, never request that site "
                    "again; it will not start working."
                ),
            },
            "is_remote": {
                "type": "boolean",
                "description": "True to filter for remote-only postings.",
            },
            "hours_old": {
                "type": "integer",
                "description": (
                    f"Max posting age in hours. Defaults to {JOB_SEARCH_HOURS_OLD} (past "
                    "week) if omitted. LinkedIn/Indeed respect this precisely; other "
                    "sites round up to whole days."
                ),
            },
            "results_wanted": {
                "type": "integer",
                "description": "Results to fetch per site (default 15). Keep modest to avoid rate limiting.",
            },
            "job_type": {
                "type": "string",
                "enum": ["fulltime", "parttime", "internship", "contract"],
                "description": "Optional employment-type filter.",
            },
            "linkedin_fetch_description": {
                "type": "boolean",
                "description": (
                    "True to fetch each LinkedIn posting's full description + direct "
                    "URL (slower, one extra request per result) — use when you need "
                    "LinkedIn's full text, not just the summary."
                ),
            },
        },
        "required": ["search_term"],
    },
}


if __name__ == "__main__":
    results = search(
        "junior data scientist",
        location="Halifax, NS",
        site_name=["indeed", "linkedin"],
        results_wanted=5,
    )
    for r in results:
        print(r.get("title"), "-", r.get("company"), "-", r.get("job_url"))
