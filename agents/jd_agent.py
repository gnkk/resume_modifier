"""
Supplied job description -> job dict.

The entry point for the SECOND pipeline. When the user already has the
posting they want to apply to, there is nothing to search for and nothing
to select between: the job is decided. This module turns a job
description file into the same dict shape agents/search_agent.py
produces, so everything downstream (judge, writer, renderers) runs
unchanged.

Deliberately NOT an agentic loop. The model does one narrow job here:
pull the metadata a human would read off the top of a posting (title,
company, location, URL, posted date). The requirements text is passed
through VERBATIM and never summarized — the writer tailors against
specific phrasing, and a paraphrase would quietly drop the exact
terminology that ATS keyword alignment depends on.

Why the job description no longer goes through the context agent: it
used to be read there and folded into the candidate summary, which every
downstream prompt receives under the label "Candidate background". A
posting's requirements travelling as background about the candidate is
an invitation to confuse the two. Here it lands in
job["full_requirements"], which is where every prompt already expects a
posting to be.

Runs on Haiku via MODEL_JD_EXTRACT — structured extraction from text in
hand, not judgment. The most mechanical call in the project.
"""

import json
import re

import anthropic

from config import ANTHROPIC_API_KEY, MODEL_JD_EXTRACT
from tools.firecrawl_tool import scrape as firecrawl_scrape, is_blocked_domain
from tools.jobspy_tool import fetch_linkedin_description, linkedin_job_id
from logger_setup import get_logger

log = get_logger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_URL_START_RE = re.compile(r"^https?://", re.IGNORECASE)

# A posting that scrapes to less than this is a cookie wall, a login
# prompt, or a JavaScript shell — not a job description. Tailoring a
# resume against one would produce confident nonsense, so it fails loudly
# instead.
_MIN_USABLE_CHARS = 300


class JobUnavailableError(RuntimeError):
    """
    The supplied posting could not be read or fetched.

    Its own type so main.py can tell it apart from the RuntimeErrors the
    agents raise for API failures. Both stop the run, but only this one
    means "the job you named is unreachable" — and that is a different
    page, with a different set of next steps, from "the Anthropic API is
    down". Subclasses RuntimeError so existing handlers still catch it if
    a caller doesn't handle it specifically.
    """

# Enough of the posting for the metadata to be visible without paying for
# the whole thing twice — title, company and location live at the top, and
# an application URL is usually near the top or bottom.
_EXTRACT_HEAD_CHARS = 6000

EXTRACT_SYSTEM_PROMPT = """You are extracting metadata from a job posting \
the user has already chosen to apply to. You are NOT evaluating it, \
summarizing it, or judging fit.

Pull only what is stated in the text:
- job_title: the role title as the posting gives it.
- company: the hiring company. If the posting is from an agency on behalf \
of an unnamed client, say so rather than guessing.
- location: city/region, or "Remote" if stated remote. Include both when \
the posting gives both.
- url: the application or posting URL, if one appears in the text. Null if \
none appears \u2014 do not construct, guess, or complete a URL.
- posted_date: only if the text states one. Null otherwise.

Use null for anything the text does not state. Do not infer a company from \
an email domain, do not assume a location from a currency or phone format, \
and never invent a URL. A null is more useful downstream than a plausible \
guess, because a guess cannot be distinguished from a fact later.

Respond with ONLY valid JSON in this exact shape, no other text:
{
  "job_title": "<string or null>",
  "company": "<string or null>",
  "location": "<string or null>",
  "url": "<string or null>",
  "posted_date": "<string or null>"
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
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _first_url(text: str) -> str | None:
    """Fall back to the first URL in the posting text."""
    match = _URL_RE.search(text)
    return match.group(0).rstrip(".,;)") if match else None


def looks_like_url(value: str) -> bool:
    """True if this argument is a URL rather than a file path."""
    return bool(_URL_START_RE.match((value or "").strip()))


def fetch_job_description(url: str) -> str:
    """
    Fetch a posting's text from its URL.

    Two routes, because no single tool covers both:
      - LinkedIn goes through JobSpy, which has purpose-built LinkedIn
        scraping. Firecrawl is refused on linkedin.com outright.
      - Everything else goes through Firecrawl.

    Raises RuntimeError rather than degrading, in every failure mode.
    This is the opposite of how scrape() is treated during a job SEARCH,
    where a failed URL just means trying the next posting — here there is
    no next posting. The user named one job, and a resume tailored
    against a cookie wall or an error string would look exactly like a
    successful run.

    Raises:
        JobUnavailableError: the fetch failed, the domain blocks access
            with no route available, or the page returned too little text
            to be a real posting.
    """
    paste_instead = (
        "Open the posting in your browser, copy the description into a .txt "
        "file, and pass that file instead:\n"
        '    python main.py <resume.pdf> "-" data/input/job_description.txt\n'
        "Paste the posting's URL into the file too, on its own line, so the "
        "review page still links to the application."
    )

    if linkedin_job_id(url):
        log.info("  LinkedIn posting — fetching via JobSpy (Firecrawl can't reach LinkedIn)...")
        text = fetch_linkedin_description(url)
        if not text:
            raise JobUnavailableError(
                f"Could not fetch the LinkedIn posting at {url}. LinkedIn serves "
                "its signup wall to guest traffic unpredictably, so this fails "
                "some of the time even on a live posting — retrying in a few "
                f"minutes sometimes works.\n\n{paste_instead}"
            )
    elif is_blocked_domain(url):
        # Glassdoor and Facebook: Firecrawl refuses them and JobSpy has no
        # single-posting route for either.
        raise JobUnavailableError(
            f"{url} is on a domain that blocks automated access, and there is "
            "no working route to it from here (unlike LinkedIn, which JobSpy "
            f"can reach).\n\n{paste_instead}"
        )
    else:
        log.info("  Fetching the posting from %s ...", url)
        text = firecrawl_scrape(url)
        if not text or text.strip().startswith("[scrape"):
            raise JobUnavailableError(
                f"Could not fetch the posting at {url} — "
                f"{text.strip() if text else 'empty response'}\n\n"
                "Many employer sites block scrapers or render postings via "
                f"JavaScript.\n\n{paste_instead}"
            )

    if len(text.strip()) < _MIN_USABLE_CHARS:
        raise JobUnavailableError(
            f"The page at {url} returned only {len(text.strip())} characters — "
            "too little to be a real job description. This usually means a "
            "cookie wall, a login prompt, or a JavaScript-rendered page that "
            f"scraped empty.\n\n{paste_instead}"
        )

    log.info("  Fetched %d characters.", len(text.strip()))
    return text


def job_from_description(
    jd_text: str,
    source_path: str | None = None,
    source_url: str | None = None,
) -> dict:
    """
    Build the job dict for a user-supplied job description.

    Args:
        jd_text: The full text of the job description.
        source_path: Where it came from, for provenance in search_notes.
        source_url: The URL the user supplied, when they gave a link
            rather than a file. Authoritative — it overrides whatever the
            model or the regex fallback finds in the text, because the
            user's own link is the one they intend to apply through.

    Returns:
        A dict with the same keys search_agent.select_best_job() returns:
        job_title, company, location, url, posted_date, full_requirements,
        match_rationale, search_notes.

    Never raises for a failed extraction. The requirements text is the
    load-bearing field and the caller already has it; missing metadata
    degrades the review page's header, not the resume. An API failure
    falls back to a job dict carrying the full text with placeholder
    metadata, so the run still produces the resume the user asked for.
    """
    provenance = (
        f"Supplied by the user ({source_url or source_path or 'no source recorded'}). "
        "No job search was run."
    )
    fallback = {
        "job_title": "Role from supplied description",
        "company": None,
        "location": None,
        "url": source_url or _first_url(jd_text),
        "posted_date": None,
        "full_requirements": jd_text,
        "match_rationale": "Job chosen by the user, not by a search agent.",
        "search_notes": provenance,
    }

    if not jd_text or not jd_text.strip():
        log.error("jd_agent: the supplied job description is empty.")
        return fallback

    try:
        response = client.messages.create(
            model=MODEL_JD_EXTRACT,
            max_tokens=1024,
            system=EXTRACT_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Job posting text:\n{jd_text[:_EXTRACT_HEAD_CHARS]}\n\n"
                        "Extract the metadata now."
                    ),
                }
            ],
        )
    except anthropic.APIError as exc:
        log.error("jd_agent: Anthropic API call failed: %s", exc, exc_info=True)
        log.info("  Warning: could not extract posting metadata (%s). Continuing with the raw text.", exc)
        return fallback

    raw_text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    parsed = _parse_json(raw_text)
    if parsed is None:
        log.error(
            "jd_agent: could not parse metadata JSON (stop_reason=%s). Raw: %s",
            response.stop_reason, raw_text[:300],
        )
        log.info("  Warning: posting metadata could not be parsed. Continuing with the raw text.")
        return fallback

    return {
        "job_title": parsed.get("job_title") or fallback["job_title"],
        "company": parsed.get("company"),
        "location": parsed.get("location"),
        # The user's own link wins. Failing that, the model is told never
        # to invent a URL, so a null here is real, and the regex fallback
        # only finds a link literally present in the text.
        "url": source_url or parsed.get("url") or _first_url(jd_text),
        "posted_date": parsed.get("posted_date"),
        # Verbatim, always. Never the model's restatement of the posting.
        "full_requirements": jd_text,
        "match_rationale": "Job chosen by the user, not by a search agent.",
        "search_notes": provenance,
    }
