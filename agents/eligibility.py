"""
Work-eligibility extraction and location filtering.

Where the candidate may legally work is a FIXED, binary fact about them,
not a judgement call about a posting — so it is enforced in code, before
any model rates anything. On the run that motivated this module, the
screener rated a Salt Lake City role 10/10 for a candidate authorized
only in Canada; the judge caught it and scored it 2/10, but that cost a
full cycle out of three.

Everything here is derived from the candidate's own resume. Nothing is
hardcoded to a country — the same pipeline has to work for a candidate
authorized in the EU, the US, or anywhere else.

CONSERVATIVE BY CONSTRUCTION. A posting is dropped only when its
location is confidently outside the candidate's eligible countries.
Anything ambiguous is KEPT and left to the screener and judge. That
asymmetry is deliberate: a wrongly kept posting costs a little attention
downstream, while a wrongly dropped one is invisible and gone for the
whole run.

The ", CA" problem is the clearest case. Indeed writes Canadian
locations as "Toronto, ON, CA" and Californian ones as "San Francisco,
CA" — the same token, two countries. A province token resolves it when
present; when it is absent ("Remote, CA") the location stays ambiguous
and the posting is kept, because a real Canada-remote posting looked
exactly like that in testing.
"""

import json
import re

import anthropic

from config import ANTHROPIC_API_KEY, MODEL_ELIGIBILITY
from logger_setup import get_logger, note_model

log = get_logger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)

EXTRACT_SYSTEM_PROMPT = """You are reading a candidate's background to \
determine where they may legally work. Report only what the text states \
or clearly implies; do not guess from nationality, education location, or \
a name.

Return:
- countries: the countries where the candidate is authorized to work \
without the employer sponsoring them. Use full English country names \
("Canada", "United States", "United Kingdom", "Germany"). Usually one \
entry. Empty list if the text says nothing about work authorization.
- open_to_remote: true if the text says they are open to remote work.
- open_to_relocation: true if the text says they can relocate.
- basis: a short phrase quoting what the authorization rests on (e.g. \
"PGWP", "citizen", "permanent resident"), or null.

An empty countries list is the correct answer when the resume is silent. \
It disables location filtering entirely, which is the safe outcome — \
guessing a country would silently discard every posting elsewhere.

Respond with ONLY valid JSON, no other text:
{
  "countries": ["<country>", ...],
  "open_to_remote": <true/false>,
  "open_to_relocation": <true/false>,
  "basis": "<string or null>"
}"""


# --- Location gazetteer -------------------------------------------------
# Subnational tokens that identify a country when a posting's location
# string carries no country name. Only the countries this pipeline is
# likely to meet; anything absent simply stays ambiguous and is kept.
_SUBNATIONAL = {
    "Canada": {
        "on", "bc", "ab", "qc", "ns", "nb", "mb", "sk", "pe", "nl",
        "yt", "nt", "nu",
        "ontario", "british columbia", "alberta", "quebec", "nova scotia",
        "new brunswick", "manitoba", "saskatchewan", "prince edward island",
        "newfoundland", "yukon", "northwest territories", "nunavut",
    },
    "United States": {
        # "ca" is deliberately EXCLUDED — it collides with Canada's country
        # code in Indeed's own location strings. See the module docstring.
        "al", "ak", "az", "ar", "co", "ct", "de", "fl", "ga", "hi", "id",
        "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn",
        "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd",
        "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
        "va", "wa", "wv", "wi", "wy", "dc",
        "california", "texas", "new york", "florida", "washington",
        "massachusetts", "illinois", "colorado", "georgia", "utah",
        "arizona", "oregon", "virginia", "north carolina", "pennsylvania",
        "michigan", "minnesota", "ohio", "new jersey", "maryland",
    },
}

# Country names and the aliases postings actually use.
_COUNTRY_ALIASES = {
    "Canada": {"canada", "canadian"},
    "United States": {"united states", "usa", "u.s.", "u.s.a", "us", "america"},
    "United Kingdom": {"united kingdom", "uk", "england", "scotland", "wales", "britain"},
    "India": {"india"},
    "Australia": {"australia"},
    "Germany": {"germany", "deutschland"},
    "Ireland": {"ireland"},
    "Netherlands": {"netherlands", "holland"},
    "France": {"france"},
    "Spain": {"spain"},
    "Poland": {"poland"},
    "Singapore": {"singapore"},
    "Mexico": {"mexico"},
    "Brazil": {"brazil"},
    "Philippines": {"philippines"},
    "New Zealand": {"new zealand"},
    "United Arab Emirates": {"united arab emirates", "uae", "dubai", "abu dhabi"},
}

_TOKEN_RE = re.compile(r"[a-z][a-z.]*")


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


def extract_work_eligibility(candidate_context: str) -> dict:
    """
    Read where the candidate may work from their own background text.

    Returns a dict with countries / open_to_remote / open_to_relocation /
    basis. On any failure returns an empty countries list, which disables
    location filtering — the safe direction, since a wrong country would
    discard every posting outside it with nothing able to notice.
    """
    empty = {"countries": [], "open_to_remote": False, "open_to_relocation": False, "basis": None}

    try:
        response = client.messages.create(
            model=MODEL_ELIGIBILITY,
            max_tokens=512,
            system=EXTRACT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Candidate background:\n{candidate_context}"}],
        )
    except anthropic.APIError as exc:
        log.error("eligibility: Anthropic API call failed: %s", exc, exc_info=True)
        log.info("  Warning: could not determine work eligibility — location filtering is off this run.")
        return empty

    note_model(log, "eligibility", response)
    raw_text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    parsed = _parse_json(raw_text)
    if not isinstance(parsed, dict):
        log.error("eligibility: could not parse response. Raw: %s", raw_text[:300])
        return empty

    countries = [str(c).strip() for c in (parsed.get("countries") or []) if str(c).strip()]
    result = {
        "countries": countries,
        "open_to_remote": bool(parsed.get("open_to_remote")),
        "open_to_relocation": bool(parsed.get("open_to_relocation")),
        "basis": parsed.get("basis"),
    }

    if countries:
        log.info(
            "  Work eligibility: %s%s",
            ", ".join(countries),
            f" ({result['basis']})" if result["basis"] else "",
        )
    else:
        log.info("  Work eligibility could not be determined — location filtering is off this run.")
    return result


def _identify_country(location: str) -> str | None:
    """
    Best guess at which country a posting's location string names.

    Returns None when the string is ambiguous or unrecognized, which the
    caller treats as "keep".
    """
    if not location:
        return None
    text = location.lower()
    tokens = set(_TOKEN_RE.findall(text))

    # Explicit country names win over subnational tokens: "Vancouver, WA,
    # United States" should not resolve to Canada on a city name.
    for country, aliases in _COUNTRY_ALIASES.items():
        if any(alias in text for alias in aliases if " " in alias) or (tokens & {a for a in aliases if " " not in a}):
            return country

    for country, markers in _SUBNATIONAL.items():
        if any(marker in text for marker in markers if " " in marker) or (tokens & {m for m in markers if " " not in m}):
            return country

    return None


def filter_by_location(pool: list[dict], eligibility: dict) -> list[dict]:
    """
    Drop postings confidently located outside the candidate's eligible
    countries.

    Keeps anything whose country can't be identified, anything in an
    eligible country, and — when the candidate is open to remote — any
    posting flagged remote whose location is ambiguous. A remote posting
    that explicitly names an ineligible country ("Remote (US)") is still
    dropped, because that one is not ambiguous.
    """
    allowed = {c.strip().lower() for c in (eligibility.get("countries") or [])}
    if not allowed or not pool:
        return pool

    kept, dropped = [], []
    for record in pool:
        country = _identify_country(str(record.get("location") or ""))
        if country is None:
            kept.append(record)
            continue
        if country.lower() in allowed:
            kept.append(record)
            continue
        dropped.append(record)

    if dropped:
        log.info(
            "  Dropped %d posting(s) outside %s (e.g. %s).",
            len(dropped),
            "/".join(eligibility["countries"]),
            ", ".join(f"{r.get('title')} [{r.get('location')}]" for r in dropped[:3]),
        )
    return kept
