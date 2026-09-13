"""
Requirements-first extraction from a job posting.

The screener reads a fixed-size slice of each posting. Plain head
truncation spends that budget on whatever the employer happened to put
first — often a company mission statement — while the text that decides
whether the candidate would be screened out sits further down. This
module reorders the slice so the employer's stated requirements come
first, responsibilities second, and boilerplate last or not at all.

WHY THIS IS BUILT TO FAIL SAFE. Section headers are the least reliable
text in a posting: wording varies, formatting varies, and plenty of
postings have no headers at all. A keyword-matching extractor will hit
perhaps two thirds of a real corpus. The dangerous failure is the silent
one — extracting nothing and handing the screener an empty string, which
looks exactly like a genuinely weak posting. So:

  - a posting with no recognized requirements section falls back to head
    truncation, which is what it got before this module existed;
  - the caller is told whether extraction fired, so the hit rate is
    visible in the run log rather than inferred from bad ratings.

If the logged hit rate turns out low, this layer is costing complexity
for little gain and the right move is to delete it and let the screener
find the requirements itself — it is a language model reading 2,000
characters, which is the thing it is actually good at.

Responsibilities are deliberately KEPT, ranked below requirements rather
than excluded. They are how a posting gets caught whose requirements
list the candidate's whole stack while the actual day-to-day is a
different discipline — a requirements-only slice would rate that role
highly.
"""

import re

from logger_setup import get_logger

log = get_logger(__name__)

# A line that looks like a section heading through explicit formatting:
# a markdown header, a fully bold line, a short ALL-CAPS line, or a short
# line ending in a colon.
_EXPLICIT_HEADING_RE = re.compile(
    r"""^\s*(?:
        \#{1,6}\s*(?P<md>.+?)\s*$           # ## Requirements
        |\*\*(?P<bold>[^*]{2,80})\*\*:?\s*$ # **What we're looking for**
        |(?P<caps>[A-Z][A-Z0-9 &/'\-]{3,60})\s*$  # REQUIREMENTS
        |(?P<colon>[A-Za-z][^.!?\n]{2,60}):\s*$   # Qualifications:
    )$""",
    re.VERBOSE,
)

# Scraped postings routinely arrive with formatting stripped, leaving
# headings as bare Title Case lines ("What we're looking for") that none
# of the patterns above match — this was the majority case in testing.
# Those are recognized structurally instead: a short, punctuation-free
# line sitting alone above a block of text. Requiring a blank line before
# and content immediately after is what keeps ordinary short sentences
# from being promoted to headings.
_BARE_HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9 &/'\u2019\-]{2,58}$")

# Ranked by how much the section matters to "would this person be
# screened out". Matched against the heading text.
_REQUIREMENTS_PAT = re.compile(
    r"what\s+we(?:'re| are)?\s+looking\s+for|what\s+you(?:'ll)?\s+(?:need|bring)"
    r"|requirements?|qualifications?|who\s+you\s+are|about\s+you|your\s+profile"
    r"|the\s+ideal\s+candidate|you\s+(?:have|are|will\s+have)|must[\s-]haves?"
    r"|skills?\s*(?:and|&)\s*experience|experience\s*(?:and|&)\s*skills"
    r"|required|essential|we(?:'d| would)\s+love",
    re.IGNORECASE,
)
_RESPONSIBILITIES_PAT = re.compile(
    r"what\s+you(?:'ll| will)?\s+(?:be\s+)?do|responsibilit|the\s+role|your\s+role"
    r"|your\s+impact|day[\s-]to[\s-]day|duties|in\s+this\s+role|key\s+activities",
    re.IGNORECASE,
)
# Matched FIRST, so "nice to have" wins over the "have" in the
# requirements pattern and a "Preferred qualifications" heading is not
# promoted alongside the real one.
_DEPRIORITIZE_PAT = re.compile(
    r"nice[\s-]to[\s-]have|bonus|preferred|desirable|plus(?:es)?$"
    r"|benefits?|perks?|compensation|salary|pay\s+range|what\s+we\s+offer"
    r"|about\s+(?:us|the\s+company)|our\s+(?:mission|values|culture|story)"
    r"|equal\s+opportunit|eeo|accommodation|diversity|inclusion"
    r"|how\s+to\s+apply|application\s+process|next\s+steps|privacy|legal",
    re.IGNORECASE,
)


def _heading_text(line: str, prev_blank: bool, next_blank: bool) -> str | None:
    match = _EXPLICIT_HEADING_RE.match(line)
    if match:
        for group in ("md", "bold", "caps", "colon"):
            value = match.group(group)
            if value:
                return value.strip()

    stripped = line.strip()
    if (
        prev_blank
        and not next_blank
        and _BARE_HEADING_RE.match(stripped)
        and len(stripped.split()) <= 8
        and not stripped.endswith((".", "!", "?", ",", ";"))
    ):
        return stripped
    return None


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split into (heading, body) pairs. Text before the first heading gets ''."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = [("", [])]
    for index, line in enumerate(lines):
        prev_blank = index == 0 or not lines[index - 1].strip()
        next_blank = index + 1 >= len(lines) or not lines[index + 1].strip()
        heading = _heading_text(line, prev_blank, next_blank)
        if heading:
            sections.append((heading, []))
        else:
            sections[-1][1].append(line)
    return [(heading, "\n".join(body).strip()) for heading, body in sections]


def requirements_first(text: str, limit: int, head_chars: int = 400) -> tuple[str, bool]:
    """
    Reorder a posting so its requirements lead, truncated to `limit`.

    Always keeps the first `head_chars` of the posting regardless of
    sections — the title, the one-line role summary and the seniority
    signal usually live there, and losing them to a reordering would
    trade one blind spot for another.

    Returns:
        (text, extracted) where `extracted` is True only when a
        requirements section was actually recognized. False means the
        caller got plain head truncation and should count it as a miss.
    """
    if not text:
        return "", False
    if len(text) <= limit:
        return text, False

    sections = _split_sections(text)
    required, responsibilities = [], []

    for heading, body in sections:
        if not heading or not body:
            continue
        if _DEPRIORITIZE_PAT.search(heading):
            continue
        if _REQUIREMENTS_PAT.search(heading):
            required.append(f"{heading}\n{body}")
        elif _RESPONSIBILITIES_PAT.search(heading):
            responsibilities.append(f"{heading}\n{body}")

    if not required:
        return text[:limit], False

    head = text[:head_chars]
    rebuilt = "\n\n".join([head, *required, *responsibilities])
    return rebuilt[:limit], True
