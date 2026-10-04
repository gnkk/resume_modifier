"""
Catch template content leaking into a generated resume.

The sample resume in data/input/ is a LAYOUT spec. Its text is handed to
the writer and the resume judge verbatim, because a prose description of
a layout loses the detail that makes output look like the original — but
that means another person's employers, dates and numbers sit in the same
prompt as the candidate's own background, separated only by delimiters
and an instruction not to use them.

That instruction holds in practice and the judge is a second line of
defence, but neither is mechanical: nothing was actually checking the
output. This module does, cheaply, right before the resume is rendered.

It compares distinctive strings in the template against the draft, and
subtracts anything the candidate's own background also contains — so
"Python", "Toronto" or a shared former employer never trip it. What is
left is text that appears in the template and in the draft but nowhere in
the candidate's documents, which is exactly the shape a leak takes.

Advisory only. It logs what it finds and never blocks a render: this is a
heuristic over names and numbers, and a false positive must not cost
someone their resume.
"""

import re

from logger_setup import get_logger

log = get_logger(__name__)

# Multi-word proper nouns ("Acme Financial Group", "McGill University").
# Two words minimum — single capitalized words are overwhelmingly ordinary
# sentence starts and common technologies, and would drown the signal.
#
# The separator is [ \t]+, never \s+: a resume puts a role on one line and
# the employer on the next, and \s+ happily spans that newline. It then
# matches "Senior Product Manager Northwind" — a string that exists in no
# document on earth — and consumes the employer name that was the actual
# thing worth checking.
_PROPER_NOUN_RE = re.compile(
    r"\b[A-Z][a-zA-Z&.'\-]+(?:[ \t]+(?:of[ \t]+|and[ \t]+|the[ \t]+)?[A-Z][a-zA-Z&.'\-]+){1,3}\b"
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_URL_RE = re.compile(r"\b(?:https?://)?(?:[\w-]+\.)+(?:com|org|net|io|ca|dev|ai)(?:/[\w./-]*)?\b")
# A number with enough shape to be someone's specific claim: percentages,
# money, and counts of three digits or more.
_FIGURE_RE = re.compile(r"\b(?:\d{1,3}(?:,\d{3})+|\$[\d,.]+|\d+(?:\.\d+)?%|\d{3,})\b")

# Section headers and boilerplate the writer is SUPPOSED to copy from the
# template — matching them is the feature, not a leak.
_ALLOWED = {
    "work experience", "professional experience", "professional summary",
    "technical skills", "core competencies", "key skills", "education",
    "certifications", "personal projects", "academic projects", "work authorization",
    "additional information", "capstone project", "research project", "areas of expertise",
}


def _candidates(text: str) -> set[str]:
    found = set()
    for pattern in (_PROPER_NOUN_RE, _EMAIL_RE, _URL_RE, _FIGURE_RE):
        for match in pattern.findall(text):
            value = match.strip(" .,;:")
            if len(value) < 4 or value.isupper():
                # ALL-CAPS runs are section headers; short strings are noise.
                continue
            if value.lower() in _ALLOWED:
                continue
            found.add(value)
    return found


# Words inside a proper-noun phrase, and the connectors that don't count
# as evidence of anything.
_WORD_RE = re.compile(r"[A-Za-z][\w.'-]*")
_CONNECTORS = {"of", "and", "the"}


def _is_novel(value: str, context_lower: str) -> bool:
    """
    Is this string genuinely absent from the candidate's own background?

    Whole-string containment alone is too strict for phrases. The template
    and the candidate can both legitimately mention Python and SQL, and the
    draft can legitimately say "Python and SQL" — but that exact phrase may
    appear nowhere in the candidate's documents, so a naive check calls it
    a leak. What actually marks a leak is an unfamiliar WORD: "Northwind",
    "McGill". So a phrase counts only when at least one of its words (minus
    connectors) is absent from the candidate's background.

    Values with no alphabetic words at all — figures, emails — fall back to
    plain containment, which is the right test for them.
    """
    if value.lower() in context_lower:
        return False
    words = [w for w in _WORD_RE.findall(value) if w.lower() not in _CONNECTORS]
    if not words:
        return True
    return any(word.lower() not in context_lower for word in words)


def find_template_leaks(
    draft: str,
    style_template: str | None,
    candidate_context: str,
) -> list[str]:
    """
    Return template strings that appear in the draft but not in the
    candidate's own background.

    Args:
        draft: The resume Markdown about to be rendered.
        style_template: The sample resume's extracted text, or None.
        candidate_context: The merged candidate background — the whitelist.
            Anything here is the candidate's own and can never be a leak,
            however often it also appears in the template.

    Returns:
        The offending strings, sorted longest first so the most specific
        (and most likely genuine) match is reported first. Empty when
        there is no template or nothing leaked.
    """
    if not style_template or not style_template.strip():
        return []

    draft_lower = draft.lower()
    context_lower = candidate_context.lower()

    leaks = [
        value for value in _candidates(style_template)
        if value.lower() in draft_lower and _is_novel(value, context_lower)
    ]
    return sorted(set(leaks), key=len, reverse=True)


def warn_on_template_leaks(
    draft: str,
    style_template: str | None,
    candidate_context: str,
    job_label: str = "",
) -> list[str]:
    """
    Run the check and log the result. Advisory: never raises, never blocks.

    Logged at error level so it stands out in the run log and is coloured
    in the web UI, because a resume carrying another person's employer is
    worth looking at before it goes to an employer.
    """
    leaks = find_template_leaks(draft, style_template, candidate_context)
    if not leaks:
        return []

    shown = leaks[:8]
    log.error(
        "Possible template content in the resume%s: %s%s. These appear in "
        "data/input/sample_resume.pdf and in the draft, but nowhere in the "
        "candidate's own documents. Check the draft before sending it; the "
        "template is a layout spec and none of its facts belong here.",
        f" for {job_label}" if job_label else "",
        ", ".join(repr(value) for value in shown),
        f" (+{len(leaks) - len(shown)} more)" if len(leaks) > len(shown) else "",
    )
    return leaks
