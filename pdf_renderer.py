"""
PDF renderer.

Renders the final tailored resume (Markdown) to PDF. The resume is
NOT output as a standalone HTML file — only Markdown and PDF are
saved to disk (see main.py). Internally this module still builds an
HTML representation of the resume as an intermediate step (WeasyPrint
renders HTML/CSS to PDF), but that intermediate HTML is never written
out; it exists only in memory for the duration of the PDF render.

Uses WeasyPrint: pure-Python-installable via pip (no external CLI
binary required on PATH), and the standard choice for turning styled
HTML/CSS into print-quality PDF. On macOS it needs a couple of system
libraries for text/font rendering (Pango, cairo, gdk-pixbuf) — see
README.md "Setup" for the one-time `brew install` step. This is a
one-time environment setup, not a per-run dependency.

WHERE APPEARANCE IS CONTROLLED
------------------------------
All visual formatting lives in this module, NOT in the writer's or
judge's prompts. The writer emits Markdown, which carries structure
(heading levels, bullets, emphasis) but no spacing, alignment or
typography — it never sees the rendered page, so asking it to "fix the
spacing" can't work. Two layers do the work here:

1. STYLE (below) — the design tokens: fonts, sizes, colours, margins,
   header alignment. Change the look by editing that dict, not the CSS
   body; every value is referenced by name from the stylesheet.
2. _prepare_html() — a thin Markdown-shaping layer that gives the CSS
   something to hook onto. Plain Markdown-to-HTML produces an
   undifferentiated run of <p> tags, so the contact block, the "Skills:"
   lines and an entry's company/date line all render identically. This
   layer tags them (.contact, .kv, .entry-meta) and lifts trailing dates
   out of the meta line onto the title line, flush right, which is what
   a professionally typeset resume looks like and what plain Markdown
   cannot express.

WHAT THE PDF MAY CONTAIN
------------------------
The PDF is the employer's copy, so it carries resume content and nothing
else — no notes, no page numbers, no render metadata, no commentary.
Anything this pipeline produces ABOUT the resume goes to the review HTML
(html_renderer.py), which is the internal-facing artifact.

That is why the writer's "## Notes (not part of the resume)" section is
split off here rather than rendered. It is deliberate and useful — the
writer flagging real gaps between the candidate and the posting — so it
stays in the saved .md and is handed to the review HTML via
extract_notes(); it just never reaches the PDF.
"""

import re

import markdown as md_lib

from config import RESUME_MAX_PAGES, RESUME_PREFERRED_PAGES
from logger_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
# The one place to tune appearance. If the output doesn't match
# data/input/sample_resume.pdf, adjust here first — these are the knobs
# that account for most of the visual difference between two resumes that
# carry the same content.
STYLE = {
    # Font stack. Only fonts actually installed locally render — WeasyPrint
    # has no web-font fallback and silently substitutes. macOS-safe swaps:
    #   serif : 'Charter, Georgia, "Times New Roman", serif'
    #   sans  : '"Helvetica Neue", Helvetica, Arial, sans-serif'
    #   humanist sans: '"Avenir Next", "Gill Sans", "Helvetica Neue", sans-serif'
    "font_body": '"Helvetica Neue", Helvetica, Arial, sans-serif',
    "font_headings": '"Helvetica Neue", Helvetica, Arial, sans-serif',

    # Body size drives everything else; most resume spacing below is in pt
    # so the rhythm stays fixed when this changes. 9.5-10.5pt is the usable
    # range — smaller reads as cramped, larger costs a page.
    "base_size": "10pt",
    "line_height": "1.34",

    # Page geometry. Tightening these is the cheapest way to win back a
    # page; going below 0.45in starts to look crowded and some ATS
    # PDF-to-text extractors clip hard at the edges.
    "page_margin": "0.5in 0.62in 0.55in 0.62in",

    # Colour. Set accent to "#141414" for an all-black, maximally
    # conservative resume.
    "accent": "#1f4e79",
    "text": "#141414",
    "muted": "#4a4a4a",
    "rule": "#9aa3ab",

    # Header block alignment. "left" matches the sample resume: name,
    # tagline and contact all flush left, no centring.
    "header_align": "left",
}

# Page numbers in the bottom margin. Off: the PDF carries resume content
# only (see "WHAT THE PDF MAY CONTAIN" above). Set True if you specifically
# want "2 / 3" footers on a multi-page copy.
SHOW_PAGE_NUMBERS = False


# ---------------------------------------------------------------------------
# Markdown shaping
# ---------------------------------------------------------------------------

# The writer is instructed (agents/writer.py) to put any gaps/caveats under
# a final "## Notes (not part of the resume)" heading. Match any Notes
# heading at any level so a reworded variant still gets caught.
_NOTES_HEADING_RE = re.compile(r"^#{1,6}\s*Notes\b.*$", re.IGNORECASE | re.MULTILINE)

# A trailing horizontal rule that existed only to fence the notes off.
_TRAILING_RULE_RE = re.compile(r"(?:\n[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*)+\s*$")

# A standalone line that is entirely bold — how the writer renders degree
# and certification titles, where a `###` heading would be the structurally
# equivalent form.
_BOLD_LINE_RE = re.compile(r"^\*\*(.+?)\*\*$")

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_POINT = rf"(?:{_MONTH}\s+)?\d{{4}}"
# A trailing meta-line segment that is purely a date or date range —
# "Apr 2023 – Nov 2023", "2024 – 2026", "Jan 2022 – Present", "Aug 2026".
# Anchored on both ends on purpose: a segment like "Team size: 15+" or
# "CGPA 8.7" must not be mistaken for a date and hoisted out of the line.
_DATE_RE = re.compile(
    rf"^(?:{_POINT}\s*(?:–|—|-|‐|to)\s*(?:{_POINT}|Present|Current|Now|Ongoing)|{_POINT})$",
    re.IGNORECASE,
)

_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<br\s*/?>")


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


def _strip_notes(markdown_text: str) -> str:
    """
    Remove the writer's "Notes (not part of the resume)" section.

    The notes are kept in the saved Markdown — they're the writer's honest
    account of where the candidate doesn't match the posting, which is
    worth reading before an interview. They are not resume content, and
    rendering them appended several paragraphs of internal commentary to
    the employer-facing PDF and blew past the page cap.
    """
    match = _NOTES_HEADING_RE.search(markdown_text)
    if not match:
        return markdown_text

    body = markdown_text[: match.start()]
    body = _TRAILING_RULE_RE.sub("", body)
    return body.rstrip() + "\n"


def extract_notes(markdown_text: str) -> str:
    """
    Return the writer's notes section (Markdown, heading excluded), or "".

    The counterpart to _strip_notes: what that function removes from the
    PDF, this one hands to the review HTML, so the writer's flagged gaps
    end up somewhere Nate actually reads them rather than being silently
    discarded. Called by main.py.
    """
    match = _NOTES_HEADING_RE.search(markdown_text)
    if not match:
        return ""
    return markdown_text[match.end():].strip()


def _split_header(markdown_text: str) -> tuple[str, str]:
    """
    Split the leading name/tagline/contact block from the sectioned body.

    Everything before the first `##` heading is the header. Returns
    ("", markdown_text) when the resume has no such block, so a resume
    that opens straight into a section still renders.
    """
    lines = markdown_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("## "):
            return "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
    return "", markdown_text


def _promote_bold_entry_titles(body_md: str) -> str:
    """
    Turn a fully-bold first line followed by a detail line into an `###`
    heading plus its meta line.

    The writer renders Experience entries as `### Title` + meta line, but
    Education and Certifications entries as `**Degree**` + meta line —
    structurally identical, visually not, because one becomes a heading and
    the other a paragraph. Normalizing them here means every entry in the
    document gets the same typographic treatment and the same right-aligned
    date, rather than Education quietly rendering as body text.

    Requires a following line so that a lone bold line (the header tagline,
    a bold lead-in) is left alone, and matches only when the whole line is
    bold, so "**Skills:** Python, ..." is untouched.
    """
    blocks = re.split(r"\n\s*\n", body_md)
    out = []
    for block in blocks:
        lines = block.split("\n")
        match = _BOLD_LINE_RE.match(lines[0].strip())
        if match and len(lines) > 1 and lines[1].strip():
            lines[0] = f"### {match.group(1).strip()}"
            block = "\n".join(lines)
        out.append(block)
    return "\n\n".join(out)


def _lift_date(meta_html: str) -> tuple[str, str]:
    """
    Pull a trailing date segment off an entry's meta line.

    "Quest Global — client: X | Team size: 15+ | Apr 2023 – Nov 2023"
    becomes ("Quest Global — client: X | Team size: 15+", "Apr 2023 – Nov 2023")
    so the date can be set flush right on the title line. Returns
    (meta_html, "") unchanged whenever the last segment isn't unambiguously
    a date — a meta line that reads oddly is a much worse failure than a
    date that stays inline.
    """
    chunks = _BR_RE.split(meta_html)
    tail = chunks[-1].strip()
    if not tail:
        return meta_html, ""

    head, sep, last = tail.rpartition("|")
    candidate = last.strip()
    if not _DATE_RE.match(_strip_tags(candidate)):
        return meta_html, ""

    if sep:
        chunks[-1] = head.strip()
    else:
        chunks = chunks[:-1]

    rebuilt = "<br />".join(chunk.strip() for chunk in chunks if chunk.strip())
    return rebuilt, candidate


def _shape_entries(html: str) -> str:
    """
    Give each `<h3>` + following paragraph the shape of a resume entry:
    role on the left, date flush right, employer/location on a muted line
    beneath.

    The date span is emitted BEFORE the role text because it is floated
    right — a float is positioned on the line box it opens, so putting it
    first is what keeps it on the title's own line rather than dropping to
    the next one.
    """
    pattern = re.compile(r"<h3>(.*?)</h3>\s*<p>(.*?)</p>", re.DOTALL)

    def replace(match: re.Match) -> str:
        title, meta = match.group(1).strip(), match.group(2).strip()
        meta, date = _lift_date(meta)

        date_html = f'<span class="date">{date}</span>' if date else ""
        meta_html = f'\n<p class="entry-meta">{meta}</p>' if meta else ""
        return f'<h3>{date_html}<span class="role">{title}</span></h3>{meta_html}'

    return pattern.sub(replace, html)


def _tag_kv_paragraphs(html: str) -> str:
    """
    Tag "**Label:** value" paragraphs — the Skills section's grouped rows —
    so they can be spaced as a list of labelled rows rather than as prose.
    """
    return re.sub(r"<p>(<strong>[^<]*:</strong>)", r'<p class="kv">\1', html)


def _render_header_html(header_md: str) -> str:
    """
    Render the name/tagline/contact block.

    Parsed LINE BY LINE rather than through the Markdown converter,
    because the writer's output shape here is not dependable. It is
    instructed to reproduce the template's contact block, and a template
    whose name is set in caps produces a bare line — 'GOKUL NATH KUNNATH
    KANDY' with no '#' — which Markdown reads as body text, not a
    heading. Worse, with nl2br and no blank lines between them the four
    header lines collapse into ONE paragraph, so nothing downstream can
    tell the name from the phone number. That shipped a resume with the
    candidate's name set at 9pt muted grey.

    The structural contract is positional and holds for every shape the
    writer produces: first line is the name, an optional second line is
    the professional tagline, everything after is contact detail. A
    leading '#' is stripped if present, so an h1-style header works too.
    """
    lines = [line.strip() for line in header_md.splitlines() if line.strip()]
    if not lines:
        return ""

    def inline(text: str) -> str:
        """Convert one line's inline Markdown, without the wrapping <p>."""
        html = md_lib.markdown(text, extensions=["extra"]).strip()
        return re.sub(r"^<p>(.*)</p>$", r"\1", html, flags=re.DOTALL)

    name = re.sub(r"^#{1,6}\s*", "", lines[0]).strip()
    name = re.sub(r"^\*\*(.*)\*\*$", r"\1", name).strip()
    parts = [f"<h1>{inline(name)}</h1>"]

    rest = lines[1:]
    # A tagline is a short line of prose. Contact lines carry separators,
    # an email, or a URL — so anything with those markers is contact,
    # which keeps a template that omits the tagline from styling its
    # contact line at tagline size.
    if rest and not re.search(r"[|@]|https?://|\bwww\.|\.com\b", rest[0]):
        tagline = re.sub(r"^\*\*(.*)\*\*$", r"\1", rest[0]).strip()
        parts.append(f'<p class="tagline">{inline(tagline)}</p>')
        rest = rest[1:]

    parts.extend(f'<p class="contact">{inline(line)}</p>' for line in rest)

    return '<header class="resume-header">\n' + "\n".join(parts) + "\n</header>"


def _prepare_html(resume_markdown: str) -> str:
    """Markdown in, styled-and-tagged resume HTML out."""
    cleaned = _strip_notes(resume_markdown)
    header_md, body_md = _split_header(cleaned)

    body_html = md_lib.markdown(
        _promote_bold_entry_titles(body_md),
        extensions=["extra", "sane_lists", "nl2br"],
    )
    body_html = _tag_kv_paragraphs(_shape_entries(body_html))

    return f'{_render_header_html(header_md)}\n<main class="resume-body">\n{body_html}\n</main>'


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
# Tokens are substituted by name (%%TOKEN%%) rather than with str.format,
# so CSS braces stay single and readable instead of doubled.
_PAGE_NUMBER_CSS = """
    @bottom-center {
      content: counter(page) " / " counter(pages);
      font-family: %%FONT_BODY%%;
      font-size: 7.5pt;
      color: %%RULE%%;
      letter-spacing: 0.04em;
    }
"""

_RESUME_PDF_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>%%TITLE%%</title>
<style>
  /* Explicit page size/margins so the resume's page count is deterministic
     and the writer's max-3/preferred-2-page constraint is something this
     renderer actually enforces, not just a suggestion in a prompt. Letter
     size matches North American resume convention (this project targets
     Canadian job applications). */
  @page {
    size: Letter;
    margin: %%PAGE_MARGIN%%;
%%PAGE_NUMBERS%%  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    font-family: %%FONT_BODY%%;
    font-size: %%BASE_SIZE%%;
    line-height: %%LINE_HEIGHT%%;
    color: %%TEXT%%;
    /* Ragged right, never justified: WeasyPrint doesn't hyphenate by
       default, so justified text opens rivers of whitespace between the
       long technical terms a resume like this is full of. */
    text-align: left;
    orphans: 2;
    widows: 2;
  }

  /* ---------- Header: name, tagline, contact ----------
     Matched to data/input/sample_resume.pdf: flush left, and NO rule under
     the block. The sample's only horizontal rules sit under the section
     headings, so adding one here doubled up the dividers and made the
     header look like a section of its own. */
  .resume-header {
    text-align: %%HEADER_ALIGN%%;
    padding-bottom: 0;
    margin-bottom: 2pt;
  }
  .resume-header h1 {
    font-family: %%FONT_HEADINGS%%;
    /* Large, heavy and TIGHTLY tracked — the sample sets an all-caps name
       at roughly twice body size with normal letter-spacing. Opening the
       tracking here reads as horizontally stretched, not spacious; the
       breathing room comes from the margin below. */
    font-size: 20pt;
    font-weight: 700;
    letter-spacing: 0;
    line-height: 1.15;
    margin: 0 0 6pt 0;
    color: %%ACCENT%%;
  }
  /* Regular weight, not bold: in the sample this line is clearly lighter
     than the name and only a little larger than the contact text. */
  .tagline {
    font-size: 11.5pt;
    font-weight: 400;
    letter-spacing: 0;
    color: %%TEXT%%;
    margin: 0 0 6pt 0;
  }
  .contact {
    font-size: 8.5pt;
    line-height: 1.5;
    color: %%TEXT%%;
    margin: 0;
  }
  /* The work-authorization line — last in the block — is italic in the
     sample, which sets it apart from the contact details above it. */
  .contact:last-child { font-style: italic; }
  .contact a { color: %%TEXT%%; }

  /* ---------- Section headers ---------- */
  /* Section headers: navy caps over a navy rule, as in the sample. */
  h2 {
    font-family: %%FONT_HEADINGS%%;
    font-size: 9.8pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: %%ACCENT%%;
    margin: 12pt 0 5pt 0;
    padding-bottom: 2.5pt;
    border-bottom: 0.9pt solid %%ACCENT%%;
    /* Never leave a section header alone at the foot of a page. */
    break-after: avoid;
    page-break-after: avoid;
  }
  /* The header block already supplies the space above the first section. */
  .resume-body > h2:first-child { margin-top: 9pt; }

  /* ---------- Entries: role / degree / project ---------- */
  h3 {
    font-family: %%FONT_HEADINGS%%;
    font-size: 10.5pt;
    font-weight: 700;
    margin: 7.5pt 0 0 0;
    break-after: avoid;
    page-break-after: avoid;
  }
  /* Contain the floated date so it can't spill onto the meta line. */
  h3::after { content: ""; display: block; clear: both; }
  h3 .date {
    float: right;
    font-size: 9pt;
    font-weight: 400;
    color: %%MUTED%%;
    letter-spacing: 0.01em;
    margin-left: 12pt;
    white-space: nowrap;
  }

  /* Employer, client, location — the line directly under an entry title.
     Deliberately NO break-after:avoid here. `avoid` chains: h2 -> h3 ->
     meta -> next h3 -> next meta welds an entire dateless section (Education,
     Certifications) into one unbreakable block, which then jumps to the next
     page and leaves a two-inch hole at the foot of the previous one. The
     h3 and ul rules already keep a title with its meta line and first
     bullet, which is the grouping that actually matters. */
  .entry-meta {
    font-size: 9.3pt;
    color: %%MUTED%%;
    margin: 1pt 0 3pt 0;
  }

  p { margin: 0 0 4pt 0; }

  /* Skills rows: "Machine Learning & AI: ..." */
  .kv { margin: 0 0 3.5pt 0; }
  .kv strong { color: %%TEXT%%; }

  /* ---------- Bullets ---------- */
  ul {
    margin: 0 0 6pt 0;
    /* Marker sits in the margin, wrapped lines align under the text, not
       under the bullet. */
    padding-left: 12pt;
    list-style-type: disc;
    list-style-position: outside;
    break-before: avoid;
    page-break-before: avoid;
  }
  li {
    margin: 0 0 2.5pt 0;
    padding-left: 2pt;
    /* Keep a bullet's wrapped lines together across a page break. */
    break-inside: avoid;
    page-break-inside: avoid;
  }
  li:last-child { margin-bottom: 0; }
  li > ul { margin-top: 2.5pt; }

  a { color: inherit; text-decoration: none; }
  strong { font-weight: 700; color: %%TEXT%%; }
  em { font-style: italic; }
  /* Section separation is the job of the h2 rule; a stray Markdown `---`
     would only double it up. */
  hr { display: none; }
</style>
</head>
<body>
%%BODY%%
</body>
</html>
"""


def _fill(template: str, values: dict) -> str:
    for key, value in values.items():
        template = template.replace(f"%%{key.upper()}%%", value)
    return template


def _build_document(body_html: str, title: str) -> str:
    page_numbers = _fill(_PAGE_NUMBER_CSS, STYLE) if SHOW_PAGE_NUMBERS else ""
    return _fill(
        _RESUME_PDF_TEMPLATE,
        {**STYLE, "title": title, "body": body_html, "page_numbers": page_numbers},
    )


def render_resume_pdf(resume_markdown: str, output_path: str, title: str = "Tailored Resume") -> str:
    """
    Render the tailored resume Markdown directly to a PDF file.

    The writer's "Notes (not part of the resume)" section is excluded from
    the PDF; it remains in the Markdown the caller has already saved.

    Args:
        resume_markdown: The approved resume content (Markdown).
        output_path: Where to write the .pdf file.
        title: PDF document title.

    Returns:
        The output_path, for convenience chaining.

    Raises:
        RuntimeError: if Markdown-to-HTML conversion or the WeasyPrint
            PDF render fails (e.g. missing system libraries like Pango,
            or a malformed output_path). The resume content itself
            (Markdown) has already been saved by the caller by this
            point, so a PDF render failure loses only the PDF format,
            not the resume content.
    """
    try:
        body_html = _prepare_html(resume_markdown)
    except Exception as exc:  # noqa: BLE001 — 3rd-party markdown parser, guard broadly
        log.error("pdf_renderer: failed converting resume Markdown to HTML: %s", exc, exc_info=True)
        raise RuntimeError(f"pdf_renderer: could not convert resume Markdown to HTML — {exc}") from exc

    full_html = _build_document(body_html, title)

    # Imported here rather than at module scope on purpose: WeasyPrint
    # resolves its native Pango/glib/cairo bindings at IMPORT time, so a
    # missing system library raises OSError on `from weasyprint import HTML`
    # — which, at module scope, would kill the whole pipeline at startup
    # before a single agent ran. Deferring it here keeps that failure
    # contained to the PDF step, which is what the rest of this module (and
    # main.py's error handling) already assumes: by the time this runs, the
    # Markdown resume is on disk, so an environment problem costs the PDF
    # format only, never the resume content.
    try:
        from weasyprint import HTML
    except (OSError, ImportError) as exc:
        log.error("pdf_renderer: WeasyPrint could not be imported: %s", exc, exc_info=True)
        raise RuntimeError(
            f"pdf_renderer: WeasyPrint could not load its native libraries ({exc}). "
            "This is an environment setup issue, not a code problem — on macOS run "
            "`brew install pango`, and if you're in a conda env also set "
            "DYLD_LIBRARY_PATH (see README.md 'Setup'). The resume Markdown has "
            "already been saved separately, so no content was lost."
        ) from exc

    try:
        # Lay the document out first, then write it. Rendering in two steps
        # costs nothing extra and makes the page count observable — the
        # writer is told to stay within the cap, but until now nothing
        # verified whether the rendered result actually did.
        document = HTML(string=full_html).render()
        page_count = len(document.pages)
        document.write_pdf(output_path)

        if page_count > RESUME_MAX_PAGES:
            log.error(
                "pdf_renderer: rendered resume is %d pages, over the %d-page cap.",
                page_count, RESUME_MAX_PAGES,
            )
            log.info(
                "  Warning: the resume rendered to %d pages, over the %d-page cap "
                "(%d preferred). Cut the lowest-value content, or tighten "
                "STYLE['page_margin']/['base_size'] in pdf_renderer.py.",
                page_count, RESUME_MAX_PAGES, RESUME_PREFERRED_PAGES,
            )
        elif page_count > RESUME_PREFERRED_PAGES:
            log.info(
                "  Note: the resume rendered to %d pages — within the %d-page cap, "
                "but %d is preferred.",
                page_count, RESUME_MAX_PAGES, RESUME_PREFERRED_PAGES,
            )
        else:
            log.info("  Resume rendered to %d page(s).", page_count)
    except OSError as exc:
        # Most common real-world cause: missing system libraries (Pango,
        # glib) that WeasyPrint's cffi bindings need — this is an
        # environment setup issue, not a code bug, so say so explicitly.
        log.error("pdf_renderer: WeasyPrint failed to render PDF to %s: %s", output_path, exc, exc_info=True)
        raise RuntimeError(
            f"pdf_renderer: WeasyPrint could not render the PDF ({exc}). This is "
            "usually a missing system library (Pango/glib) rather than a code "
            "problem — see README.md 'Setup' for the brew/apt install step. The "
            "resume Markdown has already been saved separately, so no content was lost."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — last-resort guard around a 3rd-party renderer
        log.error("pdf_renderer: unexpected error rendering PDF to %s: %s", output_path, exc, exc_info=True)
        raise RuntimeError(f"pdf_renderer: unexpected error rendering PDF — {exc}") from exc

    return output_path
