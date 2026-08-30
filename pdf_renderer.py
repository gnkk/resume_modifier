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
"""

import markdown as md_lib
from weasyprint import HTML

_RESUME_PDF_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{
    --text: #1a1a1a;
    --accent: #1f4e79;
    --border: #ddd;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Helvetica Neue", Arial, sans-serif;
    color: var(--text);
    line-height: 1.55;
  }}
  .page {{ padding: 0.4in; }}
  h1 {{
    font-size: 1.9em;
    margin: 0 0 4px 0;
    color: var(--accent);
    border-bottom: 2px solid var(--accent);
    padding-bottom: 8px;
  }}
  h2 {{
    font-size: 1.15em;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--accent);
    margin-top: 20px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
  }}
  h3 {{ font-size: 1.02em; margin-bottom: 2px; margin-top: 14px; }}
  p {{ margin: 6px 0; }}
  ul {{ margin: 6px 0 12px 0; padding-left: 22px; }}
  li {{ margin-bottom: 4px; }}
  a {{ color: var(--accent); text-decoration: none; }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 18px 0; }}
  strong {{ color: #000; }}
</style>
</head>
<body>
  <div class="page">
{body}
  </div>
</body>
</html>
"""


def render_resume_pdf(resume_markdown: str, output_path: str, title: str = "Tailored Resume") -> str:
    """
    Render the tailored resume Markdown directly to a PDF file.

    Args:
        resume_markdown: The approved resume content (Markdown).
        output_path: Where to write the .pdf file.
        title: PDF document title.

    Returns:
        The output_path, for convenience chaining.
    """
    body_html = md_lib.markdown(resume_markdown, extensions=["extra", "sane_lists"])
    full_html = _RESUME_PDF_TEMPLATE.format(title=title, body=body_html)
    HTML(string=full_html).write_pdf(output_path)
    return output_path
