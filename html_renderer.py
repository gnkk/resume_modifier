"""
HTML renderer.

Converts the final tailored resume (Markdown) into a single, clean,
vertically-scrollable HTML page. This same HTML is also the source
for the PDF export (see pdf_renderer.py), so the two outputs always
match exactly.

Uses the `markdown` library for Markdown -> HTML conversion, wrapped
in a print-friendly, single-column template with generous line-height
for readability while scrolling.
"""

import markdown as md_lib

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --text: #1a1a1a;
    --muted: #555;
    --accent: #1f4e79;
    --border: #ddd;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 0;
    background: #f4f4f4;
    font-family: "Helvetica Neue", Arial, sans-serif;
    color: var(--text);
    line-height: 1.55;
  }}
  .page {{
    max-width: 820px;
    margin: 0 auto;
    background: #fff;
    padding: 48px 56px;
    min-height: 100vh;
  }}
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
    margin-top: 28px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
  }}
  h3 {{
    font-size: 1.02em;
    margin-bottom: 2px;
    margin-top: 16px;
  }}
  p {{
    margin: 6px 0;
    color: var(--text);
  }}
  ul {{
    margin: 6px 0 12px 0;
    padding-left: 22px;
  }}
  li {{
    margin-bottom: 4px;
  }}
  a {{
    color: var(--accent);
    text-decoration: none;
  }}
  hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
  }}
  strong {{ color: #000; }}
  @media print {{
    body {{ background: #fff; }}
    .page {{ box-shadow: none; padding: 0.4in; max-width: 100%; }}
  }}
  @media (max-width: 600px) {{
    .page {{ padding: 24px 20px; }}
  }}
</style>
</head>
<body>
  <div class="page">
{body}
  </div>
</body>
</html>
"""


def render_resume_html(resume_markdown: str, title: str = "Tailored Resume") -> str:
    """
    Render the final tailored resume Markdown as a full standalone
    HTML page, single-column and readable by scrolling top to bottom.

    Args:
        resume_markdown: The approved resume content (Markdown).
        title: HTML <title> / page heading fallback.

    Returns:
        A full HTML document as a string.
    """
    body_html = md_lib.markdown(
        resume_markdown,
        extensions=["extra", "sane_lists"],
    )
    return PAGE_TEMPLATE.format(title=title, body=body_html)
