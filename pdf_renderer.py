"""
PDF renderer.

Converts the same HTML produced by html_renderer.py into a PDF, so
the two output formats are always visually consistent — one source
of truth, two renderings.

Uses WeasyPrint: pure-Python-installable via pip (no external CLI
binary required on PATH), and the standard choice for turning styled
HTML/CSS into print-quality PDF. On macOS it needs a couple of system
libraries for text/font rendering (Pango, cairo, gdk-pixbuf) — see
README.md "Setup" for the one-time `brew install` step. This is a
one-time environment setup, not a per-run dependency.
"""

from weasyprint import HTML


def render_resume_pdf(resume_html: str, output_path: str) -> str:
    """
    Render a full HTML document to a PDF file.

    Args:
        resume_html: Full HTML document string, as produced by
            html_renderer.render_resume_html().
        output_path: Where to write the .pdf file.

    Returns:
        The output_path, for convenience chaining.
    """
    HTML(string=resume_html).write_pdf(output_path)
    return output_path
