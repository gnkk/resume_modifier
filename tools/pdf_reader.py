"""
PDF reader tool.

Extracts plain text from a PDF file (e.g. your existing resume,
or a job posting saved as PDF). Kept dependency-light: pypdf only.
"""

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from logger_setup import get_logger

log = get_logger(__name__)


def read_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file_path: Path to the .pdf file.

    Returns:
        The extracted text as a single string, pages joined by
        double newlines. On failure (missing file, corrupt/encrypted
        PDF, unreadable pages), returns a short "[read_pdf failed: ...]"
        string instead of raising, so a tool-calling agent sees this as
        normal tool output and can react (e.g. tell the user the file
        couldn't be read) rather than the whole process crashing on a
        single bad file.
    """
    try:
        reader = PdfReader(file_path)
    except FileNotFoundError:
        log.error("read_pdf: file not found: %s", file_path)
        return f"[read_pdf failed: no file found at '{file_path}'. Check the path and try again.]"
    except PdfReadError as exc:
        log.error("read_pdf: could not parse PDF %s: %s", file_path, exc, exc_info=True)
        return f"[read_pdf failed: '{file_path}' does not look like a valid/readable PDF — {exc}]"
    except Exception as exc:  # noqa: BLE001 — last-resort guard around a 3rd-party parser
        log.error("read_pdf: unexpected error opening %s: %s", file_path, exc, exc_info=True)
        return f"[read_pdf failed: unexpected error opening '{file_path}' — {exc}]"

    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 — a single bad page shouldn't lose the rest
            log.error("read_pdf: failed extracting text from page %d of %s: %s", i, file_path, exc, exc_info=True)
            text = ""
        pages_text.append(text.strip())

    if not any(pages_text):
        log.error("read_pdf: extracted no text at all from %s (possibly a scanned/image-only PDF)", file_path)
        return (
            f"[read_pdf failed: '{file_path}' produced no extractable text. "
            "It may be a scanned/image-only PDF with no text layer.]"
        )

    return "\n\n".join(pages_text)


# --- Anthropic tool schema for this function ---
# Import this into your orchestrator agent's `tools` list.
PDF_READER_TOOL_SCHEMA = {
    "name": "read_pdf",
    "description": (
        "Extract text content from a PDF file on disk, such as the "
        "user's existing resume or a saved job posting. If the file is "
        "missing, corrupt, or has no extractable text, this returns a "
        "short '[read_pdf failed: ...]' message instead of erroring out — "
        "treat that as a signal to inform the user rather than a fatal error."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the PDF file to read.",
            }
        },
        "required": ["file_path"],
    },
}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pdf_reader.py <path_to_pdf>")
        sys.exit(1)

    print(read_pdf(sys.argv[1]))
