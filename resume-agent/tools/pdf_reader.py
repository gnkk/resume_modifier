"""
PDF reader tool.

Extracts plain text from a PDF file (e.g. your existing resume,
or a job posting saved as PDF). Kept dependency-light: pypdf only.
"""

from pypdf import PdfReader


def read_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file_path: Path to the .pdf file.

    Returns:
        The extracted text as a single string, pages joined by
        double newlines.
    """
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text.strip())
    return "\n\n".join(pages_text)


# --- Anthropic tool schema for this function ---
# Import this into your orchestrator agent's `tools` list.
PDF_READER_TOOL_SCHEMA = {
    "name": "read_pdf",
    "description": (
        "Extract text content from a PDF file on disk, such as the "
        "user's existing resume or a saved job posting."
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
