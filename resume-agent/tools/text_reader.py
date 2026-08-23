"""
Text file reader tool.

Reads a plain .txt file — used for a job description supplied
directly as a file, rather than found via web search.
"""


def read_text_file(file_path: str) -> str:
    """
    Read a plain text file.

    Args:
        file_path: Path to the .txt file.

    Returns:
        The file's full text content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# --- Anthropic tool schema for this function ---
TEXT_READER_TOOL_SCHEMA = {
    "name": "read_text_file",
    "description": (
        "Read a plain .txt file from disk, such as a job description "
        "the user has saved directly rather than one to be found via "
        "web search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the .txt file to read.",
            }
        },
        "required": ["file_path"],
    },
}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python text_reader.py <path_to_txt>")
        sys.exit(1)

    print(read_text_file(sys.argv[1]))
