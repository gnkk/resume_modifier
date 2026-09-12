"""
Text file reader tool.

Reads a plain .txt file — used for a job description supplied
directly as a file, rather than found via web search.
"""

from logger_setup import get_logger

log = get_logger(__name__)


def read_text_file(file_path: str) -> str:
    """
    Read a plain text file.

    Args:
        file_path: Path to the .txt file.

    Returns:
        The file's full text content. On failure (missing file,
        permission error, bad encoding), returns a short
        "[read_text_file failed: ...]" string instead of raising, so a
        tool-calling agent sees this as normal tool output and can
        react rather than the whole process crashing.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        log.error("read_text_file: file not found: %s", file_path)
        return f"[read_text_file failed: no file found at '{file_path}'. Check the path and try again.]"
    except UnicodeDecodeError as exc:
        log.error("read_text_file: encoding error reading %s: %s", file_path, exc, exc_info=True)
        return f"[read_text_file failed: '{file_path}' is not valid UTF-8 text — {exc}]"
    except OSError as exc:
        log.error("read_text_file: OS error reading %s: %s", file_path, exc, exc_info=True)
        return f"[read_text_file failed: could not read '{file_path}' — {exc}]"


# --- Anthropic tool schema for this function ---
TEXT_READER_TOOL_SCHEMA = {
    "name": "read_text_file",
    "description": (
        "Read a plain .txt file from disk, such as a job description "
        "the user has saved directly rather than one to be found via "
        "web search. If the file is missing or unreadable, this returns "
        "a short '[read_text_file failed: ...]' message instead of "
        "erroring out — treat that as a signal to inform the user rather "
        "than a fatal error."
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
