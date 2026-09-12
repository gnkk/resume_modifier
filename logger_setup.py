"""
Central logging setup.

Two destinations, same verbosity, different detail:
  - Console: INFO and above — the run narration the pipeline has always
    printed (job found, scores, cycle progress), just routed through the
    logging module instead of raw print() calls so it's consistent and
    filterable.
  - Log file: INFO and above too, with timestamps, levels and module
    names attached. One timestamped file per run under
    data/output/logs/, so a run leaves a permanent, greppable record of
    what it actually did instead of scrolling off the terminal.

The file handler used to be ERROR-only. That made a failed run
diagnosable but a SUCCESSFUL one invisible — which angles were planned,
what the pool screened down to, what each judge cycle scored and why,
all gone the moment the terminal scrolled. Those are exactly the things
worth comparing across runs when tuning the agents' prompts.

One consequence worth knowing: several places in this project log a
detailed log.error() for the record and then a plain-language
log.info("Warning: ...") for the person watching. Both now land in the
file, so a handled failure appears twice — once with its traceback,
once in the narration. That is redundancy, not a bug.

Usage, in any module:
    from logger_setup import get_logger
    log = get_logger(__name__)
    log.info("some narration")       # console + file
    log.error("something failed", exc_info=True)   # console + file, with traceback

main.py calls configure_logging() once at startup, before any other
project import runs its own module-level get_logger() call, so the
root logger is fully configured (handlers attached) before anything
tries to log through it.
"""

import logging
import os
from datetime import datetime

try:
    from config import DATA_LOGS_DIR as _LOG_DIR
except ImportError:
    # config.py raises RuntimeError (not ImportError) if API keys are
    # missing, so this fallback only triggers on a genuine import issue
    # (e.g. this module used standalone outside the project). Keeps
    # logging usable even then rather than crashing at import time.
    _LOG_DIR = os.path.join("data", "output", "logs")

_CONFIGURED = False
_LOG_FILE_PATH = None
_RUN_TIMESTAMP = None


def configure_logging(log_dir: str = _LOG_DIR) -> str:
    """
    Attach a console (INFO+) and file (INFO+) handler to the root
    logger. Safe to call more than once — only configures on the
    first call, returns the existing log file path on subsequent calls.

    Args:
        log_dir: Directory to write the timestamped log file into.
            Created if it doesn't exist.

    Returns:
        The path of the log file for this run.
    """
    global _CONFIGURED, _LOG_FILE_PATH, _RUN_TIMESTAMP

    if _CONFIGURED:
        return _LOG_FILE_PATH

    os.makedirs(log_dir, exist_ok=True)
    _RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE_PATH = os.path.join(log_dir, f"run_{_RUN_TIMESTAMP}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)  # lowest level any handler below might want

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(_LOG_FILE_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _CONFIGURED = True
    return _LOG_FILE_PATH


def get_logger(name: str) -> logging.Logger:
    """
    Get a module-scoped logger. If configure_logging() hasn't run yet
    (e.g. this module is imported/used outside main.py, such as in a
    standalone script or test), fall back to a console-only INFO
    logger so log calls never silently vanish or crash.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A standard library logging.Logger.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


def get_log_file_path() -> str | None:
    """Return the current run's log file path, or None if not yet configured."""
    return _LOG_FILE_PATH


def get_run_timestamp() -> str | None:
    """
    Return this run's timestamp string (e.g. '20260831_143022'), or None
    if configure_logging() hasn't run yet. Callers that generate their
    own versioned/timestamped output files (see main.py) should reuse
    this rather than generating a second, slightly different timestamp,
    so a run's log file and its output files share one identifying
    suffix and are trivially easy to correlate.
    """
    if not _CONFIGURED:
        configure_logging()
    return _RUN_TIMESTAMP
