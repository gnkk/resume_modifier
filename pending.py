"""
Paused-run state for the suggestion gate.

WHY THIS EXISTS. The judge's suggestions used to arrive after the PDF had
already been rendered — advice about a document that was finished, which
could only be acted on by running the whole pipeline again. The gate moves
the decision in front of the render: the run stops with a draft and a list
of suggestions, you tick the ones you want, and only then is the PDF
written.

WHY IT GOES TO DISK RATHER THAN BLOCKING. A run is launched by webapp/app.py
as a subprocess whose stdout is streamed to the browser. Waiting for the
user inside that subprocess would mean a process parked indefinitely
holding the single run lock, with the Flask side having to push a decision
back into its stdin. Writing the state out and exiting instead makes the
pause free: nothing is held open, the browser can be closed and reopened,
and the finalize half is an ordinary second invocation (finalize.py) that
reads the file back.

Each job in a multi-job run gets its own pending file, so three jobs means
three independent decisions rather than one all-or-nothing batch.

Files live in DATA_WORK_DIR, never in data/output/ — they are machinery,
not deliverables. A pending file is deleted once its resume renders.
"""

import json
import os
import time

from config import DATA_WORK_DIR
from logger_setup import get_logger

log = get_logger(__name__)

# Pending state older than this is stale: the run that wrote it was
# abandoned, and the postings it targeted have very likely moved on. Kept
# generous — a decision left overnight is normal use, not abandonment.
PENDING_TTL_SECONDS = 7 * 24 * 60 * 60


def _pending_dir() -> str:
    return os.path.join(DATA_WORK_DIR, "pending")


def _path_for(pending_id: str) -> str:
    # Basename only: pending_id reaches this from an HTTP request, and a
    # path separator in it would otherwise write or read outside the
    # working directory.
    return os.path.join(_pending_dir(), f"{os.path.basename(pending_id)}.json")


def write_pending(pending_id: str, state: dict) -> str:
    """
    Save one job's paused state and return the file path.

    Raises:
        RuntimeError: if the state can't be written. Unlike most failures
            in this pipeline this one is fatal for the job — the draft
            exists only in memory at this point, and losing it means the
            writer/judge cycles have to be paid for again.
    """
    os.makedirs(_pending_dir(), exist_ok=True)
    path = _path_for(pending_id)
    payload = {"pending_id": pending_id, "created_at": time.time(), **state}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    except (OSError, TypeError) as exc:
        log.error("pending: failed writing %s: %s", path, exc, exc_info=True)
        raise RuntimeError(
            f"pending: could not save the paused run to '{path}' — {exc}. The "
            "draft exists only in memory, so it would be lost."
        ) from exc
    return path


def read_pending(pending_id: str) -> dict:
    """
    Load one job's paused state.

    Raises:
        FileNotFoundError: if there is no such pending run — usually
            because it was already finalized.
    """
    path = _path_for(pending_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No pending run '{pending_id}'. It may already have been "
            "finalized, or the working directory was cleared."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_pending() -> list[dict]:
    """
    Every pending run awaiting a decision, newest first.

    Returns a light summary per entry rather than the full state — this
    feeds a list in the UI, which needs the job title and the suggestions,
    not the candidate context and the whole draft.
    """
    directory = _pending_dir()
    if not os.path.isdir(directory):
        return []

    entries = []
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            # One unreadable file must not hide the rest.
            log.error("pending: skipping unreadable %s: %s", name, exc)
            continue

        if time.time() - state.get("created_at", 0) > PENDING_TTL_SECONDS:
            continue

        job = state.get("job") or {}
        review = state.get("resume_review") or {}
        entries.append(
            {
                "pending_id": state.get("pending_id", name[:-5]),
                "created_at": state.get("created_at"),
                "job_title": job.get("job_title"),
                "company": job.get("company"),
                "url": job.get("url"),
                "fitness_score": review.get("fitness_score"),
                "fitness_summary": review.get("fitness_summary"),
                "approved": review.get("approved"),
                "suggestions": state.get("suggestions") or [],
                "gaps": review.get("gaps") or [],
            }
        )

    entries.sort(key=lambda e: e.get("created_at") or 0, reverse=True)
    return entries


def clear_pending(pending_id: str) -> None:
    """Delete a pending run's state once its resume has been rendered."""
    try:
        os.remove(_path_for(pending_id))
    except FileNotFoundError:
        pass
    except OSError as exc:
        # Not fatal: a leftover file costs a stale row in the pending list,
        # which the TTL eventually clears anyway.
        log.error("pending: could not remove state for %s: %s", pending_id, exc)
