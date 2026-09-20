"""
Local web UI for the resume agent.

A small Flask app that puts every tunable in one form, launches a run as
a subprocess, streams the log back live, and links the outputs.

WHY A SERVER AND NOT A STATIC PAGE. A browser file input hands JavaScript
a filename, never a path — "job_description.txt" with no indication of
where it lives. The pipeline needs a real path, so the file picker here
is server-side: the backend lists what is actually in data/input/ and the
form chooses from that. A static page could only ever build a command for
you to copy.

HOW SETTINGS REACH THE PIPELINE. Not by rewriting config.py. The form's
values go into the subprocess's ENVIRONMENT, and config.py reads each
tunable through _env_int/_env_bool/_env_str with the file's value as the
default. So config.py stays the single source of truth for what every
knob means and why, its comments survive, and an overridden run changes
nothing on disk.

SCOPE. This binds to 127.0.0.1 and runs pipeline invocations as
subprocesses with your API keys in their environment. It has no
authentication and is not built to face a network — it is a local control
panel for a tool you already run from your own terminal.
"""

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

app = Flask(__name__)


# --- Run state ---------------------------------------------------------
# One run at a time, deliberately. Concurrent runs would interleave in the
# shared log, both write timestamped files into data/output/, and both
# append to selected_jobs.json — and there is no use case for it on a
# single-user local tool.
_run_lock = threading.Lock()
_current = {"process": None, "lines": [], "status": "idle"}


# Which settings the form may override, and how each is parsed. Names match
# config.py EXACTLY — that is what makes the environment mechanism work
# without a translation table that could drift out of sync.
TUNABLES = {
    "JOB_SCRAPE_CEILING": "int",
    "BM25_KEEP": "int",
    "POOL_SCREEN_MIN_FIT": "int",
    "POOL_SCREEN_BATCH_SIZE": "int",
    "POOL_SCREEN_SNIPPET_CHARS": "int",
    "JOB_SEARCH_DAYS_OLD": "int",
    "JOB_SEARCH_SITES": "list",
    "JOBSPY_COUNTRY_INDEED": "str",
    "JOBSPY_FETCH_LINKEDIN_DESCRIPTIONS": "bool",
    "MAX_JOB_SEARCH_CYCLES": "int",
    "MAX_RESUME_REVISE_CYCLES": "int",
    "JUDGE_APPROVAL_SCORE": "int",
    "RESUME_APPROVAL_SCORE": "int",
    "MIN_VIABLE_JOB_SCORE": "int",
    "RESUME_MAX_PAGES": "int",
    "RESUME_PREFERRED_PAGES": "int",
    "SKIP_PREVIOUSLY_SELECTED": "bool",
    "JOB_MATCH_COUNT": "int",
    "MODEL_CONTEXT": "str",
    "MODEL_PLANNER": "str",
    "MODEL_SCREENER": "str",
    "MODEL_WRITER": "str",
    "MODEL_JUDGE": "str",
    "MODEL_JD_EXTRACT": "str",
    "MODEL_ELIGIBILITY": "str",
    "SUGGESTION_GATE": "bool",
}


def _defaults() -> dict:
    """Read current defaults straight out of config.py to populate the form."""
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        import config  # noqa: PLC0415 — late import so a missing .env surfaces here, not at startup

        values = {}
        for name in TUNABLES:
            value = getattr(config, name, "")
            values[name] = ",".join(value) if isinstance(value, list) else value
        return values
    except Exception as exc:  # noqa: BLE001 — a bad config should render the page, not 500 it
        app.logger.error("could not read config defaults: %s", exc)
        return {name: "" for name in TUNABLES}


def _list_files(directory: Path, suffixes: tuple[str, ...]) -> list[dict]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            {"name": p.name, "path": str(p.relative_to(PROJECT_ROOT))}
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in suffixes
        ),
        key=lambda f: f["name"].lower(),
    )


def _stream_process(process: subprocess.Popen) -> None:
    """Pump the subprocess's output into the buffer the browser polls."""
    for raw in process.stdout:
        _current["lines"].append(raw.rstrip("\n"))
    process.wait()
    _current["status"] = "finished" if process.returncode == 0 else f"exited {process.returncode}"


@app.route("/")
def index():
    return render_template(
        "index.html",
        defaults=_defaults(),
        resumes=_list_files(INPUT_DIR, (".pdf",)),
        jd_files=_list_files(INPUT_DIR, (".txt", ".md")),
    )


@app.route("/outputs")
def outputs():
    """
    Most recent output files, newest first, for the results panel.

    Walks subdirectories because each job now gets its own folder holding
    resume.pdf and review.html. The name shown is the path relative to
    data/output/, since "resume.pdf" on its own would be identical for
    every job in the list.
    """
    if not OUTPUT_DIR.is_dir():
        return jsonify([])
    files = sorted(
        (
            p for p in OUTPUT_DIR.rglob("*")
            if p.is_file() and p.suffix in (".pdf", ".html")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return jsonify(
        [
            {"name": str(p.relative_to(OUTPUT_DIR)), "size": p.stat().st_size}
            for p in files[:12]
        ]
    )


@app.route("/outputs/<path:filename>")
def output_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/run", methods=["POST"])
def run():
    """
    Launch a pipeline run.

    Returns 409 rather than queueing when one is already going: two
    concurrent runs would interleave in the log, race on
    selected_jobs.json, and confuse the streamed output. Refusing is
    clearer than silently serialising.
    """
    if not _run_lock.acquire(blocking=False):
        return jsonify({"error": "A run is already in progress."}), 409

    started = False
    try:
        payload = request.get_json(force=True) or {}
        resume = (payload.get("resume") or "").strip()
        mode = payload.get("mode", "both")
        target_role = (payload.get("target_role") or "").strip()
        job_source = (payload.get("job_source") or "").strip()

        if mode not in ("search", "write", "both"):
            return jsonify({"error": f"Unknown mode '{mode}'."}), 400
        if not resume:
            return jsonify({"error": "Choose a resume PDF."}), 400
        if mode == "write" and not job_source:
            return jsonify({"error": "Writing from a supplied job needs a posting — pick a file or paste a URL."}), 400
        if mode in ("search", "both") and not target_role:
            return jsonify({"error": "Searching needs a target role description."}), 400

        # main.py treats a third argument as "a job was supplied", which
        # implies write-only. So the argument is passed ONLY in write mode;
        # search and both are distinguished by RUN_MODE instead.
        argv = [sys.executable, "-u", "main.py", resume, target_role or "-"]
        if mode == "write":
            argv.append(job_source)

        env = os.environ.copy()
        env["RUN_MODE"] = mode
        settings = payload.get("settings") or {}
        for name, kind in TUNABLES.items():
            value = settings.get(name)
            if value is None or value == "":
                continue
            if kind == "bool":
                env[name] = "true" if value in (True, "true", "on", 1) else "false"
            else:
                env[name] = str(value)

        _current["lines"] = []
        _current["status"] = "running"
        process = subprocess.Popen(
            argv,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _current["process"] = process

        def watch():
            try:
                _stream_process(process)
            finally:
                _run_lock.release()

        threading.Thread(target=watch, daemon=True).start()
        started = True
        return jsonify({"started": True})
    except Exception as exc:  # noqa: BLE001
        app.logger.error("failed to start run: %s", exc, exc_info=True)
        return jsonify({"error": f"Could not start the run: {exc}"}), 500
    finally:
        # Released here on EVERY path that did not actually start a run —
        # validation failures included. Leaving it held on a 400 wedged the
        # lock permanently, so one empty form field made every later click
        # return 409 with no way back short of restarting the server. On
        # the success path watch() owns the release instead, because the
        # run is still going when this returns.
        if not started:
            _run_lock.release()


# --- Suggestion gate ---------------------------------------------------
# A gated run ends with its draft and the judge's suggestions parked on
# disk (see pending.py) instead of a PDF. These two routes are the other
# half: list what is waiting, then apply the boxes the user ticked.
#
# Finalizing runs in a subprocess for the same reason a run does — it
# makes a Claude call and renders a PDF, either of which can take longer
# than a request should hold a worker — and it takes the same run lock,
# because it writes into data/output/ exactly as a run does.

def _pipeline_module():
    """Import from the project root, which is not on sys.path by default."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import pending  # noqa: PLC0415 — late import; a missing .env surfaces per-request

    return pending


def _history_module():
    """Same late-import treatment for the selected-jobs ledger."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import job_history  # noqa: PLC0415

    return job_history


# --- Selected-jobs history ---------------------------------------------
# Jobs targeted in earlier runs are filtered out of the pool before
# screening, and that record is deliberately independent of data/output/ —
# deleting a job's folder does not put the posting back in play. These two
# routes are how you put it back deliberately.
#
# No subprocess here, unlike /run and /finalize: this reads and rewrites one
# small JSON file and returns in milliseconds, so it does not take the run
# lock either. A history edit during a run is harmless — the pool for that
# run was already filtered when it was built.

@app.route("/history")
def history():
    """Jobs on file, each with the row number the forget route expects."""
    try:
        return jsonify(_history_module().list_selections())
    except Exception as exc:  # noqa: BLE001 — an empty panel beats a 500
        app.logger.error("could not read job history: %s", exc, exc_info=True)
        return jsonify([])


@app.route("/history/forget", methods=["POST"])
def history_forget():
    """
    Drop entries so those postings can be picked again.

    Takes row numbers (as /history returns them) or free text matched
    against company, title and URL — the same keys the CLI takes, because
    two parsers for one operation is how they drift apart.
    """
    try:
        payload = request.get_json(force=True) or {}
        keys = payload.get("keys")
        if not isinstance(keys, list) or not keys:
            return jsonify({"error": "Nothing selected."}), 400
        removed = _history_module().forget_selections(keys)
        return jsonify({"removed": len(removed), "entries": removed})
    except Exception as exc:  # noqa: BLE001
        app.logger.error("could not edit job history: %s", exc, exc_info=True)
        return jsonify({"error": f"Could not edit the history: {exc}"}), 500


@app.route("/pending")
def pending_runs():
    """Drafts waiting on a suggestion decision, newest first."""
    try:
        return jsonify(_pipeline_module().list_pending())
    except Exception as exc:  # noqa: BLE001 — an empty list is better than a 500 in the panel
        app.logger.error("could not list pending runs: %s", exc, exc_info=True)
        return jsonify([])


@app.route("/finalize", methods=["POST"])
def finalize():
    """
    Apply the ticked suggestions and render.

    An empty `selected` is a real choice — "none of these, render what I
    already approved" — not a missing field, so it is accepted rather than
    rejected as invalid.
    """
    if not _run_lock.acquire(blocking=False):
        return jsonify({"error": "A run is already in progress."}), 409

    started = False
    try:
        payload = request.get_json(force=True) or {}
        pending_id = (payload.get("pending_id") or "").strip()
        selected = payload.get("selected")
        if not pending_id:
            return jsonify({"error": "Which pending run? No id was sent."}), 400
        if selected is None:
            selected = []
        if not isinstance(selected, list):
            return jsonify({"error": "'selected' must be a list of suggestion numbers."}), 400

        # Indices travel as a comma-joined string, matching finalize.py's
        # CLI, so there is one parsing path rather than two that can drift.
        argument = ",".join(str(item) for item in selected) if selected else "none"

        _current["lines"] = []
        _current["status"] = "running"
        process = subprocess.Popen(
            [sys.executable, "-u", "finalize.py", pending_id, argument],
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _current["process"] = process

        def watch():
            try:
                _stream_process(process)
            finally:
                _run_lock.release()

        threading.Thread(target=watch, daemon=True).start()
        started = True
        return jsonify({"started": True})
    except Exception as exc:  # noqa: BLE001
        app.logger.error("failed to finalize: %s", exc, exc_info=True)
        return jsonify({"error": f"Could not finalize: {exc}"}), 500
    finally:
        if not started:
            _run_lock.release()


@app.route("/log")
def log():
    """Poll-based log tail. Simpler than SSE and adequate for one client."""
    since = request.args.get("since", type=int, default=0)
    lines = _current["lines"][since:]
    return jsonify({"lines": lines, "next": since + len(lines), "status": _current["status"]})


@app.route("/stop", methods=["POST"])
def stop():
    process = _current.get("process")
    if process and process.poll() is None:
        process.terminate()
        return jsonify({"stopped": True})
    return jsonify({"stopped": False, "error": "No run in progress."}), 400


if __name__ == "__main__":
    # 127.0.0.1 only. This runs pipeline invocations with your API keys in
    # the environment and has no auth — not safe to expose beyond this
    # machine.
    app.run(host="127.0.0.1", port=5001, debug=False)
