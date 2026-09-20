"""
Cross-run record of jobs already selected.

Everything else this pipeline writes to data/output/ is per-run and
timestamped. This file is the exception: one JSON ledger that accumulates
across runs, holding the job each run finally settled on.

Why it exists: the search pool is rebuilt from scratch every run against
the same boards, the same candidate, and an overlapping recency window,
so consecutive runs land on the same posting and produce the same resume
for it. The ledger lets build_job_pool() drop postings already targeted
before the screening pass ever looks at them.

Only the FINAL selected job of each run is recorded. Picks the judge
rejected mid-run are deliberately NOT recorded: those were often rejected
only because something better sat next to them in that run's pool, and a
posting shouldn't be barred from future runs on that basis. The
within-run loop already excludes them for the run that's in progress.

Never load-bearing. A missing, unreadable, or corrupt ledger degrades to
"no history" and the run proceeds normally — losing deduplication is a
far smaller failure than refusing to search.

DELIBERATELY NOT TIED TO data/output/. Deleting a job's output folder
does not make that posting available again, and should not: the folder
holds a rendered PDF, while this ledger records that the posting was
targeted. Wanting to delete the one is usually about tidying disk; wanting
to undo the other is a decision, and it gets its own explicit command
rather than being a side effect of a file deletion.

Editing it: this module is runnable.
    python job_history.py --list
    python job_history.py --forget 2,5      by number, as --list shows them
    python job_history.py --forget acme     by company, title or URL text
    python job_history.py --clear
The web UI exposes the same thing as checkboxes.
"""

import json
import os
import re
import sys
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from config import SELECTED_JOBS_HISTORY_PATH
from logger_setup import get_logger

log = get_logger(__name__)

# Tracking/pagination junk that varies per scrape for the SAME posting.
# Anything not listed here is preserved, because on some boards the query
# string carries the job identity itself — an Indeed URL is
# .../viewjob?jk=<id>, so a blanket "strip the query" would collapse every
# Indeed posting in the pool into one key and silently drop the lot.
_TRACKING_PARAMS = {
    "refid", "reffid", "trackingid", "trk", "trkinfo", "originalsubdomain",
    "position", "pagenum", "ebp", "lipi", "licu", "src", "source", "from",
    "tk", "xkcb", "xpse", "vjs", "advn", "adid", "sjdu", "acatk", "spa",
}

_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_url(url: str | None) -> str:
    """
    Reduce a posting URL to a stable identity key.

    Lowercases scheme/host, drops the fragment and known tracking
    parameters, and sorts what's left so parameter order can't produce two
    keys for one posting. Returns "" for anything unusable.
    """
    if not url or not isinstance(url, str):
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""

    kept = sorted(
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=False)
        if key.lower() not in _TRACKING_PARAMS
    )
    path = parts.path.rstrip("/")
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, urlencode(kept), "")
    )


def _fingerprint(company: str | None, title: str | None) -> str:
    """
    Secondary identity key: normalized company + title.

    The same opening is routinely posted to several boards under different
    URLs, so URL matching alone lets a duplicate straight back through.
    This catches those. It is deliberately the weaker of the two keys —
    a company genuinely running two openings with one title will be
    collapsed into one here, which is the accepted cost.
    """
    company_key = _ALNUM_RE.sub("", (company or "").lower())
    title_key = _ALNUM_RE.sub("", (title or "").lower())
    if not company_key or not title_key:
        return ""
    return f"{company_key}|{title_key}"


def load_history(path: str = SELECTED_JOBS_HISTORY_PATH) -> list[dict]:
    """
    Read the ledger. Returns [] when it doesn't exist yet (the normal
    first-run case) or can't be read.
    """
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("job_history: could not read %s: %s", path, exc, exc_info=True)
        log.info(
            "  Warning: the selected-jobs history at %s could not be read (%s). "
            "Continuing without duplicate filtering for this run.",
            path, exc,
        )
        return []

    if not isinstance(data, list):
        log.error("job_history: %s does not contain a JSON list; ignoring it.", path)
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def seen_keys(history: list[dict]) -> tuple[set[str], set[str]]:
    """Return (normalized urls, company|title fingerprints) from the ledger."""
    urls = {_normalize_url(entry.get("url")) for entry in history}
    prints = {
        _fingerprint(entry.get("company"), entry.get("job_title"))
        for entry in history
    }
    return urls - {""}, prints - {""}


def filter_seen(pool: list[dict], history: list[dict] | None = None) -> list[dict]:
    """
    Drop postings already selected in an earlier run.

    Matches on normalized URL first, then on company+title, so the same
    opening scraped from a different board is still caught. Logs what was
    dropped rather than doing it silently — a pool that quietly shrinks is
    hard to tell apart from a thin market.
    """
    history = load_history() if history is None else history
    if not history or not pool:
        return pool

    urls, prints = seen_keys(history)
    if not urls and not prints:
        return pool

    kept, dropped = [], []
    for record in pool:
        url_key = _normalize_url(record.get("job_url"))
        print_key = _fingerprint(record.get("company"), record.get("title"))
        if (url_key and url_key in urls) or (print_key and print_key in prints):
            dropped.append(record)
            continue
        kept.append(record)

    if dropped:
        log.info(
            "  Skipped %d posting(s) already selected in earlier runs (e.g. %s).",
            len(dropped),
            ", ".join(
                f"{r.get('title')} at {r.get('company')}" for r in dropped[:3]
            ),
        )
    return kept


def record_selection(
    job: dict,
    run_timestamp: str | None = None,
    review: dict | None = None,
    path: str = SELECTED_JOBS_HISTORY_PATH,
) -> bool:
    """
    Append this run's final job pick to the ledger.

    Skips silently when the job has no URL (a failed selection cycle —
    recording it would bar nothing and clutter the ledger) and when the
    same posting is already recorded, so a deliberate re-run against an
    existing target doesn't add a second row.

    Returns True when a row was written. Never raises: a ledger write
    failing must not take down a run whose real outputs are already on
    disk.
    """
    url = job.get("url")
    if not url:
        return False

    history = load_history(path)
    urls, _ = seen_keys(history)
    if _normalize_url(url) in urls:
        return False

    history.append(
        {
            "run": run_timestamp,
            "url": url,
            "job_title": job.get("job_title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "posted_date": job.get("posted_date"),
            "job_match_score": (review or {}).get("job_match_score"),
            "judge_approved": (review or {}).get("approved"),
        }
    )

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, default=str)
    except (OSError, TypeError) as exc:
        log.error("job_history: could not write %s: %s", path, exc, exc_info=True)
        log.info(
            "  Warning: this run's job pick could not be added to %s (%s). "
            "A future run may select it again.",
            path, exc,
        )
        return False

    log.info("  Recorded this run's pick in %s (%d job(s) on file).", path, len(history))
    return True


# ---------------------------------------------------------------------
# Editing the ledger
# ---------------------------------------------------------------------

def _write_history(history: list[dict], path: str) -> bool:
    """Persist the ledger. Returns False on failure rather than raising."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, default=str)
        return True
    except (OSError, TypeError) as exc:
        log.error("job_history: could not write %s: %s", path, exc, exc_info=True)
        return False


def list_selections(path: str = SELECTED_JOBS_HISTORY_PATH) -> list[dict]:
    """
    The ledger with a 1-based `n` on each row, newest last.

    The number is what every editing command takes, so what you read and
    what you type are the same thing.
    """
    return [{"n": i, **entry} for i, entry in enumerate(load_history(path), 1)]


def forget_selections(
    keys: list,
    path: str = SELECTED_JOBS_HISTORY_PATH,
) -> list[dict]:
    """
    Remove entries so those postings can be selected again.

    `keys` accepts row numbers as shown by list_selections(), or free text
    matched case-insensitively against a row's company, title or URL. Text
    matching is there because the reason you want a job back is usually
    "redo the Acme one", not "redo row 7".

    Returns the removed entries. Removing nothing is not an error — it
    means nothing matched, which the caller reports.

    Note what this does NOT do: it does not delete the resume that was
    produced for that posting. The output folder is yours to keep or bin
    independently; this only controls whether the job can be picked again.
    """
    history = load_history(path)
    if not history or not keys:
        return []

    doomed: set[int] = set()
    for key in keys:
        text = str(key).strip()
        if not text:
            continue
        if text.isdigit():
            index = int(text) - 1
            if 0 <= index < len(history):
                doomed.add(index)
            continue
        needle = text.lower()
        for index, entry in enumerate(history):
            haystack = " ".join(
                str(entry.get(field) or "")
                for field in ("company", "job_title", "url")
            ).lower()
            if needle in haystack:
                doomed.add(index)

    if not doomed:
        return []

    removed = [history[i] for i in sorted(doomed)]
    remaining = [entry for i, entry in enumerate(history) if i not in doomed]

    if not _write_history(remaining, path):
        log.info("  Warning: the history could not be saved, so nothing was removed.")
        return []

    for entry in removed:
        log.info(
            "  Removed from history: %s at %s — it can be selected again.",
            entry.get("job_title"), entry.get("company"),
        )
    return removed


def clear_history(path: str = SELECTED_JOBS_HISTORY_PATH) -> int:
    """Empty the ledger. Returns how many entries were dropped."""
    count = len(load_history(path))
    if count and _write_history([], path):
        log.info("  Cleared %d entry(ies) from the job history.", count)
        return count
    return 0


def _print_history(path: str) -> None:
    entries = list_selections(path)
    if not entries:
        print("No jobs on file — nothing is being filtered out.")
        return
    print(f"{len(entries)} job(s) on file (these are skipped in future runs):\n")
    for entry in entries:
        print(
            f"  {entry['n']:>3}. {entry.get('job_title')} at {entry.get('company')}"
            f"  [{entry.get('run') or 'no run id'}]"
        )
        print(f"       {entry.get('url')}")
    print("\nRedo one: python job_history.py --forget <number(s) or text>")


def main() -> None:
    path = SELECTED_JOBS_HISTORY_PATH
    args = sys.argv[1:]

    if not args or args[0] in ("--list", "-l"):
        _print_history(path)
        return

    if args[0] == "--clear":
        count = clear_history(path)
        print(f"Cleared {count} entry(ies)." if count else "Nothing to clear.")
        return

    if args[0] == "--forget" and len(args) > 1:
        keys = [k for k in ",".join(args[1:]).split(",") if k.strip()]
        removed = forget_selections(keys, path)
        if not removed:
            print("Nothing matched — run --list to see what's on file.")
            return
        print(f"Removed {len(removed)} entry(ies); those jobs can be selected again:")
        for entry in removed:
            print(f"  - {entry.get('job_title')} at {entry.get('company')}")
        return

    print(
        "Usage:\n"
        "  python job_history.py --list\n"
        "  python job_history.py --forget 2,5\n"
        "  python job_history.py --forget acme\n"
        "  python job_history.py --clear"
    )


if __name__ == "__main__":
    main()
