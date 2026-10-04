"""
The second half of a gated run: apply the chosen suggestions, then render.

main.py now stops after the writer/judge loop and saves its state (see
pending.py). This module picks that state back up with the user's
selections, has the writer apply ONLY those edits, and renders the PDF and
review pages exactly as an ungated run would.

Two callers, one code path — finalize_pending() is what webapp/app.py's
/finalize route calls, and what this module's CLI wraps:

    python finalize.py <id> 1,3      apply suggestions 1 and 3
    python finalize.py <id> all      apply all of them
    python finalize.py <id> none     render the draft untouched
    python finalize.py --list        show what is waiting

Selections are 1-based because that is how main.py numbers them in the log
and how the UI lists them; off-by-one here would silently apply the wrong
edit, so parsing is strict and rejects anything out of range rather than
clamping.
"""

import sys

from logger_setup import configure_logging, get_logger, log_cache_summary

configure_logging()
log = get_logger(__name__)

from agents.writer import apply_selected_suggestions
from pending import read_pending, list_pending, clear_pending


def resolve_selection(suggestions: list[str], selected: list) -> list[str]:
    """
    Turn the caller's selection into the actual suggestion strings.

    Accepts 1-based indices (from the CLI or a checkbox list) or the
    suggestion strings themselves (which is what a form POST is likely to
    send back). Mixed input is fine.

    Raises:
        ValueError: on an index outside the list. A selection that doesn't
            line up means the caller is working from a different set of
            suggestions than the one on disk — applying the subset that
            happened to fit would edit the resume in a way nobody asked
            for.
    """
    resolved = []
    for item in selected:
        if isinstance(item, int) or (isinstance(item, str) and item.strip().isdigit()):
            position = int(item)
            if not 1 <= position <= len(suggestions):
                raise ValueError(
                    f"Suggestion {position} does not exist — there are "
                    f"{len(suggestions)}."
                )
            resolved.append(suggestions[position - 1])
        else:
            text = str(item).strip()
            if text and text not in suggestions:
                raise ValueError(
                    "That suggestion isn't in this pending run — it may have "
                    "been finalized already."
                )
            if text:
                resolved.append(text)

    # De-duplicate while keeping the user's order; an index and its text
    # both selected is the same edit twice.
    seen = set()
    return [s for s in resolved if not (s in seen or seen.add(s))]


def finalize_pending(pending_id: str, selected: list) -> str:
    """
    Apply the selected suggestions to a paused run and render its outputs.

    Args:
        pending_id: The id main.py logged for this job (its file suffix).
        selected: 1-based indices and/or suggestion strings. An empty list
            renders the draft unchanged, which is a legitimate choice —
            "none of these" is a decision, not a no-op.

    Returns:
        The path of the rendered PDF.
    """
    # Imported here rather than at module scope: main.py is a script as
    # well as a module, and importing it at load time would make this
    # module's own import order depend on main's. Deferring also keeps
    # --list working if main.py has a problem.
    from main import render_outputs, job_folder_name  # noqa: PLC0415
    from config import DATA_OUTPUT_DIR  # noqa: PLC0415
    import os  # noqa: PLC0415

    state = read_pending(pending_id)
    suggestions = state.get("suggestions") or []
    chosen = resolve_selection(suggestions, selected)

    job = state["job"]
    log.info(
        "Finalizing %s at %s (%d of %d suggestion(s) selected)...",
        job.get("job_title"), job.get("company"), len(chosen), len(suggestions),
    )
    for suggestion in chosen:
        log.info("  applying: %s", suggestion)

    final_draft = apply_selected_suggestions(
        state["candidate_context"],
        job,
        state["draft"],
        chosen,
        style_template=state.get("style_template"),
        job_review=state.get("job_review"),
        # From the saved state, not the environment: this runs as a later,
        # separate process that never saw the original run's settings.
        # "" (not None) when the run had none, so the writer doesn't fall
        # back to whatever WRITER_INSTRUCTIONS happens to be set now.
        instructions=state.get("writer_instructions") or "",
    )

    suffix = state.get("suffix", pending_id)
    render_outputs(
        final_draft,
        job,
        state.get("job_review") or {},
        state.get("resume_review") or {},
        state.get("job_history") or [],
        state.get("resume_history") or [],
        suffix,
        state.get("job_supplied", False),
    )

    # Only now. Clearing before the render would leave a failed render with
    # no draft to retry from, and the writer/judge cycles behind it are the
    # expensive part of the run.
    clear_pending(pending_id)
    log.info("Done.")
    log_cache_summary(log)
    return os.path.join(DATA_OUTPUT_DIR, job_folder_name(job, suffix), "resume.pdf")


def _parse_cli_selection(raw: str, count: int) -> list:
    """Turn the CLI's selection argument into what finalize_pending wants."""
    value = raw.strip().lower()
    if value in ("none", "-", "0"):
        return []
    if value == "all":
        return list(range(1, count + 1))
    return [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in ("--list", "-l"):
        entries = list_pending()
        if not entries:
            print("Nothing waiting.")
            return
        for entry in entries:
            print(
                f"\n{entry['pending_id']}  —  {entry.get('job_title')} at "
                f"{entry.get('company')}  (fitness {entry.get('fitness_score')}/10)"
            )
            for number, suggestion in enumerate(entry.get("suggestions", []), 1):
                print(f"  {number}. {suggestion}")
        print("\nApply with: python finalize.py <id> 1,3   (or 'all' / 'none')")
        return

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python finalize.py --list\n"
            "  python finalize.py <id> 1,3     apply those suggestions\n"
            "  python finalize.py <id> all     apply all of them\n"
            "  python finalize.py <id> none    render the draft as-is"
        )
        sys.exit(1)

    pending_id, raw_selection = sys.argv[1], sys.argv[2]
    try:
        state = read_pending(pending_id)
        selection = _parse_cli_selection(raw_selection, len(state.get("suggestions") or []))
        finalize_pending(pending_id, selection)
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        print("Run 'python finalize.py --list' to see what is waiting.")
        sys.exit(1)
    except ValueError as exc:
        print(f"\n{exc}")
        sys.exit(1)
    except RuntimeError as exc:
        log.error("finalize: %s", exc, exc_info=True)
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
