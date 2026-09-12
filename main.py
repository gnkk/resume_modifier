"""
Entry point: runs the resume-building pipeline end to end.

TWO PIPELINES share this entry point, chosen by whether a job
description file is supplied on the command line:

  SEARCH MODE (no job description given) — the full four-stage pipeline
  described below. Finds a job, vets it, then writes and reviews a resume
  for it. Stops without writing anything if nothing found clears the
  viability floor.

  SUPPLIED-JOB MODE (third argument given) — the job is already
  decided, so stage 2 collapses: no pool, no scraping of a search
  space, no selection cycles, no viability gate. The third argument is
  either a posting URL or a path to a .txt file. A URL is fetched via
  Firecrawl; LinkedIn, Glassdoor and Facebook links are refused up
  front, because those domains block automated access and the fetch
  would fail regardless — for those, paste the text into a file. The
  posting is parsed into the same job dict shape (agents/jd_agent.py),
  the judge assesses the match for INFORMATION only, and the run
  proceeds straight to writing and reviewing the resume. A weak match is
  reported loudly on the review page but never stops the run — the user
  has already decided to apply.

Everything after stage 2 is identical in both modes.

Pipeline (two independent, SEQUENTIAL judge loops):
    1. Context agent        -> reads the real resume PDF (+ GitHub MCP context, once GITHUB_PAT
                                is set, + an optional job description .txt). The sample/template
                                resume is NOT read here — it goes straight to the writer.
    2. Search <-> Judge      -> search agent proposes a single best-matching job
                                (JobSpy across LinkedIn/Indeed as the primary tool, with
                                Firecrawl as fallback support, restricted to postings from
                                the past two weeks, with anything already selected in an
                                earlier run filtered out); the judge evaluates ONLY that job pick
                                (match quality + link trustworthiness). If not approved,
                                the search agent tries again with the judge's feedback and
                                the rejected pick excluded. Runs for up to
                                MAX_JOB_SEARCH_CYCLES cycles, or until the judge approves
                                early — whichever comes first. Nothing else happens until
                                this stage finishes.
    3. Writer <-> Judge      -> only once a job is locked in from stage 2: the writer
                                drafts an ATS-ready resume for that job, following the
                                sample/template resume's raw text verbatim as its layout
                                spec (read in load_style_template below). The judge
                                evaluates ONLY the resume (fitness score, strengths, gaps,
                                suggestions, template fidelity). If not approved, the writer
                                revises against the judge's feedback. Runs for up to
                                MAX_RESUME_REVISE_CYCLES cycles, or until the judge
                                approves early — whichever comes first.
    4. Render output         -> tailored_resume.md + tailored_resume.pdf (the resume
                                itself, and nothing else — the writer's notes are
                                stripped from the PDF), resume_review.html (both judge
                                verdicts, both scores, the job application link, and
                                the writer's notes) + resume_review.json (full detail)

Logging: console shows the same run narration as always (INFO level).
The same narration, plus exceptions and tool/parse failures with
tracebacks, is written to data/output/logs/run_<timestamp>.log — see
logger_setup.py. The file is a full record of the run, not just its
failures, so two runs can be diffed against each other when tuning the
agents.

Usage:
    python main.py <path_to_resume.pdf> "<target role/company description>" [job URL or .txt path]

Examples:
    # Search mode — the role description steers every search angle.
    python main.py data/input/my_resume.pdf "Data Scientist, remote Canada"

    # Supplied-job mode — no search runs, so the role description is
    # ignored; pass anything short.
    python main.py data/input/my_resume.pdf "-" https://careers.example.com/jobs/1234
    python main.py data/input/my_resume.pdf "-" data/input/job_description.txt

Optional: place a sample/template resume at data/input/sample_resume.pdf
and the writer will follow its layout exactly (structure only, never its
content) — see config.SAMPLE_RESUME_PATH.
"""

import sys
import os
import json

from logger_setup import configure_logging, get_logger, get_log_file_path, get_run_timestamp

# Configure logging before any other project module runs its own
# get_logger(__name__) at import time, so every module's logger is
# attached to fully-configured handlers from the start.
configure_logging()
log = get_logger(__name__)

from agents.context_agent import gather_candidate_context
from agents.search_agent import build_job_pool, select_best_job
from agents.writer import draft_resume, revise_resume
from agents.judge import review_job, review_resume
from agents.jd_agent import (
    job_from_description,
    fetch_job_description,
    looks_like_url,
    JobUnavailableError,
)
from tools.pdf_reader import read_pdf
from tools.text_reader import read_text_file
from html_renderer import render_review_html, render_no_job_html, render_job_unavailable_html
from pdf_renderer import render_resume_pdf, extract_notes
from job_history import record_selection
from config import (
    DATA_OUTPUT_DIR,
    MAX_JOB_SEARCH_CYCLES,
    MAX_RESUME_REVISE_CYCLES,
    SAMPLE_RESUME_PATH,
    JOB_POOL_TARGET_SIZE,
    MIN_VIABLE_JOB_SCORE,
)


def load_style_template(path: str = SAMPLE_RESUME_PATH) -> str | None:
    """
    Read the sample/template resume's text, to be handed to the writer
    verbatim as a layout spec.

    Deliberately read here rather than summarized by the context agent: a
    prose description of a layout ("medium-length bullets, dates like Jan
    2022 - Present") loses the detail that actually makes output match the
    template, and the writer ends up producing a generic resume. Returns
    None when no template exists — an entirely supported case.
    """
    if not os.path.isfile(path):
        log.info(
            "  No template resume at %s — the writer will use standard ATS "
            "section conventions instead (this is optional; nothing is wrong).",
            path,
        )
        return None

    text = read_pdf(path)
    if text.lstrip().startswith("[read_pdf failed"):
        log.error("main: could not read template resume at %s: %s", path, text[:200])
        log.info("  Warning: template resume could not be read (%s). Continuing without it.", path)
        return None

    log.info("  Using template resume: %s (%d chars)", path, len(text))
    return text


def _save_job_pool(pool: list[dict], run_ts: str) -> None:
    """
    Write the scraped pool to disk for this run.

    Written per-run and never reused across runs: JOB_SEARCH_HOURS_OLD
    restricts postings to the past two weeks, so a pool from an earlier run
    has already drifted and would reintroduce exactly the stale listings the
    judge's link check exists to catch. The file is for inspection and
    crash recovery — when a later stage fails, the scrape doesn't have to
    be repeated to see what the agent was choosing between.

    Not to be confused with data/output/selected_jobs.json, which DOES
    persist across runs — that one holds only each run's final pick, for
    duplicate filtering (see job_history.py).
    """
    if not pool:
        return
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    pool_path = os.path.join(DATA_OUTPUT_DIR, f"job_pool_{run_ts}.json")
    try:
        with open(pool_path, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2, default=str)
        log.info("  Job pool saved: %s", pool_path)
    except (OSError, TypeError) as exc:
        log.error("main: failed writing job pool to %s: %s", pool_path, exc, exc_info=True)
        log.info("  Warning: could not save the job pool (%s). Continuing.", exc)


def run_job_search_loop(
    candidate_context: str,
    target_role: str,
    run_ts: str,
) -> tuple[dict, dict, list[dict]]:
    """
    Build the job pool once, then run the selection <-> judge cycle up to
    MAX_JOB_SEARCH_CYCLES times, or until the judge approves early.

    Scraping happens exactly once, before the loop. Each cycle picks a
    different posting from that pool rather than re-searching — repeated
    near-identical queries were returning near-identical postings, so
    re-scraping per cycle bought nothing. No resume is written during
    this stage; the judge evaluates ONLY the job pick.

    Returns:
        (final_job, final_job_review, history, pool_size) where history is
        a list of {"cycle": int, "job": dict, "review": dict} entries and
        pool_size is the post-filtering pool the cycles drew from — the
        caller needs it to explain an empty-handed run.

        On early approval, that approved pick is returned. If no cycle is
        approved, the BEST-SCORING pick is returned rather than the last
        one — later cycles chase specific judge concerns and can trade a
        strong overall match for one that merely answers the last
        complaint, so the final cycle is not reliably the strongest. Every
        pick is scored by the same judge against the same candidate, so
        the scores are comparable. (This mirrors run_resume_revise_loop,
        and matters more now that the judge holds a firm approval bar:
        exhausting all three cycles is a normal outcome, not a rare one.)
    """
    log.info("  Building job pool (target %d postings, one scraping pass)...", JOB_POOL_TARGET_SIZE)
    pool = build_job_pool(candidate_context, target_role)

    _save_job_pool(pool, run_ts)

    if not pool:
        # Running three selection cycles against an empty pool costs three
        # model calls to produce three identical empty results.
        log.info("  The job pool is empty — nothing matched any angle in the window.")
        return (
            {
                "job_title": None, "company": None, "location": None, "url": None,
                "posted_date": None, "full_requirements": None, "match_rationale": None,
                "search_notes": "No postings survived scraping, duplicate filtering, and fit screening.",
            },
            {
                "job_match_score": None,
                "job_match_summary": "No posting was available to review.",
                "job_link_trustworthy": None,
                "job_concerns": [],
                "approved": False,
            },
            [],
            0,
        )

    history = []
    rejected = []
    best = None  # (score, cycle, job, review)
    extra_searches_used = 0

    for cycle in range(1, MAX_JOB_SEARCH_CYCLES + 1):
        if cycle == 1:
            log.info("  [job cycle 1/%d] Selecting best job from pool...", MAX_JOB_SEARCH_CYCLES)
        else:
            log.info(
                "  [job cycle %d/%d] Selecting a different job from pool against judge feedback...",
                cycle, MAX_JOB_SEARCH_CYCLES,
            )

        job, extra_searches_used = select_best_job(
            pool,
            candidate_context,
            target_role,
            rejected_jobs=rejected or None,
            extra_searches_used=extra_searches_used,
        )

        if not job.get("url"):
            log.info("    Warning: the selection agent could not identify a job posting.")
            log.info("    Notes: %s", job.get("search_notes"))
            if job.get("raw_response"):
                log.info("    --- raw model response (for debugging) ---")
                log.info("    %s", job["raw_response"].replace("\n", "\n    ")[:2000])
                log.info("    --- end raw response ---")

        log.info("  [job cycle %d/%d] Judge reviewing job pick...", cycle, MAX_JOB_SEARCH_CYCLES)
        review = review_job(job, candidate_context)
        history.append({"cycle": cycle, "job": job, "review": review})

        job_score = review.get("job_match_score")
        approved = review.get("approved", False)
        log.info(
            "    -> %s at %s | job match: %s/10, approved: %s%s",
            job.get("job_title"), job.get("company"), job_score, approved,
            f" | {review.get('job_match_summary')}" if review.get("job_match_summary") else "",
        )

        # An unparseable review has no score; rank it below every real one
        # rather than letting it win by default.
        score_value = job_score if isinstance(job_score, (int, float)) else -1
        if best is None or score_value > best[0]:
            best = (score_value, cycle, job, review)

        if approved:
            return job, review, history, len(pool)

        if cycle == MAX_JOB_SEARCH_CYCLES:
            if best[1] != cycle:
                log.info(
                    "  No job pick was approved — keeping the best-scoring pick "
                    "(cycle %d, %s/10) over the final one (cycle %d, %s/10).",
                    best[1], best[0] if best[0] >= 0 else "unscored",
                    cycle, job_score if job_score is not None else "unscored",
                )
            else:
                log.info(
                    "  No job pick was approved — the final pick (cycle %d) was also "
                    "the best-scoring, keeping it.",
                    cycle,
                )
            return best[2], best[3], history, len(pool)

        rejected.append({"job": job, "review": review})

    # Unreachable given the loop above, but keeps type-checkers happy.
    return best[2], best[3], history, len(pool)


def run_resume_revise_loop(
    candidate_context: str,
    job: dict,
    style_template: str | None = None,
    job_review: dict | None = None,
) -> tuple[str, dict, list[dict]]:
    """
    Run the writer <-> judge cycle up to MAX_RESUME_REVISE_CYCLES
    times, or until the judge approves the resume early — whichever
    comes first. Only called once the job from run_job_search_loop is
    locked in; the judge is evaluating ONLY the resume here.

    Returns:
        (final_draft, final_review, history) where history is a list of
        {"cycle": int, "review": dict} entries for every cycle run.

        On early approval, that approved draft is returned. If no cycle is
        approved, the BEST-SCORING draft is returned rather than the last
        one — revisions chase specific feedback and can overcorrect, so the
        final cycle is not reliably the strongest. Every draft is scored by
        the same judge against the same job, so the scores are comparable.

        The loop also stops early when a revision fails to improve on the
        cycle before it. A score that doesn't move means the judge's
        remaining objections are ones the writer cannot act on — almost
        always limitations of the candidate's background rather than of the
        draft — and another cycle will produce the same verdict at full
        cost. job_review is passed down for the same reason: both the
        writer and the judge are told which gaps stage 1 already accepted,
        so neither tries to close them.
    """
    history = []
    best = None  # (score, cycle, draft, review)
    previous_score = None

    log.info("  [resume cycle 1/%d] Writer drafting initial resume...", MAX_RESUME_REVISE_CYCLES)
    draft = draft_resume(
        candidate_context, job, style_template=style_template, job_review=job_review
    )

    for cycle in range(1, MAX_RESUME_REVISE_CYCLES + 1):
        log.info("  [resume cycle %d/%d] Judge reviewing resume draft...", cycle, MAX_RESUME_REVISE_CYCLES)
        review = review_resume(
            draft, candidate_context, job, style_template=style_template, job_review=job_review
        )
        history.append({"cycle": cycle, "review": review})

        score = review.get("fitness_score")
        approved = review.get("approved", False)
        log.info(
            "    -> resume fitness: %s/10, approved: %s%s",
            score, approved,
            f" | {review.get('fitness_summary')}" if review.get("fitness_summary") else "",
        )

        # An unparseable review has no score; rank it below every real one
        # rather than letting it win by default.
        score_value = score if isinstance(score, (int, float)) else -1
        if best is None or score_value > best[0]:
            best = (score_value, cycle, draft, review)

        if approved:
            return draft, review, history

        if cycle == MAX_RESUME_REVISE_CYCLES:
            if best[1] != cycle:
                log.info(
                    "  No cycle was approved — keeping the best-scoring draft "
                    "(cycle %d, %s/10) over the final one (cycle %d, %s/10).",
                    best[1], best[0] if best[0] >= 0 else "unscored",
                    cycle, score if score is not None else "unscored",
                )
            else:
                log.info(
                    "  No cycle was approved — the final draft (cycle %d) was also "
                    "the best-scoring, keeping it.",
                    cycle,
                )
            return best[2], best[3], history

        # A revision that didn't move the score is the signature of a judge
        # objecting to something the writer cannot change. Another cycle
        # costs two model calls to reproduce the same verdict.
        if previous_score is not None and score_value <= previous_score:
            log.info(
                "  Stopping early: cycle %d scored %s against cycle %d's %s — the "
                "revision didn't improve on it, so the remaining objections are "
                "likely not fixable by rewriting. Keeping the best draft (cycle %d).",
                cycle, score if score is not None else "unscored",
                cycle - 1, previous_score if previous_score >= 0 else "unscored",
                best[1],
            )
            return best[2], best[3], history

        previous_score = score_value

        log.info(
            "  [resume cycle %d/%d] Writer revising against judge feedback...",
            cycle + 1, MAX_RESUME_REVISE_CYCLES,
        )
        draft = revise_resume(
            candidate_context, job, draft, review,
            style_template=style_template, job_review=job_review,
        )

    # Unreachable given the loop above, but keeps type-checkers happy.
    return best[2], best[3], history


def run_supplied_job_stage(candidate_context: str, job_source: str) -> tuple[dict, dict]:
    """
    The job stage for the SUPPLIED-JOB pipeline: no search, no selection.

    job_source is either a URL or a path to a text file. A URL is fetched
    through Firecrawl; LinkedIn, Glassdoor and Facebook links are refused
    up front, since those domains block automated access and the fetch
    would fail anyway.

    The user has already decided to apply, so nothing here is allowed to
    stop the run on match quality. review_job() still runs, but purely to
    produce the match assessment and the concerns that feed the writer's
    and the resume judge's settled-gaps blocks — its score gates nothing.
    A low score is surfaced loudly and reported on the review page,
    because it is useful interview preparation, but overriding a decision
    the user has already made is not this tool's job.

    Returns:
        (job, job_review) in the same shapes run_job_search_loop produces.

    Raises:
        JobUnavailableError: if the posting can't be read or fetched.
            Unlike a thin job pool, this is unrecoverable — it is the one
            input the entire run is built around, and there is no second
            candidate to fall back to.
    """
    source_url = None
    if looks_like_url(job_source):
        source_url = job_source.strip()
        jd_text = fetch_job_description(source_url)
    else:
        jd_text = read_text_file(job_source)
        if jd_text.lstrip().startswith("[read_text_file failed"):
            log.error("main: could not read the job description at %s: %s", job_source, jd_text[:200])
            raise JobUnavailableError(
                f"Could not read the job description at '{job_source}'.\n\n"
                f"{jd_text.strip()[:300]}"
            )
        log.info("  Read job description: %s (%d chars)", job_source, len(jd_text))

    job = job_from_description(jd_text, source_path=job_source, source_url=source_url)
    log.info(
        "  -> Target: %s at %s (%s)",
        job.get("job_title"), job.get("company") or "company not stated",
        job.get("location") or "location not stated",
    )
    if not job.get("url"):
        log.info("  Note: no application URL was found — the review page will have no link.")

    log.info("  Judge assessing the match (for information — this does not gate the run)...")
    review = review_job(job, candidate_context)
    score = review.get("job_match_score")
    log.info("    -> job match: %s/10 | %s", score, review.get("job_match_summary", ""))
    if isinstance(score, (int, float)) and score < MIN_VIABLE_JOB_SCORE:
        log.info(
            "    Heads up: %s/10 is below the %s/10 bar a searched job would have "
            "needed. Proceeding anyway because you chose this posting — read the "
            "concerns on the review page before applying.",
            score, MIN_VIABLE_JOB_SCORE,
        )
    return job, review


def _job_viability(job: dict, job_review: dict, pool_size: int) -> tuple[bool, str]:
    """
    Decide whether this run's best job pick is worth writing a resume for.

    Three outcomes are possible at this point, and only the third should
    stop the run: an approved pick, an unapproved-but-real match worth
    applying to, and nothing worth applying to at all. The last case used
    to flow straight into the writer, which produced a polished, plausible
    resume for a job the judge had just rated as a poor match — an output
    that looks like a successful run and shouldn't be sent anywhere. That
    is worse than no output.

    Returns (is_viable, reason). The reason is written for a human reading
    the no-job page, not for a log line.
    """
    if not job or not job.get("url"):
        if pool_size == 0:
            return False, (
                "No job postings matched this candidate at all. Nothing survived "
                "scraping, duplicate filtering against earlier runs, and fit "
                "screening within the search window."
            )
        return False, (
            f"The selection agent could not identify a usable posting from the "
            f"{pool_size}-posting pool, so nothing reached the judge."
        )

    score = job_review.get("job_match_score")
    if not isinstance(score, (int, float)):
        return False, (
            "The judge could not produce a usable score for the selected posting, "
            "so there is no basis for writing a resume against it. This usually "
            "means the review response failed to parse — see the run log."
        )

    if score < MIN_VIABLE_JOB_SCORE:
        return False, (
            f"The best posting found scored {score}/10, below the "
            f"{MIN_VIABLE_JOB_SCORE}/10 viability floor. Writing a resume for it "
            "would produce a polished application for a role this candidate is "
            "not a real match for, so the run stopped here instead."
        )

    return True, ""


def _validate_inputs(resume_path: str, job_description_path: str | None) -> None:
    """Fail fast with a clear message if required input files are missing."""
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(
            f"Resume file not found: '{resume_path}'. Check the path and try again."
        )
    # A URL is validated by fetching it, not by the filesystem.
    if (
        job_description_path
        and not looks_like_url(job_description_path)
        and not os.path.isfile(job_description_path)
    ):
        raise FileNotFoundError(
            f"Job description not found: '{job_description_path}'. Pass either a "
            "path to a text file or a posting URL starting with http:// or https://."
        )


def main():
    if len(sys.argv) not in (3, 4):
        print(
            'Usage: python main.py <path_to_resume.pdf> "<target role description>" '
            "[job URL or path to job_description.txt]"
        )
        sys.exit(1)

    resume_path, target_role = sys.argv[1], sys.argv[2]
    job_description_path = sys.argv[3] if len(sys.argv) == 4 else None

    log_file_path = get_log_file_path()
    run_ts = get_run_timestamp()
    log.info("Logging this run to: %s\n", log_file_path)

    # Two pipelines share this entry point. Supplying a job description
    # selects the second one: no pool, no search, no selection cycles, and
    # no viability gate — the job is already decided. Both still run four
    # stages; only stage 2 differs.
    job_supplied = bool(job_description_path)

    try:
        _validate_inputs(resume_path, job_description_path)

        log.info("[1/4] Gathering candidate context from resume + GitHub...")
        # The job description is deliberately NOT passed here. It would be
        # folded into the candidate summary, which every downstream prompt
        # receives labelled "Candidate background" — a posting's
        # requirements travelling as facts about the candidate. It goes to
        # job["full_requirements"] instead (see agents/jd_agent.py).
        candidate_context = gather_candidate_context(resume_path)
        style_template = load_style_template()

        if job_supplied:
            log.info("[2/4] Using the supplied job posting (no search)...")
            try:
                job, job_review = run_supplied_job_stage(candidate_context, job_description_path)
            except JobUnavailableError as exc:
                # Halt and report. There is no fallback candidate here, and
                # the failure message carries the copy-paste workaround the
                # reader needs, so it goes on the page verbatim.
                log.info("\nStopping before the writing stage: %s", exc)
                os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
                fail_path = os.path.join(DATA_OUTPUT_DIR, f"job_unavailable_{run_ts}.html")
                try:
                    page = render_job_unavailable_html(
                        reason=str(exc),
                        job_source=job_description_path,
                        run_timestamp=run_ts,
                    )
                    with open(fail_path, "w", encoding="utf-8") as f:
                        f.write(page)
                    log.info("  Report: %s", fail_path)
                except (RuntimeError, OSError) as render_exc:
                    log.error("main: could not write the job-unavailable report: %s", render_exc, exc_info=True)
                    log.info("  Warning: the report could not be written (%s).", render_exc)
                log.info("See the full run log for details: %s", log_file_path)
                # Unlike an empty search, this is a failure the user must
                # act on, so it exits non-zero.
                sys.exit(1)
            job_history = [{"cycle": 1, "job": job, "review": job_review}]
        else:
            log.info(
                "[2/4] Running search <-> judge loop (max %d cycles) to lock in a job...",
                MAX_JOB_SEARCH_CYCLES,
            )
            job, job_review, job_history, pool_size = run_job_search_loop(
                candidate_context, target_role, run_ts
            )
            log.info(
                "  -> Final pick: %s at %s (%s)",
                job.get("job_title"), job.get("company"), job.get("location"),
            )
            log.info("     %s", job.get("url"))

            viable, reason = _job_viability(job, job_review, pool_size)
            if not viable:
                log.info("\nStopping before the writing stage: %s", reason)
                os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
                no_job_path = os.path.join(DATA_OUTPUT_DIR, f"no_job_found_{run_ts}.html")
                try:
                    page = render_no_job_html(
                        reason=reason,
                        job_history=job_history,
                        target_role=target_role,
                        pool_size=pool_size,
                        max_job_cycles=MAX_JOB_SEARCH_CYCLES,
                        min_viable_score=MIN_VIABLE_JOB_SCORE,
                        run_timestamp=run_ts,
                    )
                    with open(no_job_path, "w", encoding="utf-8") as f:
                        f.write(page)
                    log.info("  Report: %s", no_job_path)
                except (RuntimeError, OSError) as exc:
                    log.error("main: could not write the no-job report: %s", exc, exc_info=True)
                    log.info("  Warning: the no-job report could not be written (%s).", exc)
                log.info("  Nothing was written to the resume history, so these postings "
                         "remain available to a future run.")
                log.info("See the full run log for details: %s", log_file_path)
                # Not an error: a week with no good match is a legitimate outcome,
                # so scheduled runs shouldn't alarm on it.
                return

        # Recorded as soon as the job is locked in, not at the end of the
        # run: if the writer or judge stage crashes, this posting has still
        # been consumed, and a re-run that landed on it again would repeat
        # the same failure rather than trying something new.
        record_selection(job, run_timestamp=run_ts, review=job_review)

        log.info(
            "[3/4] Running writer <-> judge loop (max %d cycles) to tailor the resume...",
            MAX_RESUME_REVISE_CYCLES,
        )
        final_draft, resume_review, resume_history = run_resume_revise_loop(
            candidate_context, job, style_template=style_template, job_review=job_review
        )

        log.info("[4/4] Rendering output...")
        os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

        md_path = os.path.join(DATA_OUTPUT_DIR, f"tailored_resume_{run_ts}.md")
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(final_draft)
        except OSError as exc:
            log.error("main: failed writing resume Markdown to %s: %s", md_path, exc, exc_info=True)
            raise RuntimeError(f"Could not write resume Markdown to '{md_path}' — {exc}") from exc

        review_path = os.path.join(DATA_OUTPUT_DIR, f"resume_review_{run_ts}.json")
        try:
            with open(review_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "job": job,
                        "job_search_stage": {
                            "final_review": job_review,
                            "cycle_history": job_history,
                        },
                        "resume_revise_stage": {
                            "final_review": resume_review,
                            "cycle_history": resume_history,
                        },
                    },
                    f,
                    indent=2,
                )
        except (OSError, TypeError) as exc:
            # TypeError: something non-JSON-serializable ended up in one of
            # the dicts (shouldn't happen, but don't let it take down a run
            # that otherwise succeeded).
            log.error("main: failed writing review JSON to %s: %s", review_path, exc, exc_info=True)
            log.info(
                "  Warning: could not save resume_review.json (%s). "
                "Continuing to render the resume and review HTML.",
                exc,
            )

        title = f"{job.get('job_title', 'Tailored Resume')} — {job.get('company', '')}".strip(" —")

        pdf_path = os.path.join(DATA_OUTPUT_DIR, f"tailored_resume_{run_ts}.pdf")
        pdf_ok = True
        try:
            render_resume_pdf(final_draft, pdf_path, title=title)
        except RuntimeError as exc:
            # PDF rendering is the one step most likely to fail purely on
            # environment grounds (missing system libraries) rather than a
            # real bug — don't let it take down a run that already produced
            # a valid Markdown resume and both judge verdicts.
            log.error("main: PDF rendering failed: %s", exc, exc_info=True)
            log.info("  Warning: PDF rendering failed (%s). Markdown resume was still saved.", exc)
            pdf_ok = False

        review_html_path = os.path.join(DATA_OUTPUT_DIR, f"resume_review_{run_ts}.html")
        html_ok = True
        try:
            review_html = render_review_html(
                job_review,
                resume_review,
                job,
                job_cycle_number=len(job_history),
                max_job_cycles=MAX_JOB_SEARCH_CYCLES,
                resume_cycle_number=len(resume_history),
                max_resume_cycles=MAX_RESUME_REVISE_CYCLES,
                job_supplied=job_supplied,
                # The writer's gap flags. They belong here, not in the resume
                # PDF an employer receives — see pdf_renderer.py.
                writer_notes=extract_notes(final_draft),
                title=f"Review — {title}" if title else "Resume & Job Review",
            )
            with open(review_html_path, "w", encoding="utf-8") as f:
                f.write(review_html)
        except (RuntimeError, OSError) as exc:
            log.error("main: review HTML rendering/write failed: %s", exc, exc_info=True)
            log.info("  Warning: review HTML could not be generated (%s).", exc)
            html_ok = False

        log.info(
            "\nDone. Job search: %d cycle(s). Resume revise: %d cycle(s).",
            len(job_history), len(resume_history),
        )
        log.info("  Markdown:    %s", md_path)
        log.info("  PDF:         %s", pdf_path if pdf_ok else "(failed — see warning above)")
        log.info("  Review HTML: %s", review_html_path if html_ok else "(failed — see warning above)")
        log.info("  Review JSON: %s", review_path)
        log.info(
            "\nJob match score:    %s/10 — %s",
            job_review.get("job_match_score"), job_review.get("job_match_summary", ""),
        )
        log.info(
            "Resume fitness score: %s/10 — %s",
            resume_review.get("fitness_score"), resume_review.get("fitness_summary", ""),
        )

    except FileNotFoundError as exc:
        log.error("main: input file missing: %s", exc, exc_info=True)
        log.info("\nError: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        # Raised deliberately by our own agents/renderers for genuinely
        # fatal conditions (API unreachable, etc.) — already logged with
        # full detail at the raise site; show a clean message here.
        log.error("main: pipeline aborted: %s", exc, exc_info=True)
        log.info("\nError: %s", exc)
        log.info("See the full run log for details: %s", log_file_path)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 — top-level safety net so the run always leaves a log
        log.error("main: unexpected error, pipeline aborted: %s", exc, exc_info=True)
        log.info("\nUnexpected error: %s", exc)
        log.info("See the full run log for details: %s", log_file_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
