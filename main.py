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
    2. Search <-> Judge      -> the pool is scraped once, BM25-ranked against the resume,
                                then screened by a model that rates every survivor 1-10 and
                                drops the duds (JobSpy across LinkedIn/Indeed as the primary
                                tool, restricted to postings from the past two weeks, with
                                anything already selected in an earlier run filtered out).
                                Code then walks that ranking: the best remaining posting goes
                                to the judge with its full description attached, and the judge
                                evaluates ONLY that pick (match quality + link
                                trustworthiness). If not approved, the next posting down is
                                tried. Runs for up to MAX_JOB_SEARCH_CYCLES cycles, until the
                                judge approves, or until the pool is exhausted. Nothing else
                                happens until this stage finishes.
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
import re
import json

from logger_setup import configure_logging, get_logger, get_log_file_path, get_run_timestamp, log_cache_summary

# Configure logging before any other project module runs its own
# get_logger(__name__) at import time, so every module's logger is
# attached to fully-configured handlers from the start.
configure_logging()
log = get_logger(__name__)

from agents.context_agent import gather_candidate_context
from agents.search_agent import build_job_pool, next_candidate
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
from html_renderer import (
    render_review_html,
    render_no_job_html,
    render_job_unavailable_html,
    render_jobs_found_html,
)
from pdf_renderer import render_resume_pdf, extract_notes
from job_history import record_selection
from config import (
    DATA_OUTPUT_DIR,
    DATA_WORK_DIR,
    MAX_JOB_SEARCH_CYCLES,
    MAX_RESUME_REVISE_CYCLES,
    SAMPLE_RESUME_PATH,
    JOB_SCRAPE_CEILING,
    MIN_VIABLE_JOB_SCORE,
    RUN_MODE,
    JOB_MATCH_COUNT,
    SUGGESTION_GATE,
)
from pending import write_pending


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

    Not to be confused with data/selected_jobs.json, which DOES
    persist across runs — that one holds only each run's final pick, for
    duplicate filtering (see job_history.py).
    """
    if not pool:
        return
    os.makedirs(DATA_WORK_DIR, exist_ok=True)
    pool_path = os.path.join(DATA_WORK_DIR, f"job_pool_{run_ts}.json")
    try:
        with open(pool_path, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2, default=str)
        log.info("  Job pool saved: %s", pool_path)
    except (OSError, TypeError) as exc:
        log.error("main: failed writing job pool to %s: %s", pool_path, exc, exc_info=True)
        log.info("  Warning: could not save the job pool (%s). Continuing.", exc)


def find_matching_jobs(
    candidate_context: str,
    target_role: str,
    run_ts: str,
    want: int = 1,
) -> tuple[list[dict], list[dict], int]:
    """
    Build the job pool once, then walk it until `want` jobs are worth
    applying to (or the pool/cycle budget runs out).

    Scraping happens exactly once, before the loop. Each cycle takes the
    next posting from the fit-ranked pool rather than re-searching.

    Two acceptance tiers, and the distinction matters. A posting the judge
    APPROVES is taken immediately. A posting that clears
    MIN_VIABLE_JOB_SCORE but not the approval bar is held as a fallback
    and only used to top up at the end — so a run asking for three jobs
    returns the three best available rather than the first three that were
    merely good enough, and a run that finds two approved plus one
    fallback is honest about which is which.

    The cycle budget is MAX_JOB_SEARCH_CYCLES PER job requested. Without
    that scaling, asking for three jobs from a three-cycle budget would
    usually return one.

    Returns:
        (selections, history, pool_size) where selections is a list of
        {"job": dict, "review": dict, "approved": bool} ordered best
        first, and history is every posting put to the judge, for the
        report pages. An empty selections list means nothing cleared the
        viability floor.
    """
    log.info("  Building job pool (scrape ceiling %d, one scraping pass)...", JOB_SCRAPE_CEILING)
    pool = build_job_pool(candidate_context, target_role)
    _save_job_pool(pool, run_ts)

    if not pool:
        # Running cycles against an empty pool costs model calls to
        # produce identical empty results.
        log.info("  The job pool is empty — nothing matched any angle in the window.")
        return [], [], 0

    history: list[dict] = []
    accepted: list[dict] = []
    fallbacks: list[tuple[float, dict, dict]] = []
    tried_urls: set[str] = set()
    budget = MAX_JOB_SEARCH_CYCLES * want

    for cycle in range(1, budget + 1):
        if len(accepted) >= want:
            break

        log.info("  [job cycle %d/%d] Taking the next posting from the pool...", cycle, budget)
        job = next_candidate(pool, tried_urls)
        if job is None:
            log.info("  Pool exhausted after %d cycle(s) — no postings left to try.", cycle - 1)
            break

        log.info("    -> %s at %s", job.get("job_title"), job.get("company"))
        if not job.get("full_requirements"):
            log.info(
                "    Warning: no description text available for this posting. The "
                "judge and writer will be working from almost nothing."
            )
        if job.get("url"):
            tried_urls.add(job["url"])

        review = review_job(job, candidate_context)
        score = review.get("job_match_score")
        approved = bool(review.get("approved", False))
        log.info(
            "    -> job match: %s/10 | approved: %s | %s",
            score, approved, review.get("job_match_summary", ""),
        )
        history.append({"cycle": cycle, "job": job, "review": review})

        score_value = score if isinstance(score, (int, float)) else -1
        if approved:
            accepted.append({"job": job, "review": review, "approved": True})
            log.info("    Accepted (%d of %d requested).", len(accepted), want)
        elif score_value >= MIN_VIABLE_JOB_SCORE:
            fallbacks.append((score_value, job, review))

    # Top up with the best unapproved-but-viable picks, strongest first.
    fallbacks.sort(key=lambda f: f[0], reverse=True)
    for score_value, job, review in fallbacks:
        if len(accepted) >= want:
            break
        accepted.append({"job": job, "review": review, "approved": False})
        log.info(
            "    Filling a slot with an unapproved but viable pick (%s/10): %s at %s.",
            score_value, job.get("job_title"), job.get("company"),
        )

    if not accepted:
        log.info("  Nothing in the pool cleared the %s/10 viability floor.", MIN_VIABLE_JOB_SCORE)
    elif len(accepted) < want:
        log.info(
            "  Found %d job(s) worth applying to, short of the %d requested — "
            "the pool did not hold more that cleared the bar.",
            len(accepted), want,
        )

    return accepted, history, len(pool)


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


def _no_jobs_reason(pool_size: int) -> str:
    """
    Explain an empty-handed search, for the no-job page.

    find_matching_jobs() already applies the viability floor, so by the
    time this is called the only question is WHY nothing cleared it.
    """
    if pool_size == 0:
        return (
            "No job postings matched this candidate at all. Nothing survived "
            "scraping, duplicate filtering against earlier runs, work-eligibility "
            "filtering, and fit screening within the search window."
        )
    return (
        f"Nothing in the {pool_size}-posting pool reached the "
        f"{MIN_VIABLE_JOB_SCORE}/10 viability floor. Writing a resume for a "
        "weaker match would produce a polished application for a role this "
        "candidate is not a real match for, so the run stopped here instead."
    )


def _write_report(path: str, page: str, label: str) -> bool:
    """Write a rendered HTML report, logging rather than raising on failure."""
    try:
        os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
        log.info("  %s: %s", label, path)
        return True
    except OSError as exc:
        log.error("main: could not write %s to %s: %s", label, path, exc, exc_info=True)
        log.info("  Warning: %s could not be written (%s).", label, exc)
        return False


def _slug(text: str, limit: int = 40) -> str:
    """
    Turn a job title or company into something safe for a folder name.

    Kept readable rather than minimal: the folder name is how you find a
    job's resume months later, and `20260920_164005_1` alone tells you
    nothing about which application it was.
    """
    cleaned = re.sub(r"[^\w\s-]", "", str(text or "")).strip()
    cleaned = re.sub(r"[\s_-]+", "-", cleaned)
    return cleaned[:limit].strip("-")


def job_folder_name(job: dict, suffix: str) -> str:
    """
    The per-job folder inside data/output/.

    Timestamp first so the listing sorts chronologically, then company and
    title so it is identifiable at a glance:
        20260920_164005_1_RBC_Senior-Data-Scientist
    """
    parts = [suffix, _slug(job.get("company"), 30), _slug(job.get("job_title"))]
    return "_".join(part for part in parts if part)


def render_outputs(
    final_draft: str,
    job: dict,
    job_review: dict,
    resume_review: dict,
    job_history: list[dict],
    resume_history: list[dict],
    suffix: str,
    job_supplied: bool,
) -> None:
    """
    Write one job's artifacts.

    WHERE THINGS GO. Each job gets its own folder under data/output/,
    containing only the two things worth opening: the resume PDF and the
    review page. The resume Markdown and the review JSON are intermediates
    — one is the safety net behind the PDF, the other is diagnostics — and
    go to data/work/ under the same suffix, so a job's folder is a clean
    pair of files rather than a pile of four formats of the same thing.

    `suffix` distinguishes jobs within a run — plain run timestamp for a
    single job, timestamp_1/_2/_3 when several were requested.

    Every step past the Markdown degrades rather than raises. By this
    point the resume content exists; a WeasyPrint library problem or an
    unserializable value in a review dict should cost you one format, not
    the run — and with several jobs in flight, not the jobs after it
    either.
    """
    job_dir = os.path.join(DATA_OUTPUT_DIR, job_folder_name(job, suffix))
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(DATA_WORK_DIR, exist_ok=True)

    md_path = os.path.join(DATA_WORK_DIR, f"tailored_resume_{suffix}.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_draft)
    except OSError as exc:
        log.error("main: failed writing resume Markdown to %s: %s", md_path, exc, exc_info=True)
        raise RuntimeError(f"Could not write resume Markdown to '{md_path}' — {exc}") from exc

    review_path = os.path.join(DATA_WORK_DIR, f"resume_review_{suffix}.json")
    try:
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job": job,
                    "job_search_stage": {"final_review": job_review, "cycle_history": job_history},
                    "resume_revise_stage": {"final_review": resume_review, "cycle_history": resume_history},
                },
                f, indent=2, default=str,
            )
    except (OSError, TypeError) as exc:
        log.error("main: failed writing review JSON to %s: %s", review_path, exc, exc_info=True)
        log.info("  Warning: could not save the review JSON (%s). Continuing.", exc)

    title = f"{job.get('job_title', 'Tailored Resume')} — {job.get('company', '')}".strip(" —")

    pdf_path = os.path.join(job_dir, "resume.pdf")
    pdf_ok = True
    try:
        render_resume_pdf(final_draft, pdf_path, title=title)
    except RuntimeError as exc:
        log.error("main: PDF rendering failed: %s", exc, exc_info=True)
        log.info("  Warning: PDF rendering failed (%s). Markdown resume was still saved.", exc)
        pdf_ok = False

    review_html_path = os.path.join(job_dir, "review.html")
    html_ok = True
    try:
        review_html = render_review_html(
            job_review, resume_review, job,
            job_cycle_number=len(job_history),
            max_job_cycles=MAX_JOB_SEARCH_CYCLES,
            resume_cycle_number=len(resume_history),
            max_resume_cycles=MAX_RESUME_REVISE_CYCLES,
            job_supplied=job_supplied,
            writer_notes=extract_notes(final_draft),
            title=f"Review — {title}" if title else "Resume & Job Review",
        )
        with open(review_html_path, "w", encoding="utf-8") as f:
            f.write(review_html)
    except (RuntimeError, OSError) as exc:
        log.error("main: review HTML rendering/write failed: %s", exc, exc_info=True)
        log.info("  Warning: review HTML could not be generated (%s).", exc)
        html_ok = False

    log.info("    Folder:      %s", job_dir)
    log.info("    PDF:         %s", "resume.pdf" if pdf_ok else "(failed — see warning above)")
    log.info("    Review:      %s", "review.html" if html_ok else "(failed — see warning above)")
    log.info("    (working copies: %s, %s)", md_path, review_path)


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
            "[job URL or path to job_description.txt]\n"
            "Environment: RUN_MODE=search|write|both, JOB_MATCH_COUNT=1..3"
        )
        sys.exit(1)

    resume_path, target_role = sys.argv[1], sys.argv[2]
    job_source = sys.argv[3] if len(sys.argv) == 4 else None

    log_file_path = get_log_file_path()
    run_ts = get_run_timestamp()
    log.info("Logging this run to: %s\n", log_file_path)

    # Supplying a job source always means "write" — there is nothing to
    # search for. Otherwise RUN_MODE decides, defaulting to both stages.
    mode = "write" if job_source else (RUN_MODE if RUN_MODE in ("search", "both") else "both")
    writing = mode in ("write", "both")
    searching = mode in ("search", "both")
    want = JOB_MATCH_COUNT if searching else 1
    job_supplied = mode == "write"

    total_stages = 1 + (1 if searching else 1) + (2 if writing else 1)
    stage = 0

    def step(message, *args):
        nonlocal stage
        stage += 1
        log.info(f"[{stage}/{total_stages}] " + message, *args)

    log.info(
        "Mode: %s%s",
        {"search": "search only (no resume written)",
         "write": "write only (job supplied)",
         "both": "search and write"}[mode],
        f" — looking for up to {want} job(s)" if searching else "",
    )

    try:
        _validate_inputs(resume_path, job_source)

        step("Gathering candidate context from resume + GitHub...")
        # The job description is deliberately NOT passed here. It would be
        # folded into the candidate summary, which every downstream prompt
        # receives labelled "Candidate background" — a posting's
        # requirements travelling as facts about the candidate. It goes to
        # job["full_requirements"] instead (see agents/jd_agent.py).
        candidate_context = gather_candidate_context(resume_path)
        style_template = load_style_template() if writing else None

        # --- Job stage -------------------------------------------------
        if job_supplied:
            step("Using the supplied job posting (no search)...")
            try:
                job, job_review = run_supplied_job_stage(candidate_context, job_source)
            except JobUnavailableError as exc:
                # Halt and report. There is no fallback candidate here, and
                # the failure message carries the copy-paste workaround the
                # reader needs, so it goes on the page verbatim.
                log.info("\nStopping: %s", exc)
                _write_report(
                    os.path.join(DATA_OUTPUT_DIR, f"job_unavailable_{run_ts}.html"),
                    render_job_unavailable_html(
                        reason=str(exc), job_source=job_source, run_timestamp=run_ts
                    ),
                    "Report",
                )
                log.info("See the full run log for details: %s", log_file_path)
                sys.exit(1)
            selections = [{"job": job, "review": job_review, "approved": None}]
            job_history = [{"cycle": 1, "job": job, "review": job_review}]
            pool_size = 0
        else:
            step(
                "Searching for up to %d matching job(s) (max %d judge cycles each)...",
                want, MAX_JOB_SEARCH_CYCLES,
            )
            selections, job_history, pool_size = find_matching_jobs(
                candidate_context, target_role, run_ts, want=want
            )

            if not selections:
                reason = _no_jobs_reason(pool_size)
                log.info("\nStopping: %s", reason)
                _write_report(
                    os.path.join(DATA_OUTPUT_DIR, f"no_job_found_{run_ts}.html"),
                    render_no_job_html(
                        reason=reason, job_history=job_history, target_role=target_role,
                        pool_size=pool_size, max_job_cycles=MAX_JOB_SEARCH_CYCLES,
                        min_viable_score=MIN_VIABLE_JOB_SCORE, run_timestamp=run_ts,
                    ),
                    "Report",
                )
                log.info(
                    "  Nothing was written to the job history, so these postings "
                    "remain available to a future run."
                )
                log.info("See the full run log for details: %s", log_file_path)
                # Not an error: a fortnight with no good match is a legitimate
                # outcome, so scheduled runs shouldn't alarm on it.
                return

            log.info("  Selected %d job(s):", len(selections))
            for index, selection in enumerate(selections, 1):
                picked = selection["job"]
                log.info(
                    "    %d. %s at %s — %s/10%s",
                    index, picked.get("job_title"), picked.get("company"),
                    selection["review"].get("job_match_score"),
                    " (approved)" if selection["approved"] else " (viable, not approved)",
                )
                log.info("       %s", picked.get("url"))

        # --- Search-only: report and stop ------------------------------
        if not writing:
            step("Writing the job report (resume writing is off)...")
            _write_report(
                os.path.join(DATA_OUTPUT_DIR, f"jobs_found_{run_ts}.html"),
                render_jobs_found_html(
                    selections=selections, target_role=target_role,
                    pool_size=pool_size, run_timestamp=run_ts,
                ),
                "Job report",
            )
            # Deliberately NOT recorded in selected_jobs.json. Nothing was
            # applied for, so these postings must stay available to a later
            # run that does write resumes for them.
            log.info(
                "\nDone. %d job(s) found. Nothing was written to the job history, "
                "so a later run can still pick these up.",
                len(selections),
            )
            return

        # --- Writing stage, once per selected job ----------------------
        step(
            "Writing and reviewing %d resume(s) (max %d cycles each)...",
            len(selections), MAX_RESUME_REVISE_CYCLES,
        )
        results = []
        for index, selection in enumerate(selections, 1):
            job, job_review = selection["job"], selection["review"]
            if len(selections) > 1:
                log.info(
                    "  --- Job %d of %d: %s at %s ---",
                    index, len(selections), job.get("job_title"), job.get("company"),
                )

            # Recorded as writing starts for this job, not at the end: if a
            # later stage crashes, the posting has still been consumed, and
            # a re-run landing on it again would repeat the same failure
            # rather than trying something new.
            record_selection(job, run_timestamp=run_ts, review=job_review)

            final_draft, resume_review, resume_history = run_resume_revise_loop(
                candidate_context, job, style_template=style_template, job_review=job_review
            )
            results.append((job, job_review, resume_review, resume_history, final_draft))

        # --- Suggestion gate, or render --------------------------------
        # With the gate on, the run STOPS here rather than rendering. The
        # judge's remaining suggestions are offered for selection and
        # applied by finalize.py, which then renders. Rendering first and
        # reporting suggestions afterwards produced advice about a PDF that
        # was already written — see config.SUGGESTION_GATE.
        if SUGGESTION_GATE:
            step("Pausing for your decisions on the judge's suggestions...")
            paused = 0
            for index, (job, job_review, resume_review, resume_history, final_draft) in enumerate(results, 1):
                suffix = run_ts if len(results) == 1 else f"{run_ts}_{index}"
                suggestions = resume_review.get("suggestions")
                suggestions = [s for s in suggestions if str(s).strip()] if isinstance(suggestions, list) else []

                if not suggestions:
                    # Nothing to decide, so nothing to wait for. Pausing
                    # here would make the user click through an empty list
                    # to get the PDF the pipeline could already have given
                    # them.
                    log.info(
                        "  %s at %s — the judge left no suggestions; rendering directly.",
                        job.get("job_title"), job.get("company"),
                    )
                    render_outputs(
                        final_draft, job, job_review, resume_review,
                        job_history, resume_history, suffix, job_supplied,
                    )
                    continue

                write_pending(
                    suffix,
                    {
                        "suffix": suffix,
                        "job": job,
                        "job_review": job_review,
                        "resume_review": resume_review,
                        "job_history": job_history,
                        "resume_history": resume_history,
                        "draft": final_draft,
                        "candidate_context": candidate_context,
                        "style_template": style_template,
                        "job_supplied": job_supplied,
                        "suggestions": suggestions,
                    },
                )
                paused += 1
                log.info(
                    "  %s at %s — %d suggestion(s) waiting for your selection (id: %s)",
                    job.get("job_title"), job.get("company"), len(suggestions), suffix,
                )
                for number, suggestion in enumerate(suggestions, 1):
                    log.info("      %d. %s", number, suggestion)

            if paused:
                log.info(
                    "\nPaused. Pick the suggestions you want in the web UI, or run:\n"
                    "  python finalize.py <id> <numbers>   e.g. python finalize.py %s 1,3\n"
                    "  python finalize.py <id> none        to render the draft as-is\n"
                    "The PDF is written once you choose.",
                    run_ts,
                )
            log_cache_summary(log)
            return

        step("Rendering output...")
        for index, (job, job_review, resume_review, resume_history, final_draft) in enumerate(results, 1):
            suffix = run_ts if len(results) == 1 else f"{run_ts}_{index}"
            if len(results) > 1:
                log.info("  Job %d — %s at %s:", index, job.get("job_title"), job.get("company"))
            render_outputs(
                final_draft, job, job_review, resume_review,
                job_history, resume_history, suffix, job_supplied,
            )

        log.info("\nDone. %d resume(s) written.", len(results))
        for job, job_review, resume_review, _, _ in results:
            log.info(
                "  %s at %s — job match %s/10, resume fitness %s/10",
                job.get("job_title"), job.get("company"),
                job_review.get("job_match_score"), resume_review.get("fitness_score"),
            )
        log_cache_summary(log)

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
