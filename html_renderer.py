"""
HTML renderer for the judge's review.

Renders the judge's final verdicts from BOTH pipeline stages — the
job-match review (stage 1: search agent <-> judge) and the resume
review (stage 2: writer agent <-> judge) — plus the job application
link, as a single standalone HTML page. This is the ONLY HTML output
this project produces; the resume itself is output as Markdown + PDF
(see pdf_renderer.py), not HTML.

Uses simple string templating for the review fields (already plain
strings/lists), plus a Markdown pass for the writer's notes section,
which arrives as Markdown from the writer.

This page is the internal-facing artifact: anything the pipeline has to
say ABOUT the resume belongs here, never in the PDF an employer sees
(see pdf_renderer.py, "WHAT THE PDF MAY CONTAIN").
"""

from html import escape

import markdown as md_lib

from logger_setup import get_logger

log = get_logger(__name__)

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --text: #1a1a1a;
    --muted: #555;
    --accent: #1f4e79;
    --border: #ddd;
    --good: #1e7d34;
    --bad: #b3261e;
    --warn: #a15c00;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 0;
    background: #f4f4f4;
    font-family: "Helvetica Neue", Arial, sans-serif;
    color: var(--text);
    line-height: 1.55;
  }}
  .page {{
    max-width: 820px;
    margin: 0 auto;
    background: #fff;
    padding: 48px 56px;
    min-height: 100vh;
  }}
  h1 {{
    font-size: 1.9em;
    margin: 0 0 4px 0;
    color: var(--accent);
    border-bottom: 2px solid var(--accent);
    padding-bottom: 8px;
  }}
  h2 {{
    font-size: 1.15em;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--accent);
    margin-top: 28px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
  }}
  .subtitle {{
    color: var(--muted);
    margin: 0 0 20px 0;
  }}
  .scorecard {{
    display: flex;
    gap: 16px;
    margin: 16px 0 24px 0;
    flex-wrap: wrap;
  }}
  .score-box {{
    flex: 1;
    min-width: 200px;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
  }}
  .score-box .label {{
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin-bottom: 4px;
  }}
  .score-box .cycles {{
    font-size: 0.75em;
    color: var(--muted);
    margin-top: 2px;
  }}
  .score-box .score {{
    font-size: 2em;
    font-weight: 700;
    color: var(--accent);
  }}
  .score-box .score.good {{ color: var(--good); }}
  .score-box .score.warn {{ color: var(--warn); }}
  .score-box .score.bad {{ color: var(--bad); }}
  .verdict-banner {{
    padding: 10px 16px;
    border-radius: 6px;
    font-weight: 600;
    margin-bottom: 8px;
    margin-right: 8px;
    display: inline-block;
  }}
  .verdict-banner.approved {{
    background: #e6f4ea;
    color: var(--good);
    border: 1px solid #b7ddc0;
  }}
  .verdict-banner.pending {{
    background: #fdecea;
    color: var(--bad);
    border: 1px solid #f3c2bd;
  }}
  .job-link {{
    display: inline-block;
    margin: 8px 0 4px 0;
    padding: 10px 16px;
    background: var(--accent);
    color: #fff !important;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 600;
  }}
  .job-link:hover {{ opacity: 0.9; }}
  .job-meta {{ color: var(--muted); margin: 4px 0 16px 0; }}
  .trust-tag {{
    display: inline-block;
    font-size: 0.85em;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 12px;
    margin-left: 8px;
  }}
  .trust-tag.yes {{ background: #e6f4ea; color: var(--good); }}
  .trust-tag.no {{ background: #fdecea; color: var(--bad); }}
  .trust-tag.unknown {{ background: #f1f1f1; color: var(--muted); }}
  ul {{ margin: 6px 0 16px 0; padding-left: 22px; }}
  li {{ margin-bottom: 6px; }}
  p {{ margin: 6px 0; }}
  a {{ color: var(--accent); }}
  strong {{ color: #000; }}
  .cycle-note {{ color: var(--muted); font-size: 0.9em; margin-top: 32px; }}
  .notes {{
    border-left: 3px solid var(--border);
    padding: 2px 0 2px 18px;
    margin-top: 8px;
  }}
  .notes ul {{ margin: 6px 0 14px 0; }}
  .notes p {{ margin: 8px 0; }}
  @media (max-width: 600px) {{
    .page {{ padding: 24px 20px; }}
    .scorecard {{ flex-direction: column; }}
  }}
</style>
</head>
<body>
  <div class="page">
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>

    <div class="verdict-banner {job_verdict_class}">Job match: {job_verdict_text}</div>
    <div class="verdict-banner {resume_verdict_class}">Resume: {resume_verdict_text}</div>

    <div class="scorecard">
      <div class="score-box">
        <div class="label">Job match score</div>
        <div class="score {job_class}">{job_match_score}/10</div>
        <div class="cycles">{job_cycles_text}</div>
      </div>
      <div class="score-box">
        <div class="label">Resume fitness score</div>
        <div class="score {fitness_class}">{fitness_score}/10</div>
        <div class="cycles">after {resume_cycle_number} of max {max_resume_cycles} revise cycle(s)</div>
      </div>
    </div>

    <h2>Job application</h2>
    <p><strong>{job_title}</strong> at <strong>{company}</strong> ({location})</p>
    <p class="job-meta">Posted: {posted_date}
      <span class="trust-tag {trust_class}">{trust_text}</span>
    </p>
    <p><a class="job-link" href="{job_url}" target="_blank" rel="noopener noreferrer">Open job application &rarr;</a></p>

    <h2>{job_section_heading}</h2>
    <p>{job_match_summary}</p>
    <p><strong>Concerns</strong></p>
    <ul>{job_concerns_html}</ul>

    <h2>Stage 2 &mdash; Resume remarks</h2>
    <p>{fitness_summary}</p>
    <p><strong>Strengths</strong></p>
    <ul>{strengths_html}</ul>
    <p><strong>Gaps</strong></p>
    <ul>{gaps_html}</ul>
    <p><strong>Suggestions</strong></p>
    <ul>{suggestions_html}</ul>

{notes_section}
    <p class="cycle-note">{footer_note}</p>
  </div>
</body>
</html>
"""


def _score_class(score) -> str:
    if score is None:
        return ""
    if score >= 8:
        return "good"
    if score >= 5:
        return "warn"
    return "bad"


def _list_to_html(items) -> str:
    if not items:
        return "<li><em>None noted.</em></li>"
    return "".join(f"<li>{escape(str(item))}</li>" for item in items)


def _notes_section_html(writer_notes: str | None) -> str:
    """
    Render the writer's "Notes (not part of the resume)" section, or "".

    These are the writer's own flags about where the candidate genuinely
    doesn't match the posting. They used to be rendered into the resume
    PDF, which put internal commentary in front of employers; they land
    here instead, where they're useful interview prep. Not escaped: this
    is Markdown the writer produced, converted to HTML on purpose.
    """
    if not writer_notes or not writer_notes.strip():
        return ""
    try:
        body = md_lib.markdown(writer_notes, extensions=["extra", "sane_lists"])
    except Exception as exc:  # noqa: BLE001 — 3rd-party parser, never worth a failed page
        log.error("html_renderer: could not convert writer notes to HTML: %s", exc, exc_info=True)
        body = f"<pre>{escape(writer_notes)}</pre>"
    return (
        "    <h2>Writer's notes &mdash; gaps flagged (not part of the resume)</h2>\n"
        f'    <div class="notes">{body}</div>\n'
    )


def render_review_html(
    job_review: dict,
    resume_review: dict,
    job: dict,
    job_cycle_number: int,
    max_job_cycles: int,
    resume_cycle_number: int,
    max_resume_cycles: int,
    writer_notes: str | None = None,
    job_supplied: bool = False,
    title: str = "Resume & Job Review",
) -> str:
    """
    Render the judge's final verdicts from both pipeline stages — job
    match remarks + score, resume remarks + score — and the job
    application link, as a standalone HTML page.

    Args:
        job_review: The dict returned by judge.review_job() for the
            final job-search cycle (job_match_score, job_match_summary,
            job_link_trustworthy, job_concerns, approved).
        resume_review: The dict returned by judge.review_resume() for
            the final resume-revise cycle (fitness_score,
            fitness_summary, strengths, gaps, suggestions, approved).
        job: The job dict from search_agent.find_best_job() (the final,
            approved-or-cycle-exhausted pick).
        job_cycle_number: Which cycle the job review came from (1-indexed).
        max_job_cycles: The configured MAX_JOB_SEARCH_CYCLES, for display.
        resume_cycle_number: Which cycle the resume review came from (1-indexed).
        max_resume_cycles: The configured MAX_RESUME_REVISE_CYCLES, for display.
        writer_notes: The writer's "Notes (not part of the resume)" section
            as Markdown, from pdf_renderer.extract_notes(). Rendered here
            rather than in the resume PDF.
        job_supplied: True when the user supplied the job description and
            no search ran. The job score is then reported for information
            only — it did not gate anything, because the user had already
            decided to apply — and the page says so rather than implying
            a search stage that never happened.
        title: HTML <title> / page heading.

    Returns:
        A full HTML document as a string.
    """
    job_approved = job_review.get("approved", False)
    resume_approved = resume_review.get("approved", False)
    job_match_score = job_review.get("job_match_score")
    fitness_score = resume_review.get("fitness_score")
    trust = job_review.get("job_link_trustworthy")

    if trust is True:
        trust_class, trust_text = "yes", "Link looks trustworthy"
    elif trust is False:
        trust_class, trust_text = "no", "Link flagged as questionable"
    else:
        trust_class, trust_text = "unknown", "Trust not assessed"

    job_url = job.get("url") or "#"

    if job_supplied:
        subtitle = (
            "Resume written against a job description you supplied. "
            "No job search was run."
        )
        job_cycles_text = "assessed for information — this score gated nothing"
        job_section_heading = "Job match assessment (for your information)"
        footer_note = (
            f"Supplied job description — no search stage. "
            f"Resume revise stage: {resume_cycle_number}/{max_resume_cycles} cycle(s)."
        )
    else:
        subtitle = "Judge's final review — job match (stage 1) and resume fitness (stage 2)"
        job_cycles_text = f"after {job_cycle_number} of max {max_job_cycles} search cycle(s)"
        job_section_heading = "Stage 1 &mdash; Job match remarks"
        footer_note = (
            f"Job search stage: {job_cycle_number}/{max_job_cycles} cycle(s). "
            f"Resume revise stage: {resume_cycle_number}/{max_resume_cycles} cycle(s)."
        )

    try:
        return PAGE_TEMPLATE.format(
            title=escape(title),
            subtitle=subtitle,
            job_cycles_text=job_cycles_text,
            job_section_heading=job_section_heading,
            footer_note=footer_note,
            job_verdict_class="approved" if job_approved else "pending",
            job_verdict_text=(
                "Supplied by you" if job_supplied
                else ("✓ Approved" if job_approved else "✗ Not approved")
            ),
            resume_verdict_class="approved" if resume_approved else "pending",
            resume_verdict_text="✓ Approved — ready to send" if resume_approved else "✗ Not yet approved",
            job_class=_score_class(job_match_score),
            job_match_score=job_match_score if job_match_score is not None else "—",
            job_cycle_number=job_cycle_number,
            max_job_cycles=max_job_cycles,
            fitness_class=_score_class(fitness_score),
            fitness_score=fitness_score if fitness_score is not None else "—",
            resume_cycle_number=resume_cycle_number,
            max_resume_cycles=max_resume_cycles,
            job_title=escape(str(job.get("job_title") or "Unknown role")),
            company=escape(str(job.get("company") or "Unknown company")),
            location=escape(str(job.get("location") or "Unknown location")),
            posted_date=escape(str(job.get("posted_date") or "unknown")),
            trust_class=trust_class,
            trust_text=trust_text,
            job_url=escape(job_url, quote=True),
            job_match_summary=escape(str(job_review.get("job_match_summary") or "")),
            job_concerns_html=_list_to_html(job_review.get("job_concerns")),
            fitness_summary=escape(str(resume_review.get("fitness_summary") or "")),
            strengths_html=_list_to_html(resume_review.get("strengths")),
            gaps_html=_list_to_html(resume_review.get("gaps")),
            suggestions_html=_list_to_html(resume_review.get("suggestions")),
            notes_section=_notes_section_html(writer_notes),
        )
    except (KeyError, ValueError) as exc:
        # KeyError: a template placeholder wasn't supplied as a kwarg above
        # (a template/code mismatch). ValueError: a stray literal '{' or '}'
        # somewhere in review text broke str.format's parsing. Either way,
        # this should never happen in normal operation, but a broken HTML
        # render shouldn't crash the whole pipeline after the resume itself
        # (md + pdf) has already been saved successfully.
        log.error("html_renderer: failed to render review HTML: %s", exc, exc_info=True)
        raise RuntimeError(
            f"html_renderer: could not render the review HTML page — {exc}. "
            "The resume Markdown/PDF were saved separately and are unaffected."
        ) from exc


STOP_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{ --text: #1a1a1a; --muted: #555; --accent: #1f4e79; --border: #ddd; --bad: #b3261e; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0; background: #f4f4f4;
    font-family: "Helvetica Neue", Arial, sans-serif;
    color: var(--text); line-height: 1.55;
  }}
  .page {{ max-width: 820px; margin: 0 auto; background: #fff; padding: 48px 56px; min-height: 100vh; }}
  h1 {{ font-size: 1.9em; margin: 0 0 4px 0; color: var(--accent);
       border-bottom: 2px solid var(--accent); padding-bottom: 8px; }}
  h2 {{ font-size: 1.15em; text-transform: uppercase; letter-spacing: 0.04em;
       color: var(--accent); margin-top: 28px; margin-bottom: 8px;
       border-bottom: 1px solid var(--border); padding-bottom: 4px; }}
  .subtitle {{ color: var(--muted); margin: 0 0 20px 0; }}
  .banner {{ padding: 14px 18px; border-radius: 6px; font-weight: 600;
             background: #fdecea; color: var(--bad); border: 1px solid #f3c2bd; }}
  .reason {{ margin: 16px 0 0 0; font-size: 1.05em; white-space: pre-wrap; }}
  .considered {{ border: 1px solid var(--border); border-radius: 8px;
                 padding: 14px 18px; margin-bottom: 12px; }}
  .considered .head {{ font-weight: 600; }}
  .considered .score {{ color: var(--muted); font-size: 0.9em; margin: 2px 0 8px 0; }}
  ul {{ margin: 6px 0 0 0; padding-left: 22px; }}
  li {{ margin-bottom: 4px; }}
  p {{ margin: 6px 0; }}
  a {{ color: var(--accent); }}
  code {{ background: #f1f1f1; padding: 1px 5px; border-radius: 3px; font-size: 0.92em; }}
  .note {{ color: var(--muted); font-size: 0.9em; margin-top: 32px; }}
  @media (max-width: 600px) {{ .page {{ padding: 24px 20px; }} }}
</style>
</head>
<body>
  <div class="page">
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>

    <div class="banner">{banner}</div>
    <p class="reason">{reason}</p>

{sections_html}
    <p class="note">Run {run_timestamp}.</p>
  </div>
</body>
</html>
"""


def _section(heading: str, body_html: str) -> str:
    return f"    <h2>{heading}</h2>\n    {body_html}\n"


def _bullets(items: list[str]) -> str:
    """Bullet list from pre-built HTML fragments (callers may include <code>)."""
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def _render_stop_page(
    title: str,
    subtitle: str,
    banner: str,
    reason: str,
    sections_html: str,
    run_timestamp: str | None,
) -> str:
    """
    Shared renderer for every page that reports a run stopping early.

    One template, several callers: a run can end before the writing stage
    for reasons that have nothing in common (nothing worth applying to vs
    a posting that couldn't be fetched), and each needs its own "what to
    try next". Only the sections differ.
    """
    try:
        return STOP_TEMPLATE.format(
            title=escape(title),
            subtitle=escape(subtitle),
            banner=escape(banner),
            reason=escape(reason),
            sections_html=sections_html,
            run_timestamp=escape(str(run_timestamp or "unknown")),
        )
    except (KeyError, ValueError) as exc:
        log.error("html_renderer: failed to render the stop page: %s", exc, exc_info=True)
        raise RuntimeError(f"html_renderer: could not render the stop page — {exc}.") from exc


def _considered_html(job_history: list[dict]) -> str:
    """Render each posting the selection agent put forward, with the judge's verdict."""
    if not job_history:
        return "<p><em>No posting was put forward for review.</em></p>"

    blocks = []
    for entry in job_history:
        job = entry.get("job", {}) or {}
        review = entry.get("review", {}) or {}
        concerns = review.get("job_concerns")
        concerns_html = (
            "<ul>" + _list_to_html(concerns) + "</ul>"
            if isinstance(concerns, list) and concerns
            else ""
        )
        title = escape(str(job.get("job_title") or "No posting identified"))
        company = escape(str(job.get("company") or "—"))
        url = job.get("url")
        head = f"{title} at {company}"
        if url:
            head = (
                f'<a href="{escape(str(url), quote=True)}" target="_blank" '
                f'rel="noopener noreferrer">{head}</a>'
            )
        blocks.append(
            '<div class="considered">'
            f'<div class="head">Cycle {entry.get("cycle")} &mdash; {head}</div>'
            f'<div class="score">Judge score: {review.get("job_match_score")}/10</div>'
            f"<p>{escape(str(review.get('job_match_summary') or 'No verdict recorded.'))}</p>"
            f"{concerns_html}"
            "</div>"
        )
    return "".join(blocks)


def render_no_job_html(
    reason: str,
    job_history: list[dict],
    target_role: str,
    pool_size: int,
    max_job_cycles: int,
    min_viable_score: int,
    run_timestamp: str | None = None,
    title: str = "No suitable job found",
) -> str:
    """
    Render the page produced when a SEARCH run finds nothing worth
    applying to.

    This exists so a run that finds nothing still leaves a readable
    artifact. The alternative — writing a polished resume for a job the
    judge rated as a poor match — produces something that looks like a
    successful run and shouldn't be sent, which is worse than no output.

    Reports what was searched and every posting the selection agent put
    forward with the judge's verdict, so the outcome can be acted on
    (widen the search, lower the floor) rather than just re-run blindly.
    """
    searched = _bullets([
        f"Target role hint: <strong>{escape(str(target_role or '—'))}</strong>",
        f"Postings in the pool after duplicate and fit filtering: <strong>{pool_size}</strong>",
        f"Selection cycles run: <strong>{len(job_history)}</strong> of max {max_job_cycles}",
        f"Viability floor: <strong>{min_viable_score}/10</strong>",
    ])
    next_steps = _bullets([
        "Broaden the target role hint on the command line — it steers every search angle.",
        "Widen the intake window (<code>JOB_SEARCH_HOURS_OLD</code>) or the pool size "
        "(<code>JOB_POOL_TARGET_SIZE</code>, <code>JOB_POOL_OVERSCAN_FACTOR</code>) in config.py.",
        "Loosen the screening cut (<code>POOL_SCREEN_MIN_FIT</code>) if the pool came back small.",
        "Lower <code>MIN_VIABLE_JOB_SCORE</code> to let a weaker match through to the writer.",
        "If earlier runs consumed the good postings, check "
        "<code>data/output/selected_jobs.json</code> — anything listed there is filtered out.",
        "Or skip the search entirely and pass a posting URL or a saved description as "
        "the third argument.",
    ])

    sections = (
        _section("What was searched", searched)
        + _section("Postings considered", _considered_html(job_history))
        + _section("What to try next", next_steps)
    )
    return _render_stop_page(
        title=title,
        subtitle="No resume was written for this run.",
        banner="No suitable job found — pipeline stopped before the writing stage",
        reason=reason,
        sections_html=sections,
        run_timestamp=run_timestamp,
    )


def render_job_unavailable_html(
    reason: str,
    job_source: str,
    run_timestamp: str | None = None,
    title: str = "Job posting could not be read",
) -> str:
    """
    Render the page produced when a SUPPLIED job posting can't be fetched
    or read.

    Distinct from render_no_job_html: nothing here is a judgement about
    match quality. The user named one job and the pipeline could not get
    its text, so there is nothing to tailor against and no second
    candidate to fall back to. The reason carries the fetch error
    verbatim, including its copy-paste workaround, because that is the
    action the reader needs.
    """
    requested = _bullets([f"Requested: <code>{escape(str(job_source))}</code>"])
    next_steps = _bullets([
        "LinkedIn serves its signup wall to guest traffic unpredictably — retrying "
        "in a few minutes sometimes works.",
        "Otherwise open the posting, copy the description into a .txt file, and pass "
        "that file as the third argument instead of the URL.",
        "Paste the posting's URL into that file on its own line, so the review page "
        "still links to the application.",
        "Check the link is still live and public — expired postings and "
        "login-walled pages both land here.",
    ])

    sections = (
        _section("What was requested", requested)
        + _section("What to try next", next_steps)
    )
    return _render_stop_page(
        title=title,
        subtitle="No resume was written for this run.",
        banner="Could not read the job posting — pipeline stopped before the writing stage",
        reason=reason,
        sections_html=sections,
        run_timestamp=run_timestamp,
    )
