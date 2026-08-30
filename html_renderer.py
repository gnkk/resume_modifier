"""
HTML renderer for the judge's review.

Renders the judge's final verdicts from BOTH pipeline stages — the
job-match review (stage 1: search agent <-> judge) and the resume
review (stage 2: writer agent <-> judge) — plus the job application
link, as a single standalone HTML page. This is the ONLY HTML output
this project produces; the resume itself is output as Markdown + PDF
(see pdf_renderer.py), not HTML.

Uses simple string templating (no Markdown conversion needed here,
since the review fields are already plain strings/lists).
"""

from html import escape

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
  @media (max-width: 600px) {{
    .page {{ padding: 24px 20px; }}
    .scorecard {{ flex-direction: column; }}
  }}
</style>
</head>
<body>
  <div class="page">
    <h1>{title}</h1>
    <p class="subtitle">Judge's final review — job match (stage 1) and resume fitness (stage 2)</p>

    <div class="verdict-banner {job_verdict_class}">Job match: {job_verdict_text}</div>
    <div class="verdict-banner {resume_verdict_class}">Resume: {resume_verdict_text}</div>

    <div class="scorecard">
      <div class="score-box">
        <div class="label">Job match score</div>
        <div class="score {job_class}">{job_match_score}/10</div>
        <div class="cycles">after {job_cycle_number} of max {max_job_cycles} search cycle(s)</div>
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

    <h2>Stage 1 &mdash; Job match remarks</h2>
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

    <p class="cycle-note">Job search stage: {job_cycle_number}/{max_job_cycles} cycle(s). Resume revise stage: {resume_cycle_number}/{max_resume_cycles} cycle(s).</p>
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


def render_review_html(
    job_review: dict,
    resume_review: dict,
    job: dict,
    job_cycle_number: int,
    max_job_cycles: int,
    resume_cycle_number: int,
    max_resume_cycles: int,
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

    return PAGE_TEMPLATE.format(
        title=escape(title),
        job_verdict_class="approved" if job_approved else "pending",
        job_verdict_text="✓ Approved" if job_approved else "✗ Not approved",
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
    )
