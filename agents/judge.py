"""
Judge agent (formerly "reflector").

Runs as TWO independent evaluations, used in two separate sequential
loops (see main.py):

  1. review_job() — evaluates ONLY the job the search agent picked:
     is this genuinely the single best match for the candidate, does
     the posting look live and current, does the application URL look
     like a real, working, direct application link. Drives the
     search-agent <-> judge loop (up to MAX_JOB_SEARCH_CYCLES cycles,
     or early approval) that runs BEFORE any resume is written.

  2. review_resume() — evaluates ONLY the drafted resume against the
     (by then already-approved-or-cycle-exhausted) job: fitness score,
     strengths, gaps, concrete improvement suggestions. Drives the
     writer <-> judge loop (up to MAX_RESUME_REVISE_CYCLES cycles, or
     early approval) that runs AFTER the job stage is done.

These used to be one combined call; they're now split because the job
pick should be locked in (or exhausted) before the writer ever starts
drafting — there's no point revising a resume for a job that might
still change.

Both use adaptive thinking at high effort so the judge reasons
carefully rather than pattern-matching a quick verdict — these are
judgment calls, not generation tasks.

Note on "reasoning model": Claude doesn't have a separate reasoning
model line. Instead, Sonnet 5 (and other current models) support
*adaptive thinking* — a mode where the model reasons through a hidden
scratchpad before answering. That's what's used here via
`thinking={"type": "adaptive"}` + `output_config={"effort": "high"}`.
"""

import json
import anthropic

from config import ANTHROPIC_API_KEY, MODEL_JUDGE, JUDGE_APPROVAL_SCORE

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# --- Stage 1: job match review ---

JOB_SYSTEM_PROMPT = f"""You are a skeptical, experienced hiring manager \
acting as a judge over a job-search agent's work. You will be shown a \
candidate's background and the SINGLE job posting a search agent picked as \
its best match. A resume has NOT been written yet — your job right now is \
only to judge whether this was genuinely the right job to target, not to \
evaluate any resume.

Assess:
1. Job match quality: on a 1-10 scale, is this genuinely the single best \
match the search agent could realistically have found for this candidate, \
given the candidate's real, demonstrated background? Judge the pick itself \
— seniority fit, domain fit, skill overlap that's substantive rather than \
superficial keyword matching.
2. Job posting trustworthiness: does this look like a live, current posting \
(not stale, filled, or a generic evergreen listing), and does the URL look \
like a real, direct, working application link rather than a broken, \
placeholder, or overly generic careers-homepage URL? Flag anything suspicious.
3. Concerns: any specific reasons to doubt this was the right pick — a \
better angle the search agent likely missed, a seniority or location \
mismatch, a posting that reads as closed/expired despite the recency filter, \
or a link that doesn't look like it would actually let someone apply. Be \
specific enough that the search agent could act on this feedback and find a \
genuinely different, better posting next time.
4. An explicit approval decision: set "approved" to true only if this job is \
a genuinely strong, trustworthy match with a working-looking application \
link — good enough that a resume should now be written for it. A job scoring \
{JUDGE_APPROVAL_SCORE} or higher should usually be approved, but you may \
withhold approval below that if there's a real issue, or approve slightly \
below it if remaining concerns are inherent to a thin job market rather than \
a fixable search problem (never approve just because cycles are running out \
— say so plainly in job_match_summary instead).

Be honest even when the news isn't good. If you're reviewing a re-pick after \
rejecting an earlier one, say plainly whether this new pick actually resolves \
the concerns you raised before, or whether the search agent missed the point.

Respond ONLY with valid JSON in this exact shape, no other text:
{{
  "job_match_score": <1-10 integer>,
  "job_match_summary": "<1-2 sentence verdict>",
  "job_link_trustworthy": <true/false>,
  "job_concerns": ["<specific, actionable point>", ...],
  "approved": <true/false>
}}"""


def review_job(job: dict, candidate_context: str) -> dict:
    """
    Critically review the job the search agent picked — no resume
    involved yet. Drives the search-agent <-> judge loop that runs
    BEFORE the writer starts drafting.

    Args:
        job: The job dict from search_agent.find_best_job().
        candidate_context: The context agent's gathered candidate summary.

    Returns:
        Dict with keys: job_match_score, job_match_summary,
        job_link_trustworthy, job_concerns, approved.
    """
    job_block = (
        f"Job picked by the search agent:\n"
        f"  Title: {job.get('job_title')}\n"
        f"  Company: {job.get('company')}\n"
        f"  Location: {job.get('location')}\n"
        f"  Posting URL: {job.get('url')}\n"
        f"  Posted date: {job.get('posted_date')}\n"
        f"  Search agent's own rationale for this pick: {job.get('match_rationale')}\n"
        f"  Search agent's notes on angles tried: {job.get('search_notes')}\n"
        f"Requirements/responsibilities:\n{job.get('full_requirements')}\n"
    )
    response = client.messages.create(
        model=MODEL_JUDGE,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=JOB_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Candidate background:\n{candidate_context}\n\n"
                    f"{job_block}\n"
                    "Evaluate whether this was the right job to target now."
                ),
            }
        ],
    )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    raw_text = "\n".join(text_blocks).strip()

    try:
        result = json.loads(raw_text)
        result.setdefault("approved", result.get("job_match_score", 0) >= JUDGE_APPROVAL_SCORE)
        return result
    except json.JSONDecodeError:
        return {
            "job_match_score": None,
            "job_match_summary": "Could not parse structured review.",
            "job_link_trustworthy": None,
            "job_concerns": [],
            "approved": False,
            "raw_response": raw_text,
        }


# --- Stage 2: resume review ---

RESUME_SYSTEM_PROMPT = f"""You are a skeptical, experienced hiring manager and \
resume reviewer, acting as the final judge before a resume goes out the door. \
You will be shown a tailored resume and the specific job it was written for \
(that job has already been separately vetted and approved — you do not need \
to re-judge the job itself here, only the resume). Your job is to critically \
evaluate whether this resume would actually help the candidate get an \
interview — not to be encouraging for its own sake.

Assess:
1. Fitness for the role: how well the resume's content actually matches what \
the job requires, on a 1-10 scale.
2. Strengths: what genuinely works well and should stay.
3. Gaps or weaknesses: missing keywords, vague or unquantified claims, \
misaligned emphasis, anything an ATS or a skimming recruiter would flag.
4. Concrete improvement suggestions: specific, actionable edits — not generic \
advice like "add more detail."
5. An explicit approval decision: set "approved" to true only if this resume \
is genuinely ready to send as-is, with no further meaningful improvement \
available from another revision cycle. A resume scoring {JUDGE_APPROVAL_SCORE} \
or higher should usually be approved, but you may withhold approval below \
that if there's a real issue, or approve slightly below it if the remaining \
gaps are inherent to the candidate's actual background rather than fixable \
writing problems (never approve to paper over a genuine mismatch — say so \
in fitness_summary instead).

Be honest even when the news isn't good. A resume that overclaims or is \
poorly matched to the role does the candidate no favors if you sugarcoat it. \
If you've reviewed a previous version of this resume, say plainly whether \
this revision actually improved on the specific issues you raised before, or \
whether the writer missed the point.

Respond ONLY with valid JSON in this exact shape, no other text:
{{
  "fitness_score": <1-10 integer>,
  "fitness_summary": "<1-2 sentence verdict>",
  "strengths": ["<point>", ...],
  "gaps": ["<point>", ...],
  "suggestions": ["<specific, actionable edit>", ...],
  "approved": <true/false>
}}"""


def review_resume(resume_draft: str, candidate_context: str, job: dict) -> dict:
    """
    Critically review a drafted resume against the (already-approved)
    target job. Drives the writer <-> judge loop that runs AFTER the
    job-search stage is done.

    Args:
        resume_draft: The writer agent's output (Markdown resume) for
            this cycle.
        candidate_context: The context agent's gathered candidate summary
            — gives the judge the same grounding the writer had.
        job: The job dict from search_agent.find_best_job() (already
            vetted by review_job() in the prior pipeline stage).

    Returns:
        Dict with keys: fitness_score, fitness_summary, strengths,
        gaps, suggestions, approved.
    """
    job_block = (
        f"Target job: {job.get('job_title')} at {job.get('company')} "
        f"({job.get('location')})\n"
        f"Requirements/responsibilities:\n{job.get('full_requirements')}\n"
    )
    response = client.messages.create(
        model=MODEL_JUDGE,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=RESUME_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Candidate background:\n{candidate_context}\n\n"
                    f"{job_block}\n"
                    f"Drafted resume to review:\n{resume_draft}\n\n"
                    "Evaluate this resume now."
                ),
            }
        ],
    )

    # With adaptive thinking, response.content includes a "thinking" block
    # (the reasoning, exposed for debugging) before the "text" block with
    # the actual JSON answer — pull out just the text.
    text_blocks = [b.text for b in response.content if b.type == "text"]
    raw_text = "\n".join(text_blocks).strip()

    try:
        result = json.loads(raw_text)
        result.setdefault("approved", result.get("fitness_score", 0) >= JUDGE_APPROVAL_SCORE)
        return result
    except json.JSONDecodeError:
        # Model occasionally wraps JSON in prose despite instructions —
        # fall back to returning the raw text so nothing is silently lost.
        # approved=False so the loop treats this cycle as needing another pass
        # rather than silently exiting.
        return {
            "fitness_score": None,
            "fitness_summary": "Could not parse structured review.",
            "strengths": [],
            "gaps": [],
            "suggestions": [],
            "approved": False,
            "raw_response": raw_text,
        }
