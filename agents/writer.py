"""
Writing agent.

Takes the candidate context + the single job the search agent picked,
and drafts a tailored resume. Also handles REVISIONS: when the judge
sends back a critique, this agent rewrites against that specific
feedback rather than starting over, so each cycle actually converges.

Runs on Sonnet 5 — this is the output that represents you to
employers, so it's not the place to cut cost.
"""

import anthropic

from config import ANTHROPIC_API_KEY, MODEL_WRITER

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are an expert resume writer. Given a candidate's \
background and a specific target job's requirements, produce a tailored resume.

Rules:
- Never fabricate experience, skills, or achievements the candidate doesn't have.
- Reframe and reorder existing content to emphasize what's relevant to the target role.
- Use strong, specific action verbs and quantify impact where the source material allows it.
- Keep formatting clean and ATS-friendly: plain section headers, no tables/graphics.
- Flag clearly (in a separate notes section at the end) any gaps between the \
candidate's experience and the job's requirements — do not paper over them.
- Output the resume as clean Markdown (headers, bullet lists) — it will later \
be rendered to PDF, so avoid raw HTML and avoid decorative characters \
that don't map cleanly to Markdown.

When you are given prior feedback from a reviewing judge along with your own \
previous draft, treat the feedback as the priority: address every specific gap \
and suggestion the judge raised, don't just lightly reword the same draft. \
Preserve what the judge called out as a strength."""


def draft_resume(candidate_context: str, job: dict) -> str:
    """
    Draft an initial tailored resume.

    Args:
        candidate_context: Output from context_agent.gather_candidate_context().
        job: The job dict from search_agent.find_best_job().

    Returns:
        Tailored resume text (Markdown), including a gaps/notes section.
    """
    job_block = (
        f"Target job: {job.get('job_title')} at {job.get('company')} "
        f"({job.get('location')})\n"
        f"Posting URL: {job.get('url')}\n"
        f"Requirements/responsibilities:\n{job.get('full_requirements')}\n"
    )
    response = client.messages.create(
        model=MODEL_WRITER,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Candidate background:\n{candidate_context}\n\n"
                    f"{job_block}\n"
                    "Draft the tailored resume now."
                ),
            }
        ],
    )
    text_blocks = [b.text for b in response.content if b.type == "text"]
    return "\n".join(text_blocks)


def revise_resume(
    candidate_context: str,
    job: dict,
    previous_draft: str,
    judge_feedback: dict,
) -> str:
    """
    Revise a resume draft based on the judge's critique.

    Args:
        candidate_context: Output from context_agent.gather_candidate_context().
        job: The job dict from search_agent.find_best_job().
        previous_draft: The resume Markdown from the prior cycle.
        judge_feedback: The dict returned by judge.review_resume() for the
            previous draft (fitness_score, gaps, suggestions, etc.).

    Returns:
        Revised tailored resume text (Markdown).
    """
    job_block = (
        f"Target job: {job.get('job_title')} at {job.get('company')} "
        f"({job.get('location')})\n"
        f"Requirements/responsibilities:\n{job.get('full_requirements')}\n"
    )
    feedback_block = (
        f"Fitness score given: {judge_feedback.get('fitness_score')}/10\n"
        f"Verdict: {judge_feedback.get('fitness_summary')}\n"
        f"Strengths to preserve: {judge_feedback.get('strengths')}\n"
        f"Gaps/weaknesses to fix: {judge_feedback.get('gaps')}\n"
        f"Specific suggestions to apply: {judge_feedback.get('suggestions')}\n"
    )
    response = client.messages.create(
        model=MODEL_WRITER,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Candidate background:\n{candidate_context}\n\n"
                    f"{job_block}\n"
                    f"Your previous draft:\n{previous_draft}\n\n"
                    f"Judge's feedback on that draft:\n{feedback_block}\n\n"
                    "Revise the resume now, addressing the feedback directly."
                ),
            }
        ],
    )
    text_blocks = [b.text for b in response.content if b.type == "text"]
    return "\n".join(text_blocks)
