"""
Judge agent (formerly "reflector").

Reviews a drafted resume against the target job's requirements and
returns a structured quality assessment: fitness score, strengths,
gaps, concrete improvement suggestions, and an explicit approval
decision. Uses adaptive thinking at high effort so it reasons
carefully rather than pattern-matching a quick verdict — this is a
judgment call, not a generation task.

This agent drives the writer<->judge revise loop in main.py: it
decides when a draft is good enough (approved=True, or the score
clears JUDGE_APPROVAL_SCORE) versus when another revision cycle is
worth running.

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

SYSTEM_PROMPT = f"""You are a skeptical, experienced hiring manager and resume \
reviewer, acting as the final judge before a resume goes out the door. You \
will be shown a tailored resume and the specific job it was written for. \
Your job is to critically evaluate whether this resume would actually help \
the candidate get an interview — not to be encouraging for its own sake.

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
    Critically review a drafted resume against the job it targets.

    Args:
        resume_draft: The writer agent's output (Markdown resume) for
            this cycle.
        candidate_context: The context agent's gathered candidate summary
            — gives the judge the same grounding the writer had.
        job: The job dict from search_agent.find_best_job().

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
        system=SYSTEM_PROMPT,
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
