"""
Reflector agent.

Reviews the drafted resume against the job requirements and returns
a structured quality assessment: fitness score, strengths, gaps, and
concrete improvement suggestions. Uses adaptive thinking at high
effort so it reasons carefully rather than pattern-matching a quick
verdict — this is a judgment call, not a generation task.

Note on "reasoning model": Claude doesn't have a separate reasoning
model line. Instead, Sonnet 5 (and other current models) support
*adaptive thinking* — a mode where the model reasons through a hidden
scratchpad before answering. That's what's used here via
`thinking={"type": "adaptive"}` + `output_config={"effort": "high"}`.
"""

import json
import anthropic

from config import ANTHROPIC_API_KEY, MODEL_REFLECTOR

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a skeptical, experienced hiring manager and resume \
reviewer. You will be shown a tailored resume and the job requirements it was \
written for. Your job is to critically evaluate whether this resume would \
actually help the candidate get an interview — not to be encouraging for its \
own sake.

Assess:
1. Fitness for the role: how well the resume's content actually matches what \
the job requires, on a 1-10 scale.
2. Strengths: what genuinely works well and should stay.
3. Gaps or weaknesses: missing keywords, vague or unquantified claims, \
misaligned emphasis, anything an ATS or a skimming recruiter would flag.
4. Concrete improvement suggestions: specific, actionable edits — not generic \
advice like "add more detail."

Be honest even when the news isn't good. A resume that overclaims or is \
poorly matched to the role does the candidate no favors if you sugarcoat it.

Respond ONLY with valid JSON in this exact shape, no other text:
{
  "fitness_score": <1-10 integer>,
  "fitness_summary": "<1-2 sentence verdict>",
  "strengths": ["<point>", ...],
  "gaps": ["<point>", ...],
  "suggestions": ["<specific, actionable edit>", ...]
}"""


def review_resume(resume_draft: str, gathered_context: str) -> dict:
    """
    Critically review a drafted resume against the job it targets.

    Args:
        resume_draft: The writer agent's output (markdown resume).
        gathered_context: The orchestrator's gathered context (existing
            resume + job requirements) — gives the reflector the same
            grounding the writer had, so it can judge fit accurately.

    Returns:
        Dict with keys: fitness_score, fitness_summary, strengths,
        gaps, suggestions.
    """
    response = client.messages.create(
        model=MODEL_REFLECTOR,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Job requirements and background context:\n{gathered_context}\n\n"
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
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Model occasionally wraps JSON in prose despite instructions —
        # fall back to returning the raw text so nothing is silently lost.
        return {
            "fitness_score": None,
            "fitness_summary": "Could not parse structured review.",
            "strengths": [],
            "gaps": [],
            "suggestions": [],
            "raw_response": raw_text,
        }
