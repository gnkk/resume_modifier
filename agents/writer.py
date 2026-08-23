"""
Writing agent.

Takes the context gathered by the orchestrator (existing resume +
target job requirements) and drafts tailored resume content.
Runs on Sonnet 5 — this is the output that represents you to
employers, so it's not the place to cut cost.
"""

import anthropic

from config import ANTHROPIC_API_KEY, MODEL_WRITER

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are an expert resume writer. Given a candidate's existing \
resume content and a target job's requirements, produce a tailored resume.

Rules:
- Never fabricate experience, skills, or achievements the candidate doesn't have.
- Reframe and reorder existing content to emphasize what's relevant to the target role.
- Use strong, specific action verbs and quantify impact where the source material allows it.
- Keep formatting clean and ATS-friendly: plain section headers, no tables/graphics.
- Flag clearly (in a separate notes section) any gaps between the candidate's \
experience and the job's requirements — do not paper over them."""


def draft_resume(gathered_context: str) -> str:
    """
    Draft a tailored resume from gathered context.

    Args:
        gathered_context: Output from orchestrator.gather_context() —
            existing resume content + target job requirements.

    Returns:
        Tailored resume text (markdown), plus a notes section on gaps.
    """
    response = client.messages.create(
        model=MODEL_WRITER,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Here is the gathered context (existing resume + target "
                    f"job requirements):\n\n{gathered_context}\n\n"
                    "Draft the tailored resume now."
                ),
            }
        ],
    )
    text_blocks = [b.text for b in response.content if b.type == "text"]
    return "\n".join(text_blocks)
