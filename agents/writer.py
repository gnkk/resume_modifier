"""
Writing agent.

Takes the candidate context + the single job the search agent picked,
and drafts a tailored, ATS-ready resume. Also handles REVISIONS: when
the judge sends back a critique, this agent rewrites against that
specific feedback rather than starting over, so each cycle actually
converges.

Template fidelity:
The sample/template resume's extracted TEXT is passed to this agent
verbatim (see main.py), not as a prose description of it. That is
deliberate. Summarizing a layout into sentences like "medium-length
bullets, dates as 'Jan 2022 - Present'" loses exactly the detail that
makes output look like the template — the writer ends up
reconstructing a generic resume from a vague description. Handing over
the real thing lets the writer copy the structure directly. Only
structure transfers; the template's factual content belongs to someone
else and is explicitly off-limits.

Runs on Sonnet 5 — this is the output that represents you to
employers, so it's not the place to cut cost.
"""

import anthropic

from config import ANTHROPIC_API_KEY, MODEL_WRITER, RESUME_MAX_PAGES, RESUME_PREFERRED_PAGES
from logger_setup import get_logger, note_model

log = get_logger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = f"""You are an expert, professional resume writer and ATS \
(Applicant Tracking System) optimization specialist with years of experience \
getting candidates interviews at competitive companies. Given a candidate's \
background and a specific target job's requirements, produce a tailored, \
high-standard, ATS-ready resume that would both (a) parse cleanly through \
automated screening software and (b) read well to a human hiring manager. \
Write it the way an experienced professional resume writer would hand it to \
a paying client: polished, concise, and free of filler.

Content rules:
- Never fabricate experience, skills, employers, dates, or achievements the \
candidate doesn't have. If the source material only supports a vaguer claim, \
write the vaguer claim — do not round up.
- Reframe and reorder existing content to emphasize what's relevant to the \
target role. Mirror the job posting's own terminology where the candidate's \
real experience genuinely supports it (e.g. if the posting says "stakeholder \
management" and the candidate has done that under a different label, use the \
posting's term) — this is legitimate ATS keyword alignment, not fabrication.
- Use strong, specific action verbs (led, built, reduced, launched, \
architected — vary them, don't repeat the same verb twice) and quantify \
impact wherever the source material allows it (%, $, time saved, scale, team \
size). Never invent a number that isn't grounded in the source material.
- Flag clearly, in a separate "Notes" section at the very end (outside the \
resume body itself, not mixed into it), any real gaps between the \
candidate's experience and the job's requirements — do not paper over them, \
and do not let this section appear as part of the resume proper.

ATS-readiness rules (these are non-negotiable — a resume that fails ATS \
parsing never reaches a human, regardless of content quality):
- Use standard, literal section headers an ATS parser expects when no \
template is supplied: "Summary" (or "Professional Summary"), "Experience" \
(or "Work Experience"), "Skills", "Education", and "Projects" or \
"Certifications" only if the candidate actually has relevant content for \
them. Do not get creative with header names (no "My Journey", "What I \
Bring", etc.). When a template IS supplied, its headers win over this list.
- One column, top-to-bottom, linear reading order. No tables, no text boxes, \
no multi-column layouts, no headers/footers with content an ATS would miss — \
Markdown output naturally avoids these, but do not try to simulate columns \
with pipe tables or side-by-side content either.
- Contact info (name, phone, email, location, LinkedIn/portfolio if the \
candidate has one) goes in plain text at the very top, not embedded in an \
image, header, or footer.
- Standard, unambiguous date formats for every role and degree (e.g. "Jan \
2022 – Present", "2019 – 2022") — consistent format throughout the whole \
document, not mixed styles.
- A dedicated Skills section listing concrete technologies, tools, methods, \
and certifications as a scannable list — this is what most ATS keyword \
matching leans on hardest, so make sure every skill genuinely evidenced \
elsewhere in the resume (and relevant to the target job) appears here too.
- Bullet points under each role, not paragraphs — one accomplishment or \
responsibility per bullet, front-loaded with the action verb.
- No decorative characters, emoji, symbols, or Unicode bullets beyond a plain \
"-" or "*" — some ATS parsers mangle non-standard characters into garbage.
- Keep it within the hard page cap defined below (see "Length") — do not \
pad with filler to reach a target length, and do not force in lower-value \
content just to fill space; cut it instead.

Template fidelity (highest-priority formatting rule):
You may be given the full text of a TEMPLATE RESUME, in a clearly \
delimited block. When present, it is the layout specification for your \
output — not inspiration, not a loose cue. Replicate it exactly:
- Use its section headers, with its exact wording, capitalization, and \
top-to-bottom order. If it says "WORK EXPERIENCE", do not write \
"Professional Experience". If it has no Certifications section, do not \
invent one.
- Reproduce its contact-info block layout: same fields, same order, same \
number of lines, same separators.
- Reproduce the shape of each entry: whether title and company share a \
line, where dates sit relative to them, whether location appears, and the \
exact date format. Every entry in a section must use an IDENTICAL shape — \
never two-line headers in one section and three-line in another.
- Match its bullet count per role and typical bullet length.
- Match how it organizes Skills (flat list vs grouped, and if grouped, \
similar group count and naming style).
- Match its overall density and its summary length. If its summary is \
three lines, yours is three lines — do not write a 180-word paragraph \
because you have more to say.
Where the template is silent on something, follow the ATS rules below. \
Where it would directly violate an ATS rule (e.g. multi-column layout), \
keep its section order, grouping, and emphasis but express them in a \
single-column layout.
NEVER take factual content from the template — no names, employers, dates, \
numbers, or achievements. It supplies structure; the candidate supplies \
every fact. If no template is given, use the standard ATS section set below.

Self-check before you output (fix any that fail):
- Every entry within a section uses the identical header shape.
- No section label is repeated as the first words of its own content \
(never a "Certifications" section whose first bullet begins "Certifications:").
- Each item sits under the heading it actually belongs to — language test \
scores are not certifications; personal projects are not academic projects.
- Dates use one consistent format everywhere.
- Nothing appears in two places unless the template does that.

Length: this resume must be AT MOST {RESUME_MAX_PAGES} pages, with \
{RESUME_PREFERRED_PAGES} pages preferred, once rendered. As a rough guide at \
standard resume font size/margins, {RESUME_PREFERRED_PAGES} pages is \
approximately 600-800 words of resume content (excluding the Notes \
section), and {RESUME_MAX_PAGES} pages is approximately 900-1100 words — use \
this only as a sanity check while drafting, not a target to fill. Prioritize \
the candidate's most relevant, highest-impact content for this specific job; \
cut older, less relevant, or lower-impact material rather than exceeding the \
page cap. Never pad with filler to reach a target length either — a strong, \
focused {RESUME_PREFERRED_PAGES}-page resume beats a padded 3-page one every \
time.

Output format:
- Output the resume as clean Markdown (headers, bullet lists) — it will \
later be rendered to PDF, so avoid raw HTML and avoid decorative characters \
that don't map cleanly to Markdown.
- Spacing, margins, fonts, and alignment are applied later by a stylesheet. \
Do not try to control appearance from Markdown — no padding with blank \
lines, no runs of spaces or tabs to align dates, no ASCII rules, no tables \
used for layout. Get the STRUCTURE right (heading levels, one fact per \
line, consistent entry shape) and the rendering handles how it looks.
- Put each part of an entry on its own line — title line, then the meta line \
(company, location, dates) — rather than running them together.
- The gaps/notes section, if needed, goes at the very end under its own \
"## Notes (not part of the resume)" heading so it's unambiguous that it's \
not resume content.

When you are given prior feedback from a reviewing judge along with your own \
previous draft, treat the feedback as the priority: address every specific \
gap and suggestion the judge raised, don't just lightly reword the same \
draft. Preserve what the judge called out as a strength."""


# Generous token budget on purpose. A 3-page resume is only ~1,500 tokens of
# actual output, but Sonnet 5 may spend tokens reasoning before writing, and
# that reasoning counts against max_tokens — too small a budget gets the model
# cut off before it emits ANY resume text, yielding an empty draft rather than
# a short one. There's no cost to headroom that isn't used, and the failure
# mode without it is silent and expensive (an empty resume flows all the way
# through the judge loop to the rendered output).
MAX_OUTPUT_TOKENS = 16000


def _extract_text(response, where: str) -> str:
    """
    Pull answer text out of the response, logging diagnostics when it's
    empty or truncated so this failure is never silent again.
    """
    text_blocks = [b.text for b in response.content if b.type == "text"]
    text = "\n".join(text_blocks)

    if not text.strip():
        log.error(
            "%s: model returned no text (stop_reason=%s, usage=%s). If stop_reason "
            "is 'max_tokens', the budget was exhausted before any resume text was "
            "written — raise MAX_OUTPUT_TOKENS.",
            where, getattr(response, "stop_reason", "unknown"), getattr(response, "usage", None),
        )
    elif getattr(response, "stop_reason", None) == "max_tokens":
        log.error(
            "%s: resume was truncated mid-write (stop_reason=max_tokens, usage=%s). "
            "The output is incomplete — raise MAX_OUTPUT_TOKENS.",
            where, getattr(response, "usage", None),
        )

    return text


def _template_block(style_template: str | None) -> str:
    """
    Wrap the template resume's raw text in clear delimiters, or return an
    empty string when no template is available. Delimiters matter: the
    writer must never mistake the template's factual content for the
    candidate's.
    """
    if not style_template or not style_template.strip():
        return ""
    if style_template.lstrip().startswith("[read_pdf failed"):
        # pdf_reader returns a readable failure string rather than raising;
        # passing that through as a "template" would be worse than none.
        log.error("writer: template resume could not be read (%s) — proceeding without it.", style_template.strip()[:200])
        return ""
    return (
        "=== TEMPLATE RESUME (LAYOUT SPEC — COPY ITS STRUCTURE, NEVER ITS "
        "CONTENT) ===\n"
        f"{style_template}\n"
        "=== END TEMPLATE RESUME ===\n\n"
        "Everything above belongs to a different person. Reproduce its "
        "section headers, ordering, entry shapes, date formats, bullet "
        "density, and overall length. Take no facts from it.\n\n"
    )


def _settled_gaps_block(job_review: dict | None) -> str:
    """
    Tell the writer which shortfalls were already accepted when the job
    was chosen, so it stops trying to write around them.

    Without this the writer sees a posting requirement the candidate
    can't evidence and reaches for the only tools it has: vague phrasing
    that implies coverage, or a skills line padded with a term nothing
    else in the resume supports. Naming the gap as settled removes the
    pressure — the correct response is to spend the space on real
    strengths and flag the gap in Notes, not to paper over it.
    """
    if not job_review:
        return ""
    concerns = job_review.get("job_concerns")
    if not isinstance(concerns, list) or not concerns:
        return ""
    listed = "\n".join(f"- {concern}" for concern in concerns)
    return (
        "=== KNOWN GAPS, ALREADY ACCEPTED ===\n"
        f"{listed}\n"
        "These were weighed and accepted when this job was selected. Do NOT "
        "try to cover them: no vague phrasing that implies experience the "
        "candidate lacks, no skills-line keywords nothing else in the resume "
        "evidences, no stretched reframing of unrelated work. A gap cannot be "
        "written away — only faked. Spend the space on what the candidate has "
        "genuinely done that is closest to the requirement, and record the gap "
        "in the Notes section instead.\n"
        "=== END KNOWN GAPS ===\n\n"
    )


def draft_resume(
    candidate_context: str,
    job: dict,
    style_template: str | None = None,
    job_review: dict | None = None,
) -> str:
    """
    Draft an initial tailored, ATS-ready resume.

    Args:
        candidate_context: Output from context_agent.gather_candidate_context().
        job: The job dict from search_agent.find_best_job().
        style_template: Raw extracted text of the sample/template resume,
            used as the layout spec. Passed verbatim rather than summarized
            — see this module's docstring for why.
        job_review: The stage-1 judge's verdict on this job, if available.
            Its concerns are passed through as already-accepted gaps the
            writer should not attempt to cover.

    Returns:
        Tailored resume text (Markdown), including a gaps/notes section.

    Raises:
        RuntimeError: if the underlying Claude API call fails, or the model
            returns no resume text at all.
    """
    job_block = (
        f"Target job: {job.get('job_title')} at {job.get('company')} "
        f"({job.get('location')})\n"
        f"Posting URL: {job.get('url')}\n"
        f"Requirements/responsibilities:\n{job.get('full_requirements')}\n"
    )
    try:
        response = client.messages.create(
            model=MODEL_WRITER,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{_template_block(style_template)}"
                        f"{_settled_gaps_block(job_review)}"
                        f"Candidate background:\n{candidate_context}\n\n"
                        f"{job_block}\n"
                        "Draft the tailored, ATS-ready resume now."
                    ),
                }
            ],
        )
    except anthropic.APIError as exc:
        log.error("writer.draft_resume: Anthropic API call failed: %s", exc, exc_info=True)
        raise RuntimeError(
            f"writer.draft_resume: failed to reach the Anthropic API ({exc}). "
            "Check your ANTHROPIC_API_KEY and network connection."
        ) from exc

    note_model(log, "writer", response)
    draft = _extract_text(response, "writer.draft_resume")
    if not draft.strip():
        # Fail loudly rather than returning "". An empty draft would otherwise
        # flow through the judge loop and be rendered as an empty resume — a
        # silent, expensive failure that looks like a successful run.
        raise RuntimeError(
            "writer.draft_resume: the model produced no resume text "
            f"(stop_reason={getattr(response, 'stop_reason', 'unknown')}). "
            "Nothing downstream can proceed without a draft. See the run log "
            "for token usage."
        )
    return draft


def revise_resume(
    candidate_context: str,
    job: dict,
    previous_draft: str,
    judge_feedback: dict,
    style_template: str | None = None,
    job_review: dict | None = None,
) -> str:
    """
    Revise a resume draft based on the judge's critique.

    Args:
        candidate_context: Output from context_agent.gather_candidate_context().
        job: The job dict from search_agent.find_best_job().
        previous_draft: The resume Markdown from the prior cycle.
        judge_feedback: The dict returned by judge.review_resume() for the
            previous draft (fitness_score, gaps, suggestions, etc.).
        style_template: Raw extracted text of the sample/template resume —
            passed on every cycle, not just the first, so revisions can't
            drift away from the template while chasing the judge's feedback.

    Returns:
        Revised tailored resume text (Markdown).

    Raises:
        RuntimeError: if the underlying Claude API call fails.
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
    try:
        response = client.messages.create(
            model=MODEL_WRITER,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{_template_block(style_template)}"
                        f"{_settled_gaps_block(job_review)}"
                        f"Candidate background:\n{candidate_context}\n\n"
                        f"{job_block}\n"
                        f"Your previous draft:\n{previous_draft}\n\n"
                        f"Judge's feedback on that draft:\n{feedback_block}\n\n"
                        "Revise the resume now, addressing the feedback directly "
                        "while keeping it ATS-ready and faithful to the template."
                    ),
                }
            ],
        )
    except anthropic.APIError as exc:
        log.error("writer.revise_resume: Anthropic API call failed: %s", exc, exc_info=True)
        raise RuntimeError(
            f"writer.revise_resume: failed to reach the Anthropic API ({exc}). "
            "Check your ANTHROPIC_API_KEY and network connection."
        ) from exc

    note_model(log, "writer", response)
    revised = _extract_text(response, "writer.revise_resume")
    if not revised.strip():
        log.error(
            "writer.revise_resume: falling back to the previous draft so the "
            "pipeline doesn't lose content."
        )
        return previous_draft
    return revised
