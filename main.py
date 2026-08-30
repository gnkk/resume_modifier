"""
Entry point: runs the resume-building pipeline end to end.

Pipeline:
    1. Context agent   -> reads resume PDF (+ GitHub MCP context, once GITHUB_PAT is set)
    2. Search agent     -> finds the single best-matching job (Firecrawl real-time search,
                            restricted to postings newer than one week)
    3. Writer <-> Judge -> writer drafts, judge critiques with reasoning;
                            loop feeds the judge's comments + prior draft back
                            to the writer for up to MAX_REVISE_CYCLES rounds,
                            or until the judge approves early
    4. Render output    -> tailored_resume.html (scrollable) + tailored_resume.pdf

Usage:
    python main.py <path_to_resume.pdf> "<target role/company description>" [path_to_job_description.txt]

Examples:
    python main.py data/input/my_resume.pdf "Data Scientist, remote Canada"
    python main.py data/input/my_resume.pdf "Data Scientist at Shopify" data/input/job_description.txt
"""

import sys
import os
import json

from agents.context_agent import gather_candidate_context
from agents.search_agent import find_best_job
from agents.writer import draft_resume, revise_resume
from agents.judge import review_resume
from html_renderer import render_resume_html
from pdf_renderer import render_resume_pdf
from config import DATA_OUTPUT_DIR, MAX_REVISE_CYCLES


def run_revise_loop(candidate_context: str, job: dict) -> tuple[str, dict, list[dict]]:
    """
    Run the writer <-> judge cycle up to MAX_REVISE_CYCLES times, or
    until the judge approves early.

    Returns:
        (final_draft, final_review, history) where history is a list of
        {"cycle": int, "review": dict} entries for every cycle run,
        useful for the saved review file / debugging.
    """
    history = []

    print(f"  [cycle 1/{MAX_REVISE_CYCLES}] Writer drafting initial resume...")
    draft = draft_resume(candidate_context, job)

    for cycle in range(1, MAX_REVISE_CYCLES + 1):
        print(f"  [cycle {cycle}/{MAX_REVISE_CYCLES}] Judge reviewing draft...")
        review = review_resume(draft, candidate_context, job)
        history.append({"cycle": cycle, "review": review})

        score = review.get("fitness_score")
        approved = review.get("approved", False)
        print(
            f"    -> score: {score}/10, approved: {approved}"
            + (f" | {review.get('fitness_summary')}" if review.get("fitness_summary") else "")
        )

        if approved or cycle == MAX_REVISE_CYCLES:
            return draft, review, history

        print(f"  [cycle {cycle + 1}/{MAX_REVISE_CYCLES}] Writer revising against judge feedback...")
        draft = revise_resume(candidate_context, job, draft, review)

    # Unreachable given the loop above, but keeps type-checkers happy.
    return draft, history[-1]["review"], history


def main():
    if len(sys.argv) not in (3, 4):
        print(
            'Usage: python main.py <path_to_resume.pdf> "<target role description>" '
            "[path_to_job_description.txt]"
        )
        sys.exit(1)

    resume_path, target_role = sys.argv[1], sys.argv[2]
    job_description_path = sys.argv[3] if len(sys.argv) == 4 else None

    print(f"[1/4] Gathering candidate context from resume + GitHub...")
    candidate_context = gather_candidate_context(resume_path, job_description_path)

    print(f"[2/4] Searching for the best-matching job (last week only)...")
    job = find_best_job(candidate_context, target_role)
    if not job.get("url"):
        print("  Warning: search agent could not confidently identify a job posting.")
        print(f"  Notes: {job.get('search_notes')}")
    else:
        print(f"  -> Best match: {job.get('job_title')} at {job.get('company')} ({job.get('location')})")
        print(f"     {job.get('url')}")

    print(f"[3/4] Running writer <-> judge revise loop (max {MAX_REVISE_CYCLES} cycles)...")
    final_draft, final_review, history = run_revise_loop(candidate_context, job)

    print("[4/4] Rendering output...")
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    md_path = os.path.join(DATA_OUTPUT_DIR, "tailored_resume.md")
    with open(md_path, "w") as f:
        f.write(final_draft)

    review_path = os.path.join(DATA_OUTPUT_DIR, "resume_review.json")
    with open(review_path, "w") as f:
        json.dump(
            {"job": job, "final_review": final_review, "cycle_history": history},
            f,
            indent=2,
        )

    title = f"{job.get('job_title', 'Tailored Resume')} — {job.get('company', '')}".strip(" —")
    html_doc = render_resume_html(final_draft, title=title)
    html_path = os.path.join(DATA_OUTPUT_DIR, "tailored_resume.html")
    with open(html_path, "w") as f:
        f.write(html_doc)

    pdf_path = os.path.join(DATA_OUTPUT_DIR, "tailored_resume.pdf")
    render_resume_pdf(html_doc, pdf_path)

    print(f"\nDone after {len(history)} cycle(s).")
    print(f"  Markdown: {md_path}")
    print(f"  HTML:     {html_path}")
    print(f"  PDF:      {pdf_path}")
    print(f"  Review:   {review_path}")
    if final_review.get("fitness_score") is not None:
        print(f"\nFinal fitness score: {final_review['fitness_score']}/10")
        print(f"Verdict: {final_review.get('fitness_summary', '')}")


if __name__ == "__main__":
    main()
