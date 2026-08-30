"""
Entry point: runs the resume-building pipeline end to end.

Pipeline (two independent, SEQUENTIAL judge loops):
    1. Context agent        -> reads resume PDF (+ GitHub MCP context, once GITHUB_PAT is set)
    2. Search <-> Judge      -> search agent proposes a single best-matching job
                                (Firecrawl real-time search, restricted to postings newer
                                than one week); the judge evaluates ONLY that job pick
                                (match quality + link trustworthiness). If not approved,
                                the search agent tries again with the judge's feedback and
                                the rejected pick excluded. Runs for up to
                                MAX_JOB_SEARCH_CYCLES cycles, or until the judge approves
                                early — whichever comes first. Nothing else happens until
                                this stage finishes.
    3. Writer <-> Judge      -> only once a job is locked in from stage 2: the writer
                                drafts a resume for that job, and the judge evaluates
                                ONLY the resume (fitness score, strengths, gaps,
                                suggestions). If not approved, the writer revises against
                                the judge's feedback. Runs for up to
                                MAX_RESUME_REVISE_CYCLES cycles, or until the judge
                                approves early — whichever comes first.
    4. Render output         -> tailored_resume.md + tailored_resume.pdf (resume),
                                resume_review.html (both judge verdicts, both scores, and
                                the job application link) + resume_review.json (full detail)

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
from agents.judge import review_job, review_resume
from html_renderer import render_review_html
from pdf_renderer import render_resume_pdf
from config import DATA_OUTPUT_DIR, MAX_JOB_SEARCH_CYCLES, MAX_RESUME_REVISE_CYCLES


def run_job_search_loop(candidate_context: str, target_role: str) -> tuple[dict, dict, list[dict]]:
    """
    Run the search-agent <-> judge cycle up to MAX_JOB_SEARCH_CYCLES
    times, or until the judge approves the job pick early — whichever
    comes first. No resume is written during this stage; the judge is
    evaluating ONLY the job the search agent picked.

    Returns:
        (final_job, final_job_review, history) where history is a list
        of {"cycle": int, "job": dict, "review": dict} entries for
        every cycle run.
    """
    history = []
    rejected = []

    for cycle in range(1, MAX_JOB_SEARCH_CYCLES + 1):
        if cycle == 1:
            print(f"  [job cycle 1/{MAX_JOB_SEARCH_CYCLES}] Search agent finding best-matching job...")
            job = find_best_job(candidate_context, target_role)
        else:
            print(f"  [job cycle {cycle}/{MAX_JOB_SEARCH_CYCLES}] Search agent re-searching against judge feedback...")
            job = find_best_job(candidate_context, target_role, rejected_jobs=rejected)

        if not job.get("url"):
            print("    Warning: search agent could not confidently identify a job posting.")
            print(f"    Notes: {job.get('search_notes')}")

        print(f"  [job cycle {cycle}/{MAX_JOB_SEARCH_CYCLES}] Judge reviewing job pick...")
        review = review_job(job, candidate_context)
        history.append({"cycle": cycle, "job": job, "review": review})

        job_score = review.get("job_match_score")
        approved = review.get("approved", False)
        print(
            f"    -> {job.get('job_title')} at {job.get('company')} | "
            f"job match: {job_score}/10, approved: {approved}"
            + (f" | {review.get('job_match_summary')}" if review.get("job_match_summary") else "")
        )

        if approved or cycle == MAX_JOB_SEARCH_CYCLES:
            return job, review, history

        rejected.append({"job": job, "review": review})

    # Unreachable given the loop above, but keeps type-checkers happy.
    return history[-1]["job"], history[-1]["review"], history


def run_resume_revise_loop(candidate_context: str, job: dict) -> tuple[str, dict, list[dict]]:
    """
    Run the writer <-> judge cycle up to MAX_RESUME_REVISE_CYCLES
    times, or until the judge approves the resume early — whichever
    comes first. Only called once the job from run_job_search_loop is
    locked in; the judge is evaluating ONLY the resume here.

    Returns:
        (final_draft, final_review, history) where history is a list of
        {"cycle": int, "review": dict} entries for every cycle run.
    """
    history = []

    print(f"  [resume cycle 1/{MAX_RESUME_REVISE_CYCLES}] Writer drafting initial resume...")
    draft = draft_resume(candidate_context, job)

    for cycle in range(1, MAX_RESUME_REVISE_CYCLES + 1):
        print(f"  [resume cycle {cycle}/{MAX_RESUME_REVISE_CYCLES}] Judge reviewing resume draft...")
        review = review_resume(draft, candidate_context, job)
        history.append({"cycle": cycle, "review": review})

        score = review.get("fitness_score")
        approved = review.get("approved", False)
        print(
            f"    -> resume fitness: {score}/10, approved: {approved}"
            + (f" | {review.get('fitness_summary')}" if review.get("fitness_summary") else "")
        )

        if approved or cycle == MAX_RESUME_REVISE_CYCLES:
            return draft, review, history

        print(f"  [resume cycle {cycle + 1}/{MAX_RESUME_REVISE_CYCLES}] Writer revising against judge feedback...")
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

    print(f"[2/4] Running search <-> judge loop (max {MAX_JOB_SEARCH_CYCLES} cycles) to lock in a job...")
    job, job_review, job_history = run_job_search_loop(candidate_context, target_role)
    print(f"  -> Final pick: {job.get('job_title')} at {job.get('company')} ({job.get('location')})")
    print(f"     {job.get('url')}")

    print(f"[3/4] Running writer <-> judge loop (max {MAX_RESUME_REVISE_CYCLES} cycles) to tailor the resume...")
    final_draft, resume_review, resume_history = run_resume_revise_loop(candidate_context, job)

    print("[4/4] Rendering output...")
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    md_path = os.path.join(DATA_OUTPUT_DIR, "tailored_resume.md")
    with open(md_path, "w") as f:
        f.write(final_draft)

    review_path = os.path.join(DATA_OUTPUT_DIR, "resume_review.json")
    with open(review_path, "w") as f:
        json.dump(
            {
                "job": job,
                "job_search_stage": {
                    "final_review": job_review,
                    "cycle_history": job_history,
                },
                "resume_revise_stage": {
                    "final_review": resume_review,
                    "cycle_history": resume_history,
                },
            },
            f,
            indent=2,
        )

    title = f"{job.get('job_title', 'Tailored Resume')} — {job.get('company', '')}".strip(" —")

    pdf_path = os.path.join(DATA_OUTPUT_DIR, "tailored_resume.pdf")
    render_resume_pdf(final_draft, pdf_path, title=title)

    review_html = render_review_html(
        job_review,
        resume_review,
        job,
        job_cycle_number=len(job_history),
        max_job_cycles=MAX_JOB_SEARCH_CYCLES,
        resume_cycle_number=len(resume_history),
        max_resume_cycles=MAX_RESUME_REVISE_CYCLES,
        title=f"Review — {title}" if title else "Resume & Job Review",
    )
    review_html_path = os.path.join(DATA_OUTPUT_DIR, "resume_review.html")
    with open(review_html_path, "w") as f:
        f.write(review_html)

    print(f"\nDone. Job search: {len(job_history)} cycle(s). Resume revise: {len(resume_history)} cycle(s).")
    print(f"  Markdown:    {md_path}")
    print(f"  PDF:         {pdf_path}")
    print(f"  Review HTML: {review_html_path}")
    print(f"  Review JSON: {review_path}")
    print(f"\nJob match score:    {job_review.get('job_match_score')}/10 — {job_review.get('job_match_summary', '')}")
    print(f"Resume fitness score: {resume_review.get('fitness_score')}/10 — {resume_review.get('fitness_summary', '')}")


if __name__ == "__main__":
    main()
