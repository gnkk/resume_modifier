"""
Entry point: runs the resume-building pipeline end to end.

Usage:
    python main.py <path_to_resume.pdf> "<target role/company description>" [path_to_job_description.txt]

Examples:
    python main.py data/input/my_resume.pdf "Data Scientist at Shopify, remote Canada"
    python main.py data/input/my_resume.pdf "Data Scientist at Shopify" data/input/job_description.txt
"""

import sys
import os
import json

from agents.orchestrator import gather_context
from agents.writer import draft_resume
from agents.reflector import review_resume
from config import DATA_OUTPUT_DIR


def main():
    if len(sys.argv) not in (3, 4):
        print(
            'Usage: python main.py <path_to_resume.pdf> "<target role description>" '
            "[path_to_job_description.txt]"
        )
        sys.exit(1)

    resume_path, target_role = sys.argv[1], sys.argv[2]
    job_description_path = sys.argv[3] if len(sys.argv) == 4 else None

    print(f"[1/4] Gathering context from resume + job info for: {target_role}")
    context = gather_context(resume_path, target_role, job_description_path)

    print("[2/4] Drafting tailored resume...")
    resume_draft = draft_resume(context)

    print("[3/4] Reviewing draft for fit and quality...")
    review = review_resume(resume_draft, context)

    print("[4/4] Saving output...")
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    resume_out_path = os.path.join(DATA_OUTPUT_DIR, "tailored_resume.md")
    with open(resume_out_path, "w") as f:
        f.write(resume_draft)

    review_out_path = os.path.join(DATA_OUTPUT_DIR, "resume_review.json")
    with open(review_out_path, "w") as f:
        json.dump(review, f, indent=2)

    print(f"Done. Tailored resume written to: {resume_out_path}")
    print(f"Review written to: {review_out_path}")

    if review.get("fitness_score") is not None:
        print(f"\nFitness score: {review['fitness_score']}/10")
        print(f"Verdict: {review.get('fitness_summary', '')}")


if __name__ == "__main__":
    main()
