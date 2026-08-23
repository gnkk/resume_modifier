"""
Entry point: runs the resume-building pipeline end to end.

Usage:
    python main.py <path_to_resume.pdf> "<target role/company description>"

Example:
    python main.py data/input/my_resume.pdf "Data Scientist at Shopify, remote Canada"
"""

import sys
import os

from agents.orchestrator import gather_context
from agents.writer import draft_resume
from config import DATA_OUTPUT_DIR


def main():
    if len(sys.argv) != 3:
        print('Usage: python main.py <path_to_resume.pdf> "<target role description>"')
        sys.exit(1)

    resume_path, target_role = sys.argv[1], sys.argv[2]

    print(f"[1/3] Gathering context from resume + job search for: {target_role}")
    context = gather_context(resume_path, target_role)

    print("[2/3] Drafting tailored resume...")
    resume_draft = draft_resume(context)

    print("[3/3] Saving output...")
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(DATA_OUTPUT_DIR, "tailored_resume.md")
    with open(out_path, "w") as f:
        f.write(resume_draft)

    print(f"Done. Tailored resume written to: {out_path}")


if __name__ == "__main__":
    main()
