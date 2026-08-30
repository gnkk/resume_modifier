"""
Central config for the resume-building agent project.

Loads secrets from .env and defines which Claude model each
component uses, so you can tune cost vs. quality in one place
instead of hunting through every file.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
GITHUB_PAT = os.environ.get("GITHUB_PAT")  # not used yet — GitHub integration is a later step

if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. Copy .env.example to .env "
        "and fill in your real key."
    )
if not FIRECRAWL_API_KEY:
    raise RuntimeError(
        "FIRECRAWL_API_KEY not found. Copy .env.example to .env "
        "and fill in your real key."
    )

# --- Model selection per agent/task ---
# Every agent that makes judgment calls (what to search, what to write,
# how to judge) runs on a top-tier model per project requirements.
# Sonnet 5 is the default; swap to "claude-opus-5" per-agent below if
# you want the extra reasoning headroom for a specific step.
MODEL_CONTEXT = "claude-sonnet-5"     # reads resume PDF + (later) GitHub context
MODEL_SEARCH = "claude-sonnet-5"      # finds the single best-matching job via Firecrawl
MODEL_WRITER = "claude-sonnet-5"      # drafts/revises resume content
MODEL_JUDGE = "claude-sonnet-5"       # judges job matches and resume drafts with adaptive thinking

# --- Pipeline stages, each its own search/revise <-> judge loop ---
# Stage 1: search agent <-> judge, over the JOB the search agent picked.
# Stage 2: writer agent <-> judge, over the RESUME drafted for that job.
# Each stage runs independently and sequentially — stage 2 only starts
# once stage 1 has an approved (or cycle-exhausted) job.
MAX_JOB_SEARCH_CYCLES = 3     # hard cap on search-agent <-> judge cycles (job match)
MAX_RESUME_REVISE_CYCLES = 3  # hard cap on writer <-> judge cycles (resume fitness)
JUDGE_APPROVAL_SCORE = 8      # score (1-10) at/above which the judge can approve early,
                               # used for both the job-match score and the resume fitness score

# --- Firecrawl ---
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev"
JOB_SEARCH_TBS = "qdr:w"  # restrict job search to postings newer than this (past week)

# --- Paths ---
DATA_INPUT_DIR = "data/input"
DATA_OUTPUT_DIR = "data/output"
