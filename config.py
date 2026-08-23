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
# Reasoning/orchestration and final writing quality matter most here,
# so those get Sonnet 5. Cheap mechanical filtering can use Haiku 4.5.
MODEL_ORCHESTRATOR = "claude-sonnet-5"   # decides what to search/scrape/read next
MODEL_WRITER = "claude-sonnet-5"         # drafts/rewrites resume content
MODEL_FILTER = "claude-haiku-4-5-20251001"  # cheap pre-filtering of scraped text

# --- Firecrawl ---
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev"

# --- Paths ---
DATA_INPUT_DIR = "data/input"
DATA_OUTPUT_DIR = "data/output"
