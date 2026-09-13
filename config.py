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
# Context gathering, resume writing, and judging are the steps where output
# quality directly matters (what gets pulled in, what gets written, what
# gets approved), so those stay on Sonnet 5. The judge additionally runs
# with adaptive thinking on (see agents/judge.py) since its evaluations are
# the pipeline's quality gate.
#
# Everything the search side does runs on Haiku, but as separate
# constants rather than one. They were a single MODEL_SEARCH until the
# screening pass and the JD extractor were added, at which point both
# inherited Haiku by default rather than by decision — and the jobs are
# not equivalent. Splitting them made each testable on its own, which is
# how MODEL_PLANNER came to be promoted below.
_HAIKU = "claude-haiku-4-5-20251001"
_SONNET = "claude-sonnet-5"

MODEL_CONTEXT = _SONNET   # reads resume PDF + (later) GitHub context
MODEL_WRITER = _SONNET    # drafts/revises resume content
MODEL_JUDGE = _SONNET     # judges job matches and resume drafts, thinking on

# Plans the search angles. ON SONNET, despite being a small, once-per-run
# call — because it is the narrowest channel in the pipeline and the only
# stage whose mistakes are completely invisible. Its angles determine the
# entire scrape: a posting titled something the planner didn't think of is
# never fetched, so BM25 cannot rank it, the screener cannot rate it, and
# nothing downstream can know it existed. That used to be survivable — the
# selection agent could re-explore and had a one-per-run escape hatch to
# scrape again — but both are gone, so there is now no recovery path at
# all. One call per run makes this the cheapest quality purchase here.
MODEL_PLANNER = _SONNET

# Extracts title/company/location/URL from a supplied job description.
# The most mechanical call in the project — it is told to return null
# rather than infer anything.
MODEL_JD_EXTRACT = _HAIKU

# Reads the candidate's own resume to determine where they may legally
# work. Also pure extraction: it is told to report only what the text
# states and to return an empty list when the resume is silent, which
# disables location filtering rather than guessing a country.
MODEL_ELIGIBILITY = _HAIKU

# Rates every scraped posting 1-5 and drops the rest. THE one worth
# watching: it applies the same market calibration the Sonnet judges use,
# makes hard-blocker calls, and works from ~700-character snippets in
# batches. Its drops are permanent and invisible — a posting cut here is
# gone for the whole run and nothing downstream can know it existed. If
# the pool starts coming back wrong, promote this one to _SONNET first,
# but note it runs across ~150 postings per run, so the cost difference is
# real rather than rounding error.
MODEL_SCREENER = _HAIKU

# Picks the single posting to put in front of the judge, and re-checks the
# full description for hard blockers before committing. Second candidate
# for promotion if picks keep getting rejected for reasons visible in the
# posting text.
MODEL_SELECTOR = _HAIKU

# --- Pipeline stages, each its own search/revise <-> judge loop ---
# Stage 1: search agent <-> judge, over the JOB the search agent picked.
# Stage 2: writer agent <-> judge, over the RESUME drafted for that job.
# Each stage runs independently and sequentially — stage 2 only starts
# once stage 1 has an approved (or cycle-exhausted) job.
MAX_JOB_SEARCH_CYCLES = 3     # hard cap on search-agent <-> judge cycles (job match)
MAX_RESUME_REVISE_CYCLES = 3  # hard cap on writer <-> judge cycles (resume fitness)
JUDGE_APPROVAL_SCORE = 8      # job-match score (1-10) at/above which the judge can approve early
# Resume approval sits one point lower than the job bar on purpose. A resume can
# only ever be as good as the candidate's real background allows, so a strict
# bar just burns revision cycles re-litigating gaps no rewrite can close —
# unlike a job pick, where a better option may genuinely exist in the pool.
RESUME_APPROVAL_SCORE = 7

# Viability floor for the job stage. Distinct from JUDGE_APPROVAL_SCORE on
# purpose — there are three bands, not two:
#   score >= JUDGE_APPROVAL_SCORE : approved, resume written
#   MIN_VIABLE_JOB_SCORE .. below approval : not approved, but a real match
#       worth applying to — the resume is still written, from the
#       best-scoring pick of the run
#   below MIN_VIABLE_JOB_SCORE : nothing found worth applying to. The run
#       STOPS before the writer stage and reports why, rather than
#       producing a polished resume for a job the candidate should not
#       be targeting.
# At 7 against an approval bar of 8, the middle band is a single point
# wide: only a 7/10 pick earns a resume without being approved. Anything
# the judge scores 6 or below ends the run.
# Raise this to be pickier about what earns a resume; lower it to keep the
# pipeline producing output on thin weeks.
MIN_VIABLE_JOB_SCORE = 7

# --- JobSpy (primary job search) ---
# JobSpy scrapes LinkedIn, Indeed, ZipRecruiter, and Glassdoor directly via
# purpose-built scrapers, reaching sites Firecrawl's generic scraping cannot
# (see tools/firecrawl_tool.py BLOCKED_DOMAINS). No API key needed. See
# tools/jobspy_tool.py for the tool wrapper and agents/search_agent.py for
# how it's prioritized over Firecrawl.
# LinkedIn and Indeed only by default. ZipRecruiter and Glassdoor sit behind
# Cloudflare bot protection and reliably return 403 for JobSpy's scrapers, so
# including them by default just costs retries on every search. They remain
# valid values if you want to try them (tools/jobspy_tool.py drops any site
# automatically once it proves blocked during a run).
JOB_SEARCH_SITES = ["linkedin", "indeed"]
# 24 * 14 — postings from the past two weeks, mirrors JOB_SEARCH_TBS below.
# One week was throttling the pool on thin weeks: with the screening pass now
# cutting the scraped set down to the best JOB_POOL_TARGET_SIZE, a wider
# intake window gives it more to choose from. Postings older than this go
# stale fast — many are filled without being taken down — so widening much
# further trades pool size for pool quality.
JOB_SEARCH_HOURS_OLD = 336
# Required by JobSpy for Indeed/Glassdoor searches. Change if you're searching
# outside Canada — see python-jobspy's README for the full supported-country list.
JOBSPY_COUNTRY_INDEED = "Canada"

# --- Job pool (stage 2) ---
# The search agent scrapes ONCE per run into a pool, then picks from that pool
# on every judge cycle instead of re-scraping. Re-running near-identical
# queries was returning near-identical postings, so cycles 2 and 3 were paying
# full scrape cost to rediscover what cycle 1 already had.
# --- Pool construction ---
# Three stages narrow the pool, each doing something the previous cannot:
#
#   scrape  -> JOB_SCRAPE_CEILING postings from JobSpy across all angles
#   BM25    -> lexical rank against the resume, keep BM25_KEEP
#   screen  -> a model rates each survivor 1-10, drops below POOL_SCREEN_MIN_FIT
#
# BM25 is cheap and deterministic but blind to meaning: it cannot see
# seniority, work authorization, or contract-vs-permanent, all of which are
# short common phrases that score near zero lexically. It is therefore only
# a coarse pre-filter ('is this my field at all'), never the decider. The
# screener is what catches the disqualifiers.
JOB_SCRAPE_CEILING = 200

# Postings surviving the BM25 cut. This is the effective pool size — the
# screener drops the duds from it but nothing truncates further.
# BM25 still makes a ~62% cut on purely lexical grounds before the
# screener sees anything, and it cannot see seniority, work
# authorization, or whether the role is a different discipline — so a
# posting it misjudges is gone with no model having read it. 75 leaves
# enough depth that the judge's third cycle still has real choices.
# Raise this before raising the ceiling: it costs one Haiku call per
# extra 25, while a bigger scrape costs LinkedIn requests.
BM25_KEEP = 75

# Screener rating floor, on a 1-10 scale. Everything below is dropped
# outright. Survivors are all kept — there is no target pool size, because
# nothing reads the pool wholesale and truncating a ranked list only costs
# depth on the last judge cycle, which is exactly when depth is needed.
# NOTE: this is a judgement call, not a conversion of the old 3-of-5 floor.
# Watch the drop counts in the first runs.
POOL_SCREEN_MIN_FIT = 5

# Postings per screening call. Small enough that the model reads each
# snippet properly rather than skimming a wall of them.
POOL_SCREEN_BATCH_SIZE = 25

# How much of each posting the screener sees. Raised from 700 because the
# BM25 cut halved the number of postings screened: at 700 chars the model
# saw roughly the opening paragraph, which shows seniority but rarely the
# requirements block where work authorization, licence and employment-type
# blockers actually live. Not the full text — postings run 5-10k chars and
# 25 of those is a wall the model skims rather than reads.
POOL_SCREEN_SNIPPET_CHARS = 2000

# Fetch each LinkedIn posting's full description during the pool build. Costs
# one extra request per LinkedIn result, which was too expensive when every
# judge cycle re-scraped — but the pool is built once per run now, so the cost
# is paid once and every posting in the pool has real requirements text ready
# for the judge and writer. Set False if LinkedIn rate-limits you.
JOBSPY_FETCH_LINKEDIN_DESCRIPTIONS = True

# One targeted re-search is allowed per run, and only when the selecting agent
# states which judge concern the existing pool cannot satisfy (e.g. the judge
# wants remote and the pool is entirely on-site). Without this, a pool that was
# wrong from the start guarantees two more bad picks; ungated, it degenerates
# back into re-searching every cycle.
# NO LONGER USED. This was the selection agent's escape hatch: one extra
# scrape per run, for when the judge raised a concern nothing in the pool
# could satisfy. The selection agent is gone (code now walks the screened
# ranking), so nothing can invoke it. Kept here only as a marker of what
# was removed — if a pool that is wrong from the start turns out to be a
# real problem in practice, this is the capability to restore.
MAX_EXTRA_SEARCHES = 1

# --- Firecrawl (support tool: fallback web search + single-URL scraping) ---
# Firecrawl is no longer the primary search mechanism (JobSpy is) — it now
# only backs up JobSpy: widening a search beyond the four boards above (e.g.
# a company careers page), or scraping a specific URL when JobSpy's own
# `description` field is thin/missing. See agents/search_agent.py.
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev"
# Restrict Firecrawl's fallback search to the same window as JobSpy above.
# Google's tbs syntax: qdr:w2 = past two weeks. Keep these two in step —
# a mismatch means the fallback quietly returns older postings than the
# primary search, and nothing downstream would flag it.
JOB_SEARCH_TBS = "qdr:w2"

# --- Resume format (writer.py) ---
# Hard cap and preferred target for the tailored resume's rendered PDF
# length. See agents/writer.py's SYSTEM_PROMPT and pdf_renderer.py's @page
# size, which together make this a real, enforceable constraint rather than
# a loose suggestion.
RESUME_MAX_PAGES = 3
RESUME_PREFERRED_PAGES = 2

# --- Paths ---
DATA_INPUT_DIR = "data/input"
DATA_OUTPUT_DIR = "data/output"
DATA_LOGS_DIR = os.path.join(DATA_OUTPUT_DIR, "logs")

# Running ledger of the job each run finally settled on, carried ACROSS runs
# (unlike everything else in data/output/, which is per-run and timestamped).
# Postings already selected in an earlier run are filtered out of the pool
# before screening, so consecutive runs don't keep landing on the same
# posting and producing the same resume. See job_history.py.
SELECTED_JOBS_HISTORY_PATH = os.path.join(DATA_OUTPUT_DIR, "selected_jobs.json")

# Set False to disable that filtering — e.g. to deliberately re-run against a
# job you already targeted, to regenerate its resume after changing prompts.
SKIP_PREVIOUSLY_SELECTED = True

# Optional sample/template resume used purely as a FORMATTING model —
# section order, heading style, level of detail, tone — never as a
# source of candidate content. If a file exists at this path, the
# context agent reads it and passes it to the writer labeled explicitly
# as a style reference, not the candidate's own background. Entirely
# optional: if the file isn't present, the pipeline behaves exactly as
# before.
SAMPLE_RESUME_PATH = os.path.join(DATA_INPUT_DIR, "sample_resume.pdf")
