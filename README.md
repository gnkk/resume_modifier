# Resume Building Agent

An agentic pipeline that finds the best-matching job posted in the last
week, tailors your resume to it, and critically reviews its own work at
**two separate, sequential stages** before you see the final output:

1. **Context agent** (Claude Sonnet 5) — reads your existing resume
   (PDF), and a job description file if you provide one. GitHub project
   context is wired in but **off by default** (see "GitHub integration"
   below) — this agent does **not** search the web for jobs; that's step 2.
2. **Search agent <-> Judge** (Claude Sonnet 5, adaptive thinking, high
   effort) — the search agent uses **Firecrawl's real-time web search**
   (with its `tbs` time filter) to find the single best-matching job
   posting, restricted to postings **newer than one week**, with full
   page content scraped inline. The judge then evaluates ONLY that job
   pick: is it genuinely the best match, does the posting look live and
   current, does the application link look real and working. If the
   judge doesn't approve, the search agent tries again — with the
   rejected pick(s) and the judge's specific concerns — and must return
   a **different** posting. This runs for up to **3 cycles**, or until
   the judge approves early, whichever comes first. **No resume is
   written until this stage finishes.**
3. **Writer agent <-> Judge** (Claude Sonnet 5) — only once a job is
   locked in from step 2, the writer drafts a tailored resume from the
   gathered context and that job, flagging any experience gaps rather
   than inventing content. The judge then evaluates ONLY the resume:
   fitness score, strengths, gaps, concrete suggestions. If not
   approved, the writer revises against that feedback. This also runs
   for up to **3 cycles**, or until the judge approves early, whichever
   comes first.

These two loops are independent and run one after the other — the job
is fully settled (approved, or cycles exhausted) before the writer ever
starts drafting, since there's no point tailoring a resume to a job
that might still change.

The final resume is rendered as **Markdown** and a matching **PDF** —
same content, two formats. The judge's final remarks from both stages
(job match score + notes, resume fitness score + strengths/gaps/
suggestions, and the job application link) are rendered separately as a
standalone **HTML** review page.

## Setup

1. **Clone and enter the project**
   ```bash
   cd resume-agent
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   PDF rendering uses [WeasyPrint](https://weasyprint.org/), which needs
   a few system libraries for text/font layout (this is a one-time
   environment setup, not a per-run dependency):
   ```bash
   # macOS
   brew install pango

   # Ubuntu/Debian
   sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0
   ```
   If you skip this, everything else in the pipeline still works — only
   the final PDF export step will fail, and the Markdown output is
   unaffected.

4. **Set up API keys**

   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and fill in:
   - `ANTHROPIC_API_KEY` — from [platform.claude.com](https://platform.claude.com) (API Keys section)
   - `FIRECRAWL_API_KEY` — from your Firecrawl dashboard
   - `GITHUB_PAT` — leave blank for now (see "GitHub integration" below)

   > **Security note:** never commit `.env` (it's already in
   > `.gitignore`) and never paste real key values into chat, issues,
   > or commit messages. If a key is ever exposed, rotate it
   > immediately at the provider.

5. **Add your resume**
   Place your current resume PDF in `data/input/`.

6. **(Optional) Add a job description file**
   If you already know exactly which job you want, save the posting
   text as a `.txt` file in `data/input/` — the search agent will use
   it as a strong signal and mainly confirm the posting is still live,
   rather than searching from scratch.

## Usage

Without a job description file (search agent finds the best match):
```bash
python main.py data/input/my_resume.pdf "Data Scientist, remote Canada"
```

With a job description file (skips most of the search):
```bash
python main.py data/input/my_resume.pdf "Data Scientist at Shopify" data/input/job_description.txt
```

Output (in `data/output/`):
- `tailored_resume.md` — the final tailored resume (Markdown)
- `tailored_resume.pdf` — the same resume as a PDF
- `resume_review.html` — the judge's final remarks from both stages:
  job match score + notes, resume fitness score +
  strengths/gaps/suggestions, job posting trust notes, and a working
  link to the job application
- `resume_review.json` — the chosen job, both stages' final verdicts,
  and the full cycle-by-cycle history for each stage



## Project structure

```
resume-agent/
├── agents/
│   ├── context_agent.py   # tool-calling loop: resume PDF read (+ optional GitHub MCP)
│   ├── search_agent.py    # Firecrawl real-time search (tbs-filtered) -> single best job
│   ├── writer.py           # drafts + revises resume content
│   └── judge.py             # review_job() for stage 1, review_resume() for stage 2
├── tools/
│   ├── pdf_reader.py     # PDF text extraction
│   ├── text_reader.py    # plain .txt file reading (job descriptions)
│   └── firecrawl_tool.py # real-time web search (tbs recency filter) + page scraping
├── html_renderer.py       # renders both judge verdicts as a standalone HTML review page
├── pdf_renderer.py         # renders the tailored resume Markdown to PDF
├── data/
│   ├── input/            # put source resumes + job description files here
│   └── output/            # generated resume (md/pdf) + review (html/json) land here
├── config.py              # env vars + model selection + loop settings
├── main.py                 # entry point: runs the full two-stage pipeline
└── requirements.txt
```

## On "reasoning model"

Claude doesn't have a separate reasoning-model line the way some providers
do. Instead, current models like Sonnet 5 support **adaptive thinking** — a
mode where Claude reasons through a hidden scratchpad before answering,
controlled by an `effort` level (`standard`, `high`, `xhigh`, `max`) rather
than a manual token budget. The judge agent uses
`thinking={"type": "adaptive"}` with `output_config={"effort": "high"}` for
both `review_job()` and `review_resume()`, for the same reason: judging job
fit and judging resume quality are both evaluative, multi-factor calls, not
generation tasks, and benefit from that extra reasoning step.

## Tuning the two revise loops

In `config.py`:
- `MAX_JOB_SEARCH_CYCLES` (default 3) — hard cap on search-agent↔judge
  cycles for the job-match stage. This runs to completion (approval, or
  cycles exhausted) before the resume stage starts.
- `MAX_RESUME_REVISE_CYCLES` (default 3) — hard cap on writer↔judge
  cycles for the resume stage, which only starts once the job stage
  is done.
- `JUDGE_APPROVAL_SCORE` (default 8) — score (1–10) at/above which the
  judge can approve early instead of running all available cycles.
  Applies to both the job match score and the resume fitness score —
  the judge can still withhold approval above this score, or approve
  slightly below it, if it reasons that's the right call. See
  `agents/judge.py`'s two system prompts (`JOB_SYSTEM_PROMPT` and
  `RESUME_SYSTEM_PROMPT`).

## Tuning the job search recency window

In `config.py`, `JOB_SEARCH_TBS` (default `"qdr:w"`, past week) controls
how recent a posting must be. Firecrawl's `tbs` parameter also accepts
`"qdr:h"` (past hour), `"qdr:d"` (past day), `"qdr:m"` (past month),
`"qdr:y"` (past year), or a custom range like
`"cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY"`.

## Cost notes

- All four agents use `claude-sonnet-5` ($2/$10 per MTok in/out) by
  default. The judge's adaptive thinking adds reasoning tokens (billed
  as output) on top of the base cost, at both stages. The two revise
  loops mean the total cost is multiplied by however many cycles each
  stage actually runs (up to 3 + 3 = 6 judge calls in the worst case,
  plus the corresponding search/writer calls).
- A typical run (context read + 1-2 job-search cycles + 1-2 resume-revise
  cycles) costs roughly $0.40–$1.20 in API usage, depending on how many
  cycles each stage requires.
- Swap any `MODEL_*` constant in `config.py` to `claude-opus-5` for a
  specific agent if you want more reasoning headroom there, or to
  `claude-haiku-4-5-20251001` to cut cost during testing. Keep
  `MODEL_JUDGE` on Sonnet 5 or better — the whole point of this agent
  is a careful, honest judgment call, which is exactly where a weaker
  model costs you the most.

