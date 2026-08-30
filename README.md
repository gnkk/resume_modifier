# Resume Building Agent

An agentic pipeline that finds the best-matching job posted in the last
week, tailors your resume to it, and critically reviews its own work
across multiple revision cycles before you see the final output:

1. **Context agent** (Claude Sonnet 5) — reads your existing resume
   (PDF), and a job description file if you provide one. GitHub project
   context is wired in but **off by default** (see "GitHub integration"
   below) — this agent does **not** search the web for jobs; that's step 2.
2. **Search agent** (Claude Sonnet 5) — uses **Firecrawl's real-time
   web search** (with its `tbs` time filter) to find candidate job
   postings, restricted to postings **newer than one week**, with full
   page content scraped inline. Picks the single best-matching job with
   a realistic chance of a positive response — not just the first
   plausible listing.
3. **Writer agent** (Claude Sonnet 5) — drafts a tailored resume from
   the gathered context and the chosen job, flagging any experience
   gaps rather than inventing content.
4. **Judge agent** (Claude Sonnet 5, adaptive thinking, high effort)
   — critically reviews the draft against the job: a fitness score,
   strengths, gaps, concrete suggestions, and an explicit
   approve/revise decision. Written to push back on weak output
   rather than rubber-stamp it.

Steps 3 and 4 run in a loop: the judge's comments and the previous
draft go back to the writer for a rewrite, for up to **5 cycles**, or
until the judge approves early (fitness score clears the threshold in
`config.py`, or the judge decides no further meaningful improvement is
available).

The final approved resume is rendered as both a scrollable **HTML**
page and a matching **PDF** — same content, two formats, one source of
truth.

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
   the final PDF export step will fail, and the HTML/Markdown outputs
   are unaffected.

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
- `tailored_resume.html` — the same resume as a scrollable HTML page
- `tailored_resume.pdf` — the same resume as a PDF
- `resume_review.json` — the chosen job, the judge's final verdict, and
  the full cycle-by-cycle review history

## GitHub integration (deferred)

The context agent already has GitHub MCP wiring in `agents/context_agent.py`,
but it stays off until you set `GITHUB_PAT` in `.env` — no code changes
needed when you're ready:

1. Create a fine-grained GitHub Personal Access Token scoped to
   `Contents: Read` (and `Read/Write` if you later want the agent to
   push commits/PRs) on the repos you want it to look at.
2. Set `GITHUB_PAT=<token>` in `.env`.
3. Run the pipeline as usual — the context agent will automatically
   pull in GitHub project details (READMEs, languages, notable repos)
   to supplement the resume content.

Without a `GITHUB_PAT` set, the context agent runs on the resume PDF
(+ optional job description file) alone.

## Project structure

```
resume-agent/
├── agents/
│   ├── context_agent.py   # tool-calling loop: resume PDF read (+ optional GitHub MCP)
│   ├── search_agent.py    # Firecrawl real-time search (tbs-filtered) -> single best job
│   ├── writer.py           # drafts + revises resume content
│   └── judge.py             # critiques the draft using adaptive thinking, approves or sends back
├── tools/
│   ├── pdf_reader.py     # PDF text extraction
│   ├── text_reader.py    # plain .txt file reading (job descriptions)
│   └── firecrawl_tool.py # real-time web search (tbs recency filter) + page scraping
├── html_renderer.py       # Markdown -> scrollable HTML page
├── pdf_renderer.py         # same HTML -> PDF
├── data/
│   ├── input/            # put source resumes + job description files here
│   └── output/           # generated resume (md/html/pdf) + review land here
├── config.py              # env vars + model selection + loop settings
├── main.py                 # entry point: runs the full pipeline
└── requirements.txt
```

> **Note:** `agents/orchestrator.py`, `agents/reflector.py`, and
> `tools/tavily_tool.py` are deprecated leftovers from earlier
> iterations of this architecture. They now just raise `ImportError`
> with a pointer to their replacement — delete them:
> ```bash
> rm agents/orchestrator.py agents/reflector.py tools/tavily_tool.py
> ```

## On "reasoning model"

Claude doesn't have a separate reasoning-model line the way some providers
do. Instead, current models like Sonnet 5 support **adaptive thinking** — a
mode where Claude reasons through a hidden scratchpad before answering,
controlled by an `effort` level (`standard`, `high`, `xhigh`, `max`) rather
than a manual token budget. The judge agent uses
`thinking={"type": "adaptive"}` with `output_config={"effort": "high"}` for
exactly this reason: judging resume quality and job fit is an evaluative,
multi-factor call, not a generation task, and benefits from that extra
reasoning step.

## Tuning the revise loop

In `config.py`:
- `MAX_REVISE_CYCLES` (default 5) — hard cap on writer↔judge cycles.
- `JUDGE_APPROVAL_SCORE` (default 8) — fitness score (1–10) at/above which
  the judge can approve early instead of running all 5 cycles. The judge
  can still withhold approval above this score, or approve slightly below
  it, if it reasons that's the right call — see `agents/judge.py`'s system
  prompt.

## Tuning the job search recency window

In `config.py`, `JOB_SEARCH_TBS` (default `"qdr:w"`, past week) controls
how recent a posting must be. Firecrawl's `tbs` parameter also accepts
`"qdr:h"` (past hour), `"qdr:d"` (past day), `"qdr:m"` (past month),
`"qdr:y"` (past year), or a custom range like
`"cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY"`.

## Cost notes

- All four agents use `claude-sonnet-5` ($2/$10 per MTok in/out) by
  default. The judge's adaptive thinking adds reasoning tokens (billed
  as output) on top of the base cost, and the revise loop means the
  writer + judge cost is multiplied by however many cycles actually run.
- A typical run (context read + job search + 2-3 revise cycles) costs
  roughly $0.40–$1.20 in API usage, depending on how many cycles the
  judge requires.
- Swap any `MODEL_*` constant in `config.py` to `claude-opus-5` for a
  specific agent if you want more reasoning headroom there, or to
  `claude-haiku-4-5-20251001` to cut cost during testing. Keep
  `MODEL_JUDGE` on Sonnet 5 or better — the whole point of this agent
  is a careful, honest judgment call, which is exactly where a weaker
  model costs you the most.

## Next steps (not yet built)

- GitHub integration is wired but deferred — see "GitHub integration" above.
- Pushing the finished project / output to GitHub (open a PR with the
  tailored resume, or commit output artifacts) — planned for later.
- Support for multiple target roles in one run.
- Caching Firecrawl search/scrape results across revise cycles so a
  cycle doesn't need to re-search if the job hasn't changed.

## Known limitations / before you rely on this

Read this before running against a real job search — none of the items
below are done yet.

**Rotate your API keys.** During this rework, `.env` was read as part of
making these changes, exposing the Anthropic key, Firecrawl key, and
GitHub PAT that were live at the time. Treat all three as compromised and
regenerate them:
- Anthropic: console.anthropic.com → API Keys
- Firecrawl: your Firecrawl dashboard
- GitHub PAT: github.com/settings/tokens → revoke and reissue with the
  same scopes

**Delete deprecated files manually.** These were stubbed to raise
`ImportError` rather than removed, since file deletion wasn't available
during the rework:
```bash
rm agents/orchestrator.py agents/reflector.py tools/tavily_tool.py
```

**No live end-to-end run has been done.** Everything has been verified
by syntax-checking every file, import-checking the real module graph
with dummy env vars (including both the GitHub-off and GitHub-on code
paths), and unit-testing the control flow (writer↔judge loop early-exit
and max-cycle-exhaustion paths) and the Firecrawl `search()`/`scrape()`
parsing against mocked responses shaped like Firecrawl's real v2 API.
None of this used a real Anthropic or Firecrawl API call. Things that
are only "should be right" until a real run confirms them:
- The exact `thinking`/`output_config` parameter shapes the Anthropic
  SDK expects for adaptive thinking in `agents/judge.py`.
- The MCP beta header string (`mcp-client-2025-11-20`) and
  `mcp_toolset`/`mcp_servers` wiring in `agents/context_agent.py`,
  once you turn GitHub on.
- Whether WeasyPrint's system dependency (`pango`, etc.) is already
  satisfied on your machine — untested locally.

**The stray nested folder** `resume-agent/resume-agent/` was left
untouched and never opened to confirm what's in it. It looks like a
duplicate scaffold, but that was an assumption, not a verified fact.

The suggested next step is a real run — `python main.py
data/input/<resume>.pdf "<role>"` — checking each stage's output as it
goes, rather than more static review.
