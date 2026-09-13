# Resume Building Agent

An agentic pipeline that tailors your resume to a specific job and
critically reviews its own work before you see the output. It runs in
**two modes**, sharing the same entry point:

- **Search mode** — finds a well-matched job posted in the last two
  weeks, vets it, then writes and reviews a resume for it.
- **Supplied-job mode** — you give it the posting (a URL or a saved
  description) and it goes straight to writing and reviewing.

> **Status: not yet validated end to end.** The pipeline is complete and
> every module has been syntax- and import-checked, but no full live run
> against the Anthropic API has been performed. Treat the first run as a
> test, and read the run log rather than just the output.

## How it works

### Stage 1 — getting a job

**Search mode** (no third CLI argument):

1. A **planner** (Haiku) designs 2-6 complementary search angles from
   your background — varying job title, seniority and location, since
   employers title the same work differently.
2. Those angles run against **JobSpy**, which scrapes LinkedIn and
   Indeed directly. Scraping deliberately **over-collects** (up to
   `JOB_SCRAPE_CEILING`, ~200 postings) so every planned angle actually
   runs.
3. Postings already targeted in earlier runs are **filtered out** (see
   "Duplicate filtering" below).
4. **BM25** ranks what remains against your resume and keeps the top
   `BM25_KEEP` (~75). Pure lexical arithmetic — no model, no embeddings.
   It answers "is this even my field?" cheaply and nothing more: it is
   structurally blind to seniority, work authorization and
   contract-vs-permanent, which is why it only ever pre-filters.
5. A **screening pass** (Haiku) rates every survivor 1-10 and drops
   anything below `POOL_SCREEN_MIN_FIT`. This is where the disqualifiers
   get caught — a staff-level posting is a near-perfect lexical match to
   a mid-level candidate, so BM25 ranks it first and the screener rates
   it a 2. Each posting is rated independently, never against the others
   in its batch, so ratings stay comparable across batches and cycles.
   It reads a `POOL_SCREEN_SNIPPET_CHARS` slice (2,000 chars) of each
   posting — enough to reach past the opening summary into the
   requirements block on most postings, though not the whole text. The
   judge reads the full description before anything is decided.
6. Survivors are sorted by **rating first, BM25 rank as tie-break**, and
   all of them are kept. There is no target pool size: nothing reads the
   pool wholesale, so truncating a ranked list only costs depth on the
   last cycle — exactly when it's needed.
7. **Code** takes the best remaining posting, attaches its full
   description, and hands it to the **judge** (Sonnet 5, adaptive
   thinking at high effort), which scores it 1-10 against an anchored
   rubric and approves only at `JUDGE_APPROVAL_SCORE` or above with a
   working-looking application link. The bar is enforced in code.
8. On rejection, the next posting down is tried. Up to
   `MAX_JOB_SEARCH_CYCLES` cycles, or until the pool runs out; if none is
   approved, the **best-scoring** pick is kept.
9. If the best pick scores below `MIN_VIABLE_JOB_SCORE`, or nothing
   survived filtering, **the run stops here** and writes a report
   instead of a resume (see "When a run stops early").

> **Why there is no selection agent.** Step 7 used to be a tool-use
> agent that queried the pool, read a few postings and picked one. Given
> a pool that is already rated and ranked, it was re-deriving — three
> times per run, with no memory between cycles — an ordering the screener
> had already computed once. Worse, its two most important behaviours
> ("read the full description before committing", "never re-recommend a
> rejected posting") were prompt instructions a model could silently
> skip; both are now code. What was lost: it could respond to a judge
> concern by jumping to a different *kind* of posting rather than the
> next one down, and it had a one-per-run escape hatch to scrape again
> when the pool genuinely couldn't satisfy a concern. If rejections turn
> out to cluster on one reason, the fix is a reason-driven skip in
> `next_candidate()`, not restoring the agent.

**Supplied-job mode** (third argument given): none of the above runs.
The posting is fetched or read, parsed into the same shape, and the
judge scores it **for information only** — the score gates nothing,
because you have already decided to apply. A weak match is reported
loudly but never stops the run.

### Stage 2 — writing the resume

Identical in both modes. The **writer** (Sonnet 5) drafts a tailored,
ATS-ready resume, then the **judge** (Sonnet 5) reviews it and the
writer revises against that feedback, up to `MAX_RESUME_REVISE_CYCLES`
cycles.

Two things make this loop converge rather than spin:

- The judge **scores the draft, not the candidate**. It asks how
  completely the draft surfaces what your background can honestly show
  for this job — not how well you match the posting. That was settled
  in stage 1 and no rewrite can change it. A draft that extracts
  everything available scores highly even if you only cover part of
  what the posting asks for.
- Gaps the judge raised at **stage 1 are passed forward as settled**.
  Both the writer and the resume judge are told not to try to close
  them. A gap cannot be written away, only faked.

The loop also stops early when a revision fails to improve on the
previous cycle's score — that pattern means the remaining objections
aren't fixable by rewriting.

## Setup

1. **Create an environment**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   # or: conda create -n gk_agentic python=3.11 && conda activate gk_agentic
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   PDF rendering uses [WeasyPrint](https://weasyprint.org/), which needs
   system libraries for text/font layout (one-time setup):
   ```bash
   # macOS
   brew install pango

   # Ubuntu/Debian
   sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0
   ```
   Skipping this costs you the PDF only — the Markdown resume is
   unaffected, and the failure is reported clearly.

   > **macOS + conda:** conda environments don't always inherit
   > Homebrew's library search path. If WeasyPrint can't find
   > `libgobject`/`libpango` after `brew install pango`:
   > ```bash
   > export DYLD_LIBRARY_PATH="/opt/homebrew/opt/glib/lib:/opt/homebrew/opt/pango/lib:/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
   > ```
   > Add it to `$CONDA_PREFIX/etc/conda/activate.d/` to set it on
   > `conda activate`.

3. **Set up API keys**
   ```bash
   cp .env.example .env
   ```
   - `ANTHROPIC_API_KEY` — from [platform.claude.com](https://platform.claude.com)
   - `FIRECRAWL_API_KEY` — used for fetching supplied non-LinkedIn
     posting URLs and as search fallback
   - `GITHUB_PAT` — leave blank (see "GitHub integration")

   JobSpy needs **no API key**. If you see repeated empty results or
   429s, LinkedIn is rate-limiting — lower `JOB_POOL_OVERSCAN_FACTOR`
   or add proxies (see `tools/jobspy_tool.py`).

   > **Never commit `.env`.** It's in `.gitignore`. If a key is ever
   > exposed, rotate it immediately at the provider.

4. **Add your resume** — any PDF in `data/input/`.

5. **(Optional) Add a sample resume** at `data/input/sample_resume.pdf`
   as a layout template (see "Sample resume").

## Usage

**Search mode** — the role description steers every search angle, so
be specific:

```bash
python main.py data/input/my_resume.pdf "Machine Learning Engineer, NLP and LLMs, Canada"
```

**Supplied-job mode** — pass a posting URL or a saved `.txt`. The
second argument is ignored here but still required positionally, so
pass `"-"`:

```bash
# LinkedIn (fetched via JobSpy)
python main.py data/input/my_resume.pdf "-" https://www.linkedin.com/jobs/view/4285719301/

# Employer or ATS page (fetched via Firecrawl)
python main.py data/input/my_resume.pdf "-" https://boards.greenhouse.io/acme/jobs/1234

# Saved description — always works, no fetch involved
python main.py data/input/my_resume.pdf "-" data/input/job_description.txt
```

### Supplying a posting by URL

Two routes, because no single tool covers both:

- **LinkedIn** goes through JobSpy, which has purpose-built LinkedIn
  scraping. Firecrawl is refused on `linkedin.com` outright. This uses
  a **private** python-jobspy method (`LinkedIn._get_job_details`), so a
  package upgrade could break it — the failure degrades to a clear
  message rather than a crash. LinkedIn also serves its signup wall to
  guest traffic unpredictably, so this fails some of the time even on a
  live posting; retrying often works.
- **Everything else** goes through Firecrawl. Server-rendered ATS pages
  (Greenhouse, Lever) fetch cleanly; heavily JavaScript-driven pages may
  not.
- **Glassdoor and Facebook** have no working route. Paste the text into
  a file instead.

Any fetch failure **halts the run and writes a report** — a resume
tailored against a cookie wall would look exactly like a successful run.

## Output

Everything lands in `data/output/`, timestamped per run:

- `tailored_resume_<ts>.md` — the resume (Markdown)
- `tailored_resume_<ts>.pdf` — the same resume, styled for print
- `resume_review_<ts>.html` — job match assessment, resume fitness
  score, strengths, gaps, suggestions, the application link, and the
  writer's notes
- `resume_review_<ts>.json` — full cycle-by-cycle history
- `job_pool_<ts>.json` — the screened pool (search mode only)
- `selected_jobs.json` — running cross-run ledger (see below)
- `logs/run_<ts>.log` — full run log

**The PDF carries resume content and nothing else** — no notes, no page
numbers, no commentary. The writer is instructed to end its draft with a
"Notes (not part of the resume)" section flagging real gaps between you
and the posting; that section is stripped before the PDF is rendered and
displayed on the review page instead. It's interview preparation, not
something to send an employer. The resume judge is told the section is
expected and auto-stripped, so it doesn't read a deliberate handoff as
contamination and reject an otherwise fine draft over it.

### When a run stops early

- `no_job_found_<ts>.html` — search mode found nothing worth applying
  to. Reports what was searched, every posting considered with the
  judge's verdict, and which config levers to turn. **Exits 0** — a
  thin fortnight is a legitimate outcome, not a failure.
- `job_unavailable_<ts>.html` — a supplied posting couldn't be fetched
  or read. Carries the fetch error verbatim including its workaround.
  **Exits 1** — this needs you to act.

## Duplicate filtering

Each run's final job pick is appended to `data/output/selected_jobs.json`.
On later runs those postings are filtered out of the pool before
screening, so consecutive runs don't keep landing on the same job.

Matching is on normalized URL **and** on a company+title fingerprint,
since the same opening is routinely posted to several boards under
different URLs. Only the final pick is recorded — picks the judge
rejected mid-run stay available, since they were often rejected only
because something better sat beside them that week.

Set `SKIP_PREVIOUSLY_SELECTED = False` to disable, e.g. to regenerate a
resume for a job you already targeted. Note `data/output/` is
gitignored, so the ledger is local-only.

## Resume appearance

All visual formatting lives in `pdf_renderer.py`, **not** in any
prompt. The writer emits Markdown, which carries structure but no
typography — it never sees the rendered page, so asking it to "fix the
spacing" cannot work.

Two layers do the work:

- **`STYLE`** — the design tokens: font stacks, sizes, margins, accent
  colour, header alignment. Change the look by editing that dict.
- **A Markdown-shaping layer** that gives the CSS something to hook
  onto: it tags the contact block and skills rows, lifts trailing dates
  onto the title line flush right, and normalizes Education entries to
  the same shape as Experience.

Page count is measured, not estimated: the renderer lays the document
out before writing it and logs the real count, warning when it exceeds
`RESUME_MAX_PAGES` (default 3) or `RESUME_PREFERRED_PAGES` (default 2).

## Sample resume (layout template)

Drop a template PDF at `data/input/sample_resume.pdf` and its extracted
text is passed to the writer **verbatim** as the layout spec — section
headers and their exact wording and order, contact-block layout, entry
shape, date format, bullet density, Skills organization, overall
length. The judge receives the same text and flags structural
deviations, so fidelity holds across revision cycles.

It is never a source of factual content — the writer takes no name,
employer, date, or claim from it. Where the template conflicts with an
ATS rule (e.g. multi-column layout), the ATS rule wins and the
template's section order and emphasis are reproduced in a single
column. Entirely optional.

## Market calibration

`agents/market_context.py` holds one block of guidance about how to
read a posting, imported by the screener, the selector and both judges
so all four apply the **same** standard. When they disagreed, the loops
burned cycles — the search agent would hand up a pick it rated highly
and the judge would reject it on a requirement the search agent never
weighted.

It splits requirements into:

- **Hard blockers** — work authorization, a required licence, a stated
  degree bar, strictly-onsite in the wrong place, a genuinely different
  discipline, seniority two-plus bands off. These disqualify, and cap
  the job score at 3.
- **Soft items** — years-of-experience thresholds (within ~2 years is a
  match), anything under "nice to have", a named tool where you have a
  direct equivalent, domain familiarity, and ubiquitous boilerplate
  (CI/CD, Docker, agile). These never disqualify anything.

Edit that one file to recalibrate as the market shifts.

## Logging

- **Console** — step-by-step run narration at INFO.
- **`data/output/logs/run_<ts>.log`** — the same narration **plus**
  exceptions and tool/parse failures with tracebacks, timestamped and
  tagged by module. A full record of the run, not just its failures, so
  two runs can be diffed when tuning prompts.

Several places log a detailed `error` for the record and then a
plain-language `info` warning for whoever's watching. Both land in the
file, so a handled failure appears twice. That's redundancy, not a bug.

Every stage also logs the **resolved** model that served it, once per
run:

```
[planner ran on claude-sonnet-5-...]
[screener ran on claude-haiku-4-5-20251001]
```

Model strings in `config.py` are aliases, so this is what makes
run-to-run differences attributable: when two runs of identical code
produce different angles or ratings, the log says whether they ran on the
same thing, rather than leaving "the model changed underneath me" as an
untestable explanation sitting next to "my prompt edit worked". It's the
reason not to pin model strings — pinning buys the same attribution but
makes deprecation tracking your problem.

## Model routing

In `config.py`. Sonnet 5 where a mistake is unrecoverable; Haiku where
the work is mechanical or a wrong call gets caught downstream.

| Constant | Model | Job |
|---|---|---|
| `MODEL_CONTEXT` | Sonnet 5 | Reads the resume PDF. Everything downstream sees only its summary, so anything it drops is unrecoverable. |
| `MODEL_WRITER` | Sonnet 5 | Drafts and revises the resume. |
| `MODEL_JUDGE` | Sonnet 5 | Both judges, with adaptive thinking at high effort. |
| `MODEL_PLANNER` | Sonnet 5 | Designs the search angles. The narrowest channel in the pipeline — a posting titled something it didn't think of is never scraped, and with the selection agent gone there is no recovery path. One call per run. |
| `MODEL_SCREENER` | Haiku 4.5 | Rates and drops postings. **Watch this one** — its drops are permanent and invisible. |
| `MODEL_JD_EXTRACT` | Haiku 4.5 | Pulls metadata from a supplied description. |

If the pool starts coming back wrong, promote `MODEL_SCREENER` — but note
it runs across ~75 postings per run, so the cost difference is real.

Nothing escalates automatically. A run that screens badly produces a
thin pool and says so in the log; promotion is a deliberate edit.

## Configuration reference

All in `config.py`.

**Loops and bars**
- `MAX_JOB_SEARCH_CYCLES` (3), `MAX_RESUME_REVISE_CYCLES` (3)
- `JUDGE_APPROVAL_SCORE` (8) — job-pick approval bar, enforced in code
- `RESUME_APPROVAL_SCORE` (7) — resume approval bar
- `MIN_VIABLE_JOB_SCORE` (7) — below this, the search run stops without
  writing a resume. With the approval bar at 8, the middle band is one
  point wide: only a 7/10 pick earns a resume without being approved.

**Search and pool**
- `JOB_SEARCH_HOURS_OLD` (336, two weeks) and `JOB_SEARCH_TBS`
  (`"qdr:w2"`) — keep these two in sync
- `JOB_SEARCH_SITES` (`["linkedin", "indeed"]`) — ZipRecruiter and
  Glassdoor are valid but off by default; their Cloudflare protection
  reliably 403s JobSpy. A circuit breaker drops any site that proves
  blocked for the rest of the run.
- `JOBSPY_COUNTRY_INDEED` (`"Canada"`)
- `JOB_SCRAPE_CEILING` (200) — postings scraped before ranking
- `BM25_KEEP` (75) — survivors of the lexical cut
- `POOL_SCREEN_MIN_FIT` (5, on a 1-10 scale), `POOL_SCREEN_BATCH_SIZE` (25)
- `POOL_SCREEN_SNIPPET_CHARS` (2000) — how much of each posting the
  screener reads. Raised from 700 once BM25 halved the number screened;
  at 700 the model saw the opening paragraph but rarely the requirements
  block where blockers live.
- `MAX_EXTRA_SEARCHES` — **no longer used**, see config.py

**Resume**
- `RESUME_MAX_PAGES` (3), `RESUME_PREFERRED_PAGES` (2)
- `SKIP_PREVIOUSLY_SELECTED` (True)

## GitHub integration (deferred)

The context agent has GitHub MCP wiring in `agents/context_agent.py`,
off until you set `GITHUB_PAT` in `.env` — no code changes needed:

1. Create a fine-grained PAT scoped to `Contents: Read-only` on the
   repos you want it to see, with an expiry.
2. Set `GITHUB_PAT=<token>` in `.env`.
3. Run as usual — the context agent will pull in READMEs, languages and
   notable repos to supplement the resume.

## Project structure

```
resume-agent/
├── agents/
│   ├── context_agent.py   # reads the resume PDF (+ optional GitHub MCP)
│   ├── market_context.py  # shared job-market calibration, imported by 3 agents
│   ├── search_agent.py    # plan angles -> scrape -> BM25 -> screen -> walk the ranking
│   ├── jd_agent.py        # supplied-job mode: fetch/parse one posting
│   ├── writer.py          # drafts + revises the ATS-ready resume
│   └── judge.py           # review_job() for stage 1, review_resume() for stage 2
├── tools/
│   ├── pdf_reader.py      # PDF text extraction
│   ├── text_reader.py     # plain .txt reading
│   ├── jobspy_tool.py     # PRIMARY search; also single LinkedIn posting fetch
│   ├── bm25_tool.py       # lexical pre-filter: rank the scrape against the resume
│   └── firecrawl_tool.py  # SUPPORT: fallback search + single-URL scraping
├── job_history.py         # cross-run ledger of selected jobs
├── html_renderer.py       # review page + early-stop report pages
├── pdf_renderer.py        # resume Markdown -> styled PDF (all appearance lives here)
├── logger_setup.py        # console + per-run file logging
├── data/
│   ├── input/             # your resume, optional sample_resume.pdf, saved JDs
│   └── output/            # generated resume, review, pool, ledger, logs
├── config.py              # keys, model routing, every tuning knob
├── main.py                # entry point: both pipelines
└── requirements.txt
```

## On "reasoning model"

Claude doesn't have a separate reasoning-model line. Current models
support **adaptive thinking** — reasoning through a hidden scratchpad
before answering, controlled by an `effort` level rather than a token
budget. Both judge calls use `thinking={"type": "adaptive"}` with
`output_config={"effort": "high"}`: judging job fit and judging resume
quality are evaluative, multi-factor calls, not generation tasks.

## Error handling

- **Recoverable, per-item failures** (a blocked scrape, one bad tool
  call, a screening batch that fails to parse) are logged and turned
  into a safe fallback so the pipeline continues. An unrated posting is
  kept rather than dropped — a failed screening call costs pool quality,
  never postings.
- **Fatal conditions** (Anthropic API unreachable, a missing input
  file, an unfetchable supplied posting) raise a clear error with an
  actionable message, log a full traceback, and stop the run cleanly.
- **Rendering failures don't lose the resume.** By the time the PDF or
  review HTML is rendered, the Markdown is already on disk, so a
  WeasyPrint library issue costs the PDF format only.

## Cost notes

- Sonnet 5 handles context, writing and judging; the judge's adaptive
  thinking adds reasoning tokens (billed as output) at both stages.
  Worst case is 3 + 3 judge calls plus the corresponding search and
  writer calls.
- The screening pass adds ~3 Haiku calls per search run (75 postings in
  batches of 25). BM25 costs nothing — it's local arithmetic.
- The overscan means ~200 postings scraped instead of 50. JobSpy adds no
  API cost, but `JOBSPY_FETCH_LINKEDIN_DESCRIPTIONS` costs one extra
  request per LinkedIn result — this is where rate limiting shows up
  first, and 200 is untested. Lower `JOB_SCRAPE_CEILING` if it bites.
  Turning the description fetch off is **not** a fix: both BM25 and the
  screener read those descriptions.
- Supplied-job mode is far cheaper — no pool, no screening, no selection
  cycles.
- Keep `MODEL_JUDGE` on Sonnet 5 or better. A careful, honest judgment
  call is exactly where a weaker model costs you most.
