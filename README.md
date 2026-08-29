# Resume Building Agent

An agentic pipeline that tailors a resume to a specific job, then critically
reviews its own output before you see it:

1. **Orchestrator agent** (Claude Sonnet 5) — reads your existing resume (PDF),
   reads a job description file if you provide one, searches/scrapes the web
   only to fill genuine gaps, and can read your GitHub repos (via GitHub's
   official MCP server) for supporting project details.
2. **Writer agent** (Claude Sonnet 5) — drafts a tailored resume from the
   gathered context, flagging any experience gaps rather than inventing content.
3. **Reflector agent** (Claude Sonnet 5, adaptive thinking) — critically
   reviews the draft against the job requirements: a fitness score, strengths,
   gaps, and concrete improvement suggestions. Written to push back on weak
   output rather than rubber-stamp it.

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

4. **Set up API keys**
   ```bash
   Make an file called .env for saving API keys and other env variables. 
   ```
   Then edit `.env` and fill in:
   - `ANTHROPIC_API_KEY` — from [platform.claude.com](https://platform.claude.com) (API Keys section)
   - `FIRECRAWL_API_KEY` — from your Firecrawl dashboard
   - `GITHUB_PAT` — a fine-grained GitHub Personal Access Token scoped to
     `Contents: Read/Write` and `Pull requests: Read/Write` on the repo(s)
     you want the agent to access

5. **Add your resume**
   Place your current resume PDF in `data/input/`.

6. **(Optional) Add a job description file**
   If you already have the job posting text, save it as a `.txt` file in
   `data/input/` — the orchestrator treats this as authoritative and only
   searches the web for genuine gaps (e.g. company culture info the posting
   doesn't cover), not to re-find the posting itself.

## Usage

Without a job description file (orchestrator searches the web for the posting):
```bash
python main.py data/input/my_resume.pdf "Data Scientist at Shopify, remote Canada"
```

With a job description file (recommended — more accurate, fewer web calls):
```bash
python main.py data/input/my_resume.pdf "Data Scientist at Shopify" data/input/job_description.txt
```

Output:
- `data/output/tailored_resume.md` — the tailored resume
- `data/output/resume_review.json` — the reflector's structured critique
  (fitness score, strengths, gaps, suggestions)

## Project structure

```
resume-agent/
├── agents/
│   ├── orchestrator.py   # tool-calling loop: PDF/JD read, web search/scrape, GitHub MCP
│   ├── writer.py         # drafts tailored resume content
│   └── reflector.py      # critiques the draft using adaptive thinking
├── tools/
│   ├── pdf_reader.py     # PDF text extraction
│   ├── text_reader.py    # plain .txt file reading (job descriptions)
│   └── firecrawl_tool.py # web search + scrape via Firecrawl API
├── data/
│   ├── input/            # put source resumes + job description files here
│   └── output/           # generated resume + review land here
├── config.py              # env vars + model selection
├── main.py                 # entry point
└── requirements.txt
```

## On "reasoning model"

Claude doesn't have a separate reasoning-model line the way some providers
do. Instead, current models like Sonnet 5 support **adaptive thinking** — a
mode where Claude reasons through a hidden scratchpad before answering,
controlled by an `effort` level (`standard`, `high`, `xhigh`, `max`) rather
than a manual token budget. The reflector agent uses
`thinking={"type": "adaptive"}` with `output_config={"effort": "high"}` for
exactly this reason: judging resume quality and job fit is an evaluative,
multi-factor call, not a generation task, and benefits from that extra
reasoning step.

## Cost notes

- Orchestrator, writer, and reflector all use `claude-sonnet-5`
  ($2/$10 per MTok in/out). The reflector's adaptive thinking adds some
  reasoning tokens (billed as output) on top of the base cost.
- A typical run (PDF/JD read + occasional search + draft + review) costs
  roughly $0.15–$0.40 in API usage.
- Swap `MODEL_WRITER`/`MODEL_ORCHESTRATOR` in `config.py` to
  `claude-haiku-4-5-20251001` if you want to cut cost further during testing.
  Keep `MODEL_REFLECTOR` on Sonnet 5 — the whole point of this agent is a
  careful, honest judgment call, which is exactly where a weaker model costs
  you the most.

## Next steps (not yet built)

- A fourth agent to push the finished project to GitHub.
- Support for multiple target roles in one run.
- Optionally loop the reflector's suggestions back into the writer for a
  second draft pass.
