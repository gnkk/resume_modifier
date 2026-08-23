# Resume Building Agent

An agentic pipeline that tailors a resume to a specific job posting:
1. **Orchestrator agent** (Claude Sonnet 5) — reads your existing resume (PDF),
   searches and scrapes the target job posting via Firecrawl.
2. **Writer agent** (Claude Sonnet 5) — drafts a tailored resume from the
   gathered context, flagging any experience gaps rather than inventing content.

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
   cp .env.example .env
   ```
   Then edit `.env` and fill in:
   - `ANTHROPIC_API_KEY` — from [platform.claude.com](https://platform.claude.com) (API Keys section)
   - `FIRECRAWL_API_KEY` — from your Firecrawl dashboard

5. **Add your resume**
   Place your current resume PDF in `data/input/`.

## Usage

```bash
python main.py data/input/my_resume.pdf "Data Scientist at Shopify, remote Canada"
```

Output is written to `data/output/tailored_resume.md`.

## Project structure

```
resume-agent/
├── agents/
│   ├── orchestrator.py   # tool-calling loop: PDF read + web search/scrape
│   └── writer.py         # drafts tailored resume content
├── tools/
│   ├── pdf_reader.py     # PDF text extraction
│   └── firecrawl_tool.py # web search + scrape via Firecrawl API
├── data/
│   ├── input/            # put source resumes here
│   └── output/           # generated resumes land here
├── config.py              # env vars + model selection
├── main.py                 # entry point
└── requirements.txt
```

## Cost notes

- Orchestrator + writer both use `claude-sonnet-5` ($2/$10 per MTok in/out) —
  agentic tool use and resume-quality writing benefit from the stronger model.
- A typical run (PDF read + a few searches/scrapes + one draft) costs roughly
  $0.10–$0.30 in API usage.
- Swap `MODEL_WRITER`/`MODEL_ORCHESTRATOR` in `config.py` to
  `claude-haiku-4-5-20251001` if you want to cut cost further during testing —
  quality of the final draft will likely drop, so switch back to Sonnet 5 for
  real output.

## Next steps (not yet built)

- A third agent to push the finished project to GitHub (planned).
- Support for multiple target roles in one run.
