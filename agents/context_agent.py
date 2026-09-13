"""
Context agent.

Owns the tool-calling loop for gathering everything about the
CANDIDATE (as opposed to the job): reads the real resume PDF passed on
the command line, plus an optional job description file if one was
provided directly.

Sample/template resume:
NOT handled here. The template is read directly in main.py and passed
to the writer (and judge) as raw text — see agents/writer.py's module
docstring. Summarizing a layout into prose here lost precisely the
detail that made output match the template, so that hop was removed.
This agent is concerned only with the candidate's factual background.

GitHub project context (via GitHub's official MCP server) is wired in
but OFF by default — this project's GitHub integration is a deferred
step. Set GITHUB_PAT in .env and it activates automatically; without
it, this agent runs on the resume PDF (+ optional job description
file, + optional sample resume) alone, no code changes needed later.

This agent deliberately does NOT search the web for jobs — that is
the search agent's job (see search_agent.py). Keeping the two split
means this agent's tool loop stays small and predictable, and the
search agent can be reused independently (e.g. re-run just the job
search without re-reading the resume).

Runs on Sonnet 5 since deciding what context is actually worth pulling
in (once GitHub is added) is a judgment call, not a mechanical extraction.
"""

import os
import anthropic

from config import ANTHROPIC_API_KEY, MODEL_CONTEXT
from tools.pdf_reader import read_pdf, PDF_READER_TOOL_SCHEMA
from tools.text_reader import read_text_file, TEXT_READER_TOOL_SCHEMA
from logger_setup import get_logger, note_model

log = get_logger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Local tools this agent executes itself.
TOOLS = [
    PDF_READER_TOOL_SCHEMA,
    TEXT_READER_TOOL_SCHEMA,
]

# GitHub's official hosted MCP server — Claude calls its tools directly;
# no local execution code needed on our side, unlike the tools above.
# Only wired in when GITHUB_PAT is set (see gather_candidate_context).
# Docs: https://docs.claude.com/en/docs/agents-and-tools/mcp-connector
GITHUB_PAT = os.environ.get("GITHUB_PAT")
GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
# Beta header required for the MCP connector + mcp_toolset tool entry.
MCP_BETA_HEADER = "mcp-client-2025-11-20"

# Generous budget: this summary carries the full resume text (plus GitHub
# context and style-reference notes when enabled) and is the sole input every
# downstream agent reasons from. Running out mid-write would silently starve
# the whole pipeline. See the same note in agents/writer.py.
MAX_OUTPUT_TOKENS = 16000

SYSTEM_PROMPT_BASE = """You are a candidate-context assistant. Your only job \
is to gather information ABOUT THE CANDIDATE — you do not look for jobs; a \
separate agent handles that.

You will be given a path to the candidate's existing resume PDF, and \
OPTIONALLY a path to a job description .txt file the user has already saved.

1. Always read the existing resume with read_pdf first. This is the \
candidate's real, factual background — the only source of truth for what \
they have actually done.
2. If a job description file path was provided, read it with read_text_file \
and include its content verbatim in your summary — this will be handed to \
the search agent as a strong signal of what to look for, or to the writer \
directly if the job is already fully decided.
3. Capture the candidate's background in full and faithfully: every role \
with employer, title, dates and what they actually did; education; skills; \
projects; certifications; contact details. Downstream agents see only your \
summary, never the original PDF — anything you leave out cannot appear on \
the resume. Do not editorialize, rank, or trim for relevance; that is the \
writer's job."""

SYSTEM_PROMPT_GITHUB_ADDENDUM = """
4. Use the GitHub tools to look at the candidate's repositories (README \
content, project structure, languages used, notable projects) for concrete \
project details worth highlighting on a resume. Prioritize pinned/recently \
updated repos and README quality over an exhaustive crawl.
5. Stop once you have the candidate's full resume content and enough GitHub \
project detail to meaningfully supplement it (or a clear note that GitHub \
had nothing to add)."""

SYSTEM_PROMPT_NO_GITHUB_ADDENDUM = """
4. Stop once you have the candidate's full resume content (and job \
description content, if provided)."""

SYSTEM_PROMPT_TAIL = """

Structure your final summary in clearly labeled sections, e.g.:
CANDIDATE BACKGROUND: <everything factual about the candidate>
JOB DESCRIPTION (if provided): <verbatim content>

Do not draft resume content and do not search the web for jobs — your output \
is a structured summary of the candidate's background, nothing else."""


_LOCAL_TOOL_NAMES = {"read_pdf", "read_text_file"}


def _execute_tool(name: str, tool_input: dict) -> str:
    try:
        if name == "read_pdf":
            return read_pdf(tool_input["file_path"])
        if name == "read_text_file":
            return read_text_file(tool_input["file_path"])
        raise ValueError(f"Unknown tool: {name}")
    except KeyError as exc:
        log.error("context_agent: tool '%s' called with missing input field: %s", name, exc, exc_info=True)
        return f"[tool error: '{name}' was called without the required '{exc}' argument.]"
    except Exception as exc:  # noqa: BLE001 — tool execution must never crash the agent loop
        log.error("context_agent: unexpected error executing tool '%s': %s", name, exc, exc_info=True)
        return f"[tool error: '{name}' failed unexpectedly — {exc}]"


def gather_candidate_context(
    resume_pdf_path: str,
    job_description_path: str | None = None,
) -> str:
    """
    Run the context agent's tool-calling loop.

    Args:
        resume_pdf_path: Path to the candidate's existing resume PDF.
        job_description_path: Optional path to a .txt file containing
            a job description the user already has in hand.

    Returns:
        A text summary of the candidate's resume content (+ GitHub
        project details, once GITHUB_PAT is configured), ready to hand
        to the search agent and/or writer.

    Raises:
        RuntimeError: if the underlying Claude API call fails in a way
            that can't be recovered from (e.g. auth failure, persistent
            network error), or if the agent produces no summary at all.
    """
    use_github = bool(GITHUB_PAT)

    user_prompt = f"My existing resume is at: {resume_pdf_path}\n"
    if job_description_path:
        user_prompt += (
            f"I also already have a job description saved at: "
            f"{job_description_path} — read it and include it in your summary.\n"
        )
    user_prompt += (
        "\nGather my resume content"
        + (" and any useful GitHub project details" if use_github else "")
        + ", then summarize clearly."
    )

    messages = [{"role": "user", "content": user_prompt}]

    system_prompt = SYSTEM_PROMPT_BASE + (
        SYSTEM_PROMPT_GITHUB_ADDENDUM if use_github else SYSTEM_PROMPT_NO_GITHUB_ADDENDUM
    ) + SYSTEM_PROMPT_TAIL

    if use_github:
        all_tools = TOOLS + [{"type": "mcp_toolset", "mcp_server_name": "github"}]
        mcp_servers = [
            {
                "type": "url",
                "url": GITHUB_MCP_URL,
                "name": "github",
                "authorization_token": GITHUB_PAT,
            }
        ]

    max_turns = 12  # safety valve against a runaway tool-calling loop
    turn = 0

    while True:
        turn += 1
        if turn > max_turns:
            log.error(
                "context_agent: exceeded %d tool-calling turns without finishing; "
                "returning what was gathered so far to avoid an infinite loop.",
                max_turns,
            )
            # Best-effort: ask the model directly for a final summary rather
            # than looping forever or crashing the pipeline over this.
            return (
                "[context_agent warning: the context-gathering loop did not "
                "finish cleanly within the turn limit. Candidate context may "
                "be incomplete.]"
            )

        try:
            if use_github:
                response = client.beta.messages.create(
                    model=MODEL_CONTEXT,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    system=system_prompt,
                    tools=all_tools,
                    mcp_servers=mcp_servers,
                    betas=[MCP_BETA_HEADER],
                    messages=messages,
                )
            else:
                response = client.messages.create(
                    model=MODEL_CONTEXT,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    system=system_prompt,
                    tools=TOOLS,
                    messages=messages,
                )
        except anthropic.APIError as exc:
            log.error("context_agent: Anthropic API call failed: %s", exc, exc_info=True)
            raise RuntimeError(
                f"context_agent: failed to reach the Anthropic API ({exc}). "
                "Check your ANTHROPIC_API_KEY and network connection."
            ) from exc

        note_model(log, "context agent", response)

        if response.stop_reason != "tool_use":
            text_blocks = [b.text for b in response.content if b.type == "text"]
            summary = "\n".join(text_blocks)
            if not summary.strip():
                log.error(
                    "context_agent: model returned no text (stop_reason=%s, usage=%s). "
                    "If stop_reason is 'max_tokens', the budget ran out before the "
                    "summary was written — raise MAX_OUTPUT_TOKENS.",
                    response.stop_reason, getattr(response, "usage", None),
                )
                raise RuntimeError(
                    "context_agent: produced no candidate summary "
                    f"(stop_reason={response.stop_reason}). Every downstream agent "
                    "depends on this, so the run cannot continue. See the log for "
                    "token usage."
                )
            return summary

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name in _LOCAL_TOOL_NAMES:
                result = _execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:8000],
                    }
                )
            # GitHub MCP tool calls (when enabled) are executed server-side
            # by Anthropic; no local handling or manual tool_result needed.
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
