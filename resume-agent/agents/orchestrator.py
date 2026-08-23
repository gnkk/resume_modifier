"""
Orchestrator agent.

Owns the tool-calling loop: reads your existing resume (PDF),
searches/scrapes a target job posting via Firecrawl, and gathers
everything the writing agent needs. Runs on Sonnet 5 since this
involves multi-step judgment about what to look up next.
"""

import json
import anthropic

from config import ANTHROPIC_API_KEY, MODEL_ORCHESTRATOR, GITHUB_PAT
from tools.pdf_reader import read_pdf, PDF_READER_TOOL_SCHEMA
from tools.text_reader import read_text_file, TEXT_READER_TOOL_SCHEMA
from tools.firecrawl_tool import (
    search,
    scrape,
    FIRECRAWL_SEARCH_TOOL_SCHEMA,
    FIRECRAWL_SCRAPE_TOOL_SCHEMA,
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Local tools this agent executes itself (PDF, text file, Firecrawl).
TOOLS = [
    PDF_READER_TOOL_SCHEMA,
    TEXT_READER_TOOL_SCHEMA,
    FIRECRAWL_SEARCH_TOOL_SCHEMA,
    FIRECRAWL_SCRAPE_TOOL_SCHEMA,
]

# GitHub's official hosted MCP server — Claude calls its tools directly;
# no local execution code needed on our side, unlike the tools above.
# Docs: https://docs.claude.com/en/docs/agents-and-tools/mcp-connector
MCP_SERVERS = [
    {
        "type": "url",
        "url": "https://api.githubcopilot.com/mcp/",
        "name": "github",
        "authorization_token": GITHUB_PAT,
    }
]

# Betas required: MCP connector, plus tool-use-with-MCP needs the toolset
# entry below in `tools`, matching Anthropic's current (Nov 2025+) API shape.
MCP_BETA_HEADER = "mcp-client-2025-11-20"

SYSTEM_PROMPT = """You are a research and repo-context assistant that gathers \
everything needed to tailor a resume for a specific job, and can also read \
the candidate's GitHub repositories for supporting project details.

You will be given a path to the user's existing resume PDF, a target role/company \
description, and OPTIONALLY a path to a job description .txt file.

Follow this priority order:
1. Always read the existing resume with read_pdf first.
2. If a job description file path was provided, read it with read_text_file. \
Treat this as the authoritative source for the job's requirements — do not \
search the web to find the posting itself, since you already have it.
3. Only use web_search / scrape_url if:
   - No job description file was provided, OR
   - The job description file lacks information you genuinely need (e.g. company \
culture, team context, recent news) that would meaningfully improve the tailored \
resume. Do not search speculatively "just in case" — only when there's a real gap.
4. If relevant, use the GitHub tools to look at the candidate's repositories \
(e.g. README content, project structure) for concrete project details worth \
highlighting on the resume.
5. Stop once you have: the candidate's current resume content, relevant GitHub \
project details, and the target job's requirements/responsibilities in enough \
detail to tailor a resume against.

Do not draft resume content yourself — that is a separate agent's job. Your \
output is a structured summary of gathered information, and should note whether \
the job requirements came from the provided file, the web, or both."""


_LOCAL_TOOL_NAMES = {"read_pdf", "read_text_file", "web_search", "scrape_url"}


def _execute_tool(name: str, tool_input: dict) -> str:
    if name == "read_pdf":
        return read_pdf(tool_input["file_path"])
    if name == "read_text_file":
        return read_text_file(tool_input["file_path"])
    if name == "web_search":
        results = search(tool_input["query"], tool_input.get("limit", 5))
        return json.dumps(results)
    if name == "scrape_url":
        return scrape(tool_input["url"])
    raise ValueError(f"Unknown tool: {name}")


def gather_context(
    resume_pdf_path: str,
    target_role_description: str,
    job_description_path: str | None = None,
) -> str:
    """
    Run the orchestrator's tool-calling loop.

    Args:
        resume_pdf_path: Path to the user's existing resume PDF.
        target_role_description: Free-text description of the target
            role/company, e.g. "Data Scientist at Shopify, Toronto".
        job_description_path: Optional path to a .txt file containing
            the job description directly. When given, this is treated
            as the authoritative source — web search is used only to
            fill genuine gaps, not to re-find the posting.

    Returns:
        A text summary combining resume content + job requirements,
        ready to hand to the writing agent.
    """
    user_prompt = (
        f"My existing resume is at: {resume_pdf_path}\n"
        f"Target role: {target_role_description}\n"
    )
    if job_description_path:
        user_prompt += (
            f"Job description file (authoritative — read this first, "
            f"only search the web for genuine gaps): {job_description_path}\n"
        )
    else:
        user_prompt += (
            "No job description file was provided — search the web to "
            "find the posting.\n"
        )
    user_prompt += "\nGather what you need, then summarize it clearly."

    messages = [{"role": "user", "content": user_prompt}]

    # Tools list: our locally-executed tools + the GitHub MCP toolset marker.
    # The mcp_toolset entry tells Claude which mcp_servers entry to pull
    # tool definitions from — GitHub's server exposes its own tool set
    # (search code, get file contents, list PRs, etc.) that we never
    # define ourselves.
    all_tools = TOOLS + [{"type": "mcp_toolset", "mcp_server_name": "github"}]

    # Tool-calling loop: keep going until Claude stops requesting tools.
    while True:
        response = client.beta.messages.create(
            model=MODEL_ORCHESTRATOR,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=all_tools,
            mcp_servers=MCP_SERVERS,
            betas=[MCP_BETA_HEADER],
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            # Final answer — extract and return the text.
            text_blocks = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_blocks)

        # Append assistant's tool-use turn, then run each tool and
        # append the results before looping again.
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name in _LOCAL_TOOL_NAMES:
                # We execute local tools (PDF, Firecrawl) ourselves.
                result = _execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:8000],  # keep context lean
                    }
                )
            # GitHub MCP tool calls are executed server-side by Anthropic
            # against the remote MCP server — no local handling needed,
            # and no manual tool_result to append for those.
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
