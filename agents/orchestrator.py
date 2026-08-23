"""
Orchestrator agent.

Owns the tool-calling loop: reads your existing resume (PDF),
searches/scrapes a target job posting via Firecrawl, and gathers
everything the writing agent needs. Runs on Sonnet 5 since this
involves multi-step judgment about what to look up next.
"""

import json
import anthropic

from config import ANTHROPIC_API_KEY, MODEL_ORCHESTRATOR
from tools.pdf_reader import read_pdf, PDF_READER_TOOL_SCHEMA
from tools.firecrawl_tool import (
    search,
    scrape,
    FIRECRAWL_SEARCH_TOOL_SCHEMA,
    FIRECRAWL_SCRAPE_TOOL_SCHEMA,
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

TOOLS = [PDF_READER_TOOL_SCHEMA, FIRECRAWL_SEARCH_TOOL_SCHEMA, FIRECRAWL_SCRAPE_TOOL_SCHEMA]

SYSTEM_PROMPT = """You are a research assistant that gathers everything needed \
to tailor a resume for a specific job.

Given a path to the user's existing resume PDF and a target role/company, you should:
1. Read the existing resume with read_pdf.
2. Use web_search to find the specific job posting (or similar postings if none given).
3. Use scrape_url to pull the full text of the most relevant posting(s).
4. Stop once you have: the candidate's current resume content, and the target \
job's requirements/responsibilities in enough detail to tailor a resume against.

Do not draft resume content yourself — that is a separate agent's job. Your \
output is a structured summary of gathered information."""


def _execute_tool(name: str, tool_input: dict) -> str:
    if name == "read_pdf":
        return read_pdf(tool_input["file_path"])
    if name == "web_search":
        results = search(tool_input["query"], tool_input.get("limit", 5))
        return json.dumps(results)
    if name == "scrape_url":
        return scrape(tool_input["url"])
    raise ValueError(f"Unknown tool: {name}")


def gather_context(resume_pdf_path: str, target_role_description: str) -> str:
    """
    Run the orchestrator's tool-calling loop.

    Args:
        resume_pdf_path: Path to the user's existing resume PDF.
        target_role_description: Free-text description of the target
            role/company, e.g. "Data Scientist at Shopify, Toronto".

    Returns:
        A text summary combining resume content + job requirements,
        ready to hand to the writing agent.
    """
    messages = [
        {
            "role": "user",
            "content": (
                f"My existing resume is at: {resume_pdf_path}\n"
                f"Target role: {target_role_description}\n\n"
                "Gather what you need, then summarize it clearly."
            ),
        }
    ]

    # Tool-calling loop: keep going until Claude stops requesting tools.
    while True:
        response = client.messages.create(
            model=MODEL_ORCHESTRATOR,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
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
            if block.type == "tool_use":
                result = _execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:8000],  # keep context lean
                    }
                )
        messages.append({"role": "user", "content": tool_results})
