# tools.py
"""Tool definitions and dispatch for the agent.

Each tool has an OpenAI-compatible schema for Ollama native tool calling
and a dispatch function that routes tool calls to the appropriate client.
"""

import logging
from typing import Iterable

from clients import SearxngClient

log = logging.getLogger(__name__)

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information on any topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}

DEFAULT_TOOLS = [WEB_SEARCH_TOOL]


def merge_tools(base: list[dict], extra: list[dict] | None) -> list[dict]:
    """Merge tool lists by function.name, keeping base tools if duplicates exist."""
    merged = []
    seen = set()

    def add_all(tools: Iterable[dict]):
        for t in tools:
            fn = (t or {}).get("function", {}) or {}
            name = fn.get("name", "") or ""
            if name and name in seen:
                continue
            if name:
                seen.add(name)
            merged.append(t)

    add_all(base or [])
    add_all(extra or [])
    return merged


def execute_tool(name: str, arguments: dict, searxng: SearxngClient) -> str:
    """Dispatch a tool call to the appropriate client.

    Returns formatted text that gets appended to the conversation
    as a tool response message.
    """
    if name == "web_search":
        query = str(arguments.get("query", "") or "")
        log.info("Executing web_search: %s", query)
        # Expect this to include URLs in the returned text so the model can cite them.
        return searxng.search_text(query, max_results=5)

    return f"Unknown tool: {name}"
