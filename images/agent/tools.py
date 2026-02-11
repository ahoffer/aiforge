"""Tool definitions and dispatch for the agent.

Each tool has an OpenAI-compatible schema for Ollama native tool calling
and a dispatch function that routes tool calls to the appropriate client.
"""

import logging

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
                "query": {
                    "type": "string",
                    "description": "Search query",
                }
            },
            "required": ["query"],
        },
    },
}

TOOLS = [WEB_SEARCH_TOOL]


def execute_tool(name: str, arguments: dict, searxng: SearxngClient) -> str:
    """Dispatch a tool call to the appropriate client.

    Returns formatted text that gets appended to the conversation
    as a tool response message.
    """
    if name == "web_search":
        query = arguments.get("query", "")
        log.info("Executing web_search: %s", query)
        return searxng.search_text(query, max_results=5)

    return f"Unknown tool: {name}"
