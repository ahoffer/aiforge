# tools.py
"""Tool definitions and dispatch for the agent.

Each tool has an OpenAI-compatible schema for Ollama native tool calling
and a dispatch function that routes tool calls to the appropriate client.
"""

import logging
from typing import Iterable

from clients import OllamaClient, QdrantClient, SearxngClient

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

QDRANT_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "qdrant_search",
        "description": "Search indexed knowledge in Qdrant vector store",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "collection": {"type": "string", "description": "Qdrant collection name"},
                "limit": {"type": "integer", "description": "Max number of matches (1-10)"},
            },
            "required": ["query", "collection"],
        },
    },
}

DEFAULT_TOOLS = [WEB_SEARCH_TOOL, QDRANT_SEARCH_TOOL]


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


def execute_tool(
    name: str,
    arguments: dict,
    searxng: SearxngClient,
    qdrant: QdrantClient,
    ollama: OllamaClient,
) -> str:
    """Dispatch a tool call to the appropriate client.

    Returns formatted text that gets appended to the conversation
    as a tool response message.
    """
    if name == "web_search":
        query = str(arguments.get("query", "") or "")
        log.info("Executing web_search: %s", query)
        # Expect this to include URLs in the returned text so the model can cite them.
        return searxng.search_text(query, max_results=5)

    if name == "qdrant_search":
        query = str(arguments.get("query", "") or "").strip()
        collection = str(arguments.get("collection", "") or "").strip()
        limit = int(arguments.get("limit", 5) or 5)
        limit = max(1, min(limit, 10))

        if not query:
            return "Error: qdrant_search requires non-empty 'query'."
        if not collection:
            return "Error: qdrant_search requires non-empty 'collection'."

        log.info("Executing qdrant_search query=%r collection=%r limit=%d", query, collection, limit)

        embeddings = ollama.embed(query)
        if not embeddings or not embeddings[0]:
            return "Error: failed to generate embedding vector for qdrant_search query."

        results = qdrant.search(collection=collection, vector=embeddings[0], limit=limit, with_payload=True)
        if not results:
            return f"No vector matches found in collection '{collection}' for query: {query}"

        lines = [f"Qdrant results for query: {query} (collection: {collection})", ""]
        for i, point in enumerate(results, 1):
            payload = point.get("payload", {})
            score = point.get("score", 0)
            lines.append(f"{i}. id={point.get('id')} score={score:.4f}")
            if payload:
                # Keep payload compact so tool output does not blow up context.
                snippet = str(payload)[:400]
                lines.append(f"   payload: {snippet}")
            lines.append("")
        return "\n".join(lines)

    return f"Unknown tool: {name}"
