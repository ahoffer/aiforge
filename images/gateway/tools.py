# tools.py
"""Tool definitions and dispatch for the agent.

Each tool has an OpenAI-compatible schema for Ollama native tool calling
and a handler function that executes the tool. Adding a tool means defining
a schema, a handler, and registering both in _HANDLERS.
"""

import logging
from typing import Iterable

from clients import OllamaClient, QdrantClient, SearxngClient

log = logging.getLogger(__name__)

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current events or real-time information that you do not already know. Always use this when the user explicitly asks to search the web. Do not use this for well-known facts or concepts unless the user requests a search.",
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


def _handle_web_search(arguments: dict, *, searxng: SearxngClient, **_kw) -> str:
    query = str(arguments.get("query", "") or "")
    log.info("Executing web_search: %s", query)
    return searxng.search_text(query, max_results=5)


def _handle_qdrant_search(
    arguments: dict, *, qdrant: QdrantClient, ollama: OllamaClient, **_kw
) -> str:
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


# Single registration point for server-side tools. graph.py never changes.
_HANDLERS: dict[str, tuple[dict, callable]] = {
    "web_search": (WEB_SEARCH_TOOL, _handle_web_search),
    "qdrant_search": (QDRANT_SEARCH_TOOL, _handle_qdrant_search),
}

DEFAULT_TOOLS = [schema for schema, _handler in _HANDLERS.values()]


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
    """Dispatch a tool call to the registered handler.

    Returns formatted text that gets appended to the conversation
    as a tool response message.
    """
    entry = _HANDLERS.get(name)
    if not entry:
        return f"Unknown tool: {name}"
    _schema, handler = entry
    return handler(arguments, searxng=searxng, qdrant=qdrant, ollama=ollama)
