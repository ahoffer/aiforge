# tools.py
"""LangChain tool definitions for the agent.

Each @tool function becomes a BaseTool with auto-generated JSON schema
derived from the type annotations and docstring. The TOOLS list is
passed to ChatOllama.bind_tools() and ToolNode() in graph.py.

Retriever tools for known Qdrant collections are auto-created from the
QDRANT_COLLECTIONS env var at module load time using QdrantVectorStore
and OllamaEmbeddings. The manual qdrant_search tool remains for ad-hoc
collection access.
"""

import json
import logging

from langchain_core.tools import tool

from clients import OllamaClient, QdrantClient, SearxngClient
from config import QDRANT_URL, QDRANT_COLLECTIONS, OLLAMA_URL, EMBEDDING_MODEL

log = logging.getLogger(__name__)

# Module-level singletons so tool functions can call services without
# needing dependency injection. Created once at import time.
_searxng = SearxngClient()
_qdrant = QdrantClient()
_ollama = OllamaClient()

# Cap per-tool output to limit prompt injection surface from hostile
# search snippets or oversized vector payloads.
MAX_TOOL_OUTPUT_CHARS = 4000


@tool
def web_search(query: str) -> str:
    """Search the web for current events or real-time information.

    Use this when the user explicitly asks to search the web, or when
    answering questions about recent or time-sensitive topics. Do not
    use this for well-known facts unless the user requests a search.

    Args:
        query: The search query string.
    """
    log.info("Executing web_search: %s", query)
    text = _searxng.search_text(query, max_results=5)
    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        text = text[:MAX_TOOL_OUTPUT_CHARS] + "\n[truncated]"
    return text


@tool
def qdrant_search(query: str, collection: str, limit: int = 5) -> str:
    """Search indexed knowledge in a Qdrant vector store collection.

    Use this for semantic search over previously indexed documents.
    Requires both a query and a collection name.

    Args:
        query: Natural language search query.
        collection: Qdrant collection name to search.
        limit: Maximum number of matches, 1 to 10.
    """
    query = query.strip()
    collection = collection.strip()
    limit = max(1, min(limit, 10))

    if not query:
        return "Error: qdrant_search requires non-empty 'query'."
    if not collection:
        return "Error: qdrant_search requires non-empty 'collection'."

    log.info("Executing qdrant_search query=%r collection=%r limit=%d", query, collection, limit)

    embeddings = _ollama.embed(query)
    if not embeddings or not embeddings[0]:
        return "Error: failed to generate embedding vector for qdrant_search query."

    results = _qdrant.search(collection=collection, vector=embeddings[0], limit=limit, with_payload=True)
    if not results:
        return f"No vector matches found in collection '{collection}' for query: {query}"

    lines = [f"Qdrant results for query: {query} (collection: {collection})", ""]
    for i, point in enumerate(results, 1):
        payload = point.get("payload", {})
        score = point.get("score", 0)
        lines.append(f"{i}. id={point.get('id')} score={score:.4f}")
        if payload:
            snippet = str(payload)[:400]
            lines.append(f"   payload: {snippet}")
        lines.append("")

    text = "\n".join(lines)
    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        text = text[:MAX_TOOL_OUTPUT_CHARS] + "\n[truncated]"
    return text


def _make_retriever_tools() -> list:
    """Create retriever tools for known Qdrant collections.

    Parses the QDRANT_COLLECTIONS env var, which is a JSON array of objects
    like [{"name": "docs", "description": "Project documentation"}]. For each
    entry, creates a retriever tool backed by QdrantVectorStore with
    OllamaEmbeddings. Returns an empty list if the env var is not set or
    if any required package is missing.
    """
    if not QDRANT_COLLECTIONS:
        return []

    try:
        specs = json.loads(QDRANT_COLLECTIONS)
    except (json.JSONDecodeError, TypeError):
        log.warning("QDRANT_COLLECTIONS is not valid JSON, skipping retriever tools")
        return []

    if not isinstance(specs, list) or not specs:
        return []

    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from langchain_core.tools.retriever import create_retriever_tool
    except ImportError:
        log.warning("langchain-qdrant or langchain-ollama not installed, skipping retriever tools")
        return []

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    tools = []

    for spec in specs:
        name = spec.get("name", "")
        description = spec.get("description", f"Search the {name} collection")
        if not name:
            continue

        try:
            store = QdrantVectorStore.from_existing_collection(
                collection_name=name,
                embedding=embeddings,
                url=QDRANT_URL,
            )
            retriever = store.as_retriever(search_kwargs={"k": 5})
            retriever_tool = create_retriever_tool(
                retriever,
                name=f"search_{name}",
                description=description,
            )
            tools.append(retriever_tool)
            log.info("Created retriever tool search_%s for collection %s", name, name)
        except Exception:
            log.warning("Failed to create retriever for collection %s", name, exc_info=True)

    return tools


TOOLS = [web_search, qdrant_search] + _make_retriever_tools()
