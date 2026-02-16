"""Tests for LangChain @tool definitions."""

import sys
from unittest.mock import MagicMock, patch

# Ensure real modules, not stale mocks from other test files
for mod in ("clients", "tools"):
    if mod in sys.modules and isinstance(sys.modules[mod], MagicMock):
        del sys.modules[mod]

from tools import TOOLS, web_search, qdrant_search, MAX_TOOL_OUTPUT_CHARS


def test_tools_list_has_expected_entries():
    names = [t.name for t in TOOLS]
    assert "web_search" in names
    assert "qdrant_search" in names


def test_web_search_has_json_schema():
    schema = web_search.get_input_schema().model_json_schema()
    assert "query" in schema.get("properties", {})
    assert "query" in schema.get("required", [])


def test_qdrant_search_has_json_schema():
    schema = qdrant_search.get_input_schema().model_json_schema()
    props = schema.get("properties", {})
    assert "query" in props
    assert "collection" in props
    assert "limit" in props
    assert "query" in schema.get("required", [])
    assert "collection" in schema.get("required", [])


@patch("tools._searxng")
def test_web_search_invokes_searxng(mock_searxng):
    mock_searxng.search_text.return_value = "Results for python"
    result = web_search.invoke({"query": "python"})
    assert result == "Results for python"
    mock_searxng.search_text.assert_called_once_with("python", max_results=5)


@patch("tools._qdrant")
@patch("tools._ollama")
def test_qdrant_search_invokes_clients(mock_ollama, mock_qdrant):
    mock_ollama.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_qdrant.search.return_value = [
        {"id": 1, "score": 0.92, "payload": {"text": "hello world"}},
    ]
    result = qdrant_search.invoke({"query": "hello", "collection": "docs", "limit": 3})
    assert "Qdrant results for query: hello" in result
    mock_ollama.embed.assert_called_once_with("hello")
    mock_qdrant.search.assert_called_once_with(
        collection="docs", vector=[0.1, 0.2, 0.3], limit=3, with_payload=True,
    )


def test_qdrant_search_empty_query_returns_error():
    result = qdrant_search.invoke({"query": "", "collection": "docs"})
    assert "error" in result.lower()


def test_qdrant_search_empty_collection_returns_error():
    result = qdrant_search.invoke({"query": "hello", "collection": ""})
    assert "error" in result.lower()


@patch("tools._searxng")
def test_web_search_truncates_large_output(mock_searxng):
    oversized = "x" * (MAX_TOOL_OUTPUT_CHARS + 500)
    mock_searxng.search_text.return_value = oversized
    result = web_search.invoke({"query": "big"})
    assert result.endswith("[truncated]")
    assert len(result) == MAX_TOOL_OUTPUT_CHARS + len("\n[truncated]")


@patch("tools._searxng")
def test_web_search_exception_propagates(mock_searxng):
    """Tool exceptions propagate. ToolNode catches them and returns an error ToolMessage."""
    mock_searxng.search_text.side_effect = ConnectionError("searxng down")
    raised = False
    try:
        web_search.invoke({"query": "test"})
    except ConnectionError:
        raised = True
    assert raised
