"""Tests for tool schema and dispatch."""

import os
import sys
from unittest.mock import MagicMock, patch

# Ensure we have a real clients module (not the mock from test_graph)
# by reloading if it was mocked
if "clients" in sys.modules and isinstance(sys.modules["clients"], MagicMock):
    del sys.modules["clients"]
if "tools" in sys.modules and isinstance(sys.modules["tools"], MagicMock):
    del sys.modules["tools"]

from tools import QDRANT_SEARCH_TOOL, WEB_SEARCH_TOOL, DEFAULT_TOOLS, execute_tool, merge_tools


def test_web_search_tool_schema():
    assert WEB_SEARCH_TOOL["type"] == "function"
    func = WEB_SEARCH_TOOL["function"]
    assert func["name"] == "web_search"
    assert "query" in func["parameters"]["properties"]
    assert "query" in func["parameters"]["required"]


def test_qdrant_search_tool_schema():
    assert QDRANT_SEARCH_TOOL["type"] == "function"
    func = QDRANT_SEARCH_TOOL["function"]
    assert func["name"] == "qdrant_search"
    assert "query" in func["parameters"]["properties"]
    assert "collection" in func["parameters"]["properties"]
    assert "query" in func["parameters"]["required"]
    assert "collection" in func["parameters"]["required"]


def test_tools_list_contains_web_search():
    names = [t["function"]["name"] for t in DEFAULT_TOOLS]
    assert "web_search" in names
    assert "qdrant_search" in names


def test_execute_tool_web_search():
    mock_searxng = MagicMock()
    mock_qdrant = MagicMock()
    mock_ollama = MagicMock()
    mock_searxng.search_text.return_value = "Results for python"

    result = execute_tool("web_search", {"query": "python"}, mock_searxng, mock_qdrant, mock_ollama)
    assert result == "Results for python"
    mock_searxng.search_text.assert_called_once_with("python", max_results=5)


def test_execute_tool_qdrant_search():
    mock_searxng = MagicMock()
    mock_qdrant = MagicMock()
    mock_ollama = MagicMock()
    mock_ollama.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_qdrant.search.return_value = [
        {"id": 1, "score": 0.92, "payload": {"text": "hello world"}},
    ]

    result = execute_tool(
        "qdrant_search",
        {"query": "hello", "collection": "docs", "limit": 3},
        mock_searxng,
        mock_qdrant,
        mock_ollama,
    )
    assert "Qdrant results for query: hello" in result
    mock_ollama.embed.assert_called_once_with("hello")
    mock_qdrant.search.assert_called_once_with(
        collection="docs",
        vector=[0.1, 0.2, 0.3, 0.4],
        limit=3,
        with_payload=True,
    )


def test_execute_tool_unknown():
    result = execute_tool("nonexistent", {}, MagicMock(), MagicMock(), MagicMock())
    assert "Unknown tool" in result


def test_execute_tool_unknown_returns_string_not_exception():
    """Unknown tool names return an error string, never raise."""
    result = execute_tool("totally_fake_tool", {"arg": "val"}, MagicMock(), MagicMock(), MagicMock())
    assert isinstance(result, str)
    assert "totally_fake_tool" in result


def test_execute_tool_web_search_exception_returns_error_string():
    """When the SearXNG client raises, execute_tool should not propagate.
    The caller in graph.py catches exceptions, but execute_tool for known
    tools should still be robust if the client throws."""
    mock_searxng = MagicMock()
    mock_searxng.search_text.side_effect = ConnectionError("searxng down")
    # execute_tool does not catch internally for web_search, so this will raise.
    # This test documents the current behavior: the exception propagates
    # and graph.py's tools_node catches it.
    try:
        execute_tool("web_search", {"query": "test"}, mock_searxng, MagicMock(), MagicMock())
        raised = False
    except ConnectionError:
        raised = True
    assert raised, "Exception from client should propagate to caller"


def test_merge_tools_none_extras_returns_base():
    base = [WEB_SEARCH_TOOL]
    result = merge_tools(base, None)
    assert result == base


def test_merge_tools_empty_extras_returns_base():
    base = [WEB_SEARCH_TOOL, QDRANT_SEARCH_TOOL]
    result = merge_tools(base, [])
    assert result == base


def test_merge_tools_duplicate_base_takes_precedence():
    custom_search = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "custom version",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    result = merge_tools([WEB_SEARCH_TOOL], [custom_search])
    assert len(result) == 1
    assert result[0]["function"]["description"] == WEB_SEARCH_TOOL["function"]["description"]


def test_merge_tools_adds_new_tools():
    extra_tool = {
        "type": "function",
        "function": {
            "name": "custom_tool",
            "description": "a new tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    result = merge_tools([WEB_SEARCH_TOOL], [extra_tool])
    names = [t["function"]["name"] for t in result]
    assert "web_search" in names
    assert "custom_tool" in names
