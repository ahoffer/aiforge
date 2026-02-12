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

from tools import QDRANT_SEARCH_TOOL, WEB_SEARCH_TOOL, DEFAULT_TOOLS, execute_tool


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
