"""Tests for tool schema and dispatch."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Ensure we have a real clients module (not the mock from test_graph)
# by reloading if it was mocked
if "clients" in sys.modules and isinstance(sys.modules["clients"], MagicMock):
    del sys.modules["clients"]
if "tools" in sys.modules and isinstance(sys.modules["tools"], MagicMock):
    del sys.modules["tools"]

from tools import WEB_SEARCH_TOOL, TOOLS, execute_tool


def test_web_search_tool_schema():
    assert WEB_SEARCH_TOOL["type"] == "function"
    func = WEB_SEARCH_TOOL["function"]
    assert func["name"] == "web_search"
    assert "query" in func["parameters"]["properties"]
    assert "query" in func["parameters"]["required"]


def test_tools_list_contains_web_search():
    names = [t["function"]["name"] for t in TOOLS]
    assert "web_search" in names


def test_execute_tool_web_search():
    mock_searxng = MagicMock()
    mock_searxng.search_text.return_value = "Results for python"

    result = execute_tool("web_search", {"query": "python"}, mock_searxng)
    assert result == "Results for python"
    mock_searxng.search_text.assert_called_once_with("python", max_results=5)


def test_execute_tool_unknown():
    result = execute_tool("nonexistent", {}, MagicMock())
    assert "Unknown tool" in result
