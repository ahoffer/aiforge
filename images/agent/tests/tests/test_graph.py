"""Tests for graph routing logic.

These test the pure Python functions in graph.py without needing
LangGraph or live services by mocking external dependencies.
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock langgraph before importing graph.py
_mock_langgraph = MagicMock()
_mock_langgraph_graph = MagicMock()
_mock_langgraph_graph.END = "__end__"
sys.modules["langgraph"] = _mock_langgraph
sys.modules["langgraph.graph"] = _mock_langgraph_graph

# Mock client and tools modules to avoid pulling in dependencies
_mock_clients = MagicMock()
sys.modules["clients"] = _mock_clients

# Save and mock the tools module, but restore it after graph import
# so test_tools.py can import the real module
_saved_tools = sys.modules.pop("tools", None)
sys.modules["tools"] = MagicMock()

import graph  # noqa: E402

# Restore tools module so other test files get the real one
del sys.modules["tools"]
if _saved_tools is not None:
    sys.modules["tools"] = _saved_tools

route_after_orchestrator = graph.route_after_orchestrator


def test_route_to_search_when_tool_calls_present():
    state = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "web_search"}}]},
        ],
        "search_count": 0,
    }
    assert route_after_orchestrator(state) == "search"


def test_route_to_end_when_no_tool_calls():
    state = {
        "messages": [
            {"role": "assistant", "content": "The answer is 4."},
        ],
        "search_count": 0,
    }
    assert route_after_orchestrator(state) == "__end__"


def test_route_to_end_at_max_iterations():
    state = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "web_search"}}]},
        ],
        "search_count": 5,
    }
    assert route_after_orchestrator(state) == "__end__"


def test_route_to_end_with_empty_messages():
    state = {"messages": [], "search_count": 0}
    assert route_after_orchestrator(state) == "__end__"


def test_route_search_increments_up_to_limit():
    """Successive calls should route to search until max iterations."""
    for count in range(5):
        state = {
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "web_search"}}]},
            ],
            "search_count": count,
        }
        assert route_after_orchestrator(state) == "search"

    # At exactly 5 it should stop
    state["search_count"] = 5
    assert route_after_orchestrator(state) == "__end__"
