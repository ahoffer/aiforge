"""Tests for graph routing and node logic.

These test the pure Python functions in graph.py without needing
LangGraph or live services by mocking external dependencies.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock langgraph before importing graph.py
_mock_langgraph = MagicMock()
_mock_langgraph_graph = MagicMock()
_mock_langgraph_graph.END = "__end__"
_mock_langgraph_graph_state = MagicMock()
sys.modules["langgraph"] = _mock_langgraph
sys.modules["langgraph.graph"] = _mock_langgraph_graph
sys.modules["langgraph.graph.state"] = _mock_langgraph_graph_state

_mock_langchain_core = MagicMock()
_mock_langchain_core_globals = MagicMock()
sys.modules["langchain_core"] = _mock_langchain_core
sys.modules["langchain_core.globals"] = _mock_langchain_core_globals

# Mock client and tools modules to avoid pulling in dependencies
_mock_clients = MagicMock()
sys.modules["clients"] = _mock_clients

# Save and mock the tools module, but restore it after graph import
# so test_tools.py can import the real module
_saved_tools = sys.modules.pop("tools", None)
_mock_tools = MagicMock()
sys.modules["tools"] = _mock_tools

import graph  # noqa: E402

# Restore tools module so other test files get the real one
del sys.modules["tools"]
if _saved_tools is not None:
    sys.modules["tools"] = _saved_tools

route_after_orchestrator = graph.route_after_orchestrator
orchestrator_node = graph.orchestrator_node
tools_node = graph.tools_node


# -- Routing tests --

def test_route_to_tools_when_tool_calls_present():
    state = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "web_search"}}]},
        ],
    }
    assert route_after_orchestrator(state) == "tools"


def test_route_to_end_when_no_tool_calls():
    state = {
        "messages": [
            {"role": "assistant", "content": "The answer is 4."},
        ],
    }
    assert route_after_orchestrator(state) == "__end__"


def test_route_to_end_with_empty_messages():
    state = {"messages": []}
    assert route_after_orchestrator(state) == "__end__"


def test_route_to_tools_regardless_of_iteration_count():
    """Router no longer checks iteration count. The orchestrator handles that."""
    state = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "web_search"}}]},
        ],
        "tool_iterations": 10,
    }
    assert route_after_orchestrator(state) == "tools"


# -- Orchestrator node tests --

def test_orchestrator_missing_message_returns_error():
    """When state has no message and no messages, return an error response."""
    state = {}
    result = orchestrator_node(state)
    assert "final_response" in result
    assert "error" in result["final_response"].lower()


@patch.object(graph, "OllamaClient")
def test_orchestrator_max_iterations_forces_final_answer(mock_ollama_cls):
    """When tool_iterations >= MAX, orchestrator strips tool_calls and finalizes."""
    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "role": "assistant",
        "content": "Here is what I found.",
        "tool_calls": [{"function": {"name": "web_search", "arguments": "{}"}}],
    }
    mock_ollama_cls.return_value = mock_client

    state = {
        "message": "test query",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "test query"},
        ],
        "tool_iterations": 5,
    }
    result = orchestrator_node(state)
    assert result["final_response"] == "Here is what I found."
    # tool_calls should be stripped from the response message
    last_msg = result["messages"][-1]
    assert "tool_calls" not in last_msg


@patch.object(graph, "OllamaClient")
def test_orchestrator_max_iterations_uses_fallback_when_no_content(mock_ollama_cls):
    """When model has tool_calls but empty content at max iterations, use fallback."""
    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "web_search", "arguments": "{}"}}],
    }
    mock_ollama_cls.return_value = mock_client

    state = {
        "message": "test",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "test"},
        ],
        "tool_iterations": 5,
    }
    result = orchestrator_node(state)
    assert "unable to complete" in result["final_response"].lower()


# -- Tools node tests --

def test_tools_node_no_tool_calls_returns_early():
    """Guard: tools_node returns unchanged messages if no tool_calls."""
    messages = [{"role": "assistant", "content": "done"}]
    state = {"messages": messages}
    result = tools_node(state)
    assert result["messages"] == messages
    assert "tool_iterations" not in result


@patch.object(graph, "execute_tool", return_value="search results here")
@patch.object(graph, "SearxngClient")
def test_tools_node_generates_tool_call_id(mock_searxng_cls, mock_exec):
    """When tool call has no id, tools_node generates a fallback id."""
    state = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "web_search", "arguments": '{"query": "test"}'}},
            ]},
        ],
        "sources": [],
        "search_count": 0,
        "tool_iterations": 0,
    }
    result = tools_node(state)
    tool_msg = result["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"].startswith("call_")
    assert len(tool_msg["tool_call_id"]) == 13  # "call_" + 8 hex chars


@patch.object(graph, "execute_tool", return_value="result")
@patch.object(graph, "SearxngClient")
def test_tools_node_preserves_existing_tool_call_id(mock_searxng_cls, mock_exec):
    """When tool call has an id, tools_node preserves it."""
    state = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"id": "existing_123", "function": {"name": "web_search", "arguments": "{}"}},
            ]},
        ],
        "sources": [],
        "search_count": 0,
        "tool_iterations": 0,
    }
    result = tools_node(state)
    tool_msg = result["messages"][-1]
    assert tool_msg["tool_call_id"] == "existing_123"


@patch.object(graph, "SearxngClient")
def test_tools_node_malformed_arguments_returns_error(mock_searxng_cls):
    """Malformed JSON arguments produce an error tool message instead of running."""
    state = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "web_search", "arguments": "not valid json{"}},
            ]},
        ],
        "sources": [],
        "search_count": 0,
        "tool_iterations": 0,
    }
    result = tools_node(state)
    tool_msg = result["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert "could not parse" in tool_msg["content"].lower()
    assert "tool_call_id" in tool_msg


@patch.object(graph, "execute_tool", return_value="done")
@patch.object(graph, "SearxngClient")
def test_tools_node_increments_tool_iterations(mock_searxng_cls, mock_exec):
    """Each call to tools_node increments tool_iterations by 1."""
    state = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "web_search", "arguments": "{}"}},
                {"function": {"name": "web_search", "arguments": "{}"}},
            ]},
        ],
        "sources": [],
        "search_count": 0,
        "tool_iterations": 2,
    }
    result = tools_node(state)
    # Two calls in one turn still only increments by 1
    assert result["tool_iterations"] == 3
