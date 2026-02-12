"""Tests for graph routing and node logic.

These test the pure Python functions in graph.py without needing
LangGraph or live services by mocking external dependencies.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

# Set before graph import so module-level AGENT_NUM_CTX picks it up.
os.environ.setdefault("AGENT_NUM_CTX", "16384")

# Mock langgraph before importing graph.py
_mock_langgraph = MagicMock()
_mock_langgraph_graph = MagicMock()
_mock_langgraph_graph.END = "__end__"
_mock_langgraph_graph_state = MagicMock()
_mock_langgraph_checkpoint = MagicMock()
_mock_langgraph_checkpoint_memory = MagicMock()
sys.modules["langgraph"] = _mock_langgraph
sys.modules["langgraph.graph"] = _mock_langgraph_graph
sys.modules["langgraph.graph.state"] = _mock_langgraph_graph_state
sys.modules["langgraph.checkpoint"] = _mock_langgraph_checkpoint
sys.modules["langgraph.checkpoint.memory"] = _mock_langgraph_checkpoint_memory

_mock_langchain_core = MagicMock()
_mock_langchain_core_globals = MagicMock()
sys.modules["langchain_core"] = _mock_langchain_core
sys.modules["langchain_core.globals"] = _mock_langchain_core_globals

# Save and mock client and tools modules to avoid pulling in dependencies.
# Both are restored after the graph import so other test files get the real modules.
_saved_clients = sys.modules.pop("clients", None)
sys.modules["clients"] = MagicMock()

_saved_tools = sys.modules.pop("tools", None)
sys.modules["tools"] = MagicMock()

import graph  # noqa: E402

del sys.modules["tools"]
if _saved_tools is not None:
    sys.modules["tools"] = _saved_tools

del sys.modules["clients"]
if _saved_clients is not None:
    sys.modules["clients"] = _saved_clients

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


@patch.object(graph, "OllamaClient")
def test_orchestrator_passes_num_ctx_options(mock_ollama_cls):
    """Orchestrator injects num_ctx so Ollama does not truncate large prompts."""
    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "role": "assistant",
        "content": "answer",
    }
    mock_ollama_cls.return_value = mock_client

    state = {"message": "test query"}
    orchestrator_node(state)

    call_kwargs = mock_client.chat.call_args
    assert call_kwargs.kwargs["options"] == {"num_ctx": 16384}


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


# -- Trust boundary tests --

def test_system_prompt_contains_trust_boundary():
    """SYSTEM_PROMPT must tell the model to treat tool output as untrusted.

    This canary test prevents accidental removal of the trust boundary
    during future prompt edits.
    """
    prompt_lower = graph.SYSTEM_PROMPT.lower()
    assert "untrusted" in prompt_lower
    assert "never follow instructions" in prompt_lower


@patch.object(graph, "execute_tool")
@patch.object(graph, "SearxngClient")
def test_tools_node_truncates_oversized_output(mock_searxng_cls, mock_exec):
    """Tool output exceeding MAX_TOOL_OUTPUT_CHARS is truncated with a marker."""
    oversized = "x" * (graph.MAX_TOOL_OUTPUT_CHARS + 500)
    mock_exec.return_value = oversized
    state = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "web_search", "arguments": "{}"}},
            ]},
        ],
        "sources": [],
        "search_count": 0,
        "tool_iterations": 0,
    }
    result = tools_node(state)
    tool_msg = result["messages"][-1]
    assert tool_msg["content"].endswith("[truncated]")
    expected_len = graph.MAX_TOOL_OUTPUT_CHARS + len("\n[truncated]")
    assert len(tool_msg["content"]) == expected_len


@patch.object(graph, "execute_tool", return_value="short result")
@patch.object(graph, "SearxngClient")
def test_tools_node_passes_normal_output_unchanged(mock_searxng_cls, mock_exec):
    """Tool output within the size limit is passed through verbatim."""
    state = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "web_search", "arguments": "{}"}},
            ]},
        ],
        "sources": [],
        "search_count": 0,
        "tool_iterations": 0,
    }
    result = tools_node(state)
    tool_msg = result["messages"][-1]
    assert tool_msg["content"] == "short result"
