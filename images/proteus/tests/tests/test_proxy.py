"""Tests for Proteus OpenAI-compatible message conversion."""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Ensure we have real modules (not mocks from test_graph)
if "clients" in sys.modules and isinstance(sys.modules["clients"], MagicMock):
    del sys.modules["clients"]

# Mock langgraph and graph before importing proteus.py
_mock_langgraph = MagicMock()
_mock_langgraph_graph = MagicMock()
_mock_langgraph_graph.END = "__end__"
_mock_langgraph_graph_state = MagicMock()
sys.modules.setdefault("langgraph", _mock_langgraph)
sys.modules.setdefault("langgraph.graph", _mock_langgraph_graph)
sys.modules.setdefault("langgraph.graph.state", _mock_langgraph_graph_state)
sys.modules.setdefault("langchain_core", MagicMock())
sys.modules.setdefault("langchain_core.globals", MagicMock())
sys.modules.setdefault("graph", MagicMock())

from proteus import (
    _msg_to_dict,
    OpenAIMessage,
    OpenAIToolCall,
    OpenAIFunctionCall,
)


class TestMsgToDict:

    def test_converts_openai_message(self):
        msg = OpenAIMessage(role="user", content="hello")
        d = _msg_to_dict(msg)
        assert d == {"role": "user", "content": "hello"}

    def test_preserves_dict_passthrough(self):
        d = {"role": "assistant", "content": "hi"}
        assert _msg_to_dict(d) is d

    def test_converts_tool_calls(self):
        msg = OpenAIMessage(
            role="assistant",
            tool_calls=[
                OpenAIToolCall(
                    id="call_1",
                    type="function",
                    function=OpenAIFunctionCall(
                        name="web_search",
                        arguments='{"query": "test"}',
                    ),
                )
            ],
        )
        d = _msg_to_dict(msg)
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["function"]["name"] == "web_search"

    def test_converts_tool_response(self):
        msg = OpenAIMessage(role="tool", content="result", tool_call_id="call_1")
        d = _msg_to_dict(msg)
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_1"
