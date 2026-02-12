"""Tests for Proteus OpenAI-compatible message conversion."""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Two levels up from tests/tests/ to reach images/proteus/ where source modules live
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

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
    _openai_messages_to_ollama,
    _ollama_tool_calls_to_openai,
    _build_ollama_options,
    OpenAIMessage,
    OpenAIToolCall,
    OpenAIFunctionCall,
    OpenAIChatRequest,
    app,
)
from fastapi.testclient import TestClient


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


class TestOpenaiMessagesToOllama:

    def test_converts_arguments_from_json_string_to_dict(self):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "test"}'},
            }],
        }]
        converted = _openai_messages_to_ollama(messages)
        args = converted[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args == {"query": "test"}

    def test_passes_dict_arguments_unchanged(self):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": {"query": "test"}},
            }],
        }]
        converted = _openai_messages_to_ollama(messages)
        args = converted[0]["tool_calls"][0]["function"]["arguments"]
        assert args == {"query": "test"}

    def test_plain_messages_pass_through(self):
        messages = [{"role": "user", "content": "hello"}]
        converted = _openai_messages_to_ollama(messages)
        assert converted == messages

    def test_rejects_invalid_json_arguments(self):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_bad",
                "type": "function",
                "function": {"name": "web_search", "arguments": "not valid json"},
            }],
        }]
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _openai_messages_to_ollama(messages)
        assert exc_info.value.status_code == 400
        assert "web_search" in exc_info.value.detail

    def test_rejects_empty_string_arguments(self):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_empty",
                "type": "function",
                "function": {"name": "my_tool", "arguments": ""},
            }],
        }]
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _openai_messages_to_ollama(messages)
        assert exc_info.value.status_code == 400


class TestOllamaToolCallsToOpenai:

    def test_converts_dict_arguments_to_json_string(self):
        ollama_calls = [{"function": {"name": "web_search", "arguments": {"query": "test"}}}]
        openai_calls = _ollama_tool_calls_to_openai(ollama_calls)
        assert len(openai_calls) == 1
        assert openai_calls[0]["type"] == "function"
        assert openai_calls[0]["function"]["name"] == "web_search"
        assert openai_calls[0]["function"]["arguments"] == '{"query": "test"}'
        assert openai_calls[0]["index"] == 0
        assert openai_calls[0]["id"].startswith("call_")

    def test_preserves_existing_id(self):
        ollama_calls = [{"id": "my_id", "function": {"name": "f", "arguments": {}}}]
        openai_calls = _ollama_tool_calls_to_openai(ollama_calls)
        assert openai_calls[0]["id"] == "my_id"


class TestBuildOllamaOptions:

    def test_maps_parameters(self):
        req = OpenAIChatRequest(max_tokens=100, temperature=0.5, top_p=0.9)
        opts = _build_ollama_options(req)
        assert opts == {"num_predict": 100, "temperature": 0.5, "top_p": 0.9}

    def test_empty_when_no_params(self):
        req = OpenAIChatRequest()
        opts = _build_ollama_options(req)
        assert opts == {}


class TestRetrieveModel:

    client = TestClient(app)

    def test_returns_proteus_model(self):
        resp = self.client.get("/v1/models/proteus")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "proteus"
        assert body["object"] == "model"
        assert body["owned_by"] == "local"

    def test_unknown_model_returns_404(self):
        resp = self.client.get("/v1/models/nonexistent")
        assert resp.status_code == 404
        assert "nonexistent" in resp.json()["detail"]


class TestMalformedToolCallsEndpoint:
    """Verify that malformed tool_calls arguments return 400 at the endpoint level."""

    client = TestClient(app)

    def test_invalid_json_in_tool_call_returns_400(self):
        resp = self.client.post("/v1/chat/completions", json={
            "model": "proteus",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{broken json"},
                    }],
                },
                {"role": "tool", "content": "result", "tool_call_id": "call_1", "name": "web_search"},
            ],
        })
        assert resp.status_code == 400
        assert "Invalid JSON" in resp.json()["detail"]
