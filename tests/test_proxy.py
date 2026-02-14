"""Tests for gateway OpenAI-compatible proxy functions and streaming."""

import json
import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

# Set before gateway import so module-level AGENT_NUM_CTX picks it up.
os.environ.setdefault("AGENT_NUM_CTX", "16384")

# Ensure we have real modules (not mocks from test_graph)
if "clients" in sys.modules and isinstance(sys.modules["clients"], MagicMock):
    del sys.modules["clients"]

# Mock langgraph and graph before importing gateway.py
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

from gateway import (
    _msg_to_dict,
    _openai_messages_to_ollama,
    _ollama_tool_calls_to_openai,
    _build_ollama_options,
    _SENTINEL,
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
        assert opts == {"num_ctx": 16384, "num_predict": 100, "temperature": 0.5, "top_p": 0.9}

    def test_includes_num_ctx_even_with_no_other_params(self):
        req = OpenAIChatRequest()
        opts = _build_ollama_options(req)
        assert opts == {"num_ctx": 16384}


class TestRetrieveModel:

    client = TestClient(app)

    def test_returns_gateway_model(self):
        resp = self.client.get("/v1/models/gateway")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "gateway"
        assert body["object"] == "model"
        assert body["owned_by"] == "local"

    def test_includes_context_window(self):
        resp = self.client.get("/v1/models/gateway")
        assert resp.status_code == 200
        assert resp.json()["context_window"] == 16384

    def test_unknown_model_returns_404(self):
        resp = self.client.get("/v1/models/nonexistent")
        assert resp.status_code == 404
        assert "nonexistent" in resp.json()["detail"]


class TestListModels:

    client = TestClient(app)

    def test_includes_context_window(self):
        resp = self.client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"][0]["context_window"] == 16384


class TestMalformedToolCallsEndpoint:
    """Verify that malformed tool_calls arguments return 400 at the endpoint level."""

    client = TestClient(app)

    def test_invalid_json_in_tool_call_returns_400(self):
        resp = self.client.post("/v1/chat/completions", json={
            "model": "gateway",
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


class TestMalformedStreamChunks:
    """Verify that malformed JSON chunks in the Ollama stream are skipped."""

    def test_malformed_chunk_skipped_and_valid_content_arrives(self):
        lines = [
            "this is not json",
            '{"message": {"content": "hello"}, "done": false}',
            '{"message": {"content": ""}, "done": true}',
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.aiter_lines = mock_aiter_lines

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield mock_resp

        with TestClient(app) as client:
            app.state.http.stream = mock_stream
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gateway",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        events = resp.text
        # Valid content chunk should appear in the SSE stream
        assert '"hello"' in events
        # Stream should terminate cleanly with the SSE done marker
        assert "data: [DONE]" in events
        # Malformed line should not leak into the response
        assert "this is not json" not in events


class TestToolChoiceForwarding:
    """Verify that tool_choice is forwarded to the Ollama payload."""

    def _ollama_ok_response(self):
        """Build a mock httpx.Response that looks like a successful Ollama reply."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        }
        return mock_resp

    def _base_request(self, **extra):
        return {
            "model": "gateway",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "t", "parameters": {}}}],
            **extra,
        }

    @pytest.mark.anyio
    async def test_non_streaming_forwards_tool_choice_string(self):
        mock_http = AsyncMock()
        mock_http.post.return_value = self._ollama_ok_response()

        with TestClient(app) as client:
            app.state.http = mock_http
            resp = client.post(
                "/v1/chat/completions",
                json=self._base_request(tool_choice="required"),
            )

        assert resp.status_code == 200
        payload = mock_http.post.call_args[1]["json"]
        assert payload["tool_choice"] == "required"

    @pytest.mark.anyio
    async def test_non_streaming_forwards_tool_choice_dict(self):
        choice = {"type": "function", "function": {"name": "t"}}
        mock_http = AsyncMock()
        mock_http.post.return_value = self._ollama_ok_response()

        with TestClient(app) as client:
            app.state.http = mock_http
            resp = client.post(
                "/v1/chat/completions",
                json=self._base_request(tool_choice=choice),
            )

        assert resp.status_code == 200
        payload = mock_http.post.call_args[1]["json"]
        assert payload["tool_choice"] == choice

    @pytest.mark.anyio
    async def test_non_streaming_omits_tool_choice_when_absent(self):
        mock_http = AsyncMock()
        mock_http.post.return_value = self._ollama_ok_response()

        with TestClient(app) as client:
            app.state.http = mock_http
            resp = client.post(
                "/v1/chat/completions",
                json=self._base_request(),
            )

        assert resp.status_code == 200
        payload = mock_http.post.call_args[1]["json"]
        assert "tool_choice" not in payload

    @pytest.mark.anyio
    async def test_streaming_forwards_tool_choice(self):
        lines = ['{"message": {"content": ""}, "done": true}']

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.aiter_lines = mock_aiter_lines

        captured_payload = {}

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            yield mock_resp

        with TestClient(app) as client:
            app.state.http.stream = mock_stream
            resp = client.post(
                "/v1/chat/completions",
                json=self._base_request(stream=True, tool_choice="none"),
            )

        assert resp.status_code == 200
        assert captured_payload["tool_choice"] == "none"

    @pytest.mark.anyio
    async def test_streaming_omits_tool_choice_when_absent(self):
        lines = ['{"message": {"content": ""}, "done": true}']

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.aiter_lines = mock_aiter_lines

        captured_payload = {}

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            yield mock_resp

        with TestClient(app) as client:
            app.state.http.stream = mock_stream
            resp = client.post(
                "/v1/chat/completions",
                json=self._base_request(stream=True),
            )

        assert resp.status_code == 200
        assert "tool_choice" not in captured_payload


class TestChatStream:
    """Verify that /chat/stream yields SSE events incrementally via the queue bridge."""

    def _parse_sse_events(self, text):
        """Parse SSE text into a list of (event_type, data) tuples."""
        events = []
        for block in text.strip().split("\n\n"):
            event_type = None
            data = None
            for line in block.split("\n"):
                if line.startswith("event: "):
                    event_type = line[len("event: "):]
                elif line.startswith("data: "):
                    data = line[len("data: "):]
            if event_type is not None:
                events.append((event_type, data))
        return events

    def test_events_arrive_in_order(self):
        """Two node outputs should produce node and response events in order,
        ending with a done event."""
        def mock_stream(*args, **kwargs):
            yield {"orchestrator": {"messages": ["thinking..."]}}
            yield {"orchestrator": {"final_response": "the answer"}}

        import gateway
        original = gateway.agent_graph
        mock_graph = MagicMock()
        mock_graph.stream = mock_stream

        try:
            gateway.agent_graph = mock_graph
            with TestClient(app) as client:
                resp = client.post(
                    "/chat/stream",
                    json={"message": "hello"},
                )
        finally:
            gateway.agent_graph = original

        assert resp.status_code == 200
        events = self._parse_sse_events(resp.text)
        event_types = [e[0] for e in events]

        # First node output has no final_response, so just "node"
        # Second node output has final_response, so "node" then "response"
        assert event_types == ["node", "node", "response", "done"]
        # The response event carries the answer
        response_events = [e for e in events if e[0] == "response"]
        assert response_events[0][1] == "the answer"

    def test_error_propagation(self):
        """An exception from agent_graph.stream should yield an error event."""
        def mock_stream(*args, **kwargs):
            raise RuntimeError("graph exploded")

        import gateway
        original = gateway.agent_graph
        mock_graph = MagicMock()
        mock_graph.stream = mock_stream

        try:
            gateway.agent_graph = mock_graph
            with TestClient(app) as client:
                resp = client.post(
                    "/chat/stream",
                    json={"message": "hello"},
                )
        finally:
            gateway.agent_graph = original

        assert resp.status_code == 200
        events = self._parse_sse_events(resp.text)
        event_types = [e[0] for e in events]
        assert "error" in event_types
        error_data = [e[1] for e in events if e[0] == "error"][0]
        assert "graph exploded" in error_data

    def test_empty_graph_output(self):
        """A graph that yields nothing should produce only a done event."""
        def mock_stream(*args, **kwargs):
            return iter([])

        import gateway
        original = gateway.agent_graph
        mock_graph = MagicMock()
        mock_graph.stream = mock_stream

        try:
            gateway.agent_graph = mock_graph
            with TestClient(app) as client:
                resp = client.post(
                    "/chat/stream",
                    json={"message": "hello"},
                )
        finally:
            gateway.agent_graph = original

        assert resp.status_code == 200
        events = self._parse_sse_events(resp.text)
        assert len(events) == 1
        assert events[0] == ("done", "complete")
