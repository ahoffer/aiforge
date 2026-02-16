"""Tests for gateway OpenAI-compatible proxy functions and streaming."""

import json
import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

# conftest.py loads defaults from config.env before collection

from langchain_core.messages import AIMessage, HumanMessage

from gateway import (
    _msg_to_dict,
    _openai_messages_to_ollama,
    _ollama_tool_calls_to_openai,
    _build_ollama_options,
    _estimate_tokens,
    _trim_context,
    _SENTINEL,
    OpenAIMessage,
    OpenAIToolCall,
    OpenAIFunctionCall,
    OpenAIChatRequest,
    app,
)
from config import AGENT_NUM_CTX
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
        assert opts == {"num_ctx": AGENT_NUM_CTX, "num_predict": 100, "temperature": 0.5, "top_p": 0.9}

    def test_includes_num_ctx_even_with_no_other_params(self):
        req = OpenAIChatRequest()
        opts = _build_ollama_options(req)
        assert opts == {"num_ctx": AGENT_NUM_CTX}


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
        assert resp.json()["context_window"] == AGENT_NUM_CTX

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
        assert body["data"][0]["context_window"] == AGENT_NUM_CTX


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

    def test_tokens_arrive_incrementally(self):
        """stream_mode="messages" yields (chunk, metadata) tuples.
        AIMessage chunks with content become token events."""
        chunk1 = AIMessage(content="the ")
        chunk2 = AIMessage(content="answer")
        meta = {"langgraph_node": "agent"}

        def mock_stream(*args, **kwargs):
            yield (chunk1, meta)
            yield (chunk2, meta)

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

        assert event_types == ["token", "token", "done"]
        token_data = [e[1] for e in events if e[0] == "token"]
        assert token_data == ["the ", "answer"]

    def test_tool_call_chunks_suppressed(self):
        """AIMessage chunks with tool_calls should not produce token events."""
        thinking = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "test"}, "id": "1"}])
        answer = AIMessage(content="found it")
        meta = {"langgraph_node": "agent"}

        def mock_stream(*args, **kwargs):
            yield (thinking, meta)
            yield (answer, meta)

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
        assert event_types == ["token", "done"]
        assert events[0][1] == "found it"

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


class TestEstimateTokens:

    def test_simple_message(self):
        messages = [{"role": "user", "content": "abcd" * 10}]
        assert _estimate_tokens(messages) == 10

    def test_empty_messages(self):
        assert _estimate_tokens([]) == 0

    def test_includes_tool_call_arguments(self):
        messages = [
            {"role": "assistant", "tool_calls": [
                {"function": {"arguments": '{"query": "abcd" }'}}
            ]},
        ]
        estimate = _estimate_tokens(messages)
        assert estimate > 0


class TestTrimContext:

    def _make_messages(self, tool_content_size=200, tool_count=5):
        """Build a conversation with a system prompt, user/assistant pairs,
        and tool result messages large enough to exceed a given budget."""
        msgs = [{"role": "system", "content": "You are a helper."}]
        for i in range(tool_count):
            msgs.append({"role": "user", "content": f"question {i}"})
            msgs.append({
                "role": "assistant",
                "tool_calls": [{"function": {"name": "web_search", "arguments": "{}"}}],
            })
            msgs.append({
                "role": "tool",
                "content": "x" * tool_content_size,
                "tool_call_id": f"call_{i}",
                "name": "web_search",
            })
        msgs.append({"role": "user", "content": "final question"})
        return msgs

    def test_under_budget_passes_through(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result = _trim_context(msgs, 10000)
        assert result == msgs

    def test_over_budget_trims_oldest_tool_messages(self):
        msgs = self._make_messages(tool_content_size=400, tool_count=5)
        total_before = _estimate_tokens(msgs)
        budget = total_before // 2
        result = _trim_context(msgs, budget)
        trimmed_content = [m for m in result if m.get("content") == "[trimmed]"]
        assert len(trimmed_content) > 0
        assert _estimate_tokens(result) < total_before

    def test_system_prompt_never_trimmed(self):
        msgs = self._make_messages(tool_content_size=400, tool_count=5)
        budget = 50
        result = _trim_context(msgs, budget)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helper."

    def test_last_three_user_assistant_pairs_protected(self):
        msgs = self._make_messages(tool_content_size=400, tool_count=6)
        budget = 50
        result = _trim_context(msgs, budget)
        # Last message is a user message, should be preserved
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == "final question"

    def test_empty_messages_returns_empty(self):
        assert _trim_context([], 100) == []

    def test_single_oversized_message_returns_it(self):
        msgs = [{"role": "user", "content": "x" * 10000}]
        result = _trim_context(msgs, 10)
        assert len(result) == 1
        assert result[0]["content"] == "x" * 10000

    def test_zero_budget_does_not_crash(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result = _trim_context(msgs, 0)
        assert result is not None


# -- Helpers for multi-turn test data --

def _make_multi_turn_conversation(turns, tool_result_chars=200):
    """Build a realistic OpenAI-format conversation with tool calling turns.

    Each turn consists of a user message, an assistant message with a
    tool_call, and a tool result message. The tool result content is
    padded to tool_result_chars so callers can control the token budget
    pressure.
    """
    msgs = [{"role": "system", "content": "You are a coding assistant."}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"Turn {i} question about the project"})
        call_id = f"call_{i:04d}"
        msgs.append({
            "role": "assistant",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": f"/src/file_{i}.py"}),
                },
            }],
        })
        msgs.append({
            "role": "tool",
            "content": f"# file_{i}.py\n" + "x" * tool_result_chars,
            "tool_call_id": call_id,
            "name": "read_file",
        })
    msgs.append({"role": "user", "content": "Now summarize everything."})
    return msgs


class TestTrimContextIntegrity:
    """Verify trimming preserves tool message structure required by Ollama."""

    def test_trimmed_tool_messages_retain_ids(self):
        """Every trimmed tool message must keep tool_call_id and name."""
        msgs = _make_multi_turn_conversation(turns=10, tool_result_chars=600)
        budget = _estimate_tokens(msgs) // 3
        result = _trim_context(msgs, budget)

        for msg in result:
            if msg.get("role") != "tool":
                continue
            assert "tool_call_id" in msg, "trimmed tool message lost tool_call_id"
            assert msg["tool_call_id"], "tool_call_id is empty"
            assert "name" in msg, "trimmed tool message lost name"
            assert msg["name"], "name is empty"

    def test_full_pipeline_round_trip(self):
        """Messages survive _msg_to_dict, _openai_messages_to_ollama, _trim_context."""
        openai_msgs = [
            OpenAIMessage(role="user", content="Read the config file"),
            OpenAIMessage(
                role="assistant",
                tool_calls=[OpenAIToolCall(
                    id="call_rt",
                    type="function",
                    function=OpenAIFunctionCall(
                        name="read_file",
                        arguments='{"path": "/etc/config.yaml"}',
                    ),
                )],
            ),
            OpenAIMessage(role="tool", content="key: value\n" * 100,
                          tool_call_id="call_rt", name="read_file"),
            OpenAIMessage(role="user", content="What does it say?"),
        ]

        dicts = [_msg_to_dict(m) for m in openai_msgs]
        ollama = _openai_messages_to_ollama(dicts)
        trimmed = _trim_context(ollama, max_tokens=50)

        for msg in trimmed:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    assert isinstance(args, dict), "arguments should be dict for Ollama"
            if msg.get("role") == "tool":
                assert "tool_call_id" in msg

    def test_assistant_tool_calls_never_orphaned(self):
        """When a tool result is trimmed, its assistant message must remain."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "tool_calls": [{
                "function": {"name": "read_file", "arguments": "{}"},
            }]},
            {"role": "tool", "content": "x" * 2000,
             "tool_call_id": "call_0", "name": "read_file"},
            {"role": "user", "content": "followup"},
            {"role": "assistant", "content": "here is the answer"},
            {"role": "user", "content": "thanks"},
        ]
        result = _trim_context(msgs, max_tokens=30)

        trimmed_tools = [m for m in result if m.get("content") == "[trimmed]"]
        if trimmed_tools:
            assistant_with_calls = [
                m for m in result
                if m.get("role") == "assistant" and m.get("tool_calls")
            ]
            assert len(assistant_with_calls) >= 1, \
                "assistant message with tool_calls was dropped but tool result remains"

    def test_multiple_tool_calls_per_turn(self):
        """Three tool_calls in one assistant message, three results.
        Older turn trimmed, newer turn intact."""
        msgs = [{"role": "system", "content": "sys"}]

        # Old turn with 3 tool calls
        msgs.append({"role": "user", "content": "old question"})
        old_calls = []
        for j in range(3):
            old_calls.append({
                "id": f"call_old_{j}",
                "type": "function",
                "function": {"name": f"tool_{j}", "arguments": "{}"},
            })
        msgs.append({"role": "assistant", "tool_calls": old_calls})
        for j in range(3):
            msgs.append({
                "role": "tool",
                "content": "old_data_" + "x" * 800,
                "tool_call_id": f"call_old_{j}",
                "name": f"tool_{j}",
            })

        # Recent turn with 3 tool calls
        msgs.append({"role": "user", "content": "new question"})
        new_calls = []
        for j in range(3):
            new_calls.append({
                "id": f"call_new_{j}",
                "type": "function",
                "function": {"name": f"tool_{j}", "arguments": "{}"},
            })
        msgs.append({"role": "assistant", "tool_calls": new_calls})
        for j in range(3):
            msgs.append({
                "role": "tool",
                "content": "new_data_" + "y" * 800,
                "tool_call_id": f"call_new_{j}",
                "name": f"tool_{j}",
            })
        msgs.append({"role": "user", "content": "summarize"})

        # Budget high enough that trimming the 3 old results suffices,
        # but low enough that trimming must fire
        old_tool_tokens = sum(
            _estimate_tokens([m]) for m in msgs
            if m.get("tool_call_id", "").startswith("call_old_")
        )
        budget = _estimate_tokens(msgs) - old_tool_tokens + 10
        result = _trim_context(msgs, budget)

        old_trimmed = [
            m for m in result
            if m.get("role") == "tool"
            and m.get("tool_call_id", "").startswith("call_old_")
            and m.get("content") == "[trimmed]"
        ]
        new_intact = [
            m for m in result
            if m.get("role") == "tool"
            and m.get("tool_call_id", "").startswith("call_new_")
            and m.get("content") != "[trimmed]"
        ]
        assert len(old_trimmed) == 3, f"expected 3 old trimmed, got {len(old_trimmed)}"
        assert len(new_intact) == 3, f"expected 3 new intact, got {len(new_intact)}"

    def test_realistic_payload_sizes(self):
        """Forgetools-sized payloads trigger trimming within the configured budget."""
        budget = int(AGENT_NUM_CTX * 0.75)
        # Each round contributes ~(size/4) tokens from tool content alone.
        # Build enough rounds so total comfortably exceeds the trim budget.
        target_tokens = budget * 2
        per_round_chars = 6000
        rounds = (target_tokens * 4) // per_round_chars + 1
        msgs = [{"role": "system", "content": "You are a coding assistant."}]
        for i in range(rounds):
            msgs.append({"role": "user", "content": f"Show me file {i}"})
            call_id = f"call_real_{i}"
            msgs.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": f"/src/mod{i}.py"})},
                }],
            })
            size = 4000 if i % 2 == 0 else 2000
            msgs.append({
                "role": "tool",
                "content": "x" * size,
                "tool_call_id": call_id,
                "name": "read_file",
            })
        msgs.append({"role": "user", "content": "final question"})

        before = _estimate_tokens(msgs)
        assert before > budget, "test setup: messages must exceed budget"

        result = _trim_context(msgs, budget)
        after = _estimate_tokens(result)
        assert after < budget, f"trimming failed: {after} >= {budget}"


class TestProxyMultiTurn:
    """Verify multi-turn tool conversations flow correctly through the proxy endpoint."""

    def _ollama_text_response(self, content="Here is the answer."):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": content},
            "done": True,
        }
        return mock_resp

    def _ollama_tool_response(self, tool_calls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            },
            "done": True,
        }
        return mock_resp

    def _forgetools_schemas(self):
        """OpenAI-format schemas matching the 6 forgetools."""
        return [
            {"type": "function", "function": {
                "name": "read_file",
                "description": "Read a file and return its contents with line numbers",
                "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]},
            }},
            {"type": "function", "function": {
                "name": "list_directory",
                "description": "List files and directories at the given path",
                "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
                    "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "search_files",
                "description": "Search file contents for a regex pattern",
                "parameters": {"type": "object",
                    "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["pattern"]},
            }},
            {"type": "function", "function": {
                "name": "run_command",
                "description": "Run a shell command and return output",
                "parameters": {"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]},
            }},
            {"type": "function", "function": {
                "name": "web_search",
                "description": "Search the web for current information",
                "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]},
            }},
        ]

    @pytest.mark.anyio
    async def test_turn2_tool_results_forwarded(self):
        """Tool results in turn 2 reach Ollama with arguments as dicts."""
        mock_http = AsyncMock()
        mock_http.post.return_value = self._ollama_text_response()

        messages = [
            {"role": "user", "content": "Read /etc/hostname"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "/etc/hostname"}'},
                }],
            },
            {"role": "tool", "content": "gateway-node\n",
             "tool_call_id": "call_abc", "name": "read_file"},
        ]

        with TestClient(app) as client:
            app.state.http = mock_http
            resp = client.post("/v1/chat/completions", json={
                "model": "gateway",
                "messages": messages,
                "tools": [self._forgetools_schemas()[0]],
            })

        assert resp.status_code == 200
        payload = mock_http.post.call_args[1]["json"]
        ollama_msgs = payload["messages"]

        # Arguments converted from JSON string to dict for Ollama
        assistant_msg = [m for m in ollama_msgs if m.get("tool_calls")][0]
        args = assistant_msg["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict), f"expected dict, got {type(args)}"

        # tool_call_id preserved on tool result
        tool_msg = [m for m in ollama_msgs if m.get("role") == "tool"][0]
        assert tool_msg["tool_call_id"] == "call_abc"

    @pytest.mark.anyio
    async def test_goose_exact_message_format(self):
        """Goose sends all 6 forgetools schemas. Gateway forwards valid messages."""
        mock_http = AsyncMock()
        mock_http.post.return_value = self._ollama_tool_response([{
            "function": {"name": "read_file", "arguments": {"path": "/src/main.py"}},
        }])

        with TestClient(app) as client:
            app.state.http = mock_http
            resp = client.post("/v1/chat/completions", json={
                "model": "gateway",
                "messages": [{"role": "user", "content": "Read the main source file"}],
                "tools": self._forgetools_schemas(),
            })

        assert resp.status_code == 200
        payload = mock_http.post.call_args[1]["json"]
        assert len(payload["tools"]) == 6
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    @pytest.mark.anyio
    async def test_tool_calls_response_converted(self):
        """Ollama dict arguments become JSON strings in the OpenAI response."""
        mock_http = AsyncMock()
        mock_http.post.return_value = self._ollama_tool_response([{
            "function": {
                "name": "list_directory",
                "arguments": {"path": "/src", "recursive": True},
            },
        }])

        with TestClient(app) as client:
            app.state.http = mock_http
            resp = client.post("/v1/chat/completions", json={
                "model": "gateway",
                "messages": [{"role": "user", "content": "List source files"}],
                "tools": self._forgetools_schemas(),
            })

        assert resp.status_code == 200
        body = resp.json()
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        tc = choice["message"]["tool_calls"][0]
        assert isinstance(tc["function"]["arguments"], str)
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed["path"] == "/src"

    @pytest.mark.anyio
    async def test_streaming_multi_turn(self):
        """Streaming turn 2 with tool results produces tool_calls chunk and DONE."""
        lines = [
            '{"message": {"role": "assistant", "content": "", '
            '"tool_calls": [{"function": {"name": "read_file", '
            '"arguments": {"path": "/tmp/x"}}}]}, "done": true}',
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.aiter_lines = mock_aiter_lines

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield mock_resp

        messages = [
            {"role": "user", "content": "Read /tmp/x"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_s1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
                }],
            },
            {"role": "tool", "content": "file contents here",
             "tool_call_id": "call_s1", "name": "read_file"},
            {"role": "user", "content": "What does it say?"},
        ]

        with TestClient(app) as client:
            app.state.http.stream = mock_stream
            resp = client.post("/v1/chat/completions", json={
                "model": "gateway",
                "messages": messages,
                "tools": [self._forgetools_schemas()[0]],
                "stream": True,
            })

        assert resp.status_code == 200
        assert "tool_calls" in resp.text
        assert "data: [DONE]" in resp.text
