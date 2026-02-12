"""Tests for Proteus proxy helpers and session store."""

import os
import sys
import time
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock langgraph before importing proteus.py
_mock_langgraph = MagicMock()
_mock_langgraph_graph = MagicMock()
_mock_langgraph_graph.END = "__end__"
sys.modules.setdefault("langgraph", _mock_langgraph)
sys.modules.setdefault("langgraph.graph", _mock_langgraph_graph)

# Mock graph module to avoid pulling in langgraph dependencies
sys.modules.setdefault("graph", MagicMock())

from proteus import (
    SessionStore,
    _merge_web_search,
    _partition_tool_calls,
    _reconstruct_message_from_chunks,
    _clean_for_ollama,
    _msg_to_dict,
    OpenAIMessage,
    OpenAIToolCall,
    OpenAIFunctionCall,
)
from tools import WEB_SEARCH_TOOL


class TestMergeWebSearch:

    def test_adds_web_search_when_absent(self):
        client_tools = [
            {"type": "function", "function": {"name": "read_file", "parameters": {}}}
        ]
        merged = _merge_web_search(client_tools)
        names = [t["function"]["name"] for t in merged]
        assert "web_search" in names
        assert "read_file" in names
        assert len(merged) == 2

    def test_does_not_duplicate_web_search(self):
        tools_with_search = [WEB_SEARCH_TOOL]
        merged = _merge_web_search(tools_with_search)
        web_count = sum(1 for t in merged if t["function"]["name"] == "web_search")
        assert web_count == 1

    def test_empty_tools_adds_web_search(self):
        merged = _merge_web_search(None)
        assert len(merged) == 1
        assert merged[0]["function"]["name"] == "web_search"

    def test_does_not_mutate_input(self):
        original = [{"type": "function", "function": {"name": "foo", "parameters": {}}}]
        original_len = len(original)
        _merge_web_search(original)
        assert len(original) == original_len


class TestPartitionToolCalls:

    def test_separates_web_and_client_calls(self):
        calls = [
            {"id": "1", "function": {"name": "web_search", "arguments": '{"query": "test"}'}},
            {"id": "2", "function": {"name": "read_file", "arguments": '{"path": "/tmp"}'}},
            {"id": "3", "function": {"name": "web_search", "arguments": '{"query": "other"}'}},
        ]
        web, client = _partition_tool_calls(calls)
        assert len(web) == 2
        assert len(client) == 1
        assert client[0]["id"] == "2"

    def test_all_web_search(self):
        calls = [
            {"id": "1", "function": {"name": "web_search", "arguments": '{"query": "q"}'}},
        ]
        web, client = _partition_tool_calls(calls)
        assert len(web) == 1
        assert len(client) == 0

    def test_all_client_calls(self):
        calls = [
            {"id": "1", "function": {"name": "list_files", "arguments": "{}"}},
        ]
        web, client = _partition_tool_calls(calls)
        assert len(web) == 0
        assert len(client) == 1

    def test_empty_calls(self):
        web, client = _partition_tool_calls([])
        assert web == []
        assert client == []


class TestReconstructMessageFromChunks:

    def test_text_only_chunks(self):
        chunks = [
            {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        msg = _reconstruct_message_from_chunks(chunks)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello world"
        assert "tool_calls" not in msg

    def test_tool_call_chunks(self):
        chunks = [
            {"choices": [{"delta": {"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_abc", "type": "function",
                 "function": {"name": "web_search", "arguments": ""}}
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"query":'}}
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ' "test"}'}}
            ]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        msg = _reconstruct_message_from_chunks(chunks)
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["function"]["name"] == "web_search"
        assert tc["function"]["arguments"] == '{"query": "test"}'

    def test_multiple_tool_calls(self):
        chunks = [
            {"choices": [{"delta": {"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "web_search", "arguments": '{"query": "a"}'}},
                {"index": 1, "id": "call_2", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "/tmp"}'}},
            ]}}]},
        ]
        msg = _reconstruct_message_from_chunks(chunks)
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["function"]["name"] == "web_search"
        assert msg["tool_calls"][1]["function"]["name"] == "read_file"

    def test_empty_chunks(self):
        msg = _reconstruct_message_from_chunks([])
        assert msg["role"] == "assistant"
        assert "content" not in msg
        assert "tool_calls" not in msg


class TestCleanForOllama:

    def test_strips_internal_fields(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "results", "_internal": True, "tool_call_id": "c1"},
            {"role": "assistant", "content": "answer", "_internal": True},
        ]
        cleaned = _clean_for_ollama(messages)
        for m in cleaned:
            assert "_internal" not in m
        assert cleaned[0]["content"] == "hi"
        assert cleaned[1]["content"] == "results"
        assert cleaned[1]["tool_call_id"] == "c1"


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


class TestSessionStore:

    def test_no_match_for_single_message(self):
        store = SessionStore()
        messages = [{"role": "user", "content": "hello"}]
        sid, augmented = store.find(messages)
        assert sid is None
        assert augmented is messages

    def test_save_and_find(self):
        store = SessionStore()
        # Simulate a conversation
        first_exchange = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        store.save("session1", first_exchange)

        # Client sends second message with history
        client_messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
        ]
        sid, augmented = store.find(client_messages)
        assert sid == "session1"
        assert len(augmented) == 3
        assert augmented[-1]["content"] == "how are you"

    def test_internal_messages_excluded_from_matching(self):
        store = SessionStore()
        # Stored session has internal web_search messages
        stored = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "_internal": True,
             "tool_calls": [{"function": {"name": "web_search"}}]},
            {"role": "tool", "content": "search results", "_internal": True},
            {"role": "assistant", "content": "hi there"},
        ]
        store.save("session1", stored)

        # Client only sees visible messages
        client_messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "what next"},
        ]
        sid, augmented = store.find(client_messages)
        assert sid == "session1"
        # Augmented should include the internal messages plus the new user message
        assert len(augmented) == 5
        assert augmented[-1]["content"] == "what next"

    def test_no_match_for_different_history(self):
        store = SessionStore()
        store.save("session1", [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        client_messages = [
            {"role": "user", "content": "different"},
            {"role": "assistant", "content": "conversation"},
            {"role": "user", "content": "new input"},
        ]
        sid, augmented = store.find(client_messages)
        assert sid is None

    def test_eviction_on_max_sessions(self):
        store = SessionStore()
        for i in range(105):
            store.save(f"s{i}", [{"role": "user", "content": f"msg{i}"}])
        assert len(store._sessions) == 100

    def test_ttl_expiration(self):
        store = SessionStore()
        store.save("old", [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
        ])
        # Manually expire the session
        store._sessions["old"]["ts"] = time.time() - 2000

        client_messages = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "new"},
        ]
        sid, augmented = store.find(client_messages)
        assert sid is None
