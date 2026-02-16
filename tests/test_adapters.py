"""Tests for the model adapter layer."""

import json
import sys
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from adapters import select_adapter
from adapters.base import ModelAdapter
from adapters.qwen25coder import Qwen25CoderAdapter, _extract_tool_objects
from adapters.qwen3 import Qwen3Adapter


# -- Adapter selection --

class TestSelectAdapter:

    def test_qwen25coder_prefix_match(self):
        adapter = select_adapter("qwen2.5-coder:14b")
        assert isinstance(adapter, Qwen25CoderAdapter)

    def test_qwen25coder_variant_match(self):
        adapter = select_adapter("qwen2.5-coder:7b")
        assert isinstance(adapter, Qwen25CoderAdapter)

    def test_qwen3_prefix_match(self):
        adapter = select_adapter("qwen3:14b")
        assert isinstance(adapter, Qwen3Adapter)

    def test_unknown_model_gets_base(self):
        adapter = select_adapter("llama3:8b")
        assert type(adapter) is ModelAdapter

    def test_results_are_cached(self):
        a1 = select_adapter("test-cache-model:1b")
        a2 = select_adapter("test-cache-model:1b")
        assert a1 is a2

    def test_longest_prefix_wins(self):
        """qwen2.5-coder is longer than a hypothetical qwen2.5 prefix."""
        adapter = select_adapter("qwen2.5-coder:14b")
        assert isinstance(adapter, Qwen25CoderAdapter)


# -- Base adapter passthrough --

class TestBaseAdapter:

    def setup_method(self):
        self.adapter = ModelAdapter()

    def test_normalize_tool_calls_passthrough(self):
        msg = {"content": "hello", "tool_calls": None}
        assert self.adapter.normalize_tool_calls(msg, set()) is msg

    def test_normalize_ai_message_passthrough(self):
        ai = AIMessage(content="hi")
        assert self.adapter.normalize_ai_message(ai) is ai

    def test_inject_tool_guidance_passthrough(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert self.adapter.inject_tool_guidance(msgs, None) is msgs

    def test_llm_kwargs_empty(self):
        assert self.adapter.llm_kwargs() == {}

    def test_filter_hallucinated_removes_bad_names(self):
        calls = [
            {"function": {"name": "web_search", "arguments": {}}},
            {"function": {"name": "invented_tool", "arguments": {}}},
        ]
        kept = self.adapter._filter_hallucinated(calls, {"web_search"})
        assert len(kept) == 1
        assert kept[0]["function"]["name"] == "web_search"

    def test_filter_hallucinated_noop_without_valid_names(self):
        calls = [{"function": {"name": "anything", "arguments": {}}}]
        assert self.adapter._filter_hallucinated(calls, set()) is calls

    def test_normalize_tool_calls_filters_structured_calls(self):
        """Base adapter filters hallucinated names from structured tool_calls."""
        msg = {
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {}}},
                {"function": {"name": "hallucinated", "arguments": {}}},
            ],
        }
        result = self.adapter.normalize_tool_calls(msg, {"read_file"})
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "read_file"


# -- Tool object extraction --

class TestExtractToolObjects:

    def test_fenced_json_block(self):
        text = 'Here is the call:\n```json\n{"name": "web_search", "arguments": {"query": "test"}}\n```'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 1
        assert candidates[0][1]["name"] == "web_search"

    def test_fenced_block_without_json_label(self):
        text = 'Call:\n```\n{"name": "read_file", "arguments": {"path": "/tmp"}}\n```'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 1
        assert candidates[0][1]["name"] == "read_file"

    def test_bare_json_object(self):
        text = 'I will call {"name": "run_command", "arguments": {"command": "ls"}} now.'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 1
        assert candidates[0][1]["name"] == "run_command"

    def test_invalid_json_ignored(self):
        text = '```json\n{broken json}\n```'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 0

    def test_no_name_key_ignored(self):
        text = '```json\n{"tool": "web_search", "args": {}}\n```'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 0

    def test_multiple_candidates(self):
        text = (
            '```json\n{"name": "read_file", "arguments": {"path": "/a"}}\n```\n'
            'Also: {"name": "web_search", "arguments": {"query": "test"}}'
        )
        candidates = _extract_tool_objects(text)
        names = {c[1]["name"] for c in candidates}
        assert names == {"read_file", "web_search"}

    def test_fenced_not_duplicated_as_bare(self):
        """A JSON object inside a fenced block should not also match as bare."""
        text = '```json\n{"name": "web_search", "arguments": {"query": "test"}}\n```'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 1


# -- Qwen25Coder proxy path normalization --

class TestQwen25CoderNormalizeToolCalls:

    def setup_method(self):
        self.adapter = Qwen25CoderAdapter()
        self.valid = {"web_search", "read_file", "run_command"}

    def test_structured_calls_filtered_only(self):
        """When Ollama returns structured tool_calls, just filter names."""
        msg = {
            "content": "",
            "tool_calls": [
                {"function": {"name": "web_search", "arguments": {"query": "hi"}}},
                {"function": {"name": "fake_tool", "arguments": {}}},
            ],
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_recovery_from_fenced_json(self):
        msg = {
            "content": 'Here:\n```json\n{"name": "web_search", "arguments": {"query": "python"}}\n```',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "web_search"
        assert result["tool_calls"][0]["function"]["arguments"] == {"query": "python"}
        # JSON stripped from content
        assert "web_search" not in result["content"]

    def test_recovery_from_bare_json(self):
        msg = {
            "content": 'I will call {"name": "run_command", "arguments": {"command": "ls -la"}} to list files.',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "run_command"

    def test_invalid_name_not_recovered(self):
        """Tool calls with names not in valid_names are skipped."""
        msg = {
            "content": '```json\n{"name": "nonexistent_tool", "arguments": {}}\n```',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert result.get("tool_calls") is None

    def test_no_tools_no_recovery(self):
        """Without valid_names, no recovery is attempted."""
        msg = {
            "content": '{"name": "web_search", "arguments": {"query": "test"}}',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, set())
        assert result.get("tool_calls") is None

    def test_plain_json_in_conversation_not_fabricated(self):
        """JSON objects without a name+arguments structure are not recovered."""
        msg = {
            "content": 'The config is {"database": "postgres", "port": 5432}.',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert result.get("tool_calls") is None

    def test_invalid_json_in_content_ignored(self):
        msg = {
            "content": '```json\n{this is broken json}\n```',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert result.get("tool_calls") is None


# -- Qwen25Coder agent path normalization --

class TestQwen25CoderNormalizeAIMessage:

    def setup_method(self):
        self.adapter = Qwen25CoderAdapter()

    def test_passthrough_when_tool_calls_present(self):
        """If AIMessage already has tool_calls, no recovery needed."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}],
        )
        result = self.adapter.normalize_ai_message(msg)
        assert result is msg

    def test_recovery_from_content(self):
        msg = AIMessage(
            content='```json\n{"name": "web_search", "arguments": {"query": "test"}}\n```',
        )
        result = self.adapter.normalize_ai_message(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "web_search"
        assert result.tool_calls[0]["args"] == {"query": "test"}
        assert "web_search" not in result.content

    def test_no_recovery_from_plain_text(self):
        msg = AIMessage(content="The answer is 42.")
        result = self.adapter.normalize_ai_message(msg)
        assert result is msg
        assert not result.tool_calls

    def test_empty_content_passthrough(self):
        msg = AIMessage(content="")
        result = self.adapter.normalize_ai_message(msg)
        assert result is msg


# -- Qwen25Coder tool guidance --

class TestQwen25CoderToolGuidance:

    def setup_method(self):
        self.adapter = Qwen25CoderAdapter()

    def test_injects_system_message(self):
        msgs = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "web_search"}}]
        result = self.adapter.inject_tool_guidance(msgs, tools)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "web_search" in result[0]["content"]

    def test_no_tools_passthrough(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = self.adapter.inject_tool_guidance(msgs, None)
        assert result is msgs

    def test_empty_tools_passthrough(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = self.adapter.inject_tool_guidance(msgs, [])
        assert result is msgs


# -- Qwen3 adapter --

class TestQwen3Adapter:

    def test_is_model_adapter(self):
        adapter = Qwen3Adapter()
        assert isinstance(adapter, ModelAdapter)

    def test_passthrough_behavior(self):
        adapter = Qwen3Adapter()
        msg = AIMessage(content="hello")
        assert adapter.normalize_ai_message(msg) is msg
        assert adapter.llm_kwargs() == {}
