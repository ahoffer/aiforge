"""Adapter for qwen2.5-coder family.

qwen2.5-coder emits tool calls as JSON text in the content field
instead of using structured tool_calls. This adapter recovers those
calls via regex, validates names against the client's tool list, and
strips the recovered JSON from the content so downstream consumers
see clean text plus proper tool_calls.
"""

import json
import logging
import re

from langchain_core.messages import AIMessage

from adapters.base import ModelAdapter


log = logging.getLogger(__name__)

# Fenced JSON block: ```json\n{...}\n``` or ```\n{...}\n```
_FENCED_JSON = re.compile(
    r"```(?:json)?\s*\n(\{.*?\})\s*\n```",
    re.DOTALL,
)

# Bare JSON object with a "name" key, sitting on its own
_BARE_TOOL_JSON = re.compile(
    r'(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})',
    re.DOTALL,
)

# Prompt steering for models that need explicit tool selection guidance
_TOOL_GUIDANCE_TEMPLATE = (
    "You have these tools: {tool_names}. "
    "For grep, find, sed, awk, or any shell command, use run_command. "
    "list_files only lists directory contents like ls. "
    "Use web_search for current events, real-time information, or "
    "whenever the user explicitly asks to search the web. "
    "Do not use web_search for well-known facts or concepts unless "
    "the user requests a search. "
    "When the user asks for multiple actions, call all relevant tools "
    "in one response."
)


def _extract_tool_objects(text: str) -> list[dict]:
    """Pull candidate tool-call dicts from free text.

    Tries fenced code blocks first, then bare JSON objects. Returns
    parsed dicts that have both "name" and "arguments" keys.
    """
    candidates = []
    for match in _FENCED_JSON.finditer(text):
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and "name" in obj:
                candidates.append((match.group(0), obj))
        except (json.JSONDecodeError, ValueError):
            continue

    for match in _BARE_TOOL_JSON.finditer(text):
        raw = match.group(1)
        # Skip if already captured inside a fenced block
        if any(raw in fenced for fenced, _ in candidates):
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "name" in obj:
                candidates.append((raw, obj))
        except (json.JSONDecodeError, ValueError):
            continue

    return candidates


class Qwen25CoderAdapter(ModelAdapter):
    """Recovers tool calls that qwen2.5-coder embeds in content text."""

    def normalize_tool_calls(self, ollama_msg: dict,
                             valid_names: set[str]) -> dict:
        """Recover tool_calls from content when the model skips structured output."""
        tool_calls = ollama_msg.get("tool_calls") or []
        content = ollama_msg.get("content", "") or ""

        # If the model already produced structured tool_calls, just filter
        if tool_calls:
            ollama_msg = dict(ollama_msg)
            ollama_msg["tool_calls"] = self._filter_hallucinated(
                tool_calls, valid_names)
            return ollama_msg

        # No structured calls, try to recover from content
        if not content or not valid_names:
            return ollama_msg

        candidates = _extract_tool_objects(content)
        if not candidates:
            return ollama_msg

        recovered = []
        strips = []
        for raw_text, obj in candidates:
            name = obj.get("name", "")
            if name not in valid_names:
                log.debug("recovered tool name=%r not in valid set, skipping", name)
                continue
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    continue
            recovered.append({"function": {"name": name, "arguments": args}})
            strips.append(raw_text)

        if not recovered:
            return ollama_msg

        # Strip recovered JSON from content
        cleaned = content
        for s in strips:
            cleaned = cleaned.replace(s, "")
        cleaned = cleaned.strip()

        log.info("recovered %d tool call(s) from content text", len(recovered))
        result = dict(ollama_msg)
        result["tool_calls"] = recovered
        result["content"] = cleaned
        return result

    def normalize_ai_message(self, message: AIMessage) -> AIMessage:
        """Recover tool_calls from AIMessage content on the agent path."""
        if message.tool_calls:
            return message

        content = message.content or ""
        if not content:
            return message

        candidates = _extract_tool_objects(content)
        if not candidates:
            return message

        # On the agent path we accept all parseable tool calls since
        # bind_tools already constrains the schema. LangChain ToolNode
        # will reject unknown names.
        tool_calls = []
        strips = []
        for raw_text, obj in candidates:
            name = obj.get("name", "")
            if not name:
                continue
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    continue
            tool_calls.append({
                "name": name,
                "args": args,
                "id": f"recovered_{len(tool_calls)}",
            })
            strips.append(raw_text)

        if not tool_calls:
            return message

        cleaned = content
        for s in strips:
            cleaned = cleaned.replace(s, "")
        cleaned = cleaned.strip()

        log.info("recovered %d tool call(s) from AIMessage content", len(tool_calls))
        return AIMessage(content=cleaned, tool_calls=tool_calls)

    def inject_tool_guidance(self, ollama_messages: list[dict],
                             tools: list[dict] | None) -> list[dict]:
        """Prepend tool selection guidance for models that need steering."""
        if not tools:
            return ollama_messages
        tool_names = [
            (t.get("function") or {}).get("name", "")
            for t in tools if t.get("type") == "function"
        ]
        tool_names = [n for n in tool_names if n]
        if not tool_names:
            return ollama_messages
        guidance = {"role": "system", "content": _TOOL_GUIDANCE_TEMPLATE.format(
            tool_names=", ".join(tool_names))}
        return [guidance] + ollama_messages
