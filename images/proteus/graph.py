# graph.py
"""LangGraph workflow: orchestrator + tools loop.

Two-node graph where the orchestrator calls Ollama with native tool calling.
If the LLM returns tool_calls, the tools node executes them and loops back.
If the LLM returns a text answer, the graph terminates.
"""

import json
import logging
import os
import re
import time
from typing import Literal
from uuid import uuid4

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.globals import set_debug

from clients import OllamaClient, SearxngClient
from tools import DEFAULT_TOOLS, execute_tool, merge_tools

set_debug(os.getenv("LANGGRAPH_DEBUG", "").lower() in ("1", "true"))
log = logging.getLogger(__name__)
LOG_LANGGRAPH_OUTPUT = os.getenv("LOG_LANGGRAPH_OUTPUT", "true").lower() in ("1", "true")

MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "5"))

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to web search.\n"
    "Use web_search for current, time-sensitive, or uncertain facts.\n"
    "When you use web_search results, cite sources using URLs present in the tool output.\n"
    "For stable, well-known facts, respond directly without web_search.\n"
)


_URL_RE = re.compile(r"https?://[^\s)>\"]+")


def _extract_urls(text: str) -> list[str]:
    if not text:
        return []
    urls = _URL_RE.findall(text)
    # basic cleanup: strip trailing punctuation
    cleaned = []
    for u in urls:
        cleaned.append(u.rstrip(".,;:)]}>\"'"))
    # de-dupe preserving order
    seen = set()
    out = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


class AgentState(TypedDict, total=False):
    # Input
    message: str
    conversation_id: str

    # Optional: caller-provided message history (OpenAI-compatible dicts)
    messages: list[dict]

    # Optional: caller-provided tools (OpenAI tool schema dicts)
    tools: list[dict]

    # Research tracking
    sources: list[str]          # URLs (best-effort) from tool outputs
    search_count: int           # count of web_search calls executed
    tool_iterations: int        # number of tools_node entries

    # Output
    final_response: str


def _ensure_system_prompt(messages: list[dict]) -> list[dict]:
    if messages and messages[0].get("role") == "system":
        # prepend policy to existing system prompt content
        first = dict(messages[0])
        content = str(first.get("content", ""))
        first["content"] = f"{SYSTEM_PROMPT}\n{content}" if content else SYSTEM_PROMPT
        return [first] + list(messages[1:])
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]


def orchestrator_node(state: AgentState) -> dict:
    """Call Ollama with tool schemas and inspect the response."""
    messages = list(state.get("messages", []) or [])

    # First call: initialize from message if needed
    if not messages:
        user_message = state.get("message")
        if not user_message:
            log.error("No message or messages in state")
            return {"final_response": "Error: no input message provided."}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    else:
        # Ensure we always have a system policy prompt at the front
        messages = _ensure_system_prompt(messages)

    model = os.getenv("AGENT_MODEL", "devstral:latest-agent")
    tools = merge_tools(DEFAULT_TOOLS, state.get("tools"))

    ollama = OllamaClient()
    t0 = time.time()
    try:
        response_message = ollama.chat(messages, model=model, tools=tools)
    except Exception as e:
        log.exception("Ollama chat failed")
        return {"final_response": f"Error: model request failed: {e}"}
    elapsed = time.time() - t0

    # Append assistant response
    messages = messages + [response_message]

    tool_iters = state.get("tool_iterations", 0)
    tool_calls = response_message.get("tool_calls", []) or []
    content = response_message.get("content", "") or ""

    updates: dict = {"messages": messages}

    # If budget exhausted, force final answer and prevent tool routing
    if tool_calls and tool_iters >= MAX_TOOL_ITERATIONS:
        log.warning("Max tool iterations (%d) reached, forcing final answer", MAX_TOOL_ITERATIONS)
        # Remove tool_calls from the last message stored in messages to avoid routing to tools
        response_message.pop("tool_calls", None)
        messages[-1] = response_message
        updates["messages"] = messages
        updates["final_response"] = content or (
            "I was unable to complete the request within the allowed number of tool calls."
        )
    elif tool_calls:
        tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
        log.info("Orchestrator model=%s ollama_time=%.1fs tool_calls=%s", model, elapsed, tool_names)
    else:
        log.info("Orchestrator model=%s ollama_time=%.1fs final_answer", model, elapsed)
        updates["final_response"] = content

    if LOG_LANGGRAPH_OUTPUT:
        preview = (updates.get("final_response", "") or "")[:200].replace("\n", " ")
        log.info(
            "LangGraph output node=orchestrator keys=%s tool_calls=%d final_preview=%r",
            sorted(updates.keys()),
            len(tool_calls),
            preview,
        )

    return updates


def tools_node(state: AgentState) -> dict:
    """Execute tool calls from the orchestrator and append results."""
    messages = list(state.get("messages", []) or [])
    sources = list(state.get("sources", []) or [])
    search_count = int(state.get("search_count", 0) or 0)
    tool_iterations = int(state.get("tool_iterations", 0) or 0)

    last_message = messages[-1] if messages else {}
    tool_calls = last_message.get("tool_calls", []) or []
    if not tool_calls:
        log.warning("tools_node entered with no tool_calls in last message")
        return {"messages": messages}

    searxng = SearxngClient()
    tool_iterations += 1

    for call in tool_calls:
        function = call.get("function", {}) or {}
        name = function.get("name", "") or ""
        arguments = function.get("arguments", {}) or {}
        call_id = call.get("id") or f"call_{uuid4().hex[:8]}"

        if not name:
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": "unknown",
                "content": "Error: tool call missing function.name",
            })
            continue

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                log.warning("Malformed tool arguments for %s: %r", name, function.get("arguments"))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": "Error: could not parse arguments as JSON. Provide valid JSON object arguments.",
                })
                continue

        query = arguments.get("query", "")
        t0 = time.time()
        try:
            result_text = execute_tool(name, arguments, searxng)
        except Exception as e:
            log.exception("Tool execution failed: %s", name)
            result_text = f"Error: tool execution failed for {name}: {e}"
        elapsed = time.time() - t0
        log.info("Tool %s query=%r exec_time=%.1fs", name, query, elapsed)

        # Tool result message (OpenAI-compatible)
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": result_text,
        })

        # Track sources best-effort (URLs extracted from tool output)
        if name == "web_search":
            search_count += 1
            urls = _extract_urls(result_text)
            for u in urls:
                if u not in sources:
                    sources.append(u)

    updates = {
        "messages": messages,
        "sources": sources,
        "search_count": search_count,
        "tool_iterations": tool_iterations,
    }

    if LOG_LANGGRAPH_OUTPUT:
        log.info(
            "LangGraph output node=tools keys=%s tool_iterations=%d search_count=%d sources=%d",
            sorted(updates.keys()),
            tool_iterations,
            search_count,
            len(sources),
        )
    return updates


def route_after_orchestrator(state: AgentState) -> Literal["tools", "__end__"]:
    """Route based on whether the orchestrator wants to call tools."""
    if state.get("final_response"):
        return END

    messages = state.get("messages", []) or []
    if not messages:
        return END

    last_message = messages[-1]
    tool_calls = last_message.get("tool_calls", []) or []
    if tool_calls:
        return "tools"
    return END


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "tools": "tools",
            END: END,
        },
    )

    graph.add_edge("tools", "orchestrator")
    return graph.compile()


agent_graph = build_graph()
