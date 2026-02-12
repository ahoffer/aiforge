"""LangGraph workflow: orchestrator + tools loop.

Two-node graph where the orchestrator calls Ollama with native tool calling.
If the LLM returns tool_calls, the tools node executes them and loops back.
If the LLM returns a text answer, the graph terminates.
"""

import json
import logging
import os
import time
from typing import Literal
from uuid import uuid4

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from langchain_core.globals import set_debug

from clients import OllamaClient, SearxngClient
from tools import TOOLS, execute_tool

set_debug(os.getenv("LANGGRAPH_DEBUG", "").lower() in ("1", "true"))
log = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to web search. "
    "Use web_search for current information. Cite sources from search results. "
    "For simple questions, respond directly."
)


class AgentState(TypedDict, total=False):
    """Shared state for the agent graph.

    Fields use total=False so nodes only return the fields they modify.
    Designed to grow with later phases without breaking existing nodes.
    """
    # Input
    message: str
    conversation_id: str

    # Ollama message history for /api/chat
    messages: list[dict]

    # Research tracking
    sources: list[str]
    search_count: int
    tool_iterations: int

    # Output
    final_response: str


def orchestrator_node(state: AgentState) -> dict:
    """Call Ollama with tool schemas and inspect the response.

    On first call, initializes messages with system prompt and user message.
    On subsequent calls (after tools), messages already contain tool results.
    """
    messages = state.get("messages", [])

    # First call: initialize message history
    if not messages:
        user_message = state.get("message")
        if not user_message:
            log.error("No message or messages in state")
            return {"final_response": "Error: no input message provided."}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    model = os.getenv("AGENT_MODEL", "devstral:latest-agent")
    ollama = OllamaClient()
    t0 = time.time()
    response_message = ollama.chat(messages, model=model, tools=TOOLS)
    elapsed = time.time() - t0

    messages = messages + [response_message]

    tool_calls = response_message.get("tool_calls", [])
    content = response_message.get("content", "")

    updates = {"messages": messages}

    # If the model wants more tool calls but we have exhausted the
    # iteration budget, discard the tool_calls and finalize instead.
    tool_iters = state.get("tool_iterations", 0)
    if tool_calls and tool_iters >= MAX_TOOL_ITERATIONS:
        log.warning("Max tool iterations (%d) reached, forcing final answer",
                     MAX_TOOL_ITERATIONS)
        response_message.pop("tool_calls", None)
        fallback = content or "I was unable to complete the request within the allowed number of tool calls."
        updates["final_response"] = fallback
    elif tool_calls:
        tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
        log.info("Orchestrator model=%s ollama_time=%.1fs tool_calls=%s",
                 model, elapsed, tool_names)
    else:
        log.info("Orchestrator model=%s ollama_time=%.1fs final_answer", model, elapsed)
        updates["final_response"] = content

    return updates


def tools_node(state: AgentState) -> dict:
    """Execute tool calls from the orchestrator and append results."""
    messages = list(state.get("messages", []))
    sources = list(state.get("sources", []))
    search_count = state.get("search_count", 0)
    tool_iterations = state.get("tool_iterations", 0)

    # Guard against unexpected entry with no tool_calls
    last_message = messages[-1] if messages else {}
    tool_calls = last_message.get("tool_calls", [])
    if not tool_calls:
        log.warning("tools_node entered with no tool_calls in last message")
        return {"messages": messages}

    searxng = SearxngClient()
    tool_iterations += 1

    for call in tool_calls:
        function = call.get("function", {})
        name = function.get("name", "")
        arguments = function.get("arguments", {})
        call_id = call.get("id") or f"call_{uuid4().hex[:8]}"

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                log.warning("Malformed tool arguments for %s: %r", name, function.get("arguments"))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "Error: could not parse arguments as JSON",
                })
                continue

        query = arguments.get("query", "")
        t0 = time.time()
        result_text = execute_tool(name, arguments, searxng)
        elapsed = time.time() - t0
        log.info("Tool %s query=%r exec_time=%.1fs", name, query, elapsed)

        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": result_text,
        })

        if name == "web_search":
            search_count += 1
            sources.append(f"web_search: {query}")

    return {
        "messages": messages,
        "sources": sources,
        "search_count": search_count,
        "tool_iterations": tool_iterations,
    }


def route_after_orchestrator(state: AgentState) -> Literal["tools", "__end__"]:
    """Route based on whether the orchestrator wants to call tools."""
    messages = state.get("messages", [])

    if not messages:
        return END

    last_message = messages[-1]
    tool_calls = last_message.get("tool_calls", [])

    if tool_calls:
        return "tools"

    return END


def build_graph() -> CompiledStateGraph:
    """Build and compile the 2-node orchestrator + tools graph."""
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

    # After tools, loop back to orchestrator
    graph.add_edge("tools", "orchestrator")

    return graph.compile()


agent_graph = build_graph()
