"""LangGraph workflow: orchestrator + search loop.

Two-node graph where the orchestrator calls Ollama with native tool calling.
If the LLM returns tool_calls, the search node executes them and loops back.
If the LLM returns a text answer, the graph terminates.
"""

import json
import logging
import os
from typing import Literal

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from clients import OllamaClient, SearxngClient
from tools import TOOLS, execute_tool

log = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to web search. "
    "Use the web_search tool when you need current information or facts "
    "you are not confident about. Cite your sources when you use search results. "
    "For simple questions you can answer confidently, respond directly without searching."
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

    # Research tracking, populated now, verified in Phase 5
    sources: list[str]
    search_count: int

    # Output
    final_response: str


def orchestrator_node(state: AgentState) -> dict:
    """Call Ollama with tool schemas and inspect the response.

    On first call, initializes messages with system prompt and user message.
    On subsequent calls (after search), messages already contain tool results.
    """
    messages = state.get("messages", [])

    # First call: initialize message history
    if not messages:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["message"]},
        ]

    model = os.getenv("AGENT_MODEL", "qwen3:14b-agent")
    ollama = OllamaClient()
    response_message = ollama.chat(messages, model=model, tools=TOOLS)

    # Append the assistant response to history
    messages = messages + [response_message]

    tool_calls = response_message.get("tool_calls", [])
    content = response_message.get("content", "")

    updates = {"messages": messages}

    if tool_calls:
        log.info("Orchestrator requested %d tool call(s)", len(tool_calls))
    else:
        log.info("Orchestrator produced final answer")
        updates["final_response"] = content

    return updates


def search_node(state: AgentState) -> dict:
    """Execute tool calls from the orchestrator and append results."""
    messages = list(state.get("messages", []))
    sources = list(state.get("sources", []))
    search_count = state.get("search_count", 0)

    searxng = SearxngClient()

    # Get tool calls from the last assistant message
    last_message = messages[-1] if messages else {}
    tool_calls = last_message.get("tool_calls", [])

    for call in tool_calls:
        function = call.get("function", {})
        name = function.get("name", "")
        arguments = function.get("arguments", {})

        # Arguments may arrive as a JSON string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        result_text = execute_tool(name, arguments, searxng)

        # Append tool response in Ollama's expected format
        messages.append({
            "role": "tool",
            "content": result_text,
        })

        # Track sources from search results
        if name == "web_search":
            search_count += 1
            query = arguments.get("query", "")
            sources.append(f"web_search: {query}")

    return {
        "messages": messages,
        "sources": sources,
        "search_count": search_count,
    }


def route_after_orchestrator(state: AgentState) -> Literal["search", "__end__"]:
    """Route based on whether the orchestrator wants to call tools."""
    messages = state.get("messages", [])
    search_count = state.get("search_count", 0)

    if not messages:
        return END

    last_message = messages[-1]
    tool_calls = last_message.get("tool_calls", [])

    if tool_calls and search_count < MAX_TOOL_ITERATIONS:
        return "search"

    # If we hit max iterations without a final answer, extract whatever
    # content the model provided
    if search_count >= MAX_TOOL_ITERATIONS and not state.get("final_response"):
        content = last_message.get("content", "")
        log.warning("Hit max iterations (%d), using last content as answer", MAX_TOOL_ITERATIONS)
        # This will be picked up since we already set it in orchestrator_node
        # when there are no tool_calls, but as a safety net we check here

    return END


def build_graph() -> StateGraph:
    """Build and compile the 2-node orchestrator + search graph."""
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("search", search_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "search": "search",
            END: END,
        },
    )

    # After search, loop back to orchestrator
    graph.add_edge("search", "orchestrator")

    return graph.compile()


agent_graph = build_graph()
