# agents/coder.py
"""Coder specialist: code analysis and generation.

Placeholder subgraph for the coding assistant. Currently has no file
operation tools, so it acts as a direct-answer coding advisor. File
operation tools (read, write, search) will be added incrementally
as they are built.
"""

import logging

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from config import AGENT_MODEL, AGENT_NUM_CTX, MODEL_ADAPTER, OLLAMA_URL

log = logging.getLogger(__name__)

CODER_PROMPT = (
    "You are a coding specialist. Help the user with code analysis, "
    "generation, debugging, and software design. Write clean, "
    "well-structured code and explain your reasoning.\n"
)

_LLM = ChatOllama(
    model=f"{AGENT_MODEL}-agent",
    base_url=OLLAMA_URL,
    num_ctx=AGENT_NUM_CTX,
    **MODEL_ADAPTER.llm_kwargs(),
)


def agent_node(state: MessagesState) -> dict:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=CODER_PROMPT)] + list(messages)
    response = _LLM.invoke(messages)
    response = MODEL_ADAPTER.normalize_ai_message(response)
    return {"messages": [response]}


def build_coder() -> CompiledStateGraph:
    """Single-node graph with no tools. Direct LLM conversation."""
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()
