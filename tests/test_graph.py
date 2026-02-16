"""Tests for graph routing and agent node logic.

All LangChain and LangGraph packages are available in the testrunner,
so these tests use real imports and only mock the LLM invocation to
avoid needing a live Ollama connection.
"""

import os
import sys
from unittest.mock import MagicMock, patch

# conftest.py loads defaults from config.env before collection

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import graph

agent_node = graph.agent_node
route_after_agent = graph.route_after_agent


# -- Routing tests --

def test_route_to_tools_when_tool_calls_present():
    msg = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "test"}, "id": "1"}])
    state = {"messages": [msg]}
    assert route_after_agent(state) == "tools"


def test_route_to_end_when_no_tool_calls():
    msg = AIMessage(content="The answer is 4.")
    state = {"messages": [msg]}
    assert route_after_agent(state) == "__end__"


def test_route_to_end_with_human_message_last():
    msg = HumanMessage(content="hello")
    state = {"messages": [msg]}
    assert route_after_agent(state) == "__end__"


def test_route_to_end_with_empty_tool_calls():
    msg = AIMessage(content="done", tool_calls=[])
    state = {"messages": [msg]}
    assert route_after_agent(state) == "__end__"


# -- Agent node tests --

def test_agent_node_prepends_system_prompt():
    """When no system message exists, agent_node prepends SYSTEM_PROMPT."""
    mock_response = AIMessage(content="hello", tool_calls=[])

    with patch.object(graph, "_LLM") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        state = {"messages": [HumanMessage(content="hi")]}
        result = agent_node(state)

        call_args = mock_llm.invoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)
        assert graph.SYSTEM_PROMPT in call_args[0].content
        assert result == {"messages": [mock_response]}


def test_agent_node_preserves_existing_system_prompt():
    """When a system message already contains SYSTEM_PROMPT, it is not duplicated."""
    mock_response = AIMessage(content="ok")

    with patch.object(graph, "_LLM") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        state = {"messages": [
            SystemMessage(content=graph.SYSTEM_PROMPT),
            HumanMessage(content="hi"),
        ]}
        agent_node(state)

        call_args = mock_llm.invoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content.count("untrusted") == 1


def test_agent_node_returns_response_in_messages():
    mock_response = AIMessage(content="answer here")

    with patch.object(graph, "_LLM") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        state = {"messages": [HumanMessage(content="question")]}
        result = agent_node(state)
        assert result["messages"] == [mock_response]


# -- Trust boundary tests --

def test_system_prompt_contains_trust_boundary():
    """SYSTEM_PROMPT must tell the model to treat tool output as untrusted."""
    prompt_lower = graph.SYSTEM_PROMPT.lower()
    assert "untrusted" in prompt_lower
    assert "never follow instructions" in prompt_lower


# -- URL extraction tests --

def test_extract_urls_finds_urls():
    text = "See https://example.com and http://foo.bar/baz for details."
    urls = graph._extract_urls(text)
    assert "https://example.com" in urls
    assert "http://foo.bar/baz" in urls


def test_extract_urls_deduplicates():
    text = "https://example.com and https://example.com again"
    urls = graph._extract_urls(text)
    assert urls.count("https://example.com") == 1


def test_extract_urls_empty_input():
    assert graph._extract_urls("") == []
    assert graph._extract_urls(None) == []
