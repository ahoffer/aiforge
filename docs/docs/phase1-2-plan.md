# Simplify Agent to Orchestrator + Search (Phase 1+2)

## Context

The current agent has 10 LangGraph nodes (interpreter, decision, research, synthesis, critic, ingest, resolve_context, revise, ingest_response, answer), 3 conditional routing edges, and depends on all 3 external services simultaneously. This is over-engineered for a system that hasn't proven it can do a basic search-and-answer reliably.

The user wants to rebuild incrementally. Phase 1+2 establishes the core: a LangGraph orchestrator that calls Ollama with native tool calling, and one tool (SearXNG web search). This proves the tool-calling loop works before adding crawling (Phase 3), Qdrant RAG (Phase 4), and verification/reports (Phase 5).

The key architectural insight: the system should NOT be single-shot tool calls. It should separate planning from execution, loop when research is insufficient, and track structured quality metrics (confidence, source count, contradiction detection, citation coverage). Phase 1+2 lays the groundwork by designing state that accommodates these metrics even before the verifier is built.

## Current Weaknesses (Why This Change)

1. Not using Ollama's native tool calling. The current system makes 4-5 separate LLM calls per query with custom prompts that expect specific JSON output parsed by parse_json_response. Meanwhile, qwen3:14b scores 11/12 on native tool calling via /api/chat with tools.

2. Sequential LLM calls add latency. At 43 tok/s, five sequential calls means 10-30 seconds of generation time alone. A ReAct loop with tool calling would often answer in 1-2 LLM calls.

3. Fragile JSON parsing as the glue layer. Every node depends on parse_json_response extracting structured output from free-form LLM text. Native tool calling gives structured output for free.

4. The critic loop rarely adds value at this model size. The verification philosophy (confidence scores, source counts, contradiction detection) is better implemented as deterministic checks on structured state, not another LLM call.

## Architecture

Two-node LangGraph graph with a conditional loop:

```
START -> orchestrator --[tool_call]--> search --> orchestrator (loop)
              |
              +--[text answer]--> END
```

The orchestrator calls Ollama's native /api/chat with a tools array containing web_search. If the LLM returns tool_calls, the search node executes via SearXNG and appends the result. The orchestrator re-evaluates with the new context. Max 5 iterations to prevent runaway.

State is designed to grow with later phases:

```python
class AgentState(TypedDict, total=False):
    # Input
    message: str
    conversation_id: str
    # Orchestrator message history for Ollama /api/chat
    messages: list[dict]
    # Research tracking, populated now, verified in Phase 5
    sources: list[str]
    search_count: int
    # Output
    final_response: str
    # Future Phase 5 fields: confidence, citation_coverage, contradictions
```

## Files to Modify

### images/agent/graph.py - Rewrite
Replace the 10-node graph with 2 nodes:
- orchestrator_node: Appends user message (first call) or tool result (loop), calls OllamaClient.chat() with tools, inspects response for tool_calls vs text content
- search_node: Extracts query from tool_calls, calls SearxngClient.search(), formats results, appends tool response to messages
- route_after_orchestrator(): Returns "search" if tool_calls present and under max iterations, END otherwise

### images/agent/clients/ollama.py - Add tools to chat()
Add tools parameter to chat() method. When tools are provided, include them in the /api/chat payload. Return the full message dict (not just content string) so the caller can inspect tool_calls.

Currently chat() returns str. Change it to return the full message dict when tools are provided, so the orchestrator can check for tool_calls. Signature:

```python
def chat(self, messages, model=None, stream=False, tools=None) -> dict | str | Generator
```

When tools is provided, return resp.json()["message"] (the full message dict with role, content, and optional tool_calls). When tools is None, return just the content string as before.

### images/agent/tools.py - New file
Define the web_search tool schema and a dispatch function:

```python
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}

TOOLS = [WEB_SEARCH_TOOL]

def execute_tool(name, arguments, searxng):
    """Dispatch a tool call to the appropriate client."""
```

This file grows in later phases (add crawl tool in Phase 3, rag_search in Phase 4).

### images/agent/agent.py - Simplify
- Remove Qdrant imports and /collections endpoint (Phase 4)
- Remove conversation.py import (re-add when needed)
- Simplify ChatResponse: response, sources, search_count, conversation_id
- Keep /health (check Ollama + SearXNG only), /chat, /chat/stream
- Keep asyncio.to_thread for graph invocation
- Keep structured logging

### images/agent/requirements.txt - Slim down
Remove beautifulsoup4 (Phase 3 crawling). Keep: langgraph, typing_extensions, requests, fastapi, uvicorn.

### images/agent/Dockerfile - No changes needed

### k8s/agent.yaml - Simplify ConfigMap
Remove per-node model vars (INTERPRETER_MODEL, CRITIC_MODEL, etc). Keep: OLLAMA_URL, SEARXNG_URL, AGENT_MODEL, AGENT_MAX_TOKENS.

## Files to Remove

- images/agent/nodes/ entire directory (interpreter.py, decision.py, research.py, synthesis.py, critic.py, ingest.py, parsing.py, __init__.py)
- images/agent/clients/qdrant.py (Phase 4)
- images/agent/conversation.py (re-add when needed)

## Files to Keep As-Is

- images/agent/clients/searxng.py - works perfectly
- images/agent/clients/__init__.py - update to remove QdrantClient export
- tests/test-stack.sh - agent health check still works
- tests/test-agent.sh - update to remove /collections test
- tests/test-services.sh, tests/test-tool-calling.py, benchmarks - unchanged

## Unit Tests - Rewrite

Replace images/agent/tests/ contents:
- test_tools.py - test tool schema, execute_tool dispatch
- test_graph.py - test route_after_orchestrator logic, max iteration guard
- test_clients.py - keep Ollama and SearXNG tests, add test for chat() with tools param, remove Qdrant tests

## Verification

1. Unit tests: cd images/agent && python3 -m pytest tests/ -v
2. Syntax check: python3 -c "import py_compile; ..." on all .py files
3. Integration: Deploy to K8s, run tests/test-agent.sh
4. Manual smoke test: curl -X POST http://bigfish:31400/chat -H "Content-Type: application/json" -d '{"message": "What is the latest version of Python?"}' - should trigger a web search and return a sourced answer
5. No-tool test: curl ... -d '{"message": "What is 2+2?"}' - should answer directly without searching

## Future Phases

- Phase 3: Add crawling/extraction tool
- Phase 4: Add Qdrant memory and local docs ingestion
- Phase 5: Add verifier with deterministic quality checks (confidence score, source count threshold, contradiction detection, citation coverage %)
