# Proteus Review Remediation Plan

Expert review identified seven issues. This plan addresses them in order of
effort-to-impact ratio, starting with small high-confidence fixes and building
toward the larger structural change.

## 1. JSON crash on malformed tool-call arguments

`_openai_messages_to_ollama` calls `json.loads` without a try/except. A client
sending invalid JSON in `tool_calls[].function.arguments` gets a 500 instead
of a 400.

Fix: wrap the `json.loads` call in a try/except that raises `HTTPException(400)`.
Add a test that sends malformed arguments and asserts 400.

Files: `images/proteus/proteus.py`, `images/proteus/tests/tests/test_proxy.py`

## 2. Streaming parser brittle to malformed chunks

`_openai_streaming` calls `json.loads(line)` with no guard. A malformed chunk
from Ollama raises `json.JSONDecodeError` that kills the SSE stream without
sending `[DONE]`.

Fix: wrap `json.loads(line)` in a try/except that logs a warning and skips the
bad line. Add `json.JSONDecodeError` to the exception handling in the streaming
generator. Add a test.

Files: `images/proteus/proteus.py`, `images/proteus/tests/tests/test_proxy.py`

## 3. tool_choice accepted but ignored

`OpenAIChatRequest` declares `tool_choice` but neither `_openai_non_streaming`
nor `_openai_streaming` forwards it in the Ollama payload.

Fix: include `tool_choice` in the Ollama payload when present. Ollama's
`/api/chat` accepts this field directly. Add a test that verifies the field
reaches the payload.

Files: `images/proteus/proteus.py`, `images/proteus/tests/tests/test_proxy.py`

## 4. /chat/stream is not actually streaming

`chat_stream` materializes the entire graph output with `list(agent_graph.stream(...))`
before yielding SSE events. First-token latency equals full-run latency.

Fix: replace the `list()` approach with a queue-based bridge. Run
`agent_graph.stream()` in a background thread that pushes output dicts onto an
`asyncio.Queue`. The async generator pulls from the queue and yields SSE events
incrementally. Add a test that verifies events arrive before the graph completes.

Files: `images/proteus/proteus.py`, `images/proteus/tests/tests/test_proxy.py`

## 5. Prompt injection boundary for tool outputs

Raw tool output from web search is fed into the model context as trusted
`role: "tool"` content. Hostile search snippets could steer model behavior.

Fix: add an explicit trust boundary instruction to the system prompt telling the
model to treat tool output as untrusted user-supplied data. Truncate tool output
to a reasonable maximum length.

Files: `images/proteus/graph.py`

## 6. Regression tests for all fixes

Current tests cover happy-path conversion only. Each fix above should include
its own test, but this item covers an integration-level pass to ensure the
failure modes are covered end to end.

- Invalid JSON tool args returning 400
- tool_choice forwarded in Ollama payload
- Malformed upstream chunk handled gracefully
- True incremental SSE timing for /chat/stream
- Tool output length bounded

Files: `images/proteus/tests/tests/test_proxy.py`, `images/proteus/tests/tests/test_graph.py`

## 7. Durable and bounded conversation memory

`InMemorySaver` is process-local and unbounded. Under sustained load with many
thread IDs memory grows without limit and all state is lost on restart.

Fix: swap `InMemorySaver` for `SqliteSaver` with a configurable path. Add a
TTL-based or LRU eviction policy to bound memory. This is a deployment concern
and lowest urgency.

Files: `images/proteus/graph.py`, deployment configs
