# proteus.py
"""FastAPI entry point for Proteus.

Two modes of operation:
- /chat and /chat/stream run the LangGraph agent with server-side tool
  execution, for Open WebUI and similar thick-server clients.
- /v1/chat/completions is a transparent proxy to Ollama. It preserves
  tool_calls in the response so clients like opencode and goose can
  execute tools client-side via MCP.
- /v1/embeddings proxies to Ollama's embedding endpoint for MCP servers
  that cannot reach the cluster-internal Ollama service directly.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from graph import agent_graph
from clients import OllamaClient, SearxngClient

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
AGENT_MODEL = os.getenv("AGENT_MODEL", "devstral:latest-agent")
# Feb 2025: Ollama defaults to 4096 context tokens. Injected into every request
# to prevent silent truncation of tool-heavy prompts. See ollama/ollama#5356.
AGENT_NUM_CTX = int(os.getenv("AGENT_NUM_CTX", "0"))

# Sentinel for the sync-to-async queue bridge in chat_stream
_SENTINEL = object()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
PROXY_TIMEOUT = 300.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("proteus")

LOG_LANGGRAPH_OUTPUT = os.getenv("LOG_LANGGRAPH_OUTPUT", "true").lower() in ("1", "true")


# ----------------------------
# Native request/response models
# ----------------------------

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []
    search_count: int = 0
    conversation_id: str


class HealthResponse(BaseModel):
    status: str
    services: dict[str, bool] = {}


# ----------------------------
# OpenAI-compatible models
# ----------------------------

class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string per OpenAI spec


class OpenAIToolCall(BaseModel):
    id: str
    type: str = "function"
    function: OpenAIFunctionCall


class OpenAIMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class OpenAIChatRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = "proteus"
    messages: list[OpenAIMessage] = []
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


def _msg_to_dict(msg) -> dict:
    """Convert an OpenAIMessage (or dict) to a plain dict message."""
    if isinstance(msg, dict):
        return msg
    d = {"role": msg.role}
    if msg.content is not None:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id is not None:
        d["tool_call_id"] = msg.tool_call_id
    if msg.name is not None:
        d["name"] = msg.name
    return d



# ----------------------------
# App setup
# ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Proteus starting up")
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(PROXY_TIMEOUT, connect=10.0))
    yield
    await app.state.http.aclose()
    log.info("Proteus shutting down")


# ----------------------------
# OpenAI <-> Ollama format conversion
# ----------------------------

def _openai_messages_to_ollama(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages to Ollama format.

    Main difference: tool_calls[].function.arguments is a JSON string in
    OpenAI but a dict in Ollama. Raises HTTPException 400 if arguments
    contain invalid JSON.
    """
    converted = []
    for msg in messages:
        out = dict(msg)
        if "tool_calls" in out and out["tool_calls"]:
            new_calls = []
            for tc in out["tool_calls"]:
                raw_args = tc["function"]["arguments"]
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except (json.JSONDecodeError, ValueError):
                        fn_name = tc["function"].get("name", "unknown")
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid JSON in tool_calls arguments for function '{fn_name}'",
                        )
                else:
                    parsed_args = raw_args
                new_calls.append({
                    **tc,
                    "function": {**tc["function"], "arguments": parsed_args},
                })
            out["tool_calls"] = new_calls
        converted.append(out)
    return converted


def _ollama_tool_calls_to_openai(tool_calls: list[dict]) -> list[dict]:
    """Convert Ollama tool_calls to OpenAI format."""
    openai_calls = []
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        openai_calls.append({
            "id": tc.get("id") or f"call_{uuid4().hex[:8]}",
            "type": "function",
            "index": i,
            "function": {
                "name": fn.get("name", ""),
                "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
            },
        })
    return openai_calls


def _log_tool_call_outcome(tool_calls, tools_offered: bool, stream: bool):
    """Log tool call details for proxy-path observability.

    Emits one log line per request summarizing whether the model called
    tools or answered directly. Logs tool names and argument keys only,
    not argument values, to keep logs useful without leaking payload data.
    """
    mode = "stream" if stream else "non-stream"
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "unknown")
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    arg_keys = list(json.loads(raw_args).keys())
                except (json.JSONDecodeError, AttributeError):
                    arg_keys = ["<unparseable>"]
            elif isinstance(raw_args, dict):
                arg_keys = list(raw_args.keys())
            else:
                arg_keys = []
            log.info("proxy %s tool_call name=%s arg_keys=%s", mode, name, arg_keys)
    elif tools_offered:
        log.info("proxy %s model chose direct answer despite tools offered", mode)


def _build_ollama_options(request) -> dict:
    """Extract Ollama options from the OpenAI request parameters."""
    opts = {}
    if AGENT_NUM_CTX:
        opts["num_ctx"] = AGENT_NUM_CTX
    if request.max_tokens is not None:
        opts["num_predict"] = request.max_tokens
    if request.temperature is not None:
        opts["temperature"] = request.temperature
    if request.top_p is not None:
        opts["top_p"] = request.top_p
    return opts


app = FastAPI(
    title="Proteus",
    description="LangGraph agent and transparent Ollama proxy",
    version="5.0.0",
    lifespan=lifespan,
)


def _dependency_health() -> dict[str, bool]:
    return {
        "ollama": OllamaClient().health(),
        "searxng": SearxngClient().health(),
    }


def _overall_health_status(services: dict[str, bool]) -> str:
    if services and all(services.values()):
        return "healthy"
    if services and any(services.values()):
        return "degraded"
    return "unhealthy"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    services = await asyncio.to_thread(_dependency_health)
    status = _overall_health_status(services)
    return HealthResponse(status=status, services=services)


@app.get("/health/live", response_model=HealthResponse)
async def health_live():
    # Liveness should not depend on upstream services.
    return HealthResponse(status="healthy", services={})


@app.get("/health/ready", response_model=HealthResponse)
async def health_ready():
    services = await asyncio.to_thread(_dependency_health)
    status = _overall_health_status(services)
    status_code = 200 if status == "healthy" else 503
    body = HealthResponse(status=status, services=services).model_dump()
    return JSONResponse(status_code=status_code, content=body)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    log.info("POST /chat conversation_id=%s", request.conversation_id)
    conversation_id = request.conversation_id or str(uuid4())

    try:
        result = await asyncio.to_thread(
            agent_graph.invoke,
            {"message": request.message, "conversation_id": conversation_id},
            {"configurable": {"thread_id": conversation_id}},
        )

        if LOG_LANGGRAPH_OUTPUT:
            preview = (result.get("final_response", "") or "")[:200].replace("\n", " ")
            log.info(
                "LangGraph /chat result conversation_id=%s keys=%s search_count=%s final_preview=%r",
                conversation_id,
                sorted(result.keys()),
                result.get("search_count", 0),
                preview,
            )

        return ChatResponse(
            response=result.get("final_response", "") or "",
            sources=result.get("sources", []) or [],
            search_count=int(result.get("search_count", 0) or 0),
            conversation_id=conversation_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    log.info("POST /chat/stream conversation_id=%s", request.conversation_id)
    conversation_id = request.conversation_id or str(uuid4())

    async def generate():
        graph_input = {"message": request.message, "conversation_id": conversation_id}
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run_graph():
            try:
                for output in agent_graph.stream(
                    graph_input,
                    config={"configurable": {"thread_id": conversation_id}},
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, output)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        loop.run_in_executor(None, _run_graph)

        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    yield f"event: error\ndata: {str(item)}\n\n"
                    return

                for node_name, node_output in item.items():
                    keys = sorted(node_output.keys()) if isinstance(node_output, dict) else []
                    preview = ""
                    if isinstance(node_output, dict):
                        preview = (node_output.get("final_response", "") or "")[:160].replace("\n", " ")
                    if LOG_LANGGRAPH_OUTPUT:
                        log.info(
                            "LangGraph /chat/stream node=%s conversation_id=%s keys=%s final_preview=%r",
                            node_name,
                            conversation_id,
                            keys,
                            preview,
                        )

                    yield f"event: node\ndata: {node_name}\n\n"

                    if isinstance(node_output, dict) and "final_response" in node_output:
                        response = node_output.get("final_response") or ""
                        yield f"event: response\ndata: {response}\n\n"

            yield "event: done\ndata: complete\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/v1/models")
async def list_models():
    model_info = {"id": "proteus", "object": "model", "created": 0, "owned_by": "local"}
    if AGENT_NUM_CTX:
        model_info["context_window"] = AGENT_NUM_CTX
    return {"object": "list", "data": [model_info]}


@app.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str):
    if model_id == "proteus":
        model_info = {"id": "proteus", "object": "model", "created": 0, "owned_by": "local"}
        if AGENT_NUM_CTX:
            model_info["context_window"] = AGENT_NUM_CTX
        return model_info
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    tools_count = len(request.tools) if request.tools else 0
    log.info("POST /v1/chat/completions messages=%d tools=%d stream=%s",
             len(request.messages), tools_count, request.stream)

    if request.stream:
        return await _openai_streaming(request)

    return await _openai_non_streaming(request)


async def _openai_non_streaming(request: OpenAIChatRequest):
    # Validate and convert messages before acquiring the HTTP client so
    # malformed input fails fast with 400 rather than touching resources.
    messages = [_msg_to_dict(m) for m in request.messages]
    ollama_messages = _openai_messages_to_ollama(messages)

    completion_id = f"chatcmpl-{uuid4().hex[:12]}"
    created = int(time.time())
    http: httpx.AsyncClient = app.state.http
    model = AGENT_MODEL if request.model in ("proteus", None) else request.model

    ollama_payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": False,
    }
    if request.tools:
        ollama_payload["tools"] = request.tools
    if request.tool_choice is not None:
        ollama_payload["tool_choice"] = request.tool_choice
    opts = _build_ollama_options(request)
    if opts:
        ollama_payload["options"] = opts

    try:
        resp = await http.post(f"{OLLAMA_URL}/api/chat", json=ollama_payload)
    except httpx.TimeoutException:
        return JSONResponse(status_code=504, content={
            "error": {"message": "Ollama request timed out", "type": "timeout_error"},
        })
    except httpx.ConnectError:
        return JSONResponse(status_code=502, content={
            "error": {"message": "Cannot connect to Ollama", "type": "connection_error"},
        })

    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content={
            "error": {"message": f"Ollama returned HTTP {resp.status_code}", "type": "upstream_error"},
        })

    body = resp.json()
    ollama_msg = body.get("message", {})
    content = ollama_msg.get("content", "") or ""
    tool_calls = ollama_msg.get("tool_calls")

    openai_msg = {"role": "assistant", "content": content}
    finish = "stop"
    if tool_calls:
        openai_msg["tool_calls"] = _ollama_tool_calls_to_openai(tool_calls)
        finish = "tool_calls"

    _log_tool_call_outcome(
        openai_msg.get("tool_calls"), tools_offered=bool(request.tools), stream=False)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model or "proteus",
        "choices": [{"index": 0, "message": openai_msg, "finish_reason": finish}],
    }


async def _openai_streaming(request: OpenAIChatRequest):
    # Validate and convert messages before acquiring the HTTP client so
    # malformed input fails fast with 400 rather than touching resources.
    messages = [_msg_to_dict(m) for m in request.messages]
    ollama_messages = _openai_messages_to_ollama(messages)

    completion_id = f"chatcmpl-{uuid4().hex[:12]}"
    created = int(time.time())
    http: httpx.AsyncClient = app.state.http
    model = AGENT_MODEL if request.model in ("proteus", None) else request.model

    ollama_payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": True,
    }
    if request.tools:
        ollama_payload["tools"] = request.tools
    if request.tool_choice is not None:
        ollama_payload["tool_choice"] = request.tool_choice
    opts = _build_ollama_options(request)
    if opts:
        ollama_payload["options"] = opts

    model_label = request.model or "proteus"

    async def generate():
        try:
            async with http.stream("POST", f"{OLLAMA_URL}/api/chat", json=ollama_payload) as resp:
                if resp.status_code != 200:
                    err = {"error": {"message": f"Ollama returned HTTP {resp.status_code}", "type": "upstream_error"}}
                    yield f"data: {json.dumps(err)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("Skipping malformed chunk from Ollama: %r", line[:200])
                        continue
                    ollama_msg = data.get("message", {})
                    content = ollama_msg.get("content", "")
                    is_done = data.get("done", False)

                    if content:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_label,
                            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    if is_done:
                        tool_calls = ollama_msg.get("tool_calls")
                        _log_tool_call_outcome(
                            tool_calls, tools_offered=bool(request.tools), stream=True)
                        if tool_calls:
                            openai_calls = _ollama_tool_calls_to_openai(tool_calls)
                            tc_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_label,
                                "choices": [{"index": 0, "delta": {"tool_calls": openai_calls}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(tc_chunk)}\n\n"

                        finish = "tool_calls" if tool_calls else "stop"
                        done_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_label,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                        }
                        yield f"data: {json.dumps(done_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
        except httpx.TimeoutException:
            err = {"error": {"message": "Ollama request timed out", "type": "timeout_error"}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
        except httpx.ConnectError:
            err = {"error": {"message": "Cannot connect to Ollama", "type": "connection_error"}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
        except json.JSONDecodeError as e:
            log.warning("JSON decode error in stream: %s", e)
            err = {"error": {"message": "Malformed response from upstream", "type": "parse_error"}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


class EmbeddingRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = "nomic-embed-text"
    input: str | list[str] = ""


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    http: httpx.AsyncClient = app.state.http
    model = EMBEDDING_MODEL if request.model in ("proteus", None) else request.model
    text_input = request.input if isinstance(request.input, list) else [request.input]

    try:
        resp = await http.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": model, "input": text_input},
        )
    except httpx.TimeoutException:
        return JSONResponse(status_code=504, content={
            "error": {"message": "Ollama embedding request timed out", "type": "timeout_error"},
        })
    except httpx.ConnectError:
        return JSONResponse(status_code=502, content={
            "error": {"message": "Cannot connect to Ollama", "type": "connection_error"},
        })

    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content={
            "error": {"message": f"Ollama returned HTTP {resp.status_code}", "type": "upstream_error"},
        })

    body = resp.json()
    embeddings_list = body.get("embeddings", [])

    return {
        "object": "list",
        "model": model,
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(embeddings_list)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Proteus")
    parser.add_argument("--serve", action="store_true", help="Start the HTTP server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("message", nargs="*", help="Message to process (CLI mode)")
    args = parser.parse_args()

    if args.serve:
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.message:
        message = " ".join(args.message)
        # CLI: single-shot unless you add a conversation_id and pass thread_id.
        result = agent_graph.invoke({"message": message})
        print(result.get("final_response", "No response"))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
