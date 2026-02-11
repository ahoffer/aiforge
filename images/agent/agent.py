"""FastAPI entry point for the agent."""

import argparse
import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from clients import OllamaClient, SearxngClient
from graph import agent_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")


# -- Native request/response models --

class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    response: str
    sources: list[str] = []
    search_count: int = 0
    conversation_id: str


class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str
    services: dict[str, bool]


# -- OpenAI-compatible models --

class OpenAIMessage(BaseModel):
    role: str
    content: str | None = None


class OpenAIChatRequest(BaseModel):
    model: str = "agent"
    messages: list[OpenAIMessage] = []
    stream: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    log.info("Agent starting up")
    yield
    log.info("Agent shutting down")


app = FastAPI(
    title="AI Agent",
    description="LangGraph orchestrator with tool calling",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of the agent and its dependencies."""
    ollama = OllamaClient()
    searxng = SearxngClient()

    services = {
        "ollama": ollama.health(),
        "searxng": searxng.health(),
    }

    overall = "healthy" if services["ollama"] else "degraded"

    return HealthResponse(status=overall, services=services)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the orchestrator graph."""
    conversation_id = request.conversation_id or str(uuid4())

    try:
        result = await asyncio.to_thread(agent_graph.invoke, {
            "message": request.message,
            "conversation_id": conversation_id,
        })

        return ChatResponse(
            response=result.get("final_response", ""),
            sources=result.get("sources", []),
            search_count=result.get("search_count", 0),
            conversation_id=conversation_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response with progress updates via Server-Sent Events."""
    conversation_id = request.conversation_id or str(uuid4())

    async def generate():
        try:
            graph_input = {
                "message": request.message,
                "conversation_id": conversation_id,
            }
            outputs = await asyncio.to_thread(
                lambda: list(agent_graph.stream(graph_input))
            )
            for output in outputs:
                for node_name, node_output in output.items():
                    yield f"event: node\ndata: {node_name}\n\n"

                    if "final_response" in node_output:
                        response = node_output["final_response"]
                        yield f"event: response\ndata: {response}\n\n"

            yield "event: done\ndata: complete\n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# -- OpenAI-compatible endpoints --

@app.get("/v1/models")
async def list_models():
    """Return model list for OpenAI client discovery."""
    return {
        "object": "list",
        "data": [
            {
                "id": "agent",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint.

    Runs the message through the full agent graph and returns the
    response in OpenAI ChatCompletion format. Supports both streaming
    and non-streaming modes.
    """
    # Extract last user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user" and msg.content:
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    completion_id = f"chatcmpl-{uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        return await _stream_completions(user_message, completion_id, created)

    return await _non_stream_completions(user_message, completion_id, created)


async def _non_stream_completions(message: str, completion_id: str, created: int):
    """Run graph and return a single ChatCompletion response."""
    try:
        result = await asyncio.to_thread(agent_graph.invoke, {
            "message": message,
            "conversation_id": str(uuid4()),
        })

        content = result.get("final_response", "")
        sources = result.get("sources", [])
        search_count = result.get("search_count", 0)

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": "agent",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "metadata": {
                "sources": sources,
                "search_count": search_count,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_completions(message: str, completion_id: str, created: int):
    """Stream graph output as OpenAI SSE delta chunks."""

    async def generate():
        try:
            graph_input = {
                "message": message,
                "conversation_id": str(uuid4()),
            }
            outputs = await asyncio.to_thread(
                lambda: list(agent_graph.stream(graph_input))
            )

            for output in outputs:
                for _node_name, node_output in output.items():
                    if "final_response" in node_output:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "agent",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": node_output["final_response"],
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk with finish_reason
            done_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "agent",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            log.error("Streaming error: %s", e)
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def main():
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(description="AI Agent")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the HTTP server",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "message",
        nargs="*",
        help="Message to process (CLI mode)",
    )

    args = parser.parse_args()

    if args.serve:
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.message:
        message = " ".join(args.message)
        result = agent_graph.invoke({"message": message})
        print(result.get("final_response", "No response"))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
