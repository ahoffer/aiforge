"""FastAPI entry point for the agent."""

import argparse
import asyncio
import logging
import sys
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
