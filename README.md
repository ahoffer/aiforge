# Local AI Agent Stack

Self-hosted AI platform on Kubernetes with GPU-accelerated inference, web search, vector storage, and chat.

## Architecture

Five services in the `aiforge` namespace.

| Service | Role | Cluster DNS | NodePort |
|---------|------|-------------|----------|
| Ollama | GPU model serving | ollama:11434 | :31434 |
| SearXNG | Web search aggregation | searxng:8080 | :31080 |
| Qdrant | Vector embeddings for RAG | qdrant:6333 | :31333 |
| Open WebUI | Browser chat interface | open-webui:8080 | :31380 |
| Agent | LangGraph autonomous agent | agent:8000 | :31400 |

## Quick Start

Prerequisites: Kubernetes cluster with GPU support, NVIDIA device plugin, kubectl.

```bash
./install.sh
```

Verify with `./tests/test-stack.sh` and `./tests/test-services.sh`, then open `http://localhost:31380`.

## Frontends

| Frontend | Description |
|----------|-------------|
| `goose.sh` | Terminal coding agent with MCP tool calling (web search, files, git, shell) |
| `opencode.sh` | Terminal coding assistant with MCP tool calling and TUI |
| [Open WebUI](http://localhost:31380) | Browser chat routed through the agent via OpenAI-compatible endpoint |

Local models sometimes misinterpret intent. Use explicit phrasing like "analyze this code, do not edit any files" to avoid unintended edits.

## Agent

FastAPI app in `images/agent/`, two-node LangGraph workflow:

- **Orchestrator** calls Ollama with native tool calling
- **Tools** node executes tool calls and loops back (up to 5 iterations)
- Available tools: `web_search` (SearXNG)

API endpoints:

- `POST /chat` - native chat with sources
- `POST /chat/stream` - SSE streaming with node progress
- `POST /v1/chat/completions` - OpenAI-compatible (streaming and non-streaming)
- `GET /v1/models` - model list for client discovery
- `GET /health` - dependency health check
- `GET /docs` - interactive OpenAPI docs

Build: `cd images/agent && ./build.sh` then `kubectl apply -f k8s/`

## Configuration

Launcher scripts source `defaults.sh`. Override any variable at launch.

| Variable | Default | Description |
|----------|---------|-------------|
| AGENT_MODEL | devstral:latest-agent | Tuned alias with concise output |
| GOOSE_MODEL | devstral:latest | Model for Goose |
| OPENCODE_MODEL | devstral:latest | Model for OpenCode |

```bash
GOOSE_MODEL=qwen3:8b ./clients/goose.sh  # use smaller model for Goose
OPENCODE_MODEL=qwen3:8b ./clients/opencode.sh  # use smaller model for OpenCode
```

Cluster-side settings live in ConfigMaps in `agent.yaml`.

## Testing

- `test-stack.sh` - health, inference, tool calling, token speed
- `test-services.sh` - Qdrant CRUD, SearXNG, Open WebUI, embeddings, cross-service wiring
- `test-agent.sh` - agent API including chat, streaming, and OpenAI-compatible endpoints
- `test-tool-calling.py` - 12 prompts across single-tool, no-tool, multi-tool categories
- `bench-ollama.sh` - generation speed, prompt eval, time to first token, tool call latency
- `images/agent/tests/` - unit tests (pytest)

```bash
./tests/test-stack.sh
./tests/test-services.sh
./tests/test-agent.sh
python3 tests/test-tool-calling.py
cd images/agent && python3 -m pytest tests/ -v
```

Override target model or URL: `MODEL=qwen3:8b ./tests/test-stack.sh`

## Teardown

```bash
./uninstall.sh           # keep persistent data
./uninstall.sh --purge   # remove everything including models and vectors
```

## Model Benchmark

`tests/bench-model-compare.sh` scores each model on when to call `web_search` vs answer directly. 20 calibrated prompts through the full agent stack.

| Model | Judgment | Search | NoSearch | JSON OK | E2E |
|-------|----------|--------|----------|---------|-----|
| devstral:latest | 20/20 | 12/12 | 8/8 | 20/20 | 1m 46s |
| qwen3:8b | 16/20 | 8/12 | 8/8 | 16/20 | 1m 03s |
| mistral-nemo:latest | 16/20 | 12/12 | 4/8 | 20/20 | 1m 19s |
| qwen2.5:14b | 15/20 | 9/12 | 6/8 | 17/20 | 14s |
| qwen3:14b | 13/20 | 5/12 | 8/8 | 13/20 | 1m 39s |
| llama3.1:8b | 10/20 | 10/12 | 0/8 | 18/20 | 54s |

- **Judgment** = Search + NoSearch combined score
- **Search** = correctly calls `web_search` on 12 current-info prompts
- **NoSearch** = correctly answers directly on 8 well-known-fact prompts
- **JSON OK** = valid tool call structure
- **E2E** = full stack latency

```bash
./tests/bench-model-compare.sh
```
