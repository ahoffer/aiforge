# Local AI Agent Stack

Self-hosted AI development platform on Kubernetes with GPU-accelerated LLM inference, web search, vector storage, and a chat interface.

## Architecture

Four services run in the `ai-agent` namespace. Ollama serves models with NVIDIA GPU passthrough. SearXNG aggregates web search results. Qdrant stores vector embeddings for RAG. Open WebUI provides a browser chat interface connecting all three.

| Service | Role | Cluster DNS | NodePort |
|---------|------|-------------|----------|
| Ollama | LLM inference (GPU) | ollama:11434 | :31434 |
| SearXNG | Web search aggregation | searxng:8080 | :31080 |
| Qdrant | Vector embeddings for RAG | qdrant:6333 | :31333 |
| Open WebUI | Browser chat interface | open-webui:8080 | :31380 |

## Quick Start

Prerequisites: Kubernetes cluster with GPU support, NVIDIA device plugin, kubectl.

```bash
./install.sh
```

Verify with `./tests/test-stack.sh` (8 checks) and `./tests/test-services.sh` (19 checks), then open `http://localhost:31380` or launch any agent.

## Agent Frontends

| Frontend | Latency | Description |
|----------|---------|-------------|
| `aider.sh` | 13s | Pair programming agent that proposes diffs for human review. Edits code in your working directory. No web search or tool use. |
| `goose.sh` | 17s | Autonomous multi-step agent that chains tasks together. Web search and MCP tools available via config. Good for longer autonomous sequences. |
| `ollmcp.sh` | 4s | Terminal chat with live tool calling across all 5 MCP servers. Web search via SearXNG, file access, git, shell commands. Best for exploring what the model can do with tools. |
| `opencode.sh` | 1s | Lightweight TUI for quick code questions and small edits with human review. No web search or tool use. Fastest startup of any frontend. |
| `openhands.sh` | 10s | Fully autonomous engineering agent that runs in a sandboxed container. Plans, codes, tests, and iterates without intervention. Web search via SearXNG. |
| [Open WebUI](http://localhost:31380) | - | Browser chat interface for general questions, conversation, and web research. SearXNG integration searches the web automatically. Supports RAG, document uploads, and image generation. Works like ChatGPT. |

## MCP Tool Servers

Five servers in `mcp-servers.json` extend tool-capable frontends.

| Server | Capability |
|--------|-----------|
| searxng | Web search via SearXNG metasearch |
| filesystem | Read/write access to home directory |
| git | Repository inspection, history, branches |
| shell | Whitelisted commands: ls, grep, find, python3, node, make, kubectl |
| fetch | HTTP URL content retrieval |

## Configuration

### ConfigMap Reference

All Kubernetes configuration lives in two ConfigMaps applied via `agent.yaml`.

#### ollama-config

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_HOST | 0.0.0.0:11434 | Server bind address |
| OLLAMA_KEEP_ALIVE | 5m | How long to keep models in memory before unloading |
| OLLAMA_NUM_PARALLEL | 2 | Parallel inference requests |
| OLLAMA_MAX_LOADED_MODELS | 1 | Maximum models loaded simultaneously |
| DEFAULT_MODEL | qwen3:14b | Base model pulled from Ollama registry at startup |
| AGENT_TEMPERATURE | 0.7 | Sampling temperature, lower is more focused |
| AGENT_MAX_TOKENS | 2048 | Maximum output tokens per generation |
| AGENT_TOP_P | 0.9 | Nucleus sampling threshold |
| AGENT_REPEAT_PENALTY | 1.2 | Penalty discouraging token repetition |
| AGENT_SYSTEM_PROMPT | Be concise and direct... | System prompt baked into the tuned model alias |

#### open-webui-config

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_BASE_URLS | http://ollama:11434 | Ollama API endpoint via cluster DNS |
| ENABLE_RAG_WEB_SEARCH | true | Enable RAG with web search |
| RAG_WEB_SEARCH_ENGINE | searxng | Search backend for RAG |
| SEARXNG_QUERY_URL | http://searxng:8080/search?q=\<query\>&format=json | SearXNG query endpoint |
| WEBUI_AUTH | false | Disable authentication |
| VECTOR_DB | qdrant | Vector database backend for embeddings |
| QDRANT_URI | http://qdrant:6333 | Qdrant REST endpoint via cluster DNS |
| RAG_EMBEDDING_ENGINE | ollama | Use Ollama for embedding generation |
| RAG_EMBEDDING_MODEL | nomic-embed-text | Embedding model name |
| DEFAULT_MODELS | qwen3:14b-agent | Model preselected in the chat dropdown |

#### .env

Client-side defaults sourced by all launcher scripts.

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_HOST | http://localhost:31434 | Client-side NodePort URL for reaching Ollama |
| OLLAMA_URL | $OLLAMA_HOST | Alias for tools that prefer OLLAMA_URL over OLLAMA_HOST |
| AGENT_BASE_MODEL | qwen3:14b | Base model pulled from Ollama registry |
| AGENT_MODEL | ${AGENT_BASE_MODEL}-agent | Tuned alias with baked-in verbosity controls |
| AGENT_TEMPERATURE | 0.7 | Sampling temperature, lower is more focused |
| AGENT_MAX_TOKENS | 2048 | Maximum output tokens per generation |
| AGENT_TOP_P | 0.9 | Nucleus sampling threshold |
| AGENT_REPEAT_PENALTY | 1.2 | Penalty discouraging token repetition |
| AGENT_SYSTEM_PROMPT | Be concise and direct... | System prompt baked into the tuned model alias |

### Environment Variable Override

Override any variable before launching a script to use different settings without restarting the cluster.

```bash
AGENT_MODEL=qwen3:14b ./goose.sh          # bypass the tuned alias
AGENT_TEMPERATURE=0.3 ./ollmcp.sh          # lower temperature for this session
AGENT_MODEL=qwen3:8b ./aider.sh            # use the smaller model
```

## Testing

`tests/test-stack.sh` validates health, inference, tool calling, and token speed (minimum 10 tok/s). `tests/test-services.sh` checks Qdrant CRUD, SearXNG search, Open WebUI connectivity, embeddings, and cross-service wiring. `tests/test-tool-calling.py` runs 12 prompts across single-tool, no-tool, and multi-tool categories. `tests/bench-ollama.sh` measures generation speed, prompt eval, time to first token, and tool call latency. `tests/bench-frontends.sh` compares wall-clock latency across all agent frontends.

```bash
./tests/test-stack.sh
./tests/test-services.sh
python3 tests/test-tool-calling.py
```

Override the target model or URL with environment variables, for example `MODEL=qwen3:8b ./tests/test-stack.sh`.

## Teardown

Remove the stack while keeping persistent data for future redeployment.

```bash
./uninstall.sh
```

Remove everything including downloaded models and vector data.

```bash
./uninstall.sh --purge
```
