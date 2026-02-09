# Local AI Agent Stack

A self-hosted AI development platform running on Kubernetes with GPU acceleration. Four services work together to provide LLM inference, web search, vector storage, and a chat interface, all accessible from the browser or from command-line agent tools.

## Architecture Overview

The stack runs four services within a single Kubernetes namespace named `ai-agent`. Ollama serves as the language model inference engine with NVIDIA GPU passthrough for fast token generation. SearXNG provides a privacy-respecting metasearch engine that aggregates results from multiple sources. Qdrant functions as a vector database for storing and retrieving embeddings used in retrieval-augmented generation workflows. Open WebUI presents a web-based chat interface that orchestrates interactions between these services and the user.

Services communicate using two distinct access patterns. Within the cluster, services reference each other by Kubernetes DNS names such as `http://ollama:11434` and `http://qdrant:6333`. From the host machine, clients access these services via NodePorts, exposing Ollama on port 31434, SearXNG on 31080, Qdrant on 31333, and Open WebUI on 31380. This dual pattern allows both in-cluster components and external tools like aider, goose, and opencode to interact with the services seamlessly.

At startup, Ollama automatically pulls the default model and the nomic-embed-text embedding model, then creates a tuned model alias with baked-in response parameters for concise output. All agent launcher scripts default to this tuned alias so that every tool produces consistent, focused responses.

```
┌────────────────────────────────────────────────────────┐
│                 Host Machine (NodePort)                 │
│                                                        │
│  Launcher Scripts           MCP Servers                │
│  (aider, goose,            (searxng, filesystem,       │
│   opencode, ollmcp,         git, shell, fetch)         │
│   openhands)                      │                    │
│        │                          │                    │
│        ▼                          ▼                    │
│   :31434 (Ollama)          :31080 (SearXNG)            │
│   :31380 (Open WebUI)      :31333 (Qdrant)             │
└────────┬──────────────────────────┬────────────────────┘
         │    Kubernetes Cluster    │
┌────────▼──────────────────────────▼────────────────────┐
│                 Namespace: ai-agent                     │
│                                                        │
│  ┌────────────────┐       ┌──────────────────────┐     │
│  │    Ollama      │◄──────│    Open WebUI         │     │
│  │  (GPU, 11434)  │       │    (8080)             │     │
│  │                │       └───┬──────────┬───────┘     │
│  │  Models:       │           │          │             │
│  │  qwen3:14b     │           ▼          ▼             │
│  │  nomic-embed   │     ┌─────────┐ ┌─────────┐       │
│  └────────────────┘     │ SearXNG │ │ Qdrant  │       │
│                         │ (8080)  │ │ (6333)  │       │
│                         └─────────┘ └─────────┘       │
└────────────────────────────────────────────────────────┘
```

### How Services Integrate

Open WebUI sends chat requests to Ollama at `http://ollama:11434`. Ollama processes them using the tuned model alias `qwen3:14b-agent`, which includes system prompts and response parameters. The response streams back to the chat interface.

When a user enables web search in Open WebUI, it queries SearXNG at `http://searxng:8080/search?q=<query>&format=json`. SearXNG aggregates results from multiple search backends and returns them to Open WebUI, which incorporates the retrieved information into the LLM prompt to ground responses in current web content.

Open WebUI connects to Qdrant at `http://qdrant:6333` to store and retrieve vector embeddings for uploaded documents. The nomic-embed-text model running inside Ollama generates 768-dimensional vectors through the `/api/embed` endpoint. Open WebUI is configured with `RAG_EMBEDDING_ENGINE=ollama` so all embedding computation stays local.

Host-side agent tools connect to Ollama via NodePort 31434 and to SearXNG via MCP server adapters on NodePort 31080. All launcher scripts source a common environment file and default to the same tuned model alias.

## Quick Start

Prerequisites: a Kubernetes cluster with GPU support, the NVIDIA device plugin, and kubectl configured.

```bash
./install.sh
```

This creates the `ai-agent` namespace, generates a SearXNG secret, applies the manifests, and waits for all deployments to reach ready state. After it completes, the services are available at the NodePort addresses shown above.

To verify the stack is working, run the test suites.

```bash
./test-stack.sh       # 8 checks: health, inference, tool calling, speed
./test-services.sh    # 19 checks: CRUD, cross-service wiring, config
```

Open the chat interface at `http://localhost:31380` or launch any agent from the command line.

```bash
./aider.sh            # AI pair programmer
./goose.sh            # General-purpose agent
./ollmcp.sh           # MCP client with tool servers
./opencode.sh         # Coding assistant TUI
./openhands.sh        # Autonomous agent
```

## Agent Frontends

**ollmcp** is an interactive MCP client that connects your local model to tool servers for web search, filesystem access, git operations, shell commands, and URL fetching. It is best for exploratory workflows where you want the model to freely choose which tools to invoke based on the task at hand. Launch it with `./ollmcp.sh`.

**aider** is an AI pair programmer that edits code in your project while you guide the conversation. It reads your files, proposes edits, and implements features with your feedback. Ideal for incremental code development. Launch it with `./aider.sh`.

**goose** is a general-purpose agent with extension support. It breaks down complex requests into sequences of tool calls and reasoning steps. Best suited for autonomous multi-step workflows. Launch it with `./goose.sh`.

**opencode** is a terminal-based coding assistant with a minimal TUI. It excels at explaining code, generating snippets, and discussing software design. Launch it with `./opencode.sh`.

**openhands** is an autonomous software engineering agent for end-to-end task completion. It integrates with SearXNG for research and works well for self-contained tasks. Launch it with `./openhands.sh`.

**Open WebUI** provides a browser-based chat interface at `http://localhost:31380` with session history, document uploads, web search, and RAG capabilities built in. No script launch needed.

## MCP Tool Servers

Five MCP servers are pre-configured in `mcp-servers.json` and provide capabilities to agent frontends that support MCP. SearXNG enables web search through a privacy-respecting metasearch engine. The filesystem server grants read and write access to your home directory. The git server exposes repository operations for inspecting history and branches. The shell server executes whitelisted commands including ls, grep, find, python3, node, make, and kubectl. The fetch server retrieves content from HTTP URLs. Together these servers transform the language model from a text generator into a tool-using assistant capable of exploring your system, researching information, and acting on real tasks.

## What You Can Do

Ask the model to search the web and write code in a single conversation. For example, prompt ollmcp with "Search for how to implement OAuth in Python and then write an example to /tmp/oauth.py". The model will invoke web search, analyze the results, and generate code based on what it learned.

Upload documents through Open WebUI and ask questions about their contents. The system stores document chunks as vector embeddings in Qdrant, then performs similarity searches to find relevant passages for each query. This is powerful for analyzing reports, extracting information from specs, or understanding unfamiliar codebases.

Pair program with aider by describing a feature or bug fix in natural language. Aider reads your files, understands the structure, and proposes edits with reasoning about why each change is necessary. You review, approve, or refine the suggestions while maintaining control.

Deploy autonomous workflows with openhands when you have a well-defined multi-step task. It reasons about the problem, breaks it into sub-goals, executes tool sequences, and adapts based on results.

## Configuration

### ConfigMap Reference

All Kubernetes-level configuration lives in two ConfigMaps applied via `agent.yaml`.

#### ollama-config

Controls Ollama server behavior and LLM response tuning.

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

Controls the Open WebUI chat interface and its service integrations.

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

#### agent-env.sh

Client-side defaults sourced by all launcher scripts. These connect host tools to the cluster services.

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

All launcher scripts source `agent-env.sh` which establishes defaults matching the ConfigMap. Override any variable before launching a script to use different settings without restarting the cluster.

```bash
AGENT_MODEL=qwen3:14b ./goose.sh          # bypass the tuned alias
AGENT_TEMPERATURE=0.3 ./ollmcp.sh          # lower temperature for this session
AGENT_MODEL=qwen3:8b ./aider.sh            # use the smaller model
```

### Tuning Response Verbosity

Five parameters in the `ollama-config` ConfigMap control how verbose the LLM responds. These get baked into a model alias named `${DEFAULT_MODEL}-agent` at pod startup via a Modelfile.

`AGENT_TEMPERATURE` controls randomness. Lower values like 0.3 produce focused, deterministic output. Higher values like 1.0 produce more creative, varied responses.

`AGENT_MAX_TOKENS` sets an upper limit on generation length. The default 2048 allows substantial responses while preventing runaway output.

`AGENT_TOP_P` implements nucleus sampling. Lower values prune low-probability tokens, producing more predictable output.

`AGENT_REPEAT_PENALTY` discourages the model from repeating itself. Values above 1.0 apply the penalty. Higher values like 1.5 aggressively suppress repetition.

`AGENT_SYSTEM_PROMPT` is a fixed instruction that guides model behavior. The default emphasizes conciseness.

To change these parameters, edit the ConfigMap and restart the Ollama pod.

```bash
kubectl edit configmap ollama-config -n ai-agent
kubectl rollout restart deploy/ollama -n ai-agent
```

## Testing

Three test scripts validate the stack at different levels.

`test-stack.sh` runs 8 checks covering service health, model availability, inference generation, tool calling via the OpenAI-compatible API, and token generation speed against a minimum threshold of 10 tokens per second.

`test-services.sh` runs 19 checks covering Qdrant CRUD operations, SearXNG search results, Open WebUI health and frontend serving, embedding model availability, Ollama embedding API output, agent model alias parameter verification, Open WebUI ConfigMap settings, and cross-service connectivity from Open WebUI to Qdrant and SearXNG via cluster DNS.

`test-tool-calling.py` is a Python suite of 12 prompts in three categories. Single-tool tests verify the model selects the correct tool. No-tool tests confirm it answers directly when no tool is needed. Multi-tool tests check that the model chains tools when a task requires it.

```bash
./test-stack.sh
./test-services.sh
python3 test-tool-calling.py
```

Override the target model or URL with environment variables or command-line arguments, for example `MODEL=qwen3:8b ./test-stack.sh`.

## Teardown

To remove the stack while keeping persistent data for future redeployment, run the uninstall script.

```bash
./uninstall.sh
```

To completely remove everything including downloaded models and vector data, use the purge flag.

```bash
./uninstall.sh --purge
```
