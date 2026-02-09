#!/usr/bin/env bash
# Shared environment for agent launcher scripts.
# OLLAMA_HOST is the client-side NodePort URL used by host tools to reach Ollama.
# It does not match the ConfigMap bind address (0.0.0.0:11434) which is server-side.
# Override any variable before sourcing or via environment.

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:31434}"
OLLAMA_URL="${OLLAMA_URL:-$OLLAMA_HOST}"

# Base model pulled from Ollama registry
AGENT_BASE_MODEL="${AGENT_BASE_MODEL:-qwen3:14b}"

# Tuned alias created by the Ollama postStart hook with baked-in verbosity controls.
# Falls back to the base model if the alias does not exist.
AGENT_MODEL="${AGENT_MODEL:-${AGENT_BASE_MODEL}-agent}"

# Response tuning parameters. Same defaults as the ConfigMap.
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.7}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-2048}"
AGENT_TOP_P="${AGENT_TOP_P:-0.9}"
AGENT_REPEAT_PENALTY="${AGENT_REPEAT_PENALTY:-1.2}"
AGENT_SYSTEM_PROMPT="${AGENT_SYSTEM_PROMPT:-Be concise and direct. Avoid filler phrases, unnecessary preamble, and restating the question. Get to the point.}"
