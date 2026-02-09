#!/usr/bin/env bash
# Launches OpenHands AI agent against local Ollama and SearXNG, usage: ./openhands.sh [extra-args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/agent-env.sh"

export LLM_API_KEY="${LLM_API_KEY:-ollama}"
export LLM_BASE_URL="${OLLAMA_HOST}"
export LLM_MODEL="${LLM_MODEL:-ollama/$AGENT_MODEL}"
export SEARXNG_URL="${SEARXNG_URL:-http://localhost:31080}"

if ! command -v openhands &>/dev/null; then
    echo "OpenHands CLI not found. Installing..."
    if command -v uv &>/dev/null; then
        uv tool install openhands --python 3.12
    else
        curl -fsSL https://install.openhands.dev/install.sh | sh
    fi
    echo "Installed."
fi

exec openhands --override-with-envs "$@"
