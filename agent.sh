#!/usr/bin/env bash
# Launches the autonomous multi-agent system, usage: ./agent.sh [extra-args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.env"

PYTHON_BIN="python3"
PIP_BIN="pip3"

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
    PIP_BIN="$VIRTUAL_ENV/bin/pip"
else
    if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
        echo "Creating virtual environment..."
        python3 -m venv "$SCRIPT_DIR/.venv"
    fi
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
    PIP_BIN="$SCRIPT_DIR/.venv/bin/pip"
fi

# Check dependencies
if ! "$PYTHON_BIN" -c "import requests, bs4" &>/dev/null; then
    echo "Installing dependencies..."
    "$PIP_BIN" install -q requests beautifulsoup4
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/agent.py" \
    --ollama-url "${OLLAMA_URL:-$OLLAMA_HOST}" \
    --searxng-url "$SEARXNG_URL" \
    --qdrant-url "$QDRANT_URL" \
    --embedding-model "$EMBEDDING_MODEL" \
    --interpreter-model "$INTERPRETER_MODEL" \
    --orchestrator-model "$ORCHESTRATOR_MODEL" \
    --research-model "$RESEARCH_MODEL" \
    --synthesis-model "$SYNTHESIS_MODEL" \
    --critic-model "$CRITIC_MODEL" \
    "$@"
