#!/usr/bin/env bash
# Launches Goose AI agent session against local Ollama, usage: ./goose.sh [extra-args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/agent-env.sh"
export OLLAMA_HOST
export GOOSE_PROVIDER="${GOOSE_PROVIDER:-ollama}"
export GOOSE_MODEL="${GOOSE_MODEL:-$AGENT_MODEL}"

if ! command -v goose &>/dev/null; then
    echo "Goose not found. Installing..."
    curl -fsSL https://github.com/block/goose/releases/latest/download/download_cli.sh | CONFIGURE=false bash
    echo "Installed."
fi

exec goose session "$@"
