#!/usr/bin/env bash
# Launches Aider AI pair-programming assistant against local Ollama, usage: ./aider.sh [extra-args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/agent-env.sh"
export OLLAMA_API_BASE="${OLLAMA_HOST}"

if ! command -v aider &>/dev/null; then
    echo "Aider not found. Installing via pipx..."
    pipx install aider-chat
    echo "Installed."
fi

exec aider --model "ollama/$AGENT_MODEL" "$@"
