#!/usr/bin/env bash
# Launches OpenCode AI coding assistant against local Ollama, usage: ./opencode.sh [extra-args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/agent-env.sh"

if ! command -v opencode &>/dev/null; then
    echo "OpenCode not found. Installing..."
    curl -sL "https://github.com/anomalyco/opencode/releases/latest/download/opencode-linux-x64.tar.gz" \
        | tar xz -C ~/.local/bin
    chmod +x ~/.local/bin/opencode
    echo "Installed."
fi

exec opencode -m "ollama/$AGENT_MODEL" "$@"
