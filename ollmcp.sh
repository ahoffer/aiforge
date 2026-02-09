#!/usr/bin/env bash
# Launches ollmcp MCP client for Ollama with configured MCP servers, usage: ./ollmcp.sh [extra-args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/agent-env.sh"

if ! command -v ollmcp &>/dev/null; then
    echo "Installing mcp-client-for-ollama..."
    pipx install mcp-client-for-ollama
    echo "Installed."
fi

exec ollmcp \
    -H "$OLLAMA_HOST" \
    -m "$AGENT_MODEL" \
    -j "$SCRIPT_DIR/mcp-servers.json" \
    "$@"
