#!/usr/bin/env bash
# Launches OpenCode AI coding assistant against local Ollama.
# Usage: ./opencode.sh              Start web UI on port 31580
#        ./opencode.sh run "msg"    One-shot query
#        ./opencode.sh <subcmd>     Pass through to opencode
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.env"

AGENT_MODEL="qwen3:14b-16k"
OPENCODE_PORT="${OPENCODE_PORT:-31580}"

if ! command -v opencode &>/dev/null; then
    echo "OpenCode not found. Installing..."
    curl -sL "https://github.com/anomalyco/opencode/releases/latest/download/opencode-linux-x64.tar.gz" \
        | tar xz -C ~/.local/bin
    chmod +x ~/.local/bin/opencode
    echo "Installed."
fi

if [[ $# -eq 0 ]]; then
    FQDN="$(hostname -f 2>/dev/null || hostname)"
    LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    echo "OpenCode web UI starting on port ${OPENCODE_PORT}"
    echo "  http://${FQDN}:${OPENCODE_PORT}"
    [[ -n "$LAN_IP" ]] && echo "  http://${LAN_IP}:${OPENCODE_PORT}"
    exec opencode web --port "$OPENCODE_PORT" --hostname 0.0.0.0
fi

exec opencode -m "ollama/$AGENT_MODEL" "$@"
