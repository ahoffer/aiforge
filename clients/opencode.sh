#!/usr/bin/env bash
# Launches OpenCode AI coding assistant through Proteus with MCP tool servers.
# Usage: ./opencode.sh              interactive TUI
#        ./opencode.sh <subcmd>     pass through to opencode
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../config.env"

if [[ "$AGENT_URL" == *"ollama"* ]] || [[ "$AGENT_URL" == *":11434"* ]]; then
    echo "Error: AGENT_URL must point to Proteus, not Ollama: $AGENT_URL"
    exit 1
fi

if ! command -v opencode &>/dev/null; then
    echo "OpenCode not found. Installing..."
    curl -sL "https://github.com/opencode-ai/opencode/releases/latest/download/opencode-linux-x64.tar.gz" \
        | tar xz -C ~/.local/bin
    chmod +x ~/.local/bin/opencode
    echo "Installed."
fi

# Generate config with Proteus provider and MCP servers.
# Timeout set to 10s to handle slow first-run package downloads via uvx/npx.
OPENCODE_CONFIG_DIR="${HOME}/.config/opencode"
mkdir -p "$OPENCODE_CONFIG_DIR"
OPENCODE_CONFIG_FILE="${OPENCODE_CONFIG_DIR}/config.json"
cat > "$OPENCODE_CONFIG_FILE" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "proteus": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Proteus",
      "options": {
        "baseURL": "${AGENT_URL}/v1"
      },
      "models": {
        "proteus": {
          "name": "proteus",
          "tools": true
        }
      }
    }
  },
  "mcp": {
    "shell": {
      "type": "local",
      "command": ["uvx", "mcp-shell-server"],
      "environment": {
        "ALLOW_COMMANDS": "ls,cat,head,tail,grep,find,wc,sort,uniq,diff,git,python3,node,npm,make,kubectl,curl"
      },
      "timeout": 10000
    }
  }
}
EOF

# Compatibility copy for older setups that still read opencode.json.
cp "$OPENCODE_CONFIG_FILE" "$OPENCODE_CONFIG_DIR/opencode.json"

if command -v curl &>/dev/null; then
    models_json="$(curl -fsS --max-time 3 "${AGENT_URL}/v1/models" 2>/dev/null || true)"
    if [[ -n "$models_json" ]] && ! grep -Eq '"id"[[:space:]]*:[[:space:]]*"proteus"' <<< "$models_json"; then
        echo "Error: AGENT_URL does not appear to be Proteus (missing model id=proteus): $AGENT_URL"
        exit 1
    fi
fi

# --log-level DEBUG prints full request/response payloads to log files
exec opencode --log-level DEBUG -m "proteus/proteus" "$@"
