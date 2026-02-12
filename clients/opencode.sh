#!/usr/bin/env bash
# Launches OpenCode AI coding assistant through Proteus with MCP tool servers.
# Usage: ./opencode.sh              interactive TUI
#        ./opencode.sh <subcmd>     pass through to opencode
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../defaults.sh"

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
cat > "$OPENCODE_CONFIG_DIR/opencode.json" <<EOF
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
          "name": "proteus"
        }
      }
    }
  },
  "mcp": {
    "filesystem": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/aaron"],
      "timeout": 10000
    },
    "git": {
      "type": "local",
      "command": ["uvx", "mcp-server-git"],
      "timeout": 10000
    },
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

exec opencode -m "proteus/proteus" "$@"
