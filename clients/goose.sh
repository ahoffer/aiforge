#!/usr/bin/env bash
# Launches Goose AI coding agent against local Ollama with MCP tool servers.
# Usage: ./goose.sh              interactive session
#        ./goose.sh run -t "msg" non-interactive single prompt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../defaults.sh"
export OLLAMA_HOST
export GOOSE_PROVIDER="${GOOSE_PROVIDER:-ollama}"
export GOOSE_MODEL
export GOOSE_CLI_THEME="${GOOSE_CLI_THEME:-light}"

if ! command -v goose &>/dev/null; then
    echo "Goose not found. Installing..."
    curl -fsSL https://github.com/block/goose/releases/latest/download/download_cli.sh | CONFIGURE=false bash
    echo "Installed."
fi

# Generate config with Ollama provider and MCP extensions.
# Env vars take precedence over config.yaml values at runtime.
GOOSE_CONFIG_DIR="${HOME}/.config/goose"
mkdir -p "$GOOSE_CONFIG_DIR"
cat > "$GOOSE_CONFIG_DIR/config.yaml" <<EOF
GOOSE_PROVIDER: ollama
GOOSE_MODEL: ${GOOSE_MODEL}
extensions:
  developer:
    enabled: true
    name: developer
    type: builtin
  searxng:
    enabled: true
    name: searxng
    type: stdio
    cmd: uvx
    args:
      - mcp-searxng
    envs:
      SEARXNG_URL: "${SEARXNG_URL}"
  filesystem:
    enabled: true
    name: filesystem
    type: stdio
    cmd: npx
    args:
      - "-y"
      - "@modelcontextprotocol/server-filesystem"
      - "/home/aaron"
  git:
    enabled: true
    name: git
    type: stdio
    cmd: uvx
    args:
      - mcp-server-git
  shell:
    enabled: true
    name: shell
    type: stdio
    cmd: uvx
    args:
      - mcp-shell-server
    envs:
      ALLOW_COMMANDS: "ls,cat,head,tail,grep,find,wc,sort,uniq,diff,git,python3,node,npm,make,kubectl,curl"
  fetch:
    enabled: true
    name: fetch
    type: stdio
    cmd: uvx
    args:
      - mcp-server-fetch
EOF

# Support both interactive session and non-interactive run modes.
if [[ "${1:-}" == "run" ]]; then
    shift
    exec goose run --provider "$GOOSE_PROVIDER" --model "$GOOSE_MODEL" "$@"
fi

exec goose session "$@"
