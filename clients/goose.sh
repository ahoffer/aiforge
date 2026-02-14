#!/usr/bin/env bash
# Launches Goose AI coding agent through Gateway with MCP tool servers.
# Usage: ./goose.sh              interactive session
#        ./goose.sh run -t "msg" non-interactive single prompt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../config.env"

if [[ "$AGENT_URL" == *"ollama"* ]] || [[ "$AGENT_URL" == *":11434"* ]]; then
    echo "Error: AGENT_URL must point to Gateway, not Ollama: $AGENT_URL"
    exit 1
fi

export GOOSE_PROVIDER="${GOOSE_PROVIDER:-gateway}"
export GOOSE_MODEL="${GOOSE_MODEL:-gateway}"
export GATEWAY_API_KEY="not-needed"
export GOOSE_CLI_THEME="${GOOSE_CLI_THEME:-light}"
# Show full tool parameters without truncation. Logs land in ~/.local/state/goose/logs/
export GOOSE_DEBUG="${GOOSE_DEBUG:-1}"

if ! command -v goose &>/dev/null; then
    echo "Goose not found. Installing..."
    curl -fsSL https://github.com/block/goose/releases/latest/download/download_cli.sh | CONFIGURE=false bash
    echo "Installed."
fi

# Generate custom provider definition for Gateway.
# Goose reads JSON files from custom_providers/ to discover non-built-in providers.
# The base_url must include the full path to the completions endpoint.
GOOSE_CONFIG_DIR="${HOME}/.config/goose"
mkdir -p "$GOOSE_CONFIG_DIR/custom_providers"
cat > "$GOOSE_CONFIG_DIR/custom_providers/gateway.json" <<EOF
{
  "name": "gateway",
  "engine": "openai",
  "display_name": "Gateway",
  "api_key_env": "GATEWAY_API_KEY",
  "base_url": "${AGENT_URL}/v1/chat/completions",
  "models": [
    {
      "name": "gateway",
      "context_limit": 32768
    }
  ],
  "supports_streaming": true,
  "requires_auth": false
}
EOF

# Generate config with MCP extensions.
# Six curated tools replace the generic shell escape hatch. Devstral handles
# focused schemas well. The forgetools server runs locally via stdio.
cat > "$GOOSE_CONFIG_DIR/config.yaml" <<EOF
GOOSE_PROVIDER: gateway
GOOSE_MODEL: ${GOOSE_MODEL}
extensions:
  forgetools:
    enabled: true
    name: forgetools
    type: stdio
    cmd: python3
    args:
      - ${SCRIPT_DIR}/forgetools.py
    envs:
      SEARXNG_URL: "${SEARXNG_URL}"
EOF

if command -v curl &>/dev/null; then
    models_json="$(curl -fsS --max-time 3 "${AGENT_URL}/v1/models" 2>/dev/null || true)"
    if [[ -n "$models_json" ]] && ! grep -Eq '"id"[[:space:]]*:[[:space:]]*"gateway"' <<< "$models_json"; then
        echo "Error: AGENT_URL does not appear to be Gateway (missing model id=gateway): $AGENT_URL"
        exit 1
    fi
fi

# Support both interactive session and non-interactive run modes.
if [[ "${1:-}" == "run" ]]; then
    shift
    exec goose run --provider "$GOOSE_PROVIDER" --model "$GOOSE_MODEL" "$@"
fi

exec goose session "$@"
