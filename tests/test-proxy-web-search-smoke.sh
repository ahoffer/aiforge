#!/usr/bin/env bash
# Smoke test that verifies Proteus executes server-side web_search for
# a time-sensitive prompt on /v1/chat/completions.
# Exit code: 0 on pass, 1 on failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test-lib.sh"

print_header "Proteus Web Search Smoke Test"

if ! command -v kubectl >/dev/null 2>&1; then
    echo "kubectl is required for this smoke test."
    exit 1
fi

NS="${NS:-aiforge}"
PROMPT="${PROMPT:-Who won the Super Bowl in 2026?}"
START_TIME="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "--- Trigger OpenAI-Compatible Chat Completion ---"
payload=$(python3 -c '
import json
payload = {
    "model": "proteus",
    "messages": [{"role": "user", "content": "Who won the Super Bowl in 2026?"}],
    "stream": False,
}
print(json.dumps(payload))
')

response=$(curl -s --max-time 180 "$AGENT_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>/dev/null || echo "{}")

response_ok=$(echo "$response" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    msg = d.get("choices", [{}])[0].get("message", {})
    has_content = bool((msg.get("content") or "").strip())
    print("true" if has_content else "false")
except Exception:
    print("false")
')
report "Proteus /v1/chat/completions returned content" "$response_ok"

echo ""
echo "--- Verify Proteus Server-Side web_search ---"
log_lines=$(kubectl logs -n "$NS" deploy/proteus --since-time="$START_TIME" 2>/dev/null || true)

has_completion=$(echo "$log_lines" | grep -c 'POST /v1/chat/completions' 2>/dev/null || true)
has_proxy_search=$(echo "$log_lines" | grep -c 'Proxy web_search:' 2>/dev/null || true)

if [ "$has_completion" -gt 0 ]; then
    report "Proteus received /v1/chat/completions request" "true"
else
    report "Proteus received /v1/chat/completions request" "false"
fi

if [ "$has_proxy_search" -gt 0 ]; then
    report "Proteus executed server-side web_search" "true"
else
    report "Proteus executed server-side web_search" "false"
    echo "         Prompt: $PROMPT"
    echo "         START_TIME=$START_TIME"
fi

echo ""
print_summary
