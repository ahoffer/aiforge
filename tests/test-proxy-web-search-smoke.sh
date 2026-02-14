#!/usr/bin/env bash
# Smoke test that verifies Gateway executes server-side web_search for
# a time-sensitive prompt on /chat.
# Exit code: 0 on pass, 1 on failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test.env"

print_header "Gateway Web Search Smoke Test"

if ! command -v kubectl >/dev/null 2>&1; then
    echo "kubectl is required for this smoke test."
    exit 1
fi

NS="${NS:-aiforge}"
PROMPT="${PROMPT:-Who won the Super Bowl in 2026?}"
START_TIME="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "--- Trigger /chat with time-sensitive prompt ---"
payload=$(python3 -c '
import json
payload = {
    "message": "Who won the Super Bowl in 2026?"
}
print(json.dumps(payload))
')

response=$(curl -s --max-time 180 "$AGENT_URL/chat" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>/dev/null || echo "{}")

response_ok=$(echo "$response" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    has_content = bool((d.get("response") or "").strip())
    print("true" if has_content else "false")
except Exception:
    print("false")
')
report "Gateway /chat returned content" "$response_ok"

echo ""
echo "--- Verify Gateway Server-Side web_search ---"
log_lines=$(kubectl logs -n "$NS" deploy/gateway --since-time="$START_TIME" 2>/dev/null || true)

has_chat=$(echo "$log_lines" | grep -c 'POST /chat' 2>/dev/null || true)
has_proxy_search=$(echo "$log_lines" | grep -c 'Executing web_search:' 2>/dev/null || true)

if [ "$has_chat" -gt 0 ]; then
    report "Gateway received /chat request" "true"
else
    report "Gateway received /chat request" "false"
fi

if [ "$has_proxy_search" -gt 0 ]; then
    report "Gateway executed server-side web_search" "true"
else
    report "Gateway executed server-side web_search" "false"
    echo "         Prompt: $PROMPT"
    echo "         START_TIME=$START_TIME"
fi

echo ""
print_summary
