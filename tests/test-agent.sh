#!/usr/bin/env bash
# Integration tests for the agent HTTP API.
# Validates /health, /chat, /chat/stream endpoints.
# Exit codes: 0 if all checks pass, 1 if any check fails.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test-lib.sh"

print_header "Agent Service Integration Tests"

# ---- Health endpoint ----
echo "--- Agent: Health ---"
healthJson=$(curl -s --max-time 10 "$AGENT_URL/health" 2>/dev/null || echo "{}")
healthStatus=$(echo "$healthJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('status', 'error'))
except Exception:
    print('error')
" 2>/dev/null || echo "error")

if [ "$healthStatus" = "healthy" ] || [ "$healthStatus" = "degraded" ]; then
    report "Agent /health returns status=$healthStatus" "true"
else
    report "Agent /health returns status=$healthStatus" "false"
fi

# Check that service statuses are present
serviceCount=$(echo "$healthJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(len(d.get('services', {})))
except Exception:
    print(0)
" 2>/dev/null || echo "0")

if [ "$serviceCount" -ge 2 ] 2>/dev/null; then
    report "Agent /health reports $serviceCount services" "true"
else
    report "Agent /health reports $serviceCount services (expected 2)" "false"
fi
echo ""

# ---- Chat endpoint ----
echo "--- Agent: Chat ---"
chatStart=$(date +%s%N)
chatJson=$(curl -s --max-time 120 "$AGENT_URL/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "What is 2+2?"}' 2>/dev/null || echo "{}")
chatEnd=$(date +%s%N)
chatMs=$(( (chatEnd - chatStart) / 1000000 ))

chatResponse=$(echo "$chatJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    response = d.get('response', '')
    convId = d.get('conversation_id', '')
    searchCount = d.get('search_count', 0)
    if response:
        print(f'ok|{searchCount}|{convId[:8]}|{response[:100]}')
    else:
        print(f'empty|0||')
except Exception as e:
    print(f'error||{e}|')
" 2>/dev/null || echo "error|||")

IFS='|' read -r chatStatus chatSearchCount chatConvId chatText <<< "$chatResponse"

if [ "$chatStatus" = "ok" ]; then
    report "Agent /chat responds (${chatMs}ms)" "true"
    echo "         Searches: $chatSearchCount"
    echo "         Response: $chatText"
else
    report "Agent /chat responds (${chatMs}ms, status=$chatStatus)" "false"
    echo "         $chatText"
fi
echo ""

# ---- Chat followup with conversation_id ----
echo "--- Agent: Chat Followup ---"
if [ "$chatStatus" = "ok" ] && [ -n "$chatConvId" ]; then
    followupJson=$(curl -s --max-time 120 "$AGENT_URL/chat" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"What did I just ask you?\", \"conversation_id\": \"${chatConvId}\"}" \
        2>/dev/null || echo "{}")
    followupOk=$(echo "$followupJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('true' if d.get('response') else 'false')
except Exception:
    print('false')
" 2>/dev/null || echo "false")
    report "Agent /chat followup with conversation_id" "$followupOk"
else
    report "Agent /chat followup (skipped, no conversation_id)" "true"
fi
echo ""

# ---- Streaming endpoint ----
echo "--- Agent: Chat Stream ---"
streamStart=$(date +%s%N)
streamOutput=$(curl -s --max-time 120 "$AGENT_URL/chat/stream" \
    -H "Content-Type: application/json" \
    -d '{"message": "Say hello"}' 2>/dev/null || echo "")
streamEnd=$(date +%s%N)
streamMs=$(( (streamEnd - streamStart) / 1000000 ))

hasNodeEvent=$(echo "$streamOutput" | grep -c "event: node" 2>/dev/null || true)
hasDoneEvent=$(echo "$streamOutput" | grep -c "event: done" 2>/dev/null || true)

if [ "$hasDoneEvent" -gt 0 ]; then
    report "Agent /chat/stream completes (${streamMs}ms, ${hasNodeEvent} nodes)" "true"
else
    report "Agent /chat/stream completes (${streamMs}ms)" "false"
fi
echo ""

# ---- OpenAPI docs ----
echo "--- Agent: OpenAPI Docs ---"
docsCode=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$AGENT_URL/docs" 2>/dev/null || echo "000")
if [ "$docsCode" = "200" ]; then
    report "Agent /docs serves OpenAPI UI" "true"
else
    report "Agent /docs serves OpenAPI UI (HTTP $docsCode)" "false"
fi
echo ""

# ---- Summary ----
print_summary
