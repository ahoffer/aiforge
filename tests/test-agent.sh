#!/usr/bin/env bash
# Integration tests for the agent HTTP API.
# Validates /health, /chat, /chat/stream, and /v1/chat/completions endpoints.
# Exit codes: 0 if all checks pass, 1 if any check fails.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test.env"

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

# Explicitly fail if the endpoint returns a valid but unhealthy state.
if [ "$healthStatus" = "unhealthy" ]; then
    report "Agent /health is not unhealthy" "false"
else
    report "Agent /health is not unhealthy" "true"
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
        print(f'ok|{searchCount}|{convId}|{response[:100]}')
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

# ---- OpenAI model list ----
echo "--- Agent: OpenAI Model List ---"
modelsJson=$(curl -s --max-time 10 "$AGENT_URL/v1/models" 2>/dev/null || echo "{}")
modelsOk=$(echo "$modelsJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    models = d.get('data', [])
    ids = [m.get('id') for m in models]
    print('true' if 'gateway' in ids else 'false')
except Exception:
    print('false')
" 2>/dev/null || echo "false")

if [ "$modelsOk" = "true" ]; then
    report "Agent /v1/models lists gateway model" "true"
else
    report "Agent /v1/models lists gateway model" "false"
fi
echo ""

# ---- OpenAI chat completions ----
echo "--- Agent: OpenAI Chat Completions ---"
completionsStart=$(date +%s%N)
completionsJson=$(curl -s --max-time 120 "$AGENT_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"gateway","messages":[{"role":"user","content":"What is 2+2?"}]}' \
    2>/dev/null || echo "{}")
completionsEnd=$(date +%s%N)
completionsMs=$(( (completionsEnd - completionsStart) / 1000000 ))

completionsOk=$(echo "$completionsJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    content = d.get('choices', [{}])[0].get('message', {}).get('content', '')
    print('true' if content else 'false')
except Exception:
    print('false')
" 2>/dev/null || echo "false")

if [ "$completionsOk" = "true" ]; then
    report "Agent /v1/chat/completions responds (${completionsMs}ms)" "true"
else
    report "Agent /v1/chat/completions responds (${completionsMs}ms)" "false"
fi
echo ""

# ---- OpenAI streaming completions ----
echo "--- Agent: OpenAI Streaming ---"
streamOaiStart=$(date +%s%N)
streamOaiOutput=$(curl -s --max-time 120 "$AGENT_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"gateway","messages":[{"role":"user","content":"Say hello"}],"stream":true}' \
    2>/dev/null || echo "")
streamOaiEnd=$(date +%s%N)
streamOaiMs=$(( (streamOaiEnd - streamOaiStart) / 1000000 ))

hasDataChunk=$(echo "$streamOaiOutput" | grep -c '"choices"' 2>/dev/null || true)
hasDoneMarker=$(echo "$streamOaiOutput" | grep -c '\[DONE\]' 2>/dev/null || true)

if [ "$hasDataChunk" -gt 0 ] && [ "$hasDoneMarker" -gt 0 ]; then
    report "Agent /v1/chat/completions streaming (${streamOaiMs}ms)" "true"
else
    report "Agent /v1/chat/completions streaming (${streamOaiMs}ms)" "false"
fi
echo ""

# ---- Summary ----
print_summary
