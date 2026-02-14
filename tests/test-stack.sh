#!/usr/bin/env bash
# Validates the AI agent stack by checking each service endpoint.
# Ollama is cluster-internal and tested indirectly through Gateway.
# Exit codes: 0 if all checks pass, 1 if any check fails.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test.env"

print_header "AI Agent Stack Validation"

# ---- Check 1: SearXNG health ----
echo "--- SearXNG Health ---"
searxngResponse=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$SEARXNG_URL/" 2>/dev/null || echo "000")
if [ "$searxngResponse" = "200" ]; then
    report "SearXNG responding at $SEARXNG_URL" "true"
else
    report "SearXNG responding at $SEARXNG_URL (HTTP $searxngResponse)" "false"
fi
echo ""

# ---- Check 2: Qdrant health ----
echo "--- Qdrant Health ---"
qdrantResponse=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$QDRANT_URL/healthz" 2>/dev/null || echo "000")
if [ "$qdrantResponse" = "200" ]; then
    report "Qdrant responding at $QDRANT_URL" "true"
else
    report "Qdrant responding at $QDRANT_URL (HTTP $qdrantResponse)" "false"
fi
echo ""

# ---- Check 3: Agent service health ----
echo "--- Agent Service Health ---"
agentJson=$(curl -s --max-time 10 "$AGENT_URL/health" 2>/dev/null || echo "{}")
agentStatus=$(echo "$agentJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    status = d.get('status', '')
    services = d.get('services', {})
    print(f'{status}|{json.dumps(services)}')
except Exception:
    print('unreachable|{}')
" 2>/dev/null || echo "unreachable|{}")

IFS='|' read -r agentHealthStatus agentServices <<< "$agentStatus"

if [ "$agentHealthStatus" = "healthy" ]; then
    report "Agent service healthy at $AGENT_URL" "true"
elif [ "$agentHealthStatus" = "degraded" ]; then
    report "Agent service degraded at $AGENT_URL" "true"
    echo "         Services: $agentServices"
else
    report "Agent service at $AGENT_URL ($agentHealthStatus)" "false"
fi
echo ""

# ---- Summary ----
print_summary
