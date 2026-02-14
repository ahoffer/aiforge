#!/usr/bin/env bash
# Preflight check for the testrunner. Verifies all cluster services are
# reachable and healthy before any test suite runs. Prevents wasting time
# on test runs against a broken cluster.
set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://ollama:11434}"
SEARXNG_URL="${SEARXNG_URL:-http://searxng:8080}"
QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
AGENT_URL="${AGENT_URL:-http://gateway:8000}"
AGENT_MODEL="${AGENT_MODEL:-qwen2.5-coder:14b}"

MAX_RETRIES="${PREFLIGHT_RETRIES:-3}"
RETRY_DELAY="${PREFLIGHT_RETRY_DELAY:-5}"

failures=0

check_endpoint() {
    local name="$1"
    local url="$2"
    local attempt=1

    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url" 2>/dev/null || echo "000")
        if [ "$status" = "200" ]; then
            printf "  [OK]   %-12s %s\n" "$name" "$url"
            return 0
        fi
        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            sleep "$RETRY_DELAY"
        fi
        attempt=$((attempt + 1))
    done

    printf "  [FAIL] %-12s %s (HTTP %s after %d attempts)\n" "$name" "$url" "$status" "$MAX_RETRIES"
    failures=$((failures + 1))
    return 1
}

echo "=== Preflight Check ==="
echo ""

check_endpoint "Ollama"  "$OLLAMA_URL/" || true
check_endpoint "SearXNG" "$SEARXNG_URL/" || true
check_endpoint "Qdrant"  "$QDRANT_URL/readyz" || true
check_endpoint "Gateway" "$AGENT_URL/health" || true

# Verify Ollama has the agent model available
echo ""
modelCheck=$(curl -s --max-time 10 "$OLLAMA_URL/api/tags" 2>/dev/null || echo "{}")
modelFound=$(python3 -c "
import json, sys, os
model = os.environ.get('AGENT_MODEL', '')
try:
    d = json.loads(sys.stdin.read())
    names = [m.get('name', '').split(':')[0] for m in d.get('models', [])]
    base = model.split(':')[0]
    print('yes' if base in names else 'no')
except Exception:
    print('no')
" <<< "$modelCheck" 2>/dev/null || echo "no")

if [ "$modelFound" = "yes" ]; then
    printf "  [OK]   %-12s %s present in Ollama\n" "Model" "$AGENT_MODEL"
else
    printf "  [FAIL] %-12s %s not found in Ollama\n" "Model" "$AGENT_MODEL"
    failures=$((failures + 1))
fi

echo ""

if [ "$failures" -gt 0 ]; then
    echo "Preflight FAILED: $failures check(s) did not pass."
    echo "Fix the cluster before running tests."
    exit 1
fi

echo "Preflight passed. All services healthy."
echo ""
