#!/usr/bin/env bash
# Runs all tests: unit tests then integration tests against the cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Unit Tests ==="
python3 -m pytest "$PROJECT_ROOT/tests/" -v

echo ""
echo "=== Stack Validation ==="
"$SCRIPT_DIR/test-stack.sh"

echo ""
echo "=== Service Integration ==="
"$SCRIPT_DIR/test-services.sh"

echo ""
echo "=== Agent Integration ==="
"$SCRIPT_DIR/test-agent.sh"

echo ""
echo "=== Web Search Smoke ==="
"$SCRIPT_DIR/test-proxy-web-search-smoke.sh"
