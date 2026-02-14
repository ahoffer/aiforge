#!/usr/bin/env bash
# Entrypoint for the testrunner container. Dispatches test suites based
# on the SUITE env var. Defaults to running unit, integration, and
# toolcalling suites (bench excluded because it is slow).
set -euo pipefail

SUITE="${SUITE:-all}"

# Gate all suites on cluster health
if [ "${SKIP_PREFLIGHT:-}" != "1" ]; then
    /app/tests/preflight.sh
fi

run_unit() {
    echo "=== Unit Tests ==="
    python3 -m pytest /app/tests/ -v
}

run_integration() {
    echo "=== Stack Validation ==="
    /app/tests/test-stack.sh

    echo ""
    echo "=== Service Integration ==="
    /app/tests/test-services.sh

    echo ""
    echo "=== Agent Integration ==="
    /app/tests/test-agent.sh
}

run_toolcalling() {
    echo "=== Tool Calling ==="
    python3 /app/tests/test-tool-calling.py
}

run_bench() {
    echo "=== Ollama Benchmark ==="
    /app/tests/bench-ollama.sh
}

case "$SUITE" in
    unit)
        run_unit
        ;;
    integration)
        run_integration
        ;;
    toolcalling)
        run_toolcalling
        ;;
    bench)
        run_bench
        ;;
    all)
        run_unit
        echo ""
        run_integration
        echo ""
        run_toolcalling
        ;;
    *)
        echo "Unknown suite: $SUITE" >&2
        echo "Valid suites: unit, integration, toolcalling, bench, all" >&2
        exit 1
        ;;
esac
