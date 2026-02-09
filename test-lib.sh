#!/usr/bin/env bash
# Shared test infrastructure for test-stack.sh and test-services.sh.
# Source this file after setting SCRIPT_DIR. Provides URL defaults,
# pass/fail counters, report(), print_header(), and print_summary().

set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://localhost:31434}"
SEARXNG_URL="${SEARXNG_URL:-http://localhost:31080}"
QDRANT_URL="${QDRANT_URL:-http://localhost:31333}"
OPENWEBUI_URL="${OPENWEBUI_URL:-http://localhost:31380}"

passed=0
failed=0
totalChecks=0

# Prints a pass or fail line and updates counters.
report() {
    local checkName="$1"
    local success="$2"
    totalChecks=$((totalChecks + 1))
    if [ "$success" = "true" ]; then
        passed=$((passed + 1))
        printf "  [PASS] %s\n" "$checkName"
    else
        failed=$((failed + 1))
        printf "  [FAIL] %s\n" "$checkName"
    fi
}

# Prints a banner header with the given title and URL summary.
print_header() {
    local title="$1"
    echo "================================================"
    echo "  $title"
    echo "================================================"
    echo ""
    echo "Ollama URL:      $OLLAMA_URL"
    echo "SearXNG URL:     $SEARXNG_URL"
    echo "Qdrant URL:      $QDRANT_URL"
    echo "Open WebUI URL:  $OPENWEBUI_URL"
    echo ""
}

# Prints pass/fail summary and exits with appropriate code.
print_summary() {
    echo "================================================"
    echo "  Summary"
    echo "================================================"
    echo ""
    printf "  Passed: %d / %d\n" "$passed" "$totalChecks"
    printf "  Failed: %d / %d\n" "$failed" "$totalChecks"
    echo ""

    if [ "$failed" -eq 0 ]; then
        echo "  All checks passed."
        exit 0
    else
        echo "  Some checks failed. Review output above."
        exit 1
    fi
}
