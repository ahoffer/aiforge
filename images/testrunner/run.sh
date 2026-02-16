#!/usr/bin/env bash
# Entrypoint for the testrunner container. Dispatches test suites based
# on the SUITE env var. Defaults to running unit, integration, and
# toolcalling suites (bench excluded because it is slow).
set -uo pipefail

SUITE="${SUITE:-all}"
PASS=""
FAIL=""
START=$(date +%s)

logj() {
    local level="$1"; shift
    local msg="$*"
    # Strip box-drawing characters and compress whitespace
    msg=$(echo "$msg" | sed 's/[═╔╗╚╝║]//g' | sed 's/  */ /g' | sed 's/^ //; s/ $//')
    msg="${msg//\\/\\\\}"
    msg="${msg//\"/\\\"}"
    printf '{"timestamp":"%s","level":"%s","msg":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)" "$level" "$msg"
}

run_suite() {
    local name="$1"; shift
    logj info "suite=$name starting"
    if "$@" 2>&1 | /app/logfmt; then
        PASS="$PASS $name"
        logj info "suite=$name passed"
    else
        FAIL="$FAIL $name"
        logj error "suite=$name FAILED"
    fi
}

logj info "RUN suite=$SUITE"

if [ "${SKIP_PREFLIGHT:-}" != "1" ]; then
    /app/tests/preflight.sh 2>&1 | /app/logfmt || { logj error "Preflight failed"; exit 1; }
fi

case "$SUITE" in
    unit)
        run_suite unit /app/pytest-compact /app/tests/
        ;;
    integration)
        run_suite stack /app/tests/test-stack.sh
        run_suite services /app/tests/test-services.sh
        run_suite agent /app/tests/test-agent.sh
        run_suite nativetools /app/tests/test-native-tools.sh
        ;;
    toolcalling)
        run_suite toolcalling python3 /app/tests/test-tool-calling.py
        run_suite goose python3 /app/tests/test-goose-readiness.py
        ;;
    bench)
        run_suite bench /app/tests/bench-ollama.sh
        ;;
    all)
        run_suite unit /app/pytest-compact /app/tests/
        run_suite stack /app/tests/test-stack.sh
        run_suite services /app/tests/test-services.sh
        run_suite agent /app/tests/test-agent.sh
        run_suite nativetools /app/tests/test-native-tools.sh
        run_suite toolcalling python3 /app/tests/test-tool-calling.py
        run_suite goose python3 /app/tests/test-goose-readiness.py
        ;;
    *)
        logj error "Unknown suite: $SUITE. Valid: unit, integration, toolcalling, bench, all"
        exit 1
        ;;
esac

ELAPSED=$(( $(date +%s) - START ))

if [ -n "$FAIL" ]; then
    logj error "DONE elapsed=${ELAPSED}s FAILED:$FAIL"
else
    logj info "DONE elapsed=${ELAPSED}s passed:$PASS"
fi

[ -z "$FAIL" ]
