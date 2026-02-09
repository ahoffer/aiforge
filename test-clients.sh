#!/usr/bin/env bash
# Smoke-tests each agent client by sending a single prompt and checking
# for a non-error response. Skips clients that are not installed.
# Exit codes: 0 if all checks pass, 1 if any check fails.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test-lib.sh"
source "$SCRIPT_DIR/agent-env.sh"

MODEL="${MODEL:-qwen3:14b-16k}"
TIMEOUT=120

print_header "Client Smoke Tests"
echo "Model:           $MODEL"
echo ""

# Runs a client command with timeout, captures output, reports pass/fail.
# Usage: run_client_test "name" command [args...]
# The first arg after the name must be the binary so we can check installation.
run_client_test() {
    local clientName="$1"
    shift
    local clientCmd="$1"

    if ! command -v "$clientCmd" &>/dev/null; then
        report "$clientName smoke test (not installed, skipped)" "true"
        return
    fi

    local outFile errFile
    outFile=$(mktemp)
    errFile=$(mktemp)

    local exitCode=0
    timeout "$TIMEOUT" "$@" >"$outFile" 2>"$errFile" || exitCode=$?

    local outSize
    outSize=$(wc -c < "$outFile")

    if [ "$exitCode" -eq 0 ] && [ "$outSize" -gt 0 ]; then
        report "$clientName smoke test" "true"
        echo "         Response (first 120 chars):"
        head -c 120 "$outFile" | sed 's/^/         /'
        echo ""
    else
        report "$clientName smoke test (exit=$exitCode, ${outSize}B output)" "false"
        head -c 200 "$errFile" | sed 's/^/         /'
        echo ""
    fi

    rm -f "$outFile" "$errFile"
}

# ---- aider ----
echo "--- aider ---"
# Aider needs OLLAMA_API_BASE and a temp dir so it does not modify real files.
# Uses pushd/popd instead of a subshell so report() counters propagate.
export OLLAMA_API_BASE="$OLLAMA_HOST"
aiderDir=$(mktemp -d)
pushd "$aiderDir" >/dev/null
run_client_test "aider" aider \
    --model "ollama/$MODEL" \
    --message "Say hello" \
    --yes-always \
    --no-git \
    --no-stream
popd >/dev/null
rm -rf "$aiderDir"
unset OLLAMA_API_BASE
echo ""

# ---- goose ----
echo "--- goose ---"
export OLLAMA_HOST
run_client_test "goose" goose run \
    -t "Say hello" \
    --no-session \
    --provider ollama \
    --model "$MODEL"
echo ""

# ---- opencode ----
echo "--- opencode ---"
run_client_test "opencode" opencode run \
    -m "ollama/$MODEL" \
    "Say hello"
echo ""

# ---- Summary ----
print_summary
