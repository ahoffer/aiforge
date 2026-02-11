#!/usr/bin/env bash
# Model comparison benchmark for agentic tool calling.
#
# Tests candidate models against 12 prompts measuring tool-call JSON validity,
# tool-use judgment, and latency. Hits Ollama /api/chat directly using the
# same payload format as OllamaClient.chat(), then runs one end-to-end test
# per model through the full agent stack.
#
# Usage:
#   bash tests/bench-model-compare.sh                    # all default models
#   bash tests/bench-model-compare.sh qwen3:14b          # single model
#   bash tests/bench-model-compare.sh qwen3:14b qwen3:8b # specific models

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test-lib.sh"

# ---- Section 1: Preamble ----

DEFAULT_MODELS=(
    "qwen3:14b"
    "qwen3:8b"
    "qwen2.5:14b"
    "mistral-nemo:latest"
    "llama3.1:8b"
    "devstral:latest"
)

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# Tuning params matching k8s/ollama.yaml ConfigMap
AGENT_TEMPERATURE="0.7"
AGENT_MAX_TOKENS="2048"
AGENT_TOP_P="0.9"
AGENT_REPEAT_PENALTY="1.2"

AGENT_SYSTEM_PROMPT="Be concise and direct. Avoid filler phrases. When helping with code, ALWAYS search the web for latest documentation, API references, and code examples before answering. Do not rely on potentially outdated training data. Search first, then answer."

# System prompt from graph.py used for tool-calling tests
TOOL_SYSTEM_PROMPT="You are a helpful research assistant with access to web search. Use the web_search tool when you need current information or facts you are not confident about. Cite your sources when you use search results. For simple questions you can answer confidently, respond directly without searching."

# Prompt arrays. TAGS, PROMPTS, and EXPECTED are parallel arrays.
TAGS=(
    "A1" "A2" "A3" "A4" "A5" "A6" "A7" "A8"
    "B1" "B2" "B3" "B4"
)

PROMPTS=(
    "What were the main announcements at KubeCon North America 2025?"
    "Find the latest stable release version of Python."
    "Who won the most recent Super Bowl and what was the score?"
    "What new features were added in the Ollama 0.6 release?"
    "Search for CVEs disclosed in OpenSSL during 2025."
    "What is the current population of Austin, Texas?"
    "Search for 'Kubernetes pod eviction policy' AND 'node affinity best practices'"
    "I need current info about three things: the latest Go release, the latest Rust release, and the latest Zig release."
    "What is 7 times 13?"
    "Explain what a Python decorator does in two sentences."
    "Write a bash one-liner that counts the number of lines in a file."
    "What is the capital of France?"
)

EXPECTED=(
    "search" "search" "search" "search" "search" "search" "search" "search"
    "nosearch" "nosearch" "nosearch" "nosearch"
)

BENCH_SUFFIX="-agent-bench"

print_header "Model Comparison Benchmark"
echo "Models:          ${MODELS[*]}"
echo "Prompts:         ${#PROMPTS[@]}"
echo ""

# ---- Section 2: setup_model function ----

setup_model() {
    local baseModel="$1"
    local aliasName="${baseModel}${BENCH_SUFFIX}"

    echo "  Pulling ${baseModel}..."
    pullResponse=$(curl -s --max-time 600 "$OLLAMA_URL/api/pull" \
        -d "{\"name\": \"${baseModel}\", \"stream\": false}" 2>/dev/null || echo '{"error":"pull failed"}')

    pullError=$(echo "$pullResponse" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('error', ''))
except Exception:
    print('parse error')
" 2>/dev/null || echo "parse error")

    if [ -n "$pullError" ]; then
        echo "  WARNING: Pull may have issues: $pullError"
    fi

    echo "  Creating tuned alias ${aliasName}..."
    modelfile="FROM ${baseModel}\nPARAMETER temperature ${AGENT_TEMPERATURE}\nPARAMETER num_predict ${AGENT_MAX_TOKENS}\nPARAMETER top_p ${AGENT_TOP_P}\nPARAMETER repeat_penalty ${AGENT_REPEAT_PENALTY}\nSYSTEM ${AGENT_SYSTEM_PROMPT}"

    createResponse=$(curl -s --max-time 120 "$OLLAMA_URL/api/create" \
        -d "{\"name\": \"${aliasName}\", \"modelfile\": \"${modelfile}\", \"stream\": false}" \
        2>/dev/null || echo '{"error":"create failed"}')

    echo "  Warming up ${aliasName}..."
    curl -s --max-time 120 "$OLLAMA_URL/api/generate" \
        -d "{\"model\": \"${aliasName}\", \"prompt\": \"hi\", \"stream\": false, \"options\": {\"num_predict\": 1}}" \
        > /dev/null 2>&1 || true

    echo "  Setup complete for ${aliasName}."
}

# ---- Section 3: run_prompt function ----

run_prompt() {
    local model="$1"
    local tag="$2"
    local prompt="$3"
    local expected="$4"
    local aliasName="${model}${BENCH_SUFFIX}"

    # Build JSON payload with python3 via env vars to avoid bash escaping issues
    local payload
    payload=$(BENCH_MODEL="$aliasName" BENCH_SYSTEM="$TOOL_SYSTEM_PROMPT" BENCH_PROMPT="$prompt" \
        python3 -c "
import json, os, sys
payload = {
    'model': os.environ['BENCH_MODEL'],
    'messages': [
        {'role': 'system', 'content': os.environ['BENCH_SYSTEM']},
        {'role': 'user', 'content': os.environ['BENCH_PROMPT']},
    ],
    'tools': [
        {
            'type': 'function',
            'function': {
                'name': 'web_search',
                'description': 'Search the web for current information on any topic',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'Search query',
                        }
                    },
                    'required': ['query'],
                },
            },
        }
    ],
    'stream': False,
    'options': {'enable_thinking': False},
}
json.dump(payload, sys.stdout)
" 2>/dev/null)

    if [ -z "$payload" ]; then
        echo "0|false|false|payload build failed"
        return
    fi

    local startNs endNs latencyMs
    startNs=$(date +%s%N)
    local responseJson
    responseJson=$(curl -s --max-time 120 "$OLLAMA_URL/api/chat" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null || echo '{}')
    endNs=$(date +%s%N)
    latencyMs=$(( (endNs - startNs) / 1000000 ))

    # Extract metrics with python3 via stdin to avoid quote escaping issues
    local metrics
    metrics=$(BENCH_EXPECTED="$expected" BENCH_LATENCY="$latencyMs" \
        python3 -c "
import json, os, sys

expected = os.environ['BENCH_EXPECTED']
latency = os.environ['BENCH_LATENCY']

try:
    d = json.load(sys.stdin)
except Exception:
    print(f'{latency}|false|false|parse error')
    sys.exit(0)

message = d.get('message', {})
tool_calls = message.get('tool_calls', [])
content = message.get('content', '')

json_valid = False
snippet = ''
made_tool_call = False

if tool_calls:
    made_tool_call = True
    tc = tool_calls[0]
    func = tc.get('function', {})
    name = func.get('name', '')
    args = func.get('arguments', {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    query = ''
    if isinstance(args, dict):
        query = args.get('query', '')
    json_valid = (name == 'web_search' and isinstance(query, str) and len(query) > 0)
    snippet = query[:80] if query else name[:80]
else:
    snippet = content[:80].replace(chr(10), ' ') if content else '(empty)'

if expected == 'search':
    judgment_correct = made_tool_call and json_valid
elif expected == 'nosearch':
    judgment_correct = not made_tool_call
    json_valid = True  # not applicable, count as ok
else:
    judgment_correct = False

jv = 'true' if json_valid else 'false'
jc = 'true' if judgment_correct else 'false'
print(f'{latency}|{jv}|{jc}|{snippet}')
" <<< "$responseJson" 2>/dev/null)

    if [ -z "$metrics" ]; then
        echo "${latencyMs}|false|false|python parse error"
        return
    fi

    echo "$metrics"
}

# ---- Section 4: Main loop ----

# Associative arrays for summary
declare -A modelJsonOk modelJudgment modelLatencySum modelPromptCount
declare -A modelSearchCorrect modelSearchTotal modelNosearchCorrect modelNosearchTotal

for model in "${MODELS[@]}"; do
    echo "================================================"
    echo "  Model: $model"
    echo "================================================"
    echo ""

    setup_model "$model"
    echo ""

    modelJsonOk[$model]=0
    modelJudgment[$model]=0
    modelLatencySum[$model]=0
    modelPromptCount[$model]=0
    modelSearchCorrect[$model]=0
    modelSearchTotal[$model]=0
    modelNosearchCorrect[$model]=0
    modelNosearchTotal[$model]=0

    printf "  %-4s %-7s %-9s %7s  %s\n" "Tag" "JSON" "Judgment" "Latency" "Snippet"
    printf "  %-4s %-7s %-9s %7s  %s\n" "----" "-------" "---------" "-------" "-------"

    for i in "${!TAGS[@]}"; do
        tag="${TAGS[$i]}"
        prompt="${PROMPTS[$i]}"
        expected="${EXPECTED[$i]}"

        metrics=$(run_prompt "$model" "$tag" "$prompt" "$expected") || true
        IFS='|' read -r latency jsonValid judgmentCorrect snippet <<< "$metrics"

        # Guard against empty latency from failed prompts
        latency="${latency:-0}"

        # Update counters
        modelPromptCount[$model]=$(( ${modelPromptCount[$model]} + 1 ))
        modelLatencySum[$model]=$(( ${modelLatencySum[$model]} + latency ))

        if [ "$jsonValid" = "true" ]; then
            modelJsonOk[$model]=$(( ${modelJsonOk[$model]} + 1 ))
        fi
        if [ "$judgmentCorrect" = "true" ]; then
            modelJudgment[$model]=$(( ${modelJudgment[$model]} + 1 ))
        fi

        if [ "$expected" = "search" ]; then
            modelSearchTotal[$model]=$(( ${modelSearchTotal[$model]} + 1 ))
            if [ "$judgmentCorrect" = "true" ]; then
                modelSearchCorrect[$model]=$(( ${modelSearchCorrect[$model]} + 1 ))
            fi
        else
            modelNosearchTotal[$model]=$(( ${modelNosearchTotal[$model]} + 1 ))
            if [ "$judgmentCorrect" = "true" ]; then
                modelNosearchCorrect[$model]=$(( ${modelNosearchCorrect[$model]} + 1 ))
            fi
        fi

        # Format latency with ms suffix
        printf "  %-4s %-7s %-9s %6dms  %s\n" "$tag" "$jsonValid" "$judgmentCorrect" "$latency" "$snippet"
    done

    echo ""
done

# ---- Section 5: End-to-end agent test ----

echo "================================================"
echo "  End-to-End Agent Tests"
echo "================================================"
echo ""
echo "  Testing one prompt per model through the full agent stack."
echo "  This patches the agent ConfigMap and restarts the pod."
echo ""

E2E_PROMPT="What is the latest stable release of Go?"
ORIGINAL_MODEL="qwen3:14b-agent"

declare -A e2eLatency e2eSearchCount e2eStatus

for model in "${MODELS[@]}"; do
    aliasName="${model}${BENCH_SUFFIX}"
    echo "  --- E2E: $model ---"

    echo "  Patching ConfigMap to ${aliasName}..."
    kubectl patch configmap agent-config -n ai-agent \
        --type merge -p "{\"data\":{\"AGENT_MODEL\":\"${aliasName}\"}}" \
        > /dev/null 2>&1

    echo "  Restarting agent deployment..."
    kubectl rollout restart deployment/agent -n ai-agent > /dev/null 2>&1
    kubectl rollout status deployment/agent -n ai-agent --timeout=120s 2>/dev/null || true

    # Wait a few seconds for readiness probe
    sleep 5

    echo "  Sending prompt: ${E2E_PROMPT}"
    e2eStart=$(date +%s%N)
    e2eResponse=$(curl -s --max-time 180 "$AGENT_URL/chat" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"${E2E_PROMPT}\"}" 2>/dev/null || echo '{}')
    e2eEnd=$(date +%s%N)
    e2eMs=$(( (e2eEnd - e2eStart) / 1000000 ))

    e2eResult=$(python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    response = d.get('response', d.get('final_response', ''))
    searches = d.get('search_count', 0)
    snippet = response[:80].replace(chr(10), ' ') if response else '(empty)'
    status = 'ok' if response else 'fail'
    print(f'{status}|{searches}|{snippet}')
except Exception as e:
    print(f'fail|0|{e}')
" <<< "$e2eResponse" 2>/dev/null || echo "fail|0|python error")

    IFS='|' read -r status searches snippet <<< "$e2eResult"

    e2eLatency[$model]=$e2eMs
    e2eSearchCount[$model]=$searches
    e2eStatus[$model]=$status

    printf "  Status: %s, Searches: %s, Latency: %dms\n" "$status" "$searches" "$e2eMs"
    printf "  Snippet: %s\n" "$snippet"
    echo ""
done

# Restore original model
echo "  Restoring ConfigMap to ${ORIGINAL_MODEL}..."
kubectl patch configmap agent-config -n ai-agent \
    --type merge -p "{\"data\":{\"AGENT_MODEL\":\"${ORIGINAL_MODEL}\"}}" \
    > /dev/null 2>&1
kubectl rollout restart deployment/agent -n ai-agent > /dev/null 2>&1
kubectl rollout status deployment/agent -n ai-agent --timeout=120s 2>/dev/null || true
echo "  Agent restored to ${ORIGINAL_MODEL}."
echo ""

# ---- Section 6: Comparison summary table ----

echo "================================================"
echo "  Comparison Summary"
echo "================================================"
echo ""
printf "  %-22s %9s %8s %9s %11s %13s %5s %7s\n" \
    "Model" "Judgment" "JSON OK" "Avg (ms)" "Search" "NoSearch" "E2E" "E2E ms"
printf "  %-22s %9s %8s %9s %11s %13s %5s %7s\n" \
    "----------------------" "---------" "--------" "---------" "-----------" "-------------" "-----" "-------"

for model in "${MODELS[@]}"; do
    promptCount=${modelPromptCount[$model]}
    judgScore="${modelJudgment[$model]}/${promptCount}"
    jsonScore="${modelJsonOk[$model]}/${promptCount}"

    if [ "$promptCount" -gt 0 ]; then
        avgLatency=$(( ${modelLatencySum[$model]} / promptCount ))
    else
        avgLatency=0
    fi

    searchScore="${modelSearchCorrect[$model]}/${modelSearchTotal[$model]}"
    nosearchScore="${modelNosearchCorrect[$model]}/${modelNosearchTotal[$model]}"

    e2eStatusVal="${e2eStatus[$model]:-n/a}"
    e2eMsVal="${e2eLatency[$model]:-0}"

    printf "  %-22s %9s %8s %8dms %11s %13s %5s %6dms\n" \
        "$model" "$judgScore" "$jsonScore" "$avgLatency" \
        "$searchScore" "$nosearchScore" "$e2eStatusVal" "$e2eMsVal"
done

echo ""
echo "  Judgment: model called web_search when expected and refrained when not"
echo "  JSON OK: tool_calls had valid function.name and arguments.query"
echo "  Search: correct calls out of 8 search-expected prompts"
echo "  NoSearch: correct restraint out of 4 nosearch-expected prompts"
echo ""
