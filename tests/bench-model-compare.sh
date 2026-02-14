#!/usr/bin/env bash
# Model comparison benchmark for agentic tool calling.
#
# Tests candidate models against 20 prompts (12 should-search, 8 should-not-search)
# measuring tool-call JSON validity, tool-use judgment, and latency. Hits Ollama
# /api/chat directly using the same payload format as OllamaClient.chat(), then
# runs one end-to-end test per model through the full agent stack.
#
# Usage:
#   bash tests/bench-model-compare.sh                    # all default models
#   bash tests/bench-model-compare.sh qwen3:14b          # single model
#   bash tests/bench-model-compare.sh qwen3:14b qwen3:8b # specific models

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test.env"
source "$SCRIPT_DIR/../config.env"

# ---- Section 1: Preamble ----

DEFAULT_MODELS=(
    "qwen3:14b"
    "qwen3:8b"
    "qwen2.5:14b"
    "mistral-nemo:latest"
    "llama3.1:8b"
    "qwen2.5-coder:14b"
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

AGENT_SYSTEM_PROMPT="You are a coding assistant. Read relevant files before making changes. After editing, verify your changes compile or pass linting. Use web_search only when local context is insufficient, for example to check library versions or API changes beyond your training data. Be concise and direct. Avoid filler phrases."

# System prompt from graph.py used for tool-calling tests
TOOL_SYSTEM_PROMPT="You are a helpful research assistant with access to web search. Use the web_search tool when you need current information or facts you are not confident about. Cite your sources when you use search results. For simple questions you can answer confidently, respond directly without searching."

# Prompt definitions live in bench-prompts.json. A one-line JSON append
# adds a test case instead of editing three parallel arrays.
PROMPTS_FILE="$SCRIPT_DIR/bench-prompts.json"
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "FATAL: $PROMPTS_FILE not found" >&2
    exit 1
fi

# Read JSON into tab-separated lines: tag\tprompt\texpected
PROMPT_LINES=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    prompts = json.load(f)
for p in prompts:
    print(f\"{p['tag']}\t{p['prompt']}\t{p['expected']}\")
" "$PROMPTS_FILE")

PROMPT_COUNT=$(echo "$PROMPT_LINES" | wc -l)

BENCH_SUFFIX="-agent-bench"
BENCH_JSONL="${BENCH_JSONL:-/tmp/bench-model-compare-results.jsonl}"

print_header "Model Comparison Benchmark"
echo "Models:          ${MODELS[*]}"
echo "Prompts:         $PROMPT_COUNT"
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
    local createPayload
    createPayload=$(BENCH_ALIAS="$aliasName" BENCH_BASE="$baseModel" \
        BENCH_TEMP="$AGENT_TEMPERATURE" BENCH_MAX="$AGENT_MAX_TOKENS" \
        BENCH_TOPP="$AGENT_TOP_P" BENCH_REP="$AGENT_REPEAT_PENALTY" \
        BENCH_SYS="$AGENT_SYSTEM_PROMPT" \
        python3 -c "
import json, os, sys
payload = {
    'model': os.environ['BENCH_ALIAS'],
    'from': os.environ['BENCH_BASE'],
    'system': os.environ['BENCH_SYS'],
    'parameters': {
        'temperature': float(os.environ['BENCH_TEMP']),
        'num_predict': int(os.environ['BENCH_MAX']),
        'top_p': float(os.environ['BENCH_TOPP']),
        'repeat_penalty': float(os.environ['BENCH_REP']),
    },
    'stream': False,
}
json.dump(payload, sys.stdout)
" 2>/dev/null)

    createResponse=$(curl -s --max-time 120 "$OLLAMA_URL/api/create" \
        -H "Content-Type: application/json" \
        -d "$createPayload" \
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
    'think': False,
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

    while IFS=$'\t' read -r tag prompt expected; do

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

        # Append JSONL record for CI trending
        BENCH_TAG="$tag" BENCH_JV="$jsonValid" BENCH_JC="$judgmentCorrect" \
        BENCH_LAT="$latency" BENCH_SNIP="$snippet" BENCH_MOD="$model" \
        BENCH_PROMPT_TEXT="$prompt" BENCH_EXP="$expected" \
        python3 -c "
import json, os, sys, datetime
record = {
    'model': os.environ['BENCH_MOD'],
    'tag': os.environ['BENCH_TAG'],
    'prompt': os.environ['BENCH_PROMPT_TEXT'],
    'expected': os.environ['BENCH_EXP'],
    'json_valid': os.environ['BENCH_JV'] == 'true',
    'judgment_correct': os.environ['BENCH_JC'] == 'true',
    'latency_ms': int(os.environ['BENCH_LAT']),
    'snippet': os.environ['BENCH_SNIP'],
    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
print(json.dumps(record))
" >> "$BENCH_JSONL" 2>/dev/null || true
    done <<< "$PROMPT_LINES"

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
ORIGINAL_MODEL="${AGENT_MODEL}"

restore_config() {
    echo ""
    echo "  Restoring ConfigMap to ${ORIGINAL_MODEL}..."
    kubectl patch configmap gateway-config -n aiforge \
        --type merge -p "{\"data\":{\"AGENT_MODEL\":\"${ORIGINAL_MODEL}\"}}" \
        > /dev/null 2>&1 || true
    kubectl rollout restart deployment/gateway -n aiforge > /dev/null 2>&1 || true
    kubectl rollout status deployment/gateway -n aiforge --timeout=120s 2>/dev/null || true
    echo "  Gateway restored to ${ORIGINAL_MODEL}."
}
trap restore_config EXIT

declare -A e2eLatency e2eSearchCount e2eStatus

for model in "${MODELS[@]}"; do
    aliasName="${model}${BENCH_SUFFIX}"
    echo "  --- E2E: $model ---"

    echo "  Patching ConfigMap to ${aliasName}..."
    kubectl patch configmap gateway-config -n aiforge \
        --type merge -p "{\"data\":{\"AGENT_MODEL\":\"${aliasName}\"}}" \
        > /dev/null 2>&1

    echo "  Restarting agent deployment..."
    kubectl rollout restart deployment/gateway -n aiforge > /dev/null 2>&1
    kubectl rollout status deployment/gateway -n aiforge --timeout=120s 2>/dev/null || true

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

# Restoration handled by the EXIT trap set before the loop.
echo ""

# ---- Section 6: Comparison summary table ----

# Format milliseconds as "Xm Ys" or just "Ys" when under a minute
fmt_time() {
    local ms=$1
    local totalSec=$(( ms / 1000 ))
    local min=$(( totalSec / 60 ))
    local sec=$(( totalSec % 60 ))
    if [ "$min" -gt 0 ]; then
        printf "%dm %02ds" "$min" "$sec"
    else
        printf "%ds" "$sec"
    fi
}

echo "================================================"
echo "  Comparison Summary (sorted best to worst)"
echo "================================================"
echo ""
printf "  %-22s %9s %8s %8s %11s %13s %8s\n" \
    "Model" "Judgment" "JSON OK" "Avg" "Search" "NoSearch" "E2E"
printf "  %-22s %9s %8s %8s %11s %13s %8s\n" \
    "----------------------" "---------" "--------" "--------" "-----------" "-------------" "--------"

# Sort models by judgment score descending, then by avg latency ascending
sortedModels=$(for model in "${MODELS[@]}"; do
    echo "${modelJudgment[$model]} ${modelLatencySum[$model]} ${model}"
done | sort -k1,1rn -k2,2n | awk '{print $3}')

while IFS= read -r model; do
    promptCount=${modelPromptCount[$model]}
    judgScore="${modelJudgment[$model]}/${promptCount}"
    jsonScore="${modelJsonOk[$model]}/${promptCount}"

    if [ "$promptCount" -gt 0 ]; then
        avgLatency=$(( ${modelLatencySum[$model]} / promptCount ))
    else
        avgLatency=0
    fi
    avgFmt=$(fmt_time "$avgLatency")

    searchScore="${modelSearchCorrect[$model]}/${modelSearchTotal[$model]}"
    nosearchScore="${modelNosearchCorrect[$model]}/${modelNosearchTotal[$model]}"

    e2eStatusVal="${e2eStatus[$model]:-fail}"
    e2eMsVal="${e2eLatency[$model]:-0}"
    if [ "$e2eStatusVal" = "ok" ]; then
        e2eFmt=$(fmt_time "$e2eMsVal")
    else
        e2eFmt="FAILED"
    fi

    printf "  %-22s %9s %8s %8s %11s %13s %8s\n" \
        "$model" "$judgScore" "$jsonScore" "$avgFmt" \
        "$searchScore" "$nosearchScore" "$e2eFmt"
done <<< "$sortedModels"

echo ""
echo "  Judgment: model called web_search when expected and refrained when not"
echo "  JSON OK: tool_calls had valid function.name and arguments.query"
echo "  Search: correct calls out of 12 search-expected prompts"
echo "  NoSearch: correct restraint out of 8 nosearch-expected prompts"
echo ""
echo "  Per-prompt JSONL results: $BENCH_JSONL"
echo ""
