#!/usr/bin/env bash
# Benchmarks Ollama inference performance for tuning agent workloads.
# Measures generation speed, prompt eval speed, time to first token,
# and tool call latency across multiple configurations.
#
# Usage:
#   ./bench-ollama.sh                        # defaults to devstral:latest
#   MODEL=qwen3:8b ./bench-ollama.sh         # benchmark a different model
#   ./bench-ollama.sh | tee bench-results.txt # save for comparison
#   ./bench-ollama.sh | column -ts $'\t'      # pretty-print columns

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../defaults.sh"

MODEL="${MODEL:-devstral:latest}"
TIMEOUT="${TIMEOUT:-120}"

# ---- Header ----
echo "================================================"
echo "  Ollama Performance Benchmark"
echo "================================================"
echo ""
echo "Model:      $MODEL"
echo "Ollama URL: $OLLAMA_URL"
echo "Timestamp:  $(date -Iseconds)"
echo ""

# ---- Helpers ----

# Sends a generation request and prints a tab-separated result line.
# Arguments: label, num_predict. Prompt is read from stdin.
bench_generate() {
    local label="$1"
    local numPredict="$2"

    local payload
    payload=$(python3 -c "
import json, sys
prompt = sys.stdin.read()
print(json.dumps({
    'model': '$MODEL',
    'prompt': prompt,
    'stream': False,
    'options': {'num_predict': $numPredict}
}))
" 2>/dev/null)

    local responseJson
    responseJson=$(curl -s --max-time "$TIMEOUT" "$OLLAMA_URL/api/generate" \
        -d "$payload" 2>/dev/null || echo "{}")

    python3 -c "
import sys, json
label = '$label'
try:
    d = json.loads(sys.stdin.read())
    evalCount = d.get('eval_count', 0)
    evalNs = d.get('eval_duration', 0)
    promptCount = d.get('prompt_eval_count', 0)
    promptNs = d.get('prompt_eval_duration', 0)
    totalNs = d.get('total_duration', 0)

    genTps = evalCount / (evalNs / 1e9) if evalNs > 0 else 0
    promptTps = promptCount / (promptNs / 1e9) if promptNs > 0 else 0
    totalMs = totalNs / 1e6

    print(f'{label}\t{evalCount}\t{genTps:.1f}\t{promptCount}\t{promptTps:.1f}\t{totalMs:.0f}')
except Exception as e:
    print(f'{label}\t0\t0.0\t0\t0.0\t0\tERROR: {e}')
" <<< "$responseJson"
}

# Measures time to first token via streaming mode.
# Arguments: label. Prompt is read from stdin.
bench_ttft() {
    local label="$1"
    local prompt
    prompt=$(cat)

    python3 << PYEOF
import json, time, urllib.request

label = "$label"
prompt = $(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$prompt")

try:
    payload = json.dumps({
        'model': '$MODEL',
        'prompt': prompt,
        'stream': True,
        'options': {'num_predict': 20}
    }).encode()

    req = urllib.request.Request(
        '$OLLAMA_URL/api/generate',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )

    start = time.monotonic()
    found = False
    with urllib.request.urlopen(req, timeout=$TIMEOUT) as resp:
        while True:
            rawLine = resp.readline()
            if not rawLine:
                break
            chunk = rawLine.decode().strip()
            if not chunk:
                continue
            d = json.loads(chunk)
            # Accept either response or thinking tokens as first output
            if d.get('response', '') or d.get('thinking', ''):
                ttft = (time.monotonic() - start) * 1000
                print(f'{label}\t{ttft:.0f}')
                found = True
                # Drain remaining response to avoid broken pipe
                while resp.readline():
                    pass
                break
    if not found:
        print(f'{label}\t0\tERROR: no response chunks')
except Exception as e:
    print(f'{label}\t0\tERROR: {e}')
PYEOF
}

# Measures tool call round-trip latency via the OpenAI-compatible endpoint.
# Arguments: label
bench_tool_call() {
    local label="$1"

    local payload
    payload=$(cat <<'JSONEOF'
{
    "model": "MODEL_PLACEHOLDER",
    "messages": [
        {"role": "user", "content": "List the files in /tmp"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list"
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    ],
    "stream": false
}
JSONEOF
    )
    payload="${payload//MODEL_PLACEHOLDER/$MODEL}"

    local startNs endNs elapsedMs
    startNs=$(date +%s%N)
    local responseJson
    responseJson=$(curl -s --max-time "$TIMEOUT" "$OLLAMA_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null || echo "{}")
    endNs=$(date +%s%N)
    elapsedMs=$(( (endNs - startNs) / 1000000 ))

    python3 -c "
import sys, json
label = '$label'
ms = '$elapsedMs'
try:
    d = json.loads(sys.stdin.read())
    choices = d.get('choices', [])
    if choices:
        msg = choices[0].get('message', {})
        toolCalls = msg.get('tool_calls', [])
        if toolCalls:
            funcName = toolCalls[0].get('function', {}).get('name', '')
            print(f'{label}\t{ms}\t{funcName}\tyes')
        else:
            print(f'{label}\t{ms}\t-\tno (text response)')
    else:
        print(f'{label}\t{ms}\t-\tno (empty response)')
except Exception as e:
    print(f'{label}\t{ms}\t-\tERROR: {e}')
" <<< "$responseJson"
}

# ---- Pre-warm ----
echo "Pre-warming model..."
warmupJson=$(curl -s --max-time "$TIMEOUT" "$OLLAMA_URL/api/generate" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"hi\", \"stream\": false, \"options\": {\"num_predict\": 1}}" \
    2>/dev/null || echo "{}")

warmupOk=$(python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    print('ok' if d.get('done') else 'fail')
except Exception:
    print('fail')
" <<< "$warmupJson" || echo "fail")

if [ "$warmupOk" = "ok" ]; then
    echo "Model loaded and ready."
else
    echo "WARNING: warm-up failed. Results may include model load time."
fi
echo ""

# ---- Generation benchmarks ----
printf 'BENCHMARK\tGEN_TOKENS\tGEN_TOK/S\tPROMPT_TOKENS\tPROMPT_TOK/S\tTOTAL_MS\n'

echo "Say hello in three words" | bench_generate "gen-short-50" 50
echo "Explain how a hash table works" | bench_generate "gen-medium-200" 200
echo "Write a detailed tutorial on implementing a linked list in Python" | bench_generate "gen-long-500" 500

# Long prompt eval uses a 500-word passage with short output to isolate prompt processing speed
cat << 'PROMPTEOF' | bench_generate "prompt-eval-long" 50
The history of computing stretches back thousands of years, from the abacus used in ancient civilizations to the modern supercomputers that can perform trillions of calculations per second. The first mechanical calculators appeared in the 17th century, with Blaise Pascal creating the Pascaline in 1642 and Gottfried Wilhelm Leibniz developing the Stepped Reckoner in 1694. These devices could perform basic arithmetic operations and represented a significant leap forward in computational capability. In the 19th century, Charles Babbage conceived the Analytical Engine, a design that contained many elements found in modern computers, including an arithmetic logic unit, control flow through conditional branching, and memory. Ada Lovelace wrote what is considered the first computer program for this machine, making her the world's first programmer. The early 20th century saw the development of electromechanical computers. Alan Turing published his seminal paper on computable numbers in 1936, establishing the theoretical foundation for modern computing. During World War II, several electronic computers were built for military purposes, including Colossus in Britain and ENIAC in the United States. These machines used vacuum tubes and were enormous in size, filling entire rooms. The invention of the transistor in 1947 at Bell Labs revolutionized computing by replacing vacuum tubes with smaller, more reliable, and more energy-efficient components. This led to the second generation of computers in the late 1950s and early 1960s. The development of integrated circuits in the 1960s further miniaturized computing components, leading to the third generation of computers. The microprocessor, invented in the early 1970s, placed an entire CPU on a single chip. Intel's 4004, released in 1971, was the first commercially available microprocessor. This innovation made personal computers possible and led to the founding of companies like Apple and Microsoft in the mid-1970s. The IBM PC, introduced in 1981, established a standard architecture that dominated personal computing for decades. The rise of the internet in the 1990s transformed computing from a primarily standalone activity to a connected, networked experience. The World Wide Web, created by Tim Berners-Lee in 1989, made information accessible to anyone with a computer and an internet connection. Mobile computing emerged in the 2000s with smartphones and tablets, putting powerful computers in billions of pockets worldwide. Today, computing continues to evolve with artificial intelligence, quantum computing, and edge computing pushing the boundaries of what is possible.
PROMPTEOF

echo ""

# ---- Time to first token ----
printf 'BENCHMARK\tTTFT_MS\n'
echo "Hello" | bench_ttft "ttft-minimal"
echo ""

# ---- Tool call latency ----
printf 'BENCHMARK\tLATENCY_MS\tFUNCTION\tTOOL_CALL\n'
bench_tool_call "tool-call"
echo ""

# ---- Summary ----
echo "================================================"
echo "  Benchmark complete"
echo "================================================"
