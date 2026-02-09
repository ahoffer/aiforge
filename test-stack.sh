#!/usr/bin/env bash
# Validates the AI agent stack by checking each service endpoint
# and running basic inference and tool-calling smoke tests.
# Exit codes: 0 if all checks pass, 1 if any check fails.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test-lib.sh"

MODEL="${MODEL:-qwen3:14b-16k}"

print_header "AI Agent Stack Validation"
echo "Model:           $MODEL"
echo ""

# ---- Check 1: Ollama health ----
echo "--- Ollama Health ---"
ollamaResponse=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$OLLAMA_URL/" 2>/dev/null || echo "000")
if [ "$ollamaResponse" = "200" ]; then
    report "Ollama responding at $OLLAMA_URL" "true"
else
    report "Ollama responding at $OLLAMA_URL (HTTP $ollamaResponse)" "false"
fi
echo ""

# ---- Check 2: SearXNG health ----
echo "--- SearXNG Health ---"
searxngResponse=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$SEARXNG_URL/" 2>/dev/null || echo "000")
if [ "$searxngResponse" = "200" ]; then
    report "SearXNG responding at $SEARXNG_URL" "true"
else
    report "SearXNG responding at $SEARXNG_URL (HTTP $searxngResponse)" "false"
fi
echo ""

# ---- Check 3: Qdrant health ----
echo "--- Qdrant Health ---"
qdrantResponse=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$QDRANT_URL/healthz" 2>/dev/null || echo "000")
if [ "$qdrantResponse" = "200" ]; then
    report "Qdrant responding at $QDRANT_URL" "true"
else
    report "Qdrant responding at $QDRANT_URL (HTTP $qdrantResponse)" "false"
fi
echo ""

# ---- Check 4: Open WebUI health ----
echo "--- Open WebUI Health ---"
webuiResponse=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$OPENWEBUI_URL/health" 2>/dev/null || echo "000")
if [ "$webuiResponse" = "200" ]; then
    report "Open WebUI responding at $OPENWEBUI_URL" "true"
else
    report "Open WebUI responding at $OPENWEBUI_URL (HTTP $webuiResponse)" "false"
fi
echo ""

# ---- Check 5: Model availability ----
# (renumbered from check 3 after adding Qdrant and Open WebUI)
echo "--- Model Availability ---"
tagsJson=$(curl -s --max-time 10 "$OLLAMA_URL/api/tags" 2>/dev/null || echo "{}")
modelFound=$(echo "$tagsJson" | python3 -c "
import sys, json
try:
    tags = json.load(sys.stdin)
    names = [m['name'] for m in tags.get('models', [])]
    target = '$MODEL'
    # Match with or without the ':latest' suffix
    found = any(n == target or n == target + ':latest' or n.startswith(target + ':') for n in names)
    # Also try matching just the base name
    if not found:
        found = target in names
    print('true' if found else 'false')
except Exception:
    print('false')
" 2>/dev/null || echo "false")

if [ "$modelFound" = "true" ]; then
    report "Model '$MODEL' available in Ollama" "true"
else
    report "Model '$MODEL' available in Ollama" "false"
    # Show what models are available for debugging
    echo "         Available models:"
    echo "$tagsJson" | python3 -c "
import sys, json
try:
    tags = json.load(sys.stdin)
    for m in tags.get('models', []):
        print(f'           - {m[\"name\"]}')
except Exception:
    print('           (could not parse model list)')
" 2>/dev/null || true
fi
echo ""

# ---- Pre-warm: ensure model is loaded before timed tests ----
echo "--- Pre-warming model (cold load if needed) ---"
warmupJson=$(curl -s --max-time 120 "$OLLAMA_URL/api/generate" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"hi\", \"stream\": false, \"options\": {\"num_predict\": 1}}" \
    2>/dev/null || echo "{}")
warmupOk=$(echo "$warmupJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    has_output = bool(d.get('response','').strip() or d.get('thinking','').strip())
    print('ok' if has_output or d.get('done') else 'fail')
except Exception:
    print('fail')
" 2>/dev/null || echo "fail")
if [ "$warmupOk" = "ok" ]; then
    echo "  Model loaded and ready."
else
    echo "  WARNING: Model may not be loaded. Tests may fail."
fi
echo ""

# ---- Check 6: Simple inference ----
echo "--- Inference Test ---"
inferenceStart=$(date +%s%N)
inferenceJson=$(curl -s --max-time 60 "$OLLAMA_URL/api/generate" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Say hello\", \"stream\": false, \"options\": {\"num_predict\": 30}}" \
    2>/dev/null || echo "{}")
inferenceEnd=$(date +%s%N)
inferenceMs=$(( (inferenceEnd - inferenceStart) / 1000000 ))

generatedText=$(echo "$inferenceJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    text = d.get('response', '').strip()
    if not text:
        text = d.get('thinking', '').strip()
    print(text[:100] if text else '')
except Exception:
    print('')
" 2>/dev/null || echo "")

if [ -n "$generatedText" ]; then
    report "Inference generates text (${inferenceMs}ms)" "true"
    echo "         Response: $generatedText"
    # Extract tokens per second if available
    echo "$inferenceJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    evalCount = d.get('eval_count', 0)
    evalDuration = d.get('eval_duration', 0)
    if evalCount > 0 and evalDuration > 0:
        tps = evalCount / (evalDuration / 1e9)
        print(f'         Tokens: {evalCount}, Speed: {tps:.1f} tok/s')
except Exception:
    pass
" 2>/dev/null || true
else
    report "Inference generates text (${inferenceMs}ms)" "false"
fi
echo ""

# ---- Check 7: Tool calling via OpenAI-compatible endpoint ----
echo "--- Tool Calling Test ---"
toolCallPayload=$(cat <<'JSONEOF'
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

# Substitute model name into payload
toolCallPayload="${toolCallPayload//MODEL_PLACEHOLDER/$MODEL}"

toolCallStart=$(date +%s%N)
toolCallJson=$(curl -s --max-time 60 "$OLLAMA_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$toolCallPayload" 2>/dev/null || echo "{}")
toolCallEnd=$(date +%s%N)
toolCallMs=$(( (toolCallEnd - toolCallStart) / 1000000 ))

toolCallMade=$(echo "$toolCallJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    choices = d.get('choices', [])
    if choices:
        msg = choices[0].get('message', {})
        toolCalls = msg.get('tool_calls', [])
        if toolCalls:
            tc = toolCalls[0]
            funcName = tc.get('function', {}).get('name', '')
            funcArgs = tc.get('function', {}).get('arguments', '')
            print(f'true|{funcName}|{funcArgs}')
        else:
            print('false||')
    else:
        print('false||')
except Exception as e:
    print(f'false||{e}')
" 2>/dev/null || echo "false||")

IFS='|' read -r toolCallSuccess toolFuncName toolFuncArgs <<< "$toolCallMade"

if [ "$toolCallSuccess" = "true" ]; then
    report "Tool call returned (${toolCallMs}ms)" "true"
    echo "         Function: $toolFuncName"
    echo "         Arguments: $toolFuncArgs"
else
    report "Tool call returned (${toolCallMs}ms)" "false"
    # Show raw response snippet for debugging
    echo "$toolCallJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d.get('choices', [{}])[0].get('message', {}).get('content', '')
    if msg:
        print(f'         Model responded with text instead of tool call:')
        print(f'         {msg[:200]}')
    else:
        errMsg = d.get('error', {}).get('message', '')
        if errMsg:
            print(f'         Error: {errMsg}')
except Exception:
    raw = sys.stdin.read()[:200] if hasattr(sys.stdin, 'read') else ''
    print(f'         Could not parse response')
" 2>/dev/null || true
fi
echo ""

# ---- Check 8: Token generation speed ----
# For interactive use, generation needs at least 10 tok/s to feel responsive.
# Typical reading speed is 3-4 words/sec which is roughly 4-6 tok/s,
# so 10 tok/s gives comfortable headroom.
MIN_TPS=10
echo "--- Token Speed Benchmark ---"
speedJson=$(curl -s --max-time 120 "$OLLAMA_URL/api/generate" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Write a short paragraph about the history of computing.\", \"stream\": false, \"options\": {\"num_predict\": 200}}" \
    2>/dev/null || echo "{}")

speedResult=$(echo "$speedJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    evalCount = d.get('eval_count', 0)
    evalDuration = d.get('eval_duration', 0)
    promptCount = d.get('prompt_eval_count', 0)
    promptDuration = d.get('prompt_eval_duration', 0)
    if evalCount > 0 and evalDuration > 0:
        genTps = evalCount / (evalDuration / 1e9)
        promptTps = promptCount / (promptDuration / 1e9) if promptDuration > 0 else 0
        print(f'{genTps:.1f}|{evalCount}|{promptTps:.1f}|{promptCount}')
    else:
        print('0|0|0|0')
except Exception:
    print('0|0|0|0')
" 2>/dev/null || echo "0|0|0|0")

IFS='|' read -r genTps genTokens promptTps promptTokens <<< "$speedResult"

if [ "$(echo "$genTps >= $MIN_TPS" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
    report "Token speed ${genTps} tok/s (min ${MIN_TPS})" "true"
else
    report "Token speed ${genTps} tok/s (min ${MIN_TPS})" "false"
fi
echo "         Generation: ${genTokens} tokens at ${genTps} tok/s"
echo "         Prompt eval: ${promptTokens} tokens at ${promptTps} tok/s"
echo ""

# ---- Summary ----
print_summary
