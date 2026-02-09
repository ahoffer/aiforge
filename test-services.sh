#!/usr/bin/env bash
# Validates Qdrant and Open WebUI services with functional tests
# beyond simple health checks. Tests collection CRUD, embedding storage,
# Open WebUI model listing, and SearXNG search integration.
# Exit codes: 0 if all checks pass, 1 if any check fails.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test-lib.sh"

print_header "Service Integration Tests"

# ---- Qdrant: create collection ----
echo "--- Qdrant: Collection CRUD ---"
TEST_COLLECTION="_test_$(date +%s)"

createResponse=$(curl -s -w "\n%{http_code}" --max-time 10 \
    -X PUT "$QDRANT_URL/collections/$TEST_COLLECTION" \
    -H "Content-Type: application/json" \
    -d "{
        \"vectors\": {
            \"size\": 4,
            \"distance\": \"Cosine\"
        }
    }" 2>/dev/null || echo -e "\n000")

createBody=$(echo "$createResponse" | head -n -1)
createCode=$(echo "$createResponse" | tail -n 1)

if [ "$createCode" = "200" ]; then
    report "Qdrant create collection '$TEST_COLLECTION'" "true"
else
    report "Qdrant create collection '$TEST_COLLECTION' (HTTP $createCode)" "false"
    echo "         Response: $createBody"
fi

# ---- Qdrant: insert points ----
echo ""
echo "--- Qdrant: Insert Points ---"
insertResponse=$(curl -s -w "\n%{http_code}" --max-time 10 \
    -X PUT "$QDRANT_URL/collections/$TEST_COLLECTION/points" \
    -H "Content-Type: application/json" \
    -d "{
        \"points\": [
            {\"id\": 1, \"vector\": [0.1, 0.2, 0.3, 0.4], \"payload\": {\"text\": \"hello world\"}},
            {\"id\": 2, \"vector\": [0.5, 0.6, 0.7, 0.8], \"payload\": {\"text\": \"goodbye world\"}},
            {\"id\": 3, \"vector\": [0.9, 0.1, 0.2, 0.3], \"payload\": {\"text\": \"test vector\"}}
        ]
    }" 2>/dev/null || echo -e "\n000")

insertBody=$(echo "$insertResponse" | head -n -1)
insertCode=$(echo "$insertResponse" | tail -n 1)

if [ "$insertCode" = "200" ]; then
    report "Qdrant insert 3 points" "true"
else
    report "Qdrant insert 3 points (HTTP $insertCode)" "false"
    echo "         Response: $insertBody"
fi

# ---- Qdrant: search ----
echo ""
echo "--- Qdrant: Vector Search ---"
searchResponse=$(curl -s -w "\n%{http_code}" --max-time 10 \
    -X POST "$QDRANT_URL/collections/$TEST_COLLECTION/points/search" \
    -H "Content-Type: application/json" \
    -d "{
        \"vector\": [0.1, 0.2, 0.3, 0.4],
        \"limit\": 2,
        \"with_payload\": true
    }" 2>/dev/null || echo -e "\n000")

searchBody=$(echo "$searchResponse" | head -n -1)
searchCode=$(echo "$searchResponse" | tail -n 1)

if [ "$searchCode" = "200" ]; then
    searchOk=$(echo "$searchBody" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    results = d.get('result', [])
    if len(results) >= 1 and results[0].get('id') == 1:
        print('true')
    else:
        print('false')
except Exception:
    print('false')
" 2>/dev/null || echo "false")
    report "Qdrant vector search returns nearest match" "$searchOk"
    echo "$searchBody" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for r in d.get('result', []):
        text = r.get('payload', {}).get('text', '')
        score = r.get('score', 0)
        print(f'         id={r[\"id\"]} score={score:.4f} text=\"{text}\"')
except Exception:
    pass
" 2>/dev/null || true
else
    report "Qdrant vector search (HTTP $searchCode)" "false"
fi

# ---- Qdrant: collection info ----
echo ""
echo "--- Qdrant: Collection Info ---"
infoResponse=$(curl -s --max-time 10 "$QDRANT_URL/collections/$TEST_COLLECTION" 2>/dev/null || echo "{}")
pointCount=$(echo "$infoResponse" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('result', {}).get('points_count', 0))
except Exception:
    print(0)
" 2>/dev/null || echo "0")

if [ "$pointCount" = "3" ]; then
    report "Qdrant collection has 3 points" "true"
else
    report "Qdrant collection has 3 points (got $pointCount)" "false"
fi

# ---- Qdrant: delete test collection ----
echo ""
echo "--- Qdrant: Cleanup ---"
deleteResponse=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
    -X DELETE "$QDRANT_URL/collections/$TEST_COLLECTION" 2>/dev/null || echo "000")
if [ "$deleteResponse" = "200" ]; then
    report "Qdrant delete test collection" "true"
else
    report "Qdrant delete test collection (HTTP $deleteResponse)" "false"
fi

echo ""

# ---- SearXNG: JSON search ----
echo "--- SearXNG: JSON Search ---"
searxResponse=$(curl -s -w "\n%{http_code}" --max-time 15 \
    "$SEARXNG_URL/search?q=test&format=json" 2>/dev/null || echo -e "\n000")

searxBody=$(echo "$searxResponse" | head -n -1)
searxCode=$(echo "$searxResponse" | tail -n 1)

if [ "$searxCode" = "200" ]; then
    resultCount=$(echo "$searxBody" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(len(d.get('results', [])))
except Exception:
    print(0)
" 2>/dev/null || echo "0")
    if [ "$resultCount" -gt 0 ] 2>/dev/null; then
        report "SearXNG returns search results ($resultCount results)" "true"
    else
        report "SearXNG returns search results (0 results)" "false"
    fi
else
    report "SearXNG JSON search (HTTP $searxCode)" "false"
fi

echo ""

# ---- Open WebUI: health ----
echo "--- Open WebUI: Health ---"
webuiHealth=$(curl -s --max-time 10 "$OPENWEBUI_URL/health" 2>/dev/null || echo "{}")
webuiHealthOk=$(echo "$webuiHealth" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    # Open WebUI returns {\"status\": true} on health endpoint
    print('true' if d.get('status') else 'false')
except Exception:
    print('false')
" 2>/dev/null || echo "false")

if [ "$webuiHealthOk" = "true" ]; then
    report "Open WebUI health check" "true"
else
    report "Open WebUI health check" "false"
    echo "         Response: $webuiHealth"
fi

echo ""

# ---- Open WebUI: Ollama connectivity via cluster DNS ----
# Open WebUI API endpoints require auth even with WEBUI_AUTH=false,
# so we verify the integration by confirming Ollama is reachable
# from inside the cluster via the same DNS name Open WebUI uses.
echo "--- Open WebUI: Ollama Connectivity ---"
ollamaViaCluster=$(kubectl exec -n ai-agent deploy/open-webui -- \
    curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    "http://ollama:11434/" 2>/dev/null || echo "000")

if [ "$ollamaViaCluster" = "200" ]; then
    report "Open WebUI can reach Ollama via cluster DNS" "true"
else
    report "Open WebUI can reach Ollama via cluster DNS (HTTP $ollamaViaCluster)" "false"
fi

# Also verify Open WebUI serves its frontend
webuiFrontend=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
    "$OPENWEBUI_URL/" 2>/dev/null || echo "000")
if [ "$webuiFrontend" = "200" ]; then
    report "Open WebUI serves frontend" "true"
else
    report "Open WebUI serves frontend (HTTP $webuiFrontend)" "false"
fi

# ---- Embedding model availability ----
echo ""
echo "--- Ollama: Embedding Model ---"
embedModelFound=$(curl -s --max-time 10 "$OLLAMA_URL/api/tags" 2>/dev/null \
    | python3 -c "
import sys, json
try:
    tags = json.load(sys.stdin)
    names = [m['name'] for m in tags.get('models', [])]
    found = any('nomic-embed-text' in n for n in names)
    print('true' if found else 'false')
except Exception:
    print('false')
" 2>/dev/null || echo "false")

if [ "$embedModelFound" = "true" ]; then
    report "Embedding model nomic-embed-text available" "true"
else
    report "Embedding model nomic-embed-text available" "false"
fi

# ---- Ollama embedding API ----
echo ""
echo "--- Ollama: Embedding API ---"
embedJson=$(curl -s --max-time 30 "$OLLAMA_URL/api/embed" \
    -d '{"model":"nomic-embed-text","input":"test embedding"}' 2>/dev/null || echo "{}")

embedOk=$(echo "$embedJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    embeddings = d.get('embeddings', [])
    if embeddings and len(embeddings[0]) > 0:
        print(f'true|{len(embeddings[0])}')
    else:
        print('false|0')
except Exception:
    print('false|0')
" 2>/dev/null || echo "false|0")

IFS='|' read -r embedSuccess embedDim <<< "$embedOk"
if [ "$embedSuccess" = "true" ]; then
    report "Ollama /api/embed returns vectors (dim=$embedDim)" "true"
else
    report "Ollama /api/embed returns vectors" "false"
fi

# ---- Agent model alias ----
echo ""
echo "--- Ollama: Agent Model Alias ---"
agentAliasJson=$(curl -s --max-time 10 "$OLLAMA_URL/api/show" \
    -d '{"model":"qwen3:14b-agent"}' 2>/dev/null || echo "{}")

aliasOk=$(echo "$agentAliasJson" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    mf = d.get('modelfile', '')
    hasSystem = 'SYSTEM' in mf
    hasTemp = 'temperature' in mf
    hasPredict = 'num_predict' in mf
    if hasSystem and hasTemp and hasPredict:
        print('true')
    else:
        missing = []
        if not hasSystem: missing.append('SYSTEM')
        if not hasTemp: missing.append('temperature')
        if not hasPredict: missing.append('num_predict')
        print(f'false|missing: {\", \".join(missing)}')
except Exception as e:
    print(f'false|{e}')
" 2>/dev/null || echo "false|parse error")

if [ "${aliasOk%%|*}" = "true" ]; then
    report "Agent alias qwen3:14b-agent has tuned parameters" "true"
else
    report "Agent alias qwen3:14b-agent has tuned parameters" "false"
    echo "         ${aliasOk#*|}"
fi

# ---- Open WebUI config verification ----
echo ""
echo "--- Open WebUI: Integration Config ---"
webuiEnv=$(kubectl exec -n ai-agent deploy/open-webui -- env 2>/dev/null || echo "")

checkEnv() {
    local varName="$1"
    local expectedVal="$2"
    local actualVal
    actualVal=$(echo "$webuiEnv" | grep "^${varName}=" | head -1 | cut -d= -f2-)
    if [ -z "$actualVal" ]; then
        report "Open WebUI env $varName is set" "false"
        echo "         Not set"
        return
    fi
    if [ -n "$expectedVal" ] && [ "$actualVal" != "$expectedVal" ]; then
        report "Open WebUI env $varName=$expectedVal" "false"
        echo "         Actual: $actualVal"
        return
    fi
    report "Open WebUI env $varName=$actualVal" "true"
}

checkEnv "VECTOR_DB" "qdrant"
checkEnv "QDRANT_URI" "http://qdrant:6333"
checkEnv "RAG_EMBEDDING_ENGINE" "ollama"
checkEnv "RAG_EMBEDDING_MODEL" "nomic-embed-text"
checkEnv "DEFAULT_MODELS" "qwen3:14b-agent"

# ---- Open WebUI → Qdrant connectivity ----
echo ""
echo "--- Open WebUI: Qdrant Connectivity ---"
qdrantViaCluster=$(kubectl exec -n ai-agent deploy/open-webui -- \
    curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    "http://qdrant:6333/healthz" 2>/dev/null || echo "000")

if [ "$qdrantViaCluster" = "200" ]; then
    report "Open WebUI can reach Qdrant via cluster DNS" "true"
else
    report "Open WebUI can reach Qdrant via cluster DNS (HTTP $qdrantViaCluster)" "false"
fi

# ---- Open WebUI → SearXNG connectivity ----
echo ""
echo "--- Open WebUI: SearXNG Connectivity ---"
searxViaCluster=$(kubectl exec -n ai-agent deploy/open-webui -- \
    curl -s --max-time 10 \
    "http://searxng:8080/search?q=test&format=json" 2>/dev/null || echo "{}")

searxResultCount=$(echo "$searxViaCluster" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(len(d.get('results', [])))
except Exception:
    print(0)
" 2>/dev/null || echo "0")

if [ "$searxResultCount" -gt 0 ] 2>/dev/null; then
    report "Open WebUI can search via SearXNG ($searxResultCount results)" "true"
else
    report "Open WebUI can search via SearXNG" "false"
fi

echo ""

# ---- Summary ----
print_summary
