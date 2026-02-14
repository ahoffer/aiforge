#!/usr/bin/env bash
# Validates Qdrant and SearXNG services with functional tests
# beyond simple health checks. Tests collection CRUD and SearXNG search.
# Ollama is cluster-internal and tested indirectly through Gateway.
# Exit codes: 0 if all checks pass, 1 if any check fails.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test.env"

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

# ---- Summary ----
print_summary
