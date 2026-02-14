#!/usr/bin/env bash
# Bootstraps the aiforge namespace with Ollama, SearXNG, Qdrant, and Gateway.
# Creates the SearXNG secret if needed, applies all k8s manifests, and waits for rollout.
set -euo pipefail

NS=aiforge

# Build local container images
echo "Building gateway image..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHORT_SHA=$("$SCRIPT_DIR/images/gateway/build.sh")
sed -i "s|image: gateway:.*|image: gateway:${SHORT_SHA}|" "$SCRIPT_DIR/k8s/gateway.yaml"
echo "Using image gateway:${SHORT_SHA}"

# Create namespace if it doesn't exist
kubectl get ns "$NS" >/dev/null 2>&1 || kubectl create ns "$NS"

# Create SearXNG secret only if it doesn't already exist
if ! kubectl get secret searxng-secret -n "$NS" >/dev/null 2>&1; then
    SECRET=$(openssl rand -hex 32)
    kubectl create secret generic searxng-secret \
        --namespace "$NS" \
        --from-literal=SEARXNG_SECRET_KEY="$SECRET"
    echo "Created SearXNG secret."
else
    echo "SearXNG secret already exists, keeping it."
fi

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/searxng.yaml
kubectl apply -f k8s/qdrant.yaml
kubectl apply -f k8s/ollama.yaml
kubectl apply -f k8s/postgresql.yaml
kubectl apply -f k8s/langfuse.yaml
kubectl apply -f k8s/gateway.yaml

# Force gateway to pick up the new image. kubectl apply is a no-op when only
# the image tag changes and the pod spec hash stays the same.
echo "Forcing gateway rollout for image gateway:${SHORT_SHA}..."
kubectl -n "$NS" rollout restart deploy/gateway

# Wait for all deployments
echo "Waiting for deployments..."
kubectl -n "$NS" rollout status deploy/ollama
kubectl -n "$NS" rollout status deploy/searxng
kubectl -n "$NS" rollout status deploy/qdrant
kubectl -n "$NS" rollout status deploy/postgresql
kubectl -n "$NS" rollout status deploy/langfuse
kubectl -n "$NS" rollout status deploy/gateway

# Verify the new image is actually running
RUNNING_IMAGE=$(kubectl -n "$NS" get pods -l app=gateway \
    -o jsonpath='{.items[0].spec.containers[0].image}' 2>/dev/null || true)
if [[ "$RUNNING_IMAGE" != "gateway:${SHORT_SHA}" ]]; then
    echo "Error: gateway pod is running $RUNNING_IMAGE, expected gateway:${SHORT_SHA}"
    exit 1
fi
echo "Verified gateway is running gateway:${SHORT_SHA}"

# Warm up model on GPU so first request is fast. Do this explicitly in the
# installer and verify each step rather than relying on background hooks.
echo "Warming up model on GPU (this may take a minute)..."
MODEL=$(kubectl -n "$NS" get configmap ollama-config -o jsonpath='{.data.DEFAULT_MODEL}')
WARM=false

# Re-resolve on every iteration so a pod restart does not strand us on a dead name.
resolve_ollama_pod() {
  kubectl -n "$NS" get pod -l app=ollama \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

create_model_alias() {
  kubectl -n "$NS" exec "$1" -- /bin/sh -c '
      {
        echo "FROM ${DEFAULT_MODEL}"
        echo "PARAMETER temperature ${AGENT_TEMPERATURE}"
        echo "PARAMETER num_predict ${AGENT_MAX_TOKENS}"
        echo "PARAMETER top_p ${AGENT_TOP_P}"
        echo "PARAMETER repeat_penalty ${AGENT_REPEAT_PENALTY}"
        echo "PARAMETER num_ctx ${AGENT_NUM_CTX}"
        printf "SYSTEM %s\n" "${AGENT_SYSTEM_PROMPT}"
      } > /tmp/Modelfile
      ollama create "${DEFAULT_MODEL}-agent" -f /tmp/Modelfile >/dev/null
    '
}

for i in $(seq 1 18); do
  OLLAMA_POD=$(resolve_ollama_pod)
  if [ -z "$OLLAMA_POD" ]; then
    echo "  No running Ollama pod found (attempt $i/18)..."
    sleep 10
    continue
  fi

  if ! kubectl -n "$NS" exec "$OLLAMA_POD" -- ollama show "${MODEL}-agent" >/dev/null 2>&1; then
    echo "  Creating model alias ${MODEL}-agent (attempt $i/18)..."
    create_model_alias "$OLLAMA_POD" 2>/dev/null || true
    sleep 10
    continue
  fi

  if kubectl -n "$NS" exec "$OLLAMA_POD" -- ollama run "${MODEL}-agent" "hi" >/dev/null 2>&1; then
    WARM=true
    break
  fi
  echo "  Waiting for successful warmup (attempt $i/18)..."
  sleep 10
done
if $WARM; then
  echo "Model warm and ready."
else
  echo "Warning: model warmup timed out. First request may be slow."
fi

echo
echo "Done."
echo
echo "Services available at:"
echo "  SearXNG:    http://localhost:31080"
echo "  Qdrant:     http://localhost:31333"
echo "  Gateway:    http://localhost:31400"
echo "  Langfuse:   http://localhost:31300"
echo "  Ollama:     cluster-internal (port-forward: kubectl port-forward deploy/ollama 11434:11434 -n aiforge)"
echo
