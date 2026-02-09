#!/usr/bin/env bash
# Bootstraps the ai-agent namespace with Ollama, SearXNG, Qdrant, and Open WebUI.
# Creates the SearXNG secret if needed, applies agent.yaml, and waits for rollout.
set -euo pipefail

NS=ai-agent

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

# Apply the manifest
kubectl apply -f agent.yaml

# Wait for all deployments
echo "Waiting for deployments..."
kubectl -n "$NS" rollout status deploy/ollama
kubectl -n "$NS" rollout status deploy/searxng
kubectl -n "$NS" rollout status deploy/qdrant
kubectl -n "$NS" rollout status deploy/open-webui

echo
echo "Done."
echo
echo "Services available at:"
echo "  Ollama:     http://localhost:31434"
echo "  SearXNG:    http://localhost:31080"
echo "  Qdrant:     http://localhost:31333"
echo "  Open WebUI: http://localhost:31380"
echo
