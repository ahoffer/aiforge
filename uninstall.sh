#!/usr/bin/env bash
# Removes all resources in the aiforge namespace.
# Persistent volumes are kept by default so model data survives reinstalls.
# Pass --purge to also delete persistent volume claims.
set -euo pipefail

NS=aiforge
PURGE=false

for arg in "$@"; do
    case "$arg" in
        --purge) PURGE=true ;;
        *)
            echo "Usage: $0 [--purge]"
            echo "  --purge  Also delete persistent volume claims (model data, configs, embeddings)"
            exit 1
            ;;
    esac
done

if ! kubectl get ns "$NS" >/dev/null 2>&1; then
    echo "Namespace $NS does not exist. Nothing to uninstall."
    exit 0
fi

echo "Removing deployments, services, configmaps, and secrets in $NS..."
kubectl delete deploy --all -n "$NS" --ignore-not-found
kubectl delete svc --all -n "$NS" --ignore-not-found
kubectl delete configmap --all -n "$NS" --ignore-not-found
kubectl delete secret --all -n "$NS" --ignore-not-found

# Wait for pods to terminate
echo "Waiting for pods to terminate..."
kubectl wait --for=delete pod --all -n "$NS" --timeout=120s 2>/dev/null || true

if [ "$PURGE" = "true" ]; then
    echo "Purging persistent volume claims..."
    kubectl delete pvc --all -n "$NS" --ignore-not-found
    echo "Deleting namespace $NS..."
    kubectl delete ns "$NS" --ignore-not-found
    echo
    echo "Fully purged. All data deleted."
else
    echo
    echo "Uninstalled. Persistent volume claims preserved:"
    kubectl get pvc -n "$NS" 2>/dev/null || echo "  (none)"
    echo
    echo "Run '$0 --purge' to also delete stored data."
fi
