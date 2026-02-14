#!/usr/bin/env bash
# Builds the gateway container image using nerdctl. Must run on bigfish.
set -euo pipefail

REQUIRED_HOST="bigfish"
CURRENT_HOST="$(hostname)"

if [[ "$CURRENT_HOST" != "$REQUIRED_HOST" ]]; then
    echo "Error: This script must run on $REQUIRED_HOST where nerdctl and containerd are installed." >&2
    echo "Current host: $CURRENT_HOST" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building gateway:latest image..." >&2
sudo nerdctl build -t gateway:latest .

IMAGE_SHA=$(sudo nerdctl image inspect gateway:latest --format '{{.Id}}' | sed 's/sha256://')
SHORT_SHA="${IMAGE_SHA:0:12}"
sudo nerdctl tag gateway:latest "gateway:${SHORT_SHA}"
echo "Tagged gateway:${SHORT_SHA}" >&2
echo "$SHORT_SHA"
