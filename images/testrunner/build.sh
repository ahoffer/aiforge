#!/usr/bin/env bash
# Builds the testrunner container image using nerdctl. Must run on bigfish.
# Uses project root as build context so COPY paths reach gateway and tests.
set -euo pipefail

REQUIRED_HOST="bigfish"
CURRENT_HOST="$(hostname)"

if [[ "$CURRENT_HOST" != "$REQUIRED_HOST" ]]; then
    echo "Error: This script must run on $REQUIRED_HOST where nerdctl and containerd are installed." >&2
    echo "Current host: $CURRENT_HOST" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Building testrunner:latest image..." >&2
sudo nerdctl build -t testrunner:latest -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"

IMAGE_SHA=$(sudo nerdctl image inspect testrunner:latest --format '{{.Id}}' | sed 's/sha256://')
SHORT_SHA="${IMAGE_SHA:0:12}"
sudo nerdctl tag testrunner:latest "testrunner:${SHORT_SHA}"
echo "Tagged testrunner:${SHORT_SHA}" >&2
echo "$SHORT_SHA"
