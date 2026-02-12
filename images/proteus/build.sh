#!/usr/bin/env bash
# Builds the Proteus container image using nerdctl. Must run on bigfish.
set -euo pipefail

REQUIRED_HOST="bigfish"
CURRENT_HOST="$(hostname)"

if [[ "$CURRENT_HOST" != "$REQUIRED_HOST" ]]; then
    echo "Error: This script must run on $REQUIRED_HOST where nerdctl and containerd are installed."
    echo "Current host: $CURRENT_HOST"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building proteus:latest image..."
sudo nerdctl build -t proteus:latest .
