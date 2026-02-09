#!/usr/bin/env bash
# Interactive model switcher for the agent stack.
# Sets AGENT_MODEL and MODEL so launchers and bench scripts use the chosen model.
#
# Usage:
#   ./switch-model.sh                      # pick a model, print confirmation
#   ./switch-model.sh ./bench-ollama.sh    # pick a model, run benchmark with it
#   source switch-model.sh                 # pick a model, set vars in current shell

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/agent-env.sh"

# Fetch model list from Ollama
modelsJson=$(curl -sf --max-time 10 "$OLLAMA_URL/api/tags" 2>/dev/null)
if [ $? -ne 0 ] || [ -z "$modelsJson" ]; then
    echo "ERROR: cannot reach Ollama at $OLLAMA_URL"
    return 1 2>/dev/null || exit 1
fi

# Parse into parallel arrays of names and human-readable sizes
mapfile -t modelNames < <(python3 -c "
import json, sys
tags = json.loads(sys.stdin.read())
for m in sorted(tags.get('models', []), key=lambda x: x['name']):
    print(m['name'])
" <<< "$modelsJson")

mapfile -t modelSizes < <(python3 -c "
import json, sys
tags = json.loads(sys.stdin.read())
for m in sorted(tags.get('models', []), key=lambda x: x['name']):
    sizeBytes = m.get('size', 0)
    if sizeBytes >= 1_000_000_000:
        print(f'{sizeBytes / 1_000_000_000:.1f} GB')
    else:
        print(f'{sizeBytes / 1_000_000:.0f} MB')
" <<< "$modelsJson")

if [ ${#modelNames[@]} -eq 0 ]; then
    echo "No models found in Ollama."
    return 1 2>/dev/null || exit 1
fi

# Find longest model name for alignment
maxLen=0
for name in "${modelNames[@]}"; do
    (( ${#name} > maxLen )) && maxLen=${#name}
done

echo ""
echo "Available models:"
for i in "${!modelNames[@]}"; do
    num=$((i + 1))
    printf "  %2d) %-${maxLen}s  (%s)\n" "$num" "${modelNames[$i]}" "${modelSizes[$i]}"
done
echo ""

# Read selection
count=${#modelNames[@]}
while true; do
    read -rp "Select model [1-$count]: " selection
    if [[ "$selection" =~ ^[0-9]+$ ]] && (( selection >= 1 && selection <= count )); then
        break
    fi
    echo "Invalid selection, try again."
done

chosen="${modelNames[$((selection - 1))]}"
export AGENT_MODEL="$chosen"
export MODEL="$chosen"

echo ""
echo "Selected: $chosen"

# If arguments were passed, exec the command with the model set
if [ $# -gt 0 ]; then
    echo "Running: $*"
    echo ""
    exec "$@"
fi

# When sourced, vars are already in the caller's environment.
# When executed directly, remind the user.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo ""
    echo "AGENT_MODEL and MODEL are set in this subshell only."
    echo "To set them in your current shell, run: source switch-model.sh"
fi
