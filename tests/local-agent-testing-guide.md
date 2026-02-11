# Testing and Benchmarking a Local AI Coding Agent Stack

Stack under test: Ollama (qwen3:14b) on RTX 5060 Ti 16GB, OpenCode v1.1.53, Goose v1.23.2, MCP servers (searxng, filesystem, git, shell, fetch), SearXNG and Ollama in Kubernetes.

Expected baseline performance for qwen3:14b on RTX 5060 Ti 16GB: roughly 33 tokens/second generation, roughly 943 tokens/second prompt processing at 16k context.

---

## 0. Prerequisites: Ensure Tool Calling Actually Works

Before running any benchmark, confirm that Ollama is serving qwen3:14b with enough context for tool calling. Ollama defaults to 4096 tokens of context regardless of the model's advertised limit, and agentic workflows need at least 16k.

```bash
# Create a qwen3:14b variant with 32k context
ollama run qwen3:14b
>>> /set parameter num_ctx 32768
>>> /save qwen3:14b-32k
>>> /bye
```

Verify your OpenCode config at `~/.config/opencode/config.json` includes tool support:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://<ollama-k8s-service>:11434/v1"
      },
      "models": {
        "qwen3:14b-32k": {
          "tools": true
        }
      }
    }
  }
}
```

Verify your Goose config at `~/.config/goose/config.yaml`:

```yaml
GOOSE_MODEL: qwen3:14b-32k
GOOSE_PROVIDER: openai
OPENAI_API_KEY: unused
OPENAI_HOST: http://<ollama-k8s-service>:11434
```

---

## 1. Tool Calling Reliability Tests

These 15 prompts are designed to exercise distinct tool-calling patterns and expose common failure modes. Run each prompt through both OpenCode and Goose, then score: did the model call the correct tool with correct arguments, or did it hallucinate output?

### Single Tool, Simple Args

```
Prompt 1: "List all files in /tmp"
Expected: Single filesystem list_directory call on /tmp.
Failure mode: Model prints a made-up file listing instead of calling the tool.
```

```
Prompt 2: "What is the current git branch in /home/aaron/Downloads/agents?"
Expected: Single git tool call (git branch or git status).
Failure mode: Model guesses "main" without checking.
```

```
Prompt 3: "Read the contents of /etc/hostname"
Expected: Single file read tool call.
Failure mode: Model fabricates a hostname.
```

### Single Tool, Complex Args

```
Prompt 4: "Search the web for 'Kubernetes pod eviction policy best practices 2026'"
Expected: Single searxng/web search call with the full query string.
Failure mode: Model answers from training data without searching.
```

```
Prompt 5: "Find all Python files under /home/aaron/projects that contain the string 'async def'"
Expected: Shell tool call running grep or find+grep with correct flags.
Failure mode: Model invents file paths and code snippets.
```

```
Prompt 6: "Fetch the raw content of https://httpbin.org/json and show me the keys in the response"
Expected: fetch tool call to the URL, then text processing of the JSON.
Failure mode: Model recites what it thinks httpbin returns from training data.
```

### Multi-Tool Chaining

```
Prompt 7: "Create a directory /tmp/test-agent, then create a file hello.py inside it that prints 'hello world', then run it with python3"
Expected: Three sequential tool calls (mkdir, write file, shell exec). Order matters.
Failure mode: Model tries to do everything in one tool call, or skips the mkdir step.
```

```
Prompt 8: "Check what git remotes are configured in /home/aaron/Downloads/agents, then fetch the README from the first remote's URL"
Expected: git tool call to list remotes, then a fetch tool call to the URL found.
Failure mode: Model cannot chain the output of tool 1 into tool 2.
```

```
Prompt 9: "Search the web for the current Python 3 stable release version, then create a file /tmp/python-version.txt containing that version number"
Expected: Web search, parse result, write file with extracted version.
Failure mode: Model writes a stale version from training data without searching.
```

### Deciding NOT to Use a Tool

```
Prompt 10: "What is 2 + 2?"
Expected: Direct answer "4" with no tool calls.
Failure mode: Model calls a calculator tool or shell to compute trivial arithmetic.
```

```
Prompt 11: "Explain what the MCP protocol is in two sentences."
Expected: Direct answer from knowledge, no tool calls.
Failure mode: Model searches the web for something it should know.
```

```
Prompt 12: "What does the Python 'zip' function do?"
Expected: Direct answer, no tool calls.
Failure mode: Model reads Python docs from the filesystem or web.
```

### Parallel and Conditional Tool Use

```
Prompt 13: "Check if the file /tmp/test-output.txt exists. If it does, show its contents. If not, create it with the text 'initialized'."
Expected: File existence check, then conditional branch to read or write.
Failure mode: Model always writes without checking, or hallucinates the check result.
```

```
Prompt 14: "Search the web for the latest Ollama release version AND check which version of ollama is running locally (ollama --version). Compare them."
Expected: Two independent tool calls (web search and shell), then comparison.
Failure mode: Model serializes unnecessarily or skips one call.
```

```
Prompt 15: "In /home/aaron/Downloads/agents, list all files, then for each .py file found, show its first 5 lines"
Expected: Directory listing, then a read/head call for each Python file found.
Failure mode: Model does one listing and then fabricates file contents.
```

### Scoring Sheet

For each prompt, record:

| # | Tool Called? | Correct Tool? | Correct Args? | Correct Output? | Hallucinated? | Notes |
|---|-------------|---------------|---------------|-----------------|---------------|-------|
| 1 |             |               |               |                 |               |       |

Compute: tool call accuracy = correct tool calls / total expected tool calls.

---

## 2. Code Editing Quality Tests

Create a scratch workspace and run these tasks through both agents.

```bash
mkdir -p /tmp/agent-code-tests
cd /tmp/agent-code-tests
```

### Test A: Generate a Script from Scratch

```
Prompt: "Create a Python script called fizzbuzz.py that takes a number N as a
command-line argument and prints FizzBuzz from 1 to N. Use argparse for
argument parsing. Include a main guard."
```

Verify:

```bash
python3 /tmp/agent-code-tests/fizzbuzz.py 15
# Expected output: 1, 2, Fizz, 4, Buzz, Fizz, 7, 8, Fizz, Buzz, 11, Fizz, 13, 14, FizzBuzz
```

### Test B: Fix a Bug

Create a buggy file first:

```bash
cat > /tmp/agent-code-tests/buggy.py << 'PYEOF'
import json

def parse_config(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data["settings"]["timeout"]

def main():
    config = parse_config("config.json")
    print(f"Timeout is: {config}")

if __name__ == "__main__"
    main()
PYEOF
```

```
Prompt: "The file /tmp/agent-code-tests/buggy.py has a syntax error and will also
crash if the config file is missing or the keys don't exist. Fix all the
issues and add proper error handling."
```

Verify: the agent should fix the missing colon on the `if __name__` line and add try/except for FileNotFoundError and KeyError at minimum.

### Test C: Refactor Code

```bash
cat > /tmp/agent-code-tests/messy.py << 'PYEOF'
def p(d):
    r = []
    for i in d:
        if i["a"] > 10:
            r.append({"n": i["n"], "v": i["a"] * 2})
    return r

data = [{"n": "x", "a": 5}, {"n": "y", "a": 15}, {"n": "z", "a": 20}]
print(p(data))
PYEOF
```

```
Prompt: "Refactor /tmp/agent-code-tests/messy.py. Rename the function and all
variables to be descriptive. Add type hints. Add a docstring. Keep the
behavior identical."
```

Verify: Run both original and refactored versions, compare output.

### Test D: Add Tests

```bash
cat > /tmp/agent-code-tests/calculator.py << 'PYEOF'
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
PYEOF
```

```
Prompt: "Write pytest tests for /tmp/agent-code-tests/calculator.py. Cover normal
cases, edge cases, and the zero division error. Save them to
/tmp/agent-code-tests/test_calculator.py."
```

Verify:

```bash
cd /tmp/agent-code-tests && python3 -m pytest test_calculator.py -v
```

### Test E: Multi-File Project

```
Prompt: "In /tmp/agent-code-tests/todo-app, create a simple command-line TODO app
in Python with these files: todo.py (the Todo class with add, remove, list
methods), storage.py (save/load todos as JSON), and main.py (CLI interface
using argparse with add, remove, and list subcommands). Then run it to add
two items and list them."
```

Verify: the agent creates three files, they import each other correctly, and the CLI works.

---

## 3. Web Search Integration Tests

These require the SearXNG MCP server to be accessible.

```
Prompt 16: "Search for the CVE number of the latest critical vulnerability in
OpenSSL disclosed in 2025 or 2026. Summarize what the vulnerability is."
Expected: Web search, parse results, factual summary with a real CVE number.
```

```
Prompt 17: "Look up the current weather in Austin, TX and write a Python script
that prints a formatted weather report with the data you found."
Expected: Web search for weather, then create a script with the real data embedded.
```

```
Prompt 18: "Search for the Kubernetes 1.32 release notes and list the top 3
deprecated APIs."
Expected: Web search, finds real release notes, extracts accurate deprecation info.
```

```
Prompt 19: "Find the most popular Python web framework in 2026 according to recent
surveys, then scaffold a minimal hello-world app using that framework in
/tmp/agent-code-tests/webapp/"
Expected: Search, identify framework (likely FastAPI or Django), generate working code.
```

```
Prompt 20: "Search for 'qwen3 14b BFCL benchmark score' and report what you find."
Expected: Uses web search, returns real benchmark data rather than hallucinating numbers.
```

---

## 4. Multi-Step Agentic Tasks

These tasks require planning, multiple tool calls, error recovery, and reasoning.

### Task A: Build and Test a REST API

```
Prompt: "In /tmp/agent-code-tests/rest-api, create a minimal FastAPI app with a
/items endpoint that supports GET (list all) and POST (create new). Store
items in memory as a list. Include a requirements.txt. Install the
dependencies in a venv, start the server, test it with curl, and show me the
results."
```

Score on: correct file creation, dependency installation, server startup, successful curl test, clean shutdown.

### Task B: Debug a Failing Test Suite

```bash
mkdir -p /tmp/agent-code-tests/debug-project
cat > /tmp/agent-code-tests/debug-project/utils.py << 'PYEOF'
def flatten(nested_list):
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat

def unique(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

def chunk(items, size):
    return [items[i:i+size] for i in range(0, len(items), size)]
PYEOF

cat > /tmp/agent-code-tests/debug-project/test_utils.py << 'PYEOF'
from utils import flatten, unique, chunk

def test_flatten():
    assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]
    assert flatten([]) == []

def test_unique():
    assert unique([1, 2, 2, 3, 1]) == [1, 2, 3]
    assert unique([]) == []

def test_chunk():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk([], 3) == []
    assert chunk([1], 0) == [[1]]  # This will crash: range step of 0
PYEOF
```

```
Prompt: "Run the test suite in /tmp/agent-code-tests/debug-project. There is a
failing test. Find the bug, fix it by adding a guard to the chunk function,
and re-run the tests until they all pass."
```

Score on: runs tests, identifies the zero-step bug, fixes with a ValueError or guard, re-runs to confirm.

### Task C: Git Workflow

```bash
cd /tmp/agent-code-tests && git init git-workflow-test && cd git-workflow-test
echo "initial" > README.md
git add . && git commit -m "initial commit"
```

```
Prompt: "In /tmp/agent-code-tests/git-workflow-test, create a new branch called
feature/add-greeting, add a greet.py file that has a greet(name) function,
commit it, then switch back to main and show the git log for both branches."
```

Score on: correct branch creation, file creation, commit with message, branch switch, log display.

### Task D: Investigate and Report

```
Prompt: "Examine the /etc/os-release file, check available disk space, check
memory usage, and list running Kubernetes pods (if kubectl is available).
Write a system report to /tmp/agent-code-tests/system-report.txt."
```

Score on: number of tools called correctly, report accuracy, graceful handling if kubectl is unavailable.

---

## 5. Existing Benchmarks You Can Run

### 5a. Ollama Raw Performance (tokens/second)

```bash
# Install the benchmark tool
pip install llm-benchmark

# Run it (auto-selects models based on your RAM/VRAM)
llm_benchmark run --no-sendinfo

# Or measure a specific model manually with verbose output
ollama run qwen3:14b-32k --verbose
# Then type a prompt. After the response, you will see metrics like:
#   eval count:    128
#   eval duration: 3.84s
#   eval rate:     33.33 tokens/s

# Programmatic measurement via the API
curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen3:14b-32k",
  "prompt": "Write a Python function that implements binary search.",
  "stream": false
}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
tps = d['eval_count'] / (d['eval_duration'] / 1e9)
ttft = d['prompt_eval_duration'] / 1e9
print(f'Tokens generated: {d[\"eval_count\"]}')
print(f'Generation speed: {tps:.1f} tokens/sec')
print(f'Time to first token: {ttft:.2f}s')
print(f'Total duration: {d[\"total_duration\"] / 1e9:.2f}s')
"
```

### 5b. Berkeley Function Calling Leaderboard (BFCL)

This directly measures tool-calling accuracy on standardized tests. Qwen3-14B reportedly scores around 0.971 F1 on tool-calling tasks, making it one of the best open-source models for this purpose.

```bash
# Clone and install
git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla/berkeley-function-call-leaderboard
pip install -e .

# Start Ollama serving the model (it already is via your K8s setup)
# Point BFCL at your Ollama instance's OpenAI-compatible endpoint
cat > .env << 'EOF'
REMOTE_OPENAI_BASE_URL=http://<ollama-k8s-service>:11434/v1
REMOTE_OPENAI_API_KEY=unused
EOF

# Generate responses for a test category
bfcl generate --model qwen3:14b-32k --test-category simple --skip-server-setup

# Evaluate results
bfcl evaluate --model qwen3:14b-32k --test-category simple

# Available test categories include: simple, multiple, parallel,
# parallel_multiple, java, javascript, relevance, rest, sql, chatable
# Run "bfcl generate --help" for the full list
```

### 5c. Aider Polyglot Benchmark

Tests real code editing ability across 225 exercises in 6 languages.

```bash
# Clone repos
git clone https://github.com/Aider-AI/aider.git
cd aider
git clone https://github.com/Aider-AI/polyglot-benchmark.git

# Build the docker container (runs untrusted LLM-generated code safely)
docker build -t aider-benchmark -f benchmark/Dockerfile .

# Run the container
docker run -it -v $(pwd):/aider aider-benchmark bash

# Inside the container
pip install -e .[dev]

# Run the benchmark with your Ollama-served model.
# aider can talk to any OpenAI-compatible endpoint.
export OPENAI_API_BASE=http://<ollama-k8s-service>:11434/v1
export OPENAI_API_KEY=unused

./benchmark/benchmark.py my-qwen3-14b-run \
  --model openai/qwen3:14b-32k \
  --edit-format whole \
  --threads 1 \
  --exercises-dir polyglot-benchmark

# Results appear in tmp.benchmarks/
# The benchmark reports percent of exercises solved correctly.
```

### 5d. Goose Bench

Goose has a built-in benchmark harness. Initialize a config, edit it to point at your local model, then run.

```bash
# See all available evaluation selectors
goose bench selectors

# Create a default benchmark config
goose bench init-config -n bench-config.json

# Edit bench-config.json to point at your Ollama model:
# {
#   "models": [{
#     "provider": "openai",
#     "name": "qwen3:14b-32k",
#     "parallel_safe": true,
#     "tool_shim": {
#       "use_tool_shim": false,
#       "tool_shim_model": null
#     }
#   }],
#   "evals": [{
#     "selector": "core",
#     "post_process_cmd": null,
#     "parallel_safe": true
#   }],
#   "repeat": 1
# }

# Make sure OPENAI_HOST and OPENAI_API_KEY are set
export OPENAI_HOST=http://<ollama-k8s-service>:11434
export OPENAI_API_KEY=unused

# Run the benchmark
goose bench run -c bench-config.json

# Results are written to a directory structure under the current working dir
```

### 5e. SWE-bench Lite

A subset of 300 real GitHub issues. Requires significant resources to run the full evaluation harness, but mini-swe-agent provides a lightweight alternative.

```bash
# mini-swe-agent: 100 lines, any model, runs SWE-bench tasks
git clone https://github.com/SWE-agent/mini-swe-agent.git
cd mini-swe-agent
pip install -e .

# Point at your Ollama instance
export OPENAI_BASE_URL=http://<ollama-k8s-service>:11434/v1
export OPENAI_API_KEY=unused

# Run on a single SWE-bench instance
python run.py --model qwen3:14b-32k --instance_id <instance-id>

# For the full SWE-bench Lite evaluation harness
pip install swebench
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path predictions.jsonl \
  --max_workers 4 \
  --run_id qwen3-14b-test
```

### 5f. HumanEval

The classic 164-problem code generation benchmark.

```bash
# The evaluation harness
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -e .

# You need to generate completions for each problem using your model,
# then evaluate them. A simple generation script:
cat > generate_completions.py << 'PYEOF'
import json
import requests

def generate(prompt):
    resp = requests.post(
        "http://<ollama-k8s-service>:11434/api/generate",
        json={
            "model": "qwen3:14b-32k",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 512}
        }
    )
    return resp.json()["response"]

with open("HumanEval.jsonl") as f:
    problems = [json.loads(line) for line in f]

results = []
for p in problems:
    completion = generate(p["prompt"])
    results.append({
        "task_id": p["task_id"],
        "completion": completion
    })
    print(f"Completed {p['task_id']}")

with open("completions.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
PYEOF

python generate_completions.py
# Evaluate (runs the code in a sandbox)
evaluate_functional_correctness completions.jsonl
```

### 5g. Terminal-Bench

Tests agent performance on realistic terminal tasks. 89 curated tasks covering compiling, configuring environments, running tools, and navigating filesystems.

```bash
git clone https://github.com/laude-institute/terminal-bench.git
cd terminal-bench
# Follow the README for setup. Tasks run in Docker containers.
# This benchmark is most useful for evaluating the agent wrappers
# (Goose, OpenCode) rather than the raw model.
```

---

## 6. Performance Metrics and How to Measure Them

### 6a. Tokens Per Second (generation speed)

```bash
# Method 1: Ollama verbose mode
ollama run qwen3:14b-32k --verbose
# Type your prompt. Metrics appear after the response.

# Method 2: API with timing
curl -w "\nHTTP time: %{time_total}s\n" -s \
  http://localhost:11434/api/generate -d '{
  "model": "qwen3:14b-32k",
  "prompt": "Explain quicksort in Python.",
  "stream": false
}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
gen_tps = d['eval_count'] / (d['eval_duration'] / 1e9)
prompt_tps = d['prompt_eval_count'] / (d['prompt_eval_duration'] / 1e9)
print(f'Prompt processing: {prompt_tps:.0f} tok/s ({d[\"prompt_eval_count\"]} tokens)')
print(f'Generation: {gen_tps:.1f} tok/s ({d[\"eval_count\"]} tokens)')
"

# Method 3: Dedicated benchmark tool
pip install llm-benchmark
llm_benchmark run --no-sendinfo
```

### 6b. Time to First Token (TTFT)

```bash
# Measure TTFT using streaming mode
python3 << 'PYEOF'
import requests, time, json

start = time.perf_counter()
first_token_time = None

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "qwen3:14b-32k", "prompt": "Hello", "stream": True},
    stream=True
)

for line in resp.iter_lines():
    if line:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        data = json.loads(line)
        if data.get("done"):
            end = time.perf_counter()
            break

print(f"Time to first token: {(first_token_time - start)*1000:.0f} ms")
print(f"Total generation time: {(end - start)*1000:.0f} ms")
PYEOF
```

### 6c. Tool Call Latency

Measure the overhead of a complete tool-calling round trip. This script simulates what OpenCode and Goose do internally.

```bash
python3 << 'PYEOF'
import requests, time, json

# Step 1: Send a prompt that should trigger a tool call
tools = [{
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"}
            },
            "required": ["path"]
        }
    }
}]

start = time.perf_counter()
resp = requests.post(
    "http://localhost:11434/v1/chat/completions",
    json={
        "model": "qwen3:14b-32k",
        "messages": [{"role": "user", "content": "List files in /tmp"}],
        "tools": tools,
        "stream": False
    }
)
tool_call_time = time.perf_counter() - start

data = resp.json()
choice = data["choices"][0]
has_tool_call = choice.get("message", {}).get("tool_calls") is not None
print(f"Tool call decision time: {tool_call_time*1000:.0f} ms")
print(f"Tool call made: {has_tool_call}")
if has_tool_call:
    tc = choice["message"]["tool_calls"][0]
    print(f"Function: {tc['function']['name']}")
    print(f"Arguments: {tc['function']['arguments']}")
PYEOF
```

### 6d. End-to-End Task Completion Time

Wrap your agent invocations with timing.

```bash
# For Goose
time goose run "Create a Python file /tmp/timing-test.py that prints hello world"

# For OpenCode (if it supports non-interactive mode)
time opencode --prompt "Create a Python file /tmp/timing-test.py that prints hello world"

# Or use the script below for a more detailed breakdown
python3 << 'PYEOF'
import subprocess, time

tasks = [
    ("simple_answer", "What is 2+2?"),
    ("single_tool", "List files in /tmp"),
    ("code_gen", "Write a Python function that checks if a number is prime"),
    ("multi_tool", "Create /tmp/bench-task/hello.py that prints hello, then run it"),
    ("web_search", "Search the web for the latest Python release version"),
]

for name, prompt in tasks:
    start = time.perf_counter()
    proc = subprocess.run(
        ["goose", "run", prompt],
        capture_output=True, text=True, timeout=120
    )
    elapsed = time.perf_counter() - start
    status = "OK" if proc.returncode == 0 else "FAIL"
    print(f"{name:20s} | {elapsed:6.1f}s | {status}")
PYEOF
```

### 6e. Batch Benchmark Script

Run all the manual test prompts from Section 1 through Goose and collect timing and correctness data.

```bash
cat > /tmp/agent-code-tests/run-tool-tests.sh << 'BASH'
#!/usr/bin/env bash
set -euo pipefail

PROMPTS=(
  "List all files in /tmp"
  "What is the current git branch in /home/aaron/Downloads/agents?"
  "Read the contents of /etc/hostname"
  "What is 2 + 2?"
  "Explain what the MCP protocol is in two sentences."
  "What does the Python zip function do?"
)

echo "prompt,elapsed_seconds,exit_code" > /tmp/agent-code-tests/results.csv

for prompt in "${PROMPTS[@]}"; do
  start=$(date +%s%N)
  goose run "$prompt" > /tmp/agent-code-tests/last-output.txt 2>&1
  rc=$?
  end=$(date +%s%N)
  elapsed=$(( (end - start) / 1000000 ))
  elapsed_sec=$(echo "scale=2; $elapsed / 1000" | bc)
  safe_prompt=$(echo "$prompt" | tr ',' ';')
  echo "${safe_prompt},${elapsed_sec},${rc}" >> /tmp/agent-code-tests/results.csv
  echo "[$elapsed_sec s] (rc=$rc) $prompt"
done

echo ""
echo "Results saved to /tmp/agent-code-tests/results.csv"
BASH

chmod +x /tmp/agent-code-tests/run-tool-tests.sh
```

---

## 7. Quick Reference: What to Run First

If you only have 30 minutes, run these in order:

```bash
# 1. Verify Ollama performance baseline (2 min)
ollama run qwen3:14b-32k --verbose
# Prompt: "Write a Python function that sorts a list using merge sort."

# 2. Verify tool calling works at all (2 min)
goose run "List all files in /tmp"
# Did it actually call a tool, or did it make up a listing?

# 3. Run the tool call latency test (1 min)
# (the python3 script from section 6c above)

# 4. Run a code generation test (5 min)
goose run "Create /tmp/fizzbuzz.py with a FizzBuzz implementation and run it"

# 5. Run a multi-step test (5 min)
goose run "Create directory /tmp/test-api, write a FastAPI hello world in it,
install fastapi and uvicorn in a venv, and show me the code"

# 6. Run the BFCL simple category (15 min)
# (instructions from section 5b above)
```

---

## 8. Known Failure Modes for qwen3:14b with Tool Calling

Based on community reports and benchmark data:

1. Context overflow. If num_ctx is left at the 4096 default, tool schemas consume most of the context and the model produces garbage. Always set 16k or higher.

2. Thinking tokens in tool calls. Qwen3 models in "thinking mode" may emit `<think>...</think>` blocks inside tool call arguments, breaking JSON parsing. If this happens, try appending `/no_think` to the model name in Ollama or setting `enable_thinking: false` in the API call options.

3. Hallucinated tool names. The model may invent tool names that do not exist in the schema. This happens more often when many tools are provided simultaneously.

4. Serialization of parallel calls. When the correct behavior is to make two independent tool calls, the model may serialize them or skip the second one.

5. Over-eager tool use. The model calls tools for trivial questions it should answer directly. The "deciding NOT to use a tool" tests in Section 1 specifically target this.

---

## Sources

- [Goose Benchmarking Documentation](https://block.github.io/goose/docs/tutorials/benchmarking/)
- [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [BFCL GitHub Repository](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [Aider Polyglot Benchmark](https://aider.chat/docs/leaderboards/)
- [Aider Benchmark README](https://github.com/Aider-AI/aider/blob/main/benchmark/README.md)
- [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [SWE-bench Lite](https://www.swebench.com/lite.html)
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
- [HumanEval](https://github.com/openai/human-eval)
- [Terminal-Bench](https://www.tbench.ai/)
- [Ollama Benchmark Tool](https://github.com/aidatatools/ollama-benchmark)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [RTX 5060 Ti 16GB LLM Performance](https://www.hardware-corner.net/rtx-5060-ti-16gb-llm-january-20260112/)
- [Qwen3 14B on Ollama](https://ollama.com/library/qwen3:14b)
- [Qwen3 Blog Post](https://qwenlm.github.io/blog/qwen3/)
- [OpenCode with Ollama Guide](https://github.com/p-lemonish/ollama-x-opencode)
- [OpenCode Models Documentation](https://opencode.ai/docs/models/)
- [Docker Local LLM Tool Calling Evaluation](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/)
- [Qwen3 and Gemma3 Consumer Hardware Performance](https://boredconsultant.com/2025/06/26/Qwen3-and-Gemma3-Performance-on-Consumer-Hardware/)
