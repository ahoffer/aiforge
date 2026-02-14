# Testing and Benchmarking the AIForge Agent Stack

Stack under test: Ollama on GPU, Gateway (LangGraph agent and OpenAI-compatible proxy), SearXNG, Qdrant, all running in Kubernetes.

---

## Running Automated Tests

All test suites run inside the cluster via the testrunner pod. No external network dependencies or hardcoded host addresses.

```bash
make test              # all suites (unit, integration, toolcalling)
make test SUITE=unit   # unit tests only (pytest, fast, no live services)
make test SUITE=integration  # stack and service integration checks
make test SUITE=toolcalling  # tool-calling prompt suite against the gateway
make test SUITE=bench  # Ollama performance benchmark (slow, excluded from "all")
```

The testrunner runs a preflight check before every suite to verify that Ollama, SearXNG, Qdrant, and the gateway are reachable. If preflight fails, fix the cluster before retrying. Set `SKIP_PREFLIGHT=1` to bypass.

### Building and deploying

```bash
make build    # build gateway and testrunner images, stamp k8s manifests
make deploy   # build, apply manifests, wait for rollout, verify, warmup
make warmup   # run the warmup Job to load the model into GPU memory
```

---

## Reading Test Results

### Unit tests (pytest)

Standard pytest output. Failures show the assertion diff and traceback. Tests cover gateway proxy functions, graph routing, tool dispatch, and context trimming.

### Integration tests

Each check prints `[PASS]` or `[FAIL]` with a summary count at the end. Tests verify service reachability, health endpoints, Ollama model availability, and basic agent round-trips.

### Tool-calling tests

A table shows each test name, category, pass/fail, latency, and details. Categories: `single_tool`, `no_tool`, `multi_tool`, `multi_turn`, `tool_count_stress`. A summary follows with per-category pass rates.

Results are also written to `/tmp/test-tool-calling-results.jsonl` as JSON-lines for trending. Override with `--jsonl <path>`.

### Benchmarks

`bench-ollama.sh` prints tab-separated metrics: generation tokens/sec, prompt processing tokens/sec, TTFT, and tool-call latency. Pipe to `column -ts $'\t'` for pretty columns.

`bench-model-compare.sh` tests candidate models against 20 prompts measuring tool-call JSON validity, search/no-search judgment, and latency. Results also written to `/tmp/bench-model-compare-results.jsonl`.

---

## Known Failure Modes

Based on community reports and benchmark data for small/medium open-source models with tool calling:

1. **Context overflow.** If num_ctx is left at the Ollama default of 4096, tool schemas consume most of the context and the model produces garbage. The cluster sets 16k via ConfigMap.

2. **Thinking tokens in tool calls.** Qwen3 models in "thinking mode" may emit `<think>...</think>` blocks inside tool call arguments, breaking JSON parsing. Append `/no_think` to the model name or set `enable_thinking: false` in the API call.

3. **Hallucinated tool names.** The model may invent tool names not in the schema, especially when many tools are provided. The `tool_count_stress` tests measure this ceiling.

4. **Serialization of parallel calls.** When two independent tool calls are correct, the model may serialize them or skip one.

5. **Over-eager tool use.** The model calls tools for trivial questions it can answer directly. The `no_tool` test category targets this.

---

## External Benchmarks

For deeper evaluation beyond the built-in suite, see these upstream projects:

- [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html) for standardized tool-calling accuracy
- [Aider Polyglot Benchmark](https://aider.chat/docs/leaderboards/) for code editing quality across languages
- [Goose Bench](https://block.github.io/goose/docs/tutorials/benchmarking/) for agent task completion
- [SWE-bench Lite](https://www.swebench.com/lite.html) for real GitHub issue resolution
- [HumanEval](https://github.com/openai/human-eval) for code generation
- [Terminal-Bench](https://www.tbench.ai/) for terminal-based agent tasks
