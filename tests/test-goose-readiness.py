#!/usr/bin/env python3
"""Goose readiness test suite for the Gateway proxy path.

Validates that Gateway correctly handles the exact message flow Goose
produces: model discovery, multi-turn tool calling with all 6 forgetools
schemas, large conversation trimming, and streaming tool_calls. Results
are printed in a table with per-test latency.

Usage:
    python3 test-goose-readiness.py [--url URL]
"""

import argparse
import datetime
import json
import os
import sys
import time
import requests
import re


FORGETOOLS_SCHEMAS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file and return its contents with line numbers",
        "parameters": {"type": "object",
            "properties": {"path": {"type": "string", "description": "File path to read"}},
            "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file, creating parent directories if needed",
        "parameters": {"type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"}},
            "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List files and directories at the given path",
        "parameters": {"type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"},
                "recursive": {"type": "boolean", "description": "Recurse into subdirectories"}},
            "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "search_files",
        "description": "Search file contents for a regex pattern using ripgrep",
        "parameters": {"type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "path": {"type": "string", "description": "Directory to search"}},
            "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command and return stdout and stderr",
        "parameters": {"type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command"}},
            "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web via SearXNG for current information",
        "parameters": {"type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]},
    }},
]


def extractToolCalls(response):
    """Pull tool call list from OpenAI-format response."""
    try:
        return response["choices"][0]["message"].get("tool_calls", []) or []
    except (KeyError, IndexError, TypeError):
        return []


def extractContent(response):
    """Pull text content from OpenAI-format response."""
    try:
        return response["choices"][0]["message"].get("content", "") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def sanitizeSingleLine(text):
    """Normalize text for single-line table output."""
    normalized = str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    normalized = re.sub(r"[\x00-\x1f\x7f]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def makeSnippet(text, maxLen=80):
    """Return a compact single-line snippet."""
    snippet = sanitizeSingleLine(text)
    if len(snippet) > maxLen:
        return snippet[: maxLen - 3] + "..."
    return snippet


def formatCell(text, width):
    """Fit text into a fixed-width cell."""
    value = sanitizeSingleLine(text)
    if len(value) > width:
        if width <= 3:
            return value[:width]
        return value[: width - 3] + "..."
    return value


def testModelDiscovery(baseUrl):
    """GET /v1/models should list gateway model with context_window."""
    t0 = time.perf_counter()
    try:
        resp = requests.get(f"{baseUrl}/v1/models", timeout=10)
        elapsed = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}", elapsed

        body = resp.json()
        models = body.get("data", [])
        gateway = [m for m in models if m.get("id") == "gateway"]
        if not gateway:
            return False, "gateway model not listed", elapsed
        if "context_window" not in gateway[0]:
            return False, "context_window missing from gateway model", elapsed
        return True, f"context_window={gateway[0]['context_window']}", elapsed

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return False, str(exc), elapsed


def testMultiTurnWithForgetools(baseUrl):
    """Send prompt with all 6 forgetools, get tool_calls, fabricate result, verify answer."""
    messages = [{"role": "user", "content": "Read the file /etc/hostname and tell me what it says"}]

    t0 = time.perf_counter()
    try:
        # Turn 1: expect tool_calls
        resp1 = requests.post(f"{baseUrl}/v1/chat/completions", json={
            "model": "gateway",
            "messages": messages,
            "tools": FORGETOOLS_SCHEMAS,
            "stream": False,
        }, timeout=120)
        resp1.raise_for_status()
        turn1 = resp1.json()
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return False, f"turn 1 failed: {exc}", elapsed

    calls = extractToolCalls(turn1)
    if not calls:
        elapsed = (time.perf_counter() - t0) * 1000
        content = extractContent(turn1)
        snippet = makeSnippet(content, 80)
        return False, f"turn 1: expected tool_calls, got text: {snippet}", elapsed

    # Build turn 2 with fabricated tool results
    assistantMsg = turn1["choices"][0]["message"]
    messages.append(assistantMsg)

    for call in calls:
        callId = call.get("id", "call_unknown")
        toolName = call.get("function", {}).get("name", "")
        if toolName == "read_file":
            fakeResult = "goose-test-node-42\n"
        else:
            fakeResult = f"{toolName} completed successfully"
        messages.append({
            "role": "tool",
            "tool_call_id": callId,
            "content": fakeResult,
        })

    try:
        resp2 = requests.post(f"{baseUrl}/v1/chat/completions", json={
            "model": "gateway",
            "messages": messages,
            "tools": FORGETOOLS_SCHEMAS,
            "stream": False,
        }, timeout=120)
        resp2.raise_for_status()
        turn2 = resp2.json()
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return False, f"turn 2 failed: {exc}", elapsed

    elapsed = (time.perf_counter() - t0) * 1000
    finalContent = extractContent(turn2)
    if not finalContent.strip():
        return False, "empty final answer", elapsed
    if "goose-test-node-42" in finalContent.lower() or "42" in finalContent:
        return True, f"final answer references fake result ({len(finalContent)} chars)", elapsed
    return True, f"final answer present ({len(finalContent)} chars)", elapsed


def testLargeConversation(baseUrl):
    """5-turn conversation with 2000+ char tool results per turn. Verifies trimming."""
    messages = [{"role": "system", "content": "You are a coding assistant."}]

    for i in range(5):
        messages.append({"role": "user", "content": f"Read file number {i}"})
        callId = f"call_large_{i}"
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "id": callId,
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": f"/src/module_{i}.py"}),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": callId,
            "content": f"# module_{i}.py\n" + ("def func():\n    pass\n" * 100),
        })

    # Turn 6: final prompt
    messages.append({"role": "user", "content": "Summarize all the files you read."})

    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{baseUrl}/v1/chat/completions", json={
            "model": "gateway",
            "messages": messages,
            "tools": FORGETOOLS_SCHEMAS,
            "stream": False,
        }, timeout=180)
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return False, f"request failed: {exc}", elapsed

    elapsed = (time.perf_counter() - t0) * 1000
    content = extractContent(body)
    calls = extractToolCalls(body)
    if content.strip() or calls:
        return True, f"response ok ({len(content)} chars, {len(calls)} tool_calls)", elapsed
    return False, "empty response", elapsed


def testStreamingWithForgetools(baseUrl):
    """Streaming multi-turn with forgetools. Parse SSE for tool_calls."""
    messages = [{"role": "user", "content": "List the files in /tmp"}]

    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{baseUrl}/v1/chat/completions", json={
            "model": "gateway",
            "messages": messages,
            "tools": FORGETOOLS_SCHEMAS,
            "stream": True,
        }, timeout=120, stream=True)
        resp.raise_for_status()
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return False, f"request failed: {exc}", elapsed

    chunks = []
    hasDone = False
    hasToolCalls = False

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data = line[len("data: "):]
            if data == "[DONE]":
                hasDone = True
                continue
            try:
                chunk = json.loads(data)
                chunks.append(chunk)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "tool_calls" in delta:
                    hasToolCalls = True
            except json.JSONDecodeError:
                pass

    elapsed = (time.perf_counter() - t0) * 1000

    if not hasDone:
        return False, "missing [DONE] marker", elapsed
    if not chunks:
        return False, "no chunks received", elapsed

    # Model might answer directly or call tools. Both are valid.
    if hasToolCalls:
        return True, f"stream had tool_calls ({len(chunks)} chunks)", elapsed
    return True, f"stream completed ({len(chunks)} chunks)", elapsed


def printResults(results):
    """Print results as an aligned table."""
    nameWidth = 30
    resultWidth = 6
    latencyWidth = 8
    detailsWidth = 60
    headerFmt = f"  {{:<{nameWidth}}}  {{:<{resultWidth}}}  {{:>{latencyWidth}}}  {{:<{detailsWidth}}}"
    print()
    print(headerFmt.format("Test", "Result", "Latency", "Details"))
    print("  " + "-" * (nameWidth + resultWidth + latencyWidth + detailsWidth + 8))

    for r in results:
        statusLabel = "PASS" if r["passed"] else "FAIL"
        latencyLabel = f"{r['latencyMs']:.0f}ms"
        print(headerFmt.format(
            formatCell(r["name"], nameWidth),
            formatCell(statusLabel, resultWidth),
            formatCell(latencyLabel, latencyWidth),
            formatCell(r["reason"], detailsWidth),
        ))


def writeJsonl(results, outputPath):
    """Append results as JSON-lines for CI trending."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(outputPath, "a") as f:
        for r in results:
            record = {
                "test": r["name"],
                "suite": "goose_readiness",
                "passed": r["passed"],
                "reason": r["reason"],
                "latency_ms": round(r["latencyMs"], 1),
                "timestamp": timestamp,
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Results written to {outputPath}")


def main():
    parser = argparse.ArgumentParser(description="Goose readiness test suite")
    parser.add_argument("--url", default=None,
                        help="Gateway base URL. Default: AGENT_URL env or http://bigfish:31400")
    parser.add_argument("--jsonl",
                        default=os.environ.get("TEST_JSONL_PATH", "/tmp/test-goose-readiness-results.jsonl"),
                        help="Path for JSON-lines output")
    args = parser.parse_args()

    baseUrl = args.url or os.environ.get("AGENT_URL", "http://bigfish:31400")

    print()
    print("=" * 60)
    print("  Goose Readiness Test Suite")
    print("=" * 60)
    print(f"  URL:   {baseUrl}")
    print()

    # Preflight
    try:
        preflight = requests.get(f"{baseUrl}/health", timeout=5)
        preflight.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"  Cannot reach {baseUrl}: {exc}")
        print("  Aborting.")
        sys.exit(1)

    tests = [
        ("model_discovery", testModelDiscovery),
        ("multi_turn_with_forgetools", testMultiTurnWithForgetools),
        ("large_conversation", testLargeConversation),
        ("streaming_with_forgetools", testStreamingWithForgetools),
    ]

    results = []
    for name, testFn in tests:
        print(f"  Running {name}...", end="", flush=True)
        passed, reason, latencyMs = testFn(baseUrl)
        statusSymbol = "ok" if passed else "FAIL"
        print(f" {statusSymbol} ({latencyMs:.0f}ms)")
        results.append({
            "name": name,
            "passed": passed,
            "reason": reason,
            "latencyMs": latencyMs,
        })

    printResults(results)
    writeJsonl(results, args.jsonl)

    totalPassed = sum(1 for r in results if r["passed"])
    totalCount = len(results)
    print()
    print(f"  {totalPassed}/{totalCount} passed")
    if totalPassed == totalCount:
        print("  All tests passed.")
    else:
        failedNames = [r["name"] for r in results if not r["passed"]]
        print(f"  Failed: {', '.join(failedNames)}")
    print()

    sys.exit(0 if totalPassed == totalCount else 1)


if __name__ == "__main__":
    main()
