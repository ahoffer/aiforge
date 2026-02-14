#!/usr/bin/env python3
"""Shared response parsers for test scripts and benchmarks.

Can be imported directly from Python test code or invoked from bash
via: python3 tests/lib/parse_response.py <function> < input

Supported functions when called from the command line:
  extract_tool_calls       Parse OpenAI-format tool_calls from JSON on stdin
  extract_generation_metrics  Parse Ollama /api/generate metrics from stdin
  extract_chat_metrics     Parse bench-model-compare metrics from stdin
                           (requires EXPECTED env var)
"""

import json
import os
import sys


def extract_tool_calls(response):
    """Pull tool call list from an OpenAI-format chat completion response.

    Returns a list of dicts, each with 'name' and 'arguments' keys.
    """
    try:
        choices = response.get("choices", [])
        if not choices:
            return []
        message = choices[0].get("message", {})
        raw_calls = message.get("tool_calls", []) or []
        parsed = []
        for tc in raw_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", "")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    pass
            parsed.append({
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": args,
            })
        return parsed
    except (KeyError, IndexError, TypeError):
        return []


def extract_generation_metrics(response):
    """Parse Ollama /api/generate metrics into a flat dict.

    Returns a dict with gen_tokens, gen_tps, prompt_tokens, prompt_tps,
    and total_ms.
    """
    eval_count = response.get("eval_count", 0)
    eval_ns = response.get("eval_duration", 0)
    prompt_count = response.get("prompt_eval_count", 0)
    prompt_ns = response.get("prompt_eval_duration", 0)
    total_ns = response.get("total_duration", 0)

    gen_tps = eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0
    prompt_tps = prompt_count / (prompt_ns / 1e9) if prompt_ns > 0 else 0
    total_ms = total_ns / 1e6

    return {
        "gen_tokens": eval_count,
        "gen_tps": round(gen_tps, 1),
        "prompt_tokens": prompt_count,
        "prompt_tps": round(prompt_tps, 1),
        "total_ms": round(total_ms, 0),
    }


def extract_chat_metrics(response, expected):
    """Parse an Ollama /api/chat response for bench-model-compare scoring.

    Returns a dict with latency_ms (caller-provided), json_valid,
    judgment_correct, and snippet.
    """
    message = response.get("message", {})
    tool_calls = message.get("tool_calls", [])
    content = message.get("content", "")

    json_valid = False
    snippet = ""
    made_tool_call = False

    if tool_calls:
        made_tool_call = True
        tc = tool_calls[0]
        func = tc.get("function", {})
        name = func.get("name", "")
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        query = ""
        if isinstance(args, dict):
            query = args.get("query", "")
        json_valid = name == "web_search" and isinstance(query, str) and len(query) > 0
        snippet = query[:80] if query else name[:80]
    else:
        snippet = content[:80].replace("\n", " ") if content else "(empty)"

    if expected == "search":
        judgment_correct = made_tool_call and json_valid
    elif expected == "nosearch":
        judgment_correct = not made_tool_call
        json_valid = True
    else:
        judgment_correct = False

    return {
        "json_valid": json_valid,
        "judgment_correct": judgment_correct,
        "snippet": snippet,
    }


def _cli_extract_tool_calls():
    response = json.load(sys.stdin)
    calls = extract_tool_calls(response)
    for c in calls:
        print(json.dumps(c))


def _cli_extract_generation_metrics():
    response = json.load(sys.stdin)
    metrics = extract_generation_metrics(response)
    print(json.dumps(metrics))


def _cli_extract_chat_metrics():
    expected = os.environ.get("EXPECTED", "search")
    response = json.load(sys.stdin)
    metrics = extract_chat_metrics(response, expected)
    print(json.dumps(metrics))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <function>", file=sys.stderr)
        print("Functions: extract_tool_calls, extract_generation_metrics, extract_chat_metrics", file=sys.stderr)
        sys.exit(1)

    dispatch = {
        "extract_tool_calls": _cli_extract_tool_calls,
        "extract_generation_metrics": _cli_extract_generation_metrics,
        "extract_chat_metrics": _cli_extract_chat_metrics,
    }

    func_name = sys.argv[1]
    if func_name not in dispatch:
        print(f"Unknown function: {func_name}", file=sys.stderr)
        sys.exit(1)

    try:
        dispatch[func_name]()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
