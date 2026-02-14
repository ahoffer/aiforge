#!/usr/bin/env python3
"""Comprehensive tool-calling test suite for Ollama models.

Sends prompts to the OpenAI-compatible chat completions endpoint with
tool schemas, then checks whether the model selected the right tool,
avoided tools when appropriate, or chained multiple tools. Results are
printed in a table with per-test latency.

Usage:
    python3 test-tool-calling.py [model_name] [--url URL] [--direct]

Defaults to model "gateway" through the gateway proxy at AGENT_URL env var
(fallback http://bigfish:31400). Use --direct to bypass the proxy and hit
Ollama at http://localhost:11434 with devstral:latest instead.
"""

import argparse
import json
import os
import sys
import time
import requests


# -- Tool schemas shared across tests --

LIST_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files and directories at the given path",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute directory path to list"
                }
            },
            "required": ["path"]
        }
    }
}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file at the given path",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute file path to read"
                }
            },
            "required": ["path"]
        }
    }
}

RUN_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Execute a shell command and return its output",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["command"]
        }
    }
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for a query and return results",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                }
            },
            "required": ["query"]
        }
    }
}

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file at the given path",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute file path to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    }
}

ALL_TOOLS = [LIST_FILES_TOOL, READ_FILE_TOOL, RUN_COMMAND_TOOL, WEB_SEARCH_TOOL, WRITE_FILE_TOOL]


# -- Test case definitions --
# Each test specifies the prompt, which tools to offer, what the expected
# outcome is, and a validation function. The validator receives the parsed
# API response and returns a tuple of pass/fail bool and a short reason.

def extractToolCalls(response):
    """Pull tool call list from OpenAI-format response."""
    try:
        choices = response.get("choices", [])
        if not choices:
            return []
        message = choices[0].get("message", {})
        return message.get("tool_calls", []) or []
    except (KeyError, IndexError, TypeError):
        return []


def extractContent(response):
    """Pull text content from OpenAI-format response."""
    try:
        return response["choices"][0]["message"].get("content", "") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def validateSingleTool(expectedName):
    """Return validator that checks exactly one tool call with the given name."""
    def validator(response):
        calls = extractToolCalls(response)
        if len(calls) == 0:
            content = extractContent(response)
            snippet = content[:80] + "..." if len(content) > 80 else content
            return False, f"no tool call, got text: {snippet}"
        if len(calls) > 1:
            names = [c.get("function", {}).get("name", "?") for c in calls]
            return False, f"expected 1 call, got {len(calls)}: {names}"
        actualName = calls[0].get("function", {}).get("name", "")
        if actualName != expectedName:
            return False, f"expected {expectedName}, got {actualName}"
        return True, f"called {actualName}"
    return validator


def validateNoToolCall(response):
    """Check that model answered directly without calling any tool."""
    calls = extractToolCalls(response)
    if len(calls) > 0:
        names = [c.get("function", {}).get("name", "?") for c in calls]
        return False, f"unexpected tool call: {names}"
    content = extractContent(response)
    if not content.strip():
        return False, "no tool call and no text content"
    return True, "direct answer"


def validateMultiTool(expectedNames):
    """Check that model called multiple tools matching expected names."""
    def validator(response):
        calls = extractToolCalls(response)
        actualNames = [c.get("function", {}).get("name", "?") for c in calls]
        if len(calls) == 0:
            content = extractContent(response)
            snippet = content[:80] + "..." if len(content) > 80 else content
            return False, f"no tool calls, got text: {snippet}"
        # Check that at least the expected tools appear in the call list
        missing = [n for n in expectedNames if n not in actualNames]
        if missing:
            return False, f"missing tools {missing}, got {actualNames}"
        return True, f"called {actualNames}"
    return validator


def validateToolCallWithArg(expectedName, argName, argSubstring):
    """Check single tool call, correct name, and argument contains substring."""
    def validator(response):
        calls = extractToolCalls(response)
        if len(calls) == 0:
            content = extractContent(response)
            snippet = content[:80] + "..." if len(content) > 80 else content
            return False, f"no tool call, got text: {snippet}"
        call = calls[0]
        actualName = call.get("function", {}).get("name", "")
        if actualName != expectedName:
            return False, f"expected {expectedName}, got {actualName}"
        rawArgs = call.get("function", {}).get("arguments", "")
        try:
            parsedArgs = json.loads(rawArgs) if isinstance(rawArgs, str) else rawArgs
        except json.JSONDecodeError:
            return False, f"could not parse arguments: {rawArgs[:80]}"
        argValue = str(parsedArgs.get(argName, ""))
        if argSubstring.lower() not in argValue.lower():
            return False, f"arg '{argName}' = '{argValue}', expected to contain '{argSubstring}'"
        return True, f"called {actualName}({argName}={argValue})"
    return validator


# Test definitions grouped by category

SINGLE_TOOL_TESTS = [
    {
        "name": "list_files /tmp",
        "category": "single_tool",
        "prompt": "List all files in /tmp",
        "tools": ALL_TOOLS,
        "validate": validateToolCallWithArg("list_files", "path", "/tmp"),
    },
    {
        "name": "read_file /etc/hostname",
        "category": "single_tool",
        "prompt": "Show me the contents of /etc/hostname",
        "tools": ALL_TOOLS,
        "validate": validateToolCallWithArg("read_file", "path", "/etc/hostname"),
    },
    {
        "name": "run_command git branch",
        "category": "single_tool",
        "prompt": "What is the current git branch in /home/aaron/Downloads/agents?",
        "tools": ALL_TOOLS,
        "validate": validateSingleTool("run_command"),
    },
    {
        "name": "web_search query",
        "category": "single_tool",
        "prompt": "Search the web for the latest Python 3 stable release version",
        "tools": ALL_TOOLS,
        "validate": validateSingleTool("web_search"),
    },
    {
        "name": "run_command grep",
        "category": "single_tool",
        "prompt": "Find all lines containing 'import' in /etc/hostname",
        "tools": ALL_TOOLS,
        "validate": validateSingleTool("run_command"),
    },
]

NO_TOOL_TESTS = [
    {
        "name": "arithmetic 2+2",
        "category": "no_tool",
        "prompt": "What is 2 + 2?",
        "tools": ALL_TOOLS,
        "validate": validateNoToolCall,
    },
    {
        "name": "general knowledge zip",
        "category": "no_tool",
        "prompt": "What does the Python zip function do?",
        "tools": ALL_TOOLS,
        "validate": validateNoToolCall,
    },
    {
        "name": "explain MCP",
        "category": "no_tool",
        "prompt": "Explain what the MCP protocol is in two sentences.",
        "tools": ALL_TOOLS,
        "validate": validateNoToolCall,
    },
    {
        "name": "arithmetic 17*23",
        "category": "no_tool",
        "prompt": "What is 17 multiplied by 23?",
        "tools": ALL_TOOLS,
        "validate": validateNoToolCall,
    },
]

MULTI_TOOL_TESTS = [
    {
        "name": "list then read",
        "category": "multi_tool",
        "prompt": "List files in /tmp and also read the contents of /etc/hostname",
        "tools": ALL_TOOLS,
        "validate": validateMultiTool(["list_files", "read_file"]),
    },
    {
        "name": "search and write",
        "category": "multi_tool",
        "prompt": "Search the web for the current date and write it to /tmp/today.txt",
        "tools": ALL_TOOLS,
        "validate": validateMultiTool(["web_search", "write_file"]),
    },
    {
        "name": "command and list",
        "category": "multi_tool",
        "prompt": "Run 'uname -a' and also list files in /home",
        "tools": ALL_TOOLS,
        "validate": validateMultiTool(["run_command", "list_files"]),
    },
]

# -- Multi-turn test support --
# Simulates a real client-side tool calling loop: send prompt, receive
# tool_calls, fabricate tool results, send them back, check final answer.

def runMultiTurnTest(baseUrl, modelName, testCase):
    """Execute a multi-turn tool calling loop and validate the final answer.

    Turn 1: send the user prompt with tools, expect tool_calls back.
    Turn 2: send fabricated tool results, expect a coherent text answer.
    """
    messages = [{"role": "user", "content": testCase["prompt"]}]

    # Turn 1: get tool calls
    payload = {
        "model": modelName,
        "messages": messages,
        "tools": testCase["tools"],
        "stream": False,
    }

    startTime = time.perf_counter()
    try:
        httpResponse = requests.post(
            f"{baseUrl}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        httpResponse.raise_for_status()
        turn1 = httpResponse.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
        elapsedMs = (time.perf_counter() - startTime) * 1000
        return {
            "name": testCase["name"],
            "category": testCase["category"],
            "passed": False,
            "reason": f"turn 1 failed: {exc}",
            "latencyMs": elapsedMs,
        }

    calls = extractToolCalls(turn1)
    if not calls:
        elapsedMs = (time.perf_counter() - startTime) * 1000
        content = extractContent(turn1)
        snippet = content[:80] + "..." if len(content) > 80 else content
        return {
            "name": testCase["name"],
            "category": testCase["category"],
            "passed": False,
            "reason": f"turn 1: expected tool_calls, got text: {snippet}",
            "latencyMs": elapsedMs,
        }

    # Build turn 2 messages: assistant message with tool_calls, then tool results
    assistantMsg = turn1["choices"][0]["message"]
    messages.append(assistantMsg)

    for call in calls:
        toolName = call.get("function", {}).get("name", "")
        callId = call.get("id", "call_unknown")
        fakeResult = testCase["fake_results"].get(toolName, f"{toolName} returned ok")
        messages.append({
            "role": "tool",
            "tool_call_id": callId,
            "content": fakeResult,
        })

    # Turn 2: send tool results, expect final text answer
    payload2 = {
        "model": modelName,
        "messages": messages,
        "tools": testCase["tools"],
        "stream": False,
    }

    try:
        httpResponse2 = requests.post(
            f"{baseUrl}/v1/chat/completions",
            json=payload2,
            timeout=120,
        )
        httpResponse2.raise_for_status()
        turn2 = httpResponse2.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
        elapsedMs = (time.perf_counter() - startTime) * 1000
        return {
            "name": testCase["name"],
            "category": testCase["category"],
            "passed": False,
            "reason": f"turn 2 failed: {exc}",
            "latencyMs": elapsedMs,
        }

    elapsedMs = (time.perf_counter() - startTime) * 1000
    passed, reason = testCase["validate_final"](turn2, calls)

    return {
        "name": testCase["name"],
        "category": testCase["category"],
        "passed": passed,
        "reason": reason,
        "latencyMs": elapsedMs,
    }


def validateFinalHasContent(response, turn1Calls):
    """Check that the model produced a non-empty text answer after tool results."""
    content = extractContent(response)
    if not content.strip():
        turn2Calls = extractToolCalls(response)
        if turn2Calls:
            names = [c.get("function", {}).get("name", "?") for c in turn2Calls]
            return False, f"model called more tools instead of answering: {names}"
        return False, "empty final answer"
    return True, f"final answer ({len(content)} chars)"


def validateFinalContains(substrings):
    """Check that the final answer contains all expected substrings."""
    def validator(response, turn1Calls):
        content = extractContent(response)
        if not content.strip():
            return False, "empty final answer"
        lowered = content.lower()
        missing = [s for s in substrings if s.lower() not in lowered]
        if missing:
            return False, f"answer missing expected terms: {missing}"
        return True, f"final answer contains expected terms ({len(content)} chars)"
    return validator


MULTI_TURN_TESTS = [
    {
        "name": "read file then summarize",
        "category": "multi_turn",
        "prompt": "Read /etc/hostname and tell me what the hostname is",
        "tools": ALL_TOOLS,
        "fake_results": {
            "read_file": "gateway-node-7a3b\n",
        },
        "validate_final": validateFinalContains(["gateway-node-7a3b"]),
    },
    {
        "name": "list then describe",
        "category": "multi_turn",
        "prompt": "List the files in /var/log and tell me how many there are",
        "tools": ALL_TOOLS,
        "fake_results": {
            "list_files": "syslog\nauth.log\nkern.log\ndpkg.log\nboot.log",
        },
        "validate_final": validateFinalContains(["5"]),
    },
    {
        "name": "web search then answer",
        "category": "multi_turn",
        "prompt": "Search the web for the capital of Iceland and tell me what it is",
        "tools": ALL_TOOLS,
        "fake_results": {
            "web_search": "Reykjavik is the capital and largest city of Iceland.",
        },
        "validate_final": validateFinalContains(["Reykjavik"]),
    },
]

# -- Tool count stress tests --
# Sends the same prompt with list_files plus N-1 dummy tools to find
# the model's ceiling for simultaneous tool schemas.

def _makeDummyTool(index):
    """Generate a trivially simple dummy tool schema."""
    return {
        "type": "function",
        "function": {
            "name": f"dummy_tool_{index}",
            "description": f"Dummy tool number {index} that does nothing useful",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Arbitrary input"
                    }
                },
                "required": ["input"]
            }
        }
    }


TOOL_COUNT_STRESS_LEVELS = [5, 10, 15, 20, 30, 40, 50]

TOOL_COUNT_STRESS_TESTS = []
for n in TOOL_COUNT_STRESS_LEVELS:
    dummyTools = [_makeDummyTool(i) for i in range(1, n)]
    toolSet = [LIST_FILES_TOOL] + dummyTools
    TOOL_COUNT_STRESS_TESTS.append({
        "name": f"stress {n} tools",
        "category": "tool_count_stress",
        "prompt": "List all files in /tmp",
        "tools": toolSet,
        "validate": validateToolCallWithArg("list_files", "path", "/tmp"),
    })


ALL_TESTS = SINGLE_TOOL_TESTS + NO_TOOL_TESTS + MULTI_TOOL_TESTS + MULTI_TURN_TESTS + TOOL_COUNT_STRESS_TESTS


def runTest(baseUrl, modelName, testCase):
    """Send a single test prompt and validate the response.

    Returns a dict with test name, category, pass/fail, reason, and
    latency in milliseconds.
    """
    payload = {
        "model": modelName,
        "messages": [
            {"role": "user", "content": testCase["prompt"]}
        ],
        "tools": testCase["tools"],
        "stream": False,
    }

    startTime = time.perf_counter()
    try:
        httpResponse = requests.post(
            f"{baseUrl}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        httpResponse.raise_for_status()
        responseJson = httpResponse.json()
    except requests.exceptions.Timeout:
        elapsedMs = (time.perf_counter() - startTime) * 1000
        return {
            "name": testCase["name"],
            "category": testCase["category"],
            "passed": False,
            "reason": "request timed out after 120s",
            "latencyMs": elapsedMs,
        }
    except requests.exceptions.RequestException as exc:
        elapsedMs = (time.perf_counter() - startTime) * 1000
        return {
            "name": testCase["name"],
            "category": testCase["category"],
            "passed": False,
            "reason": f"HTTP error: {exc}",
            "latencyMs": elapsedMs,
        }
    except json.JSONDecodeError:
        elapsedMs = (time.perf_counter() - startTime) * 1000
        return {
            "name": testCase["name"],
            "category": testCase["category"],
            "passed": False,
            "reason": "response was not valid JSON",
            "latencyMs": elapsedMs,
        }

    elapsedMs = (time.perf_counter() - startTime) * 1000
    passed, reason = testCase["validate"](responseJson)

    return {
        "name": testCase["name"],
        "category": testCase["category"],
        "passed": passed,
        "reason": reason,
        "latencyMs": elapsedMs,
    }


def printResults(results):
    """Print results as an aligned table with category grouping."""
    # Compute column widths
    nameWidth = max(len(r["name"]) for r in results)
    categoryWidth = max(len(r["category"]) for r in results)
    reasonWidth = max(len(r["reason"]) for r in results)
    # Cap reason width to keep table readable
    reasonWidth = min(reasonWidth, 60)

    headerFmt = f"  {{:<{nameWidth}}}  {{:<{categoryWidth}}}  {{:<6}}  {{:>8}}  {{:<{reasonWidth}}}"
    rowFmt = headerFmt

    print()
    print(headerFmt.format("Test", "Category", "Result", "Latency", "Details"))
    print("  " + "-" * (nameWidth + categoryWidth + 6 + 8 + reasonWidth + 8))

    currentCategory = None
    for testResult in results:
        if testResult["category"] != currentCategory:
            currentCategory = testResult["category"]
            if testResult != results[0]:
                print()

        statusLabel = "PASS" if testResult["passed"] else "FAIL"
        latencyLabel = f"{testResult['latencyMs']:.0f}ms"
        truncatedReason = testResult["reason"][:reasonWidth]
        print(rowFmt.format(
            testResult["name"],
            testResult["category"],
            statusLabel,
            latencyLabel,
            truncatedReason,
        ))


def printSummary(results):
    """Print pass/fail counts by category and overall."""
    print()
    print("  Summary")
    print("  " + "-" * 50)

    categories = []
    seen = set()
    for testResult in results:
        if testResult["category"] not in seen:
            categories.append(testResult["category"])
            seen.add(testResult["category"])

    totalPassed = 0
    totalCount = 0

    for category in categories:
        categoryResults = [r for r in results if r["category"] == category]
        passedCount = sum(1 for r in categoryResults if r["passed"])
        totalInCategory = len(categoryResults)
        totalPassed += passedCount
        totalCount += totalInCategory
        avgLatency = sum(r["latencyMs"] for r in categoryResults) / totalInCategory
        print(f"  {category:15s}  {passedCount}/{totalInCategory} passed  avg {avgLatency:.0f}ms")

    print(f"  {'overall':15s}  {totalPassed}/{totalCount} passed")
    print()

    if totalPassed == totalCount:
        print("  All tests passed.")
    else:
        failedNames = [r["name"] for r in results if not r["passed"]]
        print(f"  Failed tests: {', '.join(failedNames)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive tool-calling test suite for Ollama models"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model name to test. Default: gateway (proxy) or devstral:latest (--direct)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Base URL. Default: AGENT_URL env or http://bigfish:31400 (proxy), "
             "http://localhost:11434 (--direct)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Bypass gateway and test against Ollama directly",
    )
    parser.add_argument(
        "--category",
        choices=["single_tool", "no_tool", "multi_tool", "multi_turn", "tool_count_stress"],
        help="Run only tests in this category",
    )
    args = parser.parse_args()

    # Resolve defaults based on --direct flag
    if args.direct:
        if args.model is None:
            args.model = "devstral:latest"
        if args.url is None:
            args.url = "http://localhost:11434"
    else:
        if args.model is None:
            args.model = "gateway"
        if args.url is None:
            args.url = os.environ.get("AGENT_URL", "http://bigfish:31400")

    testsToRun = ALL_TESTS
    if args.category:
        testsToRun = [t for t in ALL_TESTS if t["category"] == args.category]

    modeLabel = "direct (Ollama)" if args.direct else "proxy (Gateway)"
    print()
    print("=" * 60)
    print("  Tool Calling Test Suite")
    print("=" * 60)
    print(f"  Mode:  {modeLabel}")
    print(f"  Model: {args.model}")
    print(f"  URL:   {args.url}")
    print(f"  Tests: {len(testsToRun)}")
    print()

    # Preflight check, verify endpoint is reachable
    preflightPath = "/health" if not args.direct else "/"
    try:
        preflight = requests.get(f"{args.url}{preflightPath}", timeout=5)
        preflight.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"  Cannot reach {args.url}: {exc}")
        print("  Aborting.")
        sys.exit(1)

    results = []
    for i, testCase in enumerate(testsToRun, 1):
        label = f"[{i}/{len(testsToRun)}] {testCase['name']}"
        print(f"  Running {label}...", end="", flush=True)
        if testCase["category"] == "multi_turn":
            testResult = runMultiTurnTest(args.url, args.model, testCase)
        else:
            testResult = runTest(args.url, args.model, testCase)
        statusSymbol = "ok" if testResult["passed"] else "FAIL"
        print(f" {statusSymbol} ({testResult['latencyMs']:.0f}ms)")
        results.append(testResult)

    printResults(results)
    printSummary(results)

    failedCount = sum(1 for r in results if not r["passed"])
    sys.exit(1 if failedCount > 0 else 0)


if __name__ == "__main__":
    main()
