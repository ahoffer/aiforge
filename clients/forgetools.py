#!/usr/bin/env python3
"""MCP tool server for the AIForge coding assistant.

Provides six focused tools that replace the generic shell escape hatch.
Devstral scores 20/20 on tool calling with 2 tools. Six tools with
precise descriptions is well within its capability and far better
than one shell tool that forces the model to compose commands.

Launched by goose.sh via stdio transport. Tools execute locally on
the workstation where the code lives.
"""

import os
import subprocess

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="forgetools")

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://bigfish:31080")

# Cap tool output to prevent context window exhaustion on small models
MAX_OUTPUT_CHARS = 16000


def _cap(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "\n[truncated]"
    return text


@mcp.tool()
def read_file(path: str) -> str:
    """Read a file and return its contents with line numbers."""
    path = os.path.expanduser(path)
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except IsADirectoryError:
        return f"Error: is a directory, use list_directory instead: {path}"
    numbered = [f"{i + 1:6d}  {line}" for i, line in enumerate(lines)]
    return _cap("".join(numbered))


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    path = os.path.expanduser(path)
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except OSError as e:
        return f"Error writing {path}: {e}"


@mcp.tool()
def list_directory(path: str, recursive: bool = False) -> str:
    """List files and directories at the given path."""
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        return f"Error: not a directory: {path}"
    try:
        if recursive:
            entries = []
            for root, dirs, files in os.walk(path):
                # Skip hidden and generated directories
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                rel = os.path.relpath(root, path)
                prefix = "" if rel == "." else rel + "/"
                for name in sorted(dirs):
                    entries.append(f"{prefix}{name}/")
                for name in sorted(files):
                    entries.append(f"{prefix}{name}")
            return _cap("\n".join(entries)) if entries else "(empty directory)"
        else:
            entries = sorted(os.listdir(path))
            labeled = []
            for name in entries:
                full = os.path.join(path, name)
                suffix = "/" if os.path.isdir(full) else ""
                labeled.append(f"{name}{suffix}")
            return _cap("\n".join(labeled)) if labeled else "(empty directory)"
    except PermissionError:
        return f"Error: permission denied: {path}"


@mcp.tool()
def search_files(pattern: str, path: str = ".") -> str:
    """Search file contents for a regex pattern using ripgrep. Returns matching lines with context."""
    path = os.path.expanduser(path)
    try:
        proc = subprocess.run(
            ["rg", "--no-heading", "--line-number", "--color=never",
             "--max-count=50", "-C1", pattern, path],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            return _cap(proc.stdout)
        if proc.returncode == 1:
            return f"No matches found for pattern: {pattern}"
        return f"Error: {proc.stderr.strip()}"
    except FileNotFoundError:
        # ripgrep not installed, fall back to grep
        try:
            proc = subprocess.run(
                ["grep", "-rn", "--color=never", pattern, path],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                return _cap(proc.stdout)
            return f"No matches found for pattern: {pattern}"
        except Exception as e:
            return f"Error: neither rg nor grep available: {e}"
    except subprocess.TimeoutExpired:
        return f"Error: search timed out after 30s"


@mcp.tool()
def run_command(command: str) -> str:
    """Run a shell command and return stdout and stderr. Use for build, test, git, and other tasks."""
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120,
        )
        parts = []
        if proc.stdout:
            parts.append(proc.stdout)
        if proc.stderr:
            parts.append(f"[stderr]\n{proc.stderr}")
        if proc.returncode != 0:
            parts.append(f"[exit code {proc.returncode}]")
        output = "\n".join(parts) if parts else "(no output)"
        return _cap(output)
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120s"
    except Exception as e:
        return f"Error running command: {e}"


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web via SearXNG. Use for current docs, API references, or facts beyond training data."""
    try:
        resp = httpx.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "language": "en"},
            timeout=15.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])[:5]
        if not results:
            return f"No results found for: {query}"
        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '')}")
            lines.append(f"   {r.get('url', '')}")
            content = r.get("content", "")
            if content:
                lines.append(f"   {content[:200]}")
            lines.append("")
        return "\n".join(lines)
    except httpx.TimeoutException:
        return f"Error: search timed out for query: {query}"
    except Exception as e:
        return f"Error: web search failed: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
