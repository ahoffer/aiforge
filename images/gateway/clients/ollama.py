"""Ollama service client for LLM inference and embeddings."""

import json
import logging
import os
import time
from typing import Any, Generator

import requests

log = logging.getLogger(__name__)

# Retry settings for transient failures
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0
RETRYABLE_STATUS_CODES = {502, 503, 504}


def _post_with_retry(url: str, payload: dict, timeout: int = 300, **kwargs: Any) -> requests.Response:
    """POST with exponential backoff on transient failures."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=timeout, **kwargs)
            if resp.status_code not in RETRYABLE_STATUS_CODES:
                return resp
            log.warning("Retryable HTTP %d from %s, attempt %d", resp.status_code, url, attempt + 1)
        except requests.ConnectionError as exc:
            log.warning("Connection error to %s, attempt %d: %s", url, attempt + 1, exc)
            last_exc = exc
        except requests.Timeout as exc:
            log.warning("Timeout on %s, attempt %d", url, attempt + 1)
            last_exc = exc

        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF * (2 ** attempt)
            time.sleep(wait)

    if last_exc:
        raise last_exc
    return resp


class OllamaClient:
    """Client for Ollama API providing chat and embedding capabilities."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://ollama:11434")).rstrip("/")

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        stream: bool = False,
        options: dict | None = None,
    ) -> str | Generator[str, None, None]:
        """Generate a response from the model.

        Args:
            prompt: The user prompt
            model: Model name, defaults to AGENT_MODEL env var
            system: Optional system prompt
            stream: If True, yields chunks instead of returning full response
            options: Ollama options like num_predict, temperature, etc.

        Returns:
            Complete response string or generator of chunks
        """
        model = model or os.getenv("AGENT_MODEL", "devstral:latest-agent")

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        if stream:
            return self._stream_generate(payload)

        resp = _post_with_retry(f"{self.base_url}/api/generate", payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _stream_generate(self, payload: dict) -> Generator[str, None, None]:
        """Stream generation response chunks."""
        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if chunk := data.get("response"):
                        yield chunk

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = False,
        tools: list[dict] | None = None,
        options: dict | None = None,
    ) -> dict | str | Generator[str, None, None]:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name, defaults to AGENT_MODEL env var
            stream: If True, yields chunks instead of returning full response
            tools: Optional list of tool schemas for native tool calling.
                When provided, returns the full message dict so the caller
                can inspect tool_calls. When omitted, returns content string.
            options: Ollama runtime options like num_ctx, temperature, etc.

        Returns:
            Full message dict (when tools provided), content string, or
            generator of chunks
        """
        model = model or os.getenv("AGENT_MODEL", "devstral:latest-agent")

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options

        if stream:
            return self._stream_chat(payload)

        resp = _post_with_retry(f"{self.base_url}/api/chat", payload, timeout=300)
        resp.raise_for_status()
        message = resp.json().get("message", {})

        if tools:
            return message

        return message.get("content", "")

    def _stream_chat(self, payload: dict) -> Generator[str, None, None]:
        """Stream chat response chunks."""
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if content := data.get("message", {}).get("content"):
                        yield content

    def embed(self, text: str | list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for text.

        Args:
            text: Single string or list of strings to embed
            model: Embedding model name, defaults to EMBEDDING_MODEL env var

        Returns:
            List of embedding vectors
        """
        model = model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

        if isinstance(text, str):
            text = [text]

        resp = _post_with_retry(
            f"{self.base_url}/api/embed",
            {"model": model, "input": text},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("embeddings", [])

    def health(self) -> bool:
        """Check if Ollama is healthy."""
        try:
            resp = requests.get(f"{self.base_url}/", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
