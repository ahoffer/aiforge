"""Tests for service clients with mocked HTTP calls."""

import os
import sys
from unittest.mock import patch, MagicMock

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clients import OllamaClient, SearxngClient


class TestOllamaClient:

    def test_health_success(self):
        client = OllamaClient("http://test:11434")
        with patch("clients.ollama.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert client.health() is True

    def test_health_failure(self):
        client = OllamaClient("http://test:11434")
        with patch("clients.ollama.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")
            assert client.health() is False

    def test_generate_basic(self):
        client = OllamaClient("http://test:11434")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Hello world"}
        mock_resp.status_code = 200
        with patch("clients.ollama._post_with_retry", return_value=mock_resp):
            result = client.generate("Say hello", model="test-model")
            assert result == "Hello world"

    def test_chat_without_tools_returns_string(self):
        client = OllamaClient("http://test:11434")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Hello"}
        }
        mock_resp.status_code = 200
        with patch("clients.ollama._post_with_retry", return_value=mock_resp) as mock_post:
            result = client.chat(
                [{"role": "user", "content": "Hi"}],
                model="test-model",
            )
            assert result == "Hello"
            assert isinstance(result, str)
            call_payload = mock_post.call_args[0][1]
            assert "options" not in call_payload

    def test_chat_with_tools_returns_message_dict(self):
        client = OllamaClient("http://test:11434")
        expected_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "web_search", "arguments": {"query": "test"}}}
            ],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": expected_message}
        mock_resp.status_code = 200
        with patch("clients.ollama._post_with_retry", return_value=mock_resp) as mock_post:
            tools = [{"type": "function", "function": {"name": "web_search"}}]
            result = client.chat(
                [{"role": "user", "content": "Search for test"}],
                model="test-model",
                tools=tools,
            )
            assert isinstance(result, dict)
            assert result["tool_calls"][0]["function"]["name"] == "web_search"

            # Verify tools and thinking suppression were included in payload
            call_payload = mock_post.call_args[0][1]
            assert "tools" in call_payload
            assert call_payload.get("options") == {"enable_thinking": False}

    def test_chat_with_tools_text_response(self):
        """When tools are provided but model answers directly."""
        client = OllamaClient("http://test:11434")
        expected_message = {
            "role": "assistant",
            "content": "The answer is 4.",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": expected_message}
        mock_resp.status_code = 200
        with patch("clients.ollama._post_with_retry", return_value=mock_resp):
            tools = [{"type": "function", "function": {"name": "web_search"}}]
            result = client.chat(
                [{"role": "user", "content": "What is 2+2?"}],
                model="test-model",
                tools=tools,
            )
            assert isinstance(result, dict)
            assert result["content"] == "The answer is 4."
            assert "tool_calls" not in result

    def test_embed(self):
        client = OllamaClient("http://test:11434")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_resp.status_code = 200
        with patch("clients.ollama._post_with_retry", return_value=mock_resp):
            result = client.embed("test text")
            assert len(result) == 1
            assert len(result[0]) == 3


class TestSearxngClient:

    def test_health_success(self):
        client = SearxngClient("http://test:8080")
        with patch("clients.searxng.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert client.health() is True

    def test_search(self):
        client = SearxngClient("http://test:8080")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"title": "Test", "url": "http://example.com", "content": "A result", "engine": "google"}
            ]
        }
        with patch("clients.searxng.requests.get", return_value=mock_resp):
            results = client.search("test query")
            assert len(results) == 1
            assert results[0]["title"] == "Test"

    def test_search_text(self):
        client = SearxngClient("http://test:8080")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"title": "Result 1", "url": "http://a.com", "content": "Content A", "engine": "g"}
            ]
        }
        with patch("clients.searxng.requests.get", return_value=mock_resp):
            text = client.search_text("test")
            assert "Result 1" in text
            assert "http://a.com" in text
