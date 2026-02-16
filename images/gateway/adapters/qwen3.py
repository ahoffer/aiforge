"""Adapter for qwen3 family.

qwen3 supports native tool calling through Ollama, so the base
passthrough behavior is correct. This subclass exists to document
that qwen3 was evaluated and works with defaults.
"""

from adapters.base import ModelAdapter


class Qwen3Adapter(ModelAdapter):
    pass
