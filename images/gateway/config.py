"""Shared configuration read once from environment variables.

All gateway modules import from here so model names and tunables
live in a single place. In k8s, values flow from ollama-config and
gateway-config ConfigMaps. Missing required vars fail fast with a
clear message rather than falling back to stale defaults.
"""

import os
import sys

from adapters import select_adapter


def _require(name: str) -> str:
    """Return an env var or exit with a diagnostic message."""
    val = os.getenv(name)
    if not val:
        print(
            f"FATAL: required environment variable {name} is not set. "
            f"In k8s this comes from a ConfigMap. For local dev, source config.env first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return val


OLLAMA_URL = _require("OLLAMA_URL").rstrip("/")
AGENT_MODEL = _require("AGENT_MODEL")
AGENT_NUM_CTX = int(_require("AGENT_NUM_CTX"))
EMBEDDING_MODEL = _require("EMBEDDING_MODEL")
LOG_LANGGRAPH_OUTPUT = os.getenv("LOG_LANGGRAPH_OUTPUT", "true").lower() in ("1", "true")

# PostgreSQL connection for durable conversation checkpoints.
# Falls back to empty string which graph.py interprets as InMemorySaver.
POSTGRES_URL = os.getenv("POSTGRES_URL", "")

# Qdrant URL for both manual vector search and retriever tools.
# Already consumed by clients/qdrant.py via os.getenv, but declared
# here for retriever tool construction in tools.py.
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# JSON list of collection specs for auto-created retriever tools.
# Each entry: {"name": "docs", "description": "Project documentation"}.
# Empty or unset means no retriever tools are created.
QDRANT_COLLECTIONS = os.getenv("QDRANT_COLLECTIONS", "")

# Pluggable adapter for model-specific behavior. Resolved once at startup
# so all modules share the same instance.
MODEL_ADAPTER = select_adapter(AGENT_MODEL)
