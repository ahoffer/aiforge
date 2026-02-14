"""Qdrant client for vector similarity search."""

import os
from typing import Any

import requests


class QdrantClient:
    """Client for Qdrant API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("QDRANT_URL", "http://qdrant:6333")).rstrip("/")

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 5,
        with_payload: bool = True,
    ) -> list[dict[str, Any]]:
        """Search a collection by vector and return nearest points."""
        payload = {
            "vector": vector,
            "limit": limit,
            "with_payload": with_payload,
        }
        resp = requests.post(
            f"{self.base_url}/collections/{collection}/points/search",
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("result", [])

    def health(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            resp = requests.get(f"{self.base_url}/healthz", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
