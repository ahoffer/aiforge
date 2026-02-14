"""SearXNG metasearch engine client for web search."""

import os
from typing import Any

import requests


class SearxngClient:
    """Client for SearXNG search API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("SEARXNG_URL", "http://searxng:8080")).rstrip("/")

    def search(
        self,
        query: str,
        categories: list[str] | None = None,
        engines: list[str] | None = None,
        language: str = "en",
        page: int = 1,
    ) -> list[dict[str, Any]]:
        """Perform a web search.

        Args:
            query: Search query string
            categories: Optional list like ['general', 'images', 'news']
            engines: Optional list of specific engines to use
            language: Language code, defaults to 'en'
            page: Page number for pagination

        Returns:
            List of search results with title, url, content
        """
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "pageno": page,
        }

        if categories:
            params["categories"] = ",".join(categories)
        if engines:
            params["engines"] = ",".join(engines)

        resp = requests.get(
            f"{self.base_url}/search",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "engine": r.get("engine", ""),
            }
            for r in results
        ]

    def search_text(self, query: str, max_results: int = 5) -> str:
        """Search and return results as formatted text.

        Args:
            query: Search query
            max_results: Maximum number of results to include

        Returns:
            Formatted string with search results
        """
        results = self.search(query)[:max_results]

        if not results:
            return f"No results found for: {query}"

        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['url']}")
            if r["content"]:
                lines.append(f"   {r['content'][:200]}")
            lines.append("")

        return "\n".join(lines)

    def health(self) -> bool:
        """Check if SearXNG is healthy."""
        try:
            resp = requests.get(f"{self.base_url}/", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
