from __future__ import annotations

from ..types import SearchResult
from .base import SearchProvider


class SearXNGSearchProvider(SearchProvider):
    name = "searxng"

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        endpoint = self.config.searxng_base_url.rstrip("/") + "/search"
        response = await self.client.get(
            endpoint,
            params={
                "q": query,
                "format": "json",
                "language": self.config.language,
                "safesearch": {"off": 0, "moderate": 1, "strict": 2}[self.config.safe_search],
                "pageno": offset + 1,
                "categories": "general",
            },
        )
        response.raise_for_status()
        raw = response.json().get("results", [])
        return [
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("content", "")),
                rank=offset * count + index + 1,
                source=self.name,
            )
            for index, item in enumerate(raw[:count])
            if item.get("url")
        ]
