from __future__ import annotations

from ..types import SearchResult
from .base import SearchError, SearchProvider


class TavilySearchProvider(SearchProvider):
    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        if not self.config.tavily_api_key:
            raise SearchError("BC250_TAVILY_API_KEY is required for the Tavily provider")
        # Tavily has no offset parameter. We request enough results and slice locally.
        max_results = min(20, count + offset * count)
        response = await self.client.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.config.tavily_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "topic": "general",
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
                "include_usage": True,
                "safe_search": self.config.safe_search != "off",
            },
        )
        response.raise_for_status()
        raw = response.json().get("results", [])
        start = offset * count
        raw = raw[start : start + count]
        return [
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("content", "")),
                rank=start + index + 1,
                source=self.name,
            )
            for index, item in enumerate(raw)
            if item.get("url")
        ]
