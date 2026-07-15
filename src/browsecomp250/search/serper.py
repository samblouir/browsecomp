from __future__ import annotations

from ..types import SearchResult
from .base import SearchError, SearchProvider


class SerperSearchProvider(SearchProvider):
    name = "serper"
    endpoint = "https://google.serper.dev/search"

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        if not self.config.serper_api_key:
            raise SearchError("BC250_SERPER_API_KEY is required for the Serper provider")
        response = await self.client.post(
            self.endpoint,
            headers={
                "X-API-KEY": self.config.serper_api_key,
                "Content-Type": "application/json",
            },
            json={
                "q": query,
                "num": count,
                "page": offset + 1,
                "gl": self.config.country.lower(),
                "hl": self.config.language,
            },
        )
        response.raise_for_status()
        raw = response.json().get("organic", [])
        return [
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("link", "")),
                snippet=str(item.get("snippet", "")),
                rank=int(item.get("position") or (offset * count + index + 1)),
                source=self.name,
            )
            for index, item in enumerate(raw[:count])
            if item.get("link")
        ]
