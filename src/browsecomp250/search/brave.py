from __future__ import annotations

from typing import Any

from ..types import SearchResult
from .base import SearchError, SearchProvider


class BraveSearchProvider(SearchProvider):
    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        if not self.config.brave_api_key:
            raise SearchError("BC250_BRAVE_API_KEY is required for the Brave provider")
        params: dict[str, Any] = {
            "q": query,
            "count": count,
            "offset": offset,
            "country": self.config.country,
            "search_lang": self.config.language,
            "safesearch": self.config.safe_search,
            "extra_snippets": "true",
        }
        response = await self.client.get(
            self.endpoint,
            params=params,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.config.brave_api_key,
            },
        )
        response.raise_for_status()
        data = response.json()
        raw = data.get("web", {}).get("results", [])
        return [
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("description", "")),
                rank=offset * count + index + 1,
                source=self.name,
                extra_snippets=[str(value) for value in item.get("extra_snippets", [])],
            )
            for index, item in enumerate(raw)
            if item.get("url")
        ]
