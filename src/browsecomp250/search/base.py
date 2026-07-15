from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

import httpx

from ..cache import SQLiteCache
from ..config import SearchConfig
from ..types import SearchResult


class SearchError(RuntimeError):
    pass


class SearchProvider(ABC):
    name: str

    def __init__(self, config: SearchConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=config.timeout_seconds)
        self.cache = SQLiteCache(config.cache_path, f"search:{self.name}")

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def audit_metrics(self) -> dict[str, Any]:
        """Return provider-owned counters that are safe for the run manifest."""

        return {}

    def _cache_request(self, query: str, count: int, offset: int) -> dict[str, Any]:
        return {
            "provider": self.name,
            "query": query,
            "count": count,
            "offset": offset,
            "country": self.config.country,
            "language": self.config.language,
            "safe_search": self.config.safe_search,
        }

    async def search(
        self, query: str, count: int | None = None, offset: int = 0
    ) -> list[SearchResult]:
        query = " ".join(query.split()).strip()
        if not query:
            raise SearchError("Search query is empty")
        count = min(count or self.config.results_per_call, 20)
        request = self._cache_request(query, count, offset)
        if self.config.cache_mode in {"read", "readwrite"}:
            cached = self.cache.get(request)
            if cached is not None:
                return [SearchResult(**item) for item in cached]
            if self.config.cache_mode == "read":
                raise SearchError(
                    f"Read-only search cache miss for provider={self.name}, query={query!r}"
                )

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                results = await self._search_live(query, count, offset)
                if self.config.cache_mode in {"write", "readwrite", "refresh"}:
                    self.cache.put(request, [asdict(item) for item in results])
                return results
            except (httpx.HTTPError, SearchError, ValueError, KeyError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                await asyncio.sleep(min(2**attempt, 15))
        raise SearchError(f"{self.name} search failed after retries: {last_error}")

    async def search_many(
        self,
        queries: list[str],
        count: int | None = None,
        offset: int = 0,
    ) -> list[list[SearchResult] | Exception]:
        """Run independent searches concurrently.

        Providers backed by a browser can override this to batch all queries into
        one browser launch while preserving this ordered, per-query result shape.
        """

        return await asyncio.gather(
            *(self.search(query, count=count, offset=offset) for query in queries),
            return_exceptions=True,
        )

    @abstractmethod
    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        raise NotImplementedError
