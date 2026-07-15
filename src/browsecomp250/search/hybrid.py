from __future__ import annotations

import asyncio
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config import SearchConfig
from ..types import SearchResult
from .base import SearchError, SearchProvider
from .brave import BraveSearchProvider
from .google_chrome import GoogleChromeSearchProvider


class HybridSearchProvider(SearchProvider):
    """Combine Google-in-user-Chrome discovery with Brave API results."""

    name = "hybrid"

    def __init__(self, config: SearchConfig, client: httpx.AsyncClient | None = None):
        super().__init__(config, client=client)
        self.google = GoogleChromeSearchProvider(config, client=self.client)
        self.brave = BraveSearchProvider(config, client=self.client)

    async def close(self) -> None:
        await self.google.close()
        await self.brave.close()
        await super().close()

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        mode = self.config.hybrid_mode
        if mode == "google_first":
            return await self._fallback_one(self.google, self.brave, query, count, offset)
        if mode == "brave_first":
            return await self._fallback_one(self.brave, self.google, query, count, offset)
        google, brave = await asyncio.gather(
            self.google.search(query, count=count, offset=offset),
            self.brave.search(query, count=count, offset=offset),
            return_exceptions=True,
        )
        return self._merge_or_raise(google, brave, count)

    async def search_many(
        self,
        queries: list[str],
        count: int | None = None,
        offset: int = 0,
    ) -> list[list[SearchResult] | Exception]:
        resolved_count = min(count or self.config.results_per_call, 20)
        mode = self.config.hybrid_mode
        if mode != "merge":
            return await super().search_many(queries, count=resolved_count, offset=offset)

        google_batches, brave_batches = await asyncio.gather(
            self.google.search_many(queries, count=resolved_count, offset=offset),
            self.brave.search_many(queries, count=resolved_count, offset=offset),
        )
        return [
            self._merge_or_error(google, brave, resolved_count)
            for google, brave in zip(google_batches, brave_batches, strict=True)
        ]

    @staticmethod
    async def _fallback_one(
        primary: SearchProvider,
        fallback: SearchProvider,
        query: str,
        count: int,
        offset: int,
    ) -> list[SearchResult]:
        try:
            results = await primary.search(query, count=count, offset=offset)
            if results:
                return results
        except Exception:  # noqa: BLE001 - fallback is the intended policy
            pass
        return await fallback.search(query, count=count, offset=offset)

    @classmethod
    def _merge_or_raise(
        cls,
        first: list[SearchResult] | BaseException,
        second: list[SearchResult] | BaseException,
        count: int,
    ) -> list[SearchResult]:
        result = cls._merge_or_error(first, second, count)
        if isinstance(result, Exception):
            raise result
        return result

    @classmethod
    def _merge_or_error(
        cls,
        first: list[SearchResult] | BaseException,
        second: list[SearchResult] | BaseException,
        count: int,
    ) -> list[SearchResult] | Exception:
        usable = [batch for batch in (first, second) if isinstance(batch, list)]
        if not usable:
            return SearchError(f"Both hybrid search engines failed: {first}; {second}")
        merged: list[SearchResult] = []
        seen: set[str] = set()
        for rank in range(max((len(batch) for batch in usable), default=0)):
            for batch in usable:
                if rank >= len(batch):
                    continue
                item = batch[rank]
                key = cls._url_key(item.url)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(
                    SearchResult(
                        title=item.title,
                        url=item.url,
                        snippet=item.snippet,
                        rank=len(merged) + 1,
                        source=item.source,
                        extra_snippets=item.extra_snippets,
                    )
                )
                if len(merged) >= count:
                    return merged
        return merged

    @staticmethod
    def _url_key(url: str) -> str:
        parsed = urlsplit(url)
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower().removeprefix("www."),
                parsed.path.rstrip("/"),
                parsed.query,
                "",
            )
        )


__all__ = ["HybridSearchProvider"]
