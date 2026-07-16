from __future__ import annotations

import asyncio
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config import SearchConfig
from ..types import SearchResult
from .base import SearchError, SearchProvider
from .bing_ssh import BingSSHSearchProvider
from .yahoo_ssh import YahooSSHSearchProvider


class BingYahooSSHSearchProvider(SearchProvider):
    """Interleave two server-side search engines without using a user browser."""

    name = "bing_yahoo_ssh"

    def __init__(self, config: SearchConfig, client: httpx.AsyncClient | None = None):
        super().__init__(config, client=client)
        self.bing = BingSSHSearchProvider(config, client=self.client)
        self.yahoo = YahooSSHSearchProvider(config, client=self.client)

    async def close(self) -> None:
        await self.bing.close()
        await self.yahoo.close()
        await super().close()

    async def probe_live(self, query: str | None = None, count: int = 1) -> list[SearchResult]:
        """Probe both transports directly so a warm child cache cannot mask an outage."""

        normalized = " ".join((query or self.config.live_preflight_query).split()).strip()
        if not normalized:
            raise SearchError("Search live preflight query is empty")
        resolved_count = min(count, self.config.results_per_call, 20)
        bing, yahoo = await asyncio.gather(
            self.bing.probe_live(normalized, count=resolved_count),
            self.yahoo.probe_live(normalized, count=resolved_count),
            return_exceptions=True,
        )
        results = self._merge_or_raise(bing, yahoo, resolved_count)
        if not results:
            raise SearchError(f"{self.name} live preflight returned no results")
        return results

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        bing, yahoo = await asyncio.gather(
            self.bing.search(query, count=count, offset=offset),
            self.yahoo.search(query, count=count, offset=offset),
            return_exceptions=True,
        )
        return self._merge_or_raise(bing, yahoo, count)

    async def search_many(
        self,
        queries: list[str],
        count: int | None = None,
        offset: int = 0,
    ) -> list[list[SearchResult] | Exception]:
        resolved_count = min(count or self.config.results_per_call, 20)
        bing_batches, yahoo_batches = await asyncio.gather(
            self.bing.search_many(queries, count=resolved_count, offset=offset),
            self.yahoo.search_many(queries, count=resolved_count, offset=offset),
        )
        return [
            self._merge_or_error(bing, yahoo, resolved_count)
            for bing, yahoo in zip(bing_batches, yahoo_batches, strict=True)
        ]

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
            return SearchError(f"Both server-side search engines failed: {first}; {second}")

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


__all__ = ["BingYahooSSHSearchProvider"]
