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
        self._disabled_engines: dict[str, str] = {}

    def audit_metrics(self) -> dict[str, object]:
        return {
            "engines": {
                name: {
                    "enabled": name not in self._disabled_engines,
                    "disabled_reason": self._disabled_engines.get(name, ""),
                }
                for name in ("bing", "yahoo")
            }
        }

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
        self._record_probe_health("bing", bing)
        self._record_probe_health("yahoo", yahoo)
        results = self._merge_or_raise(bing, yahoo, resolved_count)
        if not results:
            raise SearchError(f"{self.name} live preflight returned no results")
        return results

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        bing, yahoo = await self._run_enabled_pair(
            self.bing.search(query, count=count, offset=offset),
            self.yahoo.search(query, count=count, offset=offset),
        )
        self._record_single_health("bing", bing)
        self._record_single_health("yahoo", yahoo)
        return self._merge_or_raise(bing, yahoo, count)

    async def search_many(
        self,
        queries: list[str],
        count: int | None = None,
        offset: int = 0,
    ) -> list[list[SearchResult] | Exception]:
        resolved_count = min(count or self.config.results_per_call, 20)
        bing_batches, yahoo_batches = await self._run_enabled_pair(
            self.bing.search_many(queries, count=resolved_count, offset=offset),
            self.yahoo.search_many(queries, count=resolved_count, offset=offset),
        )
        bing_batches = self._normalize_batches("bing", bing_batches, len(queries))
        yahoo_batches = self._normalize_batches("yahoo", yahoo_batches, len(queries))
        return [
            self._merge_or_error(bing, yahoo, resolved_count)
            for bing, yahoo in zip(bing_batches, yahoo_batches, strict=True)
        ]

    async def _run_enabled_pair(self, bing_call, yahoo_call):
        calls = []
        names = []
        disabled_results: dict[str, SearchError] = {}
        for name, call in (("bing", bing_call), ("yahoo", yahoo_call)):
            if name in self._disabled_engines:
                call.close()
                disabled_results[name] = self._disabled_error(name)
                continue
            names.append(name)
            calls.append(call)
        completed = await asyncio.gather(*calls, return_exceptions=True)
        results = dict(zip(names, completed, strict=True))
        results.update(disabled_results)
        return results["bing"], results["yahoo"]

    def _record_probe_health(
        self,
        name: str,
        result: list[SearchResult] | BaseException,
    ) -> None:
        if isinstance(result, BaseException):
            self._disabled_engines[name] = self._error_summary(result)

    def _record_single_health(
        self,
        name: str,
        result: list[SearchResult] | BaseException,
    ) -> None:
        if isinstance(result, BaseException):
            self._disabled_engines.setdefault(name, self._error_summary(result))

    def _normalize_batches(
        self,
        name: str,
        result: list[list[SearchResult] | Exception] | BaseException,
        count: int,
    ) -> list[list[SearchResult] | Exception]:
        if isinstance(result, BaseException):
            self._disabled_engines.setdefault(name, self._error_summary(result))
            return [self._disabled_error(name) for _ in range(count)]
        if result and all(isinstance(batch, Exception) for batch in result):
            first = next(batch for batch in result if isinstance(batch, Exception))
            self._disabled_engines.setdefault(name, self._error_summary(first))
        return result

    def _disabled_error(self, name: str) -> SearchError:
        reason = self._disabled_engines.get(name, "engine unavailable")
        return SearchError(f"{name} disabled for this run after live failure: {reason}")

    @staticmethod
    def _error_summary(error: BaseException) -> str:
        return f"{type(error).__name__}: {error}"[:500]

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
