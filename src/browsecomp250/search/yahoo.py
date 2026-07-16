from __future__ import annotations

import asyncio
import re
from contextlib import suppress
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup

from ..types import SearchResult
from .base import SearchError, SearchProvider
from .shared_throttle import shared_host_throttle

_YAHOO_TARGET = re.compile(r"/RU=(?P<target>.*?)/RK=")
_CONTROL_TOKEN = re.compile(r"<\|.*?\|>")


class YahooSearchProvider(SearchProvider):
    """Server-side Yahoo organic search without a user browser."""

    name = "yahoo"
    endpoint = "https://search.yahoo.com/search"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._host_throttle = None
        # Config validation and dry construction can happen outside an event loop.
        with suppress(RuntimeError):
            self._host_throttle = self._get_host_throttle()

    def _throttle_host(self) -> str:
        return urlparse(self.endpoint).netloc

    def _get_host_throttle(self):
        if self._host_throttle is None:
            self._host_throttle = shared_host_throttle(
                namespace=self.name,
                host=self._throttle_host(),
                max_concurrency=self.config.yahoo_max_concurrency,
            )
        return self._host_throttle

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        query = " ".join(_CONTROL_TOKEN.sub(" ", query).split())
        if not query:
            raise SearchError("Yahoo query was empty after control-token cleanup")
        first = offset * count + 1
        throttle = self._get_host_throttle()
        async with throttle.semaphore:
            await self._wait_for_request_slot()
            response = await self.client.get(
                self.endpoint,
                params={"p": query, "b": first, "pz": count},
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": f"{self.config.language},en;q=0.8",
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/136.0 Safari/537.36"
                    ),
                },
            )
            if response.status_code == 429 or response.status_code >= 500:
                await self._extend_cooldown()
        response.raise_for_status()
        results = self._parse_html(response.text, count=count, offset=offset)
        return results

    async def _wait_for_request_slot(self) -> None:
        throttle = self._get_host_throttle()
        async with throttle.pace_lock:
            loop = asyncio.get_running_loop()
            delay = max(0.0, throttle.next_request_at - loop.time())
            if delay:
                await asyncio.sleep(delay)
            throttle.next_request_at = loop.time() + self.config.yahoo_min_interval_seconds

    async def _extend_cooldown(self) -> None:
        throttle = self._get_host_throttle()
        async with throttle.pace_lock:
            loop = asyncio.get_running_loop()
            throttle.next_request_at = max(
                throttle.next_request_at,
                loop.time() + self.config.yahoo_error_cooldown_seconds,
            )

    @classmethod
    def _parse_html(cls, document: str, *, count: int, offset: int) -> list[SearchResult]:
        soup = BeautifulSoup(document, "html.parser")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for container in soup.select("#web ol.searchCenterMiddle div.algo-sr"):
            anchor = container.select_one("div.compTitle a[href]")
            if anchor is None:
                continue
            url = cls._result_url(str(anchor.get("href") or ""))
            if not url or url in seen_urls:
                continue
            title_node = anchor.select_one("h3")
            title = (title_node or anchor).get_text(" ", strip=True)
            snippet_node = container.select_one("div.compText")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            seen_urls.add(url)
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    rank=offset * count + len(results) + 1,
                    source=cls.name,
                )
            )
            if len(results) >= count:
                break
        return results

    @staticmethod
    def _result_url(value: str) -> str:
        match = _YAHOO_TARGET.search(urlparse(value).path)
        return unquote(match.group("target")) if match else value
