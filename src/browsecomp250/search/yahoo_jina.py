from __future__ import annotations

import asyncio
import html
import re
from urllib.parse import urlencode, urlparse

from ..types import SearchResult
from .base import SearchError, SearchProvider

_NUMBERED_LINE = re.compile(r"^\s*\d+\.\s+(?P<body>.+)$")
_MARKDOWN_URL = re.compile(r"\]\((?P<url>https?://[^)\s]+)\)")
_MARKDOWN_TITLE = re.compile(r"###\s*(?P<title>.*?)\]\(https?://")
_MARKDOWN_DECORATION = re.compile(r"[*_`]+")


class YahooJinaSearchProvider(SearchProvider):
    """Yahoo web results rendered as markdown by the existing Jina reader."""

    name = "yahoo_jina"
    endpoint = "https://r.jina.ai/http://search.yahoo.com/search"
    _navigation_hosts = {
        "images.search.yahoo.com",
        "login.yahoo.com",
        "s.yimg.com",
        "scout.yahoo.com",
        "search.yahoo.com",
        "up.yimg.com",
        "video.search.yahoo.com",
        "www.yahoo.com",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_semaphore = asyncio.Semaphore(8)

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        first = offset * count + 1
        target = f"{self.endpoint}?{urlencode({'p': query, 'b': first, 'pz': count})}"
        async with self._request_semaphore:
            response = await self.client.get(
                target,
                headers={
                    "Accept": "text/plain",
                    "User-Agent": "BrowseComp250-Star/0.1",
                },
            )
        response.raise_for_status()
        results = self._parse_markdown(response.text, count=count, offset=offset)
        if not results:
            raise SearchError("Yahoo/Jina response contained no usable web results")
        return results

    @classmethod
    def _parse_markdown(cls, text: str, *, count: int, offset: int) -> list[SearchResult]:
        _, marker, search_results = text.partition("## Search Results")
        if not marker:
            search_results = text

        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for line in search_results.splitlines():
            numbered = _NUMBERED_LINE.match(line)
            if not numbered:
                continue
            body = numbered.group("body")
            urls = [match.group("url") for match in _MARKDOWN_URL.finditer(body)]
            result_url = next(
                (url for url in urls if cls._is_result_url(url) and url not in seen_urls),
                None,
            )
            if result_url is None:
                continue

            title_match = _MARKDOWN_TITLE.search(body)
            title = title_match.group("title") if title_match else ""
            if not title:
                title = urlparse(result_url).hostname or result_url
            title = cls._clean_text(title)

            link_end = body.find(f"]({result_url})")
            snippet = body[link_end + len(result_url) + 3 :] if link_end >= 0 else ""
            snippet = cls._clean_text(snippet)

            seen_urls.add(result_url)
            results.append(
                SearchResult(
                    title=title,
                    url=result_url,
                    snippet=snippet,
                    rank=offset * count + len(results) + 1,
                    source=cls.name,
                )
            )
            if len(results) >= count:
                break
        return results

    @classmethod
    def _is_result_url(cls, url: str) -> bool:
        host = (urlparse(html.unescape(url)).hostname or "").lower()
        return bool(host and host not in cls._navigation_hosts)

    @staticmethod
    def _clean_text(value: str) -> str:
        value = html.unescape(value)
        value = _MARKDOWN_DECORATION.sub("", value)
        return " ".join(value.split()).strip(" -")
