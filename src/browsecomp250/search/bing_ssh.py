from __future__ import annotations

import asyncio
import html
import re
import shlex
import xml.etree.ElementTree as ET
from contextlib import suppress
from urllib.parse import urlencode, urlparse

from bs4 import BeautifulSoup

from ..types import SearchResult
from .base import SearchError, SearchProvider
from .shared_throttle import shared_host_throttle

_CONTROL_TOKEN = re.compile(r"<\|.*?\|>")
_MAX_REMOTE_RESPONSE_BYTES = 4_000_000
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"
)


class BingSSHSearchProvider(SearchProvider):
    """Fetch Bing's server-rendered RSS results through an authorized SSH host."""

    name = "bing_ssh"
    endpoint = "https://www.bing.com/search"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._host_throttle = None
        # Config validation and dry construction can happen outside an event loop.
        with suppress(RuntimeError):
            self._host_throttle = self._get_host_throttle()

    def _get_host_throttle(self):
        if self._host_throttle is None:
            self._host_throttle = shared_host_throttle(
                namespace=self.name,
                host=self.config.bing_ssh_host,
                max_concurrency=self.config.bing_ssh_max_concurrency,
            )
        return self._host_throttle

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        query = " ".join(_CONTROL_TOKEN.sub(" ", query).split())
        if not query:
            raise SearchError("Bing/SSH query was empty after control-token cleanup")
        if not self.config.bing_ssh_host.strip():
            raise SearchError("Bing/SSH requires bing_ssh_host")

        first = offset * count + 1
        target = f"{self.endpoint}?{urlencode({'format': 'rss', 'q': query, 'count': count, 'first': first})}"
        remote_command = shlex.join(
            [
                self.config.bing_ssh_remote_curl_bin,
                "--location",
                "--silent",
                "--show-error",
                "--fail-with-body",
                "--max-time",
                str(self.config.timeout_seconds),
                "--user-agent",
                _USER_AGENT,
                "--header",
                "Accept: application/rss+xml,application/xml,text/xml",
                target,
            ]
        )

        throttle = self._get_host_throttle()
        async with throttle.semaphore:
            await self._wait_for_request_slot()
            process = await asyncio.create_subprocess_exec(
                self.config.bing_ssh_bin,
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={self.config.bing_ssh_connect_timeout_seconds}",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=1",
                self.config.bing_ssh_host,
                remote_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=(
                        self.config.timeout_seconds
                        + self.config.bing_ssh_connect_timeout_seconds
                        + 5
                    ),
                )
            except TimeoutError as exc:
                process.kill()
                await process.wait()
                await self._extend_cooldown()
                raise SearchError("Bing/SSH request timed out") from exc

        if process.returncode != 0:
            await self._extend_cooldown()
            detail = stderr.decode("utf-8", errors="replace").strip()[-500:]
            raise SearchError(f"Bing/SSH request failed with exit {process.returncode}: {detail}")
        if len(stdout) > _MAX_REMOTE_RESPONSE_BYTES:
            raise SearchError("Bing/SSH response exceeded the 4 MB safety limit")

        return self._parse_rss(
            stdout.decode("utf-8", errors="replace"),
            count=count,
            offset=offset,
        )

    async def _wait_for_request_slot(self) -> None:
        throttle = self._get_host_throttle()
        async with throttle.pace_lock:
            loop = asyncio.get_running_loop()
            delay = max(0.0, throttle.next_request_at - loop.time())
            if delay:
                await asyncio.sleep(delay)
            throttle.next_request_at = loop.time() + self.config.bing_ssh_min_interval_seconds

    async def _extend_cooldown(self) -> None:
        throttle = self._get_host_throttle()
        async with throttle.pace_lock:
            loop = asyncio.get_running_loop()
            throttle.next_request_at = max(
                throttle.next_request_at,
                loop.time() + self.config.bing_ssh_error_cooldown_seconds,
            )

    @classmethod
    def _parse_rss(cls, document: str, *, count: int, offset: int) -> list[SearchResult]:
        if "<!DOCTYPE" in document.upper():
            raise SearchError("Bing/SSH RSS response contained a forbidden doctype")
        try:
            root = ET.fromstring(document)
        except ET.ParseError as exc:
            raise SearchError(f"Bing/SSH returned invalid RSS: {exc}") from exc
        if root.tag != "rss":
            raise SearchError(f"Bing/SSH returned unexpected RSS root: {root.tag}")

        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for item in root.findall("./channel/item"):
            title = " ".join((item.findtext("title") or "").split())
            url = (item.findtext("link") or "").strip()
            parsed = urlparse(url)
            if not title or parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            if url in seen_urls:
                continue
            raw_description = html.unescape(item.findtext("description") or "")
            snippet = re.sub(
                r"\s+([,.;:!?])",
                r"\1",
                " ".join(
                    BeautifulSoup(raw_description, "html.parser").get_text(" ", strip=True).split()
                ),
            )
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
        if not results:
            raise SearchError("Bing/SSH RSS returned no organic results")
        return results
