from __future__ import annotations

import asyncio
import re
import shlex
from urllib.parse import unquote, urlencode, urlparse

from bs4 import BeautifulSoup

from ..types import SearchResult
from .base import SearchError, SearchProvider

_CONTROL_TOKEN = re.compile(r"<\|.*?\|>")
_QUERY_TOKEN = re.compile(r"[a-z0-9]+", flags=re.I)
_MAX_REMOTE_RESPONSE_BYTES = 4_000_000
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"
)


class BraveSSHSearchProvider(SearchProvider):
    """Fetch Brave's server-rendered organic results through an authorized SSH host."""

    name = "brave_ssh"
    endpoint = "https://search.brave.com/search"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_semaphore = asyncio.Semaphore(self.config.brave_ssh_max_concurrency)
        self._pace_lock = asyncio.Lock()
        self._next_request_at = 0.0

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        query = " ".join(_CONTROL_TOKEN.sub(" ", query).split())
        if not query:
            raise SearchError("Brave/SSH query was empty after control-token cleanup")
        if not self.config.brave_ssh_host.strip():
            raise SearchError("Brave/SSH requires brave_ssh_host")

        target = f"{self.endpoint}?{urlencode({'q': query, 'source': 'web', 'offset': offset})}"
        remote_command = shlex.join(
            [
                self.config.brave_ssh_remote_curl_bin,
                "--location",
                "--silent",
                "--show-error",
                "--fail-with-body",
                "--max-time",
                str(self.config.timeout_seconds),
                "--user-agent",
                _USER_AGENT,
                "--header",
                "Accept: text/html,application/xhtml+xml",
                target,
            ]
        )

        async with self._request_semaphore:
            await self._wait_for_request_slot()
            process = await asyncio.create_subprocess_exec(
                self.config.brave_ssh_bin,
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={self.config.brave_ssh_connect_timeout_seconds}",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=1",
                self.config.brave_ssh_host,
                remote_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=(
                        self.config.timeout_seconds
                        + self.config.brave_ssh_connect_timeout_seconds
                        + 5
                    ),
                )
            except TimeoutError as exc:
                process.kill()
                await process.wait()
                await self._extend_cooldown()
                raise SearchError("Brave/SSH request timed out") from exc

        if process.returncode != 0:
            await self._extend_cooldown()
            detail = stderr.decode("utf-8", errors="replace").strip()[-500:]
            raise SearchError(
                f"Brave/SSH request failed with exit {process.returncode}: {detail}"
            )
        if len(stdout) > _MAX_REMOTE_RESPONSE_BYTES:
            raise SearchError("Brave/SSH response exceeded the 4 MB safety limit")

        return self._parse_html(
            stdout.decode("utf-8", errors="replace"),
            query=query,
            count=count,
            offset=offset,
        )

    async def _wait_for_request_slot(self) -> None:
        async with self._pace_lock:
            loop = asyncio.get_running_loop()
            delay = max(0.0, self._next_request_at - loop.time())
            if delay:
                await asyncio.sleep(delay)
            self._next_request_at = loop.time() + self.config.brave_ssh_min_interval_seconds

    async def _extend_cooldown(self) -> None:
        async with self._pace_lock:
            loop = asyncio.get_running_loop()
            self._next_request_at = max(
                self._next_request_at,
                loop.time() + self.config.brave_ssh_error_cooldown_seconds,
            )

    @classmethod
    def _parse_html(
        cls,
        document: str,
        *,
        query: str,
        count: int,
        offset: int,
    ) -> list[SearchResult]:
        soup = BeautifulSoup(document, "html.parser")
        wrappers = soup.select(".result-wrapper")
        if not wrappers:
            title = " ".join((soup.title.get_text(" ", strip=True) if soup.title else "").split())
            raise SearchError(
                "Brave/SSH returned no organic result wrappers"
                + (f" (page title: {title[:120]})" if title else "")
            )

        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for wrapper in wrappers:
            anchor = wrapper.select_one("a[href]")
            title_node = wrapper.select_one(".search-snippet-title, .title")
            if anchor is None:
                continue
            url = str(anchor.get("href") or "").strip()
            parsed = urlparse(url)
            title = " ".join(
                (title_node or anchor).get_text(" ", strip=True).split()
            )
            if (
                not title
                or parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or url in seen_urls
                or cls._is_unusable_result(query, title, url)
                or cls._looks_like_query_mirror(query, url)
            ):
                continue
            snippet_node = wrapper.select_one(
                ".generic-snippet .content, .snippet-description, .snippet"
            )
            snippet = (
                " ".join(snippet_node.get_text(" ", strip=True).split())
                if snippet_node is not None
                else ""
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
            raise SearchError("Brave/SSH organic results were all invalid or query mirrors")
        return results

    @staticmethod
    def _looks_like_query_mirror(query: str, url: str) -> bool:
        query_terms = set(_QUERY_TOKEN.findall(query.casefold()))
        path_terms = set(_QUERY_TOKEN.findall(unquote(urlparse(url).path).casefold()))
        if len(query_terms) < 5 or len(path_terms) < 5:
            return False
        overlap = len(query_terms & path_terms)
        return overlap >= 5 and overlap / len(query_terms) >= 0.65

    @staticmethod
    def _is_unusable_result(query: str, title: str, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.hostname or "").casefold().removeprefix("www.")
        path = parsed.path.casefold()
        if host == "instagram.com" and path.startswith("/popular/"):
            return True
        query_mentions_benchmark = "browsecomp" in query.casefold()
        result_mentions_benchmark = "browsecomp" in f"{title}\n{url}".casefold()
        return result_mentions_benchmark and not query_mentions_benchmark


__all__ = ["BraveSSHSearchProvider"]
