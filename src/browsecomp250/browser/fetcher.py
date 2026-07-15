from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import asdict
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from ..cache import SQLiteCache
from ..config import BrowserConfig
from ..types import PageDocument
from .extract import extract_document
from .safety import UnsafeURLError, assert_safe_url


class BrowserError(RuntimeError):
    pass


_MARKDOWN_LINK = re.compile(r"\[([^\]]{1,300})\]\((https?://[^)\s]+)\)")


class PageFetcher:
    def __init__(self, config: BrowserConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=config.timeout_seconds,
            follow_redirects=True,
            max_redirects=config.max_redirects,
            headers={"User-Agent": config.user_agent},
        )
        self.cache = SQLiteCache(config.cache_path, "pages")

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _cache_request(self, url: str) -> dict[str, Any]:
        return {
            "url": url,
            "backend": self.config.backend,
            "max_response_bytes": self.config.max_response_bytes,
            "max_links_per_page": self.config.max_links_per_page,
            "user_agent": self.config.user_agent,
            "reader_fallback_enabled": self.config.reader_fallback_enabled,
            "reader_fallback_base_url": self.config.reader_fallback_base_url,
            "reader_fallback_min_chars": self.config.reader_fallback_min_chars,
        }

    async def fetch(self, url: str) -> PageDocument:
        request = self._cache_request(url)
        if self.config.cache_mode in {"read", "readwrite"}:
            cached = self.cache.get(request)
            if cached is not None:
                return PageDocument(**cached)
            if self.config.cache_mode == "read":
                raise BrowserError(f"Read-only page cache miss for URL: {url}")

        if self.config.backend == "playwright":
            document = await self._fetch_playwright(url)
        elif self.config.backend == "auto":
            document = await self._fetch_direct_or_reader(url)
            if len(document.text) < 300 and "html" in document.content_type.lower():
                with suppress(BrowserError):
                    document = await self._fetch_playwright(url)
        else:
            document = await self._fetch_direct_or_reader(url)

        if self.config.cache_mode in {"write", "readwrite", "refresh"}:
            self.cache.put(request, asdict(document))
        return document

    async def _fetch_direct_or_reader(self, url: str) -> PageDocument:
        try:
            direct = await self._fetch_direct(url)
        except BrowserError as direct_error:
            if not self.config.reader_fallback_enabled:
                raise
            try:
                return await self._fetch_reader(url)
            except BrowserError as reader_error:
                raise BrowserError(
                    f"Direct fetch failed ({direct_error}); reader fallback failed ({reader_error})"
                ) from reader_error

        if (
            self.config.reader_fallback_enabled
            and len(direct.text) < self.config.reader_fallback_min_chars
        ):
            with suppress(BrowserError):
                reader = await self._fetch_reader(url)
                if len(reader.text) > len(direct.text):
                    return reader
        return direct

    async def _fetch_reader(self, url: str) -> PageDocument:
        try:
            await assert_safe_url(
                url,
                block_private_networks=self.config.block_private_networks,
                allow_nonstandard_ports=self.config.allow_nonstandard_ports,
            )
        except UnsafeURLError as exc:
            raise BrowserError(str(exc)) from exc

        parsed = urlsplit(url)
        source_url = parsed._replace(fragment="").geturl()
        candidates = [source_url]
        if parsed.scheme == "https":
            candidates.append(parsed._replace(scheme="http", fragment="").geturl())
        proxy: PageDocument | None = None
        resolved_source_url = source_url
        errors: list[str] = []
        for candidate in candidates:
            try:
                await assert_safe_url(
                    candidate,
                    block_private_networks=self.config.block_private_networks,
                    allow_nonstandard_ports=self.config.allow_nonstandard_ports,
                )
                reader_url = self.config.reader_fallback_base_url.rstrip("/") + "/" + candidate
                proxy = await self._fetch_direct(reader_url)
                resolved_source_url = candidate
                break
            except (BrowserError, UnsafeURLError) as exc:
                errors.append(f"{candidate}: {exc}")
        if proxy is None:
            raise BrowserError(
                "Reader could not fetch any validated source variant: " + "; ".join(errors)
            )
        links: list[dict[str, str]] = []
        seen: set[str] = set()
        for match in _MARKDOWN_LINK.finditer(proxy.text):
            target = match.group(2).rstrip(".,;:")
            if target in seen:
                continue
            seen.add(target)
            links.append({"text": match.group(1), "url": target})
            if len(links) >= self.config.max_links_per_page:
                break
        title_match = re.search(r"^Title:\s*(.+)$", proxy.text, flags=re.M)
        return PageDocument(
            requested_url=url,
            final_url=resolved_source_url,
            title=title_match.group(1).strip() if title_match else proxy.title,
            text=proxy.text,
            content_type="text/markdown; source=reader-fallback",
            status_code=proxy.status_code,
            links=links,
            fetched_at=proxy.fetched_at,
            sha256=proxy.sha256,
            truncated=proxy.truncated,
        )

    async def _fetch_direct(self, url: str) -> PageDocument:
        current_url = url
        redirect_statuses = {301, 302, 303, 307, 308}

        for redirect_count in range(self.config.max_redirects + 1):
            try:
                await assert_safe_url(
                    current_url,
                    block_private_networks=self.config.block_private_networks,
                    allow_nonstandard_ports=self.config.allow_nonstandard_ports,
                )
            except UnsafeURLError as exc:
                raise BrowserError(str(exc)) from exc

            try:
                async with self.client.stream(
                    "GET",
                    current_url,
                    follow_redirects=False,
                    headers={
                        "Accept": "text/html,application/xhtml+xml,application/pdf,"
                        "text/plain,application/json;q=0.9,*/*;q=0.5"
                    },
                ) as response:
                    if response.status_code in redirect_statuses:
                        location = response.headers.get("location")
                        if not location:
                            raise BrowserError(
                                f"Redirect response omitted Location header: {current_url}"
                            )
                        if redirect_count >= self.config.max_redirects:
                            raise BrowserError(
                                f"Exceeded {self.config.max_redirects} redirects: {url}"
                            )
                        next_url = urljoin(str(response.url), location)
                        try:
                            # Validate before making the next request; validating only the
                            # final URL would still permit redirect-based SSRF.
                            await assert_safe_url(
                                next_url,
                                block_private_networks=self.config.block_private_networks,
                                allow_nonstandard_ports=self.config.allow_nonstandard_ports,
                            )
                        except UnsafeURLError as exc:
                            raise BrowserError(
                                f"Blocked unsafe redirect from {current_url} to {next_url}: {exc}"
                            ) from exc
                        current_url = next_url
                        continue

                    response.raise_for_status()
                    final_url = str(response.url)
                    await assert_safe_url(
                        final_url,
                        block_private_networks=self.config.block_private_networks,
                        allow_nonstandard_ports=self.config.allow_nonstandard_ports,
                    )
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.config.max_response_bytes:
                            raise BrowserError(
                                f"Response exceeded {self.config.max_response_bytes} bytes: {url}"
                            )
                        chunks.append(chunk)
                    content = b"".join(chunks)
                    content_type = response.headers.get("content-type", "application/octet-stream")
                    return extract_document(
                        content,
                        requested_url=url,
                        final_url=final_url,
                        status_code=response.status_code,
                        content_type=content_type,
                        max_links=self.config.max_links_per_page,
                    )
            except BrowserError:
                raise
            except UnsafeURLError as exc:
                raise BrowserError(str(exc)) from exc
            except (httpx.HTTPError, ValueError) as exc:
                raise BrowserError(f"Failed to fetch {current_url}: {exc}") from exc

        raise BrowserError(f"Exceeded redirect limit while fetching: {url}")

    async def _fetch_playwright(self, url: str) -> PageDocument:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise BrowserError(
                "Playwright backend requested but optional dependency is not installed. "
                "Run `pip install -e '.[browser]' && playwright install chromium`."
            ) from exc
        try:
            await assert_safe_url(
                url,
                block_private_networks=self.config.block_private_networks,
                allow_nonstandard_ports=self.config.allow_nonstandard_ports,
            )
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(user_agent=self.config.user_agent)
                    page = await context.new_page()

                    async def guard_request(route: Any) -> None:
                        request_url = route.request.url
                        scheme = urlsplit(request_url).scheme.casefold()
                        if scheme in {"data", "blob", "about"}:
                            await route.continue_()
                            return
                        try:
                            await assert_safe_url(
                                request_url,
                                block_private_networks=self.config.block_private_networks,
                                allow_nonstandard_ports=self.config.allow_nonstandard_ports,
                            )
                        except UnsafeURLError:
                            await route.abort("blockedbyclient")
                            return
                        await route.continue_()

                    await page.route("**/*", guard_request)
                    navigation = await page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=self.config.timeout_seconds * 1000,
                    )
                    html = await page.content()
                    final_url = page.url
                    status_code = navigation.status if navigation is not None else 200
                finally:
                    await browser.close()
            await assert_safe_url(
                final_url,
                block_private_networks=self.config.block_private_networks,
                allow_nonstandard_ports=self.config.allow_nonstandard_ports,
            )
            content = html.encode("utf-8")
            if len(content) > self.config.max_response_bytes:
                raise BrowserError("Rendered page exceeded maximum response size")
            return extract_document(
                content,
                requested_url=url,
                final_url=final_url,
                status_code=status_code,
                content_type="text/html; rendered=playwright",
                max_links=self.config.max_links_per_page,
            )
        except BrowserError:
            raise
        except UnsafeURLError as exc:
            raise BrowserError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"Playwright failed for {url}: {exc}") from exc
