from pathlib import Path

import httpx
import pytest

from browsecomp250.browser.fetcher import PageFetcher
from browsecomp250.config import BrowserConfig


@pytest.mark.asyncio
async def test_fetch_and_extract_html(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><head><title>Example</title></head><body><main><h1>Hello</h1><p>Useful fact.</p><a href='/x'>Next</a></main></body></html>",
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    fetcher = PageFetcher(
        BrowserConfig(
            block_private_networks=False,
            cache_mode="off",
            cache_path=tmp_path / "pages.sqlite3",
        ),
        client,
    )
    doc = await fetcher.fetch("https://example.test/")
    assert doc.title == "Example"
    assert "Useful fact" in doc.text
    assert doc.links[0]["url"] == "https://example.test/x"
    await client.aclose()


@pytest.mark.asyncio
async def test_manual_redirect_then_extract(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "first.test":
            return httpx.Response(
                302,
                headers={"location": "https://second.test/final"},
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="redirected fact",
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = PageFetcher(
        BrowserConfig(
            block_private_networks=False,
            cache_mode="off",
            cache_path=tmp_path / "pages.sqlite3",
        ),
        client,
    )
    doc = await fetcher.fetch("https://first.test/start")
    assert doc.final_url == "https://second.test/final"
    assert doc.text == "redirected fact"
    assert seen == ["https://first.test/start", "https://second.test/final"]
    await client.aclose()


@pytest.mark.asyncio
async def test_read_only_page_cache_fails_closed(tmp_path: Path) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, text="unexpected", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = PageFetcher(
        BrowserConfig(
            block_private_networks=False,
            cache_mode="read",
            cache_path=tmp_path / "pages.sqlite3",
        ),
        client,
    )
    from browsecomp250.browser.fetcher import BrowserError

    with pytest.raises(BrowserError, match="Read-only page cache miss"):
        await fetcher.fetch("https://example.test/")
    assert called is False
    await client.aclose()


@pytest.mark.asyncio
async def test_reader_fallback_preserves_original_public_url(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "blocked.test":
            raise httpx.ConnectError("origin unavailable", request=request)
        assert request.url.host == "r.jina.ai"
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text=(
                "Title: Recovered page\n\n"
                "URL Source: https://blocked.test/fact\n\n"
                "Markdown Content:\nAuthoritative fact. "
                "[Source](https://source.test/record)"
            ),
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = PageFetcher(
        BrowserConfig(
            block_private_networks=False,
            cache_mode="off",
            cache_path=tmp_path / "pages.sqlite3",
            reader_fallback_enabled=True,
        ),
        client,
    )
    doc = await fetcher.fetch("https://blocked.test/fact")
    assert seen == [
        "https://blocked.test/fact",
        "https://r.jina.ai/https://blocked.test/fact",
    ]
    assert doc.requested_url == "https://blocked.test/fact"
    assert doc.final_url == "https://blocked.test/fact"
    assert doc.title == "Recovered page"
    assert "Authoritative fact" in doc.text
    assert doc.content_type == "text/markdown; source=reader-fallback"
    assert doc.links == [{"text": "Source", "url": "https://source.test/record"}]
    await client.aclose()
