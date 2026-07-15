from pathlib import Path

import httpx
import pytest

from browsecomp250.config import SearchConfig
from browsecomp250.search.brave import BraveSearchProvider
from browsecomp250.search.google_chrome import GoogleChromeSearchProvider
from browsecomp250.search.hybrid import HybridSearchProvider
from browsecomp250.search.searxng import SearXNGSearchProvider
from browsecomp250.search.tavily import TavilySearchProvider
from browsecomp250.search.yahoo import YahooSearchProvider
from browsecomp250.search.yahoo_jina import YahooJinaSearchProvider
from browsecomp250.types import SearchResult


@pytest.mark.asyncio
async def test_brave_adapter(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Subscription-Token"] == "secret"
        return httpx.Response(
            200,
            json={
                "web": {"results": [{"title": "T", "url": "https://e.test", "description": "S"}]}
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = BraveSearchProvider(
        SearchConfig(
            provider="brave",
            brave_api_key="secret",
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
        ),
        client,
    )
    results = await provider.search("q")
    assert results[0].title == "T"
    await client.aclose()


@pytest.mark.asyncio
async def test_tavily_adapter(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer tvly"
        return httpx.Response(
            200,
            json={"results": [{"title": "T", "url": "https://e.test", "content": "C"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = TavilySearchProvider(
        SearchConfig(
            provider="tavily",
            tavily_api_key="tvly",
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
        ),
        client,
    )
    assert (await provider.search("q"))[0].snippet == "C"
    await client.aclose()


@pytest.mark.asyncio
async def test_searxng_adapter(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["format"] == "json"
        return httpx.Response(
            200,
            json={"results": [{"title": "T", "url": "https://e.test", "content": "C"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SearXNGSearchProvider(
        SearchConfig(
            provider="searxng",
            searxng_base_url="https://search.test",
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
        ),
        client,
    )
    assert (await provider.search("q"))[0].source == "searxng"
    await client.aclose()


@pytest.mark.asyncio
async def test_yahoo_jina_adapter(tmp_path: Path) -> None:
    markdown = """\
## Search Results

1. [![Image](https://s.yimg.com/icon.png) NASA https://www.nasa.gov ### Apollo 11 - NASA](https://www.nasa.gov/mission/apollo-11/) The primary objective was a crewed lunar landing.
2. [Yahoo Scout](https://scout.yahoo.com/chat?q=apollo)
3. [![Image](https://s.yimg.com/icon2.png) Britannica ### Apollo 11 | History](https://www.britannica.com/topic/Apollo-11) Independent history.
"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "search.yahoo.com" in str(request.url)
        return httpx.Response(200, text=markdown, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = YahooJinaSearchProvider(
        SearchConfig(
            provider="yahoo_jina",
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
        ),
        client,
    )
    results = await provider.search("Apollo 11", count=5)
    assert [result.url for result in results] == [
        "https://www.nasa.gov/mission/apollo-11/",
        "https://www.britannica.com/topic/Apollo-11",
    ]
    assert results[0].title == "Apollo 11 - NASA"
    assert results[0].snippet == "The primary objective was a crewed lunar landing."
    assert results[0].source == "yahoo_jina"
    await client.aclose()


@pytest.mark.asyncio
async def test_yahoo_adapter_unwraps_result_urls(tmp_path: Path) -> None:
    document = """\
<div id="web"><ol class="searchCenterMiddle">
  <li><div class="algo-sr">
    <div class="compTitle"><a href="https://r.search.yahoo.com/x/RU=https%3A%2F%2Fwww.nasa.gov%2Fmission%2Fapollo-11%2F/RK=2/x"><h3>Apollo 11 - NASA</h3></a></div>
    <div class="compText">The primary objective was a crewed lunar landing.</div>
  </div></li>
</ol></div>
"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["p"] == "Apollo 11"
        return httpx.Response(200, text=document, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = YahooSearchProvider(
        SearchConfig(
            provider="yahoo",
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
        ),
        client,
    )
    results = await provider.search("Apollo 11", count=5)
    assert results[0].url == "https://www.nasa.gov/mission/apollo-11/"
    assert results[0].title == "Apollo 11 - NASA"
    assert results[0].snippet == "The primary objective was a crewed lunar landing."
    assert results[0].source == "yahoo"
    await client.aclose()


@pytest.mark.asyncio
async def test_read_only_search_cache_fails_closed(tmp_path: Path) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"results": []}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SearXNGSearchProvider(
        SearchConfig(
            provider="searxng",
            searxng_base_url="https://search.test",
            cache_mode="read",
            cache_path=tmp_path / "cache.sqlite3",
        ),
        client,
    )
    from browsecomp250.search.base import SearchError

    with pytest.raises(SearchError, match="Read-only search cache miss"):
        await provider.search("uncached")
    assert called is False
    await client.aclose()


@pytest.mark.asyncio
async def test_google_chrome_search_many_uses_one_fanout_batch(tmp_path: Path) -> None:
    class FakeGoogleChrome(GoogleChromeSearchProvider):
        payloads: list[dict]

        def __init__(self, config):
            super().__init__(config)
            self.payloads = []

        async def _invoke_bridge(self, payload):
            self.payloads.append(payload)
            return {
                "ok": True,
                "request_tag": payload["request_tag"],
                "pid": 42,
                "self_activation_suppressed": True,
                "load_parallel": True,
                "closed_tabs": len(payload["entries"]),
                "results": [
                    {
                        "query": entry["query"],
                        "tag": entry["tag"],
                        "text": (
                            "Skip to main content\nSearch Results\n"
                            f"Result for {entry['query']} https://example.test › article\n"
                            "Useful independent snippet.\nFooter Links\n"
                        ),
                    }
                    for entry in payload["entries"]
                ],
            }

    provider = FakeGoogleChrome(
        SearchConfig(
            provider="google_chrome",
            google_chrome_host="mac.test",
            cache_mode="off",
            cache_path=tmp_path / "google.sqlite3",
        )
    )
    batches = await provider.search_many(["alpha", "beta", "gamma"], count=3)
    assert len(provider.payloads) == 1
    assert len(provider.payloads[0]["entries"]) == 3
    assert all(not isinstance(batch, Exception) for batch in batches)
    assert [batch[0].source for batch in batches if isinstance(batch, list)] == [
        "google_user_chrome",
        "google_user_chrome",
        "google_user_chrome",
    ]
    assert provider.last_batch_metadata["closed_tabs"] == 3
    await provider.close()


def test_hybrid_merge_interleaves_and_deduplicates() -> None:
    google = [
        SearchResult(title="Google A", url="https://example.test/a", source="google_user_chrome")
    ]
    brave = [
        SearchResult(title="Brave A", url="https://example.test/a", source="brave"),
        SearchResult(title="Brave B", url="https://example.test/b", source="brave"),
    ]
    merged = HybridSearchProvider._merge_or_error(google, brave, 10)
    assert isinstance(merged, list)
    assert [item.url for item in merged] == [
        "https://example.test/a",
        "https://example.test/b",
    ]
