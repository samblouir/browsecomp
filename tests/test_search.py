import asyncio
import json
from pathlib import Path

import httpx
import pytest

from browsecomp250.config import SearchConfig
from browsecomp250.search.base import SearchError
from browsecomp250.search.bing_ssh import BingSSHSearchProvider
from browsecomp250.search.bing_yahoo_ssh import BingYahooSSHSearchProvider
from browsecomp250.search.brave import BraveSearchProvider
from browsecomp250.search.brave_ssh import BraveSSHSearchProvider
from browsecomp250.search.google_chrome import GoogleChromeSearchProvider
from browsecomp250.search.hybrid import HybridSearchProvider
from browsecomp250.search.openrouter_exa import OpenRouterExaSearchProvider
from browsecomp250.search.searxng import SearXNGSearchProvider
from browsecomp250.search.tavily import TavilySearchProvider
from browsecomp250.search.yahoo import YahooSearchProvider
from browsecomp250.search.yahoo_jina import YahooJinaSearchProvider
from browsecomp250.search.yahoo_ssh import YahooSSHSearchProvider
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
async def test_live_probe_bypasses_a_warm_search_cache(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Live result",
                            "url": "https://example.test/live",
                            "description": "Transport reached.",
                        }
                    ]
                }
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = BraveSearchProvider(
        SearchConfig(
            provider="brave",
            brave_api_key="secret",
            cache_mode="readwrite",
            cache_path=tmp_path / "cache.sqlite3",
        ),
        client,
    )
    query = "OpenAI official website"
    provider.cache.put(
        provider._cache_request(query, 10, 0),
        [
            {
                "title": "Cached result",
                "url": "https://example.test/cached",
                "snippet": "Old cache entry.",
                "rank": 1,
                "source": "brave",
                "extra_snippets": [],
            }
        ],
    )

    assert (await provider.search(query))[0].title == "Cached result"
    assert requests == 0
    assert (await provider.probe_live(query, count=10))[0].title == "Live result"
    assert requests == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_permanent_search_http_error_does_not_consume_retries(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(401, json={"error": "invalid key"}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = BraveSearchProvider(
        SearchConfig(
            provider="brave",
            brave_api_key="bad-key",
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
            max_retries=4,
        ),
        client,
    )

    with pytest.raises(SearchError, match=r"failed after 1 attempt\(s\)"):
        await provider.search("credential preflight")
    assert requests == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_exa_uses_only_standardized_url_citations(tmp_path: Path) -> None:
    long_second_snippet = "Second cited passage. " + "detail " * 100 + "Tail marker."

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer or-key"
        payload = json.loads(request.content)
        assert payload["model"] == "openai/gpt-4.1-nano"
        assert payload["plugins"] == [{"id": "web", "engine": "exa", "max_results": 2}]
        assert payload["temperature"] == 0.3
        assert payload["top_p"] == 0.95
        assert payload["max_tokens"] == 16384
        assert payload["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "This carrier answer must never become search evidence.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url_citation": {
                                        "url": "https://example.test/a",
                                        "title": "First source",
                                        "content": "First cited passage.",
                                    },
                                },
                                {
                                    "type": "url_citation",
                                    "url_citation": {
                                        "url": "https://example.test/b\\",
                                        "title": "Second source",
                                        "content": long_second_snippet,
                                    },
                                },
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 123,
                    "completion_tokens": 2,
                    "total_tokens": 125,
                    "cost": 0.0052,
                },
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenRouterExaSearchProvider(
        SearchConfig(
            provider="openrouter_exa",
            openrouter_api_key="or-key",
            openrouter_search_max_snippet_chars=256,
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
        ),
        client,
    )
    results = await provider.search("citation test", count=2)
    assert (results[0].title, results[0].url, results[0].snippet, results[0].source) == (
        "First source",
        "https://example.test/a",
        "First cited passage.",
        "openrouter_exa",
    )
    assert (results[1].title, results[1].url, results[1].source) == (
        "Second source",
        "https://example.test/b",
        "openrouter_exa",
    )
    assert len(results[1].snippet) <= 256
    assert results[1].snippet.startswith("Second cited passage.")
    assert results[1].snippet.endswith("Tail marker.")
    assert "chars omitted" in results[1].snippet
    assert all("carrier answer" not in result.snippet for result in results)
    assert provider.audit_metrics() == {
        "carrier_model": "openai/gpt-4.1-nano",
        "engine": "exa",
        "requests": 1,
        "input_tokens": 123,
        "output_tokens": 2,
        "total_tokens": 125,
        "results": 2,
        "filtered_query_mirrors": 0,
        "truncated_snippets": 1,
        "snippet_chars_removed": len(long_second_snippet) - len(results[1].snippet),
        "cost_usd": 0.0052,
    }
    await client.aclose()


def test_openrouter_exa_detects_repetitive_query_mirror_but_not_real_article() -> None:
    query = '"made an album for fun" 2022 article England Roman numerals albums'
    mirrored_title = "Made an Album for Fun 2022 Article England Roman Numerals Albums"
    mirrored_url = (
        "https://spam.test/video/made-an-album-for-fun-2022-article-england-roman-numerals-albums/"
    )
    repeated = " ".join([query] * 3)
    assert OpenRouterExaSearchProvider._looks_like_query_mirror(
        query,
        title=mirrored_title,
        url=mirrored_url,
        snippet=repeated,
    )
    assert OpenRouterExaSearchProvider._looks_like_query_mirror(
        query,
        title=mirrored_title,
        url=mirrored_url,
        snippet="# Made an album for fun 2022 article England Roman numerals albums",
    )
    assert not OpenRouterExaSearchProvider._looks_like_query_mirror(
        query,
        title=mirrored_title,
        url=mirrored_url,
        snippet=(
            "The musician described the recording process, collaborators, release history, "
            "and artistic goals in a separately authored interview."
        ),
    )


def test_openrouter_exa_detects_short_empty_snippet_keyword_stuffing() -> None:
    query = '"Roman numerals" albums 2001..2007'
    assert OpenRouterExaSearchProvider._looks_like_query_mirror(
        query,
        title=(
            "English Artists Albums Named With Roman Numerals II III 2001 2007 "
            "Discography Album Name"
        ),
        url=(
            "https://spam.test/video/english-artists-albums-named-with-roman-numerals-"
            "ii-iii-2001-2007-discography-album-name/"
        ),
        snippet="",
    )
    assert not OpenRouterExaSearchProvider._looks_like_query_mirror(
        query,
        title="Roman Numerals - The Roman Numerals",
        url="https://music.test/album/roman-numerals-2006",
        snippet="A review of the band's 2006 record and its release history.",
    )


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
async def test_yahoo_jina_cleans_control_tokens_and_does_not_retry_empty_results(
    tmp_path: Path,
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.params["p"] == "exact phrase"
        return httpx.Response(200, text="## Search Results\n", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = YahooJinaSearchProvider(
        SearchConfig(
            provider="yahoo_jina",
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
            max_retries=4,
        ),
        client,
    )

    results = await provider.search("exact <|channel|> phrase", count=10)

    assert results == []
    assert calls == 1
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
async def test_yahoo_empty_results_do_not_retry(tmp_path: Path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text="<html><body>No results</body></html>", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = YahooSearchProvider(
        SearchConfig(
            provider="yahoo",
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
            max_retries=4,
            yahoo_min_interval_seconds=0,
        ),
        client,
    )

    results = await provider.search("an exact query with no matches", count=10)

    assert results == []
    assert calls == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_yahoo_adapter_bounds_concurrency_and_cleans_control_tokens(
    tmp_path: Path,
) -> None:
    active = 0
    maximum_active = 0
    queries: list[str] = []
    document = """\
<div id="web"><ol class="searchCenterMiddle">
  <li><div class="algo-sr">
    <div class="compTitle"><a href="https://example.test/result"><h3>Result</h3></a></div>
    <div class="compText">Evidence.</div>
  </div></li>
</ol></div>
"""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        queries.append(request.url.params["p"])
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, text=document, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = YahooSearchProvider(
        SearchConfig(
            provider="yahoo",
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
            yahoo_max_concurrency=2,
            yahoo_min_interval_seconds=0,
        ),
        client,
    )
    batches = await provider.search_many(
        [f"query {index} <|channel|>" for index in range(6)],
        count=1,
    )
    assert all(not isinstance(batch, Exception) and batch for batch in batches)
    assert maximum_active == 2
    assert all("<|" not in query for query in queries)
    await client.aclose()


@pytest.mark.asyncio
async def test_yahoo_ssh_adapter_uses_bounded_remote_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = b"""\
<div id="web"><ol class="searchCenterMiddle">
  <li><div class="algo-sr">
    <div class="compTitle"><a href="https://example.test/result"><h3>Result</h3></a></div>
    <div class="compText">Evidence.</div>
  </div></li>
</ol></div>
"""
    captured: tuple[object, ...] = ()

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return document, b""

    async def create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        nonlocal captured
        captured = args
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    provider = YahooSSHSearchProvider(
        SearchConfig(
            provider="yahoo_ssh",
            yahoo_ssh_host="remote.test",
            yahoo_min_interval_seconds=0,
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
        )
    )

    results = await provider.search("exact <|channel|> phrase; touch /tmp/no", count=1)

    assert results[0].url == "https://example.test/result"
    assert "remote.test" in captured
    remote_command = str(captured[-1])
    assert "<|channel|>" not in remote_command
    assert "phrase%3B+touch+%2Ftmp%2Fno" in remote_command
    assert "'Accept: text/html,application/xhtml+xml'" in remote_command
    await provider.close()


@pytest.mark.asyncio
async def test_bing_ssh_adapter_uses_bounded_escaped_rss_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = b"""\
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
  <item><title>First result</title><link>https://example.test/first</link>
    <description>Useful &amp;lt;b&amp;gt;evidence&amp;lt;/b&amp;gt;.</description></item>
</channel></rss>
"""
    captured: tuple[object, ...] = ()

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return document, b""

    async def create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        nonlocal captured
        captured = args
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    provider = BingSSHSearchProvider(
        SearchConfig(
            provider="bing_ssh",
            bing_ssh_host="remote.test",
            bing_ssh_min_interval_seconds=0,
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
        )
    )

    results = await provider.search("exact <|channel|> phrase; touch /tmp/no", count=3)

    assert results[0].url == "https://example.test/first"
    assert results[0].snippet == "Useful evidence."
    assert results[0].source == "bing_ssh"
    assert "remote.test" in captured
    remote_command = str(captured[-1])
    assert "format=rss" in remote_command
    assert "<|channel|>" not in remote_command
    assert "phrase%3B+touch+%2Ftmp%2Fno" in remote_command
    assert "'Accept: application/rss+xml,application/xml,text/xml'" in remote_command
    await provider.close()


@pytest.mark.parametrize(
    "document,match",
    [
        ("<html>challenge</html>", "unexpected RSS root"),
        ("<!DOCTYPE rss><rss><channel /></rss>", "forbidden doctype"),
        ("<rss><channel /></rss>", "no organic results"),
    ],
)
def test_bing_ssh_rejects_non_result_feeds(document: str, match: str) -> None:
    with pytest.raises(SearchError, match=match):
        BingSSHSearchProvider._parse_rss(document, count=10, offset=0)


def test_bing_ssh_parser_honors_result_count_and_offset() -> None:
    items = "".join(
        f"<item><title>Result {index}</title><link>https://example.test/{index}</link>"
        f"<description>Evidence {index}</description></item>"
        for index in range(6)
    )
    results = BingSSHSearchProvider._parse_rss(
        f"<rss><channel>{items}</channel></rss>",
        count=2,
        offset=3,
    )

    assert [item.url for item in results] == [
        "https://example.test/0",
        "https://example.test/1",
    ]
    assert [item.rank for item in results] == [7, 8]


@pytest.mark.asyncio
async def test_brave_ssh_adapter_uses_bounded_escaped_html_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = b"""\
<html><head><title>Search</title></head><body>
<div class="result-wrapper"><a href="https://example.test/result">
  <div class="title search-snippet-title">Useful result</div></a>
  <div class="generic-snippet"><div class="content">Direct evidence.</div></div>
</div></body></html>
"""
    captured: tuple[object, ...] = ()

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return document, b""

    async def create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        nonlocal captured
        captured = args
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    provider = BraveSSHSearchProvider(
        SearchConfig(
            provider="brave_ssh",
            brave_ssh_host="remote.test",
            brave_ssh_min_interval_seconds=0,
            cache_mode="off",
            cache_path=tmp_path / "cache.sqlite3",
        )
    )

    results = await provider.search("exact <|channel|> phrase; touch /tmp/no", count=3)

    assert results[0].url == "https://example.test/result"
    assert results[0].title == "Useful result"
    assert results[0].snippet == "Direct evidence."
    assert results[0].source == "brave_ssh"
    assert "remote.test" in captured
    remote_command = str(captured[-1])
    assert "<|channel|>" not in remote_command
    assert "phrase%3B+touch+%2Ftmp%2Fno" in remote_command
    assert "'Accept: text/html,application/xhtml+xml'" in remote_command
    await provider.close()


def test_brave_ssh_parser_rejects_challenge_and_query_mirrors() -> None:
    with pytest.raises(SearchError, match="no organic result wrappers"):
        BraveSSHSearchProvider._parse_html(
            "<html><title>Verify you are human</title></html>",
            query="useful evidence",
            count=10,
            offset=0,
        )

    document = """
    <div class="result-wrapper"><a href="https://www.instagram.com/popular/produced-a-play-between-1970-and-1975-prosecuted-prison/">
      <div class="title">Query mirror</div></a></div>
    <div class="result-wrapper"><a href="https://example.test/history">
      <div class="title">Independent history</div></a>
      <div class="generic-snippet"><div class="content">Evidence.</div></div></div>
    """
    results = BraveSSHSearchProvider._parse_html(
        document,
        query="produced a play between 1970 and 1975 prosecuted prison",
        count=10,
        offset=0,
    )
    assert [item.url for item in results] == ["https://example.test/history"]


def test_brave_ssh_parser_filters_social_query_pages_and_benchmark_poisoning() -> None:
    document = """
    <div class="result-wrapper"><a href="https://www.instagram.com/popular/nearby-clue-page/">
      <div class="title">Nearby clue page</div></a></div>
    <div class="result-wrapper"><a href="https://openreview.net/pdf/browsecomp-plus.pdf">
      <div class="title">BrowseComp-Plus benchmark answers</div></a></div>
    <div class="result-wrapper"><a href="https://example.test/source">
      <div class="title">Independent source</div></a></div>
    """
    results = BraveSSHSearchProvider._parse_html(
        document,
        query="historical publication culinary innovations",
        count=10,
        offset=0,
    )
    assert [item.url for item in results] == ["https://example.test/source"]


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


def test_bing_yahoo_merge_interleaves_deduplicates_and_preserves_source() -> None:
    bing = [
        SearchResult(title="Bing A", url="https://www.example.test/a/", source="bing_ssh"),
        SearchResult(title="Bing B", url="https://example.test/b", source="bing_ssh"),
    ]
    yahoo = [
        SearchResult(title="Yahoo A", url="https://example.test/a", source="yahoo_ssh"),
        SearchResult(title="Yahoo C", url="https://example.test/c", source="yahoo_ssh"),
    ]

    merged = BingYahooSSHSearchProvider._merge_or_error(bing, yahoo, 10)

    assert isinstance(merged, list)
    assert [(item.url, item.source, item.rank) for item in merged] == [
        ("https://www.example.test/a/", "bing_ssh", 1),
        ("https://example.test/b", "bing_ssh", 2),
        ("https://example.test/c", "yahoo_ssh", 3),
    ]


def test_bing_yahoo_merge_uses_one_healthy_engine() -> None:
    yahoo = [SearchResult(title="Yahoo", url="https://example.test/y", source="yahoo_ssh")]

    merged = BingYahooSSHSearchProvider._merge_or_error(SearchError("bing unavailable"), yahoo, 10)

    assert isinstance(merged, list)
    assert [item.url for item in merged] == ["https://example.test/y"]


def test_bing_yahoo_merge_reports_both_engine_failures() -> None:
    merged = BingYahooSSHSearchProvider._merge_or_error(
        SearchError("bing unavailable"), SearchError("yahoo unavailable"), 10
    )

    assert isinstance(merged, SearchError)
    assert "Both server-side search engines failed" in str(merged)


@pytest.mark.asyncio
async def test_bing_yahoo_disables_engine_that_fails_live_preflight(tmp_path: Path) -> None:
    class FakeEngine:
        def __init__(self, result):
            self.result = result
            self.search_many_calls = 0

        async def probe_live(self, query: str, count: int = 1):
            del query, count
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

        async def search_many(self, queries, count=None, offset=0):
            del count, offset
            self.search_many_calls += 1
            return [self.result for _ in queries]

        async def close(self):
            return None

    provider = BingYahooSSHSearchProvider(
        SearchConfig(
            provider="bing_yahoo_ssh",
            cache_mode="off",
            cache_path=tmp_path / "meta.sqlite3",
        )
    )
    healthy = FakeEngine(
        [SearchResult(title="Bing", url="https://example.test/b", source="bing_ssh")]
    )
    failed = FakeEngine(SearchError("HTTP/2 framing failure"))
    provider.bing = healthy
    provider.yahoo = failed

    rows = await provider.probe_live("test", count=10)
    batches = await provider.search_many(["one", "two"], count=10)

    assert [row.url for row in rows] == ["https://example.test/b"]
    assert healthy.search_many_calls == 1
    assert failed.search_many_calls == 0
    assert all(isinstance(batch, list) and batch[0].source == "bing_ssh" for batch in batches)
    assert provider.audit_metrics()["engines"]["yahoo"]["enabled"] is False
    await provider.close()
