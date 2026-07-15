from __future__ import annotations

import httpx

from ..config import SearchConfig
from .base import SearchProvider
from .brave import BraveSearchProvider
from .google_chrome import GoogleChromeSearchProvider
from .hybrid import HybridSearchProvider
from .searxng import SearXNGSearchProvider
from .serper import SerperSearchProvider
from .tavily import TavilySearchProvider
from .yahoo import YahooSearchProvider
from .yahoo_jina import YahooJinaSearchProvider


def create_search_provider(
    config: SearchConfig, client: httpx.AsyncClient | None = None
) -> SearchProvider:
    providers = {
        "brave": BraveSearchProvider,
        "google_chrome": GoogleChromeSearchProvider,
        "hybrid": HybridSearchProvider,
        "tavily": TavilySearchProvider,
        "serper": SerperSearchProvider,
        "searxng": SearXNGSearchProvider,
        "yahoo": YahooSearchProvider,
        "yahoo_jina": YahooJinaSearchProvider,
    }
    return providers[config.provider](config, client=client)


__all__ = ["SearchProvider", "create_search_provider"]
