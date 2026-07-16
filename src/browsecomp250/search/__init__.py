from __future__ import annotations

import httpx

from ..config import SearchConfig
from .base import SearchProvider
from .bing_ssh import BingSSHSearchProvider
from .bing_yahoo_ssh import BingYahooSSHSearchProvider
from .brave import BraveSearchProvider
from .brave_ssh import BraveSSHSearchProvider
from .google_chrome import GoogleChromeSearchProvider
from .hybrid import HybridSearchProvider
from .openrouter_exa import OpenRouterExaSearchProvider
from .searxng import SearXNGSearchProvider
from .serper import SerperSearchProvider
from .tavily import TavilySearchProvider
from .yahoo import YahooSearchProvider
from .yahoo_jina import YahooJinaSearchProvider
from .yahoo_ssh import YahooSSHSearchProvider


def create_search_provider(
    config: SearchConfig, client: httpx.AsyncClient | None = None
) -> SearchProvider:
    providers = {
        "bing_ssh": BingSSHSearchProvider,
        "bing_yahoo_ssh": BingYahooSSHSearchProvider,
        "brave": BraveSearchProvider,
        "brave_ssh": BraveSSHSearchProvider,
        "google_chrome": GoogleChromeSearchProvider,
        "hybrid": HybridSearchProvider,
        "openrouter_exa": OpenRouterExaSearchProvider,
        "tavily": TavilySearchProvider,
        "serper": SerperSearchProvider,
        "searxng": SearXNGSearchProvider,
        "yahoo": YahooSearchProvider,
        "yahoo_jina": YahooJinaSearchProvider,
        "yahoo_ssh": YahooSSHSearchProvider,
    }
    return providers[config.provider](config, client=client)


__all__ = ["SearchProvider", "create_search_provider"]
