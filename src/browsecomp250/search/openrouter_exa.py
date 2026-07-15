from __future__ import annotations

import asyncio
from typing import Any

from ..types import SearchResult
from .base import SearchError, SearchProvider


class OpenRouterExaSearchProvider(SearchProvider):
    """Use OpenRouter's Exa web plugin as a cited search transport.

    The carrier model's generated text is deliberately discarded. Only the
    standardized URL-citation annotations become benchmark search evidence.
    """

    name = "openrouter_exa"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_semaphore = asyncio.Semaphore(
            self.config.openrouter_search_max_concurrency
        )
        self._metrics: dict[str, int | float | str] = {
            "carrier_model": self.config.openrouter_search_model,
            "engine": "exa",
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "results": 0,
            "cost_usd": 0.0,
        }

    def audit_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        if offset:
            raise SearchError("OpenRouter Exa does not expose deterministic result pagination")

        max_results = min(count, self.config.openrouter_search_max_results)
        payload = {
            "model": self.config.openrouter_search_model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Search the web for: {query}\nReturn only: OK",
                }
            ],
            "plugins": [{"id": "web", "engine": "exa", "max_results": max_results}],
            "temperature": self.config.openrouter_search_temperature,
            "top_p": self.config.openrouter_search_top_p,
            "max_tokens": self.config.openrouter_search_max_output_tokens,
            "stream": False,
        }
        endpoint = f"{self.config.openrouter_api_base.rstrip('/')}/chat/completions"
        async with self._request_semaphore:
            self._metrics["requests"] = int(self._metrics["requests"]) + 1
            response = await self.client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.config.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        response.raise_for_status()
        body = response.json()
        self._record_usage(body.get("usage"))

        choices = body.get("choices") or []
        message = choices[0].get("message") if choices else None
        annotations = message.get("annotations") if isinstance(message, dict) else None
        if not isinstance(annotations, list):
            raise SearchError("OpenRouter Exa response contained no URL-citation annotations")

        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            citation = annotation.get("url_citation")
            if not isinstance(citation, dict):
                continue
            url = str(citation.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                SearchResult(
                    title=str(citation.get("title") or url).strip(),
                    url=url,
                    snippet=str(citation.get("content") or "").strip(),
                    rank=len(results) + 1,
                    source=self.name,
                )
            )
            if len(results) >= max_results:
                break

        if not results:
            raise SearchError("OpenRouter Exa response contained no usable URL citations")
        self._metrics["results"] = int(self._metrics["results"]) + len(results)
        return results

    def _record_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        self._metrics["input_tokens"] = int(self._metrics["input_tokens"]) + prompt_tokens
        self._metrics["output_tokens"] = (
            int(self._metrics["output_tokens"]) + completion_tokens
        )
        self._metrics["total_tokens"] = int(self._metrics["total_tokens"]) + total_tokens
        self._metrics["cost_usd"] = round(
            float(self._metrics["cost_usd"]) + float(usage.get("cost") or 0.0),
            10,
        )
