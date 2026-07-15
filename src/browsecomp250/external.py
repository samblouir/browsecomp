from __future__ import annotations

import asyncio
from typing import Any

import httpx

from .config import ExternalModelConfig


class ExternalModelError(RuntimeError):
    pass


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text", item.get("content"))
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


class ExternalModelBroker:
    """Client for the production external-model request broker."""

    def __init__(
        self,
        config: ExternalModelConfig,
        client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=config.timeout_seconds)
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def ask_many(
        self,
        requests: list[dict[str, Any]],
        *,
        request_namespace: str,
    ) -> list[dict[str, Any]]:
        if not self.config.enabled:
            raise ExternalModelError("External-model consultation is disabled")
        bounded = requests[: self.config.max_batch_size]
        return await asyncio.gather(
            *(
                self._ask_one(request, request_namespace=request_namespace, call_index=index)
                for index, request in enumerate(bounded, start=1)
            )
        )

    async def _ask_one(
        self,
        request: dict[str, Any],
        *,
        request_namespace: str,
        call_index: int,
    ) -> dict[str, Any]:
        query = str(request.get("query") or "").strip()
        if not query:
            return {"ok": False, "status": "failed", "error": "query is required"}
        provider = str(request.get("provider") or self.config.default_provider).strip().lower()
        if provider not in set(self.config.allowed_providers):
            return {
                "ok": False,
                "status": "failed",
                "error": f"provider is not allowlisted: {provider}",
            }
        model = str(request.get("model") or self.config.default_model).strip()
        system = str(request.get("system") or "").strip()
        context = str(request.get("context") or "").strip()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        content = query if not context else f"Context:\n{context}\n\nQuestion:\n{query}"
        messages.append({"role": "user", "content": content})
        max_tokens = max(
            16384,
            min(32768, int(request.get("max_tokens") or self.config.max_output_tokens)),
        )
        temperature = max(
            0.3,
            min(1.0, float(request.get("temperature", self.config.temperature))),
        )
        top_p = max(0.01, min(1.0, float(request.get("top_p", self.config.top_p))))
        body: dict[str, Any] = {
            "provider": provider,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "wait": True,
            "metadata": {
                "source": "browsecomp250_ask_external_model",
                "request_namespace": request_namespace,
                "call_index": call_index,
            },
        }
        if model:
            body["model"] = model
        headers: dict[str, str] = {}
        if self.config.admin_token:
            headers["x-frontierrl-admin-token"] = self.config.admin_token

        last_error: Exception | None = None
        async with self._semaphore:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await self.client.post(
                        self.config.api_url,
                        json=body,
                        headers=headers,
                    )
                    response.raise_for_status()
                    completed = response.json()
                    if not isinstance(completed, dict):
                        raise ExternalModelError("broker returned a non-object")
                    result = completed.get("result")
                    result = result if isinstance(result, dict) else {}
                    status = str(completed.get("status") or "failed")
                    output: dict[str, Any] = {
                        "ok": status == "succeeded",
                        "status": status,
                        "request_id": str(completed.get("id") or ""),
                        "provider": str(completed.get("provider") or provider),
                        "model": str(completed.get("model") or model),
                        "content": _flatten_content(result.get("content")),
                    }
                    if isinstance(result.get("usage"), dict):
                        output["usage"] = result["usage"]
                    if completed.get("error"):
                        output["error"] = str(completed["error"])
                    return output
                except (httpx.HTTPError, ValueError, ExternalModelError) as exc:
                    last_error = exc
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(min(2**attempt, 10))
        return {
            "ok": False,
            "status": "failed",
            "provider": provider,
            "model": model,
            "error": str(last_error),
        }


__all__ = ["ExternalModelBroker", "ExternalModelError"]
