from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..types import ModelResponse, Usage


class ModelAPIError(RuntimeError):
    pass


@dataclass(slots=True)
class ClientSettings:
    api_base: str
    api_key: str
    model: str
    temperature: float | None
    max_output_tokens: int
    timeout_seconds: float
    max_retries: int
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0


class OpenAICompatibleClient:
    def __init__(
        self,
        settings: ClientSettings,
        client: httpx.AsyncClient | None = None,
    ):
        self.settings = settings
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=settings.timeout_seconds)

    @property
    def chat_completions_url(self) -> str:
        base = self.settings.api_base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    @property
    def models_url(self) -> str:
        base = self.settings.api_base.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        return base + "/models"

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.settings.extra_headers}
        if self.settings.api_key:
            headers.setdefault("Authorization", f"Bearer {self.settings.api_key}")
        return headers

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def list_models(self) -> dict[str, Any]:
        response = await self.client.get(self.models_url, headers=self.headers())
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ModelAPIError("/models response was not an object")
        return data

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": self.settings.max_output_tokens,
            "stream": False,
            **self.settings.extra_body,
            **(extra_body or {}),
        }
        if self.settings.temperature is not None:
            body["temperature"] = self.settings.temperature
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if "max_completion_tokens" in body:
            body.pop("max_tokens", None)

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            started = time.perf_counter()
            try:
                async with asyncio.timeout(self.settings.timeout_seconds):
                    response = await self.client.post(
                        self.chat_completions_url,
                        headers=self.headers(),
                        json=body,
                        timeout=self.settings.timeout_seconds,
                    )
                if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                data = response.json()
                return self._parse_response(data, time.perf_counter() - started)
            except (
                TimeoutError,
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
            ) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code in {408, 409, 425, 429}
                    or exc.response.status_code >= 500
                )
                if not retryable or attempt >= self.settings.max_retries:
                    break
                retry_after = 0.0
                if isinstance(exc, httpx.HTTPStatusError):
                    try:
                        retry_after = float(exc.response.headers.get("retry-after", "0"))
                    except ValueError:
                        retry_after = 0.0
                delay = max(retry_after, min(2**attempt + random.random(), 20))
                await asyncio.sleep(delay)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise ModelAPIError(f"Malformed chat completion response: {exc}") from exc
        detail = ""
        if isinstance(last_error, httpx.HTTPStatusError):
            response_text = last_error.response.text.strip()
            if response_text:
                detail = f"; response={response_text[:2000]}"
        error_name = type(last_error).__name__ if last_error is not None else "UnknownError"
        raise ModelAPIError(
            f"Chat completion failed after retries: {error_name}: {last_error}{detail}"
        )

    def _parse_response(self, data: dict[str, Any], latency: float) -> ModelResponse:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelAPIError("Response has no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if content is None:
            content = ""
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        usage_data = data.get("usage") or {}
        prompt_tokens = int(usage_data.get("prompt_tokens") or usage_data.get("input_tokens") or 0)
        completion_tokens = int(
            usage_data.get("completion_tokens") or usage_data.get("output_tokens") or 0
        )
        cached_tokens = 0
        details = usage_data.get("prompt_tokens_details") or usage_data.get("input_tokens_details")
        if isinstance(details, dict):
            cached_tokens = int(details.get("cached_tokens") or 0)
        cost = (
            prompt_tokens * self.settings.input_price_per_million
            + completion_tokens * self.settings.output_price_per_million
        ) / 1_000_000
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        response_id = str(metadata.get("frontierrl_response_id") or data.get("id") or "").strip()
        conversation_id = str(
            metadata.get("frontierrl_conversation_id")
            or data.get("frontierrl_conversation_id")
            or ""
        ).strip()
        return ModelResponse(
            content=str(content),
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                cost_usd=cost,
            ),
            raw_message=dict(message),
            finish_reason=choice.get("finish_reason"),
            response_model=data.get("model"),
            latency_seconds=latency,
            response_id=response_id or None,
            conversation_id=conversation_id or None,
            metadata=dict(metadata),
        )


def settings_from_model_config(config: Any) -> ClientSettings:
    return ClientSettings(
        api_base=config.api_base,
        api_key=config.api_key,
        model=config.model,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        extra_headers=dict(getattr(config, "extra_headers_json", {}) or {}),
        extra_body=dict(config.extra_body),
        input_price_per_million=float(getattr(config, "input_price_per_million", 0.0)),
        output_price_per_million=float(getattr(config, "output_price_per_million", 0.0)),
    )
