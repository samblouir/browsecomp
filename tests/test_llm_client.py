from __future__ import annotations

import json

import httpx
import pytest

from browsecomp250.llm.client import ClientSettings, ModelAPIError, OpenAICompatibleClient


@pytest.mark.asyncio
async def test_openai_compatible_request_and_response() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["tenant"] = request.headers.get("x-tenant")
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-frlstate-test",
                "model": "served",
                "frontierrl_conversation_id": "frlconv-test",
                "metadata": {
                    "frontierrl_response_id": "chatcmpl-frlstate-test",
                    "frontierrl_conversation_id": "frlconv-test",
                },
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"action":"note","text":"x"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    raw_client = httpx.AsyncClient(transport=transport)
    client = OpenAICompatibleClient(
        ClientSettings(
            api_base="https://model.test/v1",
            api_key="secret",
            model="star",
            temperature=0.3,
            max_output_tokens=1024,
            timeout_seconds=10,
            max_retries=0,
            extra_headers={"X-Tenant": "frontierrl"},
            input_price_per_million=1.0,
            output_price_per_million=2.0,
        ),
        raw_client,
    )
    response = await client.chat([{"role": "user", "content": "hello"}])
    assert captured["authorization"] == "Bearer secret"
    assert captured["tenant"] == "frontierrl"
    assert '"model":"star"' in captured["body"].replace(" ", "")
    assert '"stream":false' in captured["body"].replace(" ", "")
    assert response.response_model == "served"
    assert response.response_id == "chatcmpl-frlstate-test"
    assert response.conversation_id == "frlconv-test"
    assert response.usage.cached_tokens == 40
    assert response.usage.cost_usd == pytest.approx(0.00014)
    await raw_client.aclose()


@pytest.mark.asyncio
async def test_max_completion_tokens_replaces_max_tokens() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {},
            },
            request=request,
        )

    raw_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(
        ClientSettings(
            api_base="https://model.test/v1",
            api_key="secret",
            model="reasoning-grader",
            temperature=None,
            max_output_tokens=16384,
            timeout_seconds=10,
            max_retries=0,
            extra_body={"max_completion_tokens": 16384},
        ),
        raw_client,
    )
    await client.chat([{"role": "user", "content": "grade"}])
    assert captured["max_completion_tokens"] == 16384
    assert "max_tokens" not in captured
    assert "temperature" not in captured
    await raw_client.aclose()


@pytest.mark.asyncio
async def test_nonretryable_api_error_includes_response_detail() -> None:
    raw_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                400,
                json={"error": {"message": "unsupported field"}},
                request=request,
            )
        )
    )
    client = OpenAICompatibleClient(
        ClientSettings(
            api_base="https://model.test/v1",
            api_key="secret",
            model="model",
            temperature=0.3,
            max_output_tokens=16384,
            timeout_seconds=10,
            max_retries=4,
        ),
        raw_client,
    )
    with pytest.raises(ModelAPIError, match="unsupported field"):
        await client.chat([{"role": "user", "content": "hello"}])
    await raw_client.aclose()
