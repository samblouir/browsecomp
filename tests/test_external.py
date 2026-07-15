import asyncio

import httpx
import pytest

from browsecomp250.config import ExternalModelConfig
from browsecomp250.external import ExternalModelBroker


@pytest.mark.asyncio
async def test_external_model_broker_runs_batch_concurrently() -> None:
    active = 0
    maximum_active = 0
    bodies: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        body = __import__("json").loads(request.content)
        bodies.append(body)
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return httpx.Response(
            200,
            json={
                "id": f"emr_{len(bodies)}",
                "provider": "mock",
                "model": "mock-external-model",
                "status": "succeeded",
                "result": {"content": f"answer: {body['messages'][-1]['content']}"},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    broker = ExternalModelBroker(
        ExternalModelConfig(
            enabled=True,
            api_url="https://broker.test/api/external-model-requests",
            default_provider="mock",
            allowed_providers=["mock"],
            max_batch_size=4,
            max_concurrency=4,
            max_retries=0,
        ),
        client=client,
    )
    results = await broker.ask_many(
        [{"query": f"question {index}"} for index in range(4)],
        request_namespace="test:item:1",
    )
    assert maximum_active == 4
    assert all(result["ok"] for result in results)
    assert all(body["wait"] is True for body in bodies)
    assert all(body["max_tokens"] == 16384 for body in bodies)
    assert all(body["temperature"] == 0.7 for body in bodies)
    assert all(body["top_p"] == 0.95 for body in bodies)
    await client.aclose()


@pytest.mark.asyncio
async def test_external_model_broker_enforces_allowlist() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None))
    broker = ExternalModelBroker(
        ExternalModelConfig(
            enabled=True,
            api_url="https://broker.test/api/external-model-requests",
            allowed_providers=["chatgpt"],
        ),
        client=client,
    )
    result = await broker.ask_many(
        [{"query": "q", "provider": "gemini"}],
        request_namespace="test:item:2",
    )
    assert result[0]["ok"] is False
    assert "allowlisted" in result[0]["error"]
    await client.aclose()
