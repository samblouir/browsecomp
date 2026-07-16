from __future__ import annotations

import asyncio
from typing import Any

import pytest

from browsecomp250.agent_external import AgentExternalModelBroker
from browsecomp250.config import AgentConfig, BrowserConfig, ExternalModelConfig
from browsecomp250.types import AgentOutcome, Usage


class _FakeModelClient:
    async def close(self) -> None:
        return None


class _FakeRunner:
    configurations: list[dict[str, Any]] = []
    active = 0
    maximum_active = 0

    def __init__(self, model_config, agent_config, browser_config, search, browser, **kwargs):
        self.__class__.configurations.append(
            {
                "model": model_config,
                "agent": agent_config,
                "browser": browser_config,
                "search": search,
                "fetcher": browser,
                "kwargs": kwargs,
            }
        )
        self.event_sink = kwargs["event_sink"]

    async def run(self, question: str, *, request_namespace: str) -> AgentOutcome:
        type(self).active += 1
        type(self).maximum_active = max(type(self).maximum_active, type(self).active)
        self.event_sink({"event": "turn_started", "step": 1})
        await asyncio.sleep(0.01)
        type(self).active -= 1
        return AgentOutcome(
            response_text="Explanation: researched result\nExact Answer: Ada\nConfidence: 80%",
            exact_answer="Ada",
            explanation='{"analysis":"researched","queries":["entity history"]}',
            confidence=80,
            citations=["https://source.test/evidence"],
            status="completed",
            steps=2,
            search_calls=1,
            page_opens=1,
            find_calls=0,
            retrieved_chars=100,
            duration_seconds=0.1,
            usage=Usage(input_tokens=20, output_tokens=10),
            transcript=[
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-search",
                            "type": "function",
                            "function": {
                                "name": "search_many",
                                "arguments": '{"queries":["entity history","candidate archive"]}',
                            },
                        }
                    ],
                },
            ],
        )


class _NoCitationRunner(_FakeRunner):
    async def run(self, question: str, *, request_namespace: str) -> AgentOutcome:
        outcome = await super().run(question, request_namespace=request_namespace)
        outcome.citations = []
        return outcome


@pytest.mark.asyncio
async def test_agent_external_broker_forces_star2_tools_and_runs_concurrently() -> None:
    _FakeRunner.configurations = []
    _FakeRunner.active = 0
    _FakeRunner.maximum_active = 0
    broker = AgentExternalModelBroker(
        ExternalModelConfig(
            enabled=True,
            mode="agent",
            agent_api_base="https://agent.test/agent/v1",
            agent_api_key="real-key",
            agent_model="frontierrl/star-2",
            agent_routing_backend_pool=["star2-a", "star2-b"],
            agent_max_steps=9,
            agent_max_search_calls=18,
            agent_max_page_opens=20,
            agent_max_find_calls=11,
            agent_max_retrieved_chars=240_000,
            agent_max_history_chars=120_000,
            agent_force_final_after_seconds=180,
            max_batch_size=4,
            max_concurrency=4,
        ),
        AgentConfig(),
        BrowserConfig(),
        search_provider=object(),
        page_fetcher=object(),
        model_client=_FakeModelClient(),
        runner_factory=_FakeRunner,
    )
    results = await broker.ask_many(
        [
            {"query": "Return research as JSON", "model": "ignored-model"},
            {"query": "Audit the evidence", "provider": "ignored-provider"},
        ],
        request_namespace="test:item",
    )
    assert _FakeRunner.maximum_active == 2
    assert all(result["ok"] for result in results)
    assert all(result["model"] == "frontierrl/star-2" for result in results)
    assert all(result["exact_answer"] == "Ada" for result in results)
    assert all(result["confidence"] == 80 for result in results)
    assert all(result["citations"] == ["https://source.test/evidence"] for result in results)
    assert results[0]["content"].startswith('{"analysis"')
    assert results[0]["agent_search_queries"] == ["entity history", "candidate archive"]
    config = _FakeRunner.configurations[0]
    assert config["model"].protocol == "tools"
    assert config["model"].temperature == 0.7
    assert config["model"].max_output_tokens == 16384
    assert config["model"].extra_body["top_p"] == 0.95
    assert config["model"].extra_body["vllm_xargs"] == {"frontierrl_max_denoising_steps": 48}
    assert config["model"].routing_backend_pool == ["star2-a", "star2-b"]
    assert config["agent"].min_search_calls_before_final == 1
    assert config["agent"].max_steps == 9
    assert config["agent"].max_search_calls == 18
    assert config["agent"].max_page_opens == 20
    assert config["agent"].max_find_calls == 11
    assert config["agent"].max_retrieved_chars == 240_000
    assert config["agent"].max_history_chars == 120_000
    assert config["agent"].force_final_after_seconds == 180
    assert config["kwargs"]["external_model_broker"] is None
    assert config["kwargs"]["external_model_config"].enabled is False


def test_agent_external_broker_rejects_non_star2_helper() -> None:
    with pytest.raises(ValueError, match="pinned to frontierrl/star-2"):
        AgentExternalModelBroker(
            ExternalModelConfig(
                enabled=True,
                mode="agent",
                agent_api_key="real-key",
                agent_model="gpt-5.6",
            ),
            AgentConfig(),
            BrowserConfig(),
            search_provider=object(),
            page_fetcher=object(),
            model_client=_FakeModelClient(),
            runner_factory=_FakeRunner,
        )


@pytest.mark.asyncio
async def test_agent_external_broker_normalizes_final_action_contract() -> None:
    broker = AgentExternalModelBroker(
        ExternalModelConfig(
            enabled=True,
            mode="agent",
            agent_api_key="real-key",
        ),
        AgentConfig(),
        BrowserConfig(),
        search_provider=object(),
        page_fetcher=object(),
        model_client=_FakeModelClient(),
        runner_factory=_FakeRunner,
    )
    result = await broker.ask_many(
        [{"query": 'Return {"action":"final","exact_answer":"short answer"}'}],
        request_namespace="test:final",
    )
    assert '"action":"final"' in result[0]["content"]
    assert '"exact_answer":"Ada"' in result[0]["content"]


@pytest.mark.asyncio
async def test_agent_external_broker_rejects_uncited_helper_final() -> None:
    broker = AgentExternalModelBroker(
        ExternalModelConfig(enabled=True, mode="agent", agent_api_key="real-key"),
        AgentConfig(require_citations=True),
        BrowserConfig(),
        search_provider=object(),
        page_fetcher=object(),
        model_client=_FakeModelClient(),
        runner_factory=_NoCitationRunner,
    )
    result = await broker.ask_many(
        [{"query": "Research and answer with sources"}],
        request_namespace="test:no-citation",
    )
    assert result[0]["ok"] is False
    assert result[0]["status"] == "failed"
    assert result[0]["error"] == "Star helper final answer omitted required citations"


@pytest.mark.asyncio
async def test_agent_external_strategy_skips_factual_answer_evidence_gates() -> None:
    _NoCitationRunner.configurations = []
    broker = AgentExternalModelBroker(
        ExternalModelConfig(enabled=True, mode="agent", agent_api_key="real-key"),
        AgentConfig(require_citations=True, require_opened_citation_support=True),
        BrowserConfig(),
        search_provider=object(),
        page_fetcher=object(),
        model_client=_FakeModelClient(),
        runner_factory=_NoCitationRunner,
    )

    result = await broker.ask_many(
        [{"task_mode": "strategy", "query": "Return a query-plan JSON object"}],
        request_namespace="test:strategy",
    )

    assert result[0]["ok"] is True
    config = _NoCitationRunner.configurations[-1]
    assert config["agent"].require_citations is False
    assert config["agent"].require_opened_citation_support is False
    assert config["agent"].min_search_calls_before_final == 0
    assert config["agent"].max_steps == 4
    assert "retrieval-strategy controller" in config["kwargs"]["system_prompt"]
    assert config["kwargs"]["initial_force_final"] is True


def test_agent_external_broker_extracts_executed_search_queries() -> None:
    transcript = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": {"query": "rare collaborator archive"},
                    }
                },
                {
                    "function": {
                        "name": "search_many",
                        "arguments": '{"queries":["candidate history","rare collaborator archive"]}',
                    }
                },
                {"function": {"name": "open", "arguments": '{"url":"https://x.test"}'}},
            ],
        }
    ]
    assert AgentExternalModelBroker._search_queries_from_transcript(transcript) == [
        "rare collaborator archive",
        "candidate history",
    ]


def test_agent_external_broker_salvages_partial_timeout_evidence() -> None:
    broker = AgentExternalModelBroker(
        ExternalModelConfig(enabled=True, mode="agent", agent_api_key="real-key"),
        AgentConfig(require_citations=True),
        BrowserConfig(),
        search_provider=object(),
        page_fetcher=object(),
        model_client=_FakeModelClient(),
        runner_factory=_FakeRunner,
    )
    result = broker._partial_timeout_result(
        namespace="test:timeout",
        error="timed out",
        events=[
            {
                "event": "model_response",
                "assistant_reasoning": "Candidate Alpha matches the rare clue.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_many",
                            "arguments": '{"queries":["Alpha archive","Alpha biography"]}',
                        }
                    }
                ],
            },
            {
                "event": "action_completed",
                "result": {
                    "pages": [
                        {"final_url": "https://source.test/alpha", "text": "Evidence"},
                        {"final_url": "file:///tmp/private", "text": "Ignore"},
                    ]
                },
            },
        ],
    )

    assert result["ok"] is False
    assert result["request_id"] == "test:timeout"
    assert "Candidate Alpha" in result["content"]
    assert result["agent_search_queries"] == ["Alpha archive", "Alpha biography"]
    assert result["citations"] == ["https://source.test/alpha"]
