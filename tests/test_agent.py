import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from browsecomp250.agent import AgentRunner
from browsecomp250.config import (
    AgentConfig,
    BrowserConfig,
    ExternalModelConfig,
    ModelConfig,
    SearchConfig,
)
from browsecomp250.types import AgentAction, ModelResponse, PageDocument, SearchResult, Usage


class FakeModel:
    def __init__(self):
        self.responses = iter(
            [
                ModelResponse(
                    content='{"action":"search","query":"clue"}',
                    usage=Usage(input_tokens=10, output_tokens=5),
                ),
                ModelResponse(
                    content='{"action":"final","explanation":"found it","exact_answer":"Answer","confidence":90,"citations":["https://example.test"]}',
                    usage=Usage(input_tokens=12, output_tokens=6),
                ),
            ]
        )

    async def chat(self, messages, **kwargs):
        return next(self.responses)

    async def close(self):
        return None


class ChainFakeModel:
    def __init__(self):
        self.calls = []
        self.responses = iter(
            [
                ModelResponse(
                    content='{"action":"search","query":"clue"}',
                    response_id="chatcmpl-frlstate-root",
                ),
                ModelResponse(
                    content='{"action":"final","explanation":"found it","exact_answer":"Answer","confidence":90,"citations":["https://example.test"]}',
                    response_id="chatcmpl-frlstate-child",
                ),
            ]
        )

    async def chat(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        return next(self.responses)

    async def close(self):
        return None


class FinalCaptureModel:
    def __init__(self):
        self.kwargs = None

    async def chat(self, messages, **kwargs):
        del messages
        self.kwargs = deepcopy(kwargs)
        return ModelResponse(
            content='{"action":"final","explanation":"best answer","exact_answer":"Answer","confidence":80,"citations":["https://example.test"]}'
        )

    async def close(self):
        return None


class AutomaticExternalModel:
    def __init__(self):
        self.responses = iter(
            [
                ModelResponse(content='{"action":"search","query":"first clue"}'),
                ModelResponse(content='{"action":"search","query":"second clue"}'),
                ModelResponse(
                    content='{"action":"final","explanation":"checked","exact_answer":"Answer","confidence":88,"citations":["https://example.test"]}'
                ),
            ]
        )

    async def chat(self, messages, **kwargs):
        del messages, kwargs
        return next(self.responses)

    async def close(self):
        return None


class DuplicateSearchModel:
    def __init__(self):
        self.responses = iter(
            [
                ModelResponse(content='{"action":"search","query":"same clue"}'),
                ModelResponse(content='{"action":"search","query":"same clue"}'),
                ModelResponse(
                    content='{"action":"final","explanation":"page checked","exact_answer":"Answer","confidence":85,"citations":["https://example.test"]}'
                ),
            ]
        )

    async def chat(self, messages, **kwargs):
        del messages, kwargs
        return next(self.responses)

    async def close(self):
        return None


class BudgetRescueModel:
    def __init__(self):
        self.responses = iter(
            [
                ModelResponse(content='{"action":"search","query":"first clue"}'),
                ModelResponse(content='{"action":"search","query":"final verification"}'),
            ]
        )

    async def chat(self, messages, **kwargs):
        del messages, kwargs
        return next(self.responses)

    async def close(self):
        return None


class FakeSearch:
    def __init__(self, tmp_path: Path):
        self.config = SearchConfig(provider="searxng", cache_path=tmp_path / "s.sqlite3")

    async def search(self, query, count=None, offset=0):
        return [SearchResult(title="Result", url="https://example.test", snippet="clue")]


class FakeBrowser:
    async def fetch(self, url):
        raise AssertionError("not expected")


class SuccessfulBrowser:
    async def fetch(self, url):
        return PageDocument(
            requested_url=url,
            final_url=url,
            title="Evidence",
            text="verified page evidence",
            content_type="text/plain",
            status_code=200,
        )


class FakeExternalModelBroker:
    def __init__(self):
        self.requests = []

    async def ask_many(self, requests, *, request_namespace):
        self.requests.append((deepcopy(requests), request_namespace))
        return [
            {
                "ok": True,
                "status": "succeeded",
                "request_id": f"emr_{index}",
                "provider": "mock",
                "model": "mock",
                "content": f"critique {item['query']}",
            }
            for index, item in enumerate(requests, start=1)
        ]


class RescueExternalModelBroker:
    async def ask_many(self, requests, *, request_namespace):
        del requests, request_namespace
        return [
            {
                "ok": True,
                "status": "succeeded",
                "request_id": "emr_rescue",
                "content": (
                    '{"action":"final","explanation":"evidence resolves it",'
                    '"exact_answer":"Answer","confidence":90,'
                    '"citations":["https://example.test"]}'
                ),
            }
        ]


@pytest.mark.asyncio
async def test_agent_search_then_final(tmp_path: Path) -> None:
    runner = AgentRunner(
        ModelConfig(api_base="http://model.test/v1", api_key="k", model="m"),
        AgentConfig(max_steps=4, max_search_calls=2, require_citations=True),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=FakeModel(),
    )
    outcome = await runner.run("Question")
    assert outcome.status == "completed"
    assert outcome.exact_answer == "Answer"
    assert outcome.search_calls == 1
    assert outcome.usage.input_tokens == 22


@pytest.mark.asyncio
async def test_agent_response_chain_sends_only_tool_delta(tmp_path: Path) -> None:
    model = ChainFakeModel()
    runner = AgentRunner(
        ModelConfig(
            api_base="http://model.test/v1",
            api_key="k",
            model="m",
            protocol="json",
            response_chain=True,
            temperature=0.3,
            max_output_tokens=16384,
        ),
        AgentConfig(max_steps=4, max_search_calls=2, require_citations=True),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=model,
    )
    outcome = await runner.run("Question", request_namespace="run:item:attempt-1")
    assert outcome.status == "completed"
    assert len(model.calls) == 2
    root_messages, root_kwargs = model.calls[0]
    assert [message["role"] for message in root_messages] == ["system", "user"]
    assert root_kwargs["extra_body"]["frontierrl_messages_mode"] == "full"
    delta_messages, delta_kwargs = model.calls[1]
    assert len(delta_messages) == 1
    assert delta_messages[0]["role"] == "user"
    assert delta_messages[0]["content"].startswith("Tool result:")
    assert delta_kwargs["extra_body"]["frontierrl_messages_mode"] == "delta"
    assert delta_kwargs["extra_body"]["frontierrl_previous_response_id"] == "chatcmpl-frlstate-root"


@pytest.mark.asyncio
async def test_agent_forces_final_tool_choice_on_last_step(tmp_path: Path) -> None:
    model = FinalCaptureModel()
    runner = AgentRunner(
        ModelConfig(
            api_base="http://model.test/v1",
            api_key="k",
            model="m",
            protocol="tools",
            temperature=0.3,
            max_output_tokens=16384,
        ),
        AgentConfig(max_steps=1, require_citations=True),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=model,
    )
    outcome = await runner.run("Question")
    assert outcome.status == "completed"
    assert model.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "final"},
    }
    assert {tool["function"]["name"] for tool in model.kwargs["tools"]} == {
        "search",
        "search_many",
        "open",
        "open_many",
        "find",
        "note",
        "final",
    }


@pytest.mark.asyncio
async def test_agent_keeps_tool_schema_stable_when_page_inspection_is_due(tmp_path: Path) -> None:
    model = FinalCaptureModel()
    runner = AgentRunner(
        ModelConfig(api_base="http://model.test/v1", api_key="k", model="m", protocol="tools"),
        AgentConfig(),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=model,
    )
    await runner._query([], "tools", require_open=True)
    assert {tool["function"]["name"] for tool in model.kwargs["tools"]} == {
        "search",
        "search_many",
        "open",
        "open_many",
        "find",
        "note",
        "final",
    }


def test_result_has_urls_detects_batched_search_candidates() -> None:
    assert AgentRunner._result_has_urls(
        {"searches": [{"query": "q", "results": [{"url": "https://example.test"}]}]}
    )
    assert not AgentRunner._result_has_urls({"searches": [{"query": "q", "results": []}]})


def test_candidate_urls_are_selected_round_robin() -> None:
    result = {
        "searches": [
            {
                "results": [
                    {"url": "https://a.test/1"},
                    {"url": "https://a.test/2"},
                ]
            },
            {
                "results": [
                    {"url": "https://b.test/1"},
                    {"url": "https://a.test/1"},
                ]
            },
        ]
    }
    assert AgentRunner._candidate_urls(result, 3) == [
        "https://a.test/1",
        "https://b.test/1",
        "https://a.test/2",
    ]


def test_unopened_candidate_urls_skip_prior_pages() -> None:
    result = {
        "results": [
            {"url": "https://already.test/page"},
            {"url": "https://new.test/one"},
            {"url": "https://new.test/two"},
        ]
    }
    opened = {
        "https://already.test/page": PageDocument(
            requested_url="https://already.test/page",
            final_url="https://already.test/page",
            title="",
            text="",
            content_type="text/plain",
            status_code=200,
        )
    }
    assert AgentRunner._unopened_candidate_urls(result, opened=opened, limit=1) == [
        "https://new.test/one"
    ]


def test_external_consultation_urls_skip_opened_and_deduplicate() -> None:
    opened = {
        "https://already.test/page": PageDocument(
            requested_url="https://already.test/page",
            final_url="https://already.test/page",
            title="",
            text="",
            content_type="text/plain",
            status_code=200,
        )
    }
    consultations = [
        {
            "content": (
                "See https://already.test/page and https://source.test/a. "
                "Cross-check https://source.test/a and http://source.test/b."
            )
        }
    ]
    assert AgentRunner._external_consultation_urls(
        consultations,
        opened=opened,
        limit=3,
    ) == ["https://source.test/a", "http://source.test/b"]


def test_forced_final_recovers_agent_backend_plain_content() -> None:
    action = AgentRunner._plain_final_action(
        "Explanation: Evidence converges.\nExact Answer: Example\nConfidence: 91%\n"
        "https://example.test/source"
    )
    assert action.action == "final"
    assert action.payload["exact_answer"] == "Example"
    assert action.payload["confidence"] == 91
    assert action.payload["citations"] == ["https://example.test/source"]


def test_search_many_is_rejected_before_overspending_budget(tmp_path: Path) -> None:
    runner = AgentRunner(
        ModelConfig(api_base="http://model.test/v1", api_key="k", model="m"),
        AgentConfig(max_search_calls=2, max_batch_size=5),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=FakeModel(),
    )
    violation = runner._action_budget_violation(
        AgentAction(action="search_many", payload={"queries": ["a", "b", "c"]}),
        search_calls=0,
        page_opens=0,
        find_calls=0,
        retrieved_chars=0,
    )
    assert violation == (
        "Action was not executed because it would exceed the budget: search calls 3>2"
    )


def test_search_many_is_clipped_to_remaining_budget(tmp_path: Path) -> None:
    runner = AgentRunner(
        ModelConfig(api_base="http://model.test/v1", api_key="k", model="m"),
        AgentConfig(max_search_calls=4, max_batch_size=5),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=FakeModel(),
    )
    action, requested_count = runner._clip_action_to_remaining_budget(
        AgentAction(action="search_many", payload={"queries": ["a", "b", "c"]}),
        search_calls=2,
        page_opens=0,
        external_model_calls=0,
    )
    assert requested_count == 3
    assert action.payload["queries"] == ["a", "b"]


@pytest.mark.asyncio
async def test_agent_executes_external_model_fanout_as_one_action(tmp_path: Path) -> None:
    broker = FakeExternalModelBroker()
    external_config = ExternalModelConfig(
        enabled=True,
        default_provider="mock",
        allowed_providers=["mock"],
        max_batch_size=4,
    )
    runner = AgentRunner(
        ModelConfig(api_base="http://model.test/v1", api_key="k", model="m"),
        AgentConfig(),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=FakeModel(),
        external_model_config=external_config,
        external_model_broker=broker,
    )
    result, deltas = await runner._execute_action(
        AgentAction(
            action="ask_external_model",
            payload={"requests": [{"query": "a"}, {"query": "b"}]},
        ),
        {},
        [],
        request_namespace="run:item:1",
    )
    assert result["ok"] is True
    assert result["attempted"] == 2
    assert len(result["consultations"]) == 2
    assert deltas[3] > 0
    assert broker.requests[0][1] == "run:item:1"


@pytest.mark.asyncio
async def test_agent_automatically_attaches_external_reviews_after_search_threshold(
    tmp_path: Path,
) -> None:
    broker = FakeExternalModelBroker()
    runner = AgentRunner(
        ModelConfig(api_base="http://model.test/v1", api_key="k", model="m"),
        AgentConfig(
            max_steps=4,
            max_search_calls=4,
            automatic_external_after_search_calls=2,
            automatic_external_requests=2,
        ),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=AutomaticExternalModel(),
        external_model_config=ExternalModelConfig(
            enabled=True,
            default_provider="mock",
            allowed_providers=["mock"],
            max_calls_per_task=4,
        ),
        external_model_broker=broker,
    )
    outcome = await runner.run("Question", request_namespace="run:item:auto")
    assert outcome.status == "completed"
    assert outcome.external_model_calls == 2
    assert len(broker.requests) == 1
    requests, namespace = broker.requests[0]
    assert namespace == hashlib.sha256(b"run:item:auto").hexdigest()[:24]
    assert len(requests) == 2
    assert all("Original research question" in request["context"] for request in requests)


@pytest.mark.asyncio
async def test_open_many_reports_failure_when_every_page_fails(tmp_path: Path) -> None:
    runner = AgentRunner(
        ModelConfig(api_base="http://model.test/v1", api_key="k", model="m"),
        AgentConfig(),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=FakeModel(),
    )
    result, deltas = await runner._execute_action(
        AgentAction(
            action="open_many",
            payload={"urls": ["https://one.test", "https://two.test"]},
        ),
        {},
        [],
        request_namespace="run:item:open-many",
    )
    assert result["ok"] is False
    assert result["succeeded"] == 0
    assert result["failed"] == 2
    assert deltas[1] == 0


def test_external_help_budget_does_not_force_final(tmp_path: Path) -> None:
    runner = AgentRunner(
        ModelConfig(api_base="http://model.test/v1", api_key="k", model="m"),
        AgentConfig(max_search_calls=10),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=FakeModel(),
        external_model_config=ExternalModelConfig(enabled=True, max_calls_per_task=8),
    )
    assert runner._near_budget(0, 0, 0, 0, 8) is False


@pytest.mark.asyncio
async def test_agent_automatically_inspects_pages_after_search_phase(tmp_path: Path) -> None:
    runner = AgentRunner(
        ModelConfig(api_base="http://model.test/v1", api_key="k", model="m"),
        AgentConfig(
            max_steps=4,
            max_search_calls=4,
            automatic_page_inspection_after_search_actions=2,
            automatic_page_inspection_count=2,
        ),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        SuccessfulBrowser(),
        model_client=AutomaticExternalModel(),
    )
    outcome = await runner.run("Question", request_namespace="run:item:auto-open")
    assert outcome.status == "completed"
    assert outcome.page_opens == 1
    tool_results = [
        row["content"]
        for row in outcome.transcript
        if row.get("role") == "user" and row.get("content", "").startswith("Tool result:")
    ]
    assert any("automatic_page_inspection" in row for row in tool_results)


@pytest.mark.asyncio
async def test_repeated_search_opens_fresh_evidence_instead_of_looping(tmp_path: Path) -> None:
    runner = AgentRunner(
        ModelConfig(
            api_base="http://model.test/v1",
            api_key="k",
            model="m",
            protocol="json",
            response_chain=False,
        ),
        AgentConfig(
            max_steps=4,
            max_search_calls=4,
            automatic_page_inspection_after_search_actions=0,
            max_consecutive_duplicate_actions=3,
        ),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        SuccessfulBrowser(),
        model_client=DuplicateSearchModel(),
    )
    outcome = await runner.run("Question", request_namespace="run:item:duplicate")
    assert outcome.status == "completed"
    assert outcome.search_calls == 1
    assert outcome.page_opens == 1
    tool_results = [
        row["content"]
        for row in outcome.transcript
        if row.get("role") == "user" and row.get("content", "").startswith("Tool result:")
    ]
    assert any("controller_recovery" in row for row in tool_results)


@pytest.mark.asyncio
async def test_hard_budget_uses_one_external_finalization_rescue(tmp_path: Path) -> None:
    runner = AgentRunner(
        ModelConfig(
            api_base="http://model.test/v1",
            api_key="k",
            model="m",
            protocol="json",
            response_chain=False,
        ),
        AgentConfig(
            max_steps=3,
            max_search_calls=1,
            automatic_finalization_rescue_after_rejections=1,
        ),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=BudgetRescueModel(),
        external_model_config=ExternalModelConfig(
            enabled=True,
            default_provider="mock",
            allowed_providers=["mock"],
            max_calls_per_task=4,
        ),
        external_model_broker=RescueExternalModelBroker(),
    )
    outcome = await runner.run("Question", request_namespace="run:item:rescue")
    assert outcome.status == "completed"
    assert outcome.exact_answer == "Answer"
    assert outcome.external_model_calls == 1


@pytest.mark.asyncio
async def test_external_tool_is_exposed_only_when_enabled(tmp_path: Path) -> None:
    model = FinalCaptureModel()
    runner = AgentRunner(
        ModelConfig(
            api_base="http://model.test/v1",
            api_key="k",
            model="m",
            protocol="tools",
        ),
        AgentConfig(),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=model,
        external_model_config=ExternalModelConfig(enabled=True),
        external_model_broker=FakeExternalModelBroker(),
    )
    await runner._query([], "tools")
    assert "ask_external_model" in {tool["function"]["name"] for tool in model.kwargs["tools"]}
