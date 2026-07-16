import hashlib
import json
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


class ToolAbstentionChainModel:
    def __init__(self):
        self.calls = []
        self.responses = iter(
            [
                ModelResponse(
                    content="",
                    response_id="chatcmpl-frlstate-abstention",
                    raw_message={
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-abstention",
                                "type": "function",
                                "function": {
                                    "name": "final",
                                    "arguments": (
                                        '{"explanation":"uncertain",'
                                        '"exact_answer":"Insufficient evidence",'
                                        '"confidence":95,"citations":[]}'
                                    ),
                                },
                            }
                        ],
                    },
                ),
                ModelResponse(
                    content="",
                    response_id="chatcmpl-frlstate-concrete",
                    raw_message={
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-concrete",
                                "type": "function",
                                "function": {
                                    "name": "final",
                                    "arguments": (
                                        '{"explanation":"best candidate",'
                                        '"exact_answer":"Concrete Answer",'
                                        '"confidence":65,'
                                        '"citations":["https://example.test"]}'
                                    ),
                                },
                            }
                        ],
                    },
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


class SurfaceConstraintModel:
    def __init__(self):
        self.calls = []
        self.responses = iter(
            [
                ModelResponse(
                    content="",
                    raw_message={
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-invalid-surface",
                                "type": "function",
                                "function": {
                                    "name": "final",
                                    "arguments": (
                                        '{"explanation":"nearby concept",'
                                        '"exact_answer":"A Study of an Intervention Program",'
                                        '"confidence":95,"citations":["https://example.test/a"]}'
                                    ),
                                },
                            }
                        ],
                    },
                ),
                ModelResponse(
                    content="",
                    raw_message={
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-valid-surface",
                                "type": "function",
                                "function": {
                                    "name": "final",
                                    "arguments": (
                                        '{"explanation":"literal title match",'
                                        '"exact_answer":"A Longitudinal Study and an Intervention",'
                                        '"confidence":85,"citations":["https://example.test/b"]}'
                                    ),
                                },
                            }
                        ],
                    },
                ),
            ]
        )

    async def chat(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        return next(self.responses)

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


class StagnatingSearchModel:
    def __init__(self):
        self.responses = iter(
            [
                ModelResponse(
                    content=('{"action":"search","query":"entity first documented use 2012..2023"}')
                ),
                ModelResponse(
                    content=('{"action":"search","query":"entity first documented use 2012-2023"}')
                ),
                ModelResponse(
                    content=(
                        '{"action":"final","explanation":"strategy evidence",'
                        '"exact_answer":"Answer","confidence":85,'
                        '"citations":["https://example.test"]}'
                    )
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


class ForcedFinalIgnoringModel:
    def __init__(self):
        self.responses = iter(
            [
                ModelResponse(content='{"action":"search","query":"same clue"}'),
                ModelResponse(content='{"action":"search","query":"same clue"}'),
                ModelResponse(content='{"action":"search","query":"same clue"}'),
            ]
        )

    async def chat(self, messages, **kwargs):
        del messages, kwargs
        return next(self.responses)

    async def close(self):
        return None


class AbstentionThenConcreteModel:
    def __init__(self):
        self.responses = iter(
            [
                ModelResponse(
                    content=(
                        '{"action":"final","explanation":"uncertain",'
                        '"exact_answer":"Not verifiable from the supplied evidence",'
                        '"confidence":95,"citations":["https://example.test"]}'
                    )
                ),
                ModelResponse(
                    content=(
                        '{"action":"final","explanation":"best supported candidate",'
                        '"exact_answer":"Concrete Answer","confidence":65,'
                        '"citations":["https://example.test"]}'
                    )
                ),
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


class RecordingSearch(FakeSearch):
    def __init__(self, tmp_path: Path):
        super().__init__(tmp_path)
        self.queries = []

    async def search(self, query, count=None, offset=0):
        del count, offset
        self.queries.append(query)
        return [SearchResult(title="Result", url="https://example.test", snippet="clue")]

    async def search_many(self, queries, count=None):
        del count
        self.queries.extend(queries)
        return [
            [SearchResult(title="Result", url=f"https://example.test/{index}", snippet="clue")]
            for index, _ in enumerate(queries, start=1)
        ]


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


class LinkedSuccessfulBrowser:
    async def fetch(self, url):
        links = []
        text = "verified page evidence"
        if url != "https://example.test/related-history":
            links = [
                {
                    "text": "Entity history and origins",
                    "url": "https://example.test/related-history",
                }
            ]
        else:
            text = (
                "Entity history and origins. According to explorer Ada Lovelace, the first "
                "documented use of the entity appeared in a primary chronicle."
            )
        return PageDocument(
            requested_url=url,
            final_url=url,
            title="Evidence",
            text=text,
            content_type="text/plain",
            status_code=200,
            links=links,
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
        del request_namespace
        result = {
            "ok": True,
            "status": "succeeded",
            "request_id": "emr_rescue",
            "content": (
                '{"action":"final","explanation":"evidence resolves it",'
                '"exact_answer":"Answer","confidence":0.9,'
                '"citations":["https://example.test"]}'
            ),
        }
        return [
            {
                **result,
                "request_id": f"emr_rescue_{index}",
            }
            for index, _ in enumerate(requests, start=1)
        ]


class QueryStrategyExternalModelBroker:
    def __init__(self):
        self.calls = []

    async def ask_many(self, requests, *, request_namespace):
        self.calls.append((deepcopy(requests), request_namespace))
        return [
            {
                "ok": True,
                "status": "succeeded",
                "request_id": "emr_query_strategy",
                "content": (
                    '{"analysis":"pivot from repeated clue wording",'
                    '"queries":["entity history explorer attribution",'
                    '"candidate chronicler entity earliest account"]}'
                ),
            }
        ]


class StructuredConsultationBroker:
    def __init__(self):
        self.calls = []

    async def ask_many(self, requests, *, request_namespace):
        self.calls.append((deepcopy(requests), request_namespace))
        results = []
        for index, request in enumerate(requests, start=1):
            if "Role: Search strategy specialist" in request["query"]:
                content = (
                    '{"analysis":"compare entities","entity_candidates":["A","B"],'
                    '"queries":["entity A primary history","entity B attribution source"]}'
                )
            else:
                content = "independent critique"
            results.append(
                {
                    "ok": True,
                    "status": "succeeded",
                    "request_id": f"emr_structured_{index}",
                    "content": content,
                }
            )
        return results


class RepairingExternalModelBroker:
    def __init__(self):
        self.calls = []

    async def ask_many(self, requests, *, request_namespace):
        self.calls.append((deepcopy(requests), request_namespace))
        if request_namespace.endswith(":finalization-reviews"):
            answers = [("Candidate One", 62), ("Candidate Two", 58)]
        elif request_namespace.endswith(":finalization-adjudication"):
            answers = [("Insufficient evidence", 96)]
        elif request_namespace.endswith(":finalization-repair"):
            answers = [("Concrete Answer", 64)]
        else:
            raise AssertionError(request_namespace)
        return [
            {
                "ok": True,
                "status": "succeeded",
                "request_id": f"emr_{len(self.calls)}_{index}",
                "content": (
                    '{"action":"final","explanation":"comparative audit",'
                    f'"exact_answer":"{answer}","confidence":{confidence},'
                    '"citations":["https://example.test"]}'
                ),
            }
            for index, (answer, confidence) in enumerate(answers, start=1)
        ]


class AbstainingExternalModelBroker(RepairingExternalModelBroker):
    async def ask_many(self, requests, *, request_namespace):
        if request_namespace.endswith(":finalization-repair"):
            self.calls.append((deepcopy(requests), request_namespace))
            return [
                {
                    "ok": True,
                    "status": "succeeded",
                    "request_id": "emr_repair_abstention",
                    "content": (
                        '{"action":"final","explanation":"still uncertain",'
                        '"exact_answer":"Cannot determine","confidence":99,'
                        '"citations":["https://example.test"]}'
                    ),
                }
            ]
        return await super().ask_many(requests, request_namespace=request_namespace)


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
    assert root_kwargs["request_headers"] == {
        "X-FRL-Conversation-Id": "bc250-29c60a6f7b8c03ccf4ae8ca6"
    }
    delta_messages, delta_kwargs = model.calls[1]
    assert len(delta_messages) == 1
    assert delta_messages[0]["role"] == "user"
    assert delta_messages[0]["content"].startswith("Tool result:")
    assert delta_kwargs["extra_body"]["frontierrl_messages_mode"] == "delta"
    assert delta_kwargs["extra_body"]["frontierrl_previous_response_id"] == "chatcmpl-frlstate-root"
    assert delta_kwargs["request_headers"] == root_kwargs["request_headers"]


@pytest.mark.asyncio
async def test_agent_response_chain_rejects_abstention_with_tool_result_delta(
    tmp_path: Path,
) -> None:
    model = ToolAbstentionChainModel()
    runner = AgentRunner(
        ModelConfig(
            api_base="http://model.test/v1",
            api_key="k",
            model="m",
            protocol="tools",
            response_chain=True,
        ),
        AgentConfig(max_steps=2, require_citations=True),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=model,
    )
    outcome = await runner.run("Question", request_namespace="run:item:tool-abstention")
    assert outcome.status == "completed"
    assert outcome.exact_answer == "Concrete Answer"
    delta_messages, delta_kwargs = model.calls[1]
    assert len(delta_messages) == 1
    assert delta_messages[0]["role"] == "tool"
    assert delta_messages[0]["tool_call_id"] == "call-abstention"
    assert "abstentions are invalid" in delta_messages[0]["content"]
    assert (
        delta_kwargs["extra_body"]["frontierrl_previous_response_id"]
        == "chatcmpl-frlstate-abstention"
    )


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


def test_benchmark_routing_headers_spread_rows_but_keep_each_chain_sticky() -> None:
    namespace = "campaign-run:bc250-023-row-0151:attempt-1"
    headers = AgentRunner._routing_headers(namespace)
    assert headers == {
        "X-FRL-Conversation-Id": ("bc250-" + hashlib.sha256(namespace.encode()).hexdigest()[:24]),
        "X-FRL-KV-Cohort-Id": ("bc250-" + hashlib.sha256(b"campaign-run").hexdigest()[:20]),
        "X-FRL-KV-Cohort-Index": "23",
    }
    assert AgentRunner._routing_headers("helper-namespace") == {
        "X-FRL-Conversation-Id": ("bc250-" + hashlib.sha256(b"helper-namespace").hexdigest()[:24])
    }
    helper_namespace = namespace + ":finalization-reviews:star2-agent:1"
    helper_headers = AgentRunner._routing_headers(helper_namespace)
    assert helper_headers["X-FRL-Conversation-Id"] == (
        "bc250-" + hashlib.sha256(helper_namespace.encode()).hexdigest()[:24]
    )
    assert helper_headers["X-FRL-KV-Cohort-Id"] == (
        "bc250-" + hashlib.sha256(b"campaign-run:star2-helpers").hexdigest()[:20]
    )
    assert helper_headers["X-FRL-KV-Cohort-Index"] == str(
        int(hashlib.sha256(helper_namespace.encode()).hexdigest()[:12], 16)
    )


def test_explicit_backend_pool_distributes_and_pins_independent_chains() -> None:
    backend_pool = [f"star2-{index}" for index in range(8)]
    namespaces = [
        f"campaign-run:bc250-{index:03d}-row-{index:04d}:attempt-1:star2-agent:1"
        for index in range(256)
    ]
    selected = {
        AgentRunner._routing_headers(
            namespace,
            routing_backend_pool=backend_pool,
        )["X-FRL-Require-Backend"]
        for namespace in namespaces
    }
    assert selected == set(backend_pool)
    namespace = namespaces[17]
    first = AgentRunner._routing_headers(namespace, routing_backend_pool=backend_pool)
    second = AgentRunner._routing_headers(namespace, routing_backend_pool=backend_pool)
    assert first == second
    assert first["X-FRL-Require-Backend"] in backend_pool


@pytest.mark.asyncio
async def test_agent_rejects_final_until_required_independent_search(tmp_path: Path) -> None:
    def tool_call(call_id: str, name: str, arguments: dict) -> ModelResponse:
        return ModelResponse(
            content="",
            raw_message={
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            },
        )

    class PrematureFinalModel:
        def __init__(self) -> None:
            self.calls = []
            self.responses = iter(
                [
                    tool_call(
                        "call-early",
                        "final",
                        {
                            "explanation": "guess",
                            "exact_answer": "Wrong",
                            "confidence": 90,
                            "citations": ["https://example.test/guess"],
                        },
                    ),
                    tool_call("call-search", "search", {"query": "candidate falsification"}),
                    tool_call(
                        "call-final",
                        "final",
                        {
                            "explanation": "verified",
                            "exact_answer": "Answer",
                            "confidence": 80,
                            "citations": ["https://example.test/evidence"],
                        },
                    ),
                ]
            )

        async def chat(self, messages, **kwargs):
            self.calls.append(deepcopy(messages))
            return next(self.responses)

        async def close(self):
            return None

    model = PrematureFinalModel()
    runner = AgentRunner(
        ModelConfig(
            api_base="http://model.test/v1",
            api_key="k",
            model="m",
            protocol="tools",
        ),
        AgentConfig(
            max_steps=3,
            max_search_calls=2,
            min_search_calls_before_final=1,
            require_citations=True,
        ),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=model,
    )
    outcome = await runner.run("Question")
    assert outcome.status == "completed"
    assert outcome.exact_answer == "Answer"
    assert outcome.search_calls == 1
    assert model.calls[1][-1]["role"] == "tool"
    assert model.calls[1][-1]["tool_call_id"] == "call-early"
    assert "Finalization is premature" in model.calls[1][-1]["content"]


def test_search_novelty_suppresses_date_range_and_near_duplicate_variants() -> None:
    action, suppressed = AgentRunner._filter_redundant_search_action(
        AgentAction(
            action="search_many",
            payload={
                "queries": [
                    '"first person to document entity use" 2012-2023',
                    '"different candidate" entity earliest account',
                ]
            },
        ),
        ['"first person to document entity use" 2012..2023'],
    )
    assert action is not None
    assert action.payload["queries"] == ['"different candidate" entity earliest account']
    assert suppressed == ['"first person to document entity use" 2012-2023']


def test_search_strategy_parser_returns_only_novel_queries() -> None:
    result = {
        "ok": True,
        "content": (
            "```json\n"
            '{"analysis":"pivot","queries":['
            '"same clue 2012-2023","entity history explorer","entity history explorer"]}'
            "\n```"
        ),
    }
    assert AgentRunner._strategy_queries_from_result(
        result,
        prior_queries=["same clue 2012..2023"],
        limit=4,
    ) == ["entity history explorer"]


def test_search_strategy_parser_falls_back_to_helper_executed_queries() -> None:
    result = {
        "ok": True,
        "content": "The helper returned prose instead of the requested query JSON.",
        "agent_search_queries": [
            "same clue 2012-2023",
            "rare collaborator archive",
            "rare collaborator archive",
        ],
    }
    assert AgentRunner._strategy_queries_from_result(
        result,
        prior_queries=["same clue 2012..2023"],
        limit=4,
    ) == ["rare collaborator archive"]


def test_consultation_strategy_uses_only_designated_role() -> None:
    consultations = [
        {
            "ok": True,
            "review_role": "Independent candidate investigator",
            "content": '{"queries":["ignore this answer query"]}',
        },
        {
            "ok": True,
            "review_role": "Search strategy specialist",
            "content": '{"queries":["already searched 2012-2023","new entity history"]}',
        },
    ]
    assert AgentRunner._consultation_strategy_queries(
        consultations,
        prior_queries=["already searched 2012..2023"],
        limit=3,
    ) == ["new entity history"]


def test_finalization_fallback_never_borrows_unrelated_context_citations() -> None:
    reviews = [
        {
            "ok": True,
            "content": (
                '{"action":"final","explanation":"unsupported guess",'
                '"exact_answer":"Candidate","confidence":99,"citations":[]}'
            ),
        }
    ]
    assert AgentRunner._best_review_fallback(reviews) is None


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


def test_strategy_candidate_urls_preserve_route_breadth_before_depth() -> None:
    result = {
        "searches": [
            {
                "results": [
                    {"url": "https://priority.test/1"},
                    {"url": "https://priority.test/2"},
                ]
            },
            {"results": [{"url": "https://alternative.test/1"}]},
        ]
    }
    assert AgentRunner._strategy_candidate_urls(result, 3) == [
        "https://priority.test/1",
        "https://alternative.test/1",
        "https://priority.test/2",
    ]


def test_related_evidence_urls_rank_semantic_same_site_links() -> None:
    pages = [
        {
            "final_url": "https://source.test/colonial-overview",
            "links": [
                {
                    "text": "Shop all products",
                    "url": "https://source.test/collections/all",
                },
                {
                    "text": "Entity history and origins",
                    "url": "https://source.test/entity-history-origins",
                },
                {
                    "text": "Independent entity history study",
                    "url": "https://journal.test/entity-history-study",
                },
            ],
        }
    ]
    assert AgentRunner._related_evidence_urls(
        pages,
        queries=["entity history origins earliest written account"],
        opened={},
        limit=2,
    ) == [
        "https://source.test/entity-history-origins",
        "https://journal.test/entity-history-study",
    ]


def test_evidence_highlights_keep_strongest_at_bounded_tail() -> None:
    pages = [
        {
            "title": "Entity history",
            "final_url": "https://source.test/history",
            "text": (
                "General entity history discusses many unrelated details and background.\n\n"
                "According to explorer Ada Lovelace, the first documented use of the entity "
                "appeared in a primary chronicle."
            ),
        },
        {
            "title": "Entity shop",
            "final_url": "https://source.test/shop",
            "text": "Buy the entity today with free delivery and seasonal discounts.",
        },
        {
            "title": "Entity navigation",
            "final_url": "https://source.test/navigation",
            "text": (
                "[Entity history origins earliest written account and first documentation]"
                "(https://source.test/history)"
            ),
        },
    ]
    highlights = AgentRunner._evidence_highlights(
        pages,
        queries=["entity history origins earliest written account"],
        limit=3,
    )
    assert highlights[-1] == {
        "title": "Entity history",
        "url": "https://source.test/history",
        "passage": (
            "According to explorer Ada Lovelace, the first documented use of the entity "
            "appeared in a primary chronicle."
        ),
    }


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
        question="Which independently documented person fits the historical clues?",
        opened=opened,
        limit=3,
    ) == ["https://source.test/a", "http://source.test/b"]


def test_external_consultation_urls_reject_query_mirror_paths_without_domain_rules() -> None:
    question = (
        "Which artist born in England made an album for fun, had a debut album between "
        "2001 and 2005, released albums with Roman numerals, and said they would never go "
        "commercial or feel threatened?"
    )
    mirror = (
        "https://spam.example/video/artist-born-in-england-made-an-album-for-fun-"
        "debut-album-between-2001-and-2005-released-albums-with-roman-numerals-"
        "would-never-go-commercial-do-not-feel-threatened/"
    )
    legitimate = "https://news.example/interviews/artist-profile-and-new-album"

    assert AgentRunner._looks_like_query_mirror_url(question, mirror)
    assert not AgentRunner._looks_like_query_mirror_url(question, legitimate)
    assert AgentRunner._external_consultation_urls(
        [{"content": f"Possible sources: {mirror} and {legitimate}"}],
        question=question,
        opened={},
        limit=3,
    ) == [legitimate]


def test_forced_final_recovers_agent_backend_plain_content() -> None:
    action = AgentRunner._plain_final_action(
        "Explanation: Evidence converges.\nExact Answer: Example\nConfidence: 91%\n"
        "https://example.test/source"
    )
    assert action.action == "final"
    assert action.payload["exact_answer"] == "Example"
    assert action.payload["confidence"] == 91
    assert action.payload["citations"] == ["https://example.test/source"]


@pytest.mark.parametrize(
    "answer",
    [
        "Unknown",
        "Insufficient evidence",
        "Not verifiable from the supplied evidence",
        "Not conclusively identifiable from the available evidence",
        "Cannot determine",
        "No conclusive answer",
    ],
)
def test_abstention_answers_are_detected(answer: str) -> None:
    assert AgentRunner._is_abstention_answer(answer)


def test_concrete_answers_are_not_abstentions() -> None:
    assert not AgentRunner._is_abstention_answer("Arbitrary Concrete Entity")


def test_surface_answer_constraints_enforce_literal_edge_words() -> None:
    ends_question = "The requested title ends with the word “Intervention”."
    assert AgentRunner._surface_answer_constraint_errors(
        ends_question,
        "A Study of an Intervention Program",
    ) == ["answer must end with 'Intervention', but its final word is 'Program'"]
    assert not AgentRunner._surface_answer_constraint_errors(
        ends_question,
        "A Longitudinal Study and an Intervention",
    )

    starts_question = "The answer begins with the word Alpha."
    assert AgentRunner._surface_answer_constraint_errors(
        starts_question,
        "Beta Alpha",
    ) == ["answer must start with 'Alpha', but its first word is 'Beta'"]
    assert not AgentRunner._surface_answer_constraint_errors(starts_question, "Alpha Beta")


@pytest.mark.asyncio
async def test_agent_rejects_final_that_violates_explicit_surface_constraint(
    tmp_path: Path,
) -> None:
    model = SurfaceConstraintModel()
    events: list[dict] = []
    runner = AgentRunner(
        ModelConfig(
            api_base="http://model.test/v1",
            api_key="k",
            model="m",
            protocol="tools",
        ),
        AgentConfig(max_steps=2, require_citations=True),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=model,
        event_sink=events.append,
    )

    outcome = await runner.run(
        "Which paper has a title ending with the word “Intervention”?",
    )

    assert outcome.status == "completed"
    assert outcome.exact_answer == "A Longitudinal Study and an Intervention"
    second_messages, _ = model.calls[1]
    correction = second_messages[-1]
    assert correction["role"] == "tool"
    assert correction["tool_call_id"] == "call-invalid-surface"
    assert "explicit surface constraint" in correction["content"]
    assert "final word is 'Program'" in correction["content"]
    assert any(event["event"] == "surface_constraint_final_rejected" for event in events)


def test_history_compaction_preserves_star_helper_milestone_and_bounds_payload(
    tmp_path: Path,
) -> None:
    events: list[dict] = []
    runner = AgentRunner(
        ModelConfig(api_base="http://model.test/v1", api_key="k", model="m"),
        AgentConfig(max_history_chars=30_000),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=FakeModel(),
        event_sink=events.append,
    )
    initial_user = "Question:\nIdentify the entity."
    messages: list[dict] = [
        {"role": "system", "content": runner.system_prompt},
        {"role": "user", "content": initial_user},
    ]
    for index in range(5):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query":"clue"}'},
                    }
                ],
            }
        )
        marker = "independent_external_consultation" if index == 0 else "search_results"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call-{index}",
                "name": "search",
                "content": f'{{"{marker}":"candidate"}}' + ("x" * 20_000),
            }
        )

    compacted = runner._compact_history(messages, initial_user, [], {})
    compacted_content = "\n".join(str(message.get("content") or "") for message in compacted)

    assert len(compacted_content) <= 30_000 + len(compacted) - 1
    assert "independent_external_consultation" in compacted_content
    assert compacted[0]["role"] == "system"
    assert compacted[1]["role"] == "user"
    assert compacted[3]["role"] == "assistant"
    assert compacted[4]["role"] == "tool"
    compaction_event = next(event for event in events if event["event"] == "history_compacted")
    assert compaction_event["before_chars"] > compaction_event["after_chars"]
    assert compaction_event["after_chars"] <= 30_000
    assert compaction_event["preserved_milestones"] == 1


@pytest.mark.asyncio
async def test_agent_retries_instead_of_accepting_abstention(tmp_path: Path) -> None:
    runner = AgentRunner(
        ModelConfig(
            api_base="http://model.test/v1",
            api_key="k",
            model="m",
            protocol="json",
        ),
        AgentConfig(max_steps=2, require_citations=True),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=AbstentionThenConcreteModel(),
    )
    outcome = await runner.run("Question")
    assert outcome.status == "completed"
    assert outcome.exact_answer == "Concrete Answer"
    assert any("abstentions are invalid" in error for error in outcome.errors)


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
    assert namespace == "run:item:auto"
    assert len(requests) == 2
    assert all("Original research question" in request["context"] for request in requests)


@pytest.mark.asyncio
async def test_agent_executes_structured_external_search_strategy(tmp_path: Path) -> None:
    broker = StructuredConsultationBroker()
    search = RecordingSearch(tmp_path)
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
            max_search_calls=6,
            automatic_external_after_search_calls=2,
            automatic_external_requests=1,
            automatic_page_inspection_after_search_actions=0,
            max_batch_size=4,
        ),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        search,
        LinkedSuccessfulBrowser(),
        model_client=AutomaticExternalModel(),
        external_model_config=ExternalModelConfig(
            enabled=True,
            default_provider="mock",
            allowed_providers=["mock"],
            max_calls_per_task=4,
        ),
        external_model_broker=broker,
    )
    outcome = await runner.run("Question", request_namespace="run:item:structured-strategy")
    assert outcome.status == "completed"
    assert outcome.errors == []
    assert outcome.search_calls == 4
    assert outcome.external_model_calls == 1
    assert outcome.page_opens == 3
    assert search.queries == [
        "first clue",
        "second clue",
        "entity A primary history",
        "entity B attribution source",
    ]
    requests, _ = broker.calls[0]
    strategy_request = requests[0]
    assert "strongest alternative" in strategy_request["query"]
    tool_results = [
        row["content"]
        for row in outcome.transcript
        if row.get("role") == "user" and row.get("content", "").startswith("Tool result:")
    ]
    assert any('"strategy_search"' in row for row in tool_results)
    assert any('"related_source_page_inspection"' in row for row in tool_results)
    assert any('"verified_evidence_highlights"' in row for row in tool_results)


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
async def test_repeated_search_uses_one_external_strategy_recovery(tmp_path: Path) -> None:
    search = RecordingSearch(tmp_path)
    broker = QueryStrategyExternalModelBroker()
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
            max_search_calls=5,
            automatic_external_after_search_calls=0,
            automatic_page_inspection_after_search_actions=0,
            max_batch_size=3,
        ),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        search,
        FakeBrowser(),
        model_client=StagnatingSearchModel(),
        external_model_config=ExternalModelConfig(
            enabled=True,
            default_provider="mock",
            allowed_providers=["mock"],
            max_calls_per_task=4,
        ),
        external_model_broker=broker,
    )
    outcome = await runner.run("Question", request_namespace="run:item:strategy")
    assert outcome.status == "completed"
    assert outcome.search_calls == 3
    assert outcome.external_model_calls == 1
    assert search.queries == [
        "entity first documented use 2012..2023",
        "entity history explorer attribution",
        "candidate chronicler entity earliest account",
    ]
    assert len(broker.calls) == 1
    assert broker.calls[0][1].endswith(":search-strategy-recovery")
    tool_results = [
        row["content"]
        for row in outcome.transcript
        if row.get("role") == "user" and row.get("content", "").startswith("Tool result:")
    ]
    assert any("external_search_strategy_recovery" in row for row in tool_results)


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
    assert outcome.confidence == 90
    assert outcome.external_model_calls == 3


@pytest.mark.asyncio
async def test_forced_final_nonfinal_action_uses_external_rescue(tmp_path: Path) -> None:
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
            max_search_calls=10,
            max_consecutive_duplicate_actions=1,
            automatic_finalization_rescue_after_rejections=1,
        ),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=ForcedFinalIgnoringModel(),
        external_model_config=ExternalModelConfig(
            enabled=True,
            default_provider="mock",
            allowed_providers=["mock"],
            max_calls_per_task=4,
        ),
        external_model_broker=RescueExternalModelBroker(),
    )
    outcome = await runner.run("Question", request_namespace="run:item:forced-final-rescue")
    assert outcome.status == "completed"
    assert outcome.exact_answer == "Answer"
    assert outcome.search_calls == 1
    assert outcome.external_model_calls == 4


@pytest.mark.asyncio
async def test_finalizer_repairs_external_abstention_within_four_calls(tmp_path: Path) -> None:
    broker = RepairingExternalModelBroker()
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
        external_model_broker=broker,
    )
    outcome = await runner.run("Question", request_namespace="run:item:repair")
    assert outcome.status == "completed"
    assert outcome.exact_answer == "Concrete Answer"
    assert outcome.external_model_calls == 4
    assert [namespace.rsplit(":", 1)[-1] for _, namespace in broker.calls] == [
        "finalization-reviews",
        "finalization-adjudication",
        "finalization-repair",
    ]


@pytest.mark.asyncio
async def test_finalizer_preserves_early_milestone_evidence(tmp_path: Path) -> None:
    broker = RepairingExternalModelBroker()
    runner = AgentRunner(
        ModelConfig(api_base="http://model.test/v1", api_key="k", model="m"),
        AgentConfig(),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=FakeModel(),
        external_model_config=ExternalModelConfig(
            enabled=True,
            default_provider="mock",
            allowed_providers=["mock"],
            max_calls_per_task=4,
        ),
        external_model_broker=broker,
    )
    messages = [
        {"role": "user", "content": "original question"},
        {
            "role": "tool",
            "content": '{"independent_external_consultation":"EARLY_ENTITY_SIGNAL"}',
        },
        *[{"role": "tool", "content": f"later evidence {index}"} for index in range(10)],
    ]
    action, result = await runner._automatic_external_finalization(
        question="Question",
        response=ModelResponse(content="latest candidate"),
        messages=messages,
        transcript=[],
        notes=[],
        request_namespace="run:item:milestone",
        request_budget=4,
    )
    assert action is not None
    assert result["ok"] is True
    first_review_context = broker.calls[0][0][0]["context"]
    assert "EARLY_ENTITY_SIGNAL" in first_review_context


@pytest.mark.asyncio
async def test_finalizer_falls_back_when_repair_also_abstains(tmp_path: Path) -> None:
    broker = AbstainingExternalModelBroker()
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
        external_model_broker=broker,
    )
    outcome = await runner.run("Question", request_namespace="run:item:fallback")
    assert outcome.status == "completed"
    assert outcome.exact_answer == "Candidate One"
    assert outcome.confidence == 62
    assert outcome.external_model_calls == 4


@pytest.mark.asyncio
async def test_wall_clock_uses_one_external_finalization_rescue(tmp_path: Path) -> None:
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
            max_search_calls=4,
            automatic_finalization_rescue_after_seconds=0.000001,
        ),
        BrowserConfig(cache_path=tmp_path / "p.sqlite3", block_private_networks=False),
        FakeSearch(tmp_path),
        FakeBrowser(),
        model_client=BudgetRescueModel(),
        external_model_config=ExternalModelConfig(
            enabled=True,
            default_provider="mock",
            allowed_providers=["mock"],
            max_calls_per_task=1,
        ),
        external_model_broker=RescueExternalModelBroker(),
    )
    outcome = await runner.run("Question", request_namespace="run:item:timed-rescue")
    assert outcome.status == "completed"
    assert outcome.exact_answer == "Answer"
    assert outcome.confidence == 90
    assert outcome.search_calls == 0
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
