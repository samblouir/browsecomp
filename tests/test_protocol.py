import json

import pytest

from browsecomp250.llm.protocol import (
    ProtocolError,
    action_from_tool_call,
    canonicalize_tool_call,
    parse_json_action,
)
from browsecomp250.llm.tools import tool_schemas


def test_parse_json_action_from_fence() -> None:
    action = parse_json_action('```json\n{"action":"search","query":"rare fact"}\n```')
    assert action.action == "search"
    assert action.payload["query"] == "rare fact"


def test_parse_embedded_object() -> None:
    action = parse_json_action('I will do this: {"action":"note","text":"x"}')
    assert action.action == "note"


def test_invalid_action() -> None:
    with pytest.raises(ProtocolError):
        parse_json_action('{"action":"shell","command":"rm -rf /"}')


def test_parse_external_model_fanout() -> None:
    action = parse_json_action(
        '{"action":"ask_external_model","requests":['
        '{"query":"challenge candidate A"},{"query":"challenge candidate B"}]}'
    )
    assert action.action == "ask_external_model"
    assert len(action.payload["requests"]) == 2


def test_external_model_requires_exactly_one_request_shape() -> None:
    with pytest.raises(ProtocolError, match="exactly one"):
        parse_json_action(
            '{"action":"ask_external_model","query":"q","requests":[{"query":"other"}]}'
        )


def test_parse_geo_search_with_expected_distances() -> None:
    action = parse_json_action(
        '{"action":"geo_search","anchors":['
        '{"query":"First landmark","radius_m":5000,"expected_distance_miles":1.2},'
        '{"query":"Second landmark","expected_distance_miles":2.4}],'
        '"category":"restaurant"}'
    )

    assert action.action == "geo_search"
    assert action.payload["anchors"][1]["expected_distance_miles"] == 2.4


@pytest.mark.parametrize("distance", [-1, 501, "one mile", True])
def test_geo_search_rejects_invalid_expected_distance(distance: object) -> None:
    with pytest.raises(ProtocolError, match="expected_distance_miles"):
        parse_json_action(
            '{"action":"geo_search","anchors":'
            f'[{{"query":"landmark","expected_distance_miles":{json.dumps(distance)}}}]}}'
        )


def test_tool_call_normalizes_singular_name_with_batch_arguments() -> None:
    action = action_from_tool_call(
        {
            "function": {
                "name": "search",
                "arguments": '{"queries":["one","two"]}',
            }
        }
    )
    assert action.action == "search_many"
    assert action.payload == {"queries": ["one", "two"]}


def test_tool_call_normalizes_singular_name_with_plural_string_argument() -> None:
    action = action_from_tool_call(
        {
            "function": {
                "name": "search",
                "arguments": '{"queries":"one precise query","count":7}',
            }
        }
    )
    assert action.action == "search"
    assert action.payload == {"query": "one precise query", "count": 7}


def test_tool_call_normalizes_batch_name_with_singular_arguments() -> None:
    action = action_from_tool_call(
        {
            "function": {
                "name": "open_many",
                "arguments": '{"url":"https://example.test"}',
            }
        }
    )
    assert action.action == "open"
    assert action.payload == {"url": "https://example.test"}


def test_tool_call_recovers_gemma_serialized_search_batch_from_function_name() -> None:
    action, canonical = canonicalize_tool_call(
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": (
                    'search_many(queries:[<|"|>WHO report introduction<|"|>,'
                    '<|"|>"Cristina Ortiz" graphic designer<|"|>]}<tool_call|>'
                ),
                "arguments": "{}",
            },
        }
    )

    assert action.action == "search_many"
    assert action.payload == {
        "queries": ["WHO report introduction", '"Cristina Ortiz" graphic designer']
    }
    assert canonical["function"] == {
        "name": "search_many",
        "arguments": (
            '{"queries":["WHO report introduction",'
            '"\\\"Cristina Ortiz\\\" graphic designer"]}'
        ),
    }


def test_tool_call_recovers_collapsed_adjacent_gemma_string_delimiter() -> None:
    action = action_from_tool_call(
        {
            "function": {
                "name": (
                    'search_many(queries:[<|"|>first query<|"|>,'
                    '<|"|>second query,,<|"|>third query<|"|>]}<tool_call|>'
                ),
                "arguments": "{}",
            }
        }
    )

    assert action.payload == {"queries": ["first query", "second query", "third query"]}


def test_tool_call_does_not_recover_unknown_serialized_function() -> None:
    with pytest.raises(ProtocolError, match="Unknown tool"):
        action_from_tool_call(
            {
                "function": {
                    "name": 'shell(command:<|"|>rm -rf /<|"|>)<tool_call|>',
                    "arguments": "{}",
                }
            }
        )


def test_external_tool_call_prefers_fanout_and_strips_routing_overrides() -> None:
    action, canonical = canonicalize_tool_call(
        {
            "function": {
                "name": "ask_external_model",
                "arguments": json.dumps(
                    {
                        "query": "redundant singular request",
                        "provider": "other",
                        "model": "other-model",
                        "context": "shared evidence",
                        "requests": [
                            {"query": "rare-anchor review", "model": "other-model"},
                            {"query": "falsify candidate", "provider": "other"},
                        ],
                    }
                ),
            }
        }
    )

    assert action.payload == {
        "requests": [
            {"query": "rare-anchor review", "context": "shared evidence"},
            {"query": "falsify candidate", "context": "shared evidence"},
        ]
    }
    assert json.loads(canonical["function"]["arguments"]) == action.payload


def test_forced_guide_step_restores_only_its_redacted_query_contract() -> None:
    action, _ = canonicalize_tool_call(
        {
            "function": {
                "name": "search",
                "arguments": '{"query":"transport-corrupted copy"}',
            }
        },
        expected_action="search_many",
        required_queries=["first redacted query", "second redacted query"],
    )

    assert action.action == "search_many"
    assert action.payload == {
        "queries": ["first redacted query", "second redacted query"]
    }


def test_forced_guide_open_uses_exact_url_even_if_worker_returns_wrong_known_tool() -> None:
    action, canonical = canonicalize_tool_call(
        {
            "function": {
                "name": "search_many",
                "arguments": '{"queries":["wrong transport action"]}',
            }
        },
        expected_action="open",
        required_urls=["https://example.test/exact-redacted-source"],
    )

    assert action.action == "open"
    assert action.payload == {"url": "https://example.test/exact-redacted-source"}
    assert canonical["function"]["name"] == "open"


def test_forced_external_step_fills_four_distinct_research_roles() -> None:
    action, canonical = canonicalize_tool_call(
        {
            "function": {
                "name": "ask_external_model",
                "arguments": json.dumps(
                    {
                        "requests": [
                            {
                                "context": "Original question and evidence.\nRole: rare-anchor solver.\nquery:",
                                "damaged serialized query": "unusable",
                            }
                        ]
                    }
                ),
            }
        },
        expected_action="ask_external_model",
        minimum_batch_size=4,
        external_context="Fallback original question and public evidence.",
    )

    requests = action.payload["requests"]
    assert len(requests) == 4
    assert all(request["query"].strip() for request in requests)
    assert {request.get("system") for request in requests if request.get("system")} == {
        "Act as the independent rare-anchor solver for this research task.",
        "Act as the independent relation-graph inverter for this research task.",
        "Act as the independent alternate-candidate falsifier for this research task.",
        "Act as the independent evidence and canonical-form auditor for this research task.",
    }
    assert json.loads(canonical["function"]["arguments"]) == action.payload


def test_forced_external_step_recovers_wrong_known_action_from_public_context() -> None:
    action, _ = canonicalize_tool_call(
        {
            "function": {
                "name": "search_many",
                "arguments": '{"queries":["candidate search"]}',
            }
        },
        expected_action="ask_external_model",
        minimum_batch_size=4,
        external_context="Original question plus accumulated public evidence.",
    )

    assert action.action == "ask_external_model"
    assert len(action.payload["requests"]) == 4
    assert all(
        request["context"] == "Original question plus accumulated public evidence."
        for request in action.payload["requests"]
    )


def test_external_tool_schema_exposes_no_provider_or_generation_overrides() -> None:
    external = next(
        tool for tool in tool_schemas() if tool["function"]["name"] == "ask_external_model"
    )["function"]
    parameters = external["parameters"]
    assert parameters["required"] == ["requests"]
    assert set(parameters["properties"]) == {"requests"}
    request_properties = parameters["properties"]["requests"]["items"]["properties"]
    assert set(request_properties) == {"query", "context", "system", "task_mode"}
    assert request_properties["task_mode"]["enum"] == ["research", "strategy"]
