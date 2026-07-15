import pytest

from browsecomp250.llm.protocol import ProtocolError, action_from_tool_call, parse_json_action


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
