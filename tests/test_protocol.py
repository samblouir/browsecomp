import pytest

from browsecomp250.llm.protocol import ProtocolError, parse_json_action


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
