from __future__ import annotations

import json
import re
from typing import Any

from ..types import AgentAction

_ALLOWED_ACTIONS = {
    "search",
    "search_many",
    "open",
    "open_many",
    "find",
    "ask_external_model",
    "note",
    "final",
}


class ProtocolError(ValueError):
    pass


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ProtocolError("No valid JSON object found in model response")


def parse_json_action(text: str) -> AgentAction:
    value = _extract_json_object(text)
    action = str(value.get("action", "")).strip().lower()
    if action not in _ALLOWED_ACTIONS:
        raise ProtocolError(f"Unknown action: {action!r}")
    payload = {key: item for key, item in value.items() if key != "action"}
    _validate_payload(action, payload)
    return AgentAction(action=action, payload=payload)  # type: ignore[arg-type]


def _validate_payload(action: str, payload: dict[str, Any]) -> None:
    if action == "search":
        if not isinstance(payload.get("query"), str) or not payload["query"].strip():
            raise ProtocolError("search requires a non-empty query")
    elif action == "search_many":
        queries = payload.get("queries")
        if (
            not isinstance(queries, list)
            or not queries
            or not all(isinstance(item, str) and item.strip() for item in queries)
        ):
            raise ProtocolError("search_many requires a non-empty string list")
    elif action == "open":
        if not isinstance(payload.get("url"), str) or not payload["url"].strip():
            raise ProtocolError("open requires a URL")
    elif action == "open_many":
        urls = payload.get("urls")
        if (
            not isinstance(urls, list)
            or not urls
            or not all(isinstance(item, str) and item.strip() for item in urls)
        ):
            raise ProtocolError("open_many requires a non-empty URL list")
    elif action == "find":
        if not isinstance(payload.get("url"), str) or not isinstance(payload.get("pattern"), str):
            raise ProtocolError("find requires url and pattern")
    elif action == "ask_external_model":
        query = payload.get("query")
        requests = payload.get("requests")
        has_query = isinstance(query, str) and bool(query.strip())
        has_requests = (
            isinstance(requests, list)
            and 1 <= len(requests) <= 4
            and all(
                isinstance(item, dict)
                and isinstance(item.get("query"), str)
                and bool(item["query"].strip())
                for item in requests
            )
        )
        if has_query == has_requests:
            raise ProtocolError(
                "ask_external_model requires exactly one of query or requests (up to four)"
            )
    elif action == "note":
        if not isinstance(payload.get("text"), str) or not payload["text"].strip():
            raise ProtocolError("note requires text")
    elif action == "final":
        if not isinstance(payload.get("exact_answer"), str):
            raise ProtocolError("final requires exact_answer")
        confidence = payload.get("confidence")
        if confidence is not None:
            try:
                numeric = float(confidence)
            except (TypeError, ValueError) as exc:
                raise ProtocolError("confidence must be numeric") from exc
            if not 0 <= numeric <= 100:
                raise ProtocolError("confidence must be between 0 and 100")


def action_from_tool_call(tool_call: dict[str, Any]) -> AgentAction:
    function = tool_call.get("function") or {}
    name = str(function.get("name", "")).strip()
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            payload = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"Invalid tool arguments for {name}: {exc}") from exc
    elif isinstance(arguments, dict):
        payload = arguments
    else:
        raise ProtocolError(f"Invalid tool arguments type for {name}")
    if name not in _ALLOWED_ACTIONS:
        raise ProtocolError(f"Unknown tool: {name}")
    _validate_payload(name, payload)
    return AgentAction(action=name, payload=payload)  # type: ignore[arg-type]
