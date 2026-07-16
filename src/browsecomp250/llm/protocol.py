from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from ..types import AgentAction

_ALLOWED_ACTIONS = {
    "search",
    "search_many",
    "open",
    "open_many",
    "find",
    "geo_search",
    "ask_external_model",
    "note",
    "final",
}


class ProtocolError(ValueError):
    pass


_GEMMA_STRING_DELIMITER = '<|"|>'
_GEMMA_TOOL_CALL_SUFFIX = "<tool_call|>"
_EXTERNAL_RESEARCH_ROLES = (
    "rare-anchor solver",
    "relation-graph inverter",
    "alternate-candidate falsifier",
    "evidence and canonical-form auditor",
)


class _GemmaFunctionArgumentsParser:
    """Parse the compact function syntax occasionally leaked into a tool name."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0

    def parse(self) -> dict[str, Any]:
        self._skip_space()
        value = self._parse_object() if self._peek() == "{" else self._parse_pairs()
        self._skip_space()
        while self._peek() in {"}", ")"}:
            self.index += 1
            self._skip_space()
        if self.index != len(self.text):
            raise ProtocolError("Unexpected trailing text in serialized tool name")
        return value

    def _parse_pairs(self, closing: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while self.index < len(self.text):
            self._skip_space()
            if closing is not None and self._peek() == closing:
                self.index += 1
                return result
            if closing is None and self._peek() in {"}", ")"}:
                return result
            key = self._parse_key()
            self._skip_space()
            self._expect(":")
            result[key] = self._parse_value()
            self._skip_space()
            if self._peek() == ",":
                self.index += 1
                continue
            if closing is not None and self._peek() == closing:
                self.index += 1
                return result
            if closing is None and self._peek() in {"", "}", ")"}:
                return result
            raise ProtocolError("Expected a separator in serialized tool arguments")
        if closing is not None:
            raise ProtocolError("Unclosed object in serialized tool arguments")
        return result

    def _parse_key(self) -> str:
        self._skip_space()
        if self.text.startswith(_GEMMA_STRING_DELIMITER, self.index):
            return self._parse_gemma_string()
        if self._peek() == '"':
            value = self._parse_json_string()
            if not isinstance(value, str):
                raise ProtocolError("Serialized tool argument key must be a string")
            return value
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", self.text[self.index :])
        if match is None:
            raise ProtocolError("Invalid key in serialized tool arguments")
        self.index += len(match.group(0))
        return match.group(0)

    def _parse_value(self) -> Any:
        self._skip_space()
        if self.text.startswith(_GEMMA_STRING_DELIMITER, self.index):
            return self._parse_gemma_string()
        char = self._peek()
        if char == '"':
            return self._parse_json_string()
        if char == "[":
            return self._parse_array()
        if char == "{":
            return self._parse_object()
        match = re.match(
            r"(?:-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|true|false|null)",
            self.text[self.index :],
        )
        if match is not None:
            token = match.group(0)
            self.index += len(token)
            return json.loads(token)
        raise ProtocolError("Unsupported value in serialized tool arguments")

    def _parse_array(self) -> list[Any]:
        self._expect("[")
        values: list[Any] = []
        while True:
            self._skip_space()
            if self._peek() == "]":
                self.index += 1
                return values
            values.append(self._parse_value())
            self._skip_space()
            if self._peek() == ",":
                self.index += 1
                continue
            if self._peek() == "]":
                self.index += 1
                return values
            # Some vLLM parser failures collapse one adjacent Gemma close/open
            # delimiter into a single token. Recover the orphaned string between
            # the current position and the next delimiter, but only inside an
            # already recognized array value.
            next_delimiter = self.text.find(_GEMMA_STRING_DELIMITER, self.index)
            if next_delimiter > self.index:
                orphan = self.text[self.index : next_delimiter].strip().lstrip(",").strip()
                if orphan:
                    if isinstance(values[-1], str):
                        values[-1] = values[-1].rstrip(",").rstrip()
                    values.append(orphan)
                    self.index = next_delimiter + len(_GEMMA_STRING_DELIMITER)
                    self._skip_space()
                    if self._peek() == ",":
                        self.index += 1
                        continue
                    if self._peek() == "]":
                        self.index += 1
                        return values
            raise ProtocolError("Expected a separator in serialized tool array")

    def _parse_object(self) -> dict[str, Any]:
        self._expect("{")
        return self._parse_pairs("}")

    def _parse_gemma_string(self) -> str:
        self.index += len(_GEMMA_STRING_DELIMITER)
        end = self.text.find(_GEMMA_STRING_DELIMITER, self.index)
        if end < 0:
            raise ProtocolError("Unclosed Gemma string in serialized tool arguments")
        value = self.text[self.index : end]
        self.index = end + len(_GEMMA_STRING_DELIMITER)
        return value

    def _parse_json_string(self) -> str:
        try:
            value, length = json.JSONDecoder().raw_decode(self.text[self.index :])
        except json.JSONDecodeError as exc:
            raise ProtocolError("Invalid JSON string in serialized tool arguments") from exc
        if not isinstance(value, str):
            raise ProtocolError("Expected a string in serialized tool arguments")
        self.index += length
        return value

    def _expect(self, expected: str) -> None:
        self._skip_space()
        if self._peek() != expected:
            raise ProtocolError(f"Expected {expected!r} in serialized tool arguments")
        self.index += 1

    def _peek(self) -> str:
        return self.text[self.index] if self.index < len(self.text) else ""

    def _skip_space(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1


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
    elif action == "geo_search":
        anchors = payload.get("anchors")
        if (
            not isinstance(anchors, list)
            or not 1 <= len(anchors) <= 4
            or not all(
                isinstance(item, dict)
                and isinstance(item.get("query"), str)
                and bool(item["query"].strip())
                for item in anchors
            )
        ):
            raise ProtocolError("geo_search requires one to four query anchors")
        for item in anchors:
            expected = item.get("expected_distance_miles")
            if expected is not None and (
                not isinstance(expected, (int, float))
                or isinstance(expected, bool)
                or not 0 <= float(expected) <= 500
            ):
                raise ProtocolError("geo_search expected_distance_miles must be between 0 and 500")
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


def _serialized_name_payload(name: str) -> tuple[str, dict[str, Any]] | None:
    if "(" not in name:
        return None
    candidate, serialized = name.split("(", 1)
    candidate = candidate.strip()
    if candidate not in _ALLOWED_ACTIONS:
        return None
    serialized = serialized.strip()
    if serialized.endswith(_GEMMA_TOOL_CALL_SUFFIX):
        serialized = serialized[: -len(_GEMMA_TOOL_CALL_SUFFIX)].rstrip()
    payload = _GemmaFunctionArgumentsParser(serialized).parse()
    return candidate, payload


def _decoded_arguments(function: dict[str, Any], name: str) -> dict[str, Any]:
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            payload = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"Invalid tool arguments for {name}: {exc}") from exc
    elif isinstance(arguments, dict):
        payload = deepcopy(arguments)
    else:
        raise ProtocolError(f"Invalid tool arguments type for {name}")
    if not isinstance(payload, dict):
        raise ProtocolError(f"Tool arguments for {name} must be an object")
    return payload


def _normalize_external_requests(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {"query", "context", "system", "task_mode"}
    requests = payload.get("requests")
    valid_requests = (
        isinstance(requests, list)
        and bool(requests)
        and all(isinstance(item, dict) for item in requests)
    )
    if valid_requests:
        common = {
            key: payload[key]
            for key in ("context", "system", "task_mode")
            if key in payload
        }
        normalized_requests = []
        for raw_request in requests[:4]:
            request = {key: value for key, value in raw_request.items() if key in allowed}
            for key, value in common.items():
                request.setdefault(key, value)
            normalized_requests.append(request)
        return {"requests": normalized_requests}
    return {key: value for key, value in payload.items() if key in allowed}


def _ensure_external_request_batch(
    payload: dict[str, Any],
    *,
    minimum_batch_size: int | None,
    fallback_context: str | None,
) -> dict[str, Any]:
    if not isinstance(minimum_batch_size, int) or minimum_batch_size <= 1:
        return payload
    target = min(4, minimum_batch_size)
    raw_requests = payload.get("requests")
    request_values = raw_requests if isinstance(raw_requests, list) else []
    requests = [
        dict(item)
        for item in request_values
        if isinstance(item, dict)
        and isinstance(item.get("query"), str)
        and bool(item["query"].strip())
    ]
    if len(requests) >= target:
        return {"requests": requests[:4]}

    context_candidates = [
        str(item.get("context") or "").removesuffix("\nquery:").strip()
        for item in request_values
        if isinstance(item, dict) and str(item.get("context") or "").strip()
    ]
    if fallback_context and fallback_context.strip():
        context_candidates.append(fallback_context.strip())
    context = max(context_candidates, key=len, default="")
    existing_text = "\n".join(
        str(value)
        for request in requests
        for value in (request.get("query"), request.get("system"))
        if value
    ).casefold()
    for role in _EXTERNAL_RESEARCH_ROLES:
        if len(requests) >= target:
            break
        if role.casefold() in existing_text:
            continue
        requests.append(
            {
                "system": f"Act as the independent {role} for this research task.",
                "query": (
                    f"As the {role}, independently identify one specific candidate, test the "
                    "complete clue chain, state unresolved gaps, and return public citation URLs."
                ),
                "context": context,
            }
        )
    while len(requests) < target:
        role = _EXTERNAL_RESEARCH_ROLES[len(requests) % len(_EXTERNAL_RESEARCH_ROLES)]
        requests.append(
            {
                "query": (
                    f"Independently re-check the supplied research task as the {role}; return a "
                    "specific candidate, decisive evidence, counterevidence, and public URLs."
                ),
                "context": context,
            }
        )
    return {"requests": requests[:4]}


def canonicalize_tool_call(
    tool_call: dict[str, Any],
    *,
    expected_action: str | None = None,
    required_queries: list[str] | None = None,
    required_urls: list[str] | None = None,
    minimum_batch_size: int | None = None,
    external_context: str | None = None,
) -> tuple[AgentAction, dict[str, Any]]:
    """Return a validated action and the canonical OpenAI tool-call representation."""

    canonical = deepcopy(tool_call)
    function = canonical.get("function")
    if not isinstance(function, dict):
        raise ProtocolError("Tool call is missing a function object")
    raw_name = str(function.get("name", "")).strip()
    recovered = _serialized_name_payload(raw_name)
    name = recovered[0] if recovered is not None else raw_name
    payload = _decoded_arguments(function, name)
    if recovered is not None:
        recovered_payload = recovered[1]
        payload = {**recovered_payload, **payload} if payload else recovered_payload
    if name not in _ALLOWED_ACTIONS:
        raise ProtocolError(f"Unknown tool: {raw_name}")

    # Reasoning models occasionally pair a singular tool name with its batch
    # argument (or vice versa). The intended operation is unambiguous, so
    # normalize the shape instead of discarding a useful evidence request.
    if name == "search" and "query" not in payload and isinstance(payload.get("queries"), list):
        name = "search_many"
    elif name == "search" and "query" not in payload and isinstance(payload.get("queries"), str):
        payload = {**payload, "query": payload["queries"]}
        payload.pop("queries", None)
    elif (
        name == "search_many" and "queries" not in payload and isinstance(payload.get("query"), str)
    ):
        name = "search"
    elif name == "open" and "url" not in payload and isinstance(payload.get("urls"), list):
        name = "open_many"
    elif name == "open" and "url" not in payload and isinstance(payload.get("urls"), str):
        payload = {**payload, "url": payload["urls"]}
        payload.pop("urls", None)
    elif name == "open_many" and "urls" not in payload and isinstance(payload.get("url"), str):
        name = "open"
    if expected_action in _ALLOWED_ACTIONS:
        related = {
            frozenset({"search", "search_many"}),
            frozenset({"open", "open_many"}),
        }
        if name == expected_action or frozenset({name, expected_action}) in related:
            name = expected_action
    if expected_action == "search" and required_queries:
        name = expected_action
        payload["query"] = str(required_queries[0])
        payload.pop("queries", None)
        payload.pop("url", None)
        payload.pop("urls", None)
    elif expected_action == "search_many" and required_queries:
        name = expected_action
        payload["queries"] = [str(value) for value in required_queries]
        payload.pop("query", None)
        payload.pop("url", None)
        payload.pop("urls", None)
    elif expected_action == "open" and required_urls:
        name = expected_action
        payload["url"] = str(required_urls[0])
        payload.pop("urls", None)
        payload.pop("query", None)
        payload.pop("queries", None)
    elif expected_action == "open_many" and required_urls:
        name = expected_action
        payload["urls"] = [str(value) for value in required_urls]
        payload.pop("url", None)
        payload.pop("query", None)
        payload.pop("queries", None)
    elif expected_action == "ask_external_model" and isinstance(minimum_batch_size, int):
        name = expected_action
    if name == "ask_external_model":
        payload = _normalize_external_requests(payload)
        payload = _ensure_external_request_batch(
            payload,
            minimum_batch_size=minimum_batch_size,
            fallback_context=external_context,
        )
    _validate_payload(name, payload)
    action = AgentAction(action=name, payload=payload)  # type: ignore[arg-type]
    function["name"] = name
    function["arguments"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    canonical["function"] = function
    return action, canonical


def action_from_tool_call(tool_call: dict[str, Any]) -> AgentAction:
    action, _ = canonicalize_tool_call(tool_call)
    return action
