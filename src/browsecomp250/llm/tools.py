from __future__ import annotations

from typing import Any


def tool_schemas(*, include_external_model: bool = True) -> list[dict[str, Any]]:
    def tool(name: str, description: str, properties: dict[str, Any], required: list[str]):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    external_request_properties = {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200000,
            "description": "A self-contained question for the independent model.",
        },
        "context": {"type": "string", "maxLength": 200000},
        "system": {"type": "string", "maxLength": 32000},
    }

    tools = [
        tool(
            "search",
            "Search the public web.",
            {
                "query": {"type": "string", "minLength": 1},
                "count": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            ["query"],
        ),
        tool(
            "search_many",
            "Run several web searches.",
            {
                "queries": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
                "count": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            ["queries"],
        ),
        tool(
            "open",
            "Fetch and extract a web page.",
            {
                "url": {"type": "string", "minLength": 1},
                "offset": {"type": "integer", "minimum": 0},
                "max_chars": {"type": "integer", "minimum": 1000},
            },
            ["url"],
        ),
        tool(
            "open_many",
            "Fetch and extract several web pages.",
            {
                "urls": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
                "offset": {"type": "integer", "minimum": 0},
                "max_chars": {"type": "integer", "minimum": 1000},
            },
            ["urls"],
        ),
        tool(
            "find",
            "Find a text pattern within an already opened page.",
            {
                "url": {"type": "string", "minLength": 1},
                "pattern": {"type": "string", "minLength": 1},
            },
            ["url", "pattern"],
        ),
        tool(
            "geo_search",
            (
                "Geocode up to four landmarks or addresses, enumerate nearby named places from "
                "OpenStreetMap, and estimate pedestrian route distances. Use this instead of "
                "guessing when a question depends on proximity, walking distance, or matching "
                "the same business across locations."
            ),
            {
                "anchors": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "minLength": 1},
                            "radius_m": {
                                "type": "integer",
                                "minimum": 100,
                                "maximum": 25000,
                                "default": 5000,
                            },
                            "expected_distance_miles": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 500,
                                "description": (
                                    "Route distance stated for this anchor, converted to miles. "
                                    "Supply it whenever the question gives a distance."
                                ),
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "restaurant",
                        "food",
                        "lodging",
                        "retail",
                        "attraction",
                        "named_place",
                    ],
                    "default": "named_place",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 50,
                },
                "include_walking_routes": {"type": "boolean", "default": True},
            },
            ["anchors"],
        ),
        tool(
            "ask_external_model",
            (
                "Ask an independent external model for difficult reasoning, current facts, or "
                "critique. Supply one to four independent consultations in requests. Continue "
                "researching after reading the answers; external responses are evidence, not "
                "authority. Routing and generation settings are selected by the deployment."
            ),
            {
                "requests": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": external_request_properties,
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            ["requests"],
        ),
        tool(
            "note",
            "Save a compact research note.",
            {"text": {"type": "string"}},
            ["text"],
        ),
        tool(
            "final",
            (
                "Submit one concrete final short answer. Missing evidence lowers confidence but "
                "must not produce an abstention or meta-answer."
            ),
            {
                "explanation": {"type": "string"},
                "exact_answer": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 100},
                "citations": {"type": "array", "items": {"type": "string"}},
            },
            ["explanation", "exact_answer", "confidence", "citations"],
        ),
    ]
    if not include_external_model:
        tools = [
            item for item in tools if item.get("function", {}).get("name") != "ask_external_model"
        ]
    return tools
