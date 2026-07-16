from __future__ import annotations

import pytest

from browsecomp250.oracle_sources import (
    _is_common_numeric_answer,
    _question_evidence_terms,
    apply_redacted_source_cache,
    discover_redacted_public_sources,
    private_source_queries,
    redact_private_answer,
)
from browsecomp250.search.base import SearchError


def _record() -> dict:
    return {
        "item": {
            "item_id": "item-1",
            "topic": "History",
            "question_text": (
                "A rare archive records a 1912 expedition and a later museum catalog. "
                "Who led the expedition?"
            ),
        },
        "question_model": {
            "answer_type": "person",
            "lexical_anchors": ["rare archive", "1912 expedition"],
        },
        "oracle": {
            "gold_answer": "Secret Person",
            "comparison_aliases": ["S. Person"],
            "answer_conditioned_verification_queries": [
                '"Secret Person" "1912 expedition"',
            ],
            "evidence_sources": [],
        },
    }


def test_private_source_queries_use_answer_only_in_private_locator() -> None:
    queries = private_source_queries(_record())

    assert queries
    assert any("Secret Person" in query for query in queries)
    assert any("1912" in query for query in queries)


def test_public_bootstrap_text_redacts_answer_aliases() -> None:
    record = _record()

    assert redact_private_answer(
        "Secret Person, also called S. Person, led the expedition.", record
    ) == "${candidate}, also called ${candidate}, led the expedition."


def test_composite_answer_redacts_and_matches_identity_component() -> None:
    record = _record()
    record["oracle"]["gold_answer"] = "Rosalea Murphy, 1912"
    record["oracle"]["comparison_aliases"] = []

    assert redact_private_answer(
        "Rosalea Murphy founded the restaurant and was born in 1912.", record
    ) == "${candidate} founded the restaurant and was born in ${candidate}."


def test_evidence_terms_exclude_answer_and_generic_terminal_words() -> None:
    terms = _question_evidence_terms(
        "What time did 18 students tour the boulevard with an urban planner?",
        ["12:30 PM"],
    )

    assert "time" not in terms
    assert "12" not in terms
    assert {"students", "tour", "boulevard", "urban", "planner"} <= terms


def test_common_numeric_answers_receive_stricter_local_context_gate() -> None:
    assert _is_common_numeric_answer(["12:30 PM"])
    assert _is_common_numeric_answer(["42 percent"])
    assert not _is_common_numeric_answer(["Red Lake"])


def test_source_cache_applies_only_to_rows_without_mapped_sources() -> None:
    empty = _record()
    mapped = _record()
    mapped["item"]["item_id"] = "item-2"
    mapped["oracle"]["evidence_sources"] = [{"url": "https://original.example"}]
    cache = {
        "rows": {
            "item-1": {"sources": [{"url": "https://bootstrap.example"}]},
            "item-2": {"sources": [{"url": "https://wrong.example"}]},
        }
    }

    enriched, applied = apply_redacted_source_cache([empty, mapped], cache)

    assert applied == 1
    assert enriched[0]["oracle"]["evidence_sources"] == [
        {"url": "https://bootstrap.example"}
    ]
    assert enriched[1]["oracle"]["evidence_sources"] == [
        {"url": "https://original.example"}
    ]
    assert empty["oracle"]["evidence_sources"] == []


@pytest.mark.asyncio
async def test_source_bootstrap_distinguishes_provider_failure_from_no_results() -> None:
    class FailedSearch:
        async def search_many(self, queries, *, count):
            del count
            return [SearchError("HTTP 402") for _ in queries]

    class UnusedBrowser:
        async def fetch(self, url):
            raise AssertionError(f"unexpected fetch: {url}")

    sources, metrics = await discover_redacted_public_sources(
        _record(),
        search=FailedSearch(),
        browser=UnusedBrowser(),
    )

    assert sources == []
    assert metrics["search_error_count"] == metrics["query_count"]
    assert metrics["search_error_types"] == {"SearchError": metrics["query_count"]}
