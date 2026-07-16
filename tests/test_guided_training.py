from __future__ import annotations

from pathlib import Path

import pytest

from browsecomp250.agent.runner import SCRIPTED_FINAL_SYSTEM_PROMPT
from browsecomp250.config import load_config
from browsecomp250.guided_training import (
    audit_system_messages,
    compile_guided_steps,
    controller_label_leaks,
    curate_scripted_training_messages,
    training_message_quality,
)
from browsecomp250.question_planning import (
    compile_question_discovery_profile,
    infer_answer_contract,
)
from scripts.run_full_guided_training import worker_resources


def _records(*, sources: bool = True):
    route = {
        "question_model": {
            "answer_type": "person",
            "answer_cardinality": 1,
            "topology": ["intersection"],
            "lexical_anchors": ["rare date"],
            "source_targets": ["archive"],
        },
        "constraints": [
            {
                "constraint_id": "C01",
                "original_text": "The person was documented in 1901.",
                "verification_rule": "Direct dated source.",
            }
        ],
        "workers": [
            {"role": "anchor", "assignment": "Find the date.", "seed_queries": ["1901"]}
        ],
        "steps": [
            {
                "step_id": "S004",
                "tool_call": {"tool": "search", "arguments": {"queries": ["rare 1901"]}},
            },
            {
                "step_id": "S005",
                "tool_call": {"tool": "search", "arguments": {"queries": ["archive 1901"]}},
            },
            {
                "step_id": "S006",
                "tool_call": {"tool": "search", "arguments": {"queries": ["site:x 1901"]}},
            },
            {
                "step_id": "S014",
                "tool_call": {
                    "tool": "search",
                    "arguments": {"queries": ['"${candidate}" contradiction']},
                },
            },
        ],
    }
    evidence_sources = (
        [
            {
                "source_id": "SRC-1",
                "role": "gold",
                "title": "Secret Person archive",
                "url": "https://example.test/secret-person",
                "overview_excerpt": "Secret Person was documented in 1901.",
            }
        ]
        if sources
        else []
    )
    full = {
        **route,
        "oracle": {
            "gold_answer": "Secret Person",
            "comparison_aliases": ["S. Person"],
            "answer_conditioned_verification_queries": [
                '"Secret Person" 1901 archive',
            ],
            "evidence_sources": evidence_sources,
            "constraint_evidence_matrix": (
                [
                    {
                        "constraint_id": "C01",
                        "required_agent_action": "Verify Secret Person in the archive.",
                        "evidence": [
                            {
                                "source_id": "SRC-1",
                                "url": "https://example.test/secret-person",
                            }
                        ],
                    }
                ]
                if sources
                else []
            ),
        },
    }
    return route, full


def test_compiler_redacts_private_label_but_keeps_public_source_url() -> None:
    route, full = _records()
    steps, review = compile_guided_steps(route, full, attempt=2)

    assert controller_label_leaks(
        question="Who was documented in 1901?",
        oracle_record=full,
        steps=steps,
        review_guidance=review,
    ) == []
    assert "Secret Person" not in "\n".join(step["instruction"] for step in steps)
    assert "${candidate}" in "\n".join(step["instruction"] for step in steps)
    assert any(
        step.get("required_urls") == ["https://example.test/secret-person"] for step in steps
    )
    open_step = next(step for step in steps if step["id"] == "open_mapped_source_01")
    assert "Secret Person" not in open_step["instruction"]
    assert "${candidate}" not in open_step["instruction"]
    assert "open with exactly this URL: https://example.test/secret-person" in open_step[
        "instruction"
    ]
    assert any(step["id"] == "mapped_evidence_ledger" for step in steps)
    assert all(step["id"] != "stop_gate_audit" for step in steps)
    recovery = next(step for step in steps if step["id"] == "answer_redacted_passage_recovery")
    assert "Secret Person" not in recovery["instruction"]
    assert "query_hints" in recovery["instruction"]
    assert "required_queries" not in recovery
    assert any(step["id"] == "answer_redacted_recovery_source_inspection" for step in steps)
    assert any(step["id"] == "helper_source_inspection" for step in steps)
    assert any(step["id"] == "candidate_gap_source_inspection" for step in steps)
    assert any(step["id"] == "candidate_verification_source_inspection" for step in steps)
    assert any(step["id"] == "pre_final_repair_source_inspection" for step in steps)
    assert steps[-1]["allowed_actions"] == ["final"]


def test_label_checker_ignores_public_url_and_does_not_match_word_substrings() -> None:
    route, full = _records()
    full["oracle"]["gold_answer"] = "Yes"
    full["oracle"]["comparison_aliases"] = []
    full["oracle"]["evidence_sources"][0]["url"] = (
        "https://en.wikipedia.org/wiki/Yes_(band)"
    )
    steps, review = compile_guided_steps(route, full)

    assert controller_label_leaks(
        question="Whose eyes appeared in the archive?",
        oracle_record=full,
        steps=steps,
        review_guidance=review,
    ) == []


def test_label_checker_ignores_procedural_number_matching_numeric_answer() -> None:
    route, full = _records()
    full["oracle"]["gold_answer"] = "2"
    full["oracle"]["comparison_aliases"] = []
    steps, review = compile_guided_steps(route, full)
    steps[0]["instruction"] += " Open mapped source 2/9."

    assert controller_label_leaks(
        question="How many qualifying records are there?",
        oracle_record=full,
        steps=steps,
        review_guidance=review,
    ) == []


def test_route_only_compiler_researches_early_then_opens_helper_sources() -> None:
    route, full = _records(sources=False)
    steps, _ = compile_guided_steps(route, full)

    helper = next(step for step in steps if step["id"] == "independent_research_helper")
    assert helper["minimum_batch_size"] == 4
    helper_index = steps.index(helper)
    helper_open = next(step for step in steps if step["id"] == "helper_source_inspection")
    candidate_search = next(step for step in steps if step["id"] == "candidate_gap_search")
    candidate_open = next(
        step for step in steps if step["id"] == "candidate_gap_source_inspection"
    )
    assert helper_index == 2
    assert helper_index < steps.index(helper_open) < steps.index(candidate_search)
    assert steps.index(candidate_search) < steps.index(candidate_open)
    assert all(not step["id"].startswith("guide_search_rung_") for step in steps)
    discovery = steps[1]["instruction"]
    assert '"Tuesday"' not in discovery
    assert '"seed_queries"' not in discovery
    review = next(step for step in steps if step["id"] == "pre_final_adversarial_review")
    assert review["minimum_batch_size"] == 2


def test_terminal_clock_request_overrides_incidental_geography_words() -> None:
    question = (
        "An urban planner worked on a boulevard in a European city. Fewer than 21 students "
        "later toured the city with the planner. Using the 12-hour clock format, what time "
        "did they reach their first stop?"
    )

    contract = infer_answer_contract(question, fallback_type="place")

    assert contract["answer_type"] == "time"
    assert contract["terminal_ask"].startswith("Using the 12-hour")


def test_directive_sentence_wins_over_name_mentioned_inside_later_clue() -> None:
    question = (
        "Using independent sources, provide the name of the mystery settlement: 1. A fire "
        "occurred nearby. 2. A hospital shares a name with the settlement."
    )

    contract = infer_answer_contract(question, fallback_type="other_short_string")

    assert contract["answer_type"] == "place"
    assert contract["terminal_ask"].startswith("Using independent sources")


def test_question_first_profile_rejects_generic_route_anchors_and_geography_bias() -> None:
    question = (
        "A person was interviewed on the first Tuesday between 2008 and 2018. The inquiry "
        "guide had nine questions about an urban planner and a boulevard. Less than a year "
        "later, fewer than 21 students toured the city with that planner. Using the 12-hour "
        "clock format, what time did they reach the first stop?"
    )
    profile = compile_question_discovery_profile(
        question,
        topic="Geography",
        route_question_model={
            "answer_type": "place",
            "answer_cardinality": {"minimum": 1, "maximum": 1, "ordered": False},
            "lexical_anchors": ["Tuesday", "Less", "These", "Using"],
            "source_targets": ["GeoNames", "gazetteers"],
        },
        route_queries=[
            ['"Tuesday"', '"Less"', '"These"'],
            ['"Tuesday" "Less"'],
            ['site:geonames.org "Tuesday"'],
        ],
    )

    assert profile["answer_type"] == "time"
    assert not {"Tuesday", "Less", "These", "Using"} & set(
        profile["question_first_anchors"]
    )
    assert "university course and studio archives" in profile["source_targets"]
    assert "GeoNames" not in profile["source_targets"]
    all_queries = [query for rung in profile["query_rungs"] for query in rung]
    assert all(query not in {'"Tuesday"', '"Less"', '"These"'} for query in all_queries)
    assert any("nine questions" in query.casefold() for query in all_queries)
    assert not any(
        '"person interviewed first Tuesday particular month 2008 2018"' in query
        for query in all_queries
    )


def test_document_and_designer_question_uses_native_document_sources_not_art_museums() -> None:
    question = (
        "A global health report credited a foreword author and an introduction author. Its "
        "cover designer studied publishing and previously worked at two agencies. What is the "
        "first and last name of the cover designer of this report?"
    )
    profile = compile_question_discovery_profile(
        question,
        topic="Art",
        route_question_model={
            "answer_type": "person",
            "source_targets": ["museum collection catalogs", "auction records"],
        },
        route_queries=[[], [], []],
    )

    assert profile["answer_type"] == "person"
    assert "report colophons and publisher credits" in profile["source_targets"]
    assert "designer portfolios and professional biographies" in profile["source_targets"]
    assert "museum collection catalogs" not in profile["source_targets"]


@pytest.mark.asyncio
async def test_guided_worker_and_star2_helpers_fail_closed_on_unsupported_finals(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/star-headline.yaml"))
    config.search.cache_path = tmp_path / "search.sqlite3"
    config.browser.cache_path = tmp_path / "pages.sqlite3"

    resources = worker_resources(
        0,
        config,
        star7_endpoint="http://star7.test/v1",
        star2_endpoint="http://star2.test/v1",
    )
    try:
        assert resources.config.agent.require_opened_citation_support is True
        assert resources.config.agent.allow_unsupported_final_at_hard_budget is False
        assert resources.external_model.agent_config.require_opened_citation_support is True
        assert resources.external_model.agent_config.allow_unsupported_final_at_hard_budget is False
    finally:
        await resources.close()


def test_compiler_adds_geo_verification_for_multiple_distance_constraints() -> None:
    route, full = _records(sources=False)
    route["constraints"].extend(
        [
            {
                "constraint_id": "C02",
                "original_text": "The place is 3 miles driving from a hotel.",
                "verification_rule": "Check the driving route.",
            },
            {
                "constraint_id": "C03",
                "original_text": "The place is 40 meters from a station.",
                "verification_rule": "Check the distance.",
            },
        ]
    )
    full["constraints"] = route["constraints"]

    steps, _ = compile_guided_steps(route, full)

    geo = next(step for step in steps if step["id"] == "geospatial_verification")
    assert geo["allowed_actions"] == ["geo_search"]
    assert steps.index(geo) < len(steps) - 1


def test_compiler_detects_multiple_distances_inside_one_constraint() -> None:
    route, full = _records(sources=False)
    route["constraints"] = [
        {
            "constraint_id": "C01",
            "original_text": (
                "The target is 3 miles driving from a hotel and 40 meters from a station."
            ),
            "verification_rule": "Check both routes.",
        }
    ]
    full["constraints"] = route["constraints"]

    steps, _ = compile_guided_steps(route, full)

    assert any(step["id"] == "geospatial_verification" for step in steps)


def test_later_attempt_guidance_does_not_reveal_private_grading_feedback() -> None:
    route, full = _records()
    steps, review = compile_guided_steps(route, full, attempt=2)
    controller_text = review + "\n" + "\n".join(step["instruction"] for step in steps)

    assert "failed private grading" not in controller_text.casefold()
    assert "prior candidate" not in controller_text.casefold()


def test_curator_keeps_only_successful_guided_actions_and_reasoning() -> None:
    contract = {"id": "open", "allowed_actions": ["open"]}
    final_contract = {"id": "finalize", "allowed_actions": ["final"]}
    events = [
        {
            "event": "scripted_guidance_step_started",
            "step": 2,
            "scripted_step_index": 0,
            "scripted_step": contract,
            "scripted_guidance_role": "user",
        },
        {
            "event": "model_response",
            "step": 2,
            "assistant_content": "",
            "assistant_reasoning": "Use the required source.",
            "tool_calls": [
                {
                    "id": "call-open",
                    "function": {"name": "open", "arguments": '{"url":"https://x"}'},
                }
            ],
        },
        {
            "event": "action_completed",
            "step": 2,
            "action": "open",
            "ok": True,
            "result": {"ok": True, "page": {"url": "https://x"}},
        },
        {
            "event": "scripted_guidance_step_completed",
            "step": 2,
            "scripted_step_index": 0,
            "action_succeeded": True,
        },
        {
            "event": "scripted_guidance_step_started",
            "step": 3,
            "scripted_step_index": 1,
            "scripted_step": final_contract,
            "scripted_guidance_role": "user",
        },
        {
            "event": "scripted_guidance_final_context_built",
            "step": 3,
            "final_system": "generic final system",
            "final_user": "public evidence",
        },
        {
            "event": "model_response",
            "step": 3,
            "assistant_content": "",
            "assistant_reasoning": "The evidence identifies one answer.",
            "tool_calls": [
                {"id": "call-final", "function": {"name": "final", "arguments": "{}"}}
            ],
        },
        {"event": "trial_final", "step": 3},
    ]
    messages = curate_scripted_training_messages(
        events,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        scripted_step_count=2,
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "user",
        "assistant",
        "tool",
        "system",
        "user",
        "assistant",
    ]
    assert [
        message["reasoning"] for message in messages if message.get("reasoning")
    ] == ["Use the required source.", "The evidence identifies one answer."]


def test_system_message_audit_allows_only_invariant_prompts() -> None:
    invariant = "generic invariant prompt"
    final_system = "generic final synthesis prompt"
    audit = audit_system_messages(
        transcript=[
            {"role": "system", "content": invariant},
            {"role": "user", "content": "item-specific guide and public evidence"},
            {"role": "system", "content": final_system},
        ],
        invariant_system_prompt=invariant,
        expected_final_system_prompt=final_system,
        events=[
            {
                "event": "scripted_guidance_final_context_built",
                "final_system": final_system,
            }
        ],
    )

    assert audit["passed"] is True
    assert audit["unexpected_system_message_count"] == 0


def test_system_message_audit_rejects_item_specific_system_content() -> None:
    audit = audit_system_messages(
        transcript=[
            {"role": "system", "content": "generic invariant prompt"},
            {"role": "system", "content": "hidden item plan or label"},
        ],
        invariant_system_prompt="generic invariant prompt",
        expected_final_system_prompt=SCRIPTED_FINAL_SYSTEM_PROMPT,
        events=[],
    )

    assert audit["passed"] is False
    assert audit["unexpected_system_message_count"] == 1


def test_system_message_audit_does_not_trust_recorded_final_system() -> None:
    item_specific = "private item plan containing the hidden reference label"
    audit = audit_system_messages(
        transcript=[
            {"role": "system", "content": "generic invariant prompt"},
            {"role": "system", "content": item_specific},
        ],
        invariant_system_prompt="generic invariant prompt",
        expected_final_system_prompt=SCRIPTED_FINAL_SYSTEM_PROMPT,
        events=[
            {
                "event": "scripted_guidance_final_context_built",
                "final_system": item_specific,
            }
        ],
    )

    assert audit["passed"] is False
    assert audit["unexpected_system_message_count"] == 1
    assert audit["unexpected_recorded_final_system_count"] == 1


def _assistant_tool_action(name: str, *, reasoning: str = "") -> dict:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": f"call-{name}", "function": {"name": name, "arguments": "{}"}}
        ],
    }
    if reasoning:
        message["reasoning"] = reasoning
    return message


def test_training_quality_allows_one_transport_action_without_reasoning() -> None:
    messages = [
        _assistant_tool_action("note", reasoning="Build the constraint ledger."),
        _assistant_tool_action("open"),
        _assistant_tool_action("search_many", reasoning="Close the remaining evidence gap."),
        _assistant_tool_action("final", reasoning="Synthesize the supported answer."),
    ]

    quality = training_message_quality(messages)

    assert quality["passed"] is True
    assert quality["assistant_reasoning_ratio"] == 0.75
    assert quality["actions_without_reasoning"] == ["open"]
    assert quality["final_reasoning_present"] is True


def test_training_quality_requires_reasoning_on_final_action() -> None:
    messages = [
        _assistant_tool_action("note", reasoning="Build the constraint ledger."),
        _assistant_tool_action("open", reasoning="Inspect the mapped source."),
        _assistant_tool_action("search_many", reasoning="Falsify the candidate."),
        _assistant_tool_action("final"),
    ]

    quality = training_message_quality(messages)

    assert quality["assistant_reasoning_ratio"] == 0.75
    assert quality["final_reasoning_present"] is False
    assert quality["passed"] is False
