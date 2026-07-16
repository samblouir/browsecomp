from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from .question_planning import compile_question_discovery_profile
from .util import canonical_json

_CANDIDATE_PLACEHOLDER = "${candidate}"
_URL_TEXT = re.compile(r"https?://[^\s\"'<>]+", re.I)
_GEO_CONSTRAINT = re.compile(
    r"\b(?:drive|driving|walk|walking|bicycle|route|distance|proximity|nearby|"
    r"within\s+\d|\d+(?:\.\d+)?\s*(?:miles?|mi|kilometers?|km|meters?|metres?|m)\b)",
    re.I,
)


def _answer_aliases(record: dict[str, Any]) -> list[str]:
    oracle = record.get("oracle") or {}
    values = [oracle.get("gold_answer"), *(oracle.get("comparison_aliases") or [])]
    aliases = {str(value).strip() for value in values if str(value).strip()}
    return sorted(aliases, key=len, reverse=True)


def _phrase_pattern(value: str) -> re.Pattern[str]:
    prefix = r"(?<!\w)" if value[:1].isalnum() else ""
    suffix = r"(?!\w)" if value[-1:].isalnum() else ""
    return re.compile(prefix + re.escape(value) + suffix, re.I)


def _contains_phrase(text: str, value: str) -> bool:
    return bool(value and _phrase_pattern(value).search(text))


def redact_oracle_text(text: str, record: dict[str, Any]) -> str:
    """Remove private answer strings while retaining the surrounding teacher clue."""
    redacted = str(text)
    for alias in _answer_aliases(record):
        redacted = _phrase_pattern(alias).sub(_CANDIDATE_PLACEHOLDER, redacted)
    return redacted


def _step_by_id(record: dict[str, Any], step_id: str) -> dict[str, Any] | None:
    return next((step for step in record.get("steps") or [] if step.get("step_id") == step_id), None)


def _tool_queries(record: dict[str, Any], step_id: str) -> list[str]:
    step = _step_by_id(record, step_id) or {}
    tool_call = step.get("tool_call") or {}
    arguments = tool_call.get("arguments") or {}
    return [str(value).strip() for value in arguments.get("queries") or [] if str(value).strip()]


def _source_constraint_ids(record: dict[str, Any]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    oracle = record.get("oracle") or {}
    for cell in oracle.get("constraint_evidence_matrix") or []:
        constraint_id = str(cell.get("constraint_id") or "").strip()
        for evidence in cell.get("evidence") or []:
            source_id = str(evidence.get("source_id") or "").strip()
            if source_id and constraint_id:
                mapping.setdefault(source_id, []).append(constraint_id)
    return {key: list(dict.fromkeys(values)) for key, values in mapping.items()}


def _source_evidence_hints(record: dict[str, Any]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    oracle = record.get("oracle") or {}
    for source in oracle.get("evidence_sources") or []:
        source_id = str(source.get("source_id") or "").strip()
        excerpt = str(source.get("overview_excerpt") or "").strip()
        if source_id and excerpt:
            mapping.setdefault(source_id, []).append(
                redact_oracle_text(excerpt, record)[:2_000]
            )
    for cell in oracle.get("constraint_evidence_matrix") or []:
        for evidence in cell.get("evidence") or []:
            source_id = str(evidence.get("source_id") or "").strip()
            excerpt = str(evidence.get("excerpt") or "").strip()
            if not source_id or not excerpt:
                continue
            redacted = redact_oracle_text(excerpt, record)
            mapping.setdefault(source_id, []).append(redacted[:2_000])
    return {
        key: list(dict.fromkeys(values))[:3]
        for key, values in mapping.items()
    }


def _candidate_recovery_queries(
    source_hints: dict[str, list[str]], record: dict[str, Any]
) -> list[str]:
    queries: list[str] = []
    for hints in source_hints.values():
        for hint in hints:
            if _CANDIDATE_PLACEHOLDER not in hint:
                continue
            before, after = hint.split(_CANDIDATE_PLACEHOLDER, 1)
            raw_after_tokens = re.findall(r"[\w]+(?:[-'][\w]+)*", after, flags=re.UNICODE)
            after_tokens = list(raw_after_tokens)
            while after_tokens and re.fullmatch(r"\d+(?:-\d+)+", after_tokens[0]):
                after_tokens.pop(0)
            before_tokens = re.findall(r"[\w]+(?:[-'][\w]+)*", before, flags=re.UNICODE)
            context_tokens = before_tokens[-8:] + raw_after_tokens[:14]
            if len(context_tokens) >= 6:
                context_query = " ".join(context_tokens)
                if _CANDIDATE_PLACEHOLDER not in redact_oracle_text(context_query, record):
                    queries.append(context_query)
            phrase_tokens = after_tokens[:12]
            if len(phrase_tokens) < 4:
                phrase_tokens = before_tokens[-12:]
            if len(phrase_tokens) < 4:
                continue
            query = '"' + " ".join(phrase_tokens) + '"'
            if _CANDIDATE_PLACEHOLDER in redact_oracle_text(query, record):
                continue
            if query not in queries:
                queries.append(query)
            if len(queries) >= 7:
                return queries
    return queries


def _requires_geo_verification(record: dict[str, Any]) -> bool:
    constraints = record.get("constraints") or []
    matches = 0
    for constraint in constraints:
        text = " ".join(
            str(constraint.get(key) or "")
            for key in ("original_text", "normalized_claim", "verification_rule")
        )
        matches += len(_GEO_CONSTRAINT.findall(text))
    return matches >= 2


def compile_guided_steps(
    route_record: dict[str, Any],
    oracle_record: dict[str, Any],
    *,
    attempt: int = 1,
) -> tuple[list[dict[str, Any]], str]:
    """Compile one private guide into answer-redacted, one-action Star turns."""
    constraints = route_record.get("constraints") or []
    question_model = route_record.get("question_model") or {}
    item = oracle_record.get("item") or route_record.get("item") or {}
    question = str(item.get("question_text") or question_model.get("normalized_question") or "")
    if not question:
        question = ". ".join(
            str(value.get("original_text") or "").strip()
            for value in constraints
            if str(value.get("original_text") or "").strip()
        )
    route_query_rungs = [
        _tool_queries(route_record, step_id)[:7]
        for step_id in ("S004", "S005", "S006")
    ]
    discovery_profile = compile_question_discovery_profile(
        question,
        topic=str(item.get("topic") or "Other"),
        route_question_model=question_model,
        route_queries=route_query_rungs,
    )
    steps: list[dict[str, Any]] = [
        {
            "id": "constraint_audit",
            "plan_step_ids": ["S001"],
            "instruction": (
                "Audit the question into an immutable constraint ledger. Preserve answer type, "
                "cardinality, dates, counts, role direction, negation, and pronoun attachment. "
                "Save one compact note containing supported/unknown/refuted cells; do not answer yet.\n"
                + canonical_json(
                    {
                        "answer_type": discovery_profile["answer_type"],
                        "answer_cardinality": discovery_profile["answer_cardinality"],
                        "answer_type_inference": discovery_profile["answer_type_inference"],
                        "terminal_ask": discovery_profile["terminal_ask"],
                        "constraints": [
                            {
                                "id": value.get("constraint_id"),
                                "clue": value.get("original_text"),
                                "verification_rule": value.get("verification_rule"),
                            }
                            for value in constraints
                        ],
                    }
                )
            ),
            "allowed_actions": ["note"],
        },
        {
            "id": "discovery_plan",
            "plan_step_ids": ["S002", "S003"],
            "instruction": (
                "Create and save one compact discovery plan with distinct rare-anchor, "
                "structured-source, and alternate-candidate branches. Do not execute a search or "
                "finalize on this turn.\n"
                + canonical_json(
                    {
                        "topology": question_model.get("topology"),
                        "rare_anchors": discovery_profile["question_first_anchors"],
                        "source_targets": discovery_profile["source_targets"],
                        "source_native_terms": discovery_profile["source_native_terms"],
                        "metadata_policy": discovery_profile["route_metadata_role"],
                        "workers": [
                            {
                                "role": "rare_anchor_researcher",
                                "assignment": (
                                    "Generate named hypotheses from the rarest relation, then test "
                                    "them against primary or contemporary sources."
                                ),
                            },
                            {
                                "role": "relation_graph_inverter",
                                "assignment": (
                                    "Start from the requested terminal fact and traverse backward "
                                    "through the constrained event, person, work, or place."
                                ),
                            },
                            {
                                "role": "alternate_candidate_falsifier",
                                "assignment": (
                                    "Maintain a minimal-pair alternative and search for the single "
                                    "clue that distinguishes it from the leading candidate."
                                ),
                            },
                        ],
                    }
                )
            ),
            "allowed_actions": ["note"],
        },
    ]

    oracle = oracle_record.get("oracle") or {}
    sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for source in oracle.get("evidence_sources") or []:
        url = str(source.get("url") or "").strip()
        normalized = url.rstrip("/").casefold()
        if not url or normalized in seen_urls:
            continue
        seen_urls.add(normalized)
        sources.append(source)

    source_constraints = _source_constraint_ids(oracle_record)
    source_hints = _source_evidence_hints(oracle_record)
    emitted_recovery_queries: set[str] = set()
    recovery_queries: list[str] = []
    recovery_contexts = [
        hint
        for hints in source_hints.values()
        for hint in hints
        if _CANDIDATE_PLACEHOLDER in hint
    ][:7]
    for index, source in enumerate(sources, start=1):
        source_id = str(source.get("source_id") or f"source-{index}")
        source_url = str(source["url"])
        title = redact_oracle_text(str(source.get("title") or "mapped source"), oracle_record)
        if _CANDIDATE_PLACEHOLDER in title:
            title = "answer-redacted mapped source title"
        assigned = source_constraints.get(source_id) or []
        steps.append(
            {
                "id": f"open_mapped_source_{index:02d}",
                "plan_step_ids": ["S004", "S005", "S006", "S011"],
                "instruction": (
                    f"Open mapped source {index}/{len(sources)} now by calling open with exactly "
                    f"this URL: {source_url}. Do not search for it, substitute another page, or "
                    "skip it even if it seems redundant or inaccessible. This is a transport-only "
                    "turn: call open immediately and defer synthesis, candidate comparison, and "
                    "constraint reassessment to the later ledger turns. Inspect the actual page "
                    "for direct evidence and identity facts; snippets are leads only. Do not "
                    "finalize. "
                    + canonical_json(
                        {
                            "source_role": source.get("role"),
                            "title_hint": title,
                            "assigned_constraint_ids": assigned,
                        }
                    )
                ),
                "allowed_actions": ["open"],
                "required_urls": [source_url],
                "advance_on_attempt": True,
            }
        )
        source_recovery_queries = [
            query
            for query in _candidate_recovery_queries(
                {source_id: source_hints.get(source_id) or []},
                oracle_record,
            )
            if query not in emitted_recovery_queries
        ]
        if source_recovery_queries:
            emitted_recovery_queries.update(source_recovery_queries)
            recovery_queries.extend(source_recovery_queries)

    if recovery_queries:
        steps.append(
            {
                "id": "answer_redacted_passage_recovery",
                "plan_step_ids": ["S006", "S010", "S021", "S022", "S023"],
                "instruction": (
                    "Run one flexible batched search after completing the mapped-source sweep. "
                    "The query hints below quote words surrounding an answer removed from guide "
                    "passages; the missing name is not supplied. Start from the best hints, but "
                    "reformulate with roles, dates, rare relations, source domains, and entities "
                    "already identified when exact quotes are unproductive. Recover and verify "
                    "the name from public results, and do not finalize.\n"
                    + canonical_json(
                        {
                            "query_hints": recovery_queries[:7],
                            "answer_redacted_passage_contexts": recovery_contexts,
                        }
                    )
                ),
                "allowed_actions": ["search_many"],
                "advance_on_attempt": True,
            }
        )
        steps.append(
            {
                "id": "answer_redacted_recovery_source_inspection",
                "plan_step_ids": ["S006", "S011", "S012"],
                "instruction": (
                    "Open the strongest independently authored public pages returned by the "
                    "answer-redacted recovery search. Use only exact URLs already present in tool "
                    "results, prefer pages that expose the candidate identity or requested terminal "
                    "fact, and exclude search pages, generated query mirrors, shops, and job pages. "
                    "Call open_many even if only one credible source remains. Do not finalize."
                ),
                "allowed_actions": ["open_many"],
                "advance_on_attempt": True,
            }
        )

    steps.append(
        {
            "id": "independent_research_helper",
            "plan_step_ids": ["S003", "S013"],
            "instruction": (
                "Launch four independent Star research agents in one ask_external_model call now, "
                "before committing to a candidate. Give every request the complete original "
                "question and all public evidence gathered so far, but assign distinct roles: "
                "rare-anchor solver, relation-graph inverter, alternate-candidate falsifier, and "
                "evidence/canonical-form auditor. Each agent must browse, open its decisive sources, "
                "propose one specific candidate, preserve the exact clue relations, report unresolved "
                "gaps, and return public citation URLs. Generation settings and routing are supplied "
                "by the deployment; do not request or name a provider or model. Do not finalize on "
                "this turn."
            ),
            "allowed_actions": ["ask_external_model"],
            "minimum_batch_size": 4,
            "advance_on_attempt": True,
        }
    )

    steps.append(
        {
            "id": "helper_source_inspection",
            "plan_step_ids": ["S006", "S011", "S012", "S013"],
            "instruction": (
                "Read all four researcher reports and call open_many on the strongest public source "
                "URLs they actually returned. Open candidate-identifying and terminal-fact sources "
                "from different evidence roles when possible. Use URLs verbatim; never invent a URL "
                "or open a generated query-mirror page. This is evidence collection, not a vote: "
                "include a minority report's source when it tests the strongest alternative. Do not "
                "finalize."
            ),
            "allowed_actions": ["open_many"],
            "advance_on_attempt": True,
        }
    )

    steps.append(
        {
            "id": "candidate_gap_search",
            "plan_step_ids": ["S004", "S005", "S006", "S016", "S021", "S022", "S023"],
            "instruction": (
                "Use the researcher reports and opened pages to run one batched candidate-specific "
                "search. Build 3-7 short queries around named candidates and the most discriminating "
                "unknown relation. Include at least one source-native or source-language route and "
                "one query that works backward from the requested terminal fact. If no credible "
                "candidate exists yet, use the rarest relation to generate named hypotheses rather "
                "than restating the whole question. Do not repeat low-yield clue bundles or finalize."
            ),
            "allowed_actions": ["search_many"],
            "advance_on_attempt": True,
        }
    )

    steps.append(
        {
            "id": "candidate_gap_source_inspection",
            "plan_step_ids": ["S011", "S012", "S016", "S021", "S022"],
            "instruction": (
                "Call open_many on the best public pages from the candidate-specific search. "
                "Prioritize a page that directly names the requested answer and separate pages that "
                "establish the identity chain. Use exact returned URLs and reject access gates, "
                "query mirrors, and pages whose title merely repeats the search. Do not finalize."
            ),
            "allowed_actions": ["open_many"],
            "advance_on_attempt": True,
        }
    )

    steps.append(
        {
            "id": "candidate_constraint_ledger",
            "plan_step_ids": ["S007", "S008", "S009"],
            "instruction": (
                "Merge the opened evidence and all researcher reports into one candidate-by-"
                "constraint ledger. Save a compact note naming the leading candidate and strongest "
                "minimal-pair alternative, with each hard constraint marked direct support, "
                "inference, unknown, or refuted. Resolve aliases before merging entities and do not "
                "promote repeated unsupported claims. Do not finalize."
            ),
            "allowed_actions": ["note"],
        }
    )

    if sources:
        steps.append(
            {
                "id": "mapped_evidence_ledger",
                "plan_step_ids": ["S007", "S008", "S009", "S011", "S012"],
                "instruction": (
                    "Review the actual mapped-source tool results and the answer-redacted passage "
                    "recovery turn. Save one compact evidence ledger that "
                    "separates direct support, contradiction, and source-access gaps. Infer the "
                    "leading candidate only from public evidence and the redacted context; never "
                    "treat ${candidate} as a supplied answer. Do not finalize."
                ),
                "allowed_actions": ["note"],
            }
        )

    templates = [
        redact_oracle_text(str(value), oracle_record)
        for value in oracle.get("answer_conditioned_verification_queries") or []
    ][:7]
    if templates:
        steps.append(
            {
                "id": "candidate_specific_verification",
                "plan_step_ids": ["S010"],
                "instruction": (
                    "Using only the leading candidate independently recovered from evidence, "
                    "replace ${candidate} in these private-guide templates and run one batched "
                    "verification search. Never treat the placeholder as evidence and do not "
                    "finalize.\n"
                    + canonical_json({"query_templates": templates})
                ),
                "allowed_actions": ["search_many"],
                "advance_on_attempt": True,
            }
        )
        steps.append(
            {
                "id": "candidate_verification_source_inspection",
                "plan_step_ids": ["S010", "S011", "S012"],
                "instruction": (
                    "Open the strongest pages returned by the candidate-specific verification "
                    "search. Use exact returned URLs and prioritize direct support for the requested "
                    "answer form plus the most fragile relation in the candidate ledger. Do not "
                    "finalize."
                ),
                "allowed_actions": ["open_many"],
                "advance_on_attempt": True,
            }
        )

    disconfirm_templates = [
        redact_oracle_text(value, oracle_record)
        for value in _tool_queries(route_record, "S014")[:7]
    ]
    steps.append(
        {
            "id": "adversarial_disconfirmation",
            "plan_step_ids": ["S013", "S014", "S015"],
            "instruction": (
                "Run one batched adversarial search for a plausible alternate candidate, identity "
                "collision, or authoritative contradiction. Substitute the current candidate into "
                "the templates and preserve exact relation direction. Do not infer contradiction "
                "from absence.\n"
                + canonical_json({"query_templates": disconfirm_templates})
            ),
            "allowed_actions": ["search_many"],
            "advance_on_attempt": True,
        }
    )
    steps.append(
        {
            "id": "adversarial_source_inspection",
            "plan_step_ids": ["S011", "S012", "S013", "S014", "S015"],
            "instruction": (
                "Open the strongest independently authored pages from the adversarial search, "
                "including any source that could falsify the current winner or support the minimal-"
                "pair alternative. Use only exact returned URLs. Distinguish a direct contradiction "
                "from missing evidence and do not finalize."
            ),
            "allowed_actions": ["open_many"],
            "advance_on_attempt": True,
        }
    )

    if attempt >= 2:
        redacted_passages = []
        for source in sources:
            excerpt = str(source.get("overview_excerpt") or "").strip()
            if excerpt:
                redacted_passages.append(redact_oracle_text(excerpt, oracle_record))
            if len(redacted_passages) >= 5:
                break
        if redacted_passages:
            steps.append(
                {
                    "id": "redacted_passage_recovery",
                    "plan_step_ids": ["S016", "S021", "S022", "S023"],
                    "instruction": (
                        "This later independent pass has additional answer-redacted source "
                        "contexts. Use them only as public-search clues. Run 2-7 searches around "
                        "their rare remaining phrases, dates, and roles to recover the candidate "
                        "from public evidence. Do not search for benchmark answers.\n"
                        + canonical_json({"redacted_source_contexts": redacted_passages})
                    ),
                    "allowed_actions": ["search_many"],
                    "advance_on_attempt": True,
                }
            )

    steps.extend(
        [
            {
                "id": "pre_final_adversarial_review",
                "plan_step_ids": ["S013", "S015", "S016", "S017", "S018"],
                "instruction": (
                    "Launch two independent Star reviewers in one ask_external_model call before "
                    "final synthesis. Give both the original question, current candidate ledger, "
                    "helper findings, and public evidence. Reviewer one must audit every hard "
                    "constraint and relation direction. Reviewer two must seek the strongest "
                    "alternative, identity collision, and canonical-answer-form error. They must "
                    "return only material blockers and concrete repair searches; minor missing "
                    "redundant corroboration is not a blocker when the requested answer is directly "
                    "supported. Do not finalize on this turn."
                ),
                "allowed_actions": ["ask_external_model"],
                "minimum_batch_size": 2,
                "advance_on_attempt": True,
            },
            {
                "id": "pre_final_repair_search",
                "plan_step_ids": ["S016", "S021", "S022", "S023", "S024"],
                "instruction": (
                    "Read the two pre-final reviews and run one batched search that repairs only "
                    "their material unresolved gaps. If neither review identifies a material gap, "
                    "run a concise candidate-plus-rarest-clue confirmation and one alternate-"
                    "candidate falsification query. Use genuinely different retrieval routes and "
                    "do not finalize on this turn."
                ),
                "allowed_actions": ["search_many"],
                "advance_on_attempt": True,
            },
            {
                "id": "pre_final_repair_source_inspection",
                "plan_step_ids": ["S011", "S012", "S016", "S021", "S022", "S024"],
                "instruction": (
                    "Open the decisive pages returned by the repair search before final synthesis. "
                    "Use exact returned URLs; prioritize direct answer support and any page needed to "
                    "resolve a reviewer's material blocker. Do not finalize on this turn."
                ),
                "allowed_actions": ["open_many"],
                "advance_on_attempt": True,
            },
        ]
    )

    if _requires_geo_verification(route_record):
        steps.append(
            {
                "id": "geospatial_verification",
                "plan_step_ids": ["S010", "S011", "S012", "S016"],
                "instruction": (
                    "The question has multiple geographic distance constraints. Call geo_search "
                    "now using the recovered candidate location and the named landmarks or "
                    "addresses from the evidence. Supply up to four concrete anchors with the "
                    "question's expected distances, choose the requested nearby category, and use "
                    "the returned route evidence to distinguish candidates. Do not finalize on "
                    "this turn."
                ),
                "allowed_actions": ["geo_search"],
                "advance_on_attempt": True,
            }
        )

    steps.append(
        {
            "id": "finalize",
            "plan_step_ids": ["S016", "S017", "S018", "S019", "S020"],
            "instruction": (
                "Audit the current winner internally against every hard constraint, identity, "
                "cardinality, source-family independence, alternate-candidate search, "
                "contradiction search, and exact answer form. Then call final exactly once with "
                "one succinct answer in the requested form, a brief evidence-grounded "
                "explanation, calibrated confidence, and inspected citation URLs. Do not search, "
                "open, consult, or save another note."
            ),
            "allowed_actions": ["final"],
        }
    )

    review_guidance = canonical_json(
        {
            "question_model": {
                **question_model,
                "answer_type": discovery_profile["answer_type"],
                "answer_cardinality": discovery_profile["answer_cardinality"],
                "lexical_anchors": discovery_profile["question_first_anchors"],
                "source_targets": discovery_profile["source_targets"],
                "answer_type_inference": discovery_profile["answer_type_inference"],
                "route_metadata_role": discovery_profile["route_metadata_role"],
            },
            "constraints": constraints,
            "mapped_sources": [
                {
                    "source_id": source.get("source_id"),
                    "role": source.get("role"),
                    "title": redact_oracle_text(str(source.get("title") or ""), oracle_record),
                    "url": source.get("url"),
                }
                for source in sources
            ],
            "required_turns": [
                {
                    "id": step["id"],
                    "plan_step_ids": step.get("plan_step_ids"),
                    "instruction": step["instruction"],
                }
                for step in steps
            ],
        }
    )
    return steps, review_guidance


def controller_label_leaks(
    *,
    question: str,
    oracle_record: dict[str, Any],
    steps: list[dict[str, Any]],
    review_guidance: str,
) -> list[str]:
    """Report answer strings introduced by the controller rather than the question or URLs."""
    controller_text = review_guidance + "\n" + "\n".join(
        str(step.get("instruction") or "")
        + "\n"
        + canonical_json(step.get("required_queries") or [])
        for step in steps
    )
    controller_without_urls = _URL_TEXT.sub("<public-source-url>", controller_text)
    leaks = []
    for alias in _answer_aliases(oracle_record):
        if alias.replace(",", "").replace(".", "").isdigit():
            continue
        if (
            not _contains_phrase(question, alias)
            and _contains_phrase(controller_without_urls, alias)
        ):
            leaks.append(alias)
    return leaks


def audit_system_messages(
    *,
    transcript: list[dict[str, Any]],
    invariant_system_prompt: str,
    events: list[dict[str, Any]],
    expected_final_system_prompt: str,
) -> dict[str, Any]:
    """Reject item-specific controller content in any model-visible system message."""
    allowed = {invariant_system_prompt, expected_final_system_prompt}
    recorded_final_systems = [
        str(event["final_system"])
        for event in events
        if event.get("event") == "scripted_guidance_final_context_built"
        and str(event.get("final_system") or "").strip()
    ]
    system_messages = [
        str(message.get("content") or "")
        for message in transcript
        if message.get("role") == "system"
    ]
    unexpected = [content for content in system_messages if content not in allowed]
    unexpected_recorded_final = [
        content for content in recorded_final_systems if content != expected_final_system_prompt
    ]
    return {
        "passed": not unexpected and not unexpected_recorded_final,
        "system_message_count": len(system_messages),
        "distinct_system_message_count": len(set(system_messages)),
        "invariant_system_prompt_sha256": hashlib.sha256(
            invariant_system_prompt.encode("utf-8")
        ).hexdigest(),
        "expected_final_system_prompt_sha256": hashlib.sha256(
            expected_final_system_prompt.encode("utf-8")
        ).hexdigest(),
        "unexpected_system_message_count": len(unexpected),
        "unexpected_recorded_final_system_count": len(unexpected_recorded_final),
    }


def _teacher_message(
    contract: dict[str, Any], index: int, count: int, *, role: str = "user"
) -> dict[str, Any]:
    return {
        "role": role,
        "content": (
            f"Teacher-forced research step {index}/{count}. This controller instruction is not "
            "factual evidence. Execute exactly one tool action satisfying this contract on this "
            "turn. Do not skip ahead, substitute a different action, or finalize early.\n"
            + canonical_json(contract)
        ),
    }


def curate_scripted_training_messages(
    events: list[dict[str, Any]],
    *,
    initial_messages: list[dict[str, Any]],
    scripted_step_count: int,
) -> list[dict[str, Any]]:
    """Build an accepted-only training trajectory while preserving Star reasoning."""
    messages = [deepcopy(value) for value in initial_messages[:2]]
    completed = [
        event
        for event in events
        if event.get("event") == "scripted_guidance_step_completed"
        and event.get("action_succeeded") is True
    ]
    emitted_indices: set[int] = set()
    for completion in completed:
        outer_step = completion.get("step")
        scripted_index = completion.get("scripted_step_index")
        if not isinstance(outer_step, int) or not isinstance(scripted_index, int):
            continue
        rows = [event for event in events if event.get("step") == outer_step]
        started = next(
            (event for event in rows if event.get("event") == "scripted_guidance_step_started"),
            None,
        )
        response = next(
            (event for event in reversed(rows) if event.get("event") == "model_response"),
            None,
        )
        action = next(
            (
                event
                for event in reversed(rows)
                if event.get("event") == "action_completed" and event.get("ok") is True
            ),
            None,
        )
        if not started or not response or not action:
            continue
        contract = started.get("scripted_step")
        if not isinstance(contract, dict):
            continue
        messages.append(
            _teacher_message(
                contract,
                scripted_index + 1,
                scripted_step_count,
                role=str(started.get("scripted_guidance_role") or "user"),
            )
        )
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": str(response.get("assistant_content") or ""),
        }
        if response.get("assistant_reasoning"):
            assistant["reasoning"] = response["assistant_reasoning"]
        tool_calls = response.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            assistant["tool_calls"] = tool_calls[:1]
        messages.append(assistant)
        result_text = json.dumps(action.get("result") or {}, ensure_ascii=False)
        if assistant.get("tool_calls"):
            tool_call = assistant["tool_calls"][0]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"accepted-{outer_step}"),
                    "name": str(
                        (tool_call.get("function") or {}).get("name")
                        or action.get("action")
                        or "tool"
                    ),
                    "content": result_text,
                }
            )
        else:
            messages.append({"role": "user", "content": "Tool result:\n" + result_text})
        emitted_indices.add(scripted_index)

    final_event = next(
        (event for event in reversed(events) if event.get("event") == "trial_final"),
        None,
    )
    if final_event is not None and isinstance(final_event.get("step"), int):
        outer_step = int(final_event["step"])
        rows = [event for event in events if event.get("step") == outer_step]
        started = next(
            (event for event in rows if event.get("event") == "scripted_guidance_step_started"),
            None,
        )
        response = next(
            (event for event in reversed(rows) if event.get("event") == "model_response"),
            None,
        )
        if started and response and isinstance(started.get("scripted_step"), dict):
            scripted_index = int(started.get("scripted_step_index") or 0)
            final_context = next(
                (
                    event
                    for event in rows
                    if event.get("event") == "scripted_guidance_final_context_built"
                ),
                None,
            )
            if final_context and final_context.get("final_system") and final_context.get("final_user"):
                messages.append(
                    {"role": "system", "content": str(final_context["final_system"])}
                )
                messages.append({"role": "user", "content": str(final_context["final_user"])})
            elif scripted_index not in emitted_indices:
                messages.append(
                    _teacher_message(
                        started["scripted_step"],
                        scripted_index + 1,
                        scripted_step_count,
                        role=str(started.get("scripted_guidance_role") or "user"),
                    )
                )
            assistant = {
                "role": "assistant",
                "content": str(response.get("assistant_content") or ""),
            }
            if response.get("assistant_reasoning"):
                assistant["reasoning"] = response["assistant_reasoning"]
            tool_calls = response.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                assistant["tool_calls"] = tool_calls[:1]
            messages.append(assistant)
    return messages


def training_message_quality(
    messages: list[dict[str, Any]],
    *,
    minimum_reasoning_ratio: float = 0.75,
) -> dict[str, Any]:
    """Evaluate accepted-only traces without penalizing transport-only tool calls."""
    if not 0.0 <= minimum_reasoning_ratio <= 1.0:
        raise ValueError("minimum_reasoning_ratio must be between 0 and 1")
    assistant_actions = [
        message
        for message in messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    names: list[str] = []
    actions_without_reasoning: list[str] = []
    final_reasoning_present = False
    for message in assistant_actions:
        tool_call = (message.get("tool_calls") or [{}])[0]
        name = str((tool_call.get("function") or {}).get("name") or "")
        names.append(name)
        has_reasoning = bool(str(message.get("reasoning") or "").strip())
        if not has_reasoning:
            actions_without_reasoning.append(name or "unknown")
        if name == "final":
            final_reasoning_present = has_reasoning

    reasoning_count = len(assistant_actions) - len(actions_without_reasoning)
    reasoning_ratio = reasoning_count / len(assistant_actions) if assistant_actions else 0.0
    failed_tool_results = 0
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            value = json.loads(str(message.get("content") or "{}"))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("ok") is False:
            failed_tool_results += 1

    final_action_present = bool(names and names[-1] == "final")
    return {
        "assistant_action_count": len(assistant_actions),
        "assistant_reasoning_count": reasoning_count,
        "assistant_reasoning_ratio": round(reasoning_ratio, 6),
        "minimum_reasoning_ratio": minimum_reasoning_ratio,
        "actions_without_reasoning": actions_without_reasoning,
        "final_reasoning_present": final_reasoning_present,
        "failed_tool_results": failed_tool_results,
        "final_action_present": final_action_present,
        "tool_names": names,
        "passed": bool(
            assistant_actions
            and reasoning_ratio >= minimum_reasoning_ratio
            and final_reasoning_present
            and failed_tool_results == 0
            and final_action_present
        ),
    }


__all__ = [
    "audit_system_messages",
    "compile_guided_steps",
    "controller_label_leaks",
    "curate_scripted_training_messages",
    "redact_oracle_text",
    "training_message_quality",
]
