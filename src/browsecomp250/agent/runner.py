from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from ..browser.extract import page_window
from ..browser.fetcher import BrowserError, PageFetcher
from ..config import AgentConfig, BrowserConfig, ExternalModelConfig, ModelConfig
from ..external import ExternalModelBroker, ExternalModelError
from ..geo import GeoResearchClient, GeoResearchError
from ..llm import ModelAPIError, OpenAICompatibleClient, ProtocolError, parse_json_action
from ..llm.client import settings_from_model_config
from ..llm.protocol import action_from_tool_call, canonicalize_tool_call
from ..llm.tools import tool_schemas
from ..prompts import AGENT_SYSTEM_PROMPT
from ..search.base import SearchError, SearchProvider
from ..types import AgentAction, AgentOutcome, PageDocument, Usage
from ..util import canonical_json, truncate_middle

_PUBLIC_URL = re.compile(r"https?://[^\s<>()\[\]{}\"'`]+", flags=re.I)
_SEARCH_DATE_RANGE = re.compile(
    r"\b(?:after|before):\d{4}(?:-\d{2}-\d{2})?\b|"
    r"\b(?:1[5-9]\d{2}|20\d{2})\s*(?:\.\.|[-–—]|\bto\b)\s*"
    r"(?:1[5-9]\d{2}|20\d{2})\b",
    flags=re.I,
)
_SEARCH_TOKEN = re.compile(r"[a-z0-9]+", flags=re.I)
_STRATEGY_PLACEHOLDER_QUERY = re.compile(
    r"^(?:query|search)(?:\s+(?:query|search))?\s*\d+$", flags=re.I
)
_BENCHMARK_REQUEST_NAMESPACE = re.compile(
    r"^(?P<run>.+):bc250-(?P<rank>\d+)-row-\d+:attempt-\d+(?P<suffix>.*)$"
)

SCRIPTED_FINAL_SYSTEM_PROMPT = (
    "You are the final synthesis stage of a completed research trajectory. "
    "All retrieval is finished and the only available action is final. Read "
    "the supplied original question and inspected public evidence, reason "
    "carefully, then call the final tool exactly once. Set exact_answer to "
    "only the requested concrete answer, include a concise evidence-based "
    "explanation, calibrated confidence, and inspected citation URLs. Never "
    "emit, request, describe, or simulate a search, open, note, or helper call."
)
_LINK_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "article",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "use",
        "uses",
        "with",
    }
)
_RESEARCH_LINK_TERMS = frozenset(
    {
        "archive",
        "attribution",
        "biography",
        "chronicle",
        "chronicles",
        "culture",
        "discovery",
        "documented",
        "documentation",
        "evidence",
        "history",
        "origin",
        "origins",
        "primary",
        "research",
        "ritual",
        "source",
        "study",
        "traditional",
    }
)
_EVIDENCE_SIGNAL_TERMS = frozenset(
    {
        "according",
        "attributed",
        "credited",
        "described",
        "documented",
        "earliest",
        "first",
        "identified",
        "named",
        "reported",
        "wrote",
    }
)
_ABSTENTION_ANSWER = re.compile(
    r"^(?:"
    r"unknown|none|n/?a|inconclusive|undetermined|"
    r"(?:not|cannot|can't|unable\s+to)\s+(?:be\s+)?(?:determine|determined|identify|identified|"
    r"verify|verified|conclude|concluded)|"
    r"(?:not\s+)?(?:conclusively\s+)?(?:identifiable|verifiable|determinable)|"
    r"(?:insufficient|inadequate|not\s+enough)\s+(?:evidence|information|data)|"
    r"no\s+(?:conclusive\s+)?(?:answer|candidate|determination)"
    r")(?:\b.*)?$",
    flags=re.I,
)
_ENDS_WITH_WORD = re.compile(
    r"\b(?:ends?|ending)\s+with\s+the\s+word\s+[\"'“”‘’]*\s*"
    r"(?P<word>[a-z0-9][a-z0-9'’-]*)",
    flags=re.I,
)
_STARTS_WITH_WORD = re.compile(
    r"\b(?:starts?|begins?|starting|beginning)\s+with\s+the\s+word\s+[\"'“”‘’]*\s*"
    r"(?P<word>[a-z0-9][a-z0-9'’-]*)",
    flags=re.I,
)
_ANSWER_WORD = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", flags=re.I)
_CONSENSUS_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", flags=re.UNICODE)
_IDENTITY_QUESTION = re.compile(
    r"(?:^\s*who\b|\b(?:tell\s+me|determine|find\s+out)\s+who\b|"
    r"\bwhat\s+(?:is|was)\s+(?:the\s+)?name\s+of|"
    r"what\s+(?:is|was)\s+(?:this|the)\s+(?:person|individual|celebrity|actor|actress|"
    r"author|artist|scientist|researcher|politician|athlete)['’]s\s+name|"
    r"\bidentify\s+(?:this|the)\s+(?:person|individual|celebrity|actor|actress|author|"
    r"artist|scientist|researcher|politician|athlete|professor|director)|"
    r"^\s*(?:what|which)\s+(?:person|individual|celebrity|actor|actress|author|artist|"
    r"scientist|researcher|politician|athlete|professor|director))\b",
    flags=re.I,
)
_GENERIC_IDENTITY_NOUN = re.compile(
    r"^(?:the|a|an)?\s*(?:(?:likely|possible|probable|specific|unnamed|implied|"
    r"famous|notable|leading|well[- ]known)\s+)*"
    r"(?:person|individual|celebrity|woman|man|female|male|actor|actress|artist|"
    r"playwright|figure|candidate|producer|author|scientist|researcher|politician|athlete|"
    r"professor|academic|director|filmmaker|musician|singer|footballer|player|model|"
    r"personality|inventor|doctor|physician|engineer|architect|composer)\b",
    flags=re.I,
)
_GENERIC_IDENTITY_QUALIFIER = re.compile(
    r"\b(?:who|that|which|likely|possibly|probably|specific|unnamed|implied|candidate)\b",
    flags=re.I,
)
_GEO_DISTANCE_CLUE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:miles?|mi\.?|kilometers?|kilometres?|km|meters?|metres?|m)\b",
    flags=re.I,
)
_GEO_RELATION_CLUE = re.compile(
    r"\b(?:walk|walking|drive|driving|distance|located|location|nearby|nearest|from)\b",
    flags=re.I,
)
_GEO_ENTITY_QUESTION = re.compile(
    r"\b(?:restaurant|hotel|motel|store|shop|business|company|chain|venue|attraction|"
    r"establishment|place)\b",
    flags=re.I,
)


class AgentRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        agent_config: AgentConfig,
        browser_config: BrowserConfig,
        search_provider: SearchProvider,
        page_fetcher: PageFetcher,
        model_client: OpenAICompatibleClient | None = None,
        geo_research: GeoResearchClient | None = None,
        external_model_config: ExternalModelConfig | None = None,
        external_model_broker: ExternalModelBroker | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        system_prompt: str | None = None,
        initial_force_final: bool = False,
    ):
        self.model_config = model_config
        self.agent_config = agent_config
        self.browser_config = browser_config
        self.search = search_provider
        self.browser = page_fetcher
        self.client = model_client or OpenAICompatibleClient(
            settings_from_model_config(model_config)
        )
        self._owns_client = model_client is None
        self.geo = geo_research
        self._owns_geo = geo_research is None
        self.external_model_config = external_model_config or ExternalModelConfig()
        self.external_model = external_model_broker
        self.event_sink = event_sink
        self.initial_force_final = initial_force_final
        self.system_prompt = AGENT_SYSTEM_PROMPT
        if system_prompt is not None:
            self.system_prompt = system_prompt.strip()
        elif agent_config.system_prompt_path is not None:
            self.system_prompt = agent_config.system_prompt_path.read_text(encoding="utf-8").strip()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.close()
        if self.geo is not None and self._owns_geo:
            await self.geo.close()

    def _geo_client(self) -> GeoResearchClient:
        if self.geo is None:
            self.geo = GeoResearchClient(
                self.browser_config.cache_path.with_name("geo-cache.sqlite3")
            )
        return self.geo

    def _emit(self, event: str, **values: Any) -> None:
        if self.event_sink is not None:
            self.event_sink({"event": event, **values})

    async def run(
        self,
        question: str,
        *,
        request_namespace: str | None = None,
        initial_guidance: str | None = None,
        review_guidance: str | None = None,
        scripted_guidance_steps: list[dict[str, Any]] | None = None,
        blocking_guidance_adversary: bool = False,
        guidance_adversary_interval_steps: int = 8,
        guidance_adversary_max_checkpoints: int = 2,
        scripted_final_block_fail_fast: bool = False,
        scripted_guidance_role: str = "user",
        scripted_step_max_attempts: int = 6,
    ) -> AgentOutcome:
        if scripted_guidance_role not in {"system", "user"}:
            raise ValueError("scripted_guidance_role must be 'system' or 'user'")
        if scripted_step_max_attempts < 1:
            raise ValueError("scripted_step_max_attempts must be at least 1")
        started = time.perf_counter()
        usage = Usage()
        transcript: list[dict[str, Any]] = []
        errors: list[str] = []
        notes: list[str] = []
        opened: dict[str, PageDocument] = {}
        evidence_journal: list[dict[str, Any]] = []
        geo_evidence: list[dict[str, Any]] = []
        search_calls = page_opens = find_calls = retrieved_chars = external_model_calls = 0
        parse_failures = 0
        protocol = self.model_config.protocol
        force_final = self.initial_force_final
        require_open = False
        search_streak = 0
        automatic_external_attempted = False
        automatic_strategy_recovery_attempted = False
        automatic_finalization_rescue_attempted = False
        last_external_completion_at = started
        forced_nonfinal_rejections = 0
        last_action_fingerprint: str | None = None
        consecutive_duplicate_actions = 0
        last_successful_search_result: dict[str, Any] | None = None
        search_query_history: list[str] = []
        pending_surface_constraint_correction: str | None = None
        pending_guidance_block: dict[str, Any] | None = None
        scripted_step_index = 0
        scripted_step_attempts = 0
        scripted_final_context_recorded = False
        guidance_adversary_checkpoints = 0
        guidance_final_reviews: dict[str, dict[str, Any]] = {}
        chain_enabled = self.model_config.response_chain and not scripted_guidance_steps
        previous_response_id: str | None = None
        chain_delta_messages: list[dict[str, Any]] | None = None
        namespace_material = request_namespace or question
        request_headers = self._routing_headers(
            namespace_material,
            routing_backend_pool=self.model_config.routing_backend_pool,
        )
        chain_namespace = request_headers["X-FRL-Conversation-Id"].removeprefix("bc250-")
        external_namespace = request_namespace or chain_namespace
        adversary_guidance = review_guidance or initial_guidance or ""

        guidance_block = ""
        if initial_guidance and initial_guidance.strip():
            guidance_block = "Research plan:\n" + initial_guidance.strip() + "\n\n"
        initial_user = (
            guidance_block
            + "Question:\n"
            + question
            + "\n\nBudgets: "
            + canonical_json(
                {
                    "max_steps": self.agent_config.max_steps,
                    "force_final_after_seconds": self.agent_config.force_final_after_seconds,
                    "max_search_calls": self.agent_config.max_search_calls,
                    "max_page_opens": self.agent_config.max_page_opens,
                    "max_find_calls": self.agent_config.max_find_calls,
                    "max_external_model_calls": self.external_model_config.max_calls_per_task,
                }
            )
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_user},
        ]
        transcript.extend(messages)
        self._emit(
            "trial_started",
            protocol=protocol,
            response_chain=chain_enabled,
            routing_conversation_id=request_headers["X-FRL-Conversation-Id"],
            routing_cohort_id=request_headers.get("X-FRL-KV-Cohort-Id"),
            routing_cohort_index=request_headers.get("X-FRL-KV-Cohort-Index"),
        )

        for step in range(1, self.agent_config.max_steps + 1):
            if time.perf_counter() - started > 0.98 * 3600:
                errors.append("Agent exceeded internal one-hour safety limit")
                break
            current_scripted_step: dict[str, Any] | None = None
            scripted_force_final = False
            scripted_required_action: str | None = None
            if scripted_guidance_steps and scripted_step_index < len(scripted_guidance_steps):
                current_scripted_step = scripted_guidance_steps[scripted_step_index]
                scripted_step_attempts += 1
                if scripted_step_attempts > scripted_step_max_attempts:
                    failure = (
                        "Teacher-forced step exceeded its bounded retry budget: "
                        + canonical_json(current_scripted_step)
                    )
                    errors.append(failure)
                    self._emit(
                        "scripted_guidance_step_exhausted",
                        step=step,
                        scripted_step_index=scripted_step_index,
                        attempts=scripted_step_attempts - 1,
                        scripted_step=current_scripted_step,
                    )
                    break
                scripted_actions = {
                    str(value).strip()
                    for value in current_scripted_step.get("allowed_actions", [])
                    if str(value).strip()
                }
                if len(scripted_actions) == 1:
                    scripted_required_action = next(iter(scripted_actions))
                scripted_force_final = scripted_required_action == "final"
                scripted_message = {
                    "role": scripted_guidance_role,
                    "content": (
                        "Teacher-forced research step "
                        f"{scripted_step_index + 1}/{len(scripted_guidance_steps)}. "
                        "This controller instruction is not factual evidence. Execute exactly one "
                        "tool action satisfying this contract on this turn. Do not skip ahead, "
                        "substitute a different action, or finalize early.\n"
                        + canonical_json(current_scripted_step)
                    ),
                }
                messages.append(scripted_message)
                transcript.append(scripted_message)
                if chain_delta_messages is not None:
                    chain_delta_messages.append(scripted_message)
                self._emit(
                    "scripted_guidance_step_started",
                    step=step,
                    scripted_step_index=scripted_step_index,
                    scripted_step=current_scripted_step,
                    scripted_guidance_role=scripted_guidance_role,
                )
            checkpoint_due = bool(
                blocking_guidance_adversary
                and initial_guidance
                and guidance_adversary_interval_steps > 0
                and step > 1
                and (step - 1) % guidance_adversary_interval_steps == 0
                and guidance_adversary_checkpoints < guidance_adversary_max_checkpoints
                and self.external_model is not None
                and self.external_model_config.enabled
            )
            if checkpoint_due:
                self._emit(
                    "blocking_guidance_adversary_started",
                    step=step,
                    phase="checkpoint",
                )
                review = await self._blocking_guidance_adversary_review(
                    question=question,
                    guidance=adversary_guidance,
                    messages=messages,
                    notes=notes,
                    opened=opened,
                    search_queries=search_query_history,
                    request_namespace=external_namespace,
                    phase="checkpoint",
                )
                guidance_adversary_checkpoints += 1
                external_model_calls += int(review.get("attempted") or 1)
                last_external_completion_at = time.perf_counter()
                pending_guidance_block = (
                    review if str(review.get("verdict") or "").upper() == "BLOCK" else None
                )
                pending_message = self._attach_blocking_guidance_review(messages, review)
                chain_delta_messages = [pending_message]
                transcript.append(
                    {
                        "role": "assistant",
                        "name": "blocking_plan_adversary",
                        "content": canonical_json(review),
                    }
                )
                self._emit(
                    "blocking_guidance_adversary_completed",
                    step=step,
                    phase="checkpoint",
                    verdict=review.get("verdict"),
                    request_id=review.get("request_id"),
                    required_next_actions=review.get("required_next_actions"),
                    review=review,
                )
            messages = self._compact_history(messages, initial_user, notes, opened)
            self._emit(
                "turn_started",
                step=step,
                search_calls=search_calls,
                page_opens=page_opens,
                find_calls=find_calls,
                retrieved_chars=retrieved_chars,
                external_model_calls=external_model_calls,
            )
            try:
                wire_messages = messages
                chain_body: dict[str, Any] = {}
                if scripted_force_final:
                    unique_opened: dict[str, PageDocument] = {}
                    for document in opened.values():
                        unique_opened[document.final_url] = document
                    opened_evidence = [
                        {
                            "url": document.final_url,
                            "title": document.title,
                            "text": truncate_middle(document.text, 6_000),
                        }
                        for document in list(unique_opened.values())[-12:]
                    ]
                    final_system = SCRIPTED_FINAL_SYSTEM_PROMPT
                    final_user = (
                        "Answer-redacted guide and constraint plan:\n"
                        + truncate_middle(review_guidance or initial_guidance or "", 40_000)
                        + "\n\nOriginal question:\n"
                        + question
                        + "\n\nFinal step contract:\n"
                        + canonical_json(current_scripted_step or {})
                        + "\n\nInspected public source evidence:\n"
                        + canonical_json(opened_evidence)
                        + "\n\nChronological public tool-evidence journal:\n"
                        + truncate_middle(canonical_json(evidence_journal[-30:]), 80_000)
                        + "\n\nSaved structured research notes:\n"
                        + truncate_middle(canonical_json(notes[-20:]), 30_000)
                        + "\n\nUse the complete evidence state above. Earlier search batches and "
                        "independent helper findings remain relevant; do not privilege only the "
                        "last search. The private reference answer is not present."
                    )
                    wire_messages = [
                        {"role": "system", "content": final_system},
                        {"role": "user", "content": final_user},
                    ]
                    messages = list(wire_messages)
                    chain_body = {}
                    if not scripted_final_context_recorded:
                        transcript.extend(
                            [
                                {
                                    "role": "system",
                                    "name": "scripted_final_context_reset",
                                    "content": final_system,
                                },
                                {
                                    "role": "user",
                                    "name": "scripted_final_evidence",
                                    "content": final_user,
                                },
                            ]
                        )
                        scripted_final_context_recorded = True
                    self._emit(
                        "scripted_guidance_final_context_built",
                        step=step,
                        opened_source_count=len(opened_evidence),
                        context_chars=sum(
                            len(str(message.get("content") or "")) for message in wire_messages
                        ),
                        final_system=final_system,
                        final_user=final_user,
                    )
                if chain_enabled:
                    chain_body = {
                        "frontierrl_messages_mode": "delta" if previous_response_id else "full",
                        "frontierrl_request_id": f"bc250:{chain_namespace}:{step}",
                    }
                    if previous_response_id:
                        chain_body["frontierrl_previous_response_id"] = previous_response_id
                        wire_messages = list(chain_delta_messages or [])
                        if not wire_messages:
                            raise ModelAPIError(
                                "Response-chain continuation had no caller-side delta"
                            )
                query_started = time.perf_counter()
                force_final_this_turn = (
                    force_final
                    or scripted_force_final
                    or (
                        self.agent_config.force_final_after_seconds > 0
                        and time.perf_counter() - started
                        >= self.agent_config.force_final_after_seconds
                    )
                    or self._near_budget(
                        search_calls,
                        page_opens,
                        find_calls,
                        retrieved_chars,
                        external_model_calls,
                    )
                    or step == self.agent_config.max_steps
                )
                query_force_final = (
                    scripted_force_final
                    if current_scripted_step is not None
                    else force_final_this_turn
                )
                query_task = asyncio.create_task(
                    self._query(
                        wire_messages,
                        protocol,
                        extra_body=chain_body,
                        force_final=query_force_final,
                        require_open=require_open,
                        required_action=scripted_required_action,
                        request_headers=request_headers,
                    )
                )
                try:
                    while True:
                        done, _ = await asyncio.wait({query_task}, timeout=15)
                        if query_task in done:
                            response = query_task.result()
                            break
                        self._emit(
                            "model_wait",
                            step=step,
                            elapsed_seconds=time.perf_counter() - query_started,
                        )
                except BaseException:
                    query_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await query_task
                    raise
                usage = usage + response.usage
                if chain_enabled:
                    if response.response_id:
                        previous_response_id = response.response_id
                    else:
                        errors.append(
                            "Endpoint omitted a response-chain ID; continuing with full-message compatibility"
                        )
                        chain_enabled = False
                        previous_response_id = None
                        chain_delta_messages = None
            except ModelAPIError as exc:
                if protocol == "auto":
                    protocol = "json"
                    errors.append(f"Native tools unavailable; fell back to JSON protocol: {exc}")
                    continue
                errors.append(str(exc))
                self._emit("model_error", step=step, error=str(exc))
                break

            raw_tool_calls = response.raw_message.get("tool_calls")
            if (
                protocol in {"tools", "auto"}
                and isinstance(raw_tool_calls, list)
                and raw_tool_calls
                and isinstance(raw_tool_calls[0], dict)
            ):
                raw_tool_call = raw_tool_calls[0]
                required_queries = (
                    current_scripted_step.get("required_queries")
                    if current_scripted_step is not None
                    else None
                )
                required_urls = (
                    current_scripted_step.get("required_urls")
                    if current_scripted_step is not None
                    else None
                )
                try:
                    _, canonical_tool_call = canonicalize_tool_call(
                        raw_tool_call,
                        expected_action=scripted_required_action,
                        required_queries=(
                            required_queries if isinstance(required_queries, list) else None
                        ),
                        required_urls=(required_urls if isinstance(required_urls, list) else None),
                    )
                except ProtocolError:
                    # Preserve malformed evidence for the ordinary protocol-retry path.
                    pass
                else:
                    if canonical_tool_call != raw_tool_call:
                        self._emit(
                            "tool_call_normalized",
                            step=step,
                            raw_tool_call=raw_tool_call,
                            tool_call=canonical_tool_call,
                            scripted_step_id=(
                                current_scripted_step.get("id")
                                if current_scripted_step is not None
                                else None
                            ),
                        )
                    response.raw_message["tool_calls"] = [canonical_tool_call]

            self._emit(
                "model_response",
                step=step,
                latency_seconds=response.latency_seconds,
                finish_reason=response.finish_reason,
                response_id=response.response_id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                assistant_content=response.content,
                assistant_reasoning=response.raw_message.get("reasoning"),
                tool_calls=response.raw_message.get("tool_calls"),
                response_metadata=response.metadata,
            )

            transcript.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.raw_message.get("tool_calls"),
                    "reasoning": response.raw_message.get("reasoning"),
                    "response_id": response.response_id,
                    "conversation_id": response.conversation_id,
                    "response_metadata": response.metadata,
                    "usage": asdict(response.usage),
                    "latency_seconds": response.latency_seconds,
                }
            )
            try:
                action, assistant_message = self._parse_action(
                    response,
                    protocol,
                    force_final=query_force_final,
                )
                parse_failures = 0
            except ProtocolError as exc:
                parse_failures += 1
                errors.append(f"Step {step} protocol error: {exc}")
                if parse_failures > self.agent_config.parse_retries:
                    self._emit(
                        "protocol_exhausted",
                        step=step,
                        failures=parse_failures,
                        error=str(exc),
                    )
                    break
                correction = (
                    "Your previous response was invalid. Return exactly one valid JSON action "
                    f"without markdown. Error: {exc}"
                )
                raw_tool_calls = response.raw_message.get("tool_calls")
                if (
                    protocol in {"tools", "auto"}
                    and isinstance(raw_tool_calls, list)
                    and raw_tool_calls
                ):
                    rejected_tool_call = raw_tool_calls[0]
                    assistant_error_message = {
                        "role": "assistant",
                        "content": response.raw_message.get("content") or "",
                        "tool_calls": [rejected_tool_call],
                    }
                    correction_message = {
                        "role": "tool",
                        "tool_call_id": rejected_tool_call.get("id", f"call-{step}"),
                        "name": str(
                            (rejected_tool_call.get("function") or {}).get("name") or "invalid"
                        ),
                        "content": canonical_json({"ok": False, "error": correction}),
                    }
                    messages.extend([assistant_error_message, correction_message])
                    transcript.append(correction_message)
                    chain_delta_messages = [correction_message]
                else:
                    messages.append({"role": "assistant", "content": response.content})
                    correction_message = {"role": "user", "content": correction}
                    messages.append(correction_message)
                    transcript.append(correction_message)
                    chain_delta_messages = [correction_message]
                self._emit("protocol_retry", step=step, error=str(exc))
                continue

            original_action = action
            action, redundant_search_queries = self._filter_redundant_search_action(
                action,
                search_query_history,
            )
            semantic_duplicate_action = action is None
            strategy_recovery: dict[str, Any] | None = None
            if (
                semantic_duplicate_action
                and original_action.action in {"search", "search_many"}
                and self.agent_config.automatic_external_strategy_recovery
                and not automatic_strategy_recovery_attempted
                and consecutive_duplicate_actions > 0
                and self.external_model is not None
                and self.external_model_config.enabled
                and external_model_calls + 1 < self.external_model_config.max_calls_per_task
                and search_calls < self.agent_config.max_search_calls
            ):
                automatic_strategy_recovery_attempted = True
                self._emit(
                    "search_strategy_recovery_started",
                    step=step,
                    repeated_queries=redundant_search_queries,
                )
                strategy_task = asyncio.create_task(
                    self._automatic_external_query_strategy(
                        question=question,
                        messages=messages,
                        notes=notes,
                        prior_queries=search_query_history,
                        repeated_action=original_action,
                        request_namespace=external_namespace,
                        limit=min(
                            self.agent_config.max_batch_size,
                            self.agent_config.max_search_calls - search_calls,
                        ),
                    )
                )
                strategy_started = time.perf_counter()
                try:
                    while True:
                        done, _ = await asyncio.wait({strategy_task}, timeout=15)
                        if strategy_task in done:
                            strategy_queries, strategy_result = strategy_task.result()
                            break
                        self._emit(
                            "search_strategy_recovery_wait",
                            step=step,
                            elapsed_seconds=time.perf_counter() - strategy_started,
                        )
                except Exception as exc:  # noqa: BLE001 - normal search recovery remains available
                    strategy_queries = []
                    strategy_result = {
                        "ok": False,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    errors.append(f"Step {step} search-strategy recovery error: {exc}")
                except BaseException:
                    strategy_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await strategy_task
                    raise
                external_model_calls += max(1, int(strategy_result.get("attempted") or 1))
                last_external_completion_at = time.perf_counter()
                transcript.append(
                    {
                        "role": "assistant",
                        "name": "external_search_strategy",
                        "content": str(strategy_result.get("content") or ""),
                        "response_metadata": {
                            "request_id": strategy_result.get("request_id"),
                            "status": strategy_result.get("status"),
                        },
                    }
                )
                if strategy_queries:
                    # The successful strategy agent already supplied an independent Star-2
                    # review for this evidence state. Do not immediately launch the automatic
                    # search-strategy role again after executing its replacement queries.
                    automatic_external_attempted = True
                    action = AgentAction(
                        action="search_many",
                        payload={"queries": strategy_queries},
                    )
                    semantic_duplicate_action = False
                    strategy_recovery = {
                        "repeated_queries": redundant_search_queries,
                        "replacement_queries": strategy_queries,
                        "request_id": strategy_result.get("request_id"),
                    }
                    self._emit(
                        "search_strategy_recovery_completed",
                        step=step,
                        query_count=len(strategy_queries),
                        replacement_queries=strategy_queries,
                        request_id=strategy_result.get("request_id"),
                    )
                else:
                    self._emit(
                        "search_strategy_recovery_failed",
                        step=step,
                        error=strategy_result.get("error") or "no novel queries returned",
                        request_id=strategy_result.get("request_id"),
                        result=strategy_result,
                    )
            if action is None:
                action = original_action

            action, clipped_from = self._clip_action_to_remaining_budget(
                action,
                search_calls=search_calls,
                page_opens=page_opens,
                external_model_calls=external_model_calls,
            )
            if clipped_from is not None:
                self._emit(
                    "action_budget_clipped",
                    step=step,
                    action=action.action,
                    requested_count=clipped_from,
                    retained_count=self._batched_action_size(action),
                )
            self._emit("action_selected", step=step, action=action.action, payload=action.payload)

            if current_scripted_step is not None and not self._action_matches_scripted_step(
                action,
                current_scripted_step,
            ):
                correction = (
                    "The teacher-forced guide rejected this action. Execute exactly the current "
                    "step contract before advancing: "
                    + canonical_json(current_scripted_step)
                )
                errors.append(correction)
                raw_tool_calls = assistant_message.get("tool_calls")
                if (
                    protocol in {"tools", "auto"}
                    and isinstance(raw_tool_calls, list)
                    and raw_tool_calls
                ):
                    messages.append(assistant_message)
                    rejected_tool_call = raw_tool_calls[0]
                    correction_message = {
                        "role": "tool",
                        "tool_call_id": rejected_tool_call.get("id", f"call-{step}"),
                        "name": str(
                            (rejected_tool_call.get("function") or {}).get("name")
                            or action.action
                        ),
                        "content": canonical_json(
                            {
                                "ok": False,
                                "error": correction,
                                "scripted_guidance_step": current_scripted_step,
                            }
                        ),
                    }
                    messages.append(correction_message)
                    transcript.append(correction_message)
                    chain_delta_messages = [correction_message]
                else:
                    messages.append({"role": "assistant", "content": response.content})
                    correction_message = {"role": "user", "content": correction}
                    messages.append(correction_message)
                    transcript.append(correction_message)
                    chain_delta_messages = [correction_message]
                self._emit(
                    "scripted_guidance_action_rejected",
                    step=step,
                    action=action.action,
                    scripted_step_index=scripted_step_index,
                    scripted_step=current_scripted_step,
                )
                continue
            if current_scripted_step is not None:
                self._emit(
                    "scripted_guidance_action_accepted",
                    step=step,
                    action=action.action,
                    scripted_step_index=scripted_step_index,
                )

            if pending_guidance_block is not None:
                if not self._action_repairs_blocking_guidance_review(
                    action,
                    pending_guidance_block,
                ):
                    required_actions = pending_guidance_block.get("required_next_actions") or []
                    correction = (
                        "This action is blocked because it does not execute the blocking "
                        "plan-adherence review's required next action. Required next actions: "
                        + canonical_json(required_actions)
                        + ". Choose a compliant tool action now."
                    )
                    errors.append(correction)
                    raw_tool_calls = assistant_message.get("tool_calls")
                    if (
                        protocol in {"tools", "auto"}
                        and isinstance(raw_tool_calls, list)
                        and raw_tool_calls
                    ):
                        messages.append(assistant_message)
                        rejected_tool_call = raw_tool_calls[0]
                        correction_message = {
                            "role": "tool",
                            "tool_call_id": rejected_tool_call.get("id", f"call-{step}"),
                            "name": str(
                                (rejected_tool_call.get("function") or {}).get("name")
                                or action.action
                            ),
                            "content": canonical_json(
                                {
                                    "ok": False,
                                    "error": correction,
                                    "blocking_plan_adversary": pending_guidance_block,
                                }
                            ),
                        }
                        messages.append(correction_message)
                        transcript.append(correction_message)
                        chain_delta_messages = [correction_message]
                    else:
                        messages.append({"role": "assistant", "content": response.content})
                        correction_message = {"role": "user", "content": correction}
                        messages.append(correction_message)
                        transcript.append(correction_message)
                        chain_delta_messages = [correction_message]
                    self._emit(
                        "blocking_guidance_action_rejected",
                        step=step,
                        action=action.action,
                        required_next_actions=required_actions,
                    )
                    continue
                self._emit(
                    "blocking_guidance_repair_action_accepted",
                    step=step,
                    action=action.action,
                    required_next_actions=pending_guidance_block.get("required_next_actions"),
                )
                pending_guidance_block = None

            if (
                action.action == "final"
                and search_calls < self.agent_config.min_search_calls_before_final
            ):
                minimum = self.agent_config.min_search_calls_before_final
                correction = (
                    "Finalization is premature. Before returning final, run independent public-web "
                    f"searches that falsify the leading candidate (minimum search calls: {minimum}; "
                    f"completed: {search_calls}). Verify every hard clue without changing its "
                    "relation type."
                )
                errors.append(correction)
                raw_tool_calls = assistant_message.get("tool_calls")
                if (
                    protocol in {"tools", "auto"}
                    and isinstance(raw_tool_calls, list)
                    and raw_tool_calls
                ):
                    messages.append(assistant_message)
                    rejected_tool_call = raw_tool_calls[0]
                    correction_message = {
                        "role": "tool",
                        "tool_call_id": rejected_tool_call.get("id", f"call-{step}"),
                        "name": str(
                            (rejected_tool_call.get("function") or {}).get("name") or "final"
                        ),
                        "content": canonical_json({"ok": False, "error": correction}),
                    }
                else:
                    messages.append({"role": "assistant", "content": response.content})
                    correction_message = {"role": "user", "content": correction}
                messages.append(correction_message)
                transcript.append(correction_message)
                chain_delta_messages = [correction_message]
                self._emit(
                    "premature_final_rejected",
                    step=step,
                    search_calls=search_calls,
                    minimum_search_calls=minimum,
                )
                continue

            if action.action == "final":
                surface_errors = self._surface_answer_constraint_errors(
                    question,
                    str(action.payload.get("exact_answer") or ""),
                )
                if surface_errors:
                    correction = (
                        "Final answer violates an explicit surface constraint in the question: "
                        + "; ".join(surface_errors)
                        + ". Continue research and return a candidate that literally satisfies the "
                        "stated answer form. Do not reinterpret starts/ends-with wording."
                    )
                    errors.append(correction)
                    raw_tool_calls = assistant_message.get("tool_calls")
                    if (
                        protocol in {"tools", "auto"}
                        and isinstance(raw_tool_calls, list)
                        and raw_tool_calls
                    ):
                        messages.append(assistant_message)
                        rejected_tool_call = raw_tool_calls[0]
                        correction_message = {
                            "role": "tool",
                            "tool_call_id": rejected_tool_call.get("id", f"call-{step}"),
                            "name": str(
                                (rejected_tool_call.get("function") or {}).get("name") or "final"
                            ),
                            "content": canonical_json({"ok": False, "error": correction}),
                        }
                    else:
                        messages.append({"role": "assistant", "content": response.content})
                        correction_message = {"role": "user", "content": correction}
                    messages.append(correction_message)
                    transcript.append(correction_message)
                    chain_delta_messages = [correction_message]
                    self._emit(
                        "surface_constraint_final_rejected",
                        step=step,
                        violations=surface_errors,
                    )
                    continue
                answer_type_errors = self._answer_type_constraint_errors(
                    question,
                    str(action.payload.get("exact_answer") or ""),
                )
                if answer_type_errors:
                    correction = (
                        "Final answer does not satisfy the requested answer type: "
                        + "; ".join(answer_type_errors)
                        + ". Continue research using a genuinely different semantic route. Return "
                        "one named entity or other concrete value, not a description of the clues."
                    )
                    errors.append(correction)
                    raw_tool_calls = assistant_message.get("tool_calls")
                    if (
                        protocol in {"tools", "auto"}
                        and isinstance(raw_tool_calls, list)
                        and raw_tool_calls
                    ):
                        messages.append(assistant_message)
                        rejected_tool_call = raw_tool_calls[0]
                        correction_message = {
                            "role": "tool",
                            "tool_call_id": rejected_tool_call.get("id", f"call-{step}"),
                            "name": str(
                                (rejected_tool_call.get("function") or {}).get("name") or "final"
                            ),
                            "content": canonical_json({"ok": False, "error": correction}),
                        }
                    else:
                        messages.append({"role": "assistant", "content": response.content})
                        correction_message = {"role": "user", "content": correction}
                    messages.append(correction_message)
                    transcript.append(correction_message)
                    chain_delta_messages = [correction_message]
                    self._emit(
                        "answer_type_final_rejected",
                        step=step,
                        violations=answer_type_errors,
                    )
                    continue
                geo_errors = self._geo_final_constraint_errors(
                    question,
                    str(action.payload.get("exact_answer") or ""),
                    geo_evidence,
                )
                if geo_errors:
                    correction = (
                        "Final answer has unverified geospatial constraints: "
                        + "; ".join(geo_errors)
                        + ". Use geo_search on the stated landmarks or addresses, compare the "
                        "nearby entities and pedestrian distances, then continue."
                    )
                    errors.append(correction)
                    raw_tool_calls = assistant_message.get("tool_calls")
                    if (
                        protocol in {"tools", "auto"}
                        and isinstance(raw_tool_calls, list)
                        and raw_tool_calls
                    ):
                        messages.append(assistant_message)
                        rejected_tool_call = raw_tool_calls[0]
                        correction_message = {
                            "role": "tool",
                            "tool_call_id": rejected_tool_call.get("id", f"call-{step}"),
                            "name": str(
                                (rejected_tool_call.get("function") or {}).get("name") or "final"
                            ),
                            "content": canonical_json({"ok": False, "error": correction}),
                        }
                    else:
                        messages.append({"role": "assistant", "content": response.content})
                        correction_message = {"role": "user", "content": correction}
                    messages.append(correction_message)
                    transcript.append(correction_message)
                    chain_delta_messages = [correction_message]
                    self._emit(
                        "geo_verification_final_rejected",
                        step=step,
                        violations=geo_errors,
                    )
                    continue
                if self.agent_config.require_opened_citation_support:
                    evidence_errors = self._final_evidence_constraint_errors(
                        question,
                        str(action.payload.get("exact_answer") or ""),
                        action.payload.get("citations") or [],
                        opened,
                    )
                    if evidence_errors:
                        if (
                            self.agent_config.allow_unsupported_final_at_hard_budget
                            and step == self.agent_config.max_steps
                        ):
                            warning = (
                                "Hard-budget best-effort final lacked fully inspected source "
                                "support: " + "; ".join(evidence_errors)
                            )
                            errors.append(warning)
                            self._emit(
                                "citation_support_final_overridden_at_hard_budget",
                                step=step,
                                violations=evidence_errors,
                            )
                        else:
                            correction = (
                                "Final answer is not grounded in an inspected citation: "
                                + "; ".join(evidence_errors)
                                + ". Open the strongest cited source or search a different route, then "
                                "return an exact answer that the inspected evidence actually names."
                            )
                            errors.append(correction)
                            raw_tool_calls = assistant_message.get("tool_calls")
                            if (
                                protocol in {"tools", "auto"}
                                and isinstance(raw_tool_calls, list)
                                and raw_tool_calls
                            ):
                                messages.append(assistant_message)
                                rejected_tool_call = raw_tool_calls[0]
                                correction_message = {
                                    "role": "tool",
                                    "tool_call_id": rejected_tool_call.get("id", f"call-{step}"),
                                    "name": str(
                                        (rejected_tool_call.get("function") or {}).get("name")
                                        or "final"
                                    ),
                                    "content": canonical_json({"ok": False, "error": correction}),
                                }
                            else:
                                messages.append({"role": "assistant", "content": response.content})
                                correction_message = {"role": "user", "content": correction}
                            messages.append(correction_message)
                            transcript.append(correction_message)
                            chain_delta_messages = [correction_message]
                            self._emit(
                                "citation_support_final_rejected",
                                step=step,
                                violations=evidence_errors,
                            )
                            continue
                if (
                    blocking_guidance_adversary
                    and initial_guidance
                    and self.external_model is not None
                    and self.external_model_config.enabled
                ):
                    final_fingerprint = canonical_json(action.payload)
                    adversary_review = guidance_final_reviews.get(final_fingerprint)
                    if adversary_review is None:
                        self._emit(
                            "blocking_guidance_adversary_started",
                            step=step,
                            phase="final",
                        )
                        adversary_review = await self._blocking_guidance_adversary_review(
                            question=question,
                            guidance=adversary_guidance,
                            messages=messages,
                            notes=notes,
                            opened=opened,
                            search_queries=search_query_history,
                            request_namespace=external_namespace,
                            phase="final",
                            proposed_final=action.payload,
                        )
                        guidance_final_reviews[final_fingerprint] = adversary_review
                        external_model_calls += int(adversary_review.get("attempted") or 1)
                        last_external_completion_at = time.perf_counter()
                        transcript.append(
                            {
                                "role": "assistant",
                                "name": "blocking_plan_adversary",
                                "content": canonical_json(adversary_review),
                            }
                        )
                    verdict = str(adversary_review.get("verdict") or "BLOCK").upper()
                    self._emit(
                        "blocking_guidance_adversary_completed",
                        step=step,
                        phase="final",
                        verdict=verdict,
                        request_id=adversary_review.get("request_id"),
                        required_next_actions=adversary_review.get("required_next_actions"),
                        review=adversary_review,
                    )
                    if verdict != "PASS":
                        required_actions = adversary_review.get("required_next_actions") or []
                        correction = (
                            "The blocking plan-adherence adversary rejected this final. "
                            + str(adversary_review.get("reason") or "Material plan steps remain unverified.")
                            + " Required next actions: "
                            + canonical_json(required_actions)
                            + ". Complete those actions using public evidence, then propose a new final."
                        )
                        errors.append(correction)
                        notes.append(
                            truncate_middle(
                                "Blocking pre-submit review:\n"
                                + canonical_json(adversary_review),
                                5_000,
                            )
                        )
                        raw_tool_calls = assistant_message.get("tool_calls")
                        if (
                            protocol in {"tools", "auto"}
                            and isinstance(raw_tool_calls, list)
                            and raw_tool_calls
                        ):
                            messages.append(assistant_message)
                            rejected_tool_call = raw_tool_calls[0]
                            correction_message = {
                                "role": "tool",
                                "tool_call_id": rejected_tool_call.get("id", f"call-{step}"),
                                "name": str(
                                    (rejected_tool_call.get("function") or {}).get("name")
                                    or "final"
                                ),
                                "content": canonical_json(
                                    {
                                        "ok": False,
                                        "error": correction,
                                        "blocking_plan_adversary": adversary_review,
                                    }
                                ),
                            }
                        else:
                            messages.append({"role": "assistant", "content": response.content})
                            correction_message = {"role": "user", "content": correction}
                        messages.append(correction_message)
                        transcript.append(correction_message)
                        chain_delta_messages = [correction_message]
                        self._emit(
                            "blocking_guidance_final_rejected",
                            step=step,
                            verdict=verdict,
                            required_next_actions=required_actions,
                        )
                        if scripted_final_block_fail_fast and scripted_force_final:
                            self._emit(
                                "scripted_guidance_final_blocked",
                                step=step,
                                verdict=verdict,
                                required_next_actions=required_actions,
                            )
                            break
                        continue
                outcome = self._final_outcome(
                    action,
                    started=started,
                    step=step,
                    usage=usage,
                    transcript=transcript,
                    errors=errors,
                    search_calls=search_calls,
                    page_opens=page_opens,
                    find_calls=find_calls,
                    retrieved_chars=retrieved_chars,
                    external_model_calls=external_model_calls,
                )
                self._emit(
                    "trial_final",
                    step=step,
                    status=outcome.status,
                    confidence=outcome.confidence,
                )
                return outcome

            action_fingerprint = canonical_json(
                {"action": action.action, "payload": action.payload}
            )
            duplicate_action = (
                semantic_duplicate_action or action_fingerprint == last_action_fingerprint
            )
            if duplicate_action:
                consecutive_duplicate_actions += 1
            else:
                last_action_fingerprint = action_fingerprint
                consecutive_duplicate_actions = 0
            budget_violation = self._action_budget_violation(
                action,
                search_calls=search_calls,
                page_opens=page_opens,
                find_calls=find_calls,
                retrieved_chars=retrieved_chars,
                external_model_calls=external_model_calls,
            )
            if force_final_this_turn:
                forced_nonfinal_rejections += 1
            else:
                forced_nonfinal_rejections = 0
            rescue_threshold = self.agent_config.automatic_finalization_rescue_after_rejections
            rescue_seconds = self.agent_config.automatic_finalization_rescue_after_seconds
            elapsed_seconds = time.perf_counter() - started
            seconds_since_external_completion = time.perf_counter() - last_external_completion_at
            forced_rescue_due = bool(
                force_final_this_turn
                and rescue_threshold > 0
                and forced_nonfinal_rejections >= rescue_threshold
            )
            time_rescue_due = bool(
                rescue_seconds > 0 and seconds_since_external_completion >= rescue_seconds
            )
            if (
                (forced_rescue_due or time_rescue_due)
                and not automatic_finalization_rescue_attempted
                and self.external_model is not None
                and self.external_model_config.enabled
                and external_model_calls < self.external_model_config.max_calls_per_task
            ):
                automatic_finalization_rescue_attempted = True
                self._emit(
                    "automatic_finalization_rescue_started",
                    step=step,
                    reason="forced_final_rejection" if forced_rescue_due else "wall_clock",
                    rejection_count=forced_nonfinal_rejections,
                    elapsed_seconds=elapsed_seconds,
                    seconds_since_external_completion=seconds_since_external_completion,
                )
                rescue_task = asyncio.create_task(
                    self._automatic_external_finalization(
                        question=question,
                        response=response,
                        messages=messages,
                        transcript=transcript,
                        notes=notes,
                        request_namespace=external_namespace,
                        request_budget=(
                            self.external_model_config.max_calls_per_task - external_model_calls
                        ),
                    )
                )
                rescue_started = time.perf_counter()
                try:
                    while True:
                        done, _ = await asyncio.wait({rescue_task}, timeout=15)
                        if rescue_task in done:
                            rescue_action, rescue_result = rescue_task.result()
                            break
                        self._emit(
                            "automatic_finalization_rescue_wait",
                            step=step,
                            elapsed_seconds=time.perf_counter() - rescue_started,
                        )
                except BaseException:
                    rescue_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await rescue_task
                    raise
                external_model_calls += int(rescue_result.get("attempted") or 1)
                last_external_completion_at = time.perf_counter()
                transcript.append(
                    {
                        "role": "assistant",
                        "name": "external_finalization_rescue",
                        "content": str(rescue_result.get("content") or ""),
                        "response_metadata": {
                            "request_id": rescue_result.get("request_id"),
                            "attempted": rescue_result.get("attempted"),
                            "review_request_ids": rescue_result.get("review_request_ids"),
                        },
                    }
                )
                if rescue_action is not None:
                    answer_errors = self._surface_answer_constraint_errors(
                        question,
                        str(rescue_action.payload.get("exact_answer") or ""),
                    )
                    answer_errors.extend(
                        self._answer_type_constraint_errors(
                            question,
                            str(rescue_action.payload.get("exact_answer") or ""),
                        )
                    )
                    evidence_errors: list[str] = []
                    if self.agent_config.require_opened_citation_support:
                        evidence_errors = self._final_evidence_constraint_errors(
                            question,
                            str(rescue_action.payload.get("exact_answer") or ""),
                            rescue_action.payload.get("citations") or [],
                            opened,
                        )
                    geo_errors = self._geo_final_constraint_errors(
                        question,
                        str(rescue_action.payload.get("exact_answer") or ""),
                        geo_evidence,
                    )
                    if answer_errors or evidence_errors or geo_errors:
                        violations = [
                            *(f"answer constraint: {error}" for error in answer_errors),
                            *(f"evidence constraint: {error}" for error in evidence_errors),
                            *(f"geospatial constraint: {error}" for error in geo_errors),
                        ]
                        pending_surface_constraint_correction = (
                            "The independent finalizer proposed an answer that fails the same "
                            "constraints as a normal final answer: "
                            + "; ".join(violations)
                            + ". Do not accept or rationalize that candidate."
                        )
                        errors.append(pending_surface_constraint_correction)
                        rescue_result = {
                            **rescue_result,
                            "ok": False,
                            "error": pending_surface_constraint_correction,
                            "answer_constraint_violations": answer_errors,
                            "evidence_constraint_violations": evidence_errors,
                            "geo_constraint_violations": geo_errors,
                        }
                        rescue_action = None
                        self._emit(
                            "automatic_finalization_rescue_rejected",
                            step=step,
                            violations=violations,
                            result=rescue_result,
                        )
                if (
                    rescue_action is not None
                    and blocking_guidance_adversary
                    and initial_guidance
                    and self.external_model is not None
                    and self.external_model_config.enabled
                ):
                    final_fingerprint = canonical_json(rescue_action.payload)
                    adversary_review = guidance_final_reviews.get(final_fingerprint)
                    if adversary_review is None:
                        self._emit(
                            "blocking_guidance_adversary_started",
                            step=step,
                            phase="final_rescue",
                        )
                        adversary_review = await self._blocking_guidance_adversary_review(
                            question=question,
                            guidance=adversary_guidance,
                            messages=messages,
                            notes=notes,
                            opened=opened,
                            search_queries=search_query_history,
                            request_namespace=external_namespace,
                            phase="final_rescue",
                            proposed_final=rescue_action.payload,
                        )
                        guidance_final_reviews[final_fingerprint] = adversary_review
                        external_model_calls += int(adversary_review.get("attempted") or 1)
                        last_external_completion_at = time.perf_counter()
                        transcript.append(
                            {
                                "role": "assistant",
                                "name": "blocking_plan_adversary",
                                "content": canonical_json(adversary_review),
                            }
                        )
                    verdict = str(adversary_review.get("verdict") or "BLOCK").upper()
                    self._emit(
                        "blocking_guidance_adversary_completed",
                        step=step,
                        phase="final_rescue",
                        verdict=verdict,
                        request_id=adversary_review.get("request_id"),
                        required_next_actions=adversary_review.get("required_next_actions"),
                        review=adversary_review,
                    )
                    if verdict != "PASS":
                        required_actions = adversary_review.get("required_next_actions") or []
                        pending_surface_constraint_correction = (
                            "The blocking plan-adherence adversary rejected the independent "
                            "finalizer's proposal. "
                            + str(
                                adversary_review.get("reason")
                                or "Material plan steps remain unverified."
                            )
                            + " Required next actions: "
                            + canonical_json(required_actions)
                            + ". Complete those actions using public evidence before finalizing."
                        )
                        errors.append(pending_surface_constraint_correction)
                        rescue_result = {
                            **rescue_result,
                            "ok": False,
                            "error": pending_surface_constraint_correction,
                            "blocking_plan_adversary": adversary_review,
                        }
                        rescue_action = None
                        self._emit(
                            "automatic_finalization_rescue_rejected",
                            step=step,
                            violations=[pending_surface_constraint_correction],
                            result=rescue_result,
                        )
                if rescue_action is not None:
                    outcome = self._final_outcome(
                        rescue_action,
                        started=started,
                        step=step,
                        usage=usage,
                        transcript=transcript,
                        errors=errors,
                        search_calls=search_calls,
                        page_opens=page_opens,
                        find_calls=find_calls,
                        retrieved_chars=retrieved_chars,
                        external_model_calls=external_model_calls,
                    )
                    self._emit(
                        "automatic_finalization_rescue_completed",
                        step=step,
                        status=outcome.status,
                        request_id=rescue_result.get("request_id"),
                        result=rescue_result,
                    )
                    return outcome
                errors.append(
                    "Automatic finalization rescue failed: "
                    + str(rescue_result.get("error") or rescue_result.get("content") or "unknown")
                )
                self._emit(
                    "automatic_finalization_rescue_failed",
                    step=step,
                    error=errors[-1],
                    result=rescue_result,
                )
            automatic_page_inspection_succeeded = False
            consensus: dict[str, Any] | None = None
            force_final_after_external_consensus = False
            try:
                if duplicate_action:
                    remaining_page_budget = max(
                        0,
                        self.agent_config.max_page_opens - page_opens,
                    )
                    recovery_urls = self._unopened_candidate_urls(
                        last_successful_search_result or {},
                        opened=opened,
                        limit=min(
                            self.agent_config.automatic_page_inspection_count,
                            remaining_page_budget,
                        ),
                    )
                    if (
                        action.action in {"search", "search_many"}
                        and recovery_urls
                        and consecutive_duplicate_actions
                        < self.agent_config.max_consecutive_duplicate_actions
                    ):
                        self._emit(
                            "duplicate_action_recovery_started",
                            step=step,
                            action=action.action,
                            duplicate_count=consecutive_duplicate_actions,
                            url_count=len(recovery_urls),
                        )
                        page_result, page_deltas = await self._execute_action(
                            AgentAction(
                                action="open_many",
                                payload={
                                    "urls": recovery_urls,
                                    "max_chars": (
                                        self.agent_config.automatic_page_inspection_max_chars
                                    ),
                                },
                            ),
                            opened,
                            notes,
                            request_namespace=external_namespace,
                        )
                        for page in page_result.get("pages") or []:
                            if isinstance(page, dict) and isinstance(page.get("links"), list):
                                page["links"] = page["links"][:20]
                        result = {
                            "ok": bool(page_result.get("ok")),
                            "repeated_action": True,
                            "duplicate_count": consecutive_duplicate_actions,
                            "controller_recovery": (
                                "The repeated search was not reissued. Fresh candidate pages "
                                "from the prior discovery batch were inspected instead."
                            ),
                            "automatic_page_inspection": page_result,
                            "next_action_guidance": (
                                "Use this page evidence to refine the answer. Do not repeat the "
                                "same search action."
                            ),
                        }
                        deltas = page_deltas
                        automatic_page_inspection_succeeded = bool(page_result.get("ok"))
                        self._emit(
                            "duplicate_action_recovery_completed",
                            step=step,
                            action=action.action,
                            duplicate_count=consecutive_duplicate_actions,
                            succeeded=int(page_result.get("succeeded") or 0),
                            failed=int(page_result.get("failed") or 0),
                        )
                    else:
                        if (
                            consecutive_duplicate_actions
                            >= self.agent_config.max_consecutive_duplicate_actions
                        ):
                            force_final = True
                        raise RuntimeError(
                            "Identical action already executed "
                            f"{consecutive_duplicate_actions} consecutive time(s). "
                            "Use existing evidence and return the final answer now."
                        )
                elif budget_violation:
                    force_final = True
                    raise RuntimeError(budget_violation)
                else:
                    result, deltas = await self._execute_action(
                        action,
                        opened,
                        notes,
                        request_namespace=external_namespace,
                    )
                    if action.action == "geo_search" and result.get("ok"):
                        geo_evidence.append(result)
                    if action.action in {"search", "search_many"}:
                        result = self._filter_query_mirror_search_results(question, result)
                        search_query_history.extend(self._search_queries(action))
                    if action.action in {"search", "search_many"} and result.get("ok"):
                        last_successful_search_result = result
                if redundant_search_queries:
                    result["suppressed_redundant_queries"] = redundant_search_queries
                    result["search_novelty_guidance"] = (
                        "These query variants were already attempted and were not reissued. Change "
                        "the entity-relation pair, source vocabulary, language, or source type; do "
                        "not merely alter quotation marks or date filters."
                    )
                if strategy_recovery is not None:
                    result["external_search_strategy_recovery"] = strategy_recovery
                projected_search_calls = search_calls + deltas[0]
                should_inspect_pages = strategy_recovery is not None or (
                    self._should_automatically_inspect_pages(
                        action=action,
                        action_result=result,
                        search_streak=search_streak,
                    )
                )
                if should_inspect_pages:
                    remaining_page_budget = max(
                        0,
                        self.agent_config.max_page_opens - page_opens,
                    )
                    inspection_limit = min(
                        (
                            self.agent_config.max_batch_size
                            if strategy_recovery is not None
                            else self.agent_config.automatic_page_inspection_count
                        ),
                        remaining_page_budget,
                    )
                    candidate_urls = (
                        self._strategy_candidate_urls(result, inspection_limit)
                        if strategy_recovery is not None
                        else self._candidate_urls(result, inspection_limit)
                    )
                    if candidate_urls:
                        self._emit(
                            "automatic_page_inspection_started",
                            step=step,
                            url_count=len(candidate_urls),
                        )
                        page_result, page_deltas = await self._execute_action(
                            AgentAction(
                                action="open_many",
                                payload={
                                    "urls": candidate_urls,
                                    "max_chars": (
                                        self.agent_config.automatic_page_inspection_max_chars
                                    ),
                                },
                            ),
                            opened,
                            notes,
                            request_namespace=external_namespace,
                        )
                        for page in page_result.get("pages") or []:
                            if isinstance(page, dict) and isinstance(page.get("links"), list):
                                page["links"] = page["links"][:20]
                        result["automatic_page_inspection"] = page_result
                        deltas = tuple(
                            left + right for left, right in zip(deltas, page_deltas, strict=True)
                        )
                        automatic_page_inspection_succeeded = bool(page_result.get("ok"))
                        self._emit(
                            "automatic_page_inspection_completed",
                            step=step,
                            succeeded=int(page_result.get("succeeded") or 0),
                            failed=int(page_result.get("failed") or 0),
                            retrieved_chars=page_deltas[3],
                        )
                if self._should_automatically_consult_external(
                    action=action,
                    action_result=result,
                    search_calls=projected_search_calls,
                    external_model_calls=external_model_calls,
                    already_attempted=automatic_external_attempted,
                ):
                    automatic_external_attempted = True
                    request_count = min(
                        self.agent_config.automatic_external_requests,
                        self.external_model_config.max_calls_per_task - external_model_calls,
                    )
                    self._emit(
                        "automatic_external_started",
                        step=step,
                        request_count=request_count,
                        search_calls=projected_search_calls,
                    )
                    consultation_task = asyncio.create_task(
                        self._automatic_external_consultations(
                            question=question,
                            current_evidence=result,
                            notes=notes,
                            request_namespace=external_namespace,
                            request_count=request_count,
                        )
                    )
                    consultation_started = time.perf_counter()
                    try:
                        while True:
                            done, _ = await asyncio.wait({consultation_task}, timeout=15)
                            if consultation_task in done:
                                consultations = consultation_task.result()
                                break
                            self._emit(
                                "automatic_external_wait",
                                step=step,
                                elapsed_seconds=time.perf_counter() - consultation_started,
                                request_count=request_count,
                            )
                    except Exception as exc:  # noqa: BLE001 - search evidence remains usable
                        consultations = [
                            {
                                "ok": False,
                                "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        ]
                        errors.append(f"Step {step} automatic external consultation error: {exc}")
                    except BaseException:
                        consultation_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await consultation_task
                        raise
                    external_model_calls += request_count
                    consultation_chars = sum(
                        len(str(item.get("content") or "")) for item in consultations
                    )
                    deltas = (deltas[0], deltas[1], deltas[2], deltas[3] + consultation_chars)
                    result["independent_external_consultation"] = {
                        "attempted": request_count,
                        "consultations": consultations,
                        "instruction": (
                            "Use these independent candidate, adversarial, and search-strategy "
                            "reviews as leads. Preserve every viable entity in a candidate-by-clue "
                            "ledger, marking each clue directly supported, inferred, unknown, or "
                            "contradicted. Verify material claims against public-web evidence and "
                            "do not collapse to one entity merely because it was proposed first."
                        ),
                    }
                    remaining_search_budget = max(
                        0,
                        self.agent_config.max_search_calls - (search_calls + deltas[0]),
                    )
                    consultation_strategy_queries = self._consultation_strategy_queries(
                        consultations,
                        prior_queries=search_query_history,
                        limit=min(self.agent_config.max_batch_size, remaining_search_budget),
                    )
                    strategy_candidate_urls: list[str] = []
                    evidence_pages: list[dict[str, Any]] = []
                    if consultation_strategy_queries:
                        self._emit(
                            "external_strategy_search_started",
                            step=step,
                            query_count=len(consultation_strategy_queries),
                        )
                        strategy_search, strategy_search_deltas = await self._execute_action(
                            AgentAction(
                                action="search_many",
                                payload={"queries": consultation_strategy_queries},
                            ),
                            opened,
                            notes,
                            request_namespace=external_namespace,
                        )
                        search_query_history.extend(consultation_strategy_queries)
                        result["independent_external_consultation"]["strategy_search"] = (
                            strategy_search
                        )
                        deltas = tuple(
                            left + right
                            for left, right in zip(
                                deltas,
                                strategy_search_deltas,
                                strict=True,
                            )
                        )
                        if strategy_search.get("ok"):
                            last_successful_search_result = strategy_search
                            strategy_candidate_urls = self._strategy_candidate_urls(
                                strategy_search,
                                self.agent_config.max_batch_size,
                            )
                        self._emit(
                            "external_strategy_search_completed",
                            step=step,
                            succeeded=int(strategy_search.get("succeeded") or 0),
                            failed=int(strategy_search.get("failed") or 0),
                            returned_urls=len(strategy_candidate_urls),
                        )
                    remaining_page_budget = max(
                        0,
                        self.agent_config.max_page_opens - page_opens - deltas[1],
                    )
                    external_urls = self._external_consultation_urls(
                        consultations,
                        question=question,
                        opened=opened,
                        limit=min(
                            self.agent_config.automatic_page_inspection_count,
                            remaining_page_budget,
                        ),
                    )
                    source_inspection_limit = min(
                        (
                            self.agent_config.max_batch_size
                            if strategy_candidate_urls
                            else self.agent_config.automatic_page_inspection_count
                        ),
                        remaining_page_budget,
                    )
                    external_urls = list(dict.fromkeys(strategy_candidate_urls + external_urls))[
                        :source_inspection_limit
                    ]
                    if external_urls:
                        self._emit(
                            "external_source_inspection_started",
                            step=step,
                            url_count=len(external_urls),
                        )
                        external_pages, external_page_deltas = await self._execute_action(
                            AgentAction(
                                action="open_many",
                                payload={
                                    "urls": external_urls,
                                    "max_chars": (
                                        self.agent_config.automatic_page_inspection_max_chars
                                    ),
                                },
                            ),
                            opened,
                            notes,
                            request_namespace=external_namespace,
                        )
                        evidence_pages.extend(
                            page
                            for page in external_pages.get("pages") or []
                            if isinstance(page, dict)
                        )
                        deltas = tuple(
                            left + right
                            for left, right in zip(
                                deltas,
                                external_page_deltas,
                                strict=True,
                            )
                        )
                        automatic_page_inspection_succeeded = (
                            automatic_page_inspection_succeeded or bool(external_pages.get("ok"))
                        )
                        self._emit(
                            "external_source_inspection_completed",
                            step=step,
                            succeeded=int(external_pages.get("succeeded") or 0),
                            failed=int(external_pages.get("failed") or 0),
                            retrieved_chars=external_page_deltas[3],
                        )
                        remaining_related_page_budget = max(
                            0,
                            self.agent_config.max_page_opens - page_opens - deltas[1],
                        )
                        related_urls = self._related_evidence_urls(
                            external_pages.get("pages") or [],
                            queries=consultation_strategy_queries,
                            opened=opened,
                            limit=min(
                                self.agent_config.automatic_page_inspection_count,
                                remaining_related_page_budget,
                            ),
                        )
                        for page in external_pages.get("pages") or []:
                            if isinstance(page, dict) and isinstance(page.get("links"), list):
                                page["links"] = page["links"][:20]
                        result["independent_external_consultation"]["source_page_inspection"] = (
                            external_pages
                        )
                        if related_urls:
                            self._emit(
                                "related_source_inspection_started",
                                step=step,
                                url_count=len(related_urls),
                            )
                            related_pages, related_page_deltas = await self._execute_action(
                                AgentAction(
                                    action="open_many",
                                    payload={
                                        "urls": related_urls,
                                        "max_chars": (
                                            self.agent_config.automatic_page_inspection_max_chars
                                        ),
                                    },
                                ),
                                opened,
                                notes,
                                request_namespace=external_namespace,
                            )
                            evidence_pages.extend(
                                page
                                for page in related_pages.get("pages") or []
                                if isinstance(page, dict)
                            )
                            for page in related_pages.get("pages") or []:
                                if isinstance(page, dict) and isinstance(page.get("links"), list):
                                    page["links"] = page["links"][:20]
                            result["independent_external_consultation"][
                                "related_source_page_inspection"
                            ] = related_pages
                            deltas = tuple(
                                left + right
                                for left, right in zip(
                                    deltas,
                                    related_page_deltas,
                                    strict=True,
                                )
                            )
                            automatic_page_inspection_succeeded = (
                                automatic_page_inspection_succeeded or bool(related_pages.get("ok"))
                            )
                            self._emit(
                                "related_source_inspection_completed",
                                step=step,
                                succeeded=int(related_pages.get("succeeded") or 0),
                                failed=int(related_pages.get("failed") or 0),
                                retrieved_chars=related_page_deltas[3],
                            )
                    evidence_highlights = self._evidence_highlights(
                        evidence_pages,
                        queries=consultation_strategy_queries,
                        limit=self.agent_config.max_batch_size,
                    )
                    if evidence_highlights:
                        # Keep this outer result field last so the bounded tool-result
                        # tail retains decisive passages in long research contexts.
                        result["verified_evidence_highlights"] = {
                            "instruction": (
                                "Reconcile these query-ranked passages against the candidate-by-"
                                "clue ledger before searching again or finalizing. They are direct "
                                "page excerpts, not an answer supplied by the controller."
                            ),
                            "passages": evidence_highlights,
                        }
                        self._emit(
                            "evidence_highlights_attached",
                            step=step,
                            passage_count=len(evidence_highlights),
                        )
                    consensus = self._external_answer_consensus(
                        consultations,
                        question=question,
                        inspected_pages=evidence_pages,
                    )
                    if consensus is not None:
                        surface_errors = self._surface_answer_constraint_errors(
                            question,
                            str(consensus["exact_answer"]),
                        )
                        if surface_errors:
                            result["independent_external_consultation"][
                                "answer_consensus_rejected"
                            ] = {
                                **consensus,
                                "violations": surface_errors,
                            }
                            self._emit(
                                "external_consensus_rejected",
                                step=step,
                                violations=surface_errors,
                            )
                        else:
                            result["independent_external_consultation"][
                                "evidence_backed_answer_consensus"
                            ] = {
                                **consensus,
                                "instruction": (
                                    "Two independent Star-2 research branches reached the same "
                                    "exact answer, and at least one source they cited was opened "
                                    "successfully and matched multiple terms from the original "
                                    "question. Reconcile this candidate with the supplied evidence, "
                                    "then return the final action on the next turn. Do not run "
                                    "another discovery search merely to restate the same hypothesis."
                                ),
                            }
                            force_final_after_external_consensus = True
                    self._emit(
                        "automatic_external_completed",
                        step=step,
                        request_count=request_count,
                        successful=sum(bool(item.get("ok")) for item in consultations),
                        returned_chars=consultation_chars,
                    )
                    last_external_completion_at = time.perf_counter()
                search_calls += deltas[0]
                page_opens += deltas[1]
                find_calls += deltas[2]
                retrieved_chars += deltas[3]
                if action.action == "ask_external_model":
                    attempted_external = int(result.get("attempted", 0))
                    external_model_calls += attempted_external
                    if attempted_external > 0:
                        automatic_external_attempted = True
                        last_external_completion_at = time.perf_counter()
                if (
                    force_final
                    and result.get("ok")
                    and not self._near_budget(
                        search_calls,
                        page_opens,
                        find_calls,
                        retrieved_chars,
                        external_model_calls,
                    )
                ):
                    force_final = False
                    self._emit(
                        "forced_final_released_after_new_evidence",
                        step=step,
                        action=action.action,
                    )
                if force_final_after_external_consensus:
                    assert consensus is not None
                    force_final = True
                    self._emit(
                        "external_consensus_finalization_requested",
                        step=step,
                        exact_answer=str(consensus["exact_answer"]),
                        agreement_count=int(consensus["agreement_count"]),
                        supporting_citation_count=len(consensus["supporting_citations"]),
                    )
                self._check_budgets(
                    search_calls,
                    page_opens,
                    find_calls,
                    retrieved_chars,
                    external_model_calls,
                )
            except (
                SearchError,
                BrowserError,
                ExternalModelError,
                GeoResearchError,
                ValueError,
                RuntimeError,
            ) as exc:
                result = {"ok": False, "error": str(exc), "action": action.action}
                errors.append(f"Step {step} {action.action} error: {exc}")

            if action.action in {"search", "search_many"}:
                if result.get("ok"):
                    if automatic_page_inspection_succeeded:
                        search_streak = 0
                        require_open = False
                        result["next_action_guidance"] = (
                            "Inspect the attached page evidence before deciding whether more "
                            "discovery is necessary."
                        )
                    else:
                        search_streak += 1
                    if search_streak >= 2 and self._result_has_urls(result):
                        require_open = True
                        result["next_action_requirement"] = (
                            "Open one or more candidate result URLs before running another search."
                        )
                elif duplicate_action and search_streak >= 1:
                    require_open = True
                    result["next_action_requirement"] = (
                        "Do not repeat the search. Open a candidate URL or finalize from existing evidence."
                    )
            elif action.action in {"open", "open_many", "geo_search"} and result.get("ok"):
                require_open = False
                search_streak = 0

            if pending_surface_constraint_correction is not None:
                result["surface_constraint_rejection"] = {
                    "ok": False,
                    "error": pending_surface_constraint_correction,
                }
                pending_surface_constraint_correction = None

            result_text = truncate_middle(json.dumps(result, ensure_ascii=False), 80_000)
            if protocol == "tools" and assistant_message.get("tool_calls"):
                messages.append(assistant_message)
                tool_call = assistant_message["tool_calls"][0]
                original_tool_name = str(
                    (tool_call.get("function") or {}).get("name") or action.action
                )
                result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call-{step}"),
                    "name": original_tool_name,
                    "content": result_text,
                }
                messages.append(result_message)
                transcript.append(result_message)
                chain_delta_messages = [result_message]
            else:
                messages.append({"role": "assistant", "content": response.content})
                result_message = {"role": "user", "content": f"Tool result:\n{result_text}"}
                messages.append(result_message)
                transcript.append(result_message)
                chain_delta_messages = [result_message]

            self._emit(
                "action_completed",
                step=step,
                action=action.action,
                ok=bool(result.get("ok")),
                search_calls=search_calls,
                page_opens=page_opens,
                find_calls=find_calls,
                retrieved_chars=retrieved_chars,
                external_model_calls=external_model_calls,
                result=result,
            )

            if result.get("ok") and action.action in {
                "search",
                "search_many",
                "open",
                "open_many",
                "find",
                "geo_search",
                "ask_external_model",
                "note",
            }:
                evidence_journal.append(
                    {
                        "step": step,
                        "scripted_step_id": (
                            current_scripted_step.get("id")
                            if current_scripted_step is not None
                            else None
                        ),
                        "action": action.action,
                        "request": truncate_middle(canonical_json(action.payload), 4_000),
                        "result": truncate_middle(
                            canonical_json(result),
                            30_000 if action.action in {"ask_external_model", "geo_search"} else 16_000,
                        ),
                    }
                )
                if len(evidence_journal) > 40:
                    del evidence_journal[:-40]

            if current_scripted_step is not None:
                if result.get("ok") or current_scripted_step.get("advance_on_attempt") is True:
                    self._emit(
                        "scripted_guidance_step_completed",
                        step=step,
                        action=action.action,
                        scripted_step_index=scripted_step_index,
                        action_succeeded=bool(result.get("ok")),
                    )
                    scripted_step_index += 1
                    scripted_step_attempts = 0
                else:
                    self._emit(
                        "scripted_guidance_step_retry",
                        step=step,
                        action=action.action,
                        scripted_step_index=scripted_step_index,
                        error=result.get("error"),
                    )

            if self._near_budget(
                search_calls,
                page_opens,
                find_calls,
                retrieved_chars,
                external_model_calls,
            ):
                force_final = True
                warning = (
                    "You are near or at the browsing budget. Use the evidence already collected "
                    "and return a final action now."
                )
                warning_message = {"role": "user", "content": warning}
                messages.append(warning_message)
                transcript.append(warning_message)
                if chain_delta_messages is not None:
                    chain_delta_messages.append(warning_message)

        outcome = AgentOutcome(
            response_text="Explanation: Agent did not produce a valid final answer.\nExact Answer: \nConfidence: 0%",
            exact_answer=None,
            explanation="Agent did not produce a valid final answer.",
            confidence=0.0,
            citations=[],
            status="no_final",
            steps=min(
                self.agent_config.max_steps,
                len([x for x in transcript if x.get("role") == "assistant"]),
            ),
            search_calls=search_calls,
            page_opens=page_opens,
            find_calls=find_calls,
            retrieved_chars=retrieved_chars,
            duration_seconds=time.perf_counter() - started,
            usage=usage,
            external_model_calls=external_model_calls,
            transcript=transcript,
            errors=errors,
        )
        self._emit("trial_no_final", status=outcome.status, errors=errors[-3:])
        return outcome

    async def _query(
        self,
        messages: list[dict[str, Any]],
        protocol: str,
        *,
        extra_body: dict[str, Any] | None = None,
        force_final: bool = False,
        require_open: bool = False,
        required_action: str | None = None,
        request_headers: dict[str, str] | None = None,
    ):
        if protocol in {"tools", "auto"}:
            tool_choice: str | dict[str, Any] = "auto"
            tools = tool_schemas(
                include_external_model=(
                    self.external_model_config.enabled and self.external_model is not None
                )
            )
            if required_action is not None:
                tools = [
                    tool
                    for tool in tools
                    if (tool.get("function") or {}).get("name") == required_action
                ]
                if not tools:
                    raise ModelAPIError(
                        f"Required scripted tool is unavailable: {required_action}"
                    )
                tool_choice = {
                    "type": "function",
                    "function": {"name": required_action},
                }
            elif force_final:
                tools = [
                    tool for tool in tools if (tool.get("function") or {}).get("name") == "final"
                ]
                tool_choice = {"type": "function", "function": {"name": "final"}}
            elif require_open:
                # Keep the caller-owned tool schema stable across response-chain
                # turns. The production Agent backend can deliberate indefinitely
                # when a previously visible tool disappears mid-task. Evidence
                # discipline is communicated in the preceding tool result instead.
                pass
            return await self.client.chat(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                extra_body=extra_body,
                request_headers=request_headers,
            )
        return await self.client.chat(
            messages,
            extra_body=extra_body,
            request_headers=request_headers,
        )

    @staticmethod
    def _routing_headers(
        namespace_material: str,
        *,
        routing_backend_pool: list[str] | None = None,
    ) -> dict[str, str]:
        chain_namespace = hashlib.sha256(namespace_material.encode("utf-8")).hexdigest()[:24]
        headers = {"X-FRL-Conversation-Id": f"bc250-{chain_namespace}"}
        if routing_backend_pool:
            backend_index = int(
                hashlib.sha256(namespace_material.encode("utf-8")).hexdigest()[:16],
                16,
            ) % len(routing_backend_pool)
            headers["X-FRL-Require-Backend"] = routing_backend_pool[backend_index]
        benchmark_namespace = _BENCHMARK_REQUEST_NAMESPACE.fullmatch(namespace_material)
        if benchmark_namespace is None:
            return headers
        suffix = benchmark_namespace.group("suffix")
        cohort_material = benchmark_namespace.group("run")
        if suffix:
            cohort_material += ":star2-helpers"
        cohort_id = hashlib.sha256(cohort_material.encode("utf-8")).hexdigest()[:20]
        cohort_index = (
            int(hashlib.sha256(namespace_material.encode("utf-8")).hexdigest()[:12], 16)
            if suffix
            else int(benchmark_namespace.group("rank"))
        )
        headers.update(
            {
                "X-FRL-KV-Cohort-Id": f"bc250-{cohort_id}",
                "X-FRL-KV-Cohort-Index": str(cohort_index),
            }
        )
        return headers

    @staticmethod
    def _result_has_urls(result: dict[str, Any]) -> bool:
        if isinstance(result.get("results"), list):
            return any(isinstance(row, dict) and row.get("url") for row in result["results"])
        searches = result.get("searches")
        if not isinstance(searches, list):
            return False
        return any(
            isinstance(row, dict)
            and isinstance(row.get("results"), list)
            and any(
                isinstance(result_row, dict) and result_row.get("url")
                for result_row in row["results"]
            )
            for row in searches
        )

    @staticmethod
    def _search_queries(action: AgentAction) -> list[str]:
        if action.action == "search":
            query = action.payload.get("query")
            return [str(query).strip()] if isinstance(query, str) and query.strip() else []
        if action.action == "search_many":
            return [
                str(query).strip()
                for query in action.payload.get("queries") or []
                if isinstance(query, str) and query.strip()
            ]
        return []

    @staticmethod
    def _search_query_terms(query: str) -> frozenset[str]:
        without_ranges = _SEARCH_DATE_RANGE.sub(" ", query.casefold())
        return frozenset(_SEARCH_TOKEN.findall(without_ranges))

    @classmethod
    def _search_query_is_redundant(cls, query: str, prior_queries: list[str]) -> bool:
        terms = cls._search_query_terms(query)
        if not terms:
            return True
        for prior_query in prior_queries:
            prior_terms = cls._search_query_terms(prior_query)
            if terms == prior_terms:
                return True
            union = terms | prior_terms
            if len(union) >= 5 and len(terms & prior_terms) / len(union) >= 0.9:
                return True
        return False

    @classmethod
    def _filter_redundant_search_action(
        cls,
        action: AgentAction,
        prior_queries: list[str],
    ) -> tuple[AgentAction | None, list[str]]:
        queries = cls._search_queries(action)
        if not queries:
            return action, []
        accepted: list[str] = []
        redundant: list[str] = []
        comparison_queries = list(prior_queries)
        for query in queries:
            if cls._search_query_is_redundant(query, comparison_queries):
                redundant.append(query)
                continue
            accepted.append(query)
            comparison_queries.append(query)
        if not accepted:
            return None, redundant
        payload = dict(action.payload)
        if action.action == "search":
            payload["query"] = accepted[0]
        else:
            payload["queries"] = accepted
        return AgentAction(action=action.action, payload=payload), redundant

    @staticmethod
    def _json_objects(text: str) -> list[dict[str, Any]]:
        decoder = json.JSONDecoder()
        objects: list[dict[str, Any]] = []
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                objects.append(value)
        if objects or r"\"" not in text:
            return objects
        start = text.find(r"{\"")
        end = text.rfind("}")
        if start >= 0 and end > start:
            marker = "__BC250_INNER_QUOTE__"
            escaped = text[start : end + 1]
            normalized = re.sub(r'\\{3}"', marker, escaped)
            normalized = normalized.replace(r"\"", '"').replace(marker, r"\"")
            try:
                value = json.loads(normalized)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(value, dict):
                    objects.append(value)
        return objects

    @classmethod
    def _strategy_queries_from_result(
        cls,
        result: dict[str, Any],
        *,
        prior_queries: list[str],
        limit: int,
    ) -> list[str]:
        if limit <= 0 or not result.get("ok"):
            return []
        candidates: list[str] = []
        for value in cls._json_objects(str(result.get("content") or "")):
            queries = value.get("queries")
            if isinstance(queries, list):
                candidates.extend(str(query).strip() for query in queries if str(query).strip())
        fallback_queries = result.get("agent_search_queries")
        if isinstance(fallback_queries, list):
            candidates.extend(
                str(query).strip() for query in fallback_queries if str(query).strip()
            )
        accepted: list[str] = []
        comparison_queries = list(prior_queries)
        for query in candidates:
            if _STRATEGY_PLACEHOLDER_QUERY.fullmatch(query.strip()):
                continue
            if cls._search_query_is_redundant(query, comparison_queries):
                continue
            accepted.append(query)
            comparison_queries.append(query)
            if len(accepted) >= limit:
                break
        return accepted

    @classmethod
    def _consultation_strategy_queries(
        cls,
        consultations: list[dict[str, Any]],
        *,
        prior_queries: list[str],
        limit: int,
    ) -> list[str]:
        accepted: list[str] = []
        comparison_queries = list(prior_queries)
        for consultation in consultations:
            if consultation.get("review_role") != "Search strategy specialist":
                continue
            queries = cls._strategy_queries_from_result(
                consultation,
                prior_queries=comparison_queries,
                limit=limit - len(accepted),
            )
            accepted.extend(queries)
            comparison_queries.extend(queries)
            if len(accepted) >= limit:
                break
        return accepted

    @staticmethod
    def _attach_blocking_guidance_review(
        messages: list[dict[str, Any]], review: dict[str, Any]
    ) -> dict[str, Any]:
        """Attach a synchronous review to the pending tool result when possible."""
        if messages and messages[-1].get("role") == "tool":
            pending = messages[-1]
            content = str(pending.get("content") or "")
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                payload = {"prior_tool_result": truncate_middle(content, 60_000)}
            if not isinstance(payload, dict):
                payload = {"prior_tool_result": payload}
            payload["blocking_plan_adversary"] = review
            pending["content"] = truncate_middle(
                json.dumps(payload, ensure_ascii=False),
                80_000,
            )
            if str(review.get("verdict") or "").upper() != "BLOCK":
                return pending

        controller_message = {
            "role": "user",
            "content": (
                "Synchronous blocking plan-adherence review:\n"
                + canonical_json(review)
                + "\nThis checkpoint is blocked. The next substantive action must execute a "
                "required_next_action that repairs the stated deviation. Do not merely "
                "acknowledge the review, repeat the blocked action, or continue broad discovery."
            ),
        }
        messages.append(controller_message)
        return controller_message

    @staticmethod
    def _action_repairs_blocking_guidance_review(
        action: AgentAction,
        review: dict[str, Any],
    ) -> bool:
        required = review.get("required_next_actions")
        if not isinstance(required, list) or not required:
            return action.action != "final"
        instruction = str(required[0]).strip().casefold()
        if re.search(r"\b(open|inspect|read|visit|fetch)\b", instruction):
            return action.action in {"open", "open_many"}
        if re.search(r"\b(search|query|look up|discover|falsif|find source)\b", instruction):
            return action.action in {"search", "search_many", "geo_search"}
        if re.search(r"\b(consult|external|ask)\b", instruction):
            return action.action == "ask_external_model"
        if re.search(r"\b(note|record|ledger|save)\b", instruction):
            return action.action == "note"
        return action.action != "final"

    @staticmethod
    def _action_matches_scripted_step(
        action: AgentAction,
        scripted_step: dict[str, Any],
    ) -> bool:
        allowed = scripted_step.get("allowed_actions")
        if not isinstance(allowed, list) or not allowed:
            return False
        allowed_actions = {str(value).strip() for value in allowed if str(value).strip()}
        if action.action not in allowed_actions:
            return False
        minimum_batch_size = scripted_step.get("minimum_batch_size")
        if isinstance(minimum_batch_size, int) and minimum_batch_size > 0:
            batch_key = {
                "search_many": "queries",
                "open_many": "urls",
                "geo_search": "anchors",
                "ask_external_model": "requests",
            }.get(action.action)
            batch = action.payload.get(batch_key) if batch_key else None
            if not isinstance(batch, list) or len(batch) < minimum_batch_size:
                return False
        required_urls = scripted_step.get("required_urls")
        if isinstance(required_urls, list) and required_urls:
            supplied_urls: list[str] = []
            url = action.payload.get("url")
            if isinstance(url, str) and url.strip():
                supplied_urls.append(url.strip())
            urls = action.payload.get("urls")
            if isinstance(urls, list):
                supplied_urls.extend(
                    str(value).strip() for value in urls if str(value).strip()
                )

            def normalize(value: str) -> str:
                return value.strip().rstrip("/").casefold()

            supplied = {normalize(value) for value in supplied_urls}
            required = {normalize(str(value)) for value in required_urls if str(value).strip()}
            if not required or not required.issubset(supplied):
                return False

        required_queries = scripted_step.get("required_queries")
        if isinstance(required_queries, list) and required_queries:
            supplied_queries: list[str] = []
            query = action.payload.get("query")
            if isinstance(query, str) and query.strip():
                supplied_queries.append(query.strip())
            queries = action.payload.get("queries")
            if isinstance(queries, list):
                supplied_queries.extend(
                    str(value).strip() for value in queries if str(value).strip()
                )

            def normalize_query(value: str) -> str:
                return " ".join(value.split()).casefold()

            supplied_query_set = {normalize_query(value) for value in supplied_queries}
            required_query_set = {
                normalize_query(str(value))
                for value in required_queries
                if str(value).strip()
            }
            if not required_query_set or not required_query_set.issubset(supplied_query_set):
                return False
        return True

    @classmethod
    def _blocking_guidance_review_payload(cls, result: dict[str, Any]) -> dict[str, Any]:
        raw_content = str(result.get("content") or "")
        objects = cls._json_objects(raw_content)
        nested: list[dict[str, Any]] = []
        for value in objects:
            explanation = value.get("explanation")
            if isinstance(explanation, str):
                nested.extend(cls._json_objects(explanation))
        for value in [*nested, *objects]:
            verdict = str(value.get("verdict") or "").strip().upper()
            verdict = {
                "BLOCKED": "BLOCK",
                "ON TRACK": "ON_TRACK",
                "ON-TRACK": "ON_TRACK",
            }.get(verdict, verdict)
            if verdict not in {"PASS", "ON_TRACK", "CONTINUE", "BLOCK"}:
                continue
            required = value.get("required_next_actions")
            if not isinstance(required, list):
                required = []
            return {
                "ok": bool(result.get("ok")),
                "verdict": verdict,
                "observed_stage": str(value.get("observed_stage") or ""),
                "followed": value.get("followed") if isinstance(value.get("followed"), list) else [],
                "material_deviations": (
                    value.get("material_deviations")
                    if isinstance(value.get("material_deviations"), list)
                    else []
                ),
                "required_next_actions": [str(item) for item in required if str(item).strip()],
                "reason": str(value.get("reason") or ""),
                "request_id": result.get("request_id"),
                "attempted": 1,
                "raw_review_content": truncate_middle(raw_content, 8_000),
            }
        textual_verdict = re.search(
            r"(?i)\bverdict\b[\s`*_\"']*[:=\-][\s`*_\"']*"
            r"(PASS|ON[ _-]?TRACK|CONTINUE|BLOCK(?:ED)?)\b",
            raw_content,
        )
        if textual_verdict:
            verdict = textual_verdict.group(1).upper().replace(" ", "_").replace("-", "_")
            if verdict == "BLOCKED":
                verdict = "BLOCK"
            required = (
                [
                    "Execute the next unmet requirement in the supplied research plan, then "
                    "request another review."
                ]
                if verdict == "BLOCK"
                else []
            )
            return {
                "ok": bool(result.get("ok")),
                "verdict": verdict,
                "observed_stage": "textual_review_fallback",
                "followed": [],
                "material_deviations": (
                    [truncate_middle(raw_content, 4_000)] if verdict == "BLOCK" else []
                ),
                "required_next_actions": required,
                "reason": truncate_middle(raw_content, 4_000),
                "request_id": result.get("request_id"),
                "attempted": 1,
                "raw_review_content": truncate_middle(raw_content, 8_000),
            }
        return {
            "ok": False,
            "verdict": "BLOCK",
            "observed_stage": "review_failed",
            "followed": [],
            "material_deviations": [
                "The blocking adversary did not return a parseable verdict.",
                truncate_middle(raw_content, 4_000),
            ],
            "required_next_actions": [
                "Re-read the supplied plan, execute its next unmet requirement, and request "
                "another blocking review."
            ],
            "reason": str(
                result.get("error")
                or raw_content
                or "Invalid plan-adherence review response."
            ),
            "request_id": result.get("request_id"),
            "attempted": 1,
            "raw_review_content": truncate_middle(raw_content, 8_000),
        }

    async def _blocking_guidance_adversary_review(
        self,
        *,
        question: str,
        guidance: str,
        messages: list[dict[str, Any]],
        notes: list[str],
        opened: dict[str, PageDocument],
        search_queries: list[str],
        request_namespace: str,
        phase: str,
        proposed_final: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.external_model is None:
            return self._blocking_guidance_review_payload(
                {"ok": False, "error": "External adversary is unavailable"}
            )
        recent_trace: list[dict[str, Any]] = []
        for message in messages[-24:]:
            row: dict[str, Any] = {
                "role": message.get("role"),
                "name": message.get("name"),
                "content": truncate_middle(str(message.get("content") or ""), 6_000),
            }
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                row["tool_calls"] = tool_calls[:4]
            recent_trace.append(row)
        opened_sources = [
            {
                "url": document.final_url,
                "title": document.title,
                "text": truncate_middle(document.text, 2_500),
            }
            for document in list(opened.values())[-12:]
        ]
        context = canonical_json(
            {
                "phase": phase,
                "original_question": question,
                "research_plan": truncate_middle(guidance, 40_000),
                "executed_search_queries": search_queries[-60:],
                "recent_action_and_evidence_trace": recent_trace,
                "opened_sources": opened_sources,
                "saved_notes": notes[-12:],
                "proposed_final": proposed_final,
            }
        )
        phase_instruction = (
            "For a final review, return PASS when the proposed exact answer is directly supported "
            "by inspected public evidence, the identity chain is coherent, and no material clue "
            "contradicts it. Require candidate comparison and falsification, but do not block a "
            "correct directly supported answer merely because an incidental clue lacks a second "
            "redundant source. A mapped source explicitly marked gold or target controls the "
            "requested source-specific fact unless stronger direct evidence contradicts it."
            if phase.startswith("final")
            else "At this checkpoint, use ON_TRACK when the trajectory is materially following "
            "the plan and BLOCK when it is cycling, skipping a required stage, or failing to open "
            "and compare evidence."
        )
        requests = [
            {
                "task_mode": "review",
                "system": (
                    "You are a strict but evidence-calibrated blocking plan-adherence adversary. "
                    "You do not know the reference answer. Never search for benchmark dumps, "
                    "canaries, leaked questions, labels, or reference answers. Judge the actual "
                    "trajectory against the supplied plan. Equivalent high-quality actions may "
                    "satisfy a plan step; do not block for cosmetic ordering differences. Block "
                    "material omissions, repeated low-yield discovery, unsupported relation "
                    "bridges, missing candidate comparison, missing falsification, or a final not "
                    "entailed by opened evidence. Honor source roles explicitly designated by the "
                    "plan: a gold or target source controls source-specific attribution, while "
                    "other sources normally corroborate clues. Another source crediting a different "
                    "person with a different historical milestone is not by itself a contradiction. "
                    "Give concrete next actions, not vague criticism. "
                    + phase_instruction
                ),
                "query": (
                    "Return exactly one JSON object with this schema and no markdown: "
                    '{"verdict":"'
                    + ("PASS|BLOCK" if phase.startswith("final") else "ON_TRACK|BLOCK")
                    + '","observed_stage":"short stage",'
                    '"followed":["specific completed requirement"],'
                    '"material_deviations":["specific deviation"],'
                    '"required_next_actions":["specific executable action"],'
                    '"reason":"concise evidence-based reason"}. '
                    "Use PASS only for the final phase; use ON_TRACK or BLOCK at checkpoints."
                ),
                "context": context,
            }
        ]
        try:
            results = await self.external_model.ask_many(
                requests,
                request_namespace=request_namespace + f":blocking-plan-{phase}",
            )
        except Exception as exc:  # noqa: BLE001 - a failed verifier must fail closed
            return self._blocking_guidance_review_payload(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
        result = results[0] if results else {"ok": False, "error": "empty review response"}
        return self._blocking_guidance_review_payload(result)

    async def _automatic_external_query_strategy(
        self,
        *,
        question: str,
        messages: list[dict[str, Any]],
        notes: list[str],
        prior_queries: list[str],
        repeated_action: AgentAction,
        request_namespace: str,
        limit: int,
    ) -> tuple[list[str], dict[str, Any]]:
        if self.external_model is None or limit <= 0:
            return [], {"ok": False, "error": "external search strategy is unavailable"}
        recent_evidence = [
            truncate_middle(str(message.get("content") or ""), 20_000)
            for message in messages
            if message.get("role") in {"tool", "user"}
        ][-8:]
        context = canonical_json(
            {
                "question": question,
                "recent_evidence": recent_evidence,
                "saved_notes": notes[-10:],
                "prior_queries": prior_queries[-40:],
                "repeated_action": {
                    "action": repeated_action.action,
                    "payload": repeated_action.payload,
                },
            }
        )
        common_system = (
            "You are a query-strategy controller for an audited public-web research task. "
            "Do not search for benchmark dumps, canaries, leaked questions, or reference "
            "answers. The primary researcher is repeating low-yield searches. Do not "
            "inherit its leading hypothesis. Extract up to three viable underlying entities "
            "from all evidence and independent reviews, including the strongest alternative, "
            "then design genuinely different retrieval routes that discriminate among them "
            "and resolve the requested relation. Prefer entity-plus-role, attribution, "
            "history, source-language, primary-record, and contrastive-candidate queries. "
            "Treat constrained related entities as search anchors: when the target is "
            "unknown, identify the rarest collaborator, author, spouse, artifact, event, "
            "quotation, or dated source first and traverse that relation back to the target. "
            "Translate scholarly paraphrases into field vocabulary (for example curated "
            "database, experimentally validated, catalog, registry, corpus, revision, or "
            "open-access paper), then enumerate authors and test them against the other "
            "relations. A publication matching only one clue is candidate generation, not "
            "identity proof. For multi-anchor distance clues, use geo_search with the "
            "stated expected distances instead of search snippets or city-level pages. "
            "Do not infer nationality from birthplace or primary occupation from one artifact. "
            "For a historical attribution, put a broad subject history or origins query "
            "first; do not lead with answer-shaped wording such as 'first person credited.' "
            "Put the highest-yield unresolved-relation query first. Use one entity per "
            "query, no OR chains, and at most twelve terms per query. Do not "
            "paraphrase the full clue repeatedly, and do not create novelty by merely "
            "changing quotes, punctuation, or date ranges. Do not copy any phrase longer "
            "than three words from the question into a query. Return exactly one JSON "
            "object and no markdown."
        )
        strategy_roles = [
            (
                "Domain-vocabulary translator",
                "Infer several plausible technical, historical, cultural, institutional, or "
                "commercial domains behind the paraphrased clues. Translate each clue into the "
                "specialist nouns likely to occur in primary sources. Every query must add a "
                "domain term absent from the benchmark wording and must identify an artifact, "
                "source class, or named candidate rather than quote the clue.",
            ),
            (
                "Relation-graph inverter",
                "Treat the question as a graph. Start from the rarest constrained related entity "
                "or dated artifact, enumerate names, then traverse toward the requested target. "
                "Produce candidate-centric and source-centric queries that test different edges. "
                "Do not reuse the primary researcher's leading candidate unless a direct source "
                "in the supplied evidence supports at least two hard relations.",
            ),
        ]
        requests = [
            {
                "task_mode": "strategy",
                "system": f"{common_system}\n\nSpecialized role: {role}. {role_task}",
                "query": (
                    f"Return exactly this schema with up to {limit} concise public-web queries: "
                    '{"analysis":"brief diagnosis","entity_candidates":["candidate"],'
                    '"queries":["query 1","query 2"]}. Every query must test a different semantic '
                    "route and target the unresolved answer rather than re-verifying clues that are "
                    "already established. Include queries for the strongest alternative entity, not "
                    "only the researcher's current favorite."
                ),
                "context": context,
            }
            for role, role_task in strategy_roles
        ]
        results = await self.external_model.ask_many(
            requests,
            request_namespace=request_namespace + ":search-strategy-recovery",
        )
        queries: list[str] = []
        comparison_queries = list(prior_queries)
        for result in results:
            role_queries = self._strategy_queries_from_result(
                result,
                prior_queries=comparison_queries,
                limit=limit - len(queries),
            )
            queries.extend(role_queries)
            comparison_queries.extend(role_queries)
            if len(queries) >= limit:
                break
        combined = {
            "ok": bool(queries),
            "status": "succeeded" if queries else "failed",
            "attempted": len(results),
            "request_id": ",".join(str(result.get("request_id") or "") for result in results),
            "content": "\n".join(str(result.get("content") or "") for result in results),
            "results": results,
        }
        if not queries:
            combined["error"] = "strategy roles returned no novel queries"
        return queries, combined

    @staticmethod
    def _batched_action_size(action: AgentAction) -> int:
        key = {
            "search_many": "queries",
            "open_many": "urls",
            "geo_search": "anchors",
            "ask_external_model": "requests",
        }.get(action.action)
        values = action.payload.get(key) if key else None
        return len(values) if isinstance(values, list) else 1

    def _clip_action_to_remaining_budget(
        self,
        action: AgentAction,
        *,
        search_calls: int,
        page_opens: int,
        external_model_calls: int,
    ) -> tuple[AgentAction, int | None]:
        budget_shapes = {
            "search_many": (
                "queries",
                max(0, self.agent_config.max_search_calls - search_calls),
            ),
            "geo_search": (
                "anchors",
                max(0, self.agent_config.max_search_calls - search_calls),
            ),
            "open_many": (
                "urls",
                max(0, self.agent_config.max_page_opens - page_opens),
            ),
            "ask_external_model": (
                "requests",
                max(
                    0,
                    self.external_model_config.max_calls_per_task - external_model_calls,
                ),
            ),
        }
        shape = budget_shapes.get(action.action)
        if shape is None:
            return action, None
        key, remaining = shape
        values = action.payload.get(key)
        if not isinstance(values, list) or remaining <= 0 or len(values) <= remaining:
            return action, None
        payload = dict(action.payload)
        payload[key] = values[:remaining]
        return AgentAction(action=action.action, payload=payload), len(values)

    def _action_budget_violation(
        self,
        action: AgentAction,
        *,
        search_calls: int,
        page_opens: int,
        find_calls: int,
        retrieved_chars: int,
        external_model_calls: int = 0,
    ) -> str | None:
        search_delta = page_delta = find_delta = external_delta = 0
        if action.action == "search":
            search_delta = 1
        elif action.action == "search_many":
            search_delta = min(
                len(action.payload.get("queries") or []),
                self.agent_config.max_batch_size,
            )
        elif action.action == "geo_search":
            search_delta = min(len(action.payload.get("anchors") or []), 4)
        elif action.action == "open":
            page_delta = 1
        elif action.action == "open_many":
            page_delta = min(
                len(action.payload.get("urls") or []),
                self.agent_config.max_batch_size,
            )
        elif action.action == "find":
            find_delta = 1
        elif action.action == "ask_external_model":
            requests = action.payload.get("requests")
            external_delta = (
                min(len(requests), self.external_model_config.max_batch_size)
                if isinstance(requests, list)
                else 1
            )

        projected = {
            "search calls": (search_calls + search_delta, self.agent_config.max_search_calls),
            "page opens": (page_opens + page_delta, self.agent_config.max_page_opens),
            "find calls": (find_calls + find_delta, self.agent_config.max_find_calls),
            "external-model calls": (
                external_model_calls + external_delta,
                self.external_model_config.max_calls_per_task,
            ),
            "retrieved characters": (
                retrieved_chars,
                self.agent_config.max_retrieved_chars,
            ),
        }
        exceeded = [
            f"{name} {actual}>{limit}"
            for name, (actual, limit) in projected.items()
            if actual > limit
        ]
        if exceeded:
            return "Action was not executed because it would exceed the budget: " + ", ".join(
                exceeded
            )
        return None

    def _should_automatically_consult_external(
        self,
        *,
        action: AgentAction,
        action_result: dict[str, Any],
        search_calls: int,
        external_model_calls: int,
        already_attempted: bool,
    ) -> bool:
        threshold = self.agent_config.automatic_external_after_search_calls
        return bool(
            not already_attempted
            and threshold > 0
            and action.action in {"search", "search_many"}
            and action_result.get("ok")
            and search_calls >= threshold
            and self.external_model_config.enabled
            and self.external_model is not None
            and external_model_calls < self.external_model_config.max_calls_per_task
        )

    def _should_automatically_inspect_pages(
        self,
        *,
        action: AgentAction,
        action_result: dict[str, Any],
        search_streak: int,
    ) -> bool:
        threshold = self.agent_config.automatic_page_inspection_after_search_actions
        return bool(
            threshold > 0
            and action.action in {"search", "search_many"}
            and action_result.get("ok")
            and self._result_has_urls(action_result)
            and search_streak + 1 >= threshold
        )

    @staticmethod
    def _candidate_urls(result: dict[str, Any], limit: int) -> list[str]:
        if limit <= 0:
            return []
        batches: list[list[dict[str, Any]]] = []
        if isinstance(result.get("results"), list):
            batches.append([row for row in result["results"] if isinstance(row, dict)])
        for search in result.get("searches") or []:
            if isinstance(search, dict) and isinstance(search.get("results"), list):
                batches.append([row for row in search["results"] if isinstance(row, dict)])
        urls: list[str] = []
        seen: set[str] = set()
        for rank in range(max((len(batch) for batch in batches), default=0)):
            for batch in batches:
                if rank >= len(batch):
                    continue
                url = str(batch[rank].get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                urls.append(url)
                if len(urls) >= limit:
                    return urls
        return urls

    @classmethod
    def _strategy_candidate_urls(cls, result: dict[str, Any], limit: int) -> list[str]:
        # Preserve one evidence slot per strategy route before inspecting deeper
        # ranks from any route. External strategists can order imperfectly, and
        # route coverage is more robust than betting two slots on the first query.
        return cls._candidate_urls(result, limit)

    @classmethod
    def _related_evidence_urls(
        cls,
        pages: list[dict[str, Any]],
        *,
        queries: list[str],
        opened: dict[str, PageDocument],
        limit: int,
    ) -> list[str]:
        if limit <= 0:
            return []
        query_terms = [
            cls._search_query_terms(query) - _LINK_STOPWORDS for query in queries if query.strip()
        ]
        opened_urls = set(opened)
        ranked: list[tuple[int, int, str]] = []
        seen: set[str] = set()
        sequence = 0
        for page in pages:
            if not isinstance(page, dict):
                continue
            source_url = str(page.get("final_url") or page.get("requested_url") or "")
            source_host = urlsplit(source_url).hostname or ""
            links = page.get("links")
            if not isinstance(links, list):
                continue
            for link in links:
                if not isinstance(link, dict):
                    continue
                raw_url = str(link.get("url") or "").strip()
                parsed = urlsplit(raw_url)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    continue
                url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
                if url in seen or url in opened_urls:
                    continue
                seen.add(url)
                text = f"{link.get('text') or ''} {parsed.path.replace('/', ' ')}"
                terms = cls._search_query_terms(text) - _LINK_STOPWORDS
                overlap = max((len(terms & item) for item in query_terms), default=0)
                research_overlap = len(terms & _RESEARCH_LINK_TERMS)
                same_host = bool(source_host and source_host == (parsed.hostname or ""))
                if overlap < 2 and not (same_host and research_overlap >= 2):
                    continue
                score = overlap * 10 + research_overlap * 3 + int(same_host) * 2
                ranked.append((-score, sequence, url))
                sequence += 1
        ranked.sort()
        return [url for _, _, url in ranked[:limit]]

    @classmethod
    def _evidence_highlights(
        cls,
        pages: list[dict[str, Any]],
        *,
        queries: list[str],
        limit: int,
    ) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        query_terms = [
            cls._search_query_terms(query) - _LINK_STOPWORDS for query in queries if query.strip()
        ]
        ranked: list[tuple[int, int, int, dict[str, str]]] = []
        for page_index, page in enumerate(pages):
            if not isinstance(page, dict) or page.get("error"):
                continue
            text = str(page.get("text") or "").strip()
            if not text:
                continue
            title = str(page.get("title") or "").strip()
            url = str(page.get("final_url") or page.get("requested_url") or "").strip()
            page_candidates: list[tuple[int, int, dict[str, str]]] = []
            for passage_index, raw_passage in enumerate(re.split(r"\n\s*\n+", text)):
                passage = re.sub(r"\s+", " ", raw_passage).strip()
                if len(passage) < 40:
                    continue
                passage_words = re.findall(r"[a-z0-9]+", passage.casefold())
                prose_without_links = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", passage)
                prose_words = re.findall(r"[a-z0-9]+", prose_without_links.casefold())
                # Reader output often contains navigation cards whose entire text is
                # a query-relevant link title. They are useful for link expansion,
                # but are not evidence excerpts and otherwise outrank page prose.
                if len(prose_words) < 8 or len(prose_words) * 3 < len(passage_words):
                    continue
                terms = cls._search_query_terms(passage) - _LINK_STOPWORDS
                query_overlap = max((len(terms & item) for item in query_terms), default=0)
                signal_overlap = len(terms & _EVIDENCE_SIGNAL_TERMS)
                research_overlap = len(terms & _RESEARCH_LINK_TERMS)
                if query_overlap < 2 and signal_overlap == 0:
                    continue
                attribution_window = re.search(
                    r"\b(?:according to|attributed to|credited (?:to|with)|documented by|"
                    r"described by|reported by|written by)\b(.{0,120})",
                    passage,
                    flags=re.I,
                )
                named_attribution = bool(
                    attribution_window
                    and re.search(
                        r"\b[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)+\b",
                        attribution_window.group(1),
                    )
                )
                score = (
                    query_overlap * 10
                    + signal_overlap * 24
                    + research_overlap * 2
                    + int(named_attribution) * 80
                )
                page_candidates.append(
                    (
                        -score,
                        passage_index,
                        {
                            "title": title,
                            "url": url,
                            "passage": truncate_middle(passage, 1_600),
                        },
                    )
                )
            page_candidates.sort()
            for negative_score, passage_index, highlight in page_candidates[:2]:
                ranked.append((negative_score, page_index, passage_index, highlight))
        ranked.sort()
        selected = ranked[:limit]
        # Weakest-to-strongest keeps the best passage at the very end of the
        # bounded tool result and finalizer evidence sample.
        selected.reverse()
        return [highlight for _, _, _, highlight in selected]

    @classmethod
    def _unopened_candidate_urls(
        cls,
        result: dict[str, Any],
        *,
        opened: dict[str, PageDocument],
        limit: int,
    ) -> list[str]:
        if limit <= 0:
            return []
        opened_urls = set(opened)
        candidates = cls._candidate_urls(result, 10_000)
        return [url for url in candidates if url not in opened_urls][:limit]

    @classmethod
    def _external_consultation_urls(
        cls,
        consultations: list[dict[str, Any]],
        *,
        question: str,
        opened: dict[str, PageDocument],
        limit: int,
    ) -> list[str]:
        if limit <= 0:
            return []
        urls: list[str] = []
        seen = set(opened)
        for consultation in consultations:
            for match in _PUBLIC_URL.finditer(str(consultation.get("content") or "")):
                url = match.group(0).rstrip(".,;:")
                if url in seen or cls._looks_like_query_mirror_url(question, url):
                    continue
                seen.add(url)
                urls.append(url)
                if len(urls) >= limit:
                    return urls
        return urls

    @classmethod
    def _external_answer_consensus(
        cls,
        consultations: list[dict[str, Any]],
        *,
        question: str,
        inspected_pages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return exact Star-2 agreement only when a cited, question-matched page was opened."""

        inspected_urls: dict[str, dict[str, Any]] = {}
        for page in inspected_pages:
            if not isinstance(page, dict) or page.get("error") or not page.get("text"):
                continue
            for key in ("requested_url", "final_url"):
                normalized_url = cls._evidence_url_key(str(page.get(key) or ""))
                if normalized_url:
                    inspected_urls[normalized_url] = page

        grouped: dict[str, list[dict[str, Any]]] = {}
        seen_request_ids: set[str] = set()
        for consultation in consultations:
            if (
                not consultation.get("ok")
                or consultation.get("status") != "succeeded"
                or consultation.get("model") != "frontierrl/star-2"
            ):
                continue
            request_id = str(consultation.get("request_id") or "").strip()
            if not request_id or request_id in seen_request_ids:
                continue
            seen_request_ids.add(request_id)
            exact_answer = str(consultation.get("exact_answer") or "").strip()
            citations = [
                str(citation).strip()
                for citation in consultation.get("citations") or []
                if isinstance(citation, str)
            ]
            if any(cls._looks_like_query_mirror_url(question, url) for url in citations):
                continue
            normalized_answer = cls._normalize_consensus_answer(exact_answer)
            if not normalized_answer or _ABSTENTION_ANSWER.match(exact_answer):
                continue
            grouped.setdefault(normalized_answer, []).append(consultation)

        qualifying = [items for items in grouped.values() if len(items) >= 2]
        if not qualifying:
            return None
        qualifying.sort(
            key=lambda items: (
                -len(items),
                -sum(cls._consensus_confidence(item) for item in items),
                cls._normalize_consensus_answer(str(items[0].get("exact_answer") or "")),
            )
        )
        agreed = sorted(
            qualifying[0],
            key=cls._consensus_confidence,
            reverse=True,
        )
        cited_urls = list(
            dict.fromkeys(
                str(citation).strip()
                for item in agreed
                for citation in item.get("citations") or []
                if isinstance(citation, str)
                and citation.strip().startswith(("http://", "https://"))
            )
        )
        supporting_citations = [
            citation
            for citation in cited_urls
            if (
                (page := inspected_urls.get(cls._evidence_url_key(citation))) is not None
                and cls._page_matches_question(question, page)
            )
        ]
        if not supporting_citations:
            return None
        return {
            "exact_answer": str(agreed[0]["exact_answer"]).strip(),
            "agreement_count": len(agreed),
            "request_ids": [
                str(item.get("request_id") or "") for item in agreed if item.get("request_id")
            ],
            "supporting_citations": supporting_citations,
        }

    @staticmethod
    def _normalize_consensus_answer(answer: str) -> str:
        words = _CONSENSUS_WORD.findall(answer.casefold())
        if words[:1] == ["the"] and len(words) > 1:
            words = words[1:]
        return " ".join(words)

    @staticmethod
    def _consensus_confidence(consultation: dict[str, Any]) -> float:
        try:
            return float(consultation.get("confidence") or 0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _page_matches_question(cls, question: str, page: dict[str, Any]) -> bool:
        question_terms = cls._search_query_terms(question) - _LINK_STOPWORDS
        if not question_terms:
            return False
        page_terms = cls._search_query_terms(f"{page.get('title') or ''}\n{page.get('text') or ''}")
        required_overlap = min(
            len(question_terms),
            min(8, max(2, (len(question_terms) + 5) // 6)),
        )
        return len(question_terms & page_terms) >= required_overlap

    @staticmethod
    def _evidence_url_key(url: str) -> str:
        parsed = urlsplit(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        host = (parsed.hostname or "").casefold()
        if host.startswith("www."):
            host = host[4:]
        try:
            parsed_port = parsed.port
        except ValueError:
            return ""
        port = f":{parsed_port}" if parsed_port else ""
        path = unquote(parsed.path or "/").rstrip("/") or "/"
        return urlunsplit(("https", host + port, path, parsed.query, ""))

    @classmethod
    def _looks_like_query_mirror_url(cls, question: str, url: str) -> bool:
        question_terms = cls._search_query_terms(question) - _LINK_STOPWORDS
        if len(question_terms) < 7:
            return False
        path = unquote(urlsplit(url).path).replace("-", " ")
        path_terms = cls._search_query_terms(path) - _LINK_STOPWORDS
        if len(path) < 80 or len(path_terms) < 8:
            return False
        overlap = len(question_terms & path_terms)
        question_coverage = overlap / len(question_terms)
        path_coverage = overlap / len(path_terms)
        solver_path = any(
            marker in path.casefold()
            for marker in ("crossword solver", "crossword clue", "question answer")
        )
        return bool(
            overlap >= 8
            and (
                question_coverage >= 0.5
                or path_coverage >= 0.65
                or (solver_path and path_coverage >= 0.5)
            )
        )

    @classmethod
    def _filter_query_mirror_search_results(
        cls,
        question: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Remove clue-restating SEO pages before either Star model can use them."""

        filtered = 0

        def safe_results(rows: Any) -> list[dict[str, Any]]:
            nonlocal filtered
            if not isinstance(rows, list):
                return []
            safe: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                url = str(row.get("url") or "")
                if url and cls._looks_like_query_mirror_url(question, url):
                    filtered += 1
                    continue
                safe.append(row)
            return safe

        if isinstance(result.get("results"), list):
            result["results"] = safe_results(result["results"])
            result["ok"] = bool(result["results"])
        searches = result.get("searches")
        if isinstance(searches, list):
            succeeded = 0
            for search in searches:
                if not isinstance(search, dict) or not isinstance(search.get("results"), list):
                    continue
                search["results"] = safe_results(search["results"])
                if search["results"]:
                    succeeded += 1
                elif "error" not in search:
                    search["integrity_error"] = "all returned pages were query mirrors"
            result["succeeded"] = succeeded
            result["failed"] = len(searches) - succeeded
            result["ok"] = succeeded > 0
        if filtered:
            result["filtered_query_mirror_results"] = filtered
            result["search_integrity_guidance"] = (
                "Clue-restating SEO pages were removed. Use independently authored sources; "
                "never infer an answer from a URL slug that mirrors the question."
            )
        return result

    async def _automatic_external_finalization(
        self,
        *,
        question: str,
        response: Any,
        messages: list[dict[str, Any]],
        transcript: list[dict[str, Any]],
        notes: list[str],
        request_namespace: str,
        request_budget: int,
    ) -> tuple[AgentAction | None, dict[str, Any]]:
        if self.external_model is None or request_budget <= 0:
            return None, {"ok": False, "error": "external model is unavailable"}
        evidence_messages = [
            message for message in messages if message.get("role") in {"tool", "user"}
        ]
        milestone_messages = [
            message
            for message in evidence_messages
            if any(
                marker in str(message.get("content") or "")
                for marker in (
                    "independent_external_consultation",
                    "external_search_strategy_recovery",
                    "automatic_page_inspection",
                )
            )
        ]
        selected_evidence: list[dict[str, Any]] = []
        selected_ids: set[int] = set()
        for message in evidence_messages[:2] + milestone_messages[-4:] + evidence_messages[-6:]:
            identity = id(message)
            if identity in selected_ids:
                continue
            selected_ids.add(identity)
            selected_evidence.append(message)
        representative_evidence = [
            truncate_middle(str(message.get("content") or ""), 8_000)
            for message in selected_evidence
        ]
        reasoning_history = [
            truncate_middle(str(message.get("reasoning") or ""), 8_000)
            for message in transcript
            if message.get("role") == "assistant" and message.get("reasoning")
        ][-4:]
        context = canonical_json(
            {
                "question": question,
                "latest_assistant_reasoning": truncate_middle(
                    str(response.raw_message.get("reasoning") or ""), 20_000
                ),
                "latest_assistant_content": truncate_middle(response.content, 10_000),
                "recent_reasoning_history": reasoning_history,
                "representative_tool_evidence": representative_evidence,
                "saved_notes": notes[-10:],
            }
        )
        review_roles = [
            (
                "Candidate matrix and independent solver",
                "Enumerate every plausible answer-type-valid candidate in the evidence and solve "
                "the question independently. Resolve any underlying entity before selecting the "
                "requested person, title, place, or other final answer. For each clue, distinguish "
                "directly supported, inferred, unknown, and contradicted. Missing source text is "
                "unknown, not a contradiction, but an exact matched source saying the required "
                "outcome did not occur is affirmative contradiction. Rank candidates comparatively "
                "and return your single best concrete answer as a final JSON action. Reject any "
                "candidate that only fits after substituting a related but different relation, such "
                "as a later breakthrough for a debut, an event for an artifact, participation for "
                "organizing, or beneficiary geography for the candidate's origin. Treat hard clues "
                "as a conjunction: one material contradiction eliminates a candidate unless a "
                "reliable source conflict is explicitly resolved. Return the narrowest exact value "
                "supported by the source, never a generic hypernym for a known specific answer. "
                "For scholarly clues, translate the paraphrase into field terminology, identify "
                "the publication independently, enumerate its authors, and verify every remaining "
                "author relation. For multiple distance constraints, use geo_search with expected "
                "distances and compare shared candidates by route error.",
            ),
            (
                "Comparative adversarial auditor",
                "Challenge the leading candidate with minimal-pair alternatives and concrete "
                "objections involving dates, aliases, units, identity, ordering, and answer type. "
                "Independently re-evaluate the underlying entity and the strongest alternative; do "
                "not inherit the latest assistant hypothesis. "
                "Classify each objection as contradiction, unresolved evidence, source-quality "
                "concern, identity mismatch, or answer-type mismatch. Source silence is not "
                "disproof, and skepticism has no veto. Compare alternatives and return the single "
                "best concrete answer as a final JSON action. Reject explanations that use likely, "
                "associated with, or similar hedges to substitute for a hard relation. For route-"
                "distance questions, independently verify that the selected place appears in the "
                "cross-anchor geo evidence rather than merely existing in the same city.",
            ),
        ]
        review_count = min(len(review_roles), max(0, request_budget - 1))
        review_requests = [
            {
                "system": (
                    "You are an independent reviewer for an audited public-web research task. Do "
                    "not look for benchmark dumps, canaries, leaked questions, or reference "
                    "answers. Use public-web research if available and never fabricate citations. "
                    "Preserve earlier milestone evidence as well as recent evidence. A directly "
                    "matched source showing that a required outcome did not occur is a contradiction, "
                    "not mere uncertainty. For an answer requested exactly as printed in a dated "
                    "article, prefer the article's explicit name form over historical guesswork. "
                    "Before accepting a source, verify that multiple independent anchors identify "
                    "the same underlying event, entity, or publication; reject coincidental pages "
                    "that share only the requested phrase or answer type. "
                    "A paper matching one technical clue does not identify an author until the "
                    "remaining author relations are directly tested. Multiple route-distance clues "
                    "require geo_search evidence with their expected distances. "
                    "This is a forced-answer task: return exactly one final JSON action with a "
                    "concrete answer of the requested type. Unresolved evidence lowers confidence "
                    "but never permits an abstention or meta-answer. Before finalizing, use public "
                    "search to try to falsify the leading candidate against every hard clue."
                ),
                "query": (
                    f"Role: {role}\n\n{task}\n\nReturn exactly: "
                    '{"action":"final","explanation":"comparative evidence audit",'
                    '"exact_answer":"short concrete answer","confidence":50,'
                    '"citations":["https://public-source.example/"]}'
                ),
                "context": context,
            }
            for role, task in review_roles[:review_count]
        ]
        reviews = (
            await self.external_model.ask_many(
                review_requests,
                request_namespace=request_namespace + ":finalization-reviews",
            )
            if review_requests
            else []
        )
        review_actions = self._concrete_review_actions(reviews, question=question)
        consensus_action = self._review_action_consensus(review_actions)
        if consensus_action is not None:
            return consensus_action, {
                "ok": True,
                "status": "succeeded",
                "content": canonical_json(consensus_action.payload),
                "attempted": review_count,
                "review_request_ids": [item.get("request_id") for item in reviews],
                "reviews": reviews,
                "controller_consensus": True,
                "consensus_count": sum(
                    self._normalize_consensus_answer(str(action.payload.get("exact_answer") or ""))
                    == self._normalize_consensus_answer(
                        str(consensus_action.payload.get("exact_answer") or "")
                    )
                    for action in review_actions
                ),
            }
        adjudication_context = canonical_json(
            {
                "evidence_bundle": truncate_middle(context, 130_000),
                "independent_reviews": [
                    {
                        "status": item.get("status"),
                        "content": truncate_middle(str(item.get("content") or ""), 20_000),
                    }
                    for item in reviews
                ],
            }
        )
        adjudication_requests = [
            {
                "system": (
                    "You are the final adjudicator for an audited public-web research task. Do not "
                    "look for benchmark dumps, canaries, leaked questions, or reference answers. "
                    "Independently compare all candidates. Distinguish directly supported, "
                    "inferred, unknown, and contradicted clues. Missing proof is unknown, not "
                    "false; only affirmative, reliable, scope-aligned conflicting evidence is a "
                    "contradiction. Preserve the requested answer type. This is a forced-answer "
                    "task: select one concrete candidate even when evidence is incomplete, and "
                    "express uncertainty only through calibrated confidence and explanation. "
                    "Never return unknown, insufficient evidence, not verifiable, cannot determine, "
                    "or any other abstention or meta-answer. A candidate fails if a clue only works "
                    "after changing the relation type; do not relabel a breakthrough as a debut, an "
                    "appointment as an album, participation as organizing, or a beneficiary as a "
                    "homeland. Independently search the leading candidate and strongest alternative "
                    "before finalizing. Treat hard clues as a conjunction and eliminate a candidate "
                    "with any unresolved material contradiction. Preserve the most specific answer "
                    "wording supported by the evidence rather than broadening it. Return exactly one "
                    "JSON object and no markdown."
                ),
                "query": (
                    "Return this exact schema: "
                    '{"action":"final","explanation":"constraint-by-constraint evidence chain",'
                    '"exact_answer":"short answer","confidence":0,'
                    '"citations":["https://public-source.example/"]}. '
                    "Do not merely follow the latest candidate or majority vote. Resolve conflicts "
                    "against direct evidence. Recheck the underlying entity before selecting an "
                    "attributed person, and treat an exact source's negative result as affirmative "
                    "contradiction rather than ambiguity. Do not let universal skepticism veto the "
                    "best-supported candidate. A hard clue labeled merely implied or likely is not "
                    "direct support. Return the best concrete exact answer now."
                ),
                "context": adjudication_context,
            }
        ]
        adjudication = await self.external_model.ask_many(
            adjudication_requests,
            request_namespace=request_namespace + ":finalization-adjudication",
        )
        result = (
            adjudication[0] if adjudication else {"ok": False, "error": "empty adjudicator result"}
        )
        attempted = review_count + 1
        result = {
            **result,
            "attempted": attempted,
            "review_request_ids": [item.get("request_id") for item in reviews],
            "reviews": reviews,
        }
        action: AgentAction | None = None
        if result.get("ok"):
            try:
                action = parse_json_action(str(result.get("content") or ""))
                if action.action != "final":
                    raise ProtocolError("external finalizer returned a non-final action")
                self._require_valid_final_for_question(action, question)
            except ProtocolError as exc:
                result = {**result, "ok": False, "error": str(exc)}
                action = None

        remaining_budget = request_budget - attempted
        if action is None or not result.get("ok"):
            repair_result: dict[str, Any] | None = None
            if remaining_budget > 0:
                candidates = self._concrete_review_actions(reviews, question=question)
                allowed_answers = list(
                    dict.fromkeys(
                        str(candidate.payload.get("exact_answer") or "").strip()
                        for candidate in candidates
                        if str(candidate.payload.get("exact_answer") or "").strip()
                    )
                )
                repair_requests = [
                    {
                        "system": (
                            "You repair forced-choice outputs for a public-web research task. The "
                            "prior adjudicator abstained or violated the final schema. Return one "
                            "concrete answer of the requested type. Missing evidence is not a "
                            "reason to abstain. Return exactly one final JSON action and no markdown."
                        ),
                        "query": (
                            "Repair the prior adjudication. Choose the best concrete answer now. "
                            + (
                                "Prefer one of these independently proposed exact answers unless "
                                f"the evidence directly contradicts all of them: {canonical_json(allowed_answers)}. "
                                if allowed_answers
                                else ""
                            )
                            + 'Return {"action":"final","explanation":"brief comparative reason",'
                            '"exact_answer":"short concrete answer","confidence":50,'
                            '"citations":["https://public-source.example/"]}.'
                        ),
                        "context": canonical_json(
                            {
                                "evidence_bundle": truncate_middle(context, 100_000),
                                "reviews": [
                                    truncate_middle(str(item.get("content") or ""), 16_000)
                                    for item in reviews
                                ],
                                "invalid_adjudication": truncate_middle(
                                    str(result.get("content") or ""), 12_000
                                ),
                            }
                        ),
                    }
                ]
                repairs = await self.external_model.ask_many(
                    repair_requests,
                    request_namespace=request_namespace + ":finalization-repair",
                )
                repair_result = (
                    repairs[0] if repairs else {"ok": False, "error": "empty repair result"}
                )
                attempted += 1
                result = {
                    **result,
                    "attempted": attempted,
                    "repair_request_id": repair_result.get("request_id"),
                    "repair": repair_result,
                }
                if repair_result.get("ok"):
                    try:
                        repaired_action = parse_json_action(str(repair_result.get("content") or ""))
                        if repaired_action.action != "final":
                            raise ProtocolError("external repair returned a non-final action")
                        self._require_valid_final_for_question(repaired_action, question)
                        action = repaired_action
                        result = {
                            **result,
                            "ok": True,
                            "content": str(repair_result.get("content") or ""),
                            "error": None,
                        }
                    except ProtocolError as exc:
                        result = {**result, "error": str(exc)}
                        action = None

            if action is None or self._is_abstention_answer(
                str(action.payload.get("exact_answer") or "")
            ):
                fallback = self._best_review_fallback(reviews, question=question)
                if fallback is None:
                    return None, {**result, "ok": False}
                action = fallback
                result = {
                    **result,
                    "ok": True,
                    "content": canonical_json({"controller_fallback": action.payload}),
                    "controller_fallback": True,
                    "error": None,
                }
        return action, result

    @staticmethod
    def _is_abstention_answer(answer: str) -> bool:
        normalized = re.sub(r"\s+", " ", answer.strip())
        return not normalized or bool(_ABSTENTION_ANSWER.match(normalized))

    @classmethod
    def _require_concrete_final(cls, action: AgentAction) -> None:
        answer = str(action.payload.get("exact_answer") or "")
        if cls._is_abstention_answer(answer):
            raise ProtocolError("final requires one concrete answer; abstentions are invalid")

    @classmethod
    def _require_valid_final_for_question(cls, action: AgentAction, question: str) -> None:
        cls._require_concrete_final(action)
        violations = cls._answer_type_constraint_errors(
            question,
            str(action.payload.get("exact_answer") or ""),
        )
        if violations:
            raise ProtocolError("; ".join(violations))

    @staticmethod
    def _answer_type_constraint_errors(question: str, answer: str) -> list[str]:
        """Reject obvious category restatements when the question asks for an identity."""

        normalized = re.sub(r"\s+", " ", answer.strip())
        if not normalized or not _IDENTITY_QUESTION.search(question):
            return []
        generic = _GENERIC_IDENTITY_NOUN.match(normalized)
        if not generic:
            return []
        tail = normalized[generic.end() :].strip()
        word_count = len(_CONSENSUS_WORD.findall(normalized))
        if word_count <= 3 or _GENERIC_IDENTITY_QUALIFIER.search(tail):
            return [
                "the question asks for a named identity, but the answer is only a category "
                "description or clue restatement"
            ]
        return []

    @classmethod
    def _final_evidence_constraint_errors(
        cls,
        question: str,
        answer: str,
        citations: list[Any],
        opened: dict[str, PageDocument],
    ) -> list[str]:
        """Require the proposed exact answer to occur in a cited, inspected relevant page."""

        mirror_citations = [
            str(citation)
            for citation in citations
            if isinstance(citation, str) and cls._looks_like_query_mirror_url(question, citation)
        ]
        if mirror_citations:
            return ["a cited URL is a clue-restating query mirror rather than independent evidence"]

        citation_keys = {
            key
            for citation in citations
            if isinstance(citation, str)
            for key in [cls._evidence_url_key(citation)]
            if key
        }
        if not citation_keys:
            return ["no valid public citation URL was supplied"]

        cited_documents: list[PageDocument] = []
        seen_documents: set[int] = set()
        for document in opened.values():
            document_keys = {
                cls._evidence_url_key(document.requested_url),
                cls._evidence_url_key(document.final_url),
            }
            if not citation_keys.intersection(document_keys) or id(document) in seen_documents:
                continue
            seen_documents.add(id(document))
            cited_documents.append(document)
        if not cited_documents:
            return ["none of the cited pages was opened and inspected"]

        normalized_answer = " ".join(_CONSENSUS_WORD.findall(answer.casefold()))
        if not normalized_answer:
            return ["the proposed exact answer has no searchable content"]
        require_literal_hashtag = answer.strip().startswith("#")
        supporting_documents: list[PageDocument] = []
        for document in cited_documents:
            page = {
                "title": document.title,
                "text": document.text,
                "requested_url": document.requested_url,
                "final_url": document.final_url,
            }
            raw_evidence = f"{document.title}\n{document.text}".casefold()
            normalized_evidence = " ".join(_CONSENSUS_WORD.findall(raw_evidence))
            answer_present = (
                answer.strip().casefold() in raw_evidence
                if require_literal_hashtag
                else normalized_answer in normalized_evidence
            )
            if answer_present and cls._page_matches_question(question, page):
                supporting_documents.append(document)
        minimum_support = cls._minimum_answer_supporting_documents(question)
        if len(supporting_documents) >= minimum_support:
            return []
        if supporting_documents:
            return [
                "a multi-hop question requires at least "
                f"{minimum_support} independently opened answer-naming sources, but only "
                f"{len(supporting_documents)} was supplied"
            ]
        return [
            "no cited, inspected page both names the proposed exact answer and materially "
            "matches the question"
        ]

    @staticmethod
    def _minimum_answer_supporting_documents(question: str) -> int:
        single_source_attribution = re.search(
            r"(?is)\baccording to (?:an?|the) "
            r"(?:article|source|paper|report|page|publication|interview|profile)\b.*?"
            r"\b(?:exactly as|as it appears|identified|credited|named|stated)\b",
            question,
        )
        if single_source_attribution:
            return 1
        sentences = [item for item in re.split(r"[.!?]+", question) if item.strip()]
        years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", question))
        distance_clues = _GEO_DISTANCE_CLUE.findall(question)
        is_multi_hop = len(question) >= 280 and (
            len(sentences) >= 3 or len(years) >= 2 or len(distance_clues) >= 2
        )
        return 2 if is_multi_hop else 1

    @staticmethod
    def _geo_final_constraint_errors(
        question: str,
        answer: str,
        geo_evidence: list[dict[str, Any]],
    ) -> list[str]:
        distance_clues = _GEO_DISTANCE_CLUE.findall(question)
        if len(distance_clues) < 2 or not _GEO_RELATION_CLUE.search(question):
            return []
        valid_evidence = [
            item
            for item in geo_evidence
            if item.get("ok") and isinstance(item.get("anchors"), list)
        ]
        if not valid_evidence:
            return [
                "the question contains multiple distance or proximity constraints, but no "
                "successful deterministic geo_search was performed"
            ]
        required_expected = min(len(distance_clues), 4)
        expected_anchors: set[str] = set()
        matching_anchors: set[str] = set()
        normalized_answer = AgentRunner._normalize_consensus_answer(answer)
        for evidence_index, evidence in enumerate(valid_evidence):
            for anchor_index, anchor in enumerate(evidence["anchors"]):
                if not isinstance(anchor, dict) or not anchor.get("ok"):
                    continue
                expected = anchor.get("expected_distance_miles")
                if expected is None:
                    continue
                anchor_key = (
                    AgentRunner._normalize_consensus_answer(str(anchor.get("query") or ""))
                    or f"{evidence_index}:{anchor_index}"
                )
                expected_anchors.add(anchor_key)
                if not _GEO_ENTITY_QUESTION.search(question):
                    continue
                for place in anchor.get("places") or []:
                    if not isinstance(place, dict):
                        continue
                    labels = {
                        AgentRunner._normalize_consensus_answer(str(place.get("name") or "")),
                        AgentRunner._normalize_consensus_answer(str(place.get("brand") or "")),
                    } - {""}
                    if any(
                        label in normalized_answer or normalized_answer in label for label in labels
                    ):
                        matching_anchors.add(anchor_key)
                        break
            for entity in evidence.get("shared_entities") or []:
                if not isinstance(entity, dict):
                    continue
                normalized_label = AgentRunner._normalize_consensus_answer(
                    str(entity.get("label") or "")
                )
                if not normalized_label or not (
                    normalized_label in normalized_answer or normalized_answer in normalized_label
                ):
                    continue
                for match in entity.get("matches") or []:
                    if not isinstance(match, dict):
                        continue
                    anchor_index = match.get("anchor_index")
                    if not isinstance(anchor_index, int) or not 0 <= anchor_index < len(
                        evidence["anchors"]
                    ):
                        continue
                    anchor = evidence["anchors"][anchor_index]
                    if (
                        not isinstance(anchor, dict)
                        or anchor.get("expected_distance_miles") is None
                    ):
                        continue
                    anchor_key = (
                        AgentRunner._normalize_consensus_answer(str(anchor.get("query") or ""))
                        or f"{evidence_index}:{anchor_index}"
                    )
                    matching_anchors.add(anchor_key)
        if len(expected_anchors) < required_expected:
            return [
                f"geo_search encoded routed evidence for only {len(expected_anchors)} distinct "
                f"stated-distance anchors; at least {required_expected} are required"
            ]
        if not _GEO_ENTITY_QUESTION.search(question):
            return []
        if len(matching_anchors) >= required_expected:
            return []
        return [
            f"the proposed place or business fits only {len(matching_anchors)} of "
            f"{required_expected} distinct routed-distance anchor candidate pools"
        ]

    @staticmethod
    def _surface_answer_constraint_errors(question: str, answer: str) -> list[str]:
        answer_words = _ANSWER_WORD.findall(answer)
        errors: list[str] = []
        for match in _ENDS_WITH_WORD.finditer(question):
            required = match.group("word")
            actual = answer_words[-1] if answer_words else "<empty>"
            if actual.casefold() != required.casefold():
                errors.append(
                    f"answer must end with {required!r}, but its final word is {actual!r}"
                )
        for match in _STARTS_WITH_WORD.finditer(question):
            required = match.group("word")
            actual = answer_words[0] if answer_words else "<empty>"
            if actual.casefold() != required.casefold():
                errors.append(
                    f"answer must start with {required!r}, but its first word is {actual!r}"
                )
        return errors

    @classmethod
    def _concrete_review_actions(
        cls,
        reviews: list[dict[str, Any]],
        *,
        question: str = "",
    ) -> list[AgentAction]:
        actions: list[AgentAction] = []
        for review in reviews:
            if not review.get("ok"):
                continue
            try:
                action = parse_json_action(str(review.get("content") or ""))
                if action.action != "final":
                    continue
                cls._require_valid_final_for_question(action, question)
            except ProtocolError:
                continue
            citations = [
                str(citation).strip()
                for citation in action.payload.get("citations") or []
                if isinstance(citation, str)
            ]
            if any(cls._looks_like_query_mirror_url(question, url) for url in citations):
                continue
            actions.append(action)
        return actions

    @classmethod
    def _best_review_fallback(
        cls,
        reviews: list[dict[str, Any]],
        *,
        question: str = "",
    ) -> AgentAction | None:
        candidates = cls._concrete_review_actions(reviews, question=question)
        cited_candidates = [
            candidate
            for candidate in candidates
            if any(
                isinstance(item, str) and item.startswith(("http://", "https://"))
                for item in candidate.payload.get("citations") or []
            )
        ]
        if not cited_candidates:
            return None
        cited_candidates.sort(
            key=lambda candidate: float(candidate.payload.get("confidence") or 0),
            reverse=True,
        )
        chosen = cited_candidates[0]
        citations = [
            str(item)
            for item in chosen.payload.get("citations") or []
            if isinstance(item, str) and item.startswith(("http://", "https://"))
        ]
        try:
            confidence = float(chosen.payload.get("confidence") or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        return AgentAction(
            action="final",
            payload={
                "explanation": (
                    "The final adjudicator did not satisfy the forced-choice contract. The "
                    "controller selected the highest-confidence concrete recommendation from the "
                    "independent comparative reviews; unresolved evidence lowers confidence but "
                    "does not convert the answer into an abstention."
                ),
                "exact_answer": str(chosen.payload.get("exact_answer") or "").strip(),
                "confidence": min(max(confidence, 0.0), 70.0),
                "citations": citations,
            },
        )

    @classmethod
    def _review_action_consensus(cls, actions: list[AgentAction]) -> AgentAction | None:
        """Return a cited final when at least two independent reviewers agree."""

        grouped: dict[str, list[AgentAction]] = {}
        for action in actions:
            answer = str(action.payload.get("exact_answer") or "").strip()
            normalized = cls._normalize_consensus_answer(answer)
            if normalized:
                grouped.setdefault(normalized, []).append(action)
        agreeing = [items for items in grouped.values() if len(items) >= 2]
        if not agreeing:
            return None

        def confidence(action: AgentAction) -> float:
            try:
                return float(action.payload.get("confidence") or 0)
            except (TypeError, ValueError):
                return 0.0

        agreeing.sort(
            key=lambda items: (
                -len(items),
                -sum(confidence(action) for action in items),
                cls._normalize_consensus_answer(str(items[0].payload.get("exact_answer") or "")),
            )
        )
        selected = sorted(agreeing[0], key=confidence, reverse=True)
        citations = list(
            dict.fromkeys(
                str(citation).strip()
                for action in selected
                for citation in action.payload.get("citations") or []
                if isinstance(citation, str)
                and citation.strip().startswith(("http://", "https://"))
            )
        )
        if not citations:
            return None
        strongest = selected[0]
        return AgentAction(
            action="final",
            payload={
                "explanation": str(strongest.payload.get("explanation") or "").strip(),
                "exact_answer": str(strongest.payload.get("exact_answer") or "").strip(),
                "confidence": confidence(strongest),
                "citations": citations,
            },
        )

    async def _automatic_external_consultations(
        self,
        *,
        question: str,
        current_evidence: dict[str, Any],
        notes: list[str],
        request_namespace: str,
        request_count: int,
    ) -> list[dict[str, Any]]:
        if self.external_model is None or request_count <= 0:
            return []
        evidence = truncate_middle(canonical_json(current_evidence), 30_000)
        saved_notes = truncate_middle("\n".join(notes[-10:]), 10_000)
        context = (
            f"Original research question:\n{question}\n\nMost recent search evidence:\n{evidence}"
        )
        if saved_notes:
            context += f"\n\nSaved research notes:\n{saved_notes}"
        roles = [
            (
                "Search strategy specialist",
                "Do not inherit the current leading hypothesis or treat repeated mentions as "
                "support. Independently identify three viable underlying entities, including a "
                "strongest alternative as a minimal pair, then use public-web tools to test the single most "
                "discriminating clue before designing seven meaningfully different searches. The "
                "queries must discriminate among candidates and resolve the requested answer "
                "relation. Include broad entity-plus-history/origins/attribution routes as well as "
                "primary-record or source-language routes. For a historical attribution, put a "
                "broad subject history or origins query first; do not lead with answer-shaped "
                "wording such as 'first person credited.' Use one entity per query, no OR chains, "
                "and at most twelve terms per query. Do not create novelty by changing only quotes, "
                "punctuation, or date ranges. In the analysis, state what evidence would falsify "
                "each candidate. Return exactly one JSON object with schema "
                '{"analysis":"brief candidate and falsification diagnosis","entity_candidates":['
                '"candidate"],"queries":["query 1","query 2"]} and no markdown.',
            ),
            (
                "Independent candidate investigator",
                "Resolve the underlying entity before proposing the exact answer. Weight the most "
                "discriminating clues first, and reject any entity affirmatively contradicted by "
                "the exact experiment, event, date, or relation in a matched source. Keep at least "
                "three viable entities until a constraint-by-constraint ledger distinguishes "
                "them. For every candidate and every clue, label direct support, inference, "
                "unknown, or contradiction. Propose the answer, primary sources or URLs, and "
                "high-information follow-up queries. Before concluding, perform one adversarial "
                "minimal-pair check against the strongest alternative, including dates, negation, "
                "causal ordering, aliases, units, and answer type. State uncertainty explicitly.",
            ),
            (
                "Adversarial constraint auditor",
                "Challenge every candidate implied by the evidence. Check dates, negation, causal "
                "ordering, aliases, units, and minimal-pair alternatives. Identify the strongest "
                "falsification tests and what evidence would resolve them. Do not let one candidate "
                "win by repeated mentions; compare clue coverage in an explicit candidate matrix.",
            ),
            (
                "Independent final-answer reviewer",
                "Infer the best current exact answer, then argue against it. Return a calibrated "
                "recommendation only after checking the requested answer type and all qualifiers.",
            ),
        ]
        requests = [
            {
                "system": (
                    "You are an independent research subagent. Do not look for benchmark dumps, "
                    "reference answers, canaries, or leaked evaluation material. Work only from "
                    "the supplied question, evidence, and ordinary public knowledge. Use public "
                    "web research if it is available to you. Clearly label memory-based leads as "
                    "unverified, do not fabricate URLs, and never claim high confidence without "
                    "direct constraint-matching evidence. Reject a page that matches only the "
                    "answer phrase but not multiple independent anchors from the original "
                    "question; a same-named event, tour, person, or publication is not identity "
                    "evidence."
                ),
                "query": f"Role: {role}\n\n{task}",
                "context": context,
            }
            for role, task in roles[:request_count]
        ]
        results = await self.external_model.ask_many(
            requests,
            request_namespace=request_namespace,
        )
        return [
            {
                **result,
                "review_role": roles[index][0] if index < len(roles) else "unknown",
            }
            for index, result in enumerate(results)
        ]

    def _parse_action(
        self,
        response: Any,
        protocol: str,
        *,
        force_final: bool = False,
    ) -> tuple[AgentAction, dict[str, Any]]:
        raw_message = dict(response.raw_message)
        tool_calls = raw_message.get("tool_calls")
        if protocol in {"tools", "auto"} and isinstance(tool_calls, list) and tool_calls:
            action = action_from_tool_call(tool_calls[0])
            if action.action == "final":
                self._require_concrete_final(action)
            # The harness executes exactly one action per turn. Retaining only
            # the first native tool call also prevents the next API request from
            # containing unanswered sibling tool calls, which strict OpenAI-style
            # servers reject.
            assistant_message = {
                "role": "assistant",
                "content": raw_message.get("content") or "",
                "tool_calls": [tool_calls[0]],
            }
            return action, assistant_message
        try:
            action = parse_json_action(response.content)
        except ProtocolError:
            if not force_final:
                raise
            action = self._plain_final_action(str(response.content or ""))
        if action.action == "final":
            self._require_concrete_final(action)
        return action, {"role": "assistant", "content": response.content}

    @staticmethod
    def _plain_final_action(content: str) -> AgentAction:
        """Recover a final answer when an Agent backend consumes a forced final tool."""

        text = content.strip()
        if not text:
            raise ProtocolError("Forced final response was empty")
        exact_match = re.search(r"^\s*Exact Answer\s*:\s*(.+?)\s*$", text, re.I | re.M)
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
        answer_match = re.search(r"^\s*(?:Final )?Answer\s*:\s*(.+?)\s*$", text, re.I | re.M)
        if exact_match:
            exact_answer = exact_match.group(1).strip()
        elif boxed:
            exact_answer = boxed[-1].strip()
        elif answer_match:
            exact_answer = answer_match.group(1).strip()
        elif len(text) <= 500 and text.count("\n") <= 2:
            exact_answer = text
        else:
            raise ProtocolError("Forced final response did not contain an extractable exact answer")
        confidence_match = re.search(r"Confidence\s*:\s*(\d+(?:\.\d+)?)\s*%?", text, re.I)
        confidence = float(confidence_match.group(1)) if confidence_match else 50.0
        citations = list(dict.fromkeys(re.findall(r"https?://[^\s<>()\[\]{}]+", text)))
        return AgentAction(
            action="final",
            payload={
                "explanation": text,
                "exact_answer": exact_answer,
                "confidence": max(0.0, min(100.0, confidence)),
                "citations": citations,
            },
        )

    async def _execute_action(
        self,
        action: AgentAction,
        opened: dict[str, PageDocument],
        notes: list[str],
        *,
        request_namespace: str,
    ) -> tuple[dict[str, Any], tuple[int, int, int, int]]:
        payload = action.payload
        if action.action == "search":
            count = min(int(payload.get("count", self.search.config.results_per_call)), 20)
            results = await self.search.search(str(payload["query"]), count=count)
            return {
                "ok": True,
                "query": payload["query"],
                "results": [item.as_prompt_dict() for item in results],
            }, (1, 0, 0, sum(len(item.snippet) for item in results))

        if action.action == "search_many":
            if not self.agent_config.enable_search_many:
                raise ValueError("search_many is disabled")
            queries = [str(item) for item in payload["queries"]][: self.agent_config.max_batch_size]
            count = min(int(payload.get("count", self.search.config.results_per_call)), 20)
            batches = await self.search.search_many(queries, count=count)
            output: list[dict[str, Any]] = []
            chars = 0
            successes = 0
            for query, batch in zip(queries, batches, strict=True):
                if isinstance(batch, Exception):
                    output.append({"query": query, "error": str(batch)})
                else:
                    successes += 1
                    chars += sum(len(item.snippet) for item in batch)
                    output.append(
                        {"query": query, "results": [item.as_prompt_dict() for item in batch]}
                    )
            return {
                "ok": successes > 0,
                "succeeded": successes,
                "failed": len(queries) - successes,
                "searches": output,
            }, (len(queries), 0, 0, chars)

        if action.action == "geo_search":
            anchors = [dict(item) for item in payload["anchors"]][:4]
            result = await self._geo_client().explore(
                anchors,
                category=str(payload.get("category") or "named_place"),
                max_results=int(payload.get("max_results") or 50),
                include_walking_routes=bool(payload.get("include_walking_routes", True)),
            )
            chars = len(canonical_json(result))
            return result, (len(anchors), 0, 0, chars)

        if action.action == "open":
            url = str(payload["url"])
            document = await self.browser.fetch(url)
            opened[url] = document
            opened[document.final_url] = document
            max_chars = min(
                int(payload.get("max_chars", self.browser_config.max_text_chars_per_open)),
                self.browser_config.max_text_chars_per_open,
            )
            window = page_window(document, int(payload.get("offset", 0)), max_chars)
            # Bound link payload independently from page text.
            window["links"] = document.links[: self.browser_config.max_links_per_page]
            return {"ok": True, "page": window}, (0, 1, 0, len(str(window["text"])))

        if action.action == "open_many":
            if not self.agent_config.enable_open_many:
                raise ValueError("open_many is disabled")
            urls = [str(item) for item in payload["urls"]][: self.agent_config.max_batch_size]
            fetched = await asyncio.gather(
                *(self.browser.fetch(url) for url in urls), return_exceptions=True
            )
            max_chars = min(
                int(payload.get("max_chars", self.browser_config.max_text_chars_per_open)),
                self.browser_config.max_text_chars_per_open,
            )
            offset = int(payload.get("offset", 0))
            pages: list[dict[str, Any]] = []
            chars = 0
            successes = 0
            for url, document in zip(urls, fetched, strict=True):
                if isinstance(document, Exception):
                    pages.append({"url": url, "error": str(document)})
                    continue
                opened[url] = document
                opened[document.final_url] = document
                window = page_window(document, offset, max_chars)
                window["links"] = document.links[: self.browser_config.max_links_per_page]
                chars += len(str(window["text"]))
                successes += 1
                pages.append(window)
            return {
                "ok": successes > 0,
                "succeeded": successes,
                "failed": len(urls) - successes,
                "pages": pages,
            }, (0, successes, 0, chars)

        if action.action == "find":
            url = str(payload["url"])
            document = opened.get(url)
            page_delta = 0
            if document is None:
                document = await self.browser.fetch(url)
                opened[url] = document
                opened[document.final_url] = document
                page_delta = 1
            pattern = str(payload["pattern"])
            try:
                regex = re.compile(pattern, flags=re.I)
            except re.error:
                regex = re.compile(re.escape(pattern), flags=re.I)
            matches: list[dict[str, Any]] = []
            for match in regex.finditer(document.text):
                start = max(0, match.start() - 500)
                end = min(len(document.text), match.end() + 500)
                matches.append({"offset": match.start(), "context": document.text[start:end]})
                if len(matches) >= 20:
                    break
            chars = sum(len(item["context"]) for item in matches)
            return {
                "ok": True,
                "url": document.final_url,
                "pattern": pattern,
                "matches": matches,
            }, (0, page_delta, 1, chars)

        if action.action == "ask_external_model":
            if self.external_model is None or not self.external_model_config.enabled:
                raise ExternalModelError("External-model consultation is unavailable")
            requests = payload.get("requests")
            if isinstance(requests, list):
                consultations = [dict(item) for item in requests if isinstance(item, dict)]
            else:
                consultations = [
                    {
                        key: value
                        for key, value in payload.items()
                        if key
                        in {
                            "query",
                            "context",
                            "system",
                            "provider",
                            "model",
                            "max_tokens",
                            "temperature",
                            "top_p",
                        }
                    }
                ]
            consultations = consultations[: self.external_model_config.max_batch_size]
            results = await self.external_model.ask_many(
                consultations,
                request_namespace=request_namespace,
            )
            chars = sum(len(str(item.get("content") or "")) for item in results)
            return {
                "ok": any(bool(item.get("ok")) for item in results),
                "attempted": len(consultations),
                "consultations": results,
                "instruction": (
                    "Treat these as independent advice. Cross-check material claims against "
                    "the web evidence and continue the same task."
                ),
            }, (0, 0, 0, chars)

        if action.action == "note":
            text = truncate_middle(str(payload["text"]), 5000)
            notes.append(text)
            return {"ok": True, "saved_note": text}, (0, 0, 0, 0)

        raise ValueError(f"Unsupported action: {action.action}")

    def _check_budgets(
        self,
        search_calls: int,
        page_opens: int,
        find_calls: int,
        retrieved_chars: int,
        external_model_calls: int,
    ) -> None:
        if search_calls > self.agent_config.max_search_calls:
            raise RuntimeError("Search-call budget exceeded")
        if page_opens > self.agent_config.max_page_opens:
            raise RuntimeError("Page-open budget exceeded")
        if find_calls > self.agent_config.max_find_calls:
            raise RuntimeError("Find-call budget exceeded")
        if retrieved_chars > self.agent_config.max_retrieved_chars:
            raise RuntimeError("Retrieved-text budget exceeded")
        if external_model_calls > self.external_model_config.max_calls_per_task:
            raise RuntimeError("External-model-call budget exceeded")

    def _near_budget(
        self,
        search_calls: int,
        page_opens: int,
        find_calls: int,
        retrieved_chars: int,
        external_model_calls: int,
    ) -> bool:
        del external_model_calls  # Exhausting optional help must not end the research task.
        return any(
            (
                search_calls >= self.agent_config.max_search_calls,
                page_opens >= self.agent_config.max_page_opens,
                find_calls >= self.agent_config.max_find_calls,
                retrieved_chars >= self.agent_config.max_retrieved_chars,
            )
        )

    def _compact_history(
        self,
        messages: list[dict[str, Any]],
        initial_user: str,
        notes: list[str],
        opened: dict[str, PageDocument],
    ) -> list[dict[str, Any]]:
        size = sum(len(str(message.get("content", ""))) for message in messages)
        if size <= self.agent_config.max_history_chars:
            return messages
        unique_pages: dict[str, PageDocument] = {}
        for document in opened.values():
            unique_pages[document.final_url] = document
        milestone_markers = (
            "independent_external_consultation",
            "external_search_strategy_recovery",
            "verified_evidence_highlights",
        )
        milestone_evidence = [
            truncate_middle(str(message.get("content") or ""), 12_000)
            for message in messages
            if message.get("role") == "tool"
            and any(marker in str(message.get("content") or "") for marker in milestone_markers)
        ][-4:]
        summary = {
            "saved_notes": notes[-20:],
            "preserved_milestone_evidence": milestone_evidence,
            "opened_pages": [
                {"url": document.final_url, "title": document.title, "sha256": document.sha256}
                for document in list(unique_pages.values())[-30:]
            ],
            "instruction": (
                "Continue the same task. Preserve the independent Star helper leads and decisive "
                "evidence above. Re-open pages when more text is needed."
            ),
        }
        prefix = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_user},
        ]
        fixed_prefix_chars = sum(len(str(message.get("content") or "")) for message in prefix)
        summary_text = "Deterministic history compaction:\n" + canonical_json(summary)
        summary_budget = max(
            0,
            self.agent_config.max_history_chars // 2 - fixed_prefix_chars,
        )
        prefix.append(
            {
                "role": "user",
                "content": truncate_middle(summary_text, summary_budget),
            }
        )
        tail_start = max(2, len(messages) - 8)
        while tail_start < len(messages) and messages[tail_start].get("role") == "tool":
            tail_start += 1
        tail = [dict(message) for message in messages[tail_start:]]

        fixed_chars = sum(len(str(message.get("content") or "")) for message in prefix)
        content_indices = [
            index for index, message in enumerate(tail) if str(message.get("content") or "")
        ]
        available_chars = max(0, self.agent_config.max_history_chars - fixed_chars)
        per_message_chars = (
            available_chars // len(content_indices) if content_indices else available_chars
        )
        for index in content_indices:
            content = str(tail[index].get("content") or "")
            tail[index]["content"] = truncate_middle(content, per_message_chars)

        compacted = [*prefix, *tail]
        compacted_size = sum(len(str(message.get("content") or "")) for message in compacted)
        self._emit(
            "history_compacted",
            before_chars=size,
            after_chars=compacted_size,
            max_chars=self.agent_config.max_history_chars,
            retained_tail_messages=len(tail),
            preserved_milestones=len(milestone_evidence),
        )
        return compacted

    def _final_outcome(
        self,
        action: AgentAction,
        *,
        started: float,
        step: int,
        usage: Usage,
        transcript: list[dict[str, Any]],
        errors: list[str],
        search_calls: int,
        page_opens: int,
        find_calls: int,
        retrieved_chars: int,
        external_model_calls: int,
    ) -> AgentOutcome:
        payload = action.payload
        answer = str(payload.get("exact_answer", "")).strip()
        explanation = str(payload.get("explanation", "")).strip()
        try:
            confidence = float(payload.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        if 0 < confidence <= 1:
            confidence *= 100
        raw_citations = payload.get("citations") or []
        citations = [str(item) for item in raw_citations if isinstance(item, str)]
        status = "completed"
        if not answer:
            status = "empty_answer"
        if self.agent_config.require_citations and not citations:
            errors.append("Final answer omitted citations")
        response_text = (
            f"Explanation: {explanation}\nExact Answer: {answer}\nConfidence: {confidence:g}%"
        )
        return AgentOutcome(
            response_text=response_text,
            exact_answer=answer or None,
            explanation=explanation,
            confidence=confidence,
            citations=citations,
            status=status,
            steps=step,
            search_calls=search_calls,
            page_opens=page_opens,
            find_calls=find_calls,
            retrieved_chars=retrieved_chars,
            duration_seconds=time.perf_counter() - started,
            usage=usage,
            external_model_calls=external_model_calls,
            transcript=transcript,
            errors=errors,
        )
