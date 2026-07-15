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
from urllib.parse import urlsplit, urlunsplit

from ..browser.extract import page_window
from ..browser.fetcher import BrowserError, PageFetcher
from ..config import AgentConfig, BrowserConfig, ExternalModelConfig, ModelConfig
from ..external import ExternalModelBroker, ExternalModelError
from ..llm import ModelAPIError, OpenAICompatibleClient, ProtocolError, parse_json_action
from ..llm.client import settings_from_model_config
from ..llm.protocol import action_from_tool_call
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


class AgentRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        agent_config: AgentConfig,
        browser_config: BrowserConfig,
        search_provider: SearchProvider,
        page_fetcher: PageFetcher,
        model_client: OpenAICompatibleClient | None = None,
        external_model_config: ExternalModelConfig | None = None,
        external_model_broker: ExternalModelBroker | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
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
        self.external_model_config = external_model_config or ExternalModelConfig()
        self.external_model = external_model_broker
        self.event_sink = event_sink
        self.system_prompt = AGENT_SYSTEM_PROMPT
        if agent_config.system_prompt_path is not None:
            self.system_prompt = agent_config.system_prompt_path.read_text(encoding="utf-8").strip()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.close()

    def _emit(self, event: str, **values: Any) -> None:
        if self.event_sink is not None:
            self.event_sink({"event": event, **values})

    async def run(
        self,
        question: str,
        *,
        request_namespace: str | None = None,
    ) -> AgentOutcome:
        started = time.perf_counter()
        usage = Usage()
        transcript: list[dict[str, Any]] = []
        errors: list[str] = []
        notes: list[str] = []
        opened: dict[str, PageDocument] = {}
        search_calls = page_opens = find_calls = retrieved_chars = external_model_calls = 0
        parse_failures = 0
        protocol = self.model_config.protocol
        force_final = False
        require_open = False
        search_streak = 0
        automatic_external_attempted = False
        automatic_strategy_recovery_attempted = False
        automatic_finalization_rescue_attempted = False
        forced_nonfinal_rejections = 0
        last_action_fingerprint: str | None = None
        consecutive_duplicate_actions = 0
        last_successful_search_result: dict[str, Any] | None = None
        search_query_history: list[str] = []
        chain_enabled = self.model_config.response_chain
        previous_response_id: str | None = None
        chain_delta_messages: list[dict[str, Any]] | None = None
        namespace_material = request_namespace or question
        request_headers = self._routing_headers(
            namespace_material,
            routing_backend_pool=self.model_config.routing_backend_pool,
        )
        chain_namespace = request_headers["X-FRL-Conversation-Id"].removeprefix("bc250-")
        external_namespace = request_namespace or chain_namespace

        initial_user = (
            "Question:\n"
            + question
            + "\n\nBudgets: "
            + canonical_json(
                {
                    "max_steps": self.agent_config.max_steps,
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
                    or self._near_budget(
                        search_calls,
                        page_opens,
                        find_calls,
                        retrieved_chars,
                        external_model_calls,
                    )
                    or step == self.agent_config.max_steps
                )
                query_task = asyncio.create_task(
                    self._query(
                        wire_messages,
                        protocol,
                        extra_body=chain_body,
                        force_final=force_final_this_turn,
                        require_open=require_open,
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
                    force_final=force_final_this_turn,
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
                external_model_calls += 1
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
                        request_id=strategy_result.get("request_id"),
                    )
                else:
                    self._emit(
                        "search_strategy_recovery_failed",
                        step=step,
                        error=strategy_result.get("error") or "no novel queries returned",
                        request_id=strategy_result.get("request_id"),
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
            forced_rescue_due = bool(
                force_final_this_turn
                and rescue_threshold > 0
                and forced_nonfinal_rejections >= rescue_threshold
            )
            time_rescue_due = bool(rescue_seconds > 0 and elapsed_seconds >= rescue_seconds)
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
                    if action.action in {"search", "search_many"}:
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
                    self._emit(
                        "automatic_external_completed",
                        step=step,
                        request_count=request_count,
                        successful=sum(bool(item.get("ok")) for item in consultations),
                        returned_chars=consultation_chars,
                    )
                search_calls += deltas[0]
                page_opens += deltas[1]
                find_calls += deltas[2]
                retrieved_chars += deltas[3]
                if action.action == "ask_external_model":
                    external_model_calls += int(result.get("attempted", 0))
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
            elif action.action in {"open", "open_many"} and result.get("ok"):
                require_open = False
                search_streak = 0

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
        request_headers: dict[str, str] | None = None,
    ):
        if protocol in {"tools", "auto"}:
            tool_choice: str | dict[str, Any] = "auto"
            tools = tool_schemas(
                include_external_model=(
                    self.external_model_config.enabled and self.external_model is not None
                )
            )
            if force_final:
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
        requests = [
            {
                "system": (
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
                    "Do not infer nationality from birthplace or primary occupation from one artifact. "
                    "For a historical attribution, put a broad subject history or origins query "
                    "first; do not lead with answer-shaped wording such as 'first person credited.' "
                    "Put the highest-yield unresolved-relation query first. Use one entity per "
                    "query, no OR chains, and at most twelve terms per query. Do not "
                    "paraphrase the full clue repeatedly, and do not create novelty by merely "
                    "changing quotes, punctuation, or date ranges. Return exactly one JSON object "
                    "and no markdown."
                ),
                "query": (
                    f"Return exactly this schema with {limit} concise public-web queries: "
                    '{"analysis":"brief diagnosis","entity_candidates":["candidate"],'
                    '"queries":["query 1","query 2"]}. Every query must test a different semantic '
                    "route and target the unresolved answer rather than re-verifying clues that are "
                    "already established. Include queries for the strongest alternative entity, not "
                    "only the researcher's current favorite."
                ),
                "context": context,
            }
        ]
        results = await self.external_model.ask_many(
            requests,
            request_namespace=request_namespace + ":search-strategy-recovery",
        )
        result = results[0] if results else {"ok": False, "error": "empty strategy result"}
        queries = self._strategy_queries_from_result(
            result,
            prior_queries=prior_queries,
            limit=limit,
        )
        return queries, result

    @staticmethod
    def _batched_action_size(action: AgentAction) -> int:
        key = {
            "search_many": "queries",
            "open_many": "urls",
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

    @staticmethod
    def _external_consultation_urls(
        consultations: list[dict[str, Any]],
        *,
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
                if url in seen:
                    continue
                seen.add(url)
                urls.append(url)
                if len(urls) >= limit:
                    return urls
        return urls

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
                "supported by the source, never a generic hypernym for a known specific answer.",
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
                "best concrete answer as a final JSON action.",
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
                    "best-supported candidate. Return the best concrete exact answer now."
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
                self._require_concrete_final(action)
            except ProtocolError as exc:
                result = {**result, "ok": False, "error": str(exc)}

        remaining_budget = request_budget - attempted
        if action is None or not result.get("ok"):
            repair_result: dict[str, Any] | None = None
            if remaining_budget > 0:
                candidates = self._concrete_review_actions(reviews)
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
                        self._require_concrete_final(repaired_action)
                        action = repaired_action
                        result = {
                            **result,
                            "ok": True,
                            "content": str(repair_result.get("content") or ""),
                            "error": None,
                        }
                    except ProtocolError as exc:
                        result = {**result, "error": str(exc)}

            if action is None or self._is_abstention_answer(
                str(action.payload.get("exact_answer") or "")
            ):
                fallback = self._best_review_fallback(reviews)
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
    def _concrete_review_actions(cls, reviews: list[dict[str, Any]]) -> list[AgentAction]:
        actions: list[AgentAction] = []
        for review in reviews:
            if not review.get("ok"):
                continue
            try:
                action = parse_json_action(str(review.get("content") or ""))
                if action.action != "final":
                    continue
                cls._require_concrete_final(action)
            except ProtocolError:
                continue
            actions.append(action)
        return actions

    @classmethod
    def _best_review_fallback(
        cls,
        reviews: list[dict[str, Any]],
    ) -> AgentAction | None:
        candidates = cls._concrete_review_actions(reviews)
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
                    "direct constraint-matching evidence."
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
        summary = {
            "saved_notes": notes[-20:],
            "opened_pages": [
                {"url": document.final_url, "title": document.title, "sha256": document.sha256}
                for document in list(unique_pages.values())[-30:]
            ],
            "instruction": "Continue the same task. Re-open pages when more text is needed.",
        }
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_user},
            {
                "role": "user",
                "content": "Deterministic history compaction:\n" + canonical_json(summary),
            },
            *messages[-8:],
        ]

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
