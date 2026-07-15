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
        action_fingerprints: set[str] = set()
        chain_enabled = self.model_config.response_chain
        previous_response_id: str | None = None
        chain_delta_messages: list[dict[str, Any]] | None = None
        namespace_material = request_namespace or question
        chain_namespace = hashlib.sha256(namespace_material.encode("utf-8")).hexdigest()[:24]

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
        self._emit("trial_started", protocol=protocol, response_chain=chain_enabled)

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
                messages.append({"role": "assistant", "content": response.content})
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
                correction_message = {"role": "user", "content": correction}
                messages.append(correction_message)
                transcript.append(correction_message)
                chain_delta_messages = [correction_message]
                self._emit("protocol_retry", step=step, error=str(exc))
                continue

            self._emit("action_selected", step=step, action=action.action, payload=action.payload)

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
            duplicate_action = action_fingerprint in action_fingerprints
            action_fingerprints.add(action_fingerprint)
            budget_violation = self._action_budget_violation(
                action,
                search_calls=search_calls,
                page_opens=page_opens,
                find_calls=find_calls,
                retrieved_chars=retrieved_chars,
                external_model_calls=external_model_calls,
            )
            try:
                if duplicate_action:
                    raise RuntimeError(
                        "Identical action already executed; inspect existing evidence or choose a "
                        "materially different action"
                    )
                if budget_violation:
                    force_final = True
                    raise RuntimeError(budget_violation)
                result, deltas = await self._execute_action(
                    action,
                    opened,
                    notes,
                    request_namespace=chain_namespace,
                )
                projected_search_calls = search_calls + deltas[0]
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
                    try:
                        consultations = await self._automatic_external_consultations(
                            question=question,
                            current_evidence=result,
                            notes=notes,
                            request_namespace=chain_namespace,
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
                            "reviews as leads. Verify material claims against public-web evidence."
                        ),
                    }
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
                result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call-{step}"),
                    "name": action.action,
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
    ):
        if protocol in {"tools", "auto"}:
            tool_choice: str | dict[str, Any] = "auto"
            tools = tool_schemas(
                include_external_model=(
                    self.external_model_config.enabled and self.external_model is not None
                )
            )
            if force_final:
                tools = [tool for tool in tools if tool.get("function", {}).get("name") == "final"]
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
            )
        return await self.client.chat(messages, extra_body=extra_body)

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
                "Independent candidate investigator",
                "Solve the research question independently. Propose the exact answer, a "
                "constraint-by-constraint evidence chain, likely primary sources or URLs, and "
                "high-information follow-up queries. State uncertainty explicitly.",
            ),
            (
                "Adversarial constraint auditor",
                "Challenge every candidate implied by the evidence. Check dates, negation, causal "
                "ordering, aliases, units, and minimal-pair alternatives. Identify the strongest "
                "falsification tests and what evidence would resolve them.",
            ),
            (
                "Search strategy specialist",
                "Design the next highest-yield public-web searches and source-opening plan. Favor "
                "quoted fragments, primary records, alternate terminology, archives, and sources "
                "that can discriminate among plausible candidates.",
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
                    "the supplied question, evidence, and ordinary public knowledge."
                ),
                "query": f"Role: {role}\n\n{task}",
                "context": context,
            }
            for role, task in roles[:request_count]
        ]
        return await self.external_model.ask_many(
            requests,
            request_namespace=request_namespace,
        )

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
