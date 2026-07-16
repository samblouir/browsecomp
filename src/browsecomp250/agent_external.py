from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from .agent import AgentRunner
from .config import AgentConfig, BrowserConfig, ExternalModelConfig, ModelConfig
from .llm import OpenAICompatibleClient, settings_from_model_config
from .types import AgentOutcome
from .util import canonical_json

_FINAL_ACTION_CONTRACT = re.compile(r'\{[^{}]*["\']action["\']\s*:\s*["\']final["\']', re.I)
_REQUIRED_HELPER_MODEL = "frontierrl/star-2"
_STRATEGY_SYSTEM_PROMPT = """
You are a retrieval-strategy controller. The supplied task asks for a query plan, not a factual
answer. Read the supplied question, evidence, and prior queries, then return exactly one native
final tool call immediately without browsing. Put the requested JSON object verbatim in final.explanation, set
final.exact_answer to "query strategy", confidence to a calibrated number, and citations to an
empty list. Never search for benchmark dumps, canaries,
leaked questions, or reference answers.
""".strip()


class AgentExternalModelBroker:
    """Run external-help requests as isolated Star agents with local web tools."""

    def __init__(
        self,
        config: ExternalModelConfig,
        agent_config: AgentConfig,
        browser_config: BrowserConfig,
        search_provider: Any,
        page_fetcher: Any,
        *,
        model_client: OpenAICompatibleClient | None = None,
        runner_factory: Callable[..., AgentRunner] = AgentRunner,
    ) -> None:
        if config.agent_model != _REQUIRED_HELPER_MODEL:
            raise ValueError(
                "Agent external help is pinned to "
                f"{_REQUIRED_HELPER_MODEL}; got {config.agent_model!r}"
            )
        self.config = config
        self.browser_config = browser_config
        self.search = search_provider
        self.browser = page_fetcher
        self.runner_factory = runner_factory
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self.model_config = ModelConfig(
            api_base=config.agent_api_base,
            api_key=config.agent_api_key,
            allow_empty_api_key=config.agent_allow_empty_api_key,
            model=config.agent_model,
            protocol="tools",
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            response_chain=config.agent_response_chain,
            extra_body={
                "top_p": config.top_p,
                "parallel_tool_calls": False,
                "vllm_xargs": {
                    "frontierrl_max_denoising_steps": config.agent_max_denoising_steps,
                },
            },
            routing_backend_pool=config.agent_routing_backend_pool,
        )
        self.agent_config = agent_config.model_copy(
            deep=True,
            update={
                "max_steps": config.agent_max_steps,
                "force_final_after_seconds": config.agent_force_final_after_seconds,
                "min_search_calls_before_final": config.agent_min_search_calls_before_final,
                "max_search_calls": config.agent_max_search_calls,
                "max_page_opens": config.agent_max_page_opens,
                "max_find_calls": config.agent_max_find_calls,
                "max_retrieved_chars": config.agent_max_retrieved_chars,
                "max_history_chars": config.agent_max_history_chars,
                "automatic_external_after_search_calls": 0,
                "automatic_finalization_rescue_after_rejections": 0,
                "automatic_finalization_rescue_after_seconds": 0,
            },
        )
        self.disabled_external_config = config.model_copy(
            deep=True,
            update={"enabled": False, "max_calls_per_task": 0},
        )
        self._owns_client = model_client is None
        self.client = model_client or OpenAICompatibleClient(
            settings_from_model_config(self.model_config)
        )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.close()

    async def ask_many(
        self,
        requests: list[dict[str, Any]],
        *,
        request_namespace: str,
    ) -> list[dict[str, Any]]:
        if not self.config.enabled:
            raise RuntimeError("External-model consultation is disabled")
        bounded = requests[: self.config.max_batch_size]
        return await asyncio.gather(
            *(
                self._ask_one(
                    request,
                    request_namespace=request_namespace,
                    call_index=index,
                )
                for index, request in enumerate(bounded, start=1)
            )
        )

    async def _ask_one(
        self,
        request: dict[str, Any],
        *,
        request_namespace: str,
        call_index: int,
    ) -> dict[str, Any]:
        query = str(request.get("query") or "").strip()
        if not query:
            return {
                "ok": False,
                "status": "failed",
                "provider": "frontierrl-agent",
                "model": self.model_config.model,
                "error": "query is required",
            }
        system = str(request.get("system") or "").strip()
        context = str(request.get("context") or "").strip()
        task_mode = str(request.get("task_mode") or "research").strip().casefold()
        if task_mode == "review":
            return await self._ask_direct_review(
                request,
                request_namespace=request_namespace,
                call_index=call_index,
            )
        strategy_mode = task_mode == "strategy"
        helper_agent_config = self.agent_config
        if strategy_mode:
            helper_agent_config = self.agent_config.model_copy(
                deep=True,
                update={
                    "max_steps": min(self.agent_config.max_steps, 4),
                    "min_search_calls_before_final": 0,
                    "require_citations": False,
                    "require_opened_citation_support": False,
                },
            )
        question_parts = [
            "Act as an independent research helper for another agent.",
            "Use the supplied public-web tools whenever they can verify a material claim.",
            (
                "Before the first search, use your own knowledge only to form several specific "
                "named hypotheses; treat them as unverified leads, never as evidence. Search those "
                "names and their rarest relations instead of repeatedly pasting the clue bundle. "
                "If no name comes to mind, vary geography, source language, historical vocabulary, "
                "and the constrained collaborator or artifact until searches expose named entities."
            ),
            (
                "Before finalizing, run independent searches that try to falsify the leading "
                "candidate and test whether every hard clue preserves its exact relation type. "
                "Treat the clues as a conjunction: do not rescue a materially contradicted "
                "candidate by reinterpretation, and return the most specific source-supported "
                "answer rather than a broader category. Do not infer nationality from birthplace "
                "or primary occupation from one artifact. When the target is unknown, search from "
                "the rarest constrained collaborator, author, spouse, artifact, quotation, or "
                "dated source and traverse that relation back to candidate targets."
            ),
            (
                "Translate paraphrased scholarly clues into field terminology such as curated "
                "database, experimentally validated, catalog, registry, revision, corpus, or "
                "open-access paper. Treat a paper matching only one clue as candidate generation, "
                "enumerate its authors, and test each author against every remaining relation. "
                "Never bridge a hard clue with words such as likely, associated with, or a "
                "specific. For questions with multiple proximity or route-distance constraints, "
                "use geo_search, supply each stated distance as expected_distance_miles, and rank "
                "shared candidates by routed distance error before selecting an answer."
            ),
        ]
        if strategy_mode:
            question_parts.append(
                "This request is for a retrieval strategy, not a factual benchmark answer. Use "
                "the supplied evidence to produce the requested query-plan JSON immediately; "
                "browse only if it materially improves the plan. Put the exact JSON object in "
                "final.explanation with no surrounding markdown, set final.exact_answer to "
                "'query strategy', and use an empty citations list."
            )
        else:
            question_parts.append(
                "Complete through the final tool. Put the full requested deliverable in final."
                "explanation; if the request specifies JSON, place that exact JSON object in the "
                "explanation with no surrounding markdown. Put only a concise recommendation in "
                "final.exact_answer. When the task asks who someone is or asks for a name, the exact "
                "answer must be a specific named entity. A category such as 'the actress,' a clue "
                "restatement, or a phrase beginning 'the celebrity who' is not an answer."
            )
        if system:
            question_parts.append(f"Request-specific instructions:\n{system}")
        question_parts.append(f"Task:\n{query}")
        if context:
            question_parts.append(f"Supplied context:\n{context}")
        question = "\n\n".join(question_parts)
        namespace = f"{request_namespace}:star2-agent:{call_index}"
        events: list[dict[str, Any]] = []

        def event_sink(event: dict[str, Any]) -> None:
            events.append(event)
            if event.get("event") in {
                "turn_started",
                "action_selected",
                "action_completed",
                "trial_final",
                "trial_no_final",
            }:
                print(
                    "[bc250-helper]"
                    f" namespace={namespace}"
                    f" step={event.get('step', '-')}"
                    f" phase={event.get('event')}"
                    f" action={event.get('action', '-')}",
                    flush=True,
                )

        runner = self.runner_factory(
            self.model_config,
            helper_agent_config,
            self.browser_config,
            self.search,
            self.browser,
            model_client=self.client,
            external_model_config=self.disabled_external_config,
            external_model_broker=None,
            event_sink=event_sink,
            system_prompt=_STRATEGY_SYSTEM_PROMPT if strategy_mode else None,
            initial_force_final=strategy_mode,
        )
        try:
            async with self._semaphore:
                outcome = await asyncio.wait_for(
                    runner.run(question, request_namespace=namespace),
                    timeout=self.config.timeout_seconds,
                )
        except TimeoutError:
            return self._partial_timeout_result(
                namespace=namespace,
                events=events,
                error=f"Star helper exceeded {self.config.timeout_seconds:g} seconds",
            )
        except Exception as exc:  # noqa: BLE001 - return one failed sibling without cancelling batch
            return {
                "ok": False,
                "status": "failed",
                "provider": "frontierrl-agent",
                "model": self.model_config.model,
                "error": f"{type(exc).__name__}: {exc}",
                "agent_events": len(events),
            }
        finally:
            close_runner = getattr(runner, "close", None)
            if callable(close_runner):
                try:
                    await close_runner()
                except Exception as exc:  # noqa: BLE001 - cleanup must not erase helper evidence
                    print(
                        f"[bc250-helper] namespace={namespace} cleanup_error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
        return self._result_from_outcome(
            request,
            outcome,
            namespace=namespace,
            events=events,
            require_citations=helper_agent_config.require_citations,
        )

    async def _ask_direct_review(
        self,
        request: dict[str, Any],
        *,
        request_namespace: str,
        call_index: int,
    ) -> dict[str, Any]:
        """Run a synchronous evidence review without wrapping it in a research-agent loop."""
        query = str(request.get("query") or "").strip()
        if not query:
            return {
                "ok": False,
                "status": "failed",
                "provider": "frontierrl-agent",
                "model": self.model_config.model,
                "error": "query is required",
            }
        system = str(request.get("system") or "").strip()
        context = str(request.get("context") or "").strip()
        namespace = f"{request_namespace}:star2-review:{call_index}"
        conversation_id = "bc250-review-" + hashlib.sha256(
            namespace.encode("utf-8")
        ).hexdigest()[:24]
        messages = [
            {
                "role": "system",
                "content": (
                    system
                    + "\n\nThis is a synchronous review-only call. Do not browse or call tools. "
                    "Return the requested verdict directly in the final answer."
                ).strip(),
            },
            {
                "role": "user",
                "content": query + ("\n\nSupplied context:\n" + context if context else ""),
            },
        ]
        try:
            async with self._semaphore:
                response = await self.client.chat(
                    messages,
                    request_headers={"X-FRL-Conversation-Id": conversation_id},
                )
        except Exception as exc:  # noqa: BLE001 - a review failure must fail closed
            return {
                "ok": False,
                "status": "failed",
                "request_id": namespace,
                "provider": "frontierrl-agent",
                "model": self.model_config.model,
                "error": f"{type(exc).__name__}: {exc}",
            }
        content = response.content.strip()
        return {
            "ok": bool(content),
            "status": "succeeded" if content else "failed",
            "request_id": response.response_id or namespace,
            "provider": "frontierrl-agent",
            "model": self.model_config.model,
            "content": content,
            "reasoning": response.raw_message.get("reasoning"),
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }

    def _partial_timeout_result(
        self,
        *,
        namespace: str,
        events: list[dict[str, Any]],
        error: str,
    ) -> dict[str, Any]:
        reasoning: list[str] = []
        transcript: list[dict[str, Any]] = []
        for event in events:
            if event.get("event") == "model_response":
                value = str(event.get("assistant_reasoning") or "").strip()
                if value:
                    reasoning.append(value)
                transcript.append(event)
        queries = self._search_queries_from_transcript(transcript)
        citations = self._source_urls_from_events(events)
        sections = ["Partial Star-2 research recovered after the helper timeout."]
        if reasoning:
            sections.append("Latest research notes:\n" + reasoning[-1][-12_000:])
        if queries:
            sections.append("Executed search queries:\n- " + "\n- ".join(queries[-12:]))
        if citations:
            sections.append("Observed source URLs:\n- " + "\n- ".join(citations[:12]))
        content = "\n\n".join(sections)
        return {
            "ok": False,
            "status": "failed",
            "request_id": namespace,
            "provider": "frontierrl-agent",
            "model": self.model_config.model,
            "content": content,
            "error": error,
            "agent_events": len(events),
            "agent_search_queries": queries,
            "citations": citations,
        }

    @staticmethod
    def _source_urls_from_events(events: list[dict[str, Any]]) -> list[str]:
        urls: list[str] = []
        stack: list[Any] = [
            event.get("result")
            for event in events
            if event.get("event") == "action_completed" and event.get("result") is not None
        ]
        visited = 0
        while stack and visited < 20_000 and len(urls) < 40:
            value = stack.pop()
            visited += 1
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in {"url", "final_url", "requested_url"} and isinstance(child, str):
                        parsed = urlparse(child)
                        if parsed.scheme in {"http", "https"} and parsed.netloc:
                            urls.append(child)
                    elif isinstance(child, (dict, list)):
                        stack.append(child)
            elif isinstance(value, list):
                stack.extend(value)
        return list(dict.fromkeys(urls))

    def _result_from_outcome(
        self,
        request: dict[str, Any],
        outcome: AgentOutcome,
        *,
        namespace: str,
        events: list[dict[str, Any]],
        require_citations: bool,
    ) -> dict[str, Any]:
        contract = f"{request.get('system') or ''}\n{request.get('query') or ''}"
        if _FINAL_ACTION_CONTRACT.search(contract):
            content = canonical_json(
                {
                    "action": "final",
                    "explanation": outcome.explanation,
                    "exact_answer": outcome.exact_answer or "",
                    "confidence": outcome.confidence,
                    "citations": outcome.citations,
                }
            )
        else:
            content = outcome.explanation.strip() or outcome.response_text.strip()
            if outcome.exact_answer and outcome.exact_answer not in content:
                content += f"\n\nRecommended exact answer: {outcome.exact_answer}"
            if outcome.citations:
                content += "\n\nSources:\n" + "\n".join(outcome.citations)
        citations_satisfied = bool(outcome.citations) or not require_citations
        succeeded = outcome.status == "completed" and bool(content) and citations_satisfied
        result = {
            "ok": succeeded,
            "status": "succeeded" if succeeded else "failed",
            "request_id": namespace,
            "provider": "frontierrl-agent",
            "model": self.model_config.model,
            "content": content,
            "exact_answer": outcome.exact_answer,
            "confidence": outcome.confidence,
            "citations": list(outcome.citations),
            "agent_search_queries": self._search_queries_from_transcript(outcome.transcript),
            "usage": {
                "prompt_tokens": outcome.usage.input_tokens,
                "completion_tokens": outcome.usage.output_tokens,
                "total_tokens": outcome.usage.input_tokens + outcome.usage.output_tokens,
            },
            "agent": {
                "status": outcome.status,
                "steps": outcome.steps,
                "search_calls": outcome.search_calls,
                "page_opens": outcome.page_opens,
                "find_calls": outcome.find_calls,
                "events": len(events),
                "errors": outcome.errors[-3:],
            },
        }
        if not citations_satisfied:
            result["error"] = "Star helper final answer omitted required citations"
        return result

    @staticmethod
    def _search_queries_from_transcript(transcript: list[dict[str, Any]]) -> list[str]:
        queries: list[str] = []
        for message in transcript:
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name") or "")
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        continue
                if not isinstance(arguments, dict):
                    continue
                if name == "search":
                    value = str(arguments.get("query") or "").strip()
                    if value:
                        queries.append(value)
                elif name == "search_many":
                    values = arguments.get("queries")
                    if isinstance(values, list):
                        queries.extend(str(value).strip() for value in values if str(value).strip())
        return list(dict.fromkeys(queries))


__all__ = ["AgentExternalModelBroker"]
