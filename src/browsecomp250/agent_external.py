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
_STRATEGY_GENERIC_ENTITY_WORDS = {
    "a",
    "an",
    "actor",
    "actress",
    "answer",
    "artist",
    "author",
    "book",
    "boulevard",
    "candidate",
    "city",
    "company",
    "country",
    "event",
    "european",
    "film",
    "first",
    "guide",
    "institution",
    "interview",
    "interviewee",
    "location",
    "organization",
    "paper",
    "participant",
    "person",
    "place",
    "planner",
    "project",
    "publication",
    "question",
    "road",
    "researcher",
    "scientist",
    "second",
    "song",
    "start",
    "stop",
    "street",
    "student",
    "students",
    "subject",
    "the",
    "this",
    "time",
    "tour",
    "tuesday",
    "university",
    "urban",
    "website",
}
_STRATEGY_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
_STRATEGY_SYSTEM_PROMPT = """
You are a retrieval-strategy controller. The supplied task asks for a query plan, not a factual
answer. Read the supplied question, evidence, and prior queries, then return exactly one native
final tool call immediately without browsing. Put the requested JSON object verbatim in final.explanation, set
final.exact_answer to "query strategy", confidence to a calibrated number, and citations to an
empty list. Never search for benchmark dumps, canaries, leaked questions, or reference answers.
A hypothesis must name a concrete person, place, project, institution, publication, or source
collection using proper nouns; a paraphrase such as "the urban planner" is not a hypothesis. Put
two to five alternatives in separate hypotheses array elements, with one candidate identity per
element. At least two queries must test named hypotheses and use different entity-relation routes
instead of repeating the clue bundle. A retry must directly repair the stated rejection reason.
Capitalizing role nouns copied from the task does not create a named hypothesis: introduce concrete
proper names that add retrieval information not already present in the supplied task.
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
        strategy_source_text = str(
            request.get("_strategy_source_text") or f"{query}\n{context}"
        )
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
            (
                "You are a leaf research worker. You cannot delegate, ask another model, or call "
                "ask_external_model. Complete the assigned research yourself with search, open, "
                "find, notes, and the final action."
            ),
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
        strategy_retry = int(request.get("_strategy_retry") or 0)
        retry_suffix = f":strategy-retry-{strategy_retry}" if strategy_retry else ""
        namespace = f"{request_namespace}:star2-agent:{call_index}{retry_suffix}"
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
        result = self._result_from_outcome(
            request,
            outcome,
            namespace=namespace,
            events=events,
            require_citations=helper_agent_config.require_citations,
        )
        if not strategy_mode and outcome.status != "completed":
            partial = self._partial_timeout_result(
                namespace=namespace,
                events=events,
                error=f"Star helper ended with status {outcome.status}",
            )
            partial["citations"] = list(
                dict.fromkeys([*partial.get("citations", []), *outcome.citations])
            )
            partial["usage"] = result.get("usage")
            partial["agent"] = result.get("agent")
            partial["exact_answer"] = outcome.exact_answer
            partial["confidence"] = outcome.confidence
            partial["partial_evidence_recovered"] = True
            return partial
        if not strategy_mode:
            return result
        strategy_error = self._strategy_quality_error(
            str(result.get("content") or ""),
            source_text=strategy_source_text,
        )
        if not strategy_error:
            result["strategy_attempts"] = strategy_retry + 1
            return result
        if strategy_retry < 2:
            retry_request = dict(request)
            retry_request["_strategy_retry"] = strategy_retry + 1
            retry_request["_strategy_source_text"] = strategy_source_text
            retry_request["query"] = (
                query
                + "\n\nThe previous strategy was rejected by the answer-blind quality gate: "
                + strategy_error
                + ". Return a materially different JSON plan. Name concrete proper-noun "
                "hypotheses from distinct domains or geographies, then write short queries around "
                "those names, source-native terminology, and different relation edges. Do not copy "
                "the clue sentences. Rejected response:\n"
                + str(result.get("content") or "")[:6_000]
            )
            retry_result = await self._ask_one(
                retry_request,
                request_namespace=request_namespace,
                call_index=call_index,
            )
            retry_result["usage"] = self._sum_usage(
                result.get("usage"), retry_result.get("usage")
            )
            retry_result["strategy_rejections"] = [
                {
                    "attempt": strategy_retry + 1,
                    "request_id": result.get("request_id"),
                    "reason": strategy_error,
                },
                *list(retry_result.get("strategy_rejections") or []),
            ]
            return retry_result
        repaired = self._repair_strategy_payload(
            str(result.get("content") or ""),
            source_text=strategy_source_text,
        )
        if repaired is not None:
            result.update(
                {
                    "ok": True,
                    "status": "succeeded",
                    "content": canonical_json(repaired),
                    "error": None,
                    "strategy_attempts": strategy_retry + 1,
                    "strategy_repaired": True,
                    "strategy_gate_warning": strategy_error,
                }
            )
            return result
        result.update(
            {
                "ok": False,
                "status": "failed",
                "error": (
                    "Answer-blind strategy quality gate rejected all attempts: " + strategy_error
                ),
                "strategy_attempts": strategy_retry + 1,
            }
        )
        return result

    @staticmethod
    def _sum_usage(*values: Any) -> dict[str, int]:
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for value in values:
            if not isinstance(value, dict):
                continue
            for key in totals:
                try:
                    totals[key] += max(0, int(value.get(key) or 0))
                except (TypeError, ValueError):
                    continue
        totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
        return totals

    @staticmethod
    def _strategy_json_object(text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("queries"), list):
                return value
        return None

    @classmethod
    def _strategy_candidate_names(cls, payload: dict[str, Any]) -> list[str]:
        hypotheses = [
            " ".join(str(value).split()).strip()
            for value in payload.get("hypotheses") or []
            if str(value).strip()
        ]
        candidates: list[str] = []
        for hypothesis in hypotheses:
            quoted = re.findall(r"['\"]([^'\"]{2,100})['\"]", hypothesis)
            candidate_values = quoted or ([hypothesis] if len(hypothesis.split()) <= 10 else [])
            for candidate in candidate_values:
                candidate = candidate.strip(" .,:;()[]{}")
                words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9'’-]+", candidate)
                if not words or len(words) > 10:
                    continue
                if all(word.casefold() in _STRATEGY_GENERIC_ENTITY_WORDS for word in words):
                    continue
                if not any(word[:1].isupper() for word in words if word):
                    continue
                if sum(character.isdigit() for character in candidate) > len(candidate) / 3:
                    continue
                candidates.append(candidate)
        return list(dict.fromkeys(candidates))[:5]

    @classmethod
    def _repair_strategy_payload(
        cls,
        text: str,
        *,
        source_text: str = "",
    ) -> dict[str, Any] | None:
        payload = cls._strategy_json_object(text)
        if payload is None:
            return None
        candidates = cls._strategy_candidate_names(payload)
        if len(candidates) < 2:
            return None
        source_classes = [
            " ".join(str(value).split()).strip()
            for value in payload.get("source_classes") or []
            if str(value).strip()
        ]
        discriminators = [
            " ".join(str(value).split()).strip()
            for value in payload.get("discriminators") or []
            if str(value).strip()
        ]
        anchors = list(dict.fromkeys(source_classes + discriminators))
        if len(source_classes) < 2 or len(discriminators) < 2 or not anchors:
            return None
        queries: list[str] = []
        for index, candidate in enumerate(candidates):
            anchor = anchors[index % len(anchors)]
            query = f'"{candidate}" {anchor}'
            if len(re.findall(r"\w+", query)) <= 14:
                queries.append(query)
        anchor_index = len(queries)
        while len(queries) < 3 and anchor_index < len(anchors) + len(candidates) * 2:
            candidate = candidates[anchor_index % len(candidates)]
            anchor = anchors[anchor_index % len(anchors)]
            query = f'"{candidate}" {anchor}'
            if query not in queries and len(re.findall(r"\w+", query)) <= 14:
                queries.append(query)
            anchor_index += 1
        repaired = {
            "hypotheses": candidates,
            "queries": queries[:7],
            "source_classes": list(dict.fromkeys(source_classes))[:7],
            "discriminators": list(dict.fromkeys(discriminators))[:7],
        }
        return (
            repaired
            if cls._strategy_quality_error(
                canonical_json(repaired),
                source_text=source_text,
            )
            is None
            else None
        )

    @classmethod
    def _strategy_quality_error(cls, text: str, *, source_text: str = "") -> str | None:
        payload = cls._strategy_json_object(text)
        if payload is None:
            return "missing a parseable JSON object with a queries array"
        queries = [
            " ".join(str(value).split())
            for value in payload.get("queries") or []
            if str(value).strip()
        ]
        source_classes = [
            str(value).strip()
            for value in payload.get("source_classes") or []
            if str(value).strip()
        ]
        discriminators = [
            str(value).strip()
            for value in payload.get("discriminators") or []
            if str(value).strip()
        ]
        candidate_names = cls._strategy_candidate_names(payload)
        if len(candidate_names) < 2:
            return "fewer than two concrete candidate hypotheses"
        source_terms = {
            value.casefold()
            for value in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", source_text)
        }
        novel_candidates = 0
        for candidate in candidate_names:
            candidate_terms = {
                value.casefold()
                for value in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", candidate)
                if value.casefold() not in _STRATEGY_GENERIC_ENTITY_WORDS
            }
            if candidate_terms - source_terms:
                novel_candidates += 1
        if source_terms and novel_candidates < 2:
            return "fewer than two hypotheses introduced concrete entity terms beyond the task clues"
        named_tokens: set[str] = set()
        for candidate in candidate_names:
            for token in re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]{2,}\b", candidate):
                normalized = token.casefold()
                if normalized not in _STRATEGY_GENERIC_ENTITY_WORDS:
                    named_tokens.add(normalized)
        if not named_tokens:
            return "hypotheses paraphrased the clues instead of naming a concrete proper noun"
        if not 3 <= len(queries) <= 7:
            return "queries must contain between three and seven entries"
        normalized_queries = {" ".join(value.casefold().split()) for value in queries}
        if len(normalized_queries) != len(queries):
            return "queries contained duplicates"
        if any(len(re.findall(r"\w+", query)) > 14 for query in queries):
            return "a query exceeded fourteen words"
        candidate_term_sets = []
        for candidate in candidate_names:
            terms = {
                value.casefold()
                for value in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", candidate)
                if value.casefold() not in _STRATEGY_GENERIC_ENTITY_WORDS
            }
            if terms:
                candidate_term_sets.append(terms)
        query_term_sets: list[set[str]] = []
        matched_candidates: set[int] = set()
        for query in queries:
            terms = {
                value.casefold()
                for value in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", query)
                if value.casefold() not in _STRATEGY_QUERY_STOPWORDS
            }
            query_term_sets.append(terms)
            for candidate_index, candidate_terms in enumerate(candidate_term_sets):
                required_matches = 1 if len(candidate_terms) <= 2 else 2
                if len(terms & candidate_terms) >= required_matches:
                    matched_candidates.add(candidate_index)
        if len(matched_candidates) < 2:
            return "queries did not explicitly test two distinct named hypotheses"
        low_overlap_pairs = 0
        for left_index, left in enumerate(query_term_sets):
            for right in query_term_sets[left_index + 1 :]:
                union = left | right
                overlap = len(left & right) / len(union) if union else 1.0
                if overlap <= 0.5:
                    low_overlap_pairs += 1
        if low_overlap_pairs < 2:
            return "queries did not test sufficiently distinct entity-relation routes"
        if len({value.casefold() for value in source_classes}) < 2:
            return "fewer than two source classes"
        if len({value.casefold() for value in discriminators}) < 2:
            return "fewer than two candidate discriminators"
        return None

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
                    request_headers={
                        **AgentRunner._routing_headers(namespace),
                        "X-FRL-Conversation-Id": conversation_id,
                    },
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
