from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from .agent import AgentRunner
from .config import AgentConfig, BrowserConfig, ExternalModelConfig, ModelConfig
from .llm import OpenAICompatibleClient, settings_from_model_config
from .types import AgentOutcome
from .util import canonical_json

_FINAL_ACTION_CONTRACT = re.compile(r'\{[^{}]*["\']action["\']\s*:\s*["\']final["\']', re.I)


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
        )
        self.agent_config = agent_config.model_copy(
            deep=True,
            update={
                "max_steps": config.agent_max_steps,
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
        question_parts = [
            "Act as an independent research helper for another agent.",
            "Use the supplied public-web tools whenever they can verify a material claim.",
            (
                "Complete through the final tool. Put the full requested deliverable in final."
                "explanation; if the request specifies JSON, place that exact JSON object in the "
                "explanation with no surrounding markdown. Put only a concise recommendation in "
                "final.exact_answer."
            ),
        ]
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
            self.agent_config,
            self.browser_config,
            self.search,
            self.browser,
            model_client=self.client,
            external_model_config=self.disabled_external_config,
            external_model_broker=None,
            event_sink=event_sink,
        )
        try:
            async with self._semaphore:
                outcome = await asyncio.wait_for(
                    runner.run(question, request_namespace=namespace),
                    timeout=self.config.timeout_seconds,
                )
        except TimeoutError:
            return {
                "ok": False,
                "status": "failed",
                "provider": "frontierrl-agent",
                "model": self.model_config.model,
                "error": f"Star helper exceeded {self.config.timeout_seconds:g} seconds",
                "agent_events": len(events),
            }
        except Exception as exc:  # noqa: BLE001 - return one failed sibling without cancelling batch
            return {
                "ok": False,
                "status": "failed",
                "provider": "frontierrl-agent",
                "model": self.model_config.model,
                "error": f"{type(exc).__name__}: {exc}",
                "agent_events": len(events),
            }
        return self._result_from_outcome(request, outcome, namespace=namespace, events=events)

    def _result_from_outcome(
        self,
        request: dict[str, Any],
        outcome: AgentOutcome,
        *,
        namespace: str,
        events: list[dict[str, Any]],
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
        return {
            "ok": outcome.status == "completed" and bool(content),
            "status": "succeeded" if outcome.status == "completed" and content else "failed",
            "request_id": namespace,
            "provider": "frontierrl-agent",
            "model": self.model_config.model,
            "content": content,
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


__all__ = ["AgentExternalModelBroker"]
