#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from browsecomp250.agent import AgentRunner
from browsecomp250.config import ModelConfig, load_config
from browsecomp250.dataset import load_items
from browsecomp250.grading import Grader
from browsecomp250.llm import OpenAICompatibleClient, settings_from_model_config
from browsecomp250.llm.protocol import action_from_tool_call
from browsecomp250.llm.tools import tool_schemas
from browsecomp250.types import AgentAction, ModelResponse
from browsecomp250.util import atomic_write_json, atomic_write_text, canonical_json, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize and curate an interrupted teacher-forced BrowseComp trace."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/star-headline.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--max-final-attempts", type=int, default=3)
    return parser.parse_args()


def read_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evidence_from_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    latest_successful_search: dict[str, Any] = {}
    for event in events:
        if event.get("event") != "action_completed":
            continue
        result = event.get("result")
        if not isinstance(result, dict):
            continue
        page = result.get("page")
        if isinstance(page, dict):
            pages.append(page)
        automatic = result.get("automatic_page_inspection")
        if isinstance(automatic, dict):
            pages.extend(value for value in automatic.get("pages") or [] if isinstance(value, dict))
        if event.get("action") in {"search", "search_many"} and result.get("ok"):
            latest_successful_search = result
    deduplicated: dict[str, dict[str, Any]] = {}
    for page in pages:
        url = str(page.get("final_url") or page.get("requested_url") or "").strip()
        if url:
            deduplicated[url] = {
                "url": url,
                "title": str(page.get("title") or ""),
                "text": str(page.get("text") or "")[:16_000],
            }
    return list(deduplicated.values()), latest_successful_search


def parse_final_action(response: ModelResponse) -> AgentAction:
    tool_calls = response.raw_message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        action = action_from_tool_call(tool_calls[0])
        if action.action != "final":
            raise RuntimeError(f"Final-only request returned {action.action!r}")
        return action
    return AgentRunner._plain_final_action(response.content)


def parse_reviewer(response: ModelResponse) -> dict[str, Any]:
    reasoning = response.raw_message.get("reasoning")
    review_text = response.content
    if isinstance(reasoning, str) and reasoning.strip():
        review_text += "\n\n" + reasoning
    return AgentRunner._blocking_guidance_review_payload(
        {
            "ok": True,
            "content": review_text,
            "request_id": response.response_id,
        }
    )


def accepted_training_messages(
    events: list[dict[str, Any]],
    *,
    system_prompt: str,
    initial_user: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user},
    ]
    by_step: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        step = event.get("step")
        if isinstance(step, int):
            by_step.setdefault(step, []).append(event)
    completed_steps = [
        event
        for event in events
        if event.get("event") == "scripted_guidance_step_completed"
    ]
    for completion in completed_steps:
        step = int(completion["step"])
        rows = by_step.get(step, [])
        started = next(
            (row for row in rows if row.get("event") == "scripted_guidance_step_started"),
            None,
        )
        model_response = next(
            (row for row in reversed(rows) if row.get("event") == "model_response"),
            None,
        )
        action_completed = next(
            (
                row
                for row in reversed(rows)
                if row.get("event") == "action_completed" and row.get("ok") is True
            ),
            None,
        )
        if not started or not model_response or not action_completed:
            continue
        contract = started.get("scripted_step") or {}
        messages.append(
            {
                "role": "system",
                "content": "Teacher-forced research step:\n" + canonical_json(contract),
            }
        )
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": str(model_response.get("assistant_content") or ""),
        }
        reasoning = model_response.get("assistant_reasoning")
        if reasoning:
            assistant["reasoning"] = reasoning
        tool_calls = model_response.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            assistant["tool_calls"] = tool_calls[:1]
        messages.append(assistant)
        result_text = json.dumps(action_completed.get("result") or {}, ensure_ascii=False)
        if assistant.get("tool_calls"):
            tool_call = assistant["tool_calls"][0]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"accepted-{step}"),
                    "name": str(
                        (tool_call.get("function") or {}).get("name")
                        or action_completed.get("action")
                        or "tool"
                    ),
                    "content": result_text,
                }
            )
        else:
            messages.append({"role": "user", "content": "Tool result:\n" + result_text})
    return messages


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)
    config = load_config(args.config)
    item = {value.subset_rank: value for value in load_items(config.dataset)}[args.rank]
    plan = args.plan.read_text(encoding="utf-8").strip()
    events = read_events(args.events)
    pages, search_evidence = evidence_from_events(events)
    if len(pages) < 4 or not search_evidence:
        raise RuntimeError("Saved trace does not contain enough successful public evidence")

    final_model = config.model.model_copy(
        deep=True,
        update={
            "api_base": "http://127.0.0.1:8003/v1",
            "model": "frontierrl/star-7",
            "temperature": 0.3,
            "max_output_tokens": 16384,
            "response_chain": False,
            "extra_body": {
                "top_p": 0.95,
                "parallel_tool_calls": False,
                "vllm_xargs": {"frontierrl_max_denoising_steps": 48},
            },
        },
    )
    reviewer_model = ModelConfig(
        api_base="http://127.0.0.1:8003/v1",
        api_key=config.external_model.agent_api_key,
        allow_empty_api_key=config.external_model.agent_allow_empty_api_key,
        model="frontierrl/star-2",
        protocol="tools",
        temperature=0.7,
        max_output_tokens=16384,
        timeout_seconds=config.external_model.timeout_seconds,
        max_retries=config.external_model.max_retries,
        extra_body={
            "top_p": 0.95,
            "vllm_xargs": {"frontierrl_max_denoising_steps": 48},
        },
    )
    final_client = OpenAICompatibleClient(settings_from_model_config(final_model))
    reviewer_client = OpenAICompatibleClient(settings_from_model_config(reviewer_model))
    grader = Grader(config.grader)
    final_tools = [
        tool
        for tool in tool_schemas(include_external_model=False)
        if (tool.get("function") or {}).get("name") == "final"
    ]
    final_system = (
        "You are the final synthesis stage of a completed research trajectory. Retrieval is "
        "finished. Read the original question and inspected public evidence, reason carefully, "
        "then call the final tool exactly once. Set exact_answer to only the requested concrete "
        "answer. Include a concise explanation, calibrated confidence, and inspected citation "
        "URLs. Never emit or request a search, open, note, or helper action."
    )
    final_user = (
        "Research plan:\n"
        + plan
        + "\n\nOriginal question:\n"
        + item.question
        + "\n\nInspected public pages:\n"
        + canonical_json(pages)
        + "\n\nVerification-search evidence:\n"
        + canonical_json(search_evidence)
    )
    final_messages: list[dict[str, Any]] = [
        {"role": "system", "content": final_system},
        {"role": "user", "content": final_user},
    ]
    attempts: list[dict[str, Any]] = []
    accepted_response: ModelResponse | None = None
    accepted_action: AgentAction | None = None
    accepted_review: dict[str, Any] | None = None

    def checkpoint(status: str, *, error: str | None = None) -> None:
        payload = {
            "schema_version": "1.0",
            "updated_at": utc_now_iso(),
            "experiment": "teacher-forced source-guided, answer-label-redacted",
            "benchmark_eligible": False,
            "item_id": item.item_id,
            "subset_rank": item.subset_rank,
            "source_index": item.source_index,
            "status": status,
            "error": error,
            "attempts": attempts,
            "source_event_path": str(args.events.resolve()),
        }
        atomic_write_json(args.output, payload)
        os.chmod(args.output, 0o600)
        print(
            json.dumps(
                {
                    "stage": status,
                    "attempts": len(attempts),
                    "error": error,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    checkpoint("starting_final_synthesis")
    try:
        for attempt in range(1, args.max_final_attempts + 1):
            started_at = time.perf_counter()
            print(
                json.dumps({"stage": "final_model_started", "attempt": attempt}),
                flush=True,
            )
            response = await final_client.chat(
                final_messages,
                tools=final_tools,
                tool_choice={"type": "function", "function": {"name": "final"}},
                request_headers={
                    "X-FRL-Conversation-Id": f"bc250-teacher-final-{item.item_id}-{attempt}"
                },
            )
            try:
                action = parse_final_action(response)
            except Exception as exc:  # noqa: BLE001
                attempts.append(
                    {
                        "attempt": attempt,
                        "accepted": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "response": asdict(response),
                    }
                )
                checkpoint("final_proposal_malformed", error=str(exc))
                final_messages.append(
                    {
                        "role": "system",
                        "content": "The final-only response was malformed. Call final exactly once.",
                    }
                )
                continue
            print(
                json.dumps(
                    {
                        "stage": "final_model_completed",
                        "attempt": attempt,
                        "seconds": round(time.perf_counter() - started_at, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            review_context = canonical_json(
                {
                    "original_question": item.question,
                    "research_plan": plan,
                    "inspected_public_pages": pages,
                    "verification_search_evidence": search_evidence,
                    "proposed_final": action.payload,
                }
            )
            review_response = await reviewer_client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a blocking evidence reviewer. You do not know the reference "
                            "answer. Return one JSON object only with verdict PASS or BLOCK, reason, "
                            "material_deviations, and required_next_actions. PASS only when the "
                            "proposed person name is directly supported by the supplied public "
                            "evidence, fits the target article attribution, and has no material "
                            "identity contradiction. Honor source roles explicitly established by "
                            "the supplied plan: a source labeled gold or target is the article whose "
                            "wording answers the question; other sources are corroboration unless "
                            "the plan says otherwise. A second source crediting another person with "
                            "a different historical milestone is not by itself a contradiction. Do "
                            "not demand a second answer-naming article when the question explicitly "
                            "asks what one identified article says. required_next_actions must be a "
                            "JSON array of strings."
                        ),
                    },
                    {"role": "user", "content": review_context},
                ],
                request_headers={
                    "X-FRL-Conversation-Id": f"bc250-teacher-review-{item.item_id}-{attempt}"
                },
            )
            review = parse_reviewer(review_response)
            attempts.append(
                {
                    "attempt": attempt,
                    "accepted": review["verdict"] == "PASS",
                    "response": asdict(response),
                    "proposed_final": action.payload,
                    "review": review,
                    "review_response": asdict(review_response),
                }
            )
            checkpoint(f"blocking_review_{review['verdict'].lower()}")
            if review["verdict"] == "PASS":
                accepted_response = response
                accepted_action = action
                accepted_review = review
                break
            final_messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": response.raw_message.get("tool_calls"),
                    },
                    {
                        "role": "system",
                        "content": (
                            "Blocking review:\n"
                            + canonical_json(review)
                            + "\nRepair the proposed final using only the supplied evidence, then "
                            "call final exactly once."
                        ),
                    },
                ]
            )
        if accepted_action is None or accepted_response is None or accepted_review is None:
            checkpoint(
                "blocked",
                error="No final proposal passed the blocking reviewer",
            )
            raise RuntimeError("No final proposal passed the blocking reviewer")
        response_text = AgentRunner._format_final(accepted_action.payload)
        checkpoint("official_grading")
        grade = await grader.grade(item.question, item.answer, response_text)
    finally:
        await grader.close()
        await reviewer_client.close()
        await final_client.close()

    system_prompt = (
        config.agent.system_prompt_path.read_text(encoding="utf-8").strip()
        if config.agent.system_prompt_path is not None
        else ""
    )
    initial_user = "Research plan:\n" + plan + "\n\nQuestion:\n" + item.question
    training_messages = accepted_training_messages(
        events,
        system_prompt=system_prompt,
        initial_user=initial_user,
    )
    training_messages.extend(
        [
            {
                "role": "system",
                "name": "scripted_final_context_reset",
                "content": final_system,
            },
            {"role": "user", "name": "scripted_final_evidence", "content": final_user},
            {
                "role": "assistant",
                "content": accepted_response.content,
                "reasoning": accepted_response.raw_message.get("reasoning"),
                "tool_calls": accepted_response.raw_message.get("tool_calls"),
            },
        ]
    )
    result = {
        "schema_version": "1.0",
        "created_at": utc_now_iso(),
        "experiment": "teacher-forced source-guided, answer-label-redacted",
        "benchmark_eligible": False,
        "item_id": item.item_id,
        "subset_rank": item.subset_rank,
        "source_index": item.source_index,
        "status": "completed" if grade.correct else "graded_incorrect",
        "correct": grade.correct,
        "answer_response": response_text,
        "extracted_answer": grade.extracted_answer,
        "grader_reasoning": grade.reasoning,
        "grader_usage": asdict(grade.usage),
        "blocking_review": accepted_review,
        "attempts": attempts,
        "accepted_training_messages": len(training_messages),
        "source_event_path": str(args.events.resolve()),
    }
    atomic_write_json(args.output, result)
    os.chmod(args.output, 0o600)
    if not grade.correct:
        raise RuntimeError("Official grader rejected the blocking-review-passed final")
    training_row = {
        "schema_version": "1.0",
        "item_id": item.item_id,
        "source_index": item.source_index,
        "trajectory_type": "teacher_forced_source_guided_answer_label_redacted",
        "messages": training_messages,
        "final_response": response_text,
        "correct": True,
        "metadata": {
            "benchmark_eligible": False,
            "blocking_final_review": True,
            "official_grader": config.grader.model,
        },
    }
    atomic_write_text(
        args.training_output,
        json.dumps(training_row, ensure_ascii=False, sort_keys=True) + "\n",
    )
    os.chmod(args.training_output, 0o600)
    return result


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    print(
        json.dumps(
            {
                "status": result["status"],
                "correct": result["correct"],
                "blocking_verdict": result["blocking_review"]["verdict"],
                "accepted_training_messages": result["accepted_training_messages"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
