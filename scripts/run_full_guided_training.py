#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from browsecomp250.agent import AgentRunner
from browsecomp250.agent.runner import SCRIPTED_FINAL_SYSTEM_PROMPT
from browsecomp250.agent_external import AgentExternalModelBroker
from browsecomp250.browser import PageFetcher
from browsecomp250.config import AppConfig, load_config
from browsecomp250.grading import Grader, grade_deterministic
from browsecomp250.guided_training import (
    audit_system_messages,
    compile_guided_steps,
    controller_label_leaks,
    curate_scripted_training_messages,
    training_message_quality,
)
from browsecomp250.llm import OpenAICompatibleClient, settings_from_model_config
from browsecomp250.prompts import AGENT_SYSTEM_PROMPT
from browsecomp250.search import create_search_provider
from browsecomp250.util import atomic_write_json, atomic_write_text, canonical_json, utc_now_iso

DEFAULT_STAR7_ENDPOINTS = ",".join(
    [
        "http://192.168.1.233:9304/v1",
        "http://192.168.1.233:9324/v1",
        "http://192.168.1.233:9334/v1",
    ]
    + [f"http://127.0.0.1:{port}/v1" for port in range(9375, 9383)]
)
DEFAULT_STAR2_ENDPOINTS = ",".join(
    ["http://192.168.1.233:9364/v1"]
    + [f"http://127.0.0.1:{port}/v1" for port in range(9383, 9390)]
)
GENERIC_INITIAL_GUIDANCE = (
    "This is an answer-redacted teacher-forced research trajectory. The controller will send "
    "exactly one operational plan step per turn. Execute only that step and one matching tool "
    "action, preserve all hard clue relations, and treat controller text as procedure rather than "
    "factual evidence. Derive the answer only from public tool evidence and your own reasoning."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create private one-plan-step-per-turn training traces for all BrowseComp guides."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/star-headline.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--guide-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=11)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--item-timeout-seconds", type=float, default=3600)
    parser.add_argument("--indices", help="Zero-based comma/range selection, for example 0-9,82")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--star7-endpoints", default=DEFAULT_STAR7_ENDPOINTS)
    parser.add_argument("--star2-endpoints", default=DEFAULT_STAR2_ENDPOINTS)
    parser.add_argument("--grader-endpoint", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--grader-model", default="frontierrl/star-2")
    parser.add_argument("--deterministic-only", action="store_true")
    return parser.parse_args()


def parse_indices(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            values.update(range(min(start, end), max(start, end) + 1))
        else:
            values.add(int(part))
    invalid = sorted(value for value in values if value < 0 or value >= total)
    if invalid:
        raise ValueError(f"Guide indices out of range: {invalid[:10]}")
    return sorted(values)


def endpoint_list(value: str) -> list[str]:
    endpoints = [part.strip().rstrip("/") for part in value.split(",") if part.strip()]
    if not endpoints:
        raise ValueError("At least one endpoint is required")
    return endpoints


def load_guide_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from browsecomp_guides.corpus import read_bundle

    artifacts = root / "artifacts"
    route = read_bundle(artifacts / "route_only_guides.aesgcm")
    oracle = read_bundle(artifacts / "full_oracle_guides.aesgcm")
    if len(route) != 1266 or len(oracle) != 1266:
        raise RuntimeError(f"Expected 1,266 guide pairs; got {len(route)} and {len(oracle)}")
    for index, (left, right) in enumerate(zip(route, oracle, strict=True)):
        if left["item"]["item_id"] != right["item"]["item_id"]:
            raise RuntimeError(f"Guide item mismatch at row {index}")
    return route, oracle


def preflight_compiled_guides(
    route_records: list[dict[str, Any]],
    oracle_records: list[dict[str, Any]],
    *,
    indices: list[int],
    max_attempts: int,
) -> dict[str, Any]:
    step_counts: list[int] = []
    mapped_source_items = 0
    for index in indices:
        route_record = route_records[index]
        oracle_record = oracle_records[index]
        question = str(oracle_record["item"]["question_text"])
        if (oracle_record.get("oracle") or {}).get("evidence_sources"):
            mapped_source_items += 1
        for attempt in range(1, max_attempts + 1):
            steps, review_guidance = compile_guided_steps(
                route_record,
                oracle_record,
                attempt=attempt,
            )
            leaks = controller_label_leaks(
                question=question,
                oracle_record=oracle_record,
                steps=steps,
                review_guidance=review_guidance,
            )
            if leaks:
                raise RuntimeError(
                    f"Guide row {index} attempt {attempt} leaked private aliases: {leaks!r}"
                )
            if not steps or steps[-1].get("allowed_actions") != ["final"]:
                raise RuntimeError(
                    f"Guide row {index} attempt {attempt} lacks a final-only terminal step"
                )
            for step in steps:
                allowed = step.get("allowed_actions")
                if not isinstance(allowed, list) or len(allowed) != 1:
                    raise RuntimeError(
                        f"Guide row {index} attempt {attempt} has a non-atomic step: {step!r}"
                    )
                if not str(step.get("id") or "").strip() or not str(
                    step.get("instruction") or ""
                ).strip():
                    raise RuntimeError(
                        f"Guide row {index} attempt {attempt} has an incomplete step"
                    )
            canonical_json(steps)
            canonical_json(json.loads(review_guidance))
            step_counts.append(len(steps))
    return {
        "selected_items": len(indices),
        "attempt_variants": len(step_counts),
        "mapped_source_items": mapped_source_items,
        "route_only_items": len(indices) - mapped_source_items,
        "minimum_steps": min(step_counts),
        "maximum_steps": max(step_counts),
        "mean_steps": round(sum(step_counts) / len(step_counts), 3),
        "controller_answer_label_leaks": 0,
        "item_specific_system_prompt_content": False,
        "scripted_guidance_role": "user",
        "passed": True,
    }


def item_directory(output_dir: Path, record: dict[str, Any]) -> Path:
    item = record["item"]
    return output_dir / "items" / f"{int(item['row_index']):04d}-{item['item_id']}"


def completed_training_path(output_dir: Path, record: dict[str, Any]) -> Path:
    return item_directory(output_dir, record) / "training.jsonl"


def existing_attempt_number(path: Path) -> int:
    values = []
    for candidate in path.glob("attempt-*-result.json"):
        try:
            values.append(int(candidate.name.split("-", 2)[1]))
        except (IndexError, ValueError):
            continue
    return max(values, default=0)


def load_attempted_candidates(path: Path) -> list[str]:
    status = path / "status.json"
    if not status.exists():
        return []
    try:
        value = json.loads(status.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    values = value.get("attempted_candidates") or value.get("rejected_candidates") or []
    return [str(item) for item in values if str(item).strip()]


def write_private_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)
    os.chmod(path, 0o600)


def write_private_text(path: Path, value: str) -> None:
    atomic_write_text(path, value)
    os.chmod(path, 0o600)


def configured_agent_system_prompt(config: AppConfig) -> str:
    path = config.agent.system_prompt_path
    if path is not None and path.exists():
        return path.read_text(encoding="utf-8").strip()
    return AGENT_SYSTEM_PROMPT


def reconstructed_initial_messages(
    config: AppConfig,
    *,
    question: str,
    scripted_step_count: int,
) -> list[dict[str, Any]]:
    agent_config = config.agent.model_copy(
        deep=True,
        update={
            "max_steps": max(80, scripted_step_count * 3 + 10),
            "max_search_calls": max(80, scripted_step_count * 7),
            "max_page_opens": max(100, scripted_step_count * 4),
            "force_final_after_seconds": 0,
        },
    )
    initial_user = (
        "Research plan:\n"
        + GENERIC_INITIAL_GUIDANCE
        + "\n\nQuestion:\n"
        + question
        + "\n\nBudgets: "
        + canonical_json(
            {
                "max_steps": agent_config.max_steps,
                "force_final_after_seconds": agent_config.force_final_after_seconds,
                "max_search_calls": agent_config.max_search_calls,
                "max_page_opens": agent_config.max_page_opens,
                "max_find_calls": agent_config.max_find_calls,
                "max_external_model_calls": config.external_model.max_calls_per_task,
            }
        )
    )
    return [
        {"role": "system", "content": configured_agent_system_prompt(config)},
        {"role": "user", "content": initial_user},
    ]


def mark_item_completed(
    item_dir: Path,
    oracle_record: dict[str, Any],
    *,
    result_status: str,
) -> None:
    status_path = item_dir / "status.json"
    status: dict[str, Any] = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}
    status.update(
        {
            "item_id": oracle_record["item"]["item_id"],
            "row_index": oracle_record["item"]["row_index"],
            "correct": True,
            "status": "completed",
            "attempts": existing_attempt_number(item_dir),
            "next_attempt": None,
            "last_result_status": result_status,
            "last_error": None,
            "updated_at": utc_now_iso(),
        }
    )
    write_private_json(status_path, status)


async def recover_verified_rows(
    *,
    output_dir: Path,
    config: AppConfig,
    oracle_records: list[dict[str, Any]],
    indices: list[int],
    grader: Grader | None,
    grader_semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    recovered: list[dict[str, Any]] = []
    system_prompt = configured_agent_system_prompt(config)
    for index in indices:
        oracle_record = oracle_records[index]
        item_dir = item_directory(output_dir, oracle_record)
        if completed_training_path(output_dir, oracle_record).exists():
            mark_item_completed(
                item_dir,
                oracle_record,
                result_status="existing_verified_training",
            )
            continue
        if not item_dir.exists():
            continue
        for result_path in sorted(item_dir.glob("attempt-*-result.json")):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if result.get("status") not in {"completed", "training_quality_rejected"}:
                continue
            attempt = int(result.get("attempt") or 0)
            response = str(result.get("answer_response") or "")
            reference = str(oracle_record["oracle"]["gold_answer"])
            try:
                grade, grade_mode = await grade_response(
                    question=str(oracle_record["item"]["question_text"]),
                    answer=reference,
                    response=response,
                    grader=grader,
                    grader_semaphore=grader_semaphore,
                )
            except Exception:  # noqa: BLE001 - leave uncertain legacy rows untouched
                continue
            if not grade.correct:
                continue
            event_path = Path(str(result.get("event_path") or ""))
            try:
                events = [
                    json.loads(line)
                    for line in event_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, json.JSONDecodeError):
                continue
            scripted_step_count = int(result.get("compiled_plan_steps") or 0)
            if scripted_step_count < 1:
                continue
            curated = curate_scripted_training_messages(
                events,
                initial_messages=reconstructed_initial_messages(
                    config,
                    question=str(oracle_record["item"]["question_text"]),
                    scripted_step_count=scripted_step_count,
                ),
                scripted_step_count=scripted_step_count,
            )
            quality = training_message_quality(curated)
            system_audit = audit_system_messages(
                transcript=curated,
                invariant_system_prompt=system_prompt,
                events=events,
                expected_final_system_prompt=SCRIPTED_FINAL_SYSTEM_PROMPT,
            )
            if not quality["passed"] or not system_audit["passed"]:
                continue
            training_row = {
                "schema_version": "1.0",
                "item_id": oracle_record["item"]["item_id"],
                "source_index": oracle_record["item"]["row_index"],
                "trajectory_type": "teacher_forced_full_guide_answer_redacted",
                "messages": curated,
                "final_response": response,
                "correct": True,
                "metadata": {
                    "attempt": attempt,
                    "benchmark_eligible": False,
                    "blocking_final_review": True,
                    "controller_answer_label_leak": False,
                    "item_specific_system_prompt_content": False,
                    "invariant_system_prompt_sha256": system_audit[
                        "invariant_system_prompt_sha256"
                    ],
                    "scripted_guidance_role": "user",
                    "compiled_plan_steps": scripted_step_count,
                    "grader_mode": grade_mode,
                    "recovered_from_prior_attempt": True,
                },
            }
            training_path = item_dir / "training.jsonl"
            write_private_text(
                training_path,
                json.dumps(training_row, ensure_ascii=False, sort_keys=True) + "\n",
            )
            recovery = {
                "schema_version": "1.0",
                "recovered_at": utc_now_iso(),
                "attempt": attempt,
                "source_result": str(result_path),
                "source_events": str(event_path),
                "training_path": str(training_path),
                "quality": quality,
                "system_message_audit": system_audit,
            }
            recovery["grader_mode"] = grade_mode
            write_private_json(item_dir / "verified-answer-recovery.json", recovery)
            mark_item_completed(
                item_dir,
                oracle_record,
                result_status="verified_answer_recovery",
            )
            recovered.append(
                {
                    "row_index": index,
                    "attempt": attempt,
                    "training_path": str(training_path),
                }
            )
            break
    return {"recovered_count": len(recovered), "recovered": recovered}


@dataclass
class WorkerResources:
    worker_id: int
    star7_endpoint: str
    star2_endpoint: str
    config: AppConfig
    model_client: OpenAICompatibleClient
    search_provider: Any
    page_fetcher: PageFetcher
    external_model: AgentExternalModelBroker

    async def close(self) -> None:
        await self.external_model.close()
        await self.page_fetcher.close()
        await self.search_provider.close()
        await self.model_client.close()


def worker_resources(
    worker_id: int,
    base_config: AppConfig,
    *,
    star7_endpoint: str,
    star2_endpoint: str,
) -> WorkerResources:
    config = base_config.model_copy(deep=True)
    config.model.api_base = star7_endpoint
    config.model.model = "frontierrl/star-7"
    config.model.temperature = None
    config.model.max_output_tokens = 16384
    config.model.response_chain = False
    config.model.routing_backend_pool = []
    config.model.extra_body = {
        "top_p": 0.95,
        "parallel_tool_calls": False,
        "vllm_xargs": {"frontierrl_max_denoising_steps": 48},
    }
    config.search.provider = "brave"
    config.search.live_preflight = False
    config.agent.require_opened_citation_support = False
    config.agent.allow_unsupported_final_at_hard_budget = False
    config.agent.automatic_external_strategy_recovery = False
    config.agent.automatic_finalization_rescue_after_rejections = 0
    config.agent.automatic_finalization_rescue_after_seconds = 0
    config.agent.force_final_after_seconds = 0
    config.external_model.agent_api_base = star2_endpoint
    config.external_model.agent_model = "frontierrl/star-2"
    config.external_model.temperature = 0.7
    config.external_model.top_p = 0.95
    config.external_model.max_output_tokens = 16384
    config.external_model.agent_max_denoising_steps = 48
    config.external_model.agent_routing_backend_pool = []
    model_client = OpenAICompatibleClient(settings_from_model_config(config.model))
    search_provider = create_search_provider(config.search)
    page_fetcher = PageFetcher(config.browser)
    helper_agent_config = config.agent.model_copy(
        deep=True,
        update={
            "require_citations": False,
            "require_opened_citation_support": False,
        },
    )
    external_model = AgentExternalModelBroker(
        config.external_model,
        helper_agent_config,
        config.browser,
        search_provider,
        page_fetcher,
    )
    return WorkerResources(
        worker_id=worker_id,
        star7_endpoint=star7_endpoint,
        star2_endpoint=star2_endpoint,
        config=config,
        model_client=model_client,
        search_provider=search_provider,
        page_fetcher=page_fetcher,
        external_model=external_model,
    )


async def probe_endpoint(endpoint: str, model: str, base_config: AppConfig) -> None:
    config = base_config.model.model_copy(
        deep=True,
        update={"api_base": endpoint, "model": model, "temperature": None},
    )
    client = OpenAICompatibleClient(settings_from_model_config(config))
    try:
        data = await client.list_models()
    finally:
        await client.close()
    models = {str(value.get("id") or "") for value in data.get("data") or []}
    if model not in models:
        raise RuntimeError(f"{endpoint} does not advertise {model}: {sorted(models)}")


async def grade_response(
    *,
    question: str,
    answer: str,
    response: str,
    grader: Grader | None,
    grader_semaphore: asyncio.Semaphore,
) -> tuple[Any, str]:
    deterministic = grade_deterministic(response, answer)
    if deterministic.correct or grader is None:
        return deterministic, "deterministic"
    async with grader_semaphore:
        official = await grader.grade(question, answer, response)
    return official, "official_llm"


async def run_attempt(
    *,
    resources: WorkerResources,
    route_record: dict[str, Any],
    oracle_record: dict[str, Any],
    attempt: int,
    item_dir: Path,
    timeout_seconds: float,
    grader: Grader | None,
    grader_semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    item = oracle_record["item"]
    question = str(item["question_text"])
    reference = str(oracle_record["oracle"]["gold_answer"])
    scripted_steps, review_guidance = compile_guided_steps(
        route_record, oracle_record, attempt=attempt
    )
    leaks = controller_label_leaks(
        question=question,
        oracle_record=oracle_record,
        steps=scripted_steps,
        review_guidance=review_guidance,
    )
    if leaks:
        raise RuntimeError(f"Controller answer-label leak detected for aliases: {leaks!r}")
    initial_guidance = GENERIC_INITIAL_GUIDANCE
    attempt_prefix = item_dir / f"attempt-{attempt:02d}"
    events_path = attempt_prefix.with_name(attempt_prefix.name + "-events.jsonl")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("", encoding="utf-8")
    os.chmod(events_path, 0o600)
    events: list[dict[str, Any]] = []

    def event_sink(event: dict[str, Any]) -> None:
        row = {"timestamp": utc_now_iso(), **event}
        events.append(row)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if event.get("event") in {
            "scripted_guidance_step_completed",
            "blocking_guidance_adversary_completed",
            "trial_final",
            "trial_no_final",
            "model_error",
        }:
            print(
                json.dumps(
                    {
                        "worker": resources.worker_id,
                        "item_index": item["row_index"],
                        "attempt": attempt,
                        "event": event.get("event"),
                        "step": event.get("step"),
                        "phase": event.get("phase"),
                        "verdict": event.get("verdict"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    agent_config = resources.config.agent.model_copy(
        deep=True,
        update={
            "max_steps": max(80, len(scripted_steps) * 3 + 10),
            "max_search_calls": max(80, len(scripted_steps) * 7),
            "max_page_opens": max(100, len(scripted_steps) * 4),
        },
    )
    runner = AgentRunner(
        resources.config.model,
        agent_config,
        resources.config.browser,
        resources.search_provider,
        resources.page_fetcher,
        model_client=resources.model_client,
        external_model_config=resources.config.external_model,
        external_model_broker=resources.external_model,
        event_sink=event_sink,
    )
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_seconds):
            outcome = await runner.run(
                question,
                request_namespace=(
                    f"full-guide:{item['item_id']}:attempt-{attempt}:worker-{resources.worker_id}"
                ),
                initial_guidance=initial_guidance,
                review_guidance=review_guidance,
                scripted_guidance_steps=scripted_steps,
                blocking_guidance_adversary=True,
                guidance_adversary_interval_steps=0,
                guidance_adversary_max_checkpoints=0,
                scripted_final_block_fail_fast=False,
                scripted_guidance_role="user",
                scripted_step_max_attempts=6,
            )
    finally:
        await runner.close()

    system_message_audit = audit_system_messages(
        transcript=outcome.transcript,
        invariant_system_prompt=runner.system_prompt,
        events=events,
        expected_final_system_prompt=SCRIPTED_FINAL_SYSTEM_PROMPT,
    )
    if not system_message_audit["passed"]:
        raise RuntimeError(
            "Dynamic or item-specific content appeared in a model-visible system message"
        )

    grade = grade_deterministic(outcome.response_text, reference)
    grade_mode = "deterministic"
    grade_error = None
    if outcome.status == "completed" and not grade.correct and grader is not None:
        try:
            grade, grade_mode = await grade_response(
                question=question,
                answer=reference,
                response=outcome.response_text,
                grader=grader,
                grader_semaphore=grader_semaphore,
            )
        except Exception as exc:  # noqa: BLE001 - retain the private attempt for recovery
            grade_error = f"{type(exc).__name__}: {exc}"

    result = {
        "schema_version": "1.0",
        "experiment": "full-oracle-plan teacher-forced answer-redacted training",
        "benchmark_eligible": False,
        "item_id": item["item_id"],
        "row_index": item["row_index"],
        "attempt": attempt,
        "worker_id": resources.worker_id,
        "star7_endpoint": resources.star7_endpoint,
        "star2_endpoint": resources.star2_endpoint,
        "status": outcome.status,
        "correct": bool(grade.correct),
        "answer_response": outcome.response_text,
        "extracted_answer": grade.extracted_answer,
        "grader_mode": grade_mode,
        "grader_reasoning": grade.reasoning,
        "grader_error": grade_error,
        "grader_usage": asdict(grade.usage),
        "model_usage": asdict(outcome.usage),
        "steps": outcome.steps,
        "compiled_plan_steps": len(scripted_steps),
        "search_calls": outcome.search_calls,
        "page_opens": outcome.page_opens,
        "find_calls": outcome.find_calls,
        "external_model_calls": outcome.external_model_calls,
        "duration_seconds": time.perf_counter() - started,
        "errors": outcome.errors,
        "event_path": str(events_path),
        "system_message_audit": system_message_audit,
    }

    if result["correct"] and outcome.status == "completed":
        curated = curate_scripted_training_messages(
            events,
            initial_messages=outcome.transcript[:2],
            scripted_step_count=len(scripted_steps),
        )
        quality = training_message_quality(curated)
        result["training_quality"] = quality
        if quality["passed"]:
            training_row = {
                "schema_version": "1.0",
                "item_id": item["item_id"],
                "source_index": item["row_index"],
                "trajectory_type": "teacher_forced_full_guide_answer_redacted",
                "messages": curated,
                "final_response": outcome.response_text,
                "correct": True,
                "metadata": {
                    "attempt": attempt,
                    "benchmark_eligible": False,
                    "blocking_final_review": True,
                    "controller_answer_label_leak": False,
                    "item_specific_system_prompt_content": False,
                    "invariant_system_prompt_sha256": system_message_audit[
                        "invariant_system_prompt_sha256"
                    ],
                    "scripted_guidance_role": "user",
                    "compiled_plan_steps": len(scripted_steps),
                    "grader_mode": grade_mode,
                },
            }
            training_path = item_dir / "training.jsonl"
            write_private_text(
                training_path,
                json.dumps(training_row, ensure_ascii=False, sort_keys=True) + "\n",
            )
            result["training_path"] = str(training_path)
        else:
            result["correct"] = False
            result["status"] = "training_quality_rejected"
    write_private_json(
        attempt_prefix.with_name(attempt_prefix.name + "-result.json"),
        result,
    )
    return result


async def consolidate(output_dir: Path, records: list[dict[str, Any]]) -> tuple[Path, int]:
    rows: list[str] = []
    for record in records:
        path = completed_training_path(output_dir, record)
        if path.exists():
            rows.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line)
    destination = output_dir / "browsecomp-full-guided-training.jsonl"
    write_private_text(destination, "\n".join(rows) + ("\n" if rows else ""))
    return destination, len(rows)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)
    base_config = load_config(args.config)
    route_records, oracle_records = load_guide_records(args.guide_root.resolve())
    indices = parse_indices(args.indices, len(route_records))
    if args.limit is not None:
        indices = indices[: args.limit]
    if not indices:
        raise ValueError("No guide items were selected")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be at least 1")
    preflight = preflight_compiled_guides(
        route_records,
        oracle_records,
        indices=indices,
        max_attempts=args.max_attempts,
    )
    star7_endpoints = endpoint_list(args.star7_endpoints)
    star2_endpoints = endpoint_list(args.star2_endpoints)
    concurrency = min(args.concurrency, len(star7_endpoints), len(indices))
    if concurrency <= 0:
        raise ValueError("No selected work or Star-7 endpoint")

    for endpoint in star7_endpoints[:concurrency]:
        await probe_endpoint(endpoint, "frontierrl/star-7", base_config)
    for endpoint in sorted(set(star2_endpoints[index % len(star2_endpoints)] for index in range(concurrency))):
        await probe_endpoint(endpoint, "frontierrl/star-2", base_config)
    if not args.deterministic_only:
        await probe_endpoint(args.grader_endpoint, args.grader_model, base_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output_dir, 0o700)
    write_private_json(args.output_dir / "compiler-preflight.json", preflight)
    status_path = args.status_file or (args.output_dir / "status.json")
    started_at = utc_now_iso()
    started_clock = time.perf_counter()
    grader_config = base_config.grader.model_copy(
        deep=True,
        update={
            "api_base": args.grader_endpoint,
            "api_key": base_config.model.api_key,
            "allow_empty_api_key": base_config.model.allow_empty_api_key,
            "model": args.grader_model,
            "temperature": 0.7,
            "max_output_tokens": 16384,
            "timeout_seconds": 360,
            "extra_headers_json": {},
            "extra_body": {
                "top_p": 0.95,
            },
        },
    )
    grader = None if args.deterministic_only else Grader(grader_config)
    grader_semaphore = asyncio.Semaphore(max(1, min(8, concurrency)))
    quality_recovery = await recover_verified_rows(
        output_dir=args.output_dir,
        config=base_config,
        oracle_records=oracle_records,
        indices=indices,
        grader=grader,
        grader_semaphore=grader_semaphore,
    )
    write_private_json(args.output_dir / "quality-recovery-summary.json", quality_recovery)

    queue: asyncio.Queue[int | None] = asyncio.Queue()
    already_correct = {
        index
        for index in indices
        if completed_training_path(args.output_dir, oracle_records[index]).exists()
    }
    already_exhausted: set[int] = set()
    for index in indices:
        if index in already_correct:
            continue
        if (
            existing_attempt_number(item_directory(args.output_dir, oracle_records[index]))
            >= args.max_attempts
        ):
            already_exhausted.add(index)
        else:
            queue.put_nowait(index)

    state: dict[str, Any] = {
        "run_id": args.output_dir.name,
        "started_at": started_at,
        "selected_total": len(indices),
        "correct": set(already_correct),
        "failed": set(already_exhausted),
        "retry_pending": {},
        "in_progress": {},
        "attempts_completed": 0,
        "last_errors": {},
    }
    state_lock = asyncio.Lock()
    async def publish_status(*, done: bool = False) -> dict[str, Any]:
        async with state_lock:
            elapsed = max(0.001, time.perf_counter() - started_clock)
            finished = len(state["correct"]) + len(state["failed"])
            rate = finished / elapsed * 3600
            remaining = max(0, len(indices) - finished)
            payload = {
                "run_id": state["run_id"],
                "started_at": state["started_at"],
                "updated_at": utc_now_iso(),
                "output_dir": str(args.output_dir.resolve()),
                "combined_output": str(
                    (args.output_dir / "browsecomp-full-guided-training.jsonl").resolve()
                ),
                "target_total_questions": len(indices),
                "current_records": len(state["correct"]),
                "failed": len(state["failed"]),
                "finished_items": finished,
                "pending": queue.qsize(),
                "retry_pending": dict(state["retry_pending"]),
                "in_progress": dict(state["in_progress"]),
                "attempts_completed": state["attempts_completed"],
                "maximum_attempts_per_item": args.max_attempts,
                "items_per_hour": rate,
                "eta_seconds": remaining / rate * 3600 if rate > 0 else None,
                "done": done,
                "last_errors": dict(list(state["last_errors"].items())[-20:]),
                "compiler_preflight": preflight,
                "quality_recovery": quality_recovery,
                "notes": (
                    "Full 1,266-item private oracle-guided training capture; answers are withheld "
                    "from controller prompts and only correct, quality-gated traces are exported."
                ),
            }
            write_private_json(status_path, payload)
            write_private_json(Path("/tmp/browsecomp_full_guided_training_status.json"), payload)
            return payload

    async def process_item(resources: WorkerResources, index: int) -> None:
        oracle_record = oracle_records[index]
        route_record = route_records[index]
        item_dir = item_directory(args.output_dir, oracle_record)
        item_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(item_dir, 0o700)
        attempted_candidates = load_attempted_candidates(item_dir)
        attempt = existing_attempt_number(item_dir) + 1
        if attempt > args.max_attempts:
            raise RuntimeError(
                f"Item {index} was queued after exhausting {args.max_attempts} attempts"
            )
        try:
            result = await run_attempt(
                resources=resources,
                route_record=route_record,
                oracle_record=oracle_record,
                attempt=attempt,
                item_dir=item_dir,
                timeout_seconds=args.item_timeout_seconds,
                grader=grader,
                grader_semaphore=grader_semaphore,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one item from the full batch
            result = {
                "item_id": oracle_record["item"]["item_id"],
                "row_index": index,
                "attempt": attempt,
                "status": "error",
                "correct": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_private_json(item_dir / f"attempt-{attempt:02d}-result.json", result)
        async with state_lock:
            state["attempts_completed"] += 1
        candidate = str(result.get("extracted_answer") or "").strip()
        if candidate and candidate not in attempted_candidates:
            attempted_candidates.append(candidate)

        correct = completed_training_path(args.output_dir, oracle_record).exists()
        retry_scheduled = not correct and attempt < args.max_attempts
        item_status = {
            "item_id": oracle_record["item"]["item_id"],
            "row_index": index,
            "correct": correct,
            "status": (
                "completed" if correct else "retry_pending" if retry_scheduled else "failed"
            ),
            "attempts": existing_attempt_number(item_dir),
            "next_attempt": attempt + 1 if retry_scheduled else None,
            "attempted_candidates": attempted_candidates,
            "last_result_status": result.get("status"),
            "last_error": result.get("error") or result.get("grader_error"),
            "updated_at": utc_now_iso(),
        }
        write_private_json(item_dir / "status.json", item_status)
        async with state_lock:
            state["in_progress"].pop(str(index), None)
            if correct:
                state["correct"].add(index)
                state["failed"].discard(index)
                state["retry_pending"].pop(str(index), None)
            elif retry_scheduled:
                state["failed"].discard(index)
                state["retry_pending"][str(index)] = attempt + 1
            else:
                state["failed"].add(index)
                state["retry_pending"].pop(str(index), None)
                state["last_errors"][str(index)] = str(
                    item_status.get("last_error") or item_status["last_result_status"]
                )
        if retry_scheduled:
            queue.put_nowait(index)
        status = await publish_status()
        print(
            json.dumps(
                {
                    "event": "attempt_finished" if retry_scheduled else "item_finished",
                    "worker": resources.worker_id,
                    "item_index": index,
                    "attempt": attempt,
                    "correct": correct,
                    "retry_scheduled": retry_scheduled,
                    "current_records": status["current_records"],
                    "finished_items": status["finished_items"],
                    "target": status["target_total_questions"],
                    "eta_seconds": status["eta_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    async def worker_loop(resources: WorkerResources) -> None:
        try:
            while True:
                index = await queue.get()
                if index is None:
                    queue.task_done()
                    return
                async with state_lock:
                    state["retry_pending"].pop(str(index), None)
                    state["in_progress"][str(index)] = {
                        "worker": resources.worker_id,
                        "star7": resources.star7_endpoint,
                        "star2": resources.star2_endpoint,
                        "started_at": utc_now_iso(),
                    }
                await publish_status()
                try:
                    await process_item(resources, index)
                finally:
                    queue.task_done()
        finally:
            await resources.close()

    await publish_status()
    resources = [
        worker_resources(
            worker_id,
            base_config,
            star7_endpoint=star7_endpoints[worker_id],
            star2_endpoint=star2_endpoints[worker_id % len(star2_endpoints)],
        )
        for worker_id in range(concurrency)
    ]
    try:
        worker_tasks = [asyncio.create_task(worker_loop(value)) for value in resources]
        await queue.join()
        for _ in resources:
            queue.put_nowait(None)
        await asyncio.gather(*worker_tasks)
    finally:
        if grader is not None:
            await grader.close()

    combined_path, combined_rows = await consolidate(
        args.output_dir, [oracle_records[index] for index in indices]
    )
    final_status = await publish_status(done=True)
    final_status["combined_output"] = str(combined_path.resolve())
    final_status["combined_records"] = combined_rows
    write_private_json(status_path, final_status)
    write_private_json(Path("/tmp/browsecomp_full_guided_training_status.json"), final_status)
    return final_status


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
