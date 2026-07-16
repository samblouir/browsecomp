#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from browsecomp250.agent import AgentRunner
from browsecomp250.agent_external import AgentExternalModelBroker
from browsecomp250.browser import PageFetcher
from browsecomp250.config import load_config
from browsecomp250.dataset import load_items
from browsecomp250.grading import Grader
from browsecomp250.llm import OpenAICompatibleClient, settings_from_model_config
from browsecomp250.search import create_search_provider
from browsecomp250.util import atomic_write_json, atomic_write_text, sha256_file, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one private, source-guided BrowseComp diagnostic with blocking reviews."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/star-headline.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--private-guide", type=Path, required=True)
    parser.add_argument("--steps", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--training-output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--review-interval", type=int, default=4)
    parser.add_argument("--max-checkpoint-reviews", type=int, default=3)
    parser.add_argument("--star7-api-base", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--star2-api-base", default="http://127.0.0.1:8003/v1")
    parser.add_argument(
        "--omit-unsupported-star7-temperature",
        action="store_true",
        help="Omit temperature when calling a diffusion worker directly; the router does this normally.",
    )
    return parser.parse_args()


def private_reference_label(guide_path: Path) -> str:
    for line in guide_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Gold answer: "):
            label = line.removeprefix("Gold answer: ").strip()
            if label:
                return label
    raise RuntimeError(f"No sealed gold-answer line found in {guide_path}")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)
    config = load_config(args.config)
    config.model.api_base = args.star7_api_base
    config.model.model = "frontierrl/star-7"
    config.model.temperature = None if args.omit_unsupported_star7_temperature else 0.3
    config.model.max_output_tokens = 16384
    config.model.extra_body = {
        **config.model.extra_body,
        "top_p": 0.95,
        "parallel_tool_calls": False,
        "vllm_xargs": {"frontierrl_max_denoising_steps": 48},
    }
    config.search.provider = "brave"
    config.search.live_preflight = True
    config.agent.max_steps = min(config.agent.max_steps, 40)
    config.agent.max_search_calls = min(config.agent.max_search_calls, 40)
    config.agent.force_final_after_seconds = (
        min(config.agent.force_final_after_seconds, 900)
        if config.agent.force_final_after_seconds > 0
        else 900
    )
    config.external_model.agent_api_base = args.star2_api_base
    config.external_model.agent_model = "frontierrl/star-2"
    config.external_model.temperature = 0.7
    config.external_model.top_p = 0.95
    config.external_model.max_output_tokens = 16384
    config.external_model.agent_max_denoising_steps = 48

    items = {item.subset_rank: item for item in load_items(config.dataset)}
    if args.rank not in items:
        raise RuntimeError(f"Subset rank {args.rank} does not exist")
    item = items[args.rank]
    guidance = args.plan.read_text(encoding="utf-8").strip()
    scripted_steps: list[dict[str, Any]] | None = None
    if args.steps is not None:
        loaded_steps = json.loads(args.steps.read_text(encoding="utf-8"))
        if not isinstance(loaded_steps, list) or not all(
            isinstance(step, dict) for step in loaded_steps
        ):
            raise RuntimeError("Scripted guide steps must be a JSON array of objects")
        scripted_steps = loaded_steps
    hidden_label = private_reference_label(args.private_guide)
    if hidden_label.casefold() in guidance.casefold():
        raise RuntimeError("Refusing to run: the research plan contains the hidden answer label")
    if scripted_steps and hidden_label.casefold() in json.dumps(scripted_steps).casefold():
        raise RuntimeError("Refusing to run: scripted guide steps contain the hidden answer label")
    if scripted_steps:
        # The blocking final reviewer and official grader remain active. The generic opened-page
        # lexical gate is redundant here and can reject a correct attributed name when a reader
        # fallback omits the one naming sentence from otherwise valid source text.
        config.agent.require_opened_citation_support = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.events.parent.mkdir(parents=True, exist_ok=True)
    args.events.touch(mode=0o600, exist_ok=True)
    os.chmod(args.events, 0o600)

    def event_sink(event: dict[str, Any]) -> None:
        row = {"timestamp": utc_now_iso(), **event}
        with args.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            "[guided-item]"
            f" step={event.get('step', '-')}"
            f" event={event.get('event', '-')}"
            f" action={event.get('action', '-')}"
            f" phase={event.get('phase', '-')}"
            f" verdict={event.get('verdict', '-')}",
            flush=True,
        )

    model_client = OpenAICompatibleClient(settings_from_model_config(config.model))
    search_provider = create_search_provider(config.search)
    page_fetcher = PageFetcher(config.browser)
    external_model = AgentExternalModelBroker(
        config.external_model,
        config.agent,
        config.browser,
        search_provider,
        page_fetcher,
    )
    grader = Grader(config.grader)
    outcome = None
    grade = None
    try:
        await search_provider.probe_live()
        runner = AgentRunner(
            config.model,
            config.agent,
            config.browser,
            search_provider,
            page_fetcher,
            model_client=model_client,
            external_model_config=config.external_model,
            external_model_broker=external_model,
            event_sink=event_sink,
        )
        async with asyncio.timeout(args.timeout_seconds):
            outcome = await runner.run(
                item.question,
                request_namespace=f"guide-plan:{item.item_id}:{args.output.stem}",
                initial_guidance=guidance,
                scripted_guidance_steps=scripted_steps,
                blocking_guidance_adversary=True,
                guidance_adversary_interval_steps=args.review_interval,
                guidance_adversary_max_checkpoints=args.max_checkpoint_reviews,
            )
        grade = await grader.grade(item.question, item.answer, outcome.response_text)
        result = {
            "schema_version": "1.0",
            "experiment": "full-guide source-guided, answer-label-redacted",
            "benchmark_eligibility": {
                "eligible": False,
                "reason": "Item-specific mapped sources and clue routes were supplied.",
            },
            "item_id": item.item_id,
            "subset_rank": item.subset_rank,
            "source_index": item.source_index,
            "plan_path": str(args.plan.resolve()),
            "plan_sha256": sha256_file(args.plan),
            "plan_chars": len(guidance),
            "plan_label_leak": False,
            "status": outcome.status,
            "correct": grade.correct,
            "answer_response": outcome.response_text,
            "extracted_answer": grade.extracted_answer,
            "grader_reasoning": grade.reasoning,
            "grader_usage": asdict(grade.usage),
            "model_usage": asdict(outcome.usage),
            "steps": outcome.steps,
            "search_calls": outcome.search_calls,
            "page_opens": outcome.page_opens,
            "find_calls": outcome.find_calls,
            "retrieved_chars": outcome.retrieved_chars,
            "external_model_calls": outcome.external_model_calls,
            "duration_seconds": outcome.duration_seconds,
            "citations": outcome.citations,
            "confidence": outcome.confidence,
            "errors": outcome.errors,
            "transcript": outcome.transcript,
            "finished_at": utc_now_iso(),
        }
    finally:
        await grader.close()
        await external_model.close()
        await page_fetcher.close()
        await search_provider.close()
        await model_client.close()

    atomic_write_json(args.output, result)
    os.chmod(args.output, 0o600)
    if args.training_output is not None and result["status"] == "completed" and result["correct"]:
        training_row = {
            "schema_version": "1.0",
            "item_id": result["item_id"],
            "source_index": result["source_index"],
            "trajectory_type": "teacher_forced_source_guided_answer_label_redacted",
            "messages": result["transcript"],
            "final_response": result["answer_response"],
            "correct": True,
            "metadata": {
                "plan_sha256": result["plan_sha256"],
                "benchmark_eligible": False,
                "blocking_final_review": True,
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
                "steps": result["steps"],
                "search_calls": result["search_calls"],
                "page_opens": result["page_opens"],
                "external_model_calls": result["external_model_calls"],
                "duration_seconds": result["duration_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
