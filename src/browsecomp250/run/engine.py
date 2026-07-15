from __future__ import annotations

import asyncio
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..agent import AgentRunner
from ..agent_external import AgentExternalModelBroker
from ..browser import PageFetcher
from ..config import AppConfig
from ..constants import SUBSET_INDICES_SHA256, __version__
from ..dataset import (
    dataset_path,
    load_items,
    validate_dataset_file,
    write_dataset_manifest,
)
from ..external import ExternalModelBroker
from ..grading import Grader
from ..llm import OpenAICompatibleClient, settings_from_model_config
from ..report import write_reports
from ..search import create_search_provider
from ..types import AgentOutcome, GradeResult, TrialRecord, Usage
from ..util import (
    atomic_write_json,
    canonical_sha256,
    environment_metadata,
    git_metadata,
    is_placeholder_secret,
    sha256_bytes,
    sqlite_family_state,
    utc_now_iso,
)
from .storage import RunStorage

_MIN_RUN_FREE_BYTES = 2 * 1024**3


class BenchmarkEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.run_dir = config.run.output_dir / config.run.name
        self.storage = RunStorage(self.run_dir)

    def _secret_fingerprint(self, value: str) -> str | None:
        return sha256_bytes(value.encode("utf-8"))[:16] if value else None

    def _validate_runtime_credentials(self) -> None:
        problems: list[str] = []
        if not self.config.model.api_key and not self.config.model.allow_empty_api_key:
            problems.append("model API key is empty")

        search_key = self.config.search.selected_api_key()
        if self.config.search.provider != "searxng":
            if not search_key and self.config.search.provider not in {"google_chrome"}:
                problems.append(f"{self.config.search.provider} search API key is empty")
            elif search_key and is_placeholder_secret(search_key):
                problems.append(f"{self.config.search.provider} search API key is a placeholder")

        if self.config.grader.mode in {"official_llm", "both"}:
            if not self.config.grader.api_key and not self.config.grader.allow_empty_api_key:
                problems.append("grader API key is empty")
            elif self.config.grader.api_key and is_placeholder_secret(self.config.grader.api_key):
                problems.append("grader API key is a placeholder")

        if self.config.external_model.enabled and self.config.external_model.mode == "agent":
            agent_key = self.config.external_model.agent_api_key
            if not agent_key and not self.config.external_model.agent_allow_empty_api_key:
                problems.append("external Star agent API key is empty")
            elif agent_key and is_placeholder_secret(agent_key):
                problems.append("external Star agent API key is a placeholder")

        if problems:
            raise RuntimeError(
                "Benchmark credential preflight failed before launch: " + "; ".join(problems)
            )

    def _build_lock(self, *, start: int = 0, limit: int | None = None) -> dict[str, Any]:
        dataset_metadata = validate_dataset_file(
            dataset_path(self.config.dataset), self.config.dataset
        )
        public_config = self.config.public_dict()
        replay_material = {
            "config": public_config,
            "dataset_sha256": dataset_metadata["sha256"],
            "subset_indices_sha256": SUBSET_INDICES_SHA256,
            "model_key_fingerprint": self._secret_fingerprint(self.config.model.api_key),
            "grader_key_fingerprint": self._secret_fingerprint(self.config.grader.api_key),
            "external_model_admin_token_fingerprint": self._secret_fingerprint(
                self.config.external_model.admin_token
            ),
            "external_agent_api_key_fingerprint": self._secret_fingerprint(
                self.config.external_model.agent_api_key
            ),
            "search_key_fingerprint": self._secret_fingerprint(
                self.config.search.selected_api_key()
            ),
            "cache_state_at_start": {
                "search": sqlite_family_state(self.config.search.cache_path),
                "pages": sqlite_family_state(self.config.browser.cache_path),
            },
        }
        resume_material = {
            "config": public_config,
            "dataset_sha256": dataset_metadata["sha256"],
            "subset_indices_sha256": SUBSET_INDICES_SHA256,
            "secret_fingerprints": {
                "model_api_key": replay_material["model_key_fingerprint"],
                "grader_api_key": replay_material["grader_key_fingerprint"],
                "search_api_key": replay_material["search_key_fingerprint"],
                "external_model_admin_token": replay_material[
                    "external_model_admin_token_fingerprint"
                ],
                "external_agent_api_key": replay_material["external_agent_api_key_fingerprint"],
            },
        }
        selection = {"start": start, "limit": limit}
        if start > 0:
            # Older run locks predate ranged execution. Preserve their start=0
            # resume compatibility while locking every nonzero held-out range.
            replay_material["selection"] = selection
            resume_material["selection"] = selection
        return {
            "schema_version": "1.0",
            "created_at": utc_now_iso(),
            "runner_version": __version__,
            "benchmark": "BrowseComp-250",
            "benchmark_official": False,
            "source_benchmark": "BrowseComp",
            "dataset": dataset_metadata,
            "subset_indices_sha256": SUBSET_INDICES_SHA256,
            "config": public_config,
            "secret_fingerprints": {
                "model_api_key": replay_material["model_key_fingerprint"],
                "grader_api_key": replay_material["grader_key_fingerprint"],
                "search_api_key": replay_material["search_key_fingerprint"],
                "external_model_admin_token": replay_material[
                    "external_model_admin_token_fingerprint"
                ],
                "external_agent_api_key": replay_material["external_agent_api_key_fingerprint"],
            },
            "environment": environment_metadata(),
            "git": git_metadata(Path.cwd()),
            "cache_state_at_start": replay_material["cache_state_at_start"],
            "selection": selection,
            "replay_hash": canonical_sha256(replay_material),
            "resume_hash": canonical_sha256(resume_material),
        }

    async def run(self, *, start: int = 0, limit: int | None = None) -> dict[str, Any]:
        if start < 0:
            raise ValueError("start must be non-negative")
        if start >= self.config.dataset.subset_size:
            raise ValueError(
                f"start must be smaller than the subset size ({self.config.dataset.subset_size})"
            )
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        self.config.run.output_dir.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(self.config.run.output_dir).free
        if free_bytes < _MIN_RUN_FREE_BYTES:
            raise RuntimeError(
                "Insufficient free disk for a durable benchmark run: "
                f"{free_bytes / 1024**3:.2f} GiB available; "
                f"at least {_MIN_RUN_FREE_BYTES / 1024**3:.0f} GiB required"
            )
        self._validate_runtime_credentials()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.storage.write_private_readme()
        write_dataset_manifest(self.config.dataset)
        self.storage.write_lock(self._build_lock(start=start, limit=limit))
        items = load_items(self.config.dataset)
        items = items[start:]
        if limit is not None:
            items = items[:limit]

        work = [
            (item, attempt) for attempt in range(1, self.config.run.attempts + 1) for item in items
        ]
        if self.config.run.shuffle:
            random.Random(self.config.run.seed).shuffle(work)
        completed = self.storage.completed_keys() if self.config.run.resume else set()
        work = [
            (item, attempt) for item, attempt in work if (item.item_id, attempt) not in completed
        ]

        self.storage.update_status(
            state="running",
            total_planned=len(items) * self.config.run.attempts,
            already_completed=len(completed),
            completed=len(completed),
            failed=0,
            remaining=len(work),
            active_trials={},
            started_at=utc_now_iso(),
        )

        model_client = OpenAICompatibleClient(settings_from_model_config(self.config.model))
        search_provider = create_search_provider(self.config.search)
        page_fetcher = PageFetcher(self.config.browser)
        if self.config.external_model.mode == "agent":
            external_model_broker = AgentExternalModelBroker(
                self.config.external_model,
                self.config.agent,
                self.config.browser,
                search_provider,
                page_fetcher,
            )
        else:
            external_model_broker = ExternalModelBroker(self.config.external_model)
        grader = Grader(self.config.grader)
        semaphore = asyncio.Semaphore(self.config.run.concurrency)
        failures: list[str] = []
        active_trials: dict[str, dict[str, Any]] = {}
        progress = {"completed": len(completed), "failed": 0}

        def publish_status(*, last_event: dict[str, Any] | None = None) -> None:
            self.storage.update_status(
                completed=progress["completed"],
                failed=progress["failed"],
                remaining=max(
                    0,
                    len(items) * self.config.run.attempts - progress["completed"],
                ),
                active_trials=dict(active_trials),
                last_event=last_event,
            )

        async def execute(item: Any, attempt: int) -> None:
            async with semaphore:
                trial_key = f"{item.item_id}/attempt-{attempt}"

                def event_sink(event: dict[str, Any]) -> None:
                    event_row = {
                        "schema_version": "1.0",
                        "timestamp": utc_now_iso(),
                        "run_id": self.config.run.name,
                        "item_id": item.item_id,
                        "subset_rank": item.subset_rank,
                        "attempt": attempt,
                        **event,
                    }
                    self.storage.append_event(event_row)
                    status_event = {
                        key: value
                        for key, value in event_row.items()
                        if key
                        not in {
                            "assistant_content",
                            "assistant_reasoning",
                            "payload",
                            "result",
                            "response_metadata",
                            "tool_calls",
                        }
                    }
                    active_trials[trial_key] = {
                        "item_id": item.item_id,
                        "subset_rank": item.subset_rank,
                        "attempt": attempt,
                        "phase": event.get("event"),
                        "step": event.get("step"),
                        "action": event.get("action"),
                        "last_event_at": event_row["timestamp"],
                    }
                    print(
                        "[bc250]"
                        f" item={item.item_id} attempt={attempt}"
                        f" step={event.get('step', '-')} phase={event.get('event')}"
                        f" action={event.get('action', '-')}",
                        flush=True,
                    )
                    publish_status(last_event=status_event)

                event_sink({"event": "trial_scheduled"})
                try:
                    record = await self._run_one(
                        item,
                        attempt,
                        model_client=model_client,
                        search_provider=search_provider,
                        page_fetcher=page_fetcher,
                        external_model_broker=external_model_broker,
                        grader=grader,
                        event_sink=event_sink,
                    )
                    progress["completed"] += 1
                    if record.status not in {"completed"}:
                        progress["failed"] += 1
                        failures.append(f"{trial_key}: status={record.status}")
                except Exception as exc:  # noqa: BLE001
                    progress["failed"] += 1
                    failures.append(f"{trial_key}: {exc}")
                    if self.config.run.fail_fast:
                        raise
                finally:
                    active_trials.pop(trial_key, None)
                    publish_status(
                        last_event={
                            "timestamp": utc_now_iso(),
                            "event": "trial_finished",
                            "item_id": item.item_id,
                            "attempt": attempt,
                        }
                    )

        try:
            if self.config.run.fail_fast:
                async with asyncio.TaskGroup() as group:
                    for item, attempt in work:
                        group.create_task(execute(item, attempt))
            else:
                await asyncio.gather(*(execute(item, attempt) for item, attempt in work))
        finally:
            await grader.close()
            await page_fetcher.close()
            await external_model_broker.close()
            await search_provider.close()
            await model_client.close()

        cache_manifest = {
            "schema_version": "1.0",
            "created_at": utc_now_iso(),
            "search": {
                **sqlite_family_state(self.config.search.cache_path),
                "namespace_entries": search_provider.cache.count(),
                "mode": self.config.search.cache_mode,
            },
            "pages": {
                **sqlite_family_state(self.config.browser.cache_path),
                "namespace_entries": page_fetcher.cache.count(),
                "mode": self.config.browser.cache_mode,
            },
            "contains_cache_contents": False,
        }
        atomic_write_json(self.run_dir / "cache.manifest.json", cache_manifest)

        records = self.storage.load_records()
        summary = write_reports(
            self.run_dir,
            records,
            confidence=self.config.report.confidence_level,
            bootstrap_samples=self.config.report.bootstrap_samples,
            write_csv=self.config.report.write_csv,
            write_html=self.config.report.write_html,
        )
        self.storage.update_status(
            state="completed_with_errors" if failures else "completed",
            finished_at=utc_now_iso(),
            n_records=len(records),
            completed=progress["completed"],
            failed=progress["failed"],
            remaining=0,
            active_trials={},
            failures=failures,
            summary=summary,
        )
        return summary

    async def _run_one(
        self,
        item: Any,
        attempt: int,
        *,
        model_client: OpenAICompatibleClient,
        search_provider: Any,
        page_fetcher: PageFetcher,
        external_model_broker: ExternalModelBroker | AgentExternalModelBroker,
        grader: Grader,
        event_sink: Any | None = None,
    ) -> TrialRecord:
        started_at = utc_now_iso()
        outcome: AgentOutcome | None = None
        grade: GradeResult | None = None
        error: str | None = None
        status = "error"
        try:
            runner = AgentRunner(
                self.config.model,
                self.config.agent,
                self.config.browser,
                search_provider,
                page_fetcher,
                model_client=model_client,
                external_model_config=self.config.external_model,
                external_model_broker=external_model_broker,
                event_sink=event_sink,
            )
            outcome = await asyncio.wait_for(
                runner.run(
                    item.question,
                    request_namespace=(f"{self.config.run.name}:{item.item_id}:attempt-{attempt}"),
                ),
                timeout=self.config.run.task_timeout_seconds,
            )
            status = outcome.status
            grade = await grader.grade(item.question, item.answer, outcome.response_text)
        except TimeoutError:
            status = "timeout"
            error = f"Task exceeded {self.config.run.task_timeout_seconds} seconds"
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            if self.config.run.fail_fast:
                raise
        finished_at = utc_now_iso()

        if outcome is None:
            outcome = AgentOutcome(
                response_text="",
                exact_answer=None,
                explanation="",
                confidence=None,
                citations=[],
                status=status,
                steps=0,
                search_calls=0,
                page_opens=0,
                find_calls=0,
                retrieved_chars=0,
                duration_seconds=0.0,
                usage=Usage(),
                transcript=[],
                errors=[error] if error else [],
            )

        grade_usage = grade.usage if grade else Usage()
        total_cost = outcome.usage.cost_usd + grade_usage.cost_usd
        record = TrialRecord(
            schema_version="1.0",
            run_id=self.config.run.name,
            item_id=item.item_id,
            subset_rank=item.subset_rank,
            source_index=item.source_index,
            attempt=attempt,
            model=self.config.model.model,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            answer_response=outcome.response_text,
            extracted_answer=(grade.extracted_answer if grade else outcome.exact_answer),
            explanation=outcome.explanation,
            confidence=outcome.confidence,
            citations=outcome.citations,
            correct=grade.correct if grade else None,
            grading=asdict(grade) if grade else None,
            metrics={
                "steps": outcome.steps,
                "search_calls": outcome.search_calls,
                "page_opens": outcome.page_opens,
                "find_calls": outcome.find_calls,
                "retrieved_chars": outcome.retrieved_chars,
                "external_model_calls": outcome.external_model_calls,
                "duration_seconds": outcome.duration_seconds,
                "input_tokens": outcome.usage.input_tokens,
                "output_tokens": outcome.usage.output_tokens,
                "cached_tokens": outcome.usage.cached_tokens,
                "model_cost_usd": outcome.usage.cost_usd,
                "grader_input_tokens": grade_usage.input_tokens,
                "grader_output_tokens": grade_usage.output_tokens,
                "grader_cost_usd": grade_usage.cost_usd,
                "total_cost_usd": total_cost,
            },
            error=error,
        )
        self.storage.append_record(record)
        if self.config.run.write_private_transcripts:
            self.storage.write_transcript(
                item.item_id,
                attempt,
                {
                    "schema_version": "1.0",
                    "warning": "PRIVATE BENCHMARK ARTIFACT — DO NOT PUBLISH",
                    "item": {
                        "item_id": item.item_id,
                        "subset_rank": item.subset_rank,
                        "source_index": item.source_index,
                        "encrypted_row_hash": item.encrypted_row_hash,
                        "question": item.question,
                        "reference_answer": item.answer,
                    },
                    "outcome": asdict(outcome),
                    "grade": asdict(grade) if grade else None,
                    "error": error,
                },
            )
        return record
