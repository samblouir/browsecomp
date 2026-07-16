from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from browsecomp250.config import (
    AgentConfig,
    AppConfig,
    BrowserConfig,
    DatasetConfig,
    GraderConfig,
    ModelConfig,
    ReportConfig,
    RunConfig,
    SearchConfig,
)
from browsecomp250.crypto import encrypt
from browsecomp250.run.engine import BenchmarkEngine
from browsecomp250.types import AgentOutcome, GradeResult, Usage


class _CountOnlyCache:
    def count(self) -> int:
        return 0


class _FakeClosable:
    def __init__(self, *args, **kwargs):
        self.cache = _CountOnlyCache()

    async def close(self) -> None:
        return None


class _FakeAgentRunner:
    def __init__(self, *args, **kwargs):
        pass

    async def run(self, question: str, **kwargs) -> AgentOutcome:
        del kwargs
        return AgentOutcome(
            response_text="Explanation: fixture\nExact Answer: fixture\nConfidence: 90%",
            exact_answer="fixture",
            explanation="fixture",
            confidence=90,
            citations=["https://example.test/evidence"],
            status="completed",
            steps=2,
            search_calls=1,
            page_opens=1,
            find_calls=0,
            retrieved_chars=100,
            duration_seconds=0.1,
            usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
            transcript=[{"role": "user", "content": question}],
        )


class _FakeGrader(_FakeClosable):
    async def grade(self, question: str, reference: str, response: str) -> GradeResult:
        return GradeResult(
            correct=True,
            extracted_answer="fixture",
            reasoning="fixture",
            grader_mode="fixture",
            usage=Usage(input_tokens=3, output_tokens=2, cost_usd=0.0001),
        )


class _FailingGrader(_FakeClosable):
    async def grade(self, question: str, reference: str, response: str) -> GradeResult:
        del question, reference, response
        raise RuntimeError("grader transport unavailable")


def _write_synthetic_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["problem", "answer", "canary"])
        writer.writeheader()
        for index in range(1266):
            canary = f"fixture-{index}"
            writer.writerow(
                {
                    "problem": encrypt(f"Question {index}", canary),
                    "answer": encrypt(f"Answer {index}", canary),
                    "canary": canary,
                }
            )


@pytest.mark.asyncio
async def test_engine_end_to_end_with_synthetic_dataset(monkeypatch, tmp_path: Path) -> None:
    import browsecomp250.run.engine as engine_module

    cache_dir = tmp_path / "cache"
    _write_synthetic_dataset(cache_dir / "browse_comp_test_set.csv")
    repository_root = Path(__file__).parents[1]
    config = AppConfig(
        run=RunConfig(
            name="fixture-run",
            output_dir=tmp_path / "runs",
            concurrency=2,
            shuffle=False,
            write_private_transcripts=True,
        ),
        dataset=DatasetConfig(
            source_url="https://example.test/encrypted.csv",
            cache_dir=cache_dir,
            subset_indices_path=repository_root / "data" / "subset_indices.json",
        ),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(provider="searxng", cache_path=tmp_path / "search.sqlite3"),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3", block_private_networks=False),
        agent=AgentConfig(),
        grader=GraderConfig(mode="deterministic"),
        report=ReportConfig(bootstrap_samples=200),
    )

    monkeypatch.setattr(engine_module, "AgentRunner", _FakeAgentRunner)
    monkeypatch.setattr(engine_module, "OpenAICompatibleClient", _FakeClosable)
    monkeypatch.setattr(engine_module, "create_search_provider", lambda config: _FakeClosable())
    monkeypatch.setattr(engine_module, "PageFetcher", _FakeClosable)
    monkeypatch.setattr(engine_module, "Grader", _FakeGrader)

    summary = await BenchmarkEngine(config).run(limit=2)
    run_dir = tmp_path / "runs" / "fixture-run"
    assert summary["n_scored"] == 2
    assert summary["accuracy"] == 1.0
    assert (run_dir / "run.lock.json").exists()
    assert (run_dir / "cache.manifest.json").exists()
    assert (run_dir / "public" / "summary.json").exists()
    assert (run_dir / "public" / "trials.csv").exists()
    assert len((run_dir / "private" / "trials.jsonl").read_text().splitlines()) == 2
    public_summary = json.loads((run_dir / "public" / "summary.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())
    assert status["assigned"] == 2
    assert status["finished"] == 2
    assert status["correct"] == 2
    assert status["incorrect"] == 0
    assert status["graded_incorrect"] == 0
    assert status["strict_incorrect"] == 0
    assert status["failed"] == 0
    assert status["execution_failed"] == 0
    assert status["pending"] == 0
    assert status["accuracy_among_finished"] == 1.0
    assert status["accuracy_among_graded"] == 1.0
    assert status["correct_fraction_of_assigned"] == 1.0
    assert public_summary["cost_breakdown_usd"]["model"] == pytest.approx(0.002)
    assert public_summary["cost_breakdown_usd"]["grader"] == pytest.approx(0.0002)


@pytest.mark.asyncio
async def test_engine_reports_grader_transport_failure_separately_from_wrong_answer(
    monkeypatch, tmp_path: Path
) -> None:
    import browsecomp250.run.engine as engine_module

    cache_dir = tmp_path / "cache"
    _write_synthetic_dataset(cache_dir / "browse_comp_test_set.csv")
    repository_root = Path(__file__).parents[1]
    config = AppConfig(
        run=RunConfig(name="fixture-grader-failure", output_dir=tmp_path / "runs"),
        dataset=DatasetConfig(
            source_url="https://example.test/encrypted.csv",
            cache_dir=cache_dir,
            subset_indices_path=repository_root / "data" / "subset_indices.json",
        ),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(provider="searxng", cache_path=tmp_path / "search.sqlite3"),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3"),
        agent=AgentConfig(),
        grader=GraderConfig(mode="deterministic"),
        report=ReportConfig(bootstrap_samples=100),
    )

    monkeypatch.setattr(engine_module, "AgentRunner", _FakeAgentRunner)
    monkeypatch.setattr(engine_module, "OpenAICompatibleClient", _FakeClosable)
    monkeypatch.setattr(engine_module, "create_search_provider", lambda config: _FakeClosable())
    monkeypatch.setattr(engine_module, "PageFetcher", _FakeClosable)
    monkeypatch.setattr(engine_module, "Grader", _FailingGrader)

    await BenchmarkEngine(config).run(limit=1)
    status = json.loads((tmp_path / "runs" / "fixture-grader-failure" / "status.json").read_text())
    assert status["finished"] == 1
    assert status["correct"] == 0
    assert status["incorrect"] == 0
    assert status["graded_incorrect"] == 0
    assert status["strict_incorrect"] == 1
    assert status["failed"] == 1
    assert status["accuracy_among_finished"] == 0
    assert status["accuracy_among_graded"] is None


@pytest.mark.asyncio
async def test_engine_runs_a_locked_heldout_range(monkeypatch, tmp_path: Path) -> None:
    import browsecomp250.run.engine as engine_module

    cache_dir = tmp_path / "cache"
    _write_synthetic_dataset(cache_dir / "browse_comp_test_set.csv")
    repository_root = Path(__file__).parents[1]
    config = AppConfig(
        run=RunConfig(
            name="fixture-heldout-run",
            output_dir=tmp_path / "runs",
            concurrency=2,
            shuffle=False,
        ),
        dataset=DatasetConfig(
            source_url="https://example.test/encrypted.csv",
            cache_dir=cache_dir,
            subset_indices_path=repository_root / "data" / "subset_indices.json",
        ),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(provider="searxng", cache_path=tmp_path / "search.sqlite3"),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3", block_private_networks=False),
        agent=AgentConfig(),
        grader=GraderConfig(mode="deterministic"),
        report=ReportConfig(bootstrap_samples=200),
    )

    monkeypatch.setattr(engine_module, "AgentRunner", _FakeAgentRunner)
    monkeypatch.setattr(engine_module, "OpenAICompatibleClient", _FakeClosable)
    monkeypatch.setattr(engine_module, "create_search_provider", lambda config: _FakeClosable())
    monkeypatch.setattr(engine_module, "PageFetcher", _FakeClosable)
    monkeypatch.setattr(engine_module, "Grader", _FakeGrader)

    summary = await BenchmarkEngine(config).run(start=5, limit=2)
    run_dir = tmp_path / "runs" / "fixture-heldout-run"
    records = [
        json.loads(line) for line in (run_dir / "private" / "trials.jsonl").read_text().splitlines()
    ]
    lock = json.loads((run_dir / "run.lock.json").read_text())
    assert summary["n_scored"] == 2
    assert [record["subset_rank"] for record in records] == [5, 6]
    assert lock["selection"] == {"start": 5, "limit": 2}


@pytest.mark.asyncio
async def test_engine_runs_exact_noncontiguous_ranks(monkeypatch, tmp_path: Path) -> None:
    import browsecomp250.run.engine as engine_module

    cache_dir = tmp_path / "cache"
    _write_synthetic_dataset(cache_dir / "browse_comp_test_set.csv")
    repository_root = Path(__file__).parents[1]
    config = AppConfig(
        run=RunConfig(
            name="fixture-exact-ranks-run",
            output_dir=tmp_path / "runs",
            concurrency=3,
            shuffle=False,
        ),
        dataset=DatasetConfig(
            source_url="https://example.test/encrypted.csv",
            cache_dir=cache_dir,
            subset_indices_path=repository_root / "data" / "subset_indices.json",
        ),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(provider="searxng", cache_path=tmp_path / "search.sqlite3"),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3", block_private_networks=False),
        agent=AgentConfig(),
        grader=GraderConfig(mode="deterministic"),
        report=ReportConfig(bootstrap_samples=200),
    )

    monkeypatch.setattr(engine_module, "AgentRunner", _FakeAgentRunner)
    monkeypatch.setattr(engine_module, "OpenAICompatibleClient", _FakeClosable)
    monkeypatch.setattr(engine_module, "create_search_provider", lambda config: _FakeClosable())
    monkeypatch.setattr(engine_module, "PageFetcher", _FakeClosable)
    monkeypatch.setattr(engine_module, "Grader", _FakeGrader)

    summary = await BenchmarkEngine(config).run(ranks=[24, 2, 10])
    run_dir = tmp_path / "runs" / "fixture-exact-ranks-run"
    records = [
        json.loads(line) for line in (run_dir / "private" / "trials.jsonl").read_text().splitlines()
    ]
    lock = json.loads((run_dir / "run.lock.json").read_text())
    assert summary["n_scored"] == 3
    assert [record["subset_rank"] for record in records] == [2, 10, 24]
    assert lock["selection"] == {"ranks": [2, 10, 24]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ranks": []}, "at least one"),
        ({"ranks": [2, 2]}, "duplicates"),
        ({"ranks": [-1]}, "between 0 and 249"),
        ({"ranks": [250]}, "between 0 and 249"),
        ({"ranks": [2], "start": 1}, "cannot be combined"),
        ({"ranks": [2], "limit": 1}, "cannot be combined"),
    ],
)
async def test_engine_rejects_invalid_exact_rank_selection(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    config = AppConfig(
        run=RunConfig(name="fixture-invalid-ranks", output_dir=tmp_path / "runs"),
        dataset=DatasetConfig(),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(provider="searxng", cache_path=tmp_path / "search.sqlite3"),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3", block_private_networks=False),
        agent=AgentConfig(),
        grader=GraderConfig(mode="deterministic"),
        report=ReportConfig(),
    )

    with pytest.raises(ValueError, match=message):
        await BenchmarkEngine(config).run(**kwargs)  # type: ignore[arg-type]
    assert not (tmp_path / "runs" / "fixture-invalid-ranks" / "run.lock.json").exists()


@pytest.mark.asyncio
async def test_engine_enforces_per_cohort_concurrency(monkeypatch, tmp_path: Path) -> None:
    import browsecomp250.run.engine as engine_module

    cache_dir = tmp_path / "cache"
    _write_synthetic_dataset(cache_dir / "browse_comp_test_set.csv")
    repository_root = Path(__file__).parents[1]
    current_by_shard = [0, 0]
    maximum_by_shard = [0, 0]
    current_total = 0
    maximum_total = 0

    class CohortTrackingRunner(_FakeAgentRunner):
        async def run(self, question: str, **kwargs) -> AgentOutcome:
            nonlocal current_total, maximum_total
            namespace = str(kwargs["request_namespace"])
            rank = int(namespace.split(":bc250-", 1)[1].split("-", 1)[0])
            shard = rank % 2
            current_by_shard[shard] += 1
            current_total += 1
            maximum_by_shard[shard] = max(maximum_by_shard[shard], current_by_shard[shard])
            maximum_total = max(maximum_total, current_total)
            try:
                await asyncio.sleep(0.02)
                return await super().run(question, **kwargs)
            finally:
                current_by_shard[shard] -= 1
                current_total -= 1

    config = AppConfig(
        run=RunConfig(
            name="fixture-cohort-run",
            output_dir=tmp_path / "runs",
            seed=0,
            concurrency=2,
            routing_cohort_size=2,
            routing_max_concurrency_per_cohort=1,
            shuffle=True,
        ),
        dataset=DatasetConfig(
            source_url="https://example.test/encrypted.csv",
            cache_dir=cache_dir,
            subset_indices_path=repository_root / "data" / "subset_indices.json",
        ),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(provider="searxng", cache_path=tmp_path / "search.sqlite3"),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3", block_private_networks=False),
        agent=AgentConfig(),
        grader=GraderConfig(mode="deterministic"),
        report=ReportConfig(bootstrap_samples=200),
    )

    monkeypatch.setattr(engine_module, "AgentRunner", CohortTrackingRunner)
    monkeypatch.setattr(engine_module, "OpenAICompatibleClient", _FakeClosable)
    monkeypatch.setattr(engine_module, "create_search_provider", lambda config: _FakeClosable())
    monkeypatch.setattr(engine_module, "PageFetcher", _FakeClosable)
    monkeypatch.setattr(engine_module, "Grader", _FakeGrader)

    summary = await BenchmarkEngine(config).run(limit=4)
    assert summary["n_scored"] == 4
    assert maximum_by_shard == [1, 1]
    assert maximum_total == 2


def test_run_config_rejects_incomplete_or_oversubscribed_cohort_limits() -> None:
    with pytest.raises(ValueError, match="must both be set or zero"):
        RunConfig(routing_cohort_size=11)
    with pytest.raises(ValueError, match="exceeds the configured routing cohort capacity"):
        RunConfig(
            concurrency=45,
            routing_cohort_size=11,
            routing_max_concurrency_per_cohort=4,
        )


@pytest.mark.asyncio
async def test_engine_fails_before_writing_lock_when_disk_space_is_too_low(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import browsecomp250.run.engine as engine_module

    config = AppConfig(
        run=RunConfig(name="fixture-low-disk", output_dir=tmp_path / "runs"),
        dataset=DatasetConfig(),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(provider="searxng", cache_path=tmp_path / "search.sqlite3"),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3", block_private_networks=False),
        agent=AgentConfig(),
        grader=GraderConfig(mode="deterministic"),
        report=ReportConfig(),
    )
    monkeypatch.setattr(
        engine_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10, used=9, free=1),
    )

    with pytest.raises(RuntimeError, match="Insufficient free disk"):
        await BenchmarkEngine(config).run(limit=1)
    assert not (tmp_path / "runs" / "fixture-low-disk" / "run.lock.json").exists()


@pytest.mark.asyncio
async def test_engine_rejects_placeholder_grader_key_before_writing_lock(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        run=RunConfig(name="fixture-placeholder-key", output_dir=tmp_path / "runs"),
        dataset=DatasetConfig(),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(provider="searxng", cache_path=tmp_path / "search.sqlite3"),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3", block_private_networks=False),
        agent=AgentConfig(),
        grader=GraderConfig(mode="official_llm", api_key="replace-me"),
        report=ReportConfig(),
    )

    with pytest.raises(RuntimeError, match="grader API key is a placeholder"):
        await BenchmarkEngine(config).run(limit=1)
    assert not (tmp_path / "runs" / "fixture-placeholder-key" / "run.lock.json").exists()


@pytest.mark.asyncio
async def test_engine_rejects_openrouter_key_for_openai_grader_before_lock(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        run=RunConfig(name="fixture-mismatched-grader-key", output_dir=tmp_path / "runs"),
        dataset=DatasetConfig(),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(provider="searxng", cache_path=tmp_path / "search.sqlite3"),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3"),
        agent=AgentConfig(),
        grader=GraderConfig(
            mode="official_llm",
            api_base="https://api.openai.com/v1",
            api_key="sk-or-v1-not-an-openai-key",
        ),
        report=ReportConfig(),
    )

    with pytest.raises(RuntimeError, match="OpenRouter key but grader endpoint is api.openai.com"):
        await BenchmarkEngine(config).run(limit=1)
    assert not (tmp_path / "runs" / "fixture-mismatched-grader-key" / "run.lock.json").exists()


@pytest.mark.asyncio
async def test_engine_rejects_placeholder_search_key_before_writing_lock(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        run=RunConfig(name="fixture-placeholder-search", output_dir=tmp_path / "runs"),
        dataset=DatasetConfig(),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(
            provider="brave",
            brave_api_key="replace-me",
            cache_path=tmp_path / "search.sqlite3",
        ),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3", block_private_networks=False),
        agent=AgentConfig(),
        grader=GraderConfig(mode="deterministic"),
        report=ReportConfig(),
    )

    with pytest.raises(RuntimeError, match="brave search API key is a placeholder"):
        await BenchmarkEngine(config).run(limit=1)
    assert not (tmp_path / "runs" / "fixture-placeholder-search" / "run.lock.json").exists()


@pytest.mark.asyncio
async def test_engine_rejects_placeholder_openrouter_search_key_before_lock(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        run=RunConfig(name="fixture-placeholder-openrouter", output_dir=tmp_path / "runs"),
        dataset=DatasetConfig(),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(
            provider="openrouter_exa",
            openrouter_api_key="change-me",
            cache_path=tmp_path / "search.sqlite3",
        ),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3", block_private_networks=False),
        agent=AgentConfig(),
        grader=GraderConfig(mode="deterministic"),
        report=ReportConfig(),
    )

    with pytest.raises(RuntimeError, match="openrouter_exa search API key is a placeholder"):
        await BenchmarkEngine(config).run(limit=1)
    assert not (tmp_path / "runs" / "fixture-placeholder-openrouter" / "run.lock.json").exists()


@pytest.mark.asyncio
async def test_engine_rejects_failed_live_search_probe_before_writing_lock(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import browsecomp250.run.engine as engine_module

    class FailingLiveProbe(_FakeClosable):
        closed = False
        probe_calls = 0

        async def probe_live(self) -> None:
            self.probe_calls += 1
            raise RuntimeError("401 invalid search credential")

        async def close(self) -> None:
            self.closed = True

    provider = FailingLiveProbe()
    config = AppConfig(
        run=RunConfig(name="fixture-live-search-failure", output_dir=tmp_path / "runs"),
        dataset=DatasetConfig(),
        model=ModelConfig(api_base="https://model.test/v1", api_key="key", model="star"),
        search=SearchConfig(
            provider="searxng",
            live_preflight=True,
            cache_path=tmp_path / "search.sqlite3",
        ),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3"),
        agent=AgentConfig(),
        grader=GraderConfig(mode="deterministic"),
        report=ReportConfig(),
    )
    monkeypatch.setattr(engine_module, "create_search_provider", lambda _config: provider)

    with pytest.raises(RuntimeError, match="Search live preflight failed before launch"):
        await BenchmarkEngine(config).run(limit=1)

    assert provider.probe_calls == 1
    assert provider.closed is True
    assert not (tmp_path / "runs" / "fixture-live-search-failure" / "run.lock.json").exists()
