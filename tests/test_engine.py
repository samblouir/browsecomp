from __future__ import annotations

import csv
import json
from pathlib import Path

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
    assert public_summary["cost_breakdown_usd"]["model"] == pytest.approx(0.002)
    assert public_summary["cost_breakdown_usd"]["grader"] == pytest.approx(0.0002)


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
