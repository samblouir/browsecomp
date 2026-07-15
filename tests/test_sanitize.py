from __future__ import annotations

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
from browsecomp250.report.sanitize import LeakDetectedError, sanitize_run, scan_public_tree


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        run=RunConfig(output_dir=tmp_path / "runs"),
        dataset=DatasetConfig(
            cache_dir=tmp_path / "cache",
            subset_indices_path=Path(__file__).parents[1] / "data" / "subset_indices.json",
        ),
        model=ModelConfig(api_base="https://model.test/v1", model="star"),
        search=SearchConfig(provider="searxng", cache_path=tmp_path / "search.sqlite3"),
        browser=BrowserConfig(cache_path=tmp_path / "pages.sqlite3"),
        agent=AgentConfig(),
        grader=GraderConfig(mode="deterministic"),
        report=ReportConfig(),
    )


def test_sanitize_copies_only_public_artifacts(monkeypatch, tmp_path: Path) -> None:
    import browsecomp250.report.sanitize as module

    run_dir = tmp_path / "run"
    public = run_dir / "public"
    private = run_dir / "private"
    public.mkdir(parents=True)
    private.mkdir()
    (public / "summary.json").write_text('{"accuracy": 1.0}\n')
    (private / "secret.txt").write_text("private")
    (run_dir / "run.lock.json").write_text(json.dumps({"config": {"model": {"api_key": "secret"}}}))
    monkeypatch.setattr(
        module,
        "iter_plaintext_for_leak_scan",
        lambda config: [("x", "unique benchmark question phrase", "reference-answer")],
    )

    destination = tmp_path / "release"
    sanitize_run(run_dir, destination, _config(tmp_path))
    assert (destination / "public" / "summary.json").exists()
    assert not (destination / "private").exists()
    lock = json.loads((destination / "run.lock.json").read_text())
    assert lock["config"]["model"]["api_key"] == "<redacted>"
    assert (destination / "SHA256SUMS.json").exists()


def test_public_leak_scan_rejects_plaintext_question(monkeypatch, tmp_path: Path) -> None:
    import browsecomp250.report.sanitize as module

    run_dir = tmp_path / "run"
    public = run_dir / "public"
    public.mkdir(parents=True)
    question = "unique benchmark question phrase long enough"
    (public / "report.txt").write_text(question)
    monkeypatch.setattr(
        module,
        "iter_plaintext_for_leak_scan",
        lambda config: [("x", question, "reference-answer")],
    )
    findings = scan_public_tree(run_dir, _config(tmp_path))
    assert any("plaintext question" in finding for finding in findings)
    with pytest.raises(LeakDetectedError):
        sanitize_run(run_dir, tmp_path / "release", _config(tmp_path))
