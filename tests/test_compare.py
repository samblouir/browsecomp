from __future__ import annotations

import csv
import json
from pathlib import Path

from browsecomp250.report.compare import mcnemar_exact_pvalue, paired_compare


def _write_run(path: Path, outcomes: list[bool], *, model: str = "m") -> None:
    public = path / "public"
    public.mkdir(parents=True)
    with (public / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "source_index",
            "attempt",
            "correct",
            "duration_seconds",
            "total_cost_usd",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, outcome in enumerate(outcomes):
            writer.writerow(
                {
                    "source_index": index,
                    "attempt": 1,
                    "correct": outcome,
                    "duration_seconds": index + 1,
                    "total_cost_usd": 0.1,
                }
            )
    lock = {
        "subset_indices_sha256": "same",
        "dataset": {"sha256": "same"},
        "config": {
            "model": {"model": model},
            "run": {
                "attempts": 1,
                "concurrency": 1,
                "task_timeout_seconds": 10,
                "shuffle": False,
                "seed": 0,
            },
            "search": {},
            "browser": {},
            "agent": {},
            "grader": {},
            "report": {},
        },
    }
    (path / "run.lock.json").write_text(json.dumps(lock), encoding="utf-8")


def test_mcnemar_and_paired_compare(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left, [True, True, False, False], model="left")
    _write_run(right, [True, False, True, False], model="right")
    result = paired_compare(left, right, bootstrap_samples=500)
    assert result["n_common_trials"] == 4
    assert result["mcnemar"]["left_only_correct"] == 1
    assert result["mcnemar"]["right_only_correct"] == 1
    assert result["mcnemar"]["exact_two_sided_pvalue"] == 1.0
    assert result["protocol_compatible"] is True
    assert mcnemar_exact_pvalue(0, 0) == 1.0
