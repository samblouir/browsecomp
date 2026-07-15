from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def _parse_bool(value: str | None) -> bool:
    return str(value).strip().casefold() == "true"


def _parse_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_public_trials(run_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    path = run_dir / "public" / "trials.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing public trials CSV: {path}")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["source_index"]), int(row["attempt"]))
            if key in rows:
                raise ValueError(f"Duplicate trial key {key} in {path}")
            rows[key] = {
                **row,
                "correct_bool": _parse_bool(row.get("correct")),
                "duration_float": _parse_float(row.get("duration_seconds")),
                "cost_float": _parse_float(row.get("total_cost_usd")),
            }
    return rows


def mcnemar_exact_pvalue(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value under Binomial(n, 0.5)."""
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    lower = min(a_only, b_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def paired_bootstrap_interval(
    pairs: list[tuple[bool, bool]],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    if not pairs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    differences: list[float] = []
    count = len(pairs)
    for _ in range(samples):
        draw = [pairs[rng.randrange(count)] for _ in range(count)]
        differences.append(
            sum((1 if left else 0) - (1 if right else 0) for left, right in draw) / count
        )
    alpha = (1 - confidence) / 2
    return (_percentile(differences, alpha), _percentile(differences, 1 - alpha))


def _load_lock(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.lock.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run lock: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_mismatches(left_lock: dict[str, Any], right_lock: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    if left_lock.get("subset_indices_sha256") != right_lock.get("subset_indices_sha256"):
        mismatches.append("subset_indices_sha256")
    if (left_lock.get("dataset") or {}).get("sha256") != (right_lock.get("dataset") or {}).get(
        "sha256"
    ):
        mismatches.append("dataset.sha256")

    left_config = left_lock.get("config") or {}
    right_config = right_lock.get("config") or {}
    # The model and run identity are expected to differ. All external scaffold
    # sections should match for a bare-model comparison.
    for section in ("search", "browser", "agent", "grader", "report"):
        if left_config.get(section) != right_config.get(section):
            mismatches.append(f"config.{section}")
    left_run = left_config.get("run") or {}
    right_run = right_config.get("run") or {}
    for field in ("attempts", "concurrency", "task_timeout_seconds", "shuffle", "seed"):
        if left_run.get(field) != right_run.get(field):
            mismatches.append(f"config.run.{field}")
    return mismatches


def paired_compare(
    left_dir: Path,
    right_dir: Path,
    *,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    left = load_public_trials(left_dir)
    right = load_public_trials(right_dir)
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError("Runs have no common (source_index, attempt) trial keys")

    pairs = [(left[key]["correct_bool"], right[key]["correct_bool"]) for key in common]
    both_correct = sum(a and b for a, b in pairs)
    left_only = sum(a and not b for a, b in pairs)
    right_only = sum(not a and b for a, b in pairs)
    both_wrong = sum(not a and not b for a, b in pairs)
    left_accuracy = sum(a for a, _ in pairs) / len(pairs)
    right_accuracy = sum(b for _, b in pairs) / len(pairs)
    interval = paired_bootstrap_interval(pairs, samples=bootstrap_samples, confidence=confidence)

    duration_differences = [
        left[key]["duration_float"] - right[key]["duration_float"]
        for key in common
        if left[key]["duration_float"] is not None and right[key]["duration_float"] is not None
    ]
    cost_differences = [
        left[key]["cost_float"] - right[key]["cost_float"]
        for key in common
        if left[key]["cost_float"] is not None and right[key]["cost_float"] is not None
    ]

    left_lock = _load_lock(left_dir)
    right_lock = _load_lock(right_dir)
    mismatches = protocol_mismatches(left_lock, right_lock)
    return {
        "schema_version": "1.0",
        "left_run": str(left_dir),
        "right_run": str(right_dir),
        "n_common_trials": len(common),
        "n_left_only_trials": len(set(left) - set(right)),
        "n_right_only_trials": len(set(right) - set(left)),
        "left_accuracy": left_accuracy,
        "right_accuracy": right_accuracy,
        "accuracy_difference_left_minus_right": left_accuracy - right_accuracy,
        "paired_bootstrap_interval": list(interval),
        "confidence_level": confidence,
        "mcnemar": {
            "both_correct": both_correct,
            "left_only_correct": left_only,
            "right_only_correct": right_only,
            "both_wrong": both_wrong,
            "discordant": left_only + right_only,
            "exact_two_sided_pvalue": mcnemar_exact_pvalue(left_only, right_only),
        },
        "mean_duration_difference_seconds_left_minus_right": (
            statistics.fmean(duration_differences) if duration_differences else None
        ),
        "mean_cost_difference_usd_left_minus_right": (
            statistics.fmean(cost_differences) if cost_differences else None
        ),
        "protocol_mismatches": mismatches,
        "protocol_compatible": not mismatches,
    }
