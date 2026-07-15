from __future__ import annotations

import csv
import html
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

from ..util import atomic_write_json, atomic_write_text, utc_now_iso


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_accuracy(
    values: list[int], *, samples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    estimates = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    estimates.sort()
    alpha = (1 - confidence) / 2
    low = estimates[max(0, int(alpha * samples))]
    high = estimates[min(samples - 1, int((1 - alpha) * samples) - 1)]
    return low, high


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (float(row["confidence"]) / 100, 1.0 if row.get("correct") else 0.0)
        for row in records
        if row.get("confidence") is not None and row.get("correct") is not None
    ]
    if not pairs:
        return {"brier": None, "ece_10": None, "n": 0}
    brier = sum((confidence - outcome) ** 2 for confidence, outcome in pairs) / len(pairs)
    ece = 0.0
    for bucket in range(10):
        lo, hi = bucket / 10, (bucket + 1) / 10
        values = [pair for pair in pairs if lo <= pair[0] < hi or (bucket == 9 and pair[0] == 1)]
        if not values:
            continue
        avg_conf = sum(pair[0] for pair in values) / len(values)
        avg_acc = sum(pair[1] for pair in values) / len(values)
        ece += len(values) / len(pairs) * abs(avg_conf - avg_acc)
    return {"brier": brier, "ece_10": ece, "n": len(pairs)}


def aggregate_records(
    records: list[dict[str, Any]], *, confidence: float = 0.95, bootstrap_samples: int = 10_000
) -> dict[str, Any]:
    # Every attempted trial belongs in the headline denominator. A timeout,
    # agent exception, missing grader response, or malformed final answer is not
    # silently dropped; it contributes a zero. `n_graded` remains available as
    # a diagnostic for distinguishing wrong answers from infrastructure failures.
    attempted = list(records)
    graded = [row for row in records if row.get("correct") is not None]
    successes = sum(row.get("correct") is True for row in attempted)
    total = len(attempted)
    wilson_low, wilson_high = wilson_interval(successes, total, confidence)
    bootstrap_low, bootstrap_high = bootstrap_accuracy(
        [1 if row.get("correct") is True else 0 for row in attempted],
        samples=bootstrap_samples,
        confidence=confidence,
    )
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_item[str(row["item_id"])].append(row)
    max_attempts = max((int(row["attempt"]) for row in records), default=0)
    pass_at_k = {
        str(k): (
            sum(
                any(item.get("correct") for item in attempts if int(item["attempt"]) <= k)
                for attempts in by_item.values()
            )
            / len(by_item)
            if by_item
            else 0.0
        )
        for k in range(1, max_attempts + 1)
    }
    metrics = [row.get("metrics") or {} for row in records]
    durations = [float(item.get("duration_seconds", 0)) for item in metrics]
    costs = [float(item.get("total_cost_usd", 0)) for item in metrics]
    search_calls = [float(item.get("search_calls", 0)) for item in metrics]
    page_opens = [float(item.get("page_opens", 0)) for item in metrics]
    input_tokens = sum(int(item.get("input_tokens", 0)) for item in metrics)
    output_tokens = sum(int(item.get("output_tokens", 0)) for item in metrics)
    cached_tokens = sum(int(item.get("cached_tokens", 0)) for item in metrics)
    model_cost = sum(float(item.get("model_cost_usd", 0)) for item in metrics)
    grader_cost = sum(float(item.get("grader_cost_usd", 0)) for item in metrics)
    steps = [float(item.get("steps", 0)) for item in metrics]
    find_calls = [float(item.get("find_calls", 0)) for item in metrics]
    retrieved_chars = [float(item.get("retrieved_chars", 0)) for item in metrics]
    total_cost = sum(costs)
    status_counts = dict(
        sorted(Counter(str(row.get("status", "unknown")) for row in records).items())
    )
    citation_count = sum(bool(row.get("citations")) for row in records)
    zero_search_count = sum(
        float((row.get("metrics") or {}).get("search_calls", 0)) == 0 for row in records
    )
    correct_zero_search_count = sum(
        row.get("correct") is True and float((row.get("metrics") or {}).get("search_calls", 0)) == 0
        for row in records
    )
    grader_parse_errors = sum(
        bool((row.get("grading") or {}).get("parse_error")) for row in records
    )
    return {
        "schema_version": "1.0",
        "generated_at": utc_now_iso(),
        "n_records": len(records),
        "n_scored": total,
        "n_graded": len(graded),
        "n_ungraded": total - len(graded),
        "n_items": len(by_item),
        "n_correct": successes,
        "accuracy": successes / total if total else 0.0,
        "confidence_level": confidence,
        "wilson_interval": [wilson_low, wilson_high],
        "bootstrap_interval": [bootstrap_low, bootstrap_high],
        "pass_at_k": pass_at_k,
        "n_errors": sum(row.get("status") not in {"completed", "empty_answer"} for row in records),
        "status_counts": status_counts,
        "grader_parse_errors": grader_parse_errors,
        "answer_rate": (
            sum(bool(row.get("extracted_answer")) for row in records) / len(records)
            if records
            else 0.0
        ),
        "citation_rate": citation_count / len(records) if records else 0.0,
        "zero_search_rate": zero_search_count / len(records) if records else 0.0,
        "correct_zero_search_count": correct_zero_search_count,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "cached": cached_tokens,
        },
        "cost_usd": total_cost,
        "cost_breakdown_usd": {"model": model_cost, "grader": grader_cost},
        "cost_per_correct_usd": total_cost / successes if successes else None,
        "duration_seconds": {
            "total": sum(durations),
            "mean": statistics.fmean(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "p90": _percentile(durations, 0.9),
            "p95": _percentile(durations, 0.95),
        },
        "search_calls": {
            "total": sum(search_calls),
            "mean": statistics.fmean(search_calls) if search_calls else None,
        },
        "page_opens": {
            "total": sum(page_opens),
            "mean": statistics.fmean(page_opens) if page_opens else None,
        },
        "find_calls": {
            "total": sum(find_calls),
            "mean": statistics.fmean(find_calls) if find_calls else None,
        },
        "steps": {
            "total": sum(steps),
            "mean": statistics.fmean(steps) if steps else None,
        },
        "retrieved_chars": {
            "total": sum(retrieved_chars),
            "mean": statistics.fmean(retrieved_chars) if retrieved_chars else None,
        },
        "calibration": _calibration(graded),
    }


def write_reports(
    run_dir: Path,
    records: list[dict[str, Any]],
    *,
    confidence: float,
    bootstrap_samples: int,
    write_csv: bool,
    write_html: bool,
) -> dict[str, Any]:
    public_dir = run_dir / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_records(records, confidence=confidence, bootstrap_samples=bootstrap_samples)
    atomic_write_json(public_dir / "summary.json", summary)
    percent = 100 * summary["accuracy"]
    low, high = (100 * x for x in summary["wilson_interval"])
    markdown = f"""# BrowseComp-250 results

- Accuracy: **{percent:.2f}%** ({summary["n_correct"]}/{summary["n_scored"]})
- {int(confidence * 100)}% Wilson interval: **{low:.2f}%–{high:.2f}%**
- Items: {summary["n_items"]}
- Trial records: {summary["n_records"]}
- Graded / ungraded: {summary["n_graded"]} / {summary["n_ungraded"]}
- Errors: {summary["n_errors"]} (all attempted errors count as incorrect)
- Citation compliance: {100 * summary["citation_rate"]:.2f}%
- Zero-search trials: {100 * summary["zero_search_rate"]:.2f}%
- Total input / output / cached tokens: {summary["tokens"]["input"]:,} / {summary["tokens"]["output"]:,} / {summary["tokens"]["cached"]:,}
- Total recorded cost: ${summary["cost_usd"]:.4f}

This report intentionally omits benchmark questions, reference answers, model explanations, predicted answers, citations, and URLs. BrowseComp-250 is a custom fixed subset, not an official split.
"""
    atomic_write_text(public_dir / "summary.md", markdown)

    public_rows = [
        {
            "item_id": row["item_id"],
            "subset_rank": row["subset_rank"],
            "source_index": row["source_index"],
            "attempt": row["attempt"],
            "model": row["model"],
            "status": row["status"],
            "correct": row.get("correct"),
            "confidence": row.get("confidence"),
            "duration_seconds": (row.get("metrics") or {}).get("duration_seconds"),
            "search_calls": (row.get("metrics") or {}).get("search_calls"),
            "page_opens": (row.get("metrics") or {}).get("page_opens"),
            "input_tokens": (row.get("metrics") or {}).get("input_tokens"),
            "output_tokens": (row.get("metrics") or {}).get("output_tokens"),
            "total_cost_usd": (row.get("metrics") or {}).get("total_cost_usd"),
            "citation_count": len(row.get("citations") or []),
            "grader_parse_error": bool((row.get("grading") or {}).get("parse_error")),
        }
        for row in records
    ]
    if write_csv:
        with (public_dir / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]) if public_rows else [])
            if public_rows:
                writer.writeheader()
                writer.writerows(public_rows)

    if write_html:
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in row) + "</tr>"
            for row in public_rows
        )
        headers = "".join(
            f"<th>{html.escape(key)}</th>" for key in (public_rows[0] if public_rows else [])
        )
        page = f"""<!doctype html><html><head><meta charset='utf-8'><title>BrowseComp-250</title>
<style>body{{font-family:system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%;font-size:.85rem}}th,td{{border:1px solid #ddd;padding:.35rem;text-align:left}}th{{position:sticky;top:0;background:#fff}}code{{white-space:pre-wrap}}</style></head><body>
<h1>BrowseComp-250</h1><p><strong>Accuracy:</strong> {percent:.2f}% ({summary["n_correct"]}/{summary["n_scored"]})</p>
<p>This publication-safe report omits questions, answers, explanations, and URLs.</p><table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table></body></html>"""
        atomic_write_text(public_dir / "report.html", page)
    return summary
