from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

DEFAULT_TOTAL_QUESTIONS = 250


def _is_graded_completion(record: dict[str, Any]) -> bool:
    return record.get("status") == "completed" and isinstance(record.get("correct"), bool)


def _has_terminal_identity(record: dict[str, Any]) -> bool:
    return isinstance(record.get("subset_rank"), int) and bool(record.get("finished_at"))


def load_campaign_records(
    runs_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load terminal attempts without silently dropping failures from accuracy.

    The returned collections are overlapping views by design:

    * terminal records have a rank and completion timestamp, including failures;
    * unscored records are anything that is not a graded completion; and
    * quarantined records cannot be assigned to a terminal rank safely.
    """

    terminal: list[dict[str, Any]] = []
    unscored: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob("*/private/trials.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                record = {
                    "status": "corrupt_record",
                    "correct": None,
                    "error": f"JSONDecodeError: {exc}",
                    "campaign_source": str(path),
                    "campaign_source_line": line_number,
                    "raw_line_chars": len(line),
                    "raw_line_sha256": sha256(line.encode("utf-8")).hexdigest(),
                }
                unscored.append(record)
                quarantined.append(record)
                continue
            if not isinstance(record, dict):
                record = {
                    "status": "invalid_record",
                    "correct": None,
                    "error": f"JSON record must be an object, got {type(record).__name__}",
                    "campaign_source": str(path),
                    "campaign_source_line": line_number,
                    "raw_line_chars": len(line),
                    "raw_line_sha256": sha256(line.encode("utf-8")).hexdigest(),
                }
                unscored.append(record)
                quarantined.append(record)
                continue

            record["campaign_source"] = str(path)
            record["campaign_source_line"] = line_number
            if _has_terminal_identity(record):
                terminal.append(record)
            else:
                quarantined.append(record)
            if not _is_graded_completion(record):
                unscored.append(record)
    return terminal, unscored, quarantined


def _record_order(row: dict[str, Any]) -> tuple[str, str, int, str, int]:
    return (
        str(row["finished_at"]),
        str(row.get("run_id") or ""),
        int(row.get("attempt") or 0),
        str(row.get("campaign_source") or ""),
        int(row.get("campaign_source_line") or 0),
    )


def build_campaign_ledgers(
    records: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_rank[int(record["subset_rank"])].append(record)

    ledgers: dict[str, list[dict[str, Any]]] = {
        "strict_first_terminal": [],
        "latest_terminal": [],
        "best_observed_after_repair": [],
        "graded_first_completion": [],
        "graded_latest_completion": [],
    }
    for rank in sorted(by_rank):
        ordered = sorted(by_rank[rank], key=_record_order)
        graded = [row for row in ordered if _is_graded_completion(row)]
        correct = [row for row in ordered if row.get("correct") is True]

        ledgers["strict_first_terminal"].append(ordered[0])
        ledgers["latest_terminal"].append(ordered[-1])
        ledgers["best_observed_after_repair"].append(correct[-1] if correct else ordered[-1])
        if graded:
            ledgers["graded_first_completion"].append(graded[0])
            ledgers["graded_latest_completion"].append(graded[-1])
    return ledgers


def ledger_summary(
    records: list[dict[str, Any]],
    *,
    total_questions: int = DEFAULT_TOTAL_QUESTIONS,
) -> dict[str, Any]:
    correct = sum(record.get("correct") is True for record in records)
    attempted = len(records)
    return {
        "attempted": attempted,
        "correct": correct,
        "incorrect": attempted - correct,
        "accuracy_among_attempted": correct / attempted if attempted else None,
        "total_questions": total_questions,
        "questions_without_terminal_record": max(total_questions - attempted, 0),
        "full_set_correct_coverage": correct / total_questions if total_questions else None,
        "subset_ranks": [int(record["subset_rank"]) for record in records],
    }


def write_campaign_ledgers(
    runs_root: Path,
    output_dir: Path,
    *,
    total_questions: int = DEFAULT_TOTAL_QUESTIONS,
) -> dict[str, Any]:
    terminal, unscored, quarantined = load_campaign_records(runs_root)
    ledgers = build_campaign_ledgers(terminal)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_records = {
        "first_pass.jsonl": ledgers["strict_first_terminal"],
        "repaired.jsonl": ledgers["best_observed_after_repair"],
        "latest_terminal.jsonl": ledgers["latest_terminal"],
        "graded_first_completion.jsonl": ledgers["graded_first_completion"],
        "graded_latest_completion.jsonl": ledgers["graded_latest_completion"],
        "unscored.jsonl": unscored,
        "quarantined.jsonl": quarantined,
    }
    for name, records in output_records.items():
        payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        (output_dir / name).write_text(payload)

    summaries = {
        name: ledger_summary(records, total_questions=total_questions)
        for name, records in ledgers.items()
    }
    summary = {
        "metric_contract": {
            "headline_development_metric": "strict_first_terminal",
            "strict_first_terminal": (
                "The earliest terminal record per distinct question; timeout, error, and "
                "no-final outcomes count as incorrect."
            ),
            "best_observed_after_repair": (
                "A development ceiling: the latest correct record when one exists, "
                "otherwise the latest terminal failure. It is not pass@1."
            ),
            "graded_completion_metrics": (
                "Diagnostic-only views that exclude terminal failures from their denominator."
            ),
        },
        **summaries,
        # Compatibility aliases now point at honestly labelled metrics.
        "first_pass": summaries["strict_first_terminal"],
        "repaired": summaries["best_observed_after_repair"],
        "terminal_record_versions": len(terminal),
        "graded_record_versions": sum(_is_graded_completion(row) for row in terminal),
        "unscored_record_versions": len(unscored),
        "quarantined_record_versions": len(quarantined),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
