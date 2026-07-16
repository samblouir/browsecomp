from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any


def load_campaign_records(runs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    unscored: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob("*/private/trials.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                unscored.append(
                    {
                        "status": "corrupt_record",
                        "correct": None,
                        "error": f"JSONDecodeError: {exc}",
                        "campaign_source": str(path),
                        "campaign_source_line": line_number,
                        "raw_line_chars": len(line),
                        "raw_line_sha256": sha256(line.encode("utf-8")).hexdigest(),
                    }
                )
                continue
            if not isinstance(record, dict):
                unscored.append(
                    {
                        "status": "invalid_record",
                        "correct": None,
                        "error": f"JSON record must be an object, got {type(record).__name__}",
                        "campaign_source": str(path),
                        "campaign_source_line": line_number,
                        "raw_line_chars": len(line),
                        "raw_line_sha256": sha256(line.encode("utf-8")).hexdigest(),
                    }
                )
                continue
            record["campaign_source"] = str(path)
            record["campaign_source_line"] = line_number
            if (
                record.get("status") == "completed"
                and isinstance(record.get("correct"), bool)
                and isinstance(record.get("subset_rank"), int)
                and record.get("finished_at")
            ):
                scored.append(record)
            else:
                unscored.append(record)
    return scored, unscored


def build_campaign_ledgers(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_rank[int(record["subset_rank"])].append(record)

    first_pass: list[dict[str, Any]] = []
    repaired: list[dict[str, Any]] = []
    for rank in sorted(by_rank):
        ordered = sorted(
            by_rank[rank],
            key=lambda row: (
                str(row["finished_at"]),
                str(row.get("run_id") or ""),
                int(row.get("attempt") or 0),
                str(row.get("campaign_source") or ""),
                int(row.get("campaign_source_line") or 0),
            ),
        )
        first_pass.append(ordered[0])
        repaired.append(ordered[-1])
    return first_pass, repaired


def ledger_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(record["correct"] is True for record in records)
    total = len(records)
    return {
        "scored": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total else None,
        "subset_ranks": [int(record["subset_rank"]) for record in records],
    }


def write_campaign_ledgers(runs_root: Path, output_dir: Path) -> dict[str, Any]:
    scored, unscored = load_campaign_records(runs_root)
    first_pass, repaired = build_campaign_ledgers(scored)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (
        ("first_pass.jsonl", first_pass),
        ("repaired.jsonl", repaired),
        ("unscored.jsonl", unscored),
    ):
        payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        (output_dir / name).write_text(payload)

    summary = {
        "first_pass": ledger_summary(first_pass),
        "repaired": ledger_summary(repaired),
        "unscored_records": len(unscored),
        "scored_record_versions": len(scored),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
