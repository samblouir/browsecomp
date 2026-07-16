from __future__ import annotations

import json
from pathlib import Path

from browsecomp250.campaign import write_campaign_ledgers


def _write_run(root: Path, name: str, records: list[dict[str, object]]) -> None:
    path = root / name / "private" / "trials.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _record(rank: int, finished_at: str, correct: bool) -> dict[str, object]:
    return {
        "run_id": finished_at,
        "subset_rank": rank,
        "status": "completed",
        "finished_at": finished_at,
        "correct": correct,
    }


def test_campaign_keeps_first_pass_and_latest_repair_separate(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_run(runs, "first", [_record(0, "2026-07-15T10:00:00+00:00", False)])
    _write_run(runs, "repair", [_record(0, "2026-07-15T11:00:00+00:00", True)])

    output = tmp_path / "ledger"
    summary = write_campaign_ledgers(runs, output)

    assert summary["first_pass"]["correct"] == 0
    assert summary["repaired"]["correct"] == 1
    assert json.loads((output / "first_pass.jsonl").read_text())["correct"] is False
    assert json.loads((output / "repaired.jsonl").read_text())["correct"] is True


def test_unscored_error_does_not_claim_first_pass_slot(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    error = {
        "run_id": "timed-out",
        "subset_rank": 2,
        "status": "error",
        "finished_at": "2026-07-15T09:00:00+00:00",
        "correct": None,
    }
    _write_run(runs, "error", [error])
    _write_run(runs, "valid", [_record(2, "2026-07-15T10:00:00+00:00", True)])

    summary = write_campaign_ledgers(runs, tmp_path / "ledger")

    assert summary["first_pass"]["scored"] == 1
    assert summary["first_pass"]["correct"] == 1
    assert summary["unscored_records"] == 1


def test_campaign_quarantines_corrupt_jsonl_without_losing_valid_rows(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    path = runs / "interrupted" / "private" / "trials.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"status":"timeout","correct":null,\n'
        + json.dumps(_record(4, "2026-07-15T12:00:00+00:00", True))
        + "\n"
    )

    output = tmp_path / "ledger"
    summary = write_campaign_ledgers(runs, output)

    assert summary["repaired"]["scored"] == 1
    assert summary["repaired"]["correct"] == 1
    assert summary["unscored_records"] == 1
    quarantined = json.loads((output / "unscored.jsonl").read_text())
    assert quarantined["status"] == "corrupt_record"
    assert quarantined["campaign_source_line"] == 1
    assert quarantined["raw_line_chars"] > 0
    assert len(quarantined["raw_line_sha256"]) == 64
    assert "JSONDecodeError" in quarantined["error"]
