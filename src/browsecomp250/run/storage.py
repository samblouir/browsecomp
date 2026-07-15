from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet

from ..types import TrialRecord
from ..util import atomic_write_json, atomic_write_text, utc_now_iso


class RunStorage:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.private_dir = run_dir / "private"
        self.public_dir = run_dir / "public"
        self.transcripts_dir = self.private_dir / "transcripts"
        self.records_path = self.private_dir / "trials.jsonl"
        self.events_path = self.private_dir / "events.jsonl"
        self._lock = threading.RLock()
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.public_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        key = os.environ.get("BC250_ARTIFACT_FERNET_KEY", "")
        self._fernet = Fernet(key.encode("ascii")) if key else None

    @property
    def lock_path(self) -> Path:
        return self.run_dir / "run.lock.json"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "status.json"

    def write_lock(self, value: dict[str, Any]) -> None:
        if self.lock_path.exists():
            existing = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if existing.get("replay_hash") != value.get("replay_hash"):
                raise FileExistsError(
                    f"Run directory already contains a different lock: {self.run_dir}"
                )
            return
        atomic_write_json(self.lock_path, value)

    def update_status(self, **values: Any) -> None:
        with self._lock:
            current: dict[str, Any] = {}
            if self.status_path.exists():
                current = json.loads(self.status_path.read_text(encoding="utf-8"))
            current.update(values)
            current["updated_at"] = utc_now_iso()
            atomic_write_json(self.status_path, current)

    def append_event(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self._lock, self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def append_record(self, record: TrialRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
        with self._lock, self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def load_records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.records_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed record at {self.records_path}:{line_number}: {exc}"
                ) from exc
        return records

    def completed_keys(self) -> set[tuple[str, int]]:
        return {(str(row["item_id"]), int(row["attempt"])) for row in self.load_records()}

    def write_transcript(self, item_id: str, attempt: int, payload: dict[str, Any]) -> Path:
        name = f"{item_id}--attempt-{attempt}.json"
        path = self.transcripts_dir / name
        raw = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        if self._fernet:
            path = path.with_suffix(".json.fernet")
            path.write_bytes(self._fernet.encrypt(raw))
        else:
            path.write_bytes(raw)
        return path

    def write_private_readme(self) -> None:
        text = """# Private artifacts\n\nThis directory may contain BrowseComp questions, model trajectories, predicted answers, and grader outputs. Do not publish it. Use `bc250 sanitize` to generate publication-safe artifacts.\n"""
        atomic_write_text(self.private_dir / "README.md", text)
