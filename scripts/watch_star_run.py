#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

TERMINAL_STATES = {"completed", "completed_with_errors"}


def parse_time(value: object) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_watcher_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch a BrowseComp-250 Star run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--stale-seconds", type=float, default=600.0)
    parser.add_argument("--status-out", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    source = run_dir / "status.json"
    output = args.status_out or (run_dir / "watcher_status.json")

    while True:
        now = time.time()
        problems: list[str] = []
        data: dict[str, object] = {}
        if source.exists():
            try:
                data = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"status_read_error={exc}")
        else:
            problems.append("status_missing")

        state = str(data.get("state") or "starting")
        updated_at = parse_time(data.get("updated_at"))
        age = max(0.0, now - updated_at) if updated_at else None
        if state not in TERMINAL_STATES and age is not None and age > args.stale_seconds:
            problems.append(f"no_progress_for_{int(age)}s")
        if state not in TERMINAL_STATES and not pid_alive(args.pid):
            problems.append("runner_dead_before_completion")

        watcher = {
            "schema_version": "1.0",
            "checked_at": datetime.now(UTC).isoformat(),
            "run_dir": str(run_dir),
            "state": state,
            "completed": int(data.get("completed") or 0),
            "failed": int(data.get("failed") or 0),
            "remaining": int(data.get("remaining") or 0),
            "active_trials": data.get("active_trials") or {},
            "status_age_seconds": age,
            "runner_pid": args.pid,
            "problems": problems,
            "done": state in TERMINAL_STATES,
        }
        write_watcher_status(output, watcher)
        print(json.dumps(watcher, sort_keys=True), flush=True)

        if watcher["done"]:
            return 0 if not problems and watcher["failed"] == 0 else 1
        if problems:
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
