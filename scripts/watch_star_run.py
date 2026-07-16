#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

TERMINAL_STATES = {"completed", "completed_with_errors"}
NOTIFIER = Path.home() / ".codex/skills/smux-notifier/scripts/smux_notify.py"


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


def latest_active_event_mtime(run_dir: Path, active_trials: object) -> float:
    if not isinstance(active_trials, dict):
        return 0.0
    latest = 0.0
    items_dir = run_dir / "items"
    for raw_index in active_trials:
        try:
            prefix = f"{int(raw_index):04d}-"
        except (TypeError, ValueError):
            continue
        for item_dir in items_dir.glob(prefix + "*"):
            for event_path in item_dir.glob("attempt-*-events.jsonl"):
                try:
                    latest = max(latest, event_path.stat().st_mtime)
                except OSError:
                    continue
    return latest


def write_watcher_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_status(data: dict[str, object]) -> dict[str, object]:
    """Normalize both benchmark-runner and private guided-training status schemas."""
    if "target_total_questions" in data:
        target = int(data.get("target_total_questions") or 0)
        completed = int(data.get("current_records") or 0)
        failed = int(data.get("failed") or 0)
        finished = int(data.get("finished_items") or completed + failed)
        done = bool(data.get("done"))
        state = "completed_with_errors" if done and failed else "completed" if done else "running"
        return {
            "state": state,
            "completed": completed,
            "failed": failed,
            "remaining": max(0, target - finished),
            "active_trials": data.get("in_progress") or {},
            "updated_at": data.get("updated_at"),
            "target": target,
        }
    return {
        "state": str(data.get("state") or "starting"),
        "completed": int(data.get("completed") or 0),
        "failed": int(data.get("failed") or 0),
        "remaining": int(data.get("remaining") or 0),
        "active_trials": data.get("active_trials") or {},
        "updated_at": data.get("updated_at"),
        "target": int(data.get("target") or 0),
    }


def send_notification(
    *,
    target: str,
    mux: str,
    message: str,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"notification_dry_run={message}", flush=True)
        return
    if not NOTIFIER.exists():
        raise RuntimeError(f"Missing smux notifier: {NOTIFIER}")
    completed = subprocess.run(
        [
            sys.executable,
            str(NOTIFIER),
            "--target",
            target,
            "--mode",
            "immediate",
            "--mux",
            mux,
            "--message",
            message,
            "--verify-submit-retries",
            "8",
            "--delay-seconds",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"smux notification failed: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch a BrowseComp-250 Star run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument(
        "--status-file",
        type=Path,
        help="Read this runner status file instead of RUN_DIR/status.json.",
    )
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--stale-seconds", type=float, default=600.0)
    parser.add_argument("--status-out", type=Path)
    parser.add_argument("--failure-confirmations", type=int, default=3)
    parser.add_argument("--notify-target")
    parser.add_argument("--notify-mux", default="smux", choices=["auto", "smux", "tmux"])
    parser.add_argument("--notify-prefix", default="vllm_handler")
    parser.add_argument("--dry-run-notify", action="store_true")
    args = parser.parse_args()
    if args.failure_confirmations < 1:
        parser.error("--failure-confirmations must be at least 1")

    run_dir = args.run_dir.resolve()
    source = args.status_file.resolve() if args.status_file else run_dir / "status.json"
    output = args.status_out or (run_dir / "watcher_status.json")
    consecutive_problem_checks = 0

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

        normalized = normalized_status(data)
        state = str(normalized["state"])
        updated_at = parse_time(normalized.get("updated_at"))
        event_mtime = latest_active_event_mtime(run_dir, normalized.get("active_trials"))
        latest_activity = max(updated_at, event_mtime)
        age = max(0.0, now - latest_activity) if latest_activity else None
        if state not in TERMINAL_STATES and age is not None and age > args.stale_seconds:
            problems.append(f"no_progress_for_{int(age)}s")
        if state not in TERMINAL_STATES and not pid_alive(args.pid):
            problems.append("runner_dead_before_completion")

        watcher = {
            "schema_version": "1.0",
            "checked_at": datetime.now(UTC).isoformat(),
            "run_dir": str(run_dir),
            "state": state,
            "completed": int(normalized["completed"]),
            "failed": int(normalized["failed"]),
            "remaining": int(normalized["remaining"]),
            "target": int(normalized["target"]),
            "active_trials": normalized["active_trials"],
            "status_age_seconds": age,
            "latest_active_event_mtime": event_mtime or None,
            "runner_pid": args.pid,
            "problems": problems,
            "done": state in TERMINAL_STATES,
        }
        write_watcher_status(output, watcher)
        print(json.dumps(watcher, sort_keys=True), flush=True)

        if watcher["done"]:
            if args.notify_target:
                message = (
                    f"{args.notify_prefix}_done: guided training reached "
                    f"{watcher['completed']}/{watcher['target']}; failed={watcher['failed']}; "
                    f"status={output}"
                )
                send_notification(
                    target=args.notify_target,
                    mux=args.notify_mux,
                    message=message,
                    dry_run=args.dry_run_notify,
                )
            return 0 if not problems and watcher["failed"] == 0 else 1
        if problems:
            consecutive_problem_checks += 1
        else:
            consecutive_problem_checks = 0
        if consecutive_problem_checks >= args.failure_confirmations:
            if args.notify_target:
                message = (
                    f"{args.notify_prefix}_alert: guided training needs attention at "
                    f"{watcher['completed']}/{watcher['target']}; "
                    f"problems={','.join(problems)}; status={output}"
                )
                send_notification(
                    target=args.notify_target,
                    mux=args.notify_mux,
                    message=message,
                    dry_run=args.dry_run_notify,
                )
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
