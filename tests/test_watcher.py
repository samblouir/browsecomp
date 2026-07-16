import importlib.util
import os
from pathlib import Path

_WATCHER_PATH = Path(__file__).parents[1] / "scripts" / "watch_star_run.py"
_SPEC = importlib.util.spec_from_file_location("watch_star_run", _WATCHER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_WATCHER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WATCHER)
latest_active_event_mtime = _WATCHER.latest_active_event_mtime


def test_run_log_counts_as_activity_while_helpers_are_running(tmp_path) -> None:
    item_dir = tmp_path / "items" / "0003-item"
    item_dir.mkdir(parents=True)
    event_path = item_dir / "attempt-01-events.jsonl"
    event_path.write_text("{}\n", encoding="utf-8")
    runner_log = tmp_path / "runner-v9.log"
    runner_log.write_text("helper progress\n", encoding="utf-8")
    os.utime(event_path, (100, 100))
    os.utime(runner_log, (200, 200))

    assert latest_active_event_mtime(tmp_path, {"3": {}}) == 200
