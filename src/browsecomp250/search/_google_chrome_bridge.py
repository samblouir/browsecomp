#!/usr/bin/env python3
"""Remote bridge for batched Google searches in an existing personal Chrome.

This file intentionally uses only the Python standard library. The local
provider copies it to the selected Mac and invokes it over SSH. It never starts
Chrome with a separate profile, never quits Chrome, and closes only tabs tagged
by the current request.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

RESULT_BEGIN = "__BC250_GOOGLE_CHROME_RESULT_BEGIN__"
RESULT_END = "__BC250_GOOGLE_CHROME_RESULT_END__"

_TAB_SCRIPT = r"""
function run(argv) {
  const operation = argv[0];
  const needle = argv[1];
  const chrome = Application("Google Chrome");
  const windows = chrome.windows();
  for (let wi = 0; wi < windows.length; wi++) {
    const win = windows[wi];
    const tabs = win.tabs();
    for (let ti = 0; ti < tabs.length; ti++) {
      const tab = tabs[ti];
      const url = String(tab.url() || "");
      if (url.indexOf(needle) !== -1) {
        if (operation === "activate") {
          win.activeTabIndex = ti + 1;
          delay(0.08);
        }
        const bounds = win.bounds();
        return JSON.stringify({
          found: true,
          window_index: wi + 1,
          tab_index: ti + 1,
          bounds: bounds,
          title: String(tab.title() || ""),
          url: String(tab.url() || ""),
          loading: Boolean(tab.loading())
        });
      }
    }
  }
  return JSON.stringify({found: false});
}
"""

_CLOSE_SCRIPT = r"""
function run(argv) {
  const needle = argv[0];
  const chrome = Application("Google Chrome");
  let closed = 0;
  const windows = chrome.windows();
  for (let wi = windows.length - 1; wi >= 0; wi--) {
    const tabs = windows[wi].tabs();
    for (let ti = tabs.length - 1; ti >= 0; ti--) {
      const url = String(tabs[ti].url() || "");
      if (url.indexOf(needle) !== -1) {
        tabs[ti].close();
        closed += 1;
      }
    }
  }
  return String(closed);
}
"""


class BridgeError(RuntimeError):
    pass


def _extract_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise BridgeError(f"command did not return JSON: {text[:500]!r}")


def _run(
    argv: list[str],
    *,
    timeout: float,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise BridgeError(f"command failed ({completed.returncode}): {detail[:1000]}")
    return completed


def _driver_call(driver: str, name: str, payload: dict[str, Any], timeout: float) -> str:
    completed = _run(
        [driver, "call", name, json.dumps(payload, separators=(",", ":"))],
        timeout=timeout,
    )
    return completed.stdout


def _tab_info(needle: str, operation: str, timeout: float) -> dict[str, Any]:
    completed = _run(
        ["osascript", "-l", "JavaScript", "-e", _TAB_SCRIPT, operation, needle],
        timeout=timeout,
    )
    return json.loads(completed.stdout.strip())


def _close_tabs(needle: str, timeout: float) -> int:
    completed = _run(
        ["osascript", "-l", "JavaScript", "-e", _CLOSE_SCRIPT, needle],
        timeout=timeout,
        check=False,
    )
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return 0


def _window_id(
    driver: str,
    pid: int,
    tab: dict[str, Any],
    timeout: float,
) -> int:
    raw = _extract_json(_driver_call(driver, "list_windows", {}, timeout))
    windows = raw.get("windows", raw) if isinstance(raw, dict) else raw
    bounds = tab.get("bounds") or []
    expected: tuple[float, float, float, float] | None = None
    if isinstance(bounds, list) and len(bounds) == 4:
        left, top, right, bottom = (float(value) for value in bounds)
        expected = (left, top, right - left, bottom - top)

    candidates: list[dict[str, Any]] = []
    for window in windows if isinstance(windows, list) else []:
        if not isinstance(window, dict) or int(window.get("pid") or -1) != pid:
            continue
        candidates.append(window)
        actual = window.get("bounds") or {}
        if expected and all(
            abs(float(actual.get(key, -100000)) - value) <= 2
            for key, value in zip(("x", "y", "width", "height"), expected, strict=True)
        ):
            return int(window["window_id"])

    title = str(tab.get("title") or "")
    title_matches = [window for window in candidates if str(window.get("title") or "") == title]
    if len(title_matches) == 1:
        return int(title_matches[0]["window_id"])
    raise BridgeError(f"could not map Chrome tab to a unique window: title={title!r}")


def _wait_for_tabs(entries: list[dict[str, Any]], timeout: float) -> dict[str, str]:
    deadline = time.monotonic() + timeout
    errors: dict[str, str] = {}
    pending = {str(entry["tag"]): entry for entry in entries}
    while pending and time.monotonic() < deadline:
        for tag in list(pending):
            try:
                info = _tab_info(tag, "inspect", min(timeout, 10))
            except Exception as exc:  # noqa: BLE001 - returned per query
                errors[tag] = str(exc)
                continue
            title = str(info.get("title") or "")
            if info.get("found") and (not info.get("loading") or title.endswith("Google Search")):
                pending.pop(tag, None)
        if pending:
            time.sleep(0.2)
    for tag in pending:
        errors[tag] = "Chrome tab did not finish loading before the deadline"
    return errors


def _run_batch(payload: dict[str, Any]) -> dict[str, Any]:
    batch_started = time.monotonic()
    driver = str(Path(str(payload["cua_driver"])).expanduser())
    bundle_id = str(payload.get("bundle_id") or "com.google.Chrome")
    timeout = float(payload.get("timeout_seconds") or 45)
    entries = [dict(item) for item in payload.get("entries") or []]
    if not entries:
        raise BridgeError("entries is empty")
    request_tag = str(payload["request_tag"])
    urls = [str(item["url"]) for item in entries]
    launch_started = time.monotonic()
    launched_raw = _driver_call(
        driver,
        "launch_app",
        {"bundle_id": bundle_id, "urls": urls},
        timeout,
    )
    launched = _extract_json(launched_raw)
    if not isinstance(launched, dict):
        raise BridgeError("launch_app returned a non-object")
    pid = int(launched.get("pid") or 0)
    if pid <= 0:
        raise BridgeError("launch_app did not return the Chrome pid")

    launch_seconds = time.monotonic() - launch_started
    load_started = time.monotonic()
    load_errors = _wait_for_tabs(entries, timeout)
    load_wait_seconds = time.monotonic() - load_started
    results: list[dict[str, Any]] = []
    for entry in entries:
        tag = str(entry["tag"])
        if tag in load_errors:
            results.append({"query": entry.get("query"), "tag": tag, "error": load_errors[tag]})
            continue
        try:
            extraction_started = time.monotonic()
            tab = _tab_info(tag, "activate", min(timeout, 10))
            if not tab.get("found"):
                raise BridgeError("tagged Chrome tab disappeared before extraction")
            window_id = _window_id(driver, pid, tab, timeout)
            page_text = _driver_call(
                driver,
                "page",
                {
                    "action": "get_text",
                    "pid": pid,
                    "window_id": window_id,
                },
                timeout,
            )
            results.append(
                {
                    "query": entry.get("query"),
                    "tag": tag,
                    "url": tab.get("url"),
                    "title": tab.get("title"),
                    "window_id": window_id,
                    "text": page_text,
                    "extraction_seconds": time.monotonic() - extraction_started,
                }
            )
        except Exception as exc:  # noqa: BLE001 - returned per query
            results.append({"query": entry.get("query"), "tag": tag, "error": str(exc)})

    return {
        "ok": any("text" in row for row in results),
        "pid": pid,
        "self_activation_suppressed": launched.get("self_activation_suppressed"),
        "request_tag": request_tag,
        "load_parallel": True,
        "launch_seconds": launch_seconds,
        "load_wait_seconds": load_wait_seconds,
        "batch_seconds": time.monotonic() - batch_started,
        "results": results,
    }


def _decode_payload(value: str) -> dict[str, Any]:
    padding = "=" * (-len(value) % 4)
    raw = base64.urlsafe_b64decode((value + padding).encode("ascii"))
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise BridgeError("payload must decode to an object")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-base64", required=True)
    args = parser.parse_args()
    payload = _decode_payload(args.payload_base64)
    request_tag = str(payload.get("request_tag") or "")
    if not request_tag.startswith("frlbc250-"):
        raise BridgeError("invalid request tag")

    def interrupted(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    result: dict[str, Any]
    closed = 0
    lock_path = "/tmp/browsecomp250-google-chrome.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            result = _run_batch(payload)
        except BaseException as exc:  # include signals so exact-tab cleanup still runs
            result = {
                "ok": False,
                "request_tag": request_tag,
                "error": str(exc) or type(exc).__name__,
                "results": [],
            }
        finally:
            try:
                closed = _close_tabs(
                    request_tag, min(float(payload.get("timeout_seconds") or 45), 15)
                )
            except Exception:
                closed = 0
    result["closed_tabs"] = closed
    result["bridge_pid"] = os.getpid()
    print(RESULT_BEGIN)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    print(RESULT_END)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(RESULT_BEGIN)
        print(json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":")))
        print(RESULT_END)
        raise SystemExit(1) from exc
