from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import AppConfig
from ..constants import BROWSECOMP_CANARY_PREFIX
from ..dataset import iter_plaintext_for_leak_scan
from ..util import atomic_write_json, redact, sha256_file


class LeakDetectedError(RuntimeError):
    pass


def scan_public_tree(run_dir: Path, config: AppConfig) -> list[str]:
    public_dir = run_dir / "public"
    if not public_dir.exists():
        raise FileNotFoundError(f"No public report directory: {public_dir}")
    texts: dict[Path, str] = {}
    for path in public_dir.rglob("*"):
        if path.is_file() and path.stat().st_size <= 20_000_000:
            texts[path] = path.read_text(encoding="utf-8", errors="ignore")
    findings: list[str] = []
    for path, text in texts.items():
        if BROWSECOMP_CANARY_PREFIX in text:
            findings.append(f"canary prefix in {path}")
    for item_id, question, answer in iter_plaintext_for_leak_scan(config.dataset):
        for path, text in texts.items():
            if len(question) >= 20 and question in text:
                findings.append(f"plaintext question {item_id} in {path}")
            if len(answer) >= 4 and answer in text:
                findings.append(f"reference answer {item_id} in {path}")
    return findings


def sanitize_run(run_dir: Path, destination: Path, config: AppConfig) -> Path:
    findings = scan_public_tree(run_dir, config)
    if findings:
        raise LeakDetectedError("Publication leak scan failed:\n" + "\n".join(findings[:50]))
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copytree(run_dir / "public", destination / "public")
    lock = json.loads((run_dir / "run.lock.json").read_text(encoding="utf-8"))
    atomic_write_json(destination / "run.lock.json", redact(lock))
    cache_manifest = run_dir / "cache.manifest.json"
    if cache_manifest.exists():
        shutil.copy2(cache_manifest, destination / "cache.manifest.json")
    checksums = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            checksums[str(path.relative_to(destination))] = sha256_file(path)
    atomic_write_json(destination / "SHA256SUMS.json", checksums)
    return destination
