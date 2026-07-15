#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "FILE_SHA256SUMS.txt"
SKIP_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".nox",
    "htmlcov",
    "__pycache__",
    "runs",
    "release",
    "dist",
    "build",
}
SKIP_FILES = {".env", ".coverage", OUTPUT.name}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in relative.parts):
        return False
    if path.name in SKIP_FILES or path.suffix in SKIP_SUFFIXES:
        return False
    if path.name == "browse_comp_test_set.csv":
        return False
    return path.is_file()


lines = []
for path in sorted(ROOT.rglob("*")):
    if not included(path):
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(lines)} checksums to {OUTPUT}")
