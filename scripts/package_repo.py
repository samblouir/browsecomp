#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "browsecomp-250-openai-compatible"
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
    ".eggs",
}
SKIP_FILES = {".env", ".coverage"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
FORBIDDEN_BASENAMES = {
    "browse_comp_test_set.csv",
    "id_rsa",
    "id_ed25519",
}


def iter_files() -> list[Path]:
    output: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if not path.is_file():
            continue
        if path.name in SKIP_FILES or path.suffix in SKIP_SUFFIXES:
            continue
        if path.name in FORBIDDEN_BASENAMES:
            raise SystemExit(f"Refusing to package forbidden file: {relative}")
        output.append(path)
    return output


def verify_no_sensitive_names(files: list[Path]) -> None:
    for path in files:
        relative = path.relative_to(ROOT).as_posix().casefold()
        if relative.startswith("private/") or "/private/" in relative:
            raise SystemExit(f"Refusing to package private artifact: {relative}")
        if "search-cache.sqlite" in relative or "page-cache.sqlite" in relative:
            raise SystemExit(f"Refusing to package cache: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT.parent / f"{PREFIX}.zip")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    if not args.skip_tests:
        subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "src", "tests"], cwd=ROOT, check=True
        )
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, env=env, check=True)

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run([sys.executable, "scripts/generate_schema.py"], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "scripts/update_checksums.py"], cwd=ROOT, check=True)

    files = iter_files()
    verify_no_sensitive_names(files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    if temp.exists():
        temp.unlink()

    # Fixed timestamps make repeated packages byte-stable when file contents are unchanged.
    timestamp = (2026, 7, 15, 0, 0, 0)
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(f"{PREFIX}/{relative}", date_time=timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if os.access(path, os.X_OK) else 0o644
            info.external_attr = mode << 16
            archive.writestr(
                info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )

    temp.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(args.output)
    print(sidecar)
    print(digest)


if __name__ == "__main__":
    main()
