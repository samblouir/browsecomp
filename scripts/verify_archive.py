#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN_PARTS = {
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
    "private",
}
FORBIDDEN_NAMES = {
    ".env",
    ".coverage",
    "browse_comp_test_set.csv",
    "id_rsa",
    "id_ed25519",
}
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"sk-[A-Za-z0-9_-]{24,}"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--sha256")
    args = parser.parse_args()

    archive_bytes = Path(args.archive).read_bytes()
    digest = hashlib.sha256(archive_bytes).hexdigest()
    if args.sha256 and digest.casefold() != args.sha256.casefold():
        raise SystemExit(f"SHA-256 mismatch: expected {args.sha256}, got {digest}")

    with zipfile.ZipFile(args.archive) as zf:
        bad: list[str] = []
        for info in zf.infolist():
            path = PurePosixPath(info.filename)
            if any(part in FORBIDDEN_PARTS for part in path.parts) or path.name in FORBIDDEN_NAMES:
                bad.append(info.filename)
                continue
            if info.file_size > 50_000_000:
                bad.append(f"oversize:{info.filename}")
                continue
            if info.file_size <= 10_000_000:
                payload = zf.read(info)
                if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
                    bad.append(f"secret-pattern:{info.filename}")
        if bad:
            raise SystemExit("Archive verification failed:\n" + "\n".join(bad))
        if zf.testzip() is not None:
            raise SystemExit("ZIP CRC verification failed")
    print(f"Archive OK: {args.archive}\nSHA-256: {digest}")


if __name__ == "__main__":
    main()
