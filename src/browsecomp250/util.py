from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-((?:[^{}]|\{[^{}]*\})*))?\}")
_SECRET_EXACT_KEYS = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "client_secret",
    "token",
    "access_token",
    "auth_token",
    "admin_token",
    "bearer_token",
    "subscription_token",
    "x_api_key",
    "x_subscription_token",
}
_PLACEHOLDER_SECRETS = {
    "change-me",
    "changeme",
    "dummy",
    "example",
    "replace-me",
    "test",
    "your-api-key",
    "your_api_key",
}


def _is_secret_key(key: object) -> bool:
    normalized = str(key).strip().casefold().replace("-", "_")
    # Metadata describing authentication policy or a one-way fingerprint is safe
    # and necessary for reproducibility.
    if normalized.startswith("allow_empty_") or "fingerprint" in normalized:
        return False
    if normalized in _SECRET_EXACT_KEYS:
        return True
    return any(
        normalized.endswith(suffix)
        for suffix in (
            "_api_key",
            "_password",
            "_client_secret",
            "_access_token",
            "_auth_token",
            "_admin_token",
            "_bearer_token",
            "_subscription_token",
        )
    )


def is_placeholder_secret(value: str) -> bool:
    """Return whether a configured credential is an obvious non-secret placeholder."""
    normalized = value.strip().strip("\"'").casefold()
    return normalized in _PLACEHOLDER_SECRETS or (
        normalized.startswith("${") and normalized.endswith("}")
    )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def expand_env(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")

    return _ENV_PATTERN.sub(replace, text)


def expand_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if base is not None and not path.is_absolute():
        path = base / path
    return path.resolve()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def atomic_write_json(path: Path, value: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(value, indent=indent, ensure_ascii=False) + "\n")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if _is_secret_key(key):
                out[key] = "<redacted>" if item else ""
            else:
                out[key] = redact(item)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars < 100:
        return text[:max_chars]
    half = (max_chars - 60) // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n...[{omitted} chars omitted]...\n{text[-half:]}"


def chunks(sequence: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def sqlite_family_state(path: Path) -> dict[str, Any]:
    """Return hashes for a SQLite database and any WAL/SHM sidecars."""
    members: list[dict[str, Any]] = []
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists() and candidate.is_file():
            members.append(
                {
                    "path": str(candidate),
                    "size_bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    return {"path": str(path), "exists": bool(members), "members": members}


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
