from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import httpx

from .config import DatasetConfig
from .crypto import decrypt
from .subset import load_indices
from .types import BenchmarkItem
from .util import atomic_write_json, sha256_file, utc_now_iso

REQUIRED_COLUMNS = {"problem", "answer", "canary"}


class DatasetError(RuntimeError):
    pass


def dataset_path(config: DatasetConfig) -> Path:
    return config.cache_dir / "browse_comp_test_set.csv"


async def download_dataset(config: DatasetConfig, *, force: bool = False) -> Path:
    target = dataset_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        validate_dataset_file(target, config)
        return target

    temp = target.with_suffix(".csv.part")
    async with (
        httpx.AsyncClient(follow_redirects=True, timeout=120) as client,
        client.stream("GET", config.source_url) as response,
    ):
        response.raise_for_status()
        with temp.open("wb") as handle:
            async for chunk in response.aiter_bytes():
                handle.write(chunk)
    temp.replace(target)
    validate_dataset_file(target, config)
    return target


def read_encrypted_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if not REQUIRED_COLUMNS.issubset(columns):
            raise DatasetError(
                f"Dataset missing required columns {sorted(REQUIRED_COLUMNS)}; found {sorted(columns)}"
            )
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def validate_dataset_file(path: Path, config: DatasetConfig) -> dict[str, object]:
    rows = read_encrypted_rows(path)
    if len(rows) != config.expected_rows:
        raise DatasetError(f"Expected {config.expected_rows} rows, found {len(rows)}")
    digest = sha256_file(path)
    if config.expected_sha256 and digest.lower() != config.expected_sha256.lower():
        raise DatasetError(
            f"Dataset SHA-256 mismatch: expected {config.expected_sha256}, got {digest}"
        )
    empty = [
        index for index, row in enumerate(rows) if not all(row.get(k) for k in REQUIRED_COLUMNS)
    ]
    if empty:
        raise DatasetError(f"Dataset has empty required fields in rows: {empty[:10]}")
    return {"path": str(path), "rows": len(rows), "sha256": digest}


def encrypted_row_hash(row: dict[str, str]) -> str:
    payload = json.dumps(
        {key: row.get(key, "") for key in sorted(REQUIRED_COLUMNS)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_items(config: DatasetConfig) -> list[BenchmarkItem]:
    path = dataset_path(config)
    validate_dataset_file(path, config)
    rows = read_encrypted_rows(path)
    indices = load_indices(config.subset_indices_path)
    items: list[BenchmarkItem] = []
    for rank, source_index in enumerate(indices):
        row = rows[source_index]
        canary = row["canary"]
        try:
            question = decrypt(row["problem"], canary)
            answer = decrypt(row["answer"], canary)
        except Exception as exc:  # noqa: BLE001
            raise DatasetError(f"Failed to decrypt source row {source_index}: {exc}") from exc
        items.append(
            BenchmarkItem(
                item_id=f"bc250-{rank:03d}-row-{source_index:04d}",
                subset_rank=rank,
                source_index=source_index,
                encrypted_row_hash=encrypted_row_hash(row),
                question=question,
                answer=answer,
                canary=canary,
            )
        )
    return items


def write_dataset_manifest(config: DatasetConfig) -> Path:
    path = dataset_path(config)
    metadata = validate_dataset_file(path, config)
    rows = read_encrypted_rows(path)
    indices = load_indices(config.subset_indices_path)
    manifest = {
        "schema_version": "1.0",
        "created_at": utc_now_iso(),
        "source_url": config.source_url,
        "source_path": str(path),
        "source_rows": metadata["rows"],
        "source_sha256": metadata["sha256"],
        "subset_size": len(indices),
        "subset_indices": indices,
        "selected_encrypted_row_hashes": [encrypted_row_hash(rows[index]) for index in indices],
        "contains_plaintext_questions": False,
        "contains_plaintext_answers": False,
    }
    destination = config.cache_dir / "browsecomp250-dataset-manifest.json"
    atomic_write_json(destination, manifest)
    return destination


def iter_plaintext_for_leak_scan(config: DatasetConfig) -> Iterable[tuple[str, str, str]]:
    for item in load_items(config):
        yield item.item_id, item.question, item.answer
