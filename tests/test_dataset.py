from __future__ import annotations

import csv
from pathlib import Path

from browsecomp250.config import DatasetConfig
from browsecomp250.crypto import encrypt
from browsecomp250.dataset import (
    dataset_path,
    load_items,
    validate_dataset_file,
    write_dataset_manifest,
)


def _write_fake_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["problem", "answer", "canary"])
        writer.writeheader()
        for index in range(1266):
            canary = f"browsecomp:unit-{index}"
            writer.writerow(
                {
                    "problem": encrypt(f"Question number {index}?", canary),
                    "answer": encrypt(f"Answer number {index}", canary),
                    "canary": canary,
                }
            )


def test_dataset_validation_selection_and_private_manifest(tmp_path: Path) -> None:
    subset = Path(__file__).parents[1] / "data" / "subset_indices.json"
    config = DatasetConfig(cache_dir=tmp_path, subset_indices_path=subset)
    path = dataset_path(config)
    _write_fake_dataset(path)

    metadata = validate_dataset_file(path, config)
    assert metadata["rows"] == 1266
    items = load_items(config)
    assert len(items) == 250
    assert items[0].question == f"Question number {items[0].source_index}?"
    assert items[0].answer == f"Answer number {items[0].source_index}"

    manifest = write_dataset_manifest(config)
    text = manifest.read_text(encoding="utf-8")
    assert "Question number" not in text
    assert "Answer number" not in text
    assert "selected_encrypted_row_hashes" in text
