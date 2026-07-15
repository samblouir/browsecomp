from __future__ import annotations

import json
import random
from pathlib import Path

from .constants import (
    OFFICIAL_DATASET_ROWS,
    SUBSET_INDICES_SHA256,
    SUBSET_SEED,
    SUBSET_SIZE,
)
from .util import canonical_sha256


def reference_indices() -> list[int]:
    """Reproduce OpenAI's `random.Random(0).sample(examples, 250)` selection."""
    return random.Random(SUBSET_SEED).sample(range(OFFICIAL_DATASET_ROWS), SUBSET_SIZE)


def load_indices(path: Path) -> list[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"Subset file must be a JSON integer list: {path}")
    validate_indices(value)
    return value


def validate_indices(indices: list[int]) -> None:
    if len(indices) != SUBSET_SIZE:
        raise ValueError(f"Expected {SUBSET_SIZE} indices, got {len(indices)}")
    if len(set(indices)) != len(indices):
        raise ValueError("Subset contains duplicate row indices")
    if min(indices) < 0 or max(indices) >= OFFICIAL_DATASET_ROWS:
        raise ValueError("Subset contains an out-of-range row index")
    digest = canonical_sha256(indices)
    if digest != SUBSET_INDICES_SHA256:
        raise ValueError(f"Subset hash mismatch: expected {SUBSET_INDICES_SHA256}, got {digest}")
    expected = reference_indices()
    if indices != expected:
        raise ValueError("Frozen subset does not match the OpenAI seed-0 sampling procedure")
