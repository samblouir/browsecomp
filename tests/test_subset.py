import json
from pathlib import Path

from browsecomp250.constants import SUBSET_INDICES_SHA256
from browsecomp250.subset import load_indices, reference_indices
from browsecomp250.util import canonical_sha256


def test_frozen_subset_matches_reference() -> None:
    path = Path(__file__).parents[1] / "data" / "subset_indices.json"
    indices = load_indices(path)
    assert indices == reference_indices()
    assert len(indices) == 250
    assert canonical_sha256(indices) == SUBSET_INDICES_SHA256


def test_packaged_subset_copy_matches_repository() -> None:
    from browsecomp250.constants import PACKAGE_ROOT

    packaged = json.loads((PACKAGE_ROOT / "data" / "subset_indices.json").read_text())
    repository = json.loads(Path("data/subset_indices.json").read_text())
    assert packaged == repository == reference_indices()
