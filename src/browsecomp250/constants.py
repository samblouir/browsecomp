from pathlib import Path

__version__ = "0.1.0"

OFFICIAL_DATASET_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)
OFFICIAL_DATASET_ROWS = 1266
SUBSET_SIZE = 250
SUBSET_SEED = 0
SUBSET_INDICES_SHA256 = "b0c3334bf37a9ee9eb653639daac477576bce36ec7bcfc5e3ec8ef88c168f4f0"
BROWSECOMP_CANARY_PREFIX = "browsecomp:"

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent.parent
REPOSITORY_SUBSET_PATH = REPOSITORY_ROOT / "data" / "subset_indices.json"
PACKAGED_SUBSET_PATH = PACKAGE_ROOT / "data" / "subset_indices.json"
DEFAULT_SUBSET_PATH = (
    REPOSITORY_SUBSET_PATH if REPOSITORY_SUBSET_PATH.exists() else PACKAGED_SUBSET_PATH
)
