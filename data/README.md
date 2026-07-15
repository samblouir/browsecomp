# Data directory

This directory intentionally contains only the frozen subset definition and its metadata.

- `subset_indices.json` — 250 zero-based source-row positions selected by `random.Random(0).sample(range(1266), 250)`.
- `subset_spec.json` — provenance, naming, and hash metadata.

The official BrowseComp CSV and decrypted questions/answers are not distributed here. `bc250 prepare` downloads the encrypted CSV into the configured user cache directory, normally `~/.cache/browsecomp250/`.

Do not commit:

- `browse_comp_test_set.csv`;
- decrypted items;
- model outputs tied to item text;
- grader outputs containing answers; or
- benchmark-derived examples.
