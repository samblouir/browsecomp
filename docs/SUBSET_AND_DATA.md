# Subset and dataset handling

## Why a fixed 250-item subset

The official BrowseComp set contains 1,266 items. A 250-item subset lowers model, search, browsing, and grading cost while retaining enough items to detect large performance differences. It does not reliably resolve very small differences: at approximately 90% accuracy, a 250-item binomial estimate has a 95% interval roughly four percentage points wide in each direction.

The subset is intended for:

- rapid but serious model comparisons;
- investor or product demonstrations with a disclosed protocol;
- search/browsing ablations;
- cost and latency frontier studies; and
- regression testing.

It is not a replacement for full BrowseComp when making definitive claims.

## Exact derivation

OpenAI's reference evaluator uses:

```python
rng = random.Random(0)
examples = rng.sample(examples, num_examples)
```

When `num_examples=250`, sampling operates on the CSV rows in their loaded order. This repository commits the corresponding integer source-row positions, preserving the order returned by `random.sample`.

To reproduce:

```bash
.venv/bin/bc250 subset
```

The command validates:

- exactly 250 indices;
- no duplicates;
- every index in `[0, 1265]`;
- canonical JSON SHA-256;
- exact equality with a fresh Python seed-0 derivation.

## Canonical hash

The index hash is computed over compact, sorted-key JSON encoding of the integer list:

```python
json.dumps(indices, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

The committed digest is:

```text
b0c3334bf37a9ee9eb653639daac477576bce36ec7bcfc5e3ec8ef88c168f4f0
```

The ordering matters because it defines `subset_rank`, stable item identifiers, and deterministic limited runs.

## Official encrypted source

The official reference code downloads:

```text
https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv
```

Required columns:

- `problem` — encrypted question;
- `answer` — encrypted reference answer;
- `canary` — per-row key material and benchmark canary.

The compatibility decryption uses SHA-256 key derivation and XOR, matching the upstream reference implementation.

## Snapshot pinning

The upstream URL is stable-looking but not content-addressed. Therefore:

1. run `bc250 prepare` from a controlled machine;
2. record the printed CSV SHA-256;
3. set `BC250_EXPECTED_DATASET_SHA256`;
4. preserve the generated dataset manifest;
5. use the same digest for every model in a comparison campaign.

The headline command refuses an unpinned dataset by default.

An emergency override exists:

```bash
.venv/bin/bc250 headline --dry-run --allow-unpinned-dataset
```

Do not use that override for a published score.

## Runtime privacy model

The harness:

- downloads the encrypted CSV to the user cache directory;
- validates the complete encrypted file;
- decrypts only selected items in process memory;
- writes questions and reference answers only to the private transcript area when private transcripts are enabled;
- never places plaintext questions or reference answers in public CSV, JSON, Markdown, or HTML reports; and
- scans publication output against all selected plaintext questions and answers.

The source CSV itself should not be copied into the repository archive or a result release.

## Item identities

Items receive IDs such as:

```text
bc250-000-row-0788
```

The ID encodes:

- zero-based subset rank; and
- zero-based source-row position.

The run lock also records the source CSV hash and the committed subset hash. The dataset cache manifest records a SHA-256 for each selected encrypted row, allowing operators to diagnose upstream row-level drift without publishing plaintext.

## Contamination and memorization

BrowseComp is public and static. A model may know questions or answers from training, evaluation leakage, prior runs, or search-engine indexing. This harness cannot prove that an answer was obtained from live browsing rather than memorization.

Mitigations:

- require citations for headline runs;
- inspect private trajectories for suspicious zero-search answers;
- report search and page-use distributions;
- compare no-browse and browse-enabled ablations;
- avoid publishing benchmark examples;
- evaluate a time-sensitive or privately held browsing set in parallel; and
- treat BrowseComp-250 as one component of a broader evidence package.
