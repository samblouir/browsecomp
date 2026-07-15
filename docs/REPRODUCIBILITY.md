# Reproducibility

## What can and cannot be reproduced

The repository can freeze:

- source code;
- configuration;
- subset indices;
- source CSV digest;
- model identifier and request fields;
- search/browser settings;
- grader settings;
- random seeds;
- budgets;
- run outputs; and
- cache snapshots retained by the operator.

It cannot guarantee that live web pages, search rankings, APIs, models behind mutable aliases, or provider routing remain unchanged.

A BrowseComp result is therefore a measurement at a specified time, not a timeless model constant.

## Run lock

Before scheduling trials, the engine writes `run.lock.json` containing:

- runner version;
- benchmark label and explicit `benchmark_official: false` marker;
- encrypted source CSV path, row count, and SHA-256;
- subset index SHA-256;
- complete resolved configuration with credentials redacted;
- one-way short fingerprints of model, grader, and search credentials;
- Python and platform metadata;
- Git commit/dirty state when available; and
- a canonical replay hash.

A run directory with a different replay hash cannot be resumed.

Credential fingerprints are not authentication tokens. They only detect accidental use of a different credential context.

## Randomness

The subset seed is fixed at zero and independent of `run.seed`.

`run.seed` controls work-order shuffling only. Model-side randomness is controlled by temperature and any provider-specific seed added through `extra_body`. Not all OpenAI-compatible endpoints honor request seeds deterministically, especially under distributed serving.

For strict deterministic studies:

- set temperature to zero;
- set a provider seed when supported;
- pin model weights and serving code;
- disable nondeterministic speculative or sampling behavior where material;
- freeze search/page caches; and
- run serially or verify that concurrency does not affect routing.

## Caches

SQLite search and page caches support five modes. Preserve the cache files and their WAL sidecars if exact replay matters.

Caches are not public-safe by default. They may contain:

- benchmark-derived search queries;
- page snippets;
- URLs;
- full extracted page text; and
- timestamps.

Archive them in encrypted internal storage, not in a public result release.

A live comparison should generally use the same cache policy for all systems. Mixing a warm cache for one model with a cold cache for another changes both latency and external evidence availability.

## Model aliases

A model name such as `star` can point to changing weights or infrastructure. Add immutable deployment metadata through custom headers/body fields or a separate internal manifest:

- model checkpoint digest;
- tokenizer digest;
- serving image digest;
- Git commit;
- quantization format;
- tensor/pipeline parallelism;
- decoding implementation;
- context window;
- rollout date; and
- endpoint region.

Do not put proprietary metadata into public artifacts unless intended. Preserve it privately and publish a stable release identifier.

## Evaluation window

Record exact start and end timestamps in UTC. For cross-model comparison, interleave model trials when possible rather than running one entire model days before another. Interleaving reduces systematic drift from changing search rankings or web content.

This repository's simple engine runs one model configuration per run. A campaign operator can alternate chunks across configurations or run separate endpoints concurrently. Preserve separate run locks.

## Source and dependency pinning

- `requirements-tested.txt` records the locally validated dependency set.
- `pyproject.toml` uses bounded compatibility ranges.
- `configs/schema.json` captures the configuration model.
- `FILE_SHA256SUMS.txt` covers the distributed repository files.
- Resume compatibility hashes immutable configuration, dataset, subset, and
  credential fingerprints. The original mutable cache snapshot remains in the
  run lock for provenance but does not make a run reject its own continuation.
- the ZIP has a separate SHA-256 sidecar.

For a long-lived campaign, build a container image and record its image digest.

## Result reproducibility checklist

Before publication, retain:

- repository ZIP and SHA-256;
- source CSV SHA-256;
- subset hash;
- complete run directory;
- model deployment manifest;
- search and browser configuration;
- grader identifier;
- dependency lock;
- cache snapshots or explicit live-cache policy;
- start/end times;
- API/service incident notes; and
- sanitized public release artifacts.
