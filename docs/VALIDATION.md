# Validation record

## Validated in the packaged development environment

The final artifact was validated on Python 3.13.5 in both the construction environment and a clean virtual environment without system site-packages. The suite contains **38 passing tests**. Tests were also run with `ResourceWarning` promoted to an error.

The release process performs:

- Python bytecode compilation for `src/` and `tests/`;
- 38 unit and integration-style tests using mock HTTP transports and a synthetic 1,266-row encrypted dataset;
- frozen subset regeneration and hash validation;
- source-wheel construction and installation into an isolated virtual environment;
- working-directory-independent validation of the installed `bc250 subset` command and embedded frozen subset;
- CLI version/subset/config invocations;
- headline dry-run validation with dummy credentials and a dummy pinned dataset digest;
- shell syntax validation for every script;
- YAML parsing and Pydantic configuration validation;
- configuration JSON-schema generation;
- repository file checksum generation and verification;
- ZIP structural integrity testing;
- archive scans for `.env`, benchmark CSVs, private keys, virtual environments, caches, bytecode, and private run artifacts; and
- verification that the archive contains no plaintext BrowseComp dataset.

## Mock-tested components

- OpenAI-compatible response parsing;
- JSON-action parsing;
- native tool-call parsing;
- model retry logic at the unit boundary;
- Brave, Tavily, and SearXNG adapters;
- direct HTML retrieval/extraction;
- redirect handling;
- private-network URL rejection;
- read-only cache failure behavior;
- semantic-grader parsing;
- deterministic grading;
- append-only cache/storage behavior;
- error-inclusive aggregation; and
- agent search-to-final flow;
- complete engine execution through locking, encrypted-data loading, trial persistence, grading, aggregation, and public reporting; and
- publication sanitizer success and deliberate benchmark-leak rejection.

## Not performed during artifact construction

The package build environment has no general outbound network access or user credentials. Therefore it does not execute:

- the official CSV download;
- a live model call to the user's Star endpoint;
- a live search-provider call;
- live web browsing;
- a live external semantic-grader call;
- a full 250-item evaluation; or
- Playwright browser installation/runtime.

These machine-specific checks are provided by:

```bash
bc250 prepare
bc250 doctor --live
bc250 run --config configs/smoke.yaml --limit 1
bc250 headline --dry-run
```

Run them before committing substantial evaluation spend.

## Publication gate

A result should not be released until all of the following pass:

```bash
bc250 verify-run <run-dir> --config <exact-config>
bc250 sanitize <run-dir> <release-dir> --config <exact-config>
```

Additionally review the release directory manually and confirm the public headline names the custom subset and complete harness.
