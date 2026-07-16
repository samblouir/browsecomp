# BrowseComp-250 OpenAI-Compatible Evaluation Harness

A reproducible, privacy-conscious evaluation repository for running a **fixed 250-item subset of BrowseComp** against any model exposed through an OpenAI-compatible `POST /v1/chat/completions` API.

This repository provides the complete evaluation system around the model:

- a frozen, deterministic 250-row subset;
- a common autonomous browsing agent;
- interchangeable search providers;
- direct HTTP and optional Playwright page retrieval;
- the BrowseComp reference semantic-grader prompt;
- resumable concurrent execution;
- per-run protocol locks and checksums;
- cost, token, latency, calibration, and uncertainty reporting;
- publication-safe artifact generation; and
- tests, CI, security controls, and operational documentation.

## Benchmark status

**BrowseComp-250 is a custom subset, not an official OpenAI benchmark split.**

It is defined as the first 250-sample behavior of OpenAI's reference implementation:

```python
random.Random(0).sample(examples, 250)
```

The exact source-row indices are committed in `data/subset_indices.json`. Their canonical JSON SHA-256 is:

```text
b0c3334bf37a9ee9eb653639daac477576bce36ec7bcfc5e3ec8ef88c168f4f0
```

The repository never redistributes the official encrypted CSV or plaintext questions and answers. It downloads the encrypted CSV at runtime from the URL used by OpenAI's `simple-evals` reference implementation, validates its row count and optional SHA-256 pin, and decrypts selected rows only in memory.

Do not label a result from this harness merely as “BrowseComp.” Use wording such as:

> BrowseComp-250 (fixed seed-0 subset, n=250), FrontierRL common browsing harness, one attempt per item.

Do not compare this score directly with a provider's full 1,266-item BrowseComp result.

## Quick start

Requirements:

- Python 3.12 or newer;
- internet access for dataset download and browsing;
- one supported search provider;
- an OpenAI-compatible model endpoint; and
- a fixed semantic grader for headline-quality results.

```bash
unzip browsecomp-250-openai-compatible.zip
cd browsecomp-250-openai-compatible

./scripts/bootstrap.sh
cp .env.example .env
```

Edit `.env`:

```dotenv
BC250_MODEL_API_BASE=http://127.0.0.1:8000/v1
BC250_MODEL_API_KEY=your-model-api-key
BC250_MODEL_NAME=star

BC250_SEARCH_PROVIDER=brave
BC250_BRAVE_API_KEY=your-brave-key

BC250_GRADER_API_BASE=https://api.openai.com/v1
BC250_GRADER_API_KEY=your-grader-key
BC250_GRADER_MODEL=gpt-5.6
```

A local endpoint that deliberately does not require authentication must be enabled explicitly:

```dotenv
BC250_MODEL_API_KEY=
BC250_ALLOW_EMPTY_MODEL_API_KEY=true
```

Prepare and pin the encrypted source snapshot:

```bash
.venv/bin/bc250 prepare --config configs/headline.yaml
```

The command prints the source CSV SHA-256. Put that value in `.env`:

```dotenv
BC250_EXPECTED_DATASET_SHA256=<printed-sha256>
```

Validate the environment:

```bash
.venv/bin/bc250 doctor --config configs/smoke.yaml
.venv/bin/bc250 doctor --config configs/smoke.yaml --live
```

Run one item with the low-cost smoke profile:

```bash
.venv/bin/bc250 run --config configs/smoke.yaml --limit 1
```

Validate the headline protocol without spending inference or search credits:

```bash
.venv/bin/bc250 headline --config configs/headline.yaml --dry-run
```

Launch all 250 items:

```bash
.venv/bin/bc250 headline --config configs/headline.yaml --yes
```

## Headline protocol

`configs/headline.yaml` fixes the following defaults:

| Component | Setting |
|---|---|
| Subset | Frozen seed-0 250-row subset |
| Attempts | 1 per item |
| Model temperature | 0.3 |
| Model output budget | At least 16,384 tokens per turn |
| Agent protocol | JSON actions |
| Maximum agent steps | 80 |
| Maximum search calls | 40 |
| Maximum page opens | 100 |
| Task timeout | 1,800 seconds |
| Grading | BrowseComp semantic LLM-grader template |
| Confidence interval | 95% Wilson plus bootstrap interval |
| Errors/timeouts | Counted as incorrect |
| Public artifacts | Questions, reference answers, model answers, URLs, and explanations omitted |

The `headline` command refuses configurations that weaken major minimum budgets, change the subset, use multiple attempts, use a nonsemantic grader, omit required credentials, or leave the dataset SHA unpinned unless an explicit escape hatch is supplied.

## Supported search providers

- Brave Search API
- OpenRouter Exa web search with standardized URL citations
- Experimental Google-in-user-Chrome and hybrid adapters (not used by Star campaigns)
- Tavily Search API
- Serper
- SearXNG
- Server-side Bing and Yahoo organic search through authorized SSH egress hosts
- A fail-open Bing/Yahoo server-side meta-search stratum that interleaves both result sets

All providers are normalized to the same internal result schema. The OpenRouter
Exa carrier's generated prose is discarded; only URL-citation annotations are
used. Fix the provider, region, language, safe-search setting, result count, and
evaluation dates across compared systems.

The Star configs default to the Brave API and permit an explicit
`BC250_SEARCH_PROVIDER` override that is frozen in each run's protocol lock.
Campaign reports must name the provider actually used; results collected with
different providers must be reported as separate strata or with per-row
provenance. The configs expose the caller-owned
`ask_external_model` tool through an isolated Star-2 agent that can use the
same search, page-open, and find tools. Hard items receive one strategy-first
helper after eight searches; further review is evidence-triggered rather than a
mandatory full council. A helper must perform an independent falsification
search before finalizing, and its reviews must preserve clue relation types and
provide their own citations. Source support remains a hard gate on normal turns;
the configured final hard-budget turn may return one concrete, answer-type-valid
best effort instead of an empty answer, with the support failure retained in the
private audit trail. Forced-final turns expose only the `final` tool, so a model
cannot escape the bounded finalization phase by selecting another search action.
The generic production-broker adapter remains
available for comparison configs. See
[`STAR_IMPLEMENTATION.md`](STAR_IMPLEMENTATION.md) for search isolation,
external-call budgets, and live verification details.

## Browser backends

- `direct`: default; HTTP/HTML/PDF/JSON/text retrieval without JavaScript.
- `playwright`: optional rendered Chromium backend.
- `auto`: direct retrieval first, Playwright fallback for sparse HTML.

The direct backend validates every redirect target **before** requesting it. Both backends block loopback, private, link-local, multicast, reserved, and unspecified network destinations by default. The Playwright backend applies the same policy to subresource requests.

Install browser support only when needed:

```bash
.venv/bin/pip install -e '.[browser]'
.venv/bin/playwright install chromium
```

## OpenAI-compatible API contract

The evaluated endpoint must accept:

```http
POST <api_base>/chat/completions
Authorization: Bearer <key>
Content-Type: application/json
```

with a body containing at least:

```json
{
  "model": "star",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.0,
  "max_tokens": 8192
}
```

and return an OpenAI-style `choices[0].message` object. Native tool calls are optional; the default JSON-action protocol only requires text generation. See `docs/MODEL_API.md`.

## Run outputs

A run is written under `runs/<run-name>/`:

```text
runs/<run-name>/
├── run.lock.json
├── status.json
├── private/
│   ├── README.md
│   ├── trials.jsonl
│   └── transcripts/
└── public/
    ├── summary.json
    ├── summary.md
    ├── trials.csv
    └── report.html
```

Private artifacts can contain benchmark questions, reference answers, model reasoning, search evidence, and grader responses. Do not publish them. Set `BC250_ARTIFACT_FERNET_KEY` to encrypt transcript files at rest.

Generate a key:

```bash
.venv/bin/python scripts/generate_fernet_key.py
```

Create publication-safe output:

```bash
.venv/bin/bc250 sanitize \
  runs/browsecomp250-headline \
  release/browsecomp250-headline \
  --config configs/headline.yaml
```

The sanitizer decrypts the selected benchmark items only for a local leak scan; it rejects public files containing benchmark questions, reference answers, or the BrowseComp canary prefix.

## Comparing systems

Run every system with:

- the same committed subset;
- the same agent code and prompt;
- the same search provider and configuration;
- the same browser backend;
- the same budgets and timeout;
- the same grader model and prompt;
- the same attempt count; and
- a documented evaluation window.

Then compare runs:

```bash
.venv/bin/bc250 compare \
  runs/star \
  runs/gpt-5-6 \
  runs/fable-5 \
  runs/grok-4-5
```

For small score differences, use paired item-level analysis rather than comparing only overlapping confidence intervals. See `docs/COMPARISON_CAMPAIGNS.md`.

## Documentation map

- `docs/QUICKSTART.md` — installation and first run
- `docs/ARCHITECTURE.md` — components and data flow
- `docs/PROTOCOL.md` — frozen evaluation contract
- `docs/SUBSET_AND_DATA.md` — subset derivation and dataset handling
- `docs/MODEL_API.md` — OpenAI-compatible endpoint integration
- `docs/SEARCH_AND_BROWSER.md` — provider and retrieval behavior
- `docs/GRADING.md` — semantic grading and diagnostics
- `docs/REPRODUCIBILITY.md` — locks, hashes, caches, and provenance
- `docs/REPORTING.md` — metrics and publication language
- `docs/COMPARISON_CAMPAIGNS.md` — fair multi-model comparisons
- `docs/SECURITY_AND_PRIVACY.md` — SSRF, secrets, and benchmark leakage
- `docs/KNOWN_LIMITATIONS.md` — interpretation constraints
- `docs/TROUBLESHOOTING.md` — operational failures
- `docs/VALIDATION.md` — validation performed and remaining live checks

## Development

```bash
make bootstrap
make test
make lint
```

The archive includes exact package versions used for local validation in `requirements-tested.txt`, while `pyproject.toml` retains bounded compatible ranges for normal installation.

## License

This harness is MIT-licensed. BrowseComp and the adapted query/grader templates are attributed in `THIRD_PARTY_NOTICES.md` and remain subject to their upstream license and terms.
