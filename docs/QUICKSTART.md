# Quick start

## 1. System requirements

The validated baseline is:

- Linux or macOS;
- Python 3.12 or 3.13;
- at least 4 GB of local disk for environments, caches, logs, and optional browser binaries;
- outbound HTTPS access;
- a model API implementing OpenAI-compatible chat completions;
- a web-search backend; and
- a semantic grader endpoint for headline runs.

No GPU is required on the evaluator host when the model is served remotely.

## 2. Install

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools
.venv/bin/pip install -e '.[dev]'
```

Or use:

```bash
./scripts/bootstrap.sh
```

Validate installation:

```bash
.venv/bin/bc250 --version
.venv/bin/bc250 subset
.venv/bin/pytest
```

## 3. Configure the model endpoint

```bash
cp .env.example .env
```

Minimum model settings:

```dotenv
BC250_MODEL_API_BASE=http://127.0.0.1:8000/v1
BC250_MODEL_API_KEY=replace-me
BC250_MODEL_NAME=star
```

The base URL may also point directly to `/chat/completions`. The client normalizes either form.

For a no-auth local service:

```dotenv
BC250_MODEL_API_KEY=
BC250_ALLOW_EMPTY_MODEL_API_KEY=true
```

Custom headers can be supplied as JSON:

```dotenv
BC250_MODEL_EXTRA_HEADERS={"X-Tenant":"frontierrl","X-Route":"star-prod"}
```

Provider-specific body fields belong under `model.extra_body` in a copied YAML configuration:

```yaml
model:
  extra_body:
    reasoning_effort: high
    chat_template_kwargs:
      enable_thinking: true
```

Do not change model request fields between compared systems unless the variation is the subject of the experiment.

## 4. Configure search

### Brave

```dotenv
BC250_SEARCH_PROVIDER=brave
BC250_BRAVE_API_KEY=replace-me
```

### Tavily

```dotenv
BC250_SEARCH_PROVIDER=tavily
BC250_TAVILY_API_KEY=replace-me
```

### Serper

```dotenv
BC250_SEARCH_PROVIDER=serper
BC250_SERPER_API_KEY=replace-me
```

### SearXNG

```dotenv
BC250_SEARCH_PROVIDER=searxng
BC250_SEARXNG_BASE_URL=http://127.0.0.1:8080
```

SearXNG must enable JSON output. It should be isolated from unrelated users so engine configuration and rate limiting remain fixed during a campaign.

### Server-side Yahoo

```dotenv
BC250_SEARCH_PROVIDER=yahoo
```

This keyless adapter parses Yahoo's server-rendered organic results and unwraps
Yahoo redirect URLs before exposing them to the agent. It does not launch or
control a user's browser. Treat it as a distinct provider in protocol locks and
result reports; do not mix its rows into a Brave-only headline claim.

## 5. Configure grading

The smoke profile uses the deterministic diagnostic grader. The headline profile requires the semantic LLM grader.

```dotenv
BC250_GRADER_API_BASE=https://api.openai.com/v1
BC250_GRADER_API_KEY=replace-me
BC250_GRADER_MODEL=gpt-5.6
```

A different OpenAI-compatible grader is supported, including custom headers and no-auth endpoints:

```dotenv
BC250_GRADER_EXTRA_HEADERS={}
BC250_ALLOW_EMPTY_GRADER_API_KEY=false
```

Disclose the exact grader identifier and evaluation date. Changing the grader can change scores even when model outputs are unchanged.

## 6. Prepare and pin the dataset

```bash
.venv/bin/bc250 prepare --config configs/headline.yaml
```

The command:

1. downloads the official encrypted CSV;
2. verifies that it has 1,266 rows and the required columns;
3. computes its SHA-256;
4. validates the frozen subset indices;
5. writes a manifest containing only encrypted-row hashes; and
6. prints the SHA-256 to pin.

Add the printed digest to `.env`:

```dotenv
BC250_EXPECTED_DATASET_SHA256=<sha256>
```

Re-run preparation. A mismatch now fails closed.

## 7. Validate service connectivity

Static checks:

```bash
.venv/bin/bc250 doctor --config configs/smoke.yaml
```

Live checks:

```bash
.venv/bin/bc250 doctor --config configs/smoke.yaml --live
```

The live mode calls the model, performs one search, and fetches one public page. It consumes API quota.

## 8. Run a staged evaluation

One item:

```bash
.venv/bin/bc250 run --config configs/smoke.yaml --limit 1
```

Five items:

```bash
.venv/bin/bc250 run --config configs/smoke.yaml --limit 5
```

A separate run name should be used when changing any protocol-affecting setting. Existing run directories are resumable only when their replay hash matches.

## 9. Run the headline profile

Dry run:

```bash
.venv/bin/bc250 headline --config configs/headline.yaml --dry-run
```

Full run:

```bash
.venv/bin/bc250 headline --config configs/headline.yaml --yes
```

The default profile schedules 250 trials with concurrency four. Concurrency affects web timing, search rate limits, and model batching; keep it fixed across compared models.

## 10. Validate and release results

```bash
.venv/bin/bc250 verify-run runs/browsecomp250-headline \
  --config configs/headline.yaml

.venv/bin/bc250 sanitize \
  runs/browsecomp250-headline \
  release/browsecomp250-headline \
  --config configs/headline.yaml
```

Review the sanitized directory manually before publication. Never upload `runs/<name>/private/`.
