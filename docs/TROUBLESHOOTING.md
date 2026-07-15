# Troubleshooting

## `Configuration root must be a mapping`

Check YAML indentation and environment interpolation. Print the resolved public configuration:

```bash
.venv/bin/bc250 print-config --config configs/headline.yaml
```

## Dataset download fails

Check DNS, proxy, outbound HTTPS, and the source URL. The download is made from the evaluator process, not the model endpoint.

Retry:

```bash
.venv/bin/bc250 prepare --config configs/headline.yaml --force
```

A `.part` file may remain after interruption and can be removed safely.

## Dataset SHA mismatch

Do not bypass immediately. Determine whether:

- the upstream file changed;
- a proxy modified line endings/content;
- the download was truncated; or
- `.env` contains an obsolete digest.

Preserve both snapshots and compare encrypted row hashes. A changed source snapshot makes runs nonidentical.

## Model returns prose instead of JSON

Use `model.protocol: json`, temperature zero, and verify that the endpoint uses the intended instruct/chat template. The agent sends bounded correction prompts, but repeated failures end as `no_final` and count incorrect.

Inspect a private transcript. Do not publish it.

## Endpoint rejects `max_tokens`

Add the alternative field:

```yaml
model:
  extra_body:
    max_completion_tokens: 8192
```

The client removes `max_tokens` when this field is present.

## Endpoint requires custom headers

```dotenv
BC250_MODEL_EXTRA_HEADERS={"X-API-Key":"...","X-Tenant":"..."}
```

Prefer the dedicated API-key field when bearer auth works. Header values are redacted from run locks based on secret-like header names.

## Native tools fail

Switch to:

```yaml
model:
  protocol: json
```

JSON actions require only text chat-completion compatibility and are the recommended common denominator.

## Search credential error

Confirm the selected provider and matching key. Only the selected provider's credential is required.

```bash
.venv/bin/bc250 doctor --config configs/smoke.yaml --live
```

## SearXNG returns HTML

Enable JSON output in SearXNG and confirm:

```bash
curl -sS 'http://127.0.0.1:8080/search?q=test&format=json'
```

Some public SearXNG instances disable JSON or rate-limit automation; use a controlled instance.

## Browser blocks a URL

The destination may resolve to a private/reserved address, use a nonstandard port, include credentials, or redirect unsafely. Do not disable safety globally. Verify the domain independently and use network isolation before adding a narrow exception in a protocol-versioned fork.

## Many pages have little text

Use the optional rendered backend:

```bash
.venv/bin/pip install -e '.[browser]'
.venv/bin/playwright install chromium
```

Then set `browser.backend: auto` or `playwright`. Re-run all compared systems with the same backend.

## `Read-only ... cache miss`

`cache_mode: read` intentionally fails closed. Use `readwrite` to permit live fallback or prepopulate the exact request in the cache.

## Grader parse errors

Inspect private `grading.grader_response`. The grader must emit a line matching:

```text
correct: yes
```

or:

```text
correct: no
```

Increase grader output tokens only if responses are truncated. Keep the change fixed across all systems.

## Run directory conflict

A run directory already contains a different replay hash. Choose a new `run.name` rather than deleting the lock:

```yaml
run:
  name: star-2026-07-15
```

## Resume does not rerun failed trials

The append-only resume policy treats every existing `(item_id, attempt)` record as completed, including failures. This preserves the original attempt count. To retry failures, create a new run/profile and disclose the retry policy; do not silently replace failed headline records.

## Public leak scan fails

The public tree contains text matching a selected question, reference answer, or canary prefix. Remove the leaked content and regenerate reports. Do not weaken the scanner merely to pass release validation.

Short common answers can cause conservative false positives if added to custom public prose. Rephrase or remove that prose rather than publishing answer-like text.
