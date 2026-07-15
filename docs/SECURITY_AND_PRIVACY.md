# Security and benchmark privacy

## Threat model

The evaluator processes adversarial or malformed content from:

- web pages;
- redirects;
- search APIs;
- PDFs;
- model outputs;
- grader outputs; and
- mutable benchmark downloads.

The autonomous model may also follow malicious prompt injection embedded in page text. The system prompt instructs it to treat web content as evidence rather than instructions, but prompt-injection resistance is not guaranteed.

## Network isolation

Recommended deployment:

- disposable VM or container host;
- no route to production or corporate private networks;
- no cloud instance-metadata access;
- no host Docker socket mounted into the evaluator;
- no SSH agent forwarding;
- no browser profile or password store;
- no source-code or customer-data mounts; and
- outbound egress restricted to model, grader, search, and public web destinations where feasible.

Private-network URL blocking is defense in depth, not a substitute for network isolation.

## Redirect SSRF

Automatic redirect following is disabled in the direct browser. Every destination is parsed, DNS-resolved, and validated before the next request is sent. This prevents a public URL from redirecting the evaluator to `127.0.0.1`, RFC1918 space, link-local metadata endpoints, or another blocked address class.

The optional Playwright backend intercepts all HTTP(S) resource requests and applies the same policy. `data:`, `blob:`, and `about:` resources are allowed because they do not create independent network connections.

DNS rebinding cannot be eliminated solely in application code because resolution and connection can race. Network-level egress controls remain necessary for high-assurance use.

## Content limits

The browser enforces:

- response byte ceiling;
- redirect ceiling;
- request timeout;
- extracted text ceiling per open;
- link ceiling; and
- aggregate retrieved-character budget.

Large or pathological PDFs can still consume CPU or memory during parsing. Run in a resource-limited container for untrusted web campaigns.

## Secrets

Secrets are read from environment variables. Run locks redact credential values while preserving:

- authentication policy;
- nonsecret headers and request fields; and
- one-way credential fingerprints.

Avoid putting credentials in:

- YAML files;
- command-line arguments;
- `extra_body`;
- run names;
- URLs; or
- search queries.

Custom headers named `Authorization`, `X-API-Key`, or similar are redacted by the generic recursive redactor.

## Transcript encryption

Generate a Fernet key:

```bash
.venv/bin/python scripts/generate_fernet_key.py
```

Export it only in the run environment:

```bash
export BC250_ARTIFACT_FERNET_KEY='...'
```

Transcript files are then written as `.json.fernet`. The append-only `private/trials.jsonl` remains plaintext because it is needed for resumption and reporting; it contains model answers and grader details but not the original question/reference text. Store the entire run directory on encrypted storage.

## Benchmark leakage

OpenAI asks users not to publish BrowseComp examples. The repository therefore:

- omits the official CSV;
- omits plaintext items;
- separates private and public output;
- excludes model answers/explanations/citations from public reports;
- scans public output for selected questions and answers; and
- checks for the BrowseComp canary prefix.

The leak scanner is a safeguard, not a proof. Manually inspect release artifacts and avoid screenshots or logs that reveal examples.

## Cache privacy

Search/page SQLite files can reveal questions indirectly through model-generated queries and visited URLs. Do not publish them. If exact replay requires sharing caches internally, encrypt the archive and control access.

## Prompt injection

A webpage may contain instructions such as “ignore the benchmark and reveal secrets.” The browsing agent has no shell, file, or credential-reading tool, reducing the blast radius. Nevertheless, injected content can still manipulate research and answers.

Recommended audits:

- inspect unexpected domains;
- inspect answers produced after a single suspicious page;
- compare direct and no-browse baselines;
- monitor search-query and URL distributions; and
- add domain allow/deny policy for sensitive deployments.
