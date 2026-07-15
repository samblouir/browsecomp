# Architecture

## Component overview

```text
Official encrypted CSV
        │
        ▼
Dataset validator ── frozen source-row indices ──► 250 in-memory items
        │
        ▼
Concurrent benchmark engine
        │
        ├──► OpenAI-compatible model client
        │       │
        │       ▼
        │    autonomous action loop
        │       ├── search / search_many
        │       ├── open / open_many
        │       ├── find
        │       ├── ask_external_model (up to four concurrent requests)
        │       ├── note
        │       └── final
        │
        ├──► Search adapter
        │       ├── Brave
        │       ├── Google in the user's existing Chrome (experimental only)
        │       ├── Hybrid Google + Brave (experimental only)
        │       ├── Tavily
        │       ├── Serper
        │       └── SearXNG
        │
        ├──► External-help backend
        │       ├── isolated Star-2 tool agents for Star campaign configs
        │       ├── generic production-broker adapter for comparison configs
        │       └── one strategy-first review with evidence-triggered escalation
        │
        ├──► Page fetcher
        │       ├── direct HTTP
        │       └── Playwright Chromium
        │
        └──► Grader
                ├── BrowseComp semantic LLM grader
                └── strict deterministic diagnostic grader

        ▼
Append-only private records + transcripts
        │
        ▼
Aggregation, confidence intervals, calibration, cost/latency summaries
        │
        ▼
Publication-safe output and leak scan
```

## Package layout

```text
src/browsecomp250/
├── agent/       autonomous browsing policy and budgets
├── browser/     safe retrieval, extraction, PDF support
├── grading/     semantic and deterministic scoring
├── llm/         OpenAI-compatible client and action protocols
├── report/      aggregation, HTML/CSV/JSON, sanitization
├── run/         orchestration, locking, resumability, storage
├── search/      search-provider adapters
├── agent_external.py  Star-2 tool-agent external-help backend
├── external.py  production external-model broker client
├── cache.py     SQLite request/response cache
├── cli.py       operator interface
├── config.py    validated YAML configuration
├── crypto.py    compatibility decryption for the official dataset
├── dataset.py   download, validation, decryption, manifests
├── subset.py    frozen seed-0 subset validation
└── types.py     internal typed records
```

## Trust boundaries

### Trusted local control plane

- committed code;
- committed configuration;
- frozen subset index file;
- local environment variables;
- run lock and hashes.

### Untrusted external inputs

- model responses;
- grader responses;
- search API responses;
- page URLs and redirect targets;
- HTML, PDF, JSON, and text page bodies;
- HTTP headers;
- cached external responses; and
- the downloaded benchmark file until validated.

No model-produced string is executed as shell code. The model has only the enumerated browsing actions implemented in `AgentRunner`.

## Agent protocols

### JSON-action protocol

The default protocol requests exactly one JSON object per model turn. It is broadly compatible with OpenAI-style servers because no native tool-call feature is required.

Example:

```json
{"action":"search","query":"rare historical clue","count":10}
```

### Native tool protocol

When `model.protocol: tools`, the client sends OpenAI function-tool schemas. The first returned tool call is executed. This mode is appropriate only when the evaluated endpoint has reliable OpenAI-compatible tool-call semantics.

### Auto protocol

`auto` initially requests native tools and falls back to JSON actions if the API call fails. It is convenient for development but introduces an adaptive protocol difference. Prefer a fixed `json` or `tools` value for a published comparison.

## History management

The agent retains the complete interaction until `agent.max_history_chars` is exceeded. It then performs deterministic compaction:

- preserves the system prompt and original question;
- keeps recent tool interactions;
- retains explicit model notes;
- records titles, URLs, and content hashes for recently opened pages; and
- instructs the model to reopen pages when needed.

No model-based summarizer is used, avoiding an unreported secondary-model dependency.

## Storage model

`private/trials.jsonl` is append-only and fsynced after each trial. This limits data loss after interruption and supports resumption.

Each trial has a stable key `(item_id, attempt)`. Existing records are skipped on resume. A run directory cannot be reused with a different replay hash.

Private transcripts can be plaintext JSON or Fernet-encrypted JSON. Public reports never require transcript decryption.
