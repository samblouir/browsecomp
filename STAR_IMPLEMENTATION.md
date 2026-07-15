# Star BrowseComp-250 implementation

This directory is an isolated BrowseComp-250 evaluation workspace. It does not
import or modify the live `frlweb` API server.

## Runtime path

The evaluator sends OpenAI-compatible native tool calls through the shared
router at `http://127.0.0.1:8003/v1`. The evaluated model is
`frontierrl/star-7`. Search and page retrieval execute in this evaluator, not
inside the model server.

Every benchmark item derives a distinct, stable `X-FRL-Conversation-Id` from
its run/item/attempt namespace and reuses it for every model turn. Star-2 helper
branches derive their own IDs from the parent namespace and helper index. This
preserves KV affinity while allowing independent items and branches to spread
across the fleet. The direct router receives normal full OpenAI message history;
the Agent-endpoint-only response-chain extension is disabled by default. It can
be explicitly re-enabled only when an endpoint implementing that extension is
selected.

## Star contract

Both `configs/star-smoke.yaml` and `configs/star-headline.yaml` enforce:

- temperature `0.3`;
- `top_p: 0.95`;
- `max_output_tokens: 16384` for every evaluated-model turn;
- native OpenAI tools with one externally executed action per turn;
- a maximum of 48 denoising steps;
- parallel `search_many` and `open_many` actions;
- a protocol-locked search provider with batched query fanout;
- selective `ask_external_model` calls, with one strategy-first Star-2 helper
  attached only after eight logical searches and later calls reserved for a
  concrete evidence dispute;
- automatic inspection of up to four top result pages after each two-search
  phase, with a safe reader fallback for blocked public origins;
- bounded duplicate-action recovery that opens fresh discovered pages before
  requiring a final answer with an unchanged tool schema;
- protocol normalization for unambiguous singular/batch tool-name mismatches;
- ten model-transport retries so brief router or worker disruptions do
  not discard a completed research trajectory;
- remaining-budget clipping for batch actions and one structured external
  finalization rescue only after two rejected forced-final turns at the hard cap;
- no wall-clock-triggered council: elapsed time alone is not evidence that more
  reviewers will improve the answer;
- bounded wall time and action budgets; and
- durable per-step heartbeat and event logs.

The semantic grader uses `gpt-5.6` with `max_completion_tokens: 16384` and
omits `temperature` because that endpoint rejects every explicit value except
its server default of `1`. The evaluated Star model remains at temperature
`0.3`. This grader transport detail must be disclosed when publishing or
comparing scores.

## Search isolation

The Star smoke, development, and headline profiles default `search.provider` to
`brave` and accept an explicit `BC250_SEARCH_PROVIDER` override. The resolved
provider is frozen in the run lock. They do not open Google searches in a
user's Chrome. This avoids personal-browser traffic challenges. If a campaign
changes providers because of an upstream outage, its final report must preserve
per-row provenance and present the resulting provider strata separately.

## External consultation

`ask_external_model` is a caller-owned native tool in this evaluator. The Star
development, smoke, and headline profiles fix every helper request to
`frontierrl/star-2` through the shared router. Each helper is a real isolated tool
agent: it can use the configured search provider, open pages, find text, take notes, and finalize. It cannot call
`ask_external_model` recursively. The tool accepts either one `query` or
`requests` containing up to four independent queries; batched requests run
concurrently. Defaults are temperature `0.7`, top-p `0.95`, 16,384 output
tokens per turn, and at most 48 denoising steps.

External answers are inserted as a normal tool result and retained in the
private benchmark transcript. Star is instructed to treat them as independent
advice and verify factual claims against browsed evidence. Calls have a separate
per-task budget and request IDs are included in audit output. Configure:

```bash
export BC250_EXTERNAL_MODEL_ENABLED=true
export BC250_EXTERNAL_AGENT_API_BASE=http://127.0.0.1:8003/v1
export BC250_MODEL_API_KEY='<account-bound-key>'
```

The helper key is redacted from public run locks and recorded only as a short
one-way fingerprint in private reproducibility metadata. Incoming provider or
model suggestions cannot redirect this mode away from Star-2. The older
`mode: broker` adapter remains available to deliberately separate comparison
profiles; it is not selected by `configs/star-dev-baseline.yaml`.

The Star profiles set `automatic_external_after_search_calls: 8`
and `automatic_external_requests: 1`. Once per item, the controller runs one
strategy-first independent investigator that identifies candidate entities,
performs a minimal-pair adversarial check, and returns discriminating search
routes. The controller executes those routes and opens their evidence. The
parent can request another focused helper only when a concrete
contradiction, identity ambiguity, or answer-type dispute remains. This avoids
launching four overlapping full browsing agents on routine items. The hard
per-item helper budget is three: one strategy helper, then at most one reviewer
and one adjudicator after a forced-final failure. Helper output is embedded in
the current search tool result, preserving the normal assistant/tool
continuation on the parent chain. Public URLs proposed by the helper are opened
automatically and attached for factual checking. A helper failure does not
discard successful search evidence, and exhausting help does not force the main
agent to finalize while browsing budget remains.

Semantic duplicate searches are rejected locally with route-change guidance.
They do not launch a Star-2 strategy agent in this profile; a full browsing
helper is reserved for evidence investigation or an explicit unresolved
conflict.

After every two consecutive successful search actions, the controller opens up
to four top URLs in round-robin query order. Failed or sparse direct HTTP
retrieval falls back to the configured public text reader after validating the
original URL against the same SSRF policy. The original or actually resolved
source URL remains the citation identity; the reader URL is never presented as
the source. This supplies page evidence without changing the model-visible tool
schema or creating a synthetic conversation turn.

## Commands

```bash
./scripts/bootstrap.sh
./scripts/prepare_star.sh
./scripts/run_star_smoke.sh 1
./scripts/run_star_headline.sh
```

Live progress is written to `runs/<name>/status.json`. Full private event logs
are in `runs/<name>/private/events.jsonl`; one durable trial record is appended
to `private/trials.jsonl` after each item. A detached run can be supervised with:

```bash
.venv/bin/python scripts/watch_star_run.py runs/browsecomp250-star7-agent-20260715 \
  --pid <runner-pid> --stale-seconds 600
```

The watcher exits nonzero if progress goes stale or the runner dies before a
terminal state. The benchmark itself remains resumable under the replay lock.
