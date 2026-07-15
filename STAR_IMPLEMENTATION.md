# Star BrowseComp-250 implementation

This directory is an isolated BrowseComp-250 evaluation workspace. It does not
import or modify the live `frlweb` API server.

## Runtime path

The evaluator sends OpenAI-compatible native tool calls to the deployed Agent
endpoint at `http://127.0.0.1:8000/agent/v1`. The evaluated model is
`frontierrl/star-7`. Search and page retrieval execute in this evaluator, not
inside the model server.

Each benchmark item uses FrontierRL `response-chain-v1`:

1. The root request sends the complete BrowseComp system and user prompt.
2. The endpoint records the exact assistant response and returns an immutable
   response ID.
3. Every continuation sends only the new `tool` result with that response ID.
4. A stable request ID makes transport retries idempotent.

This preserves the backend-owned reasoning/tool history, avoids reconstructed
assistant turns, and keeps each item on a stable KV-affinity lease.

## Star contract

Both `configs/star-smoke.yaml` and `configs/star-headline.yaml` enforce:

- temperature `0.3`;
- `top_p: 0.95`;
- `max_output_tokens: 16384` for every evaluated-model turn;
- native OpenAI tools with one externally executed action per turn;
- a maximum of 48 denoising steps;
- parallel `search_many` and `open_many` actions;
- reproducible Brave API search with batched query fanout;
- selective `ask_external_model` calls, including four-call concurrent batches;
- four concurrent independent reviews automatically attached after eight
  logical searches so difficult tasks cannot silently skip external help;
- automatic inspection of up to four top result pages after each two-search
  phase, with a safe reader fallback for blocked public origins;
- bounded duplicate-action recovery that opens fresh discovered pages before
  requiring a final answer with an unchanged tool schema;
- protocol normalization for unambiguous singular/batch tool-name mismatches;
- ten idempotent model-transport retries so brief Agent endpoint restarts do
  not discard a completed research trajectory;
- remaining-budget clipping for batch actions and one structured external
  finalization rescue if the backend still requests evidence at the hard cap;
- a 900-second wall-clock trigger for a three-review candidate, constraint, and
  falsification council followed by an independent structured adjudicator,
  preserving enough of the 1,800-second task budget for grading;
- bounded wall time and action budgets; and
- durable per-step heartbeat and event logs.

The semantic grader uses `gpt-5.6` with `max_completion_tokens: 16384` and
omits `temperature` because that endpoint rejects every explicit value except
its server default of `1`. The evaluated Star model remains at temperature
`0.3`. This grader transport detail must be disclosed when publishing or
comparing scores.

## Search isolation

The Star smoke, development, and headline profiles fix `search.provider` to
`brave`. They do not open Google searches in a user's Chrome. This keeps the
evaluation reproducible and avoids personal-browser traffic challenges. The
library retains its experimental Chrome and hybrid adapters for isolated
development, but the Star campaign configs cannot select them through an
environment override.

## External consultation

`ask_external_model` is a caller-owned native tool in this evaluator. The Star
development profile fixes every helper request to `frontierrl/star-2` through
the Agent API. Each helper is a real isolated tool agent: it can search Brave,
open pages, find text, take notes, and finalize. It cannot call
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
export BC250_EXTERNAL_AGENT_API_BASE=http://127.0.0.1:8000/agent/v1
export BC250_MODEL_API_KEY='<account-bound-key>'
```

The helper key is redacted from public run locks and recorded only as a short
one-way fingerprint in private reproducibility metadata. Incoming provider or
model suggestions cannot redirect this mode away from Star-2. The older
`mode: broker` adapter remains available to deliberately separate comparison
profiles; it is not selected by `configs/star-dev-baseline.yaml`.

The Star development profile sets `automatic_external_after_search_calls: 8`
and `automatic_external_requests: 1`. Once per item, the controller runs one
combined candidate investigator that also performs a minimal-pair adversarial
check. The parent can request another focused helper only when a concrete
contradiction, identity ambiguity, or answer-type dispute remains. This avoids
launching four overlapping full browsing agents on routine items. The hard
per-item helper budget is four. Helper output is embedded in the current search
tool result, preserving the same assistant/tool continuation and stable
response-chain history. Public URLs proposed by the helper are opened
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
