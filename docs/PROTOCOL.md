# BrowseComp-250 protocol

## Purpose

This document defines the result-comparability contract. Any material departure must be named in the result and should use a new run profile or benchmark label.

## Dataset and subset

- Source benchmark: BrowseComp.
- Source size expected by the harness: 1,266 rows.
- Custom subset size: 250 rows.
- Selection procedure: `random.Random(0).sample(examples, 250)` in source-row order.
- Committed index hash: `b0c3334bf37a9ee9eb653639daac477576bce36ec7bcfc5e3ec8ef88c168f4f0`.
- Source CSV SHA-256: pinned by the operator after controlled download.
- Plaintext items: never committed or placed in public reports.

## Unit of evaluation

A trial consists of one model attempting one selected BrowseComp item under a fixed agent, search, browser, budget, and grader configuration.

The headline profile uses exactly one attempt per item. Multiple-attempt profiles are research variants and must report both per-attempt accuracy and item-level pass@k.

## Model interface

- Endpoint: OpenAI-compatible chat completions.
- Star protocol: native OpenAI-compatible tool calls with a stable tool schema
  across response-chain continuations.
- Star temperature: 0.3.
- Star maximum completion tokens per model call: 16,384.
- No hidden human intervention.
- No access to the reference answer.
- No access to the benchmark CSV or run-private artifacts through tools.
- Response-chain transport retries reuse the same request ID; Star profiles
  permit ten retries so connection outages do not duplicate a model turn.

## Browsing tools

The model can invoke only:

- `search(query, count)`;
- `search_many(queries, count)`;
- `open(url, offset, max_chars)`;
- `open_many(urls, offset, max_chars)`;
- `find(url, pattern)`;
- `ask_external_model(query, ...)` or one call with up to four concurrent requests;
- `note(text)`; and
- `final(explanation, exact_answer, confidence, citations)`.

The model cannot execute shell commands, arbitrary Python, code interpreters, file reads, or browser-profile actions.

## Headline budgets

| Budget | Default |
|---|---:|
| Agent turns | 80 |
| Search calls | 80 |
| Page opens | 100 |
| Find calls | 80 |
| Retrieved characters | 2,000,000 |
| Retained history characters | 500,000 |
| Batch size | 7 |
| Per-task wall-clock timeout | 1,800 seconds |

After eight logical searches, the Star profiles attach four concurrent
external reviews to the current search result when the model has not already
requested external help. These are candidate-generation, adversarial-audit,
search-strategy, and independent final-review roles. The controller opens public
source URLs from those reviews and also inspects up to four top result pages
after each two-search phase. They are not treated as ground truth, do not have
access to private benchmark artifacts, and must be verified with web evidence.
This scaffold is part of the reported evaluation protocol.

If Star immediately repeats an identical search action, the controller does
not spend the search budget on a redundant request. It opens fresh, previously
uninspected URLs from the latest successful result batch. After three
consecutive identical actions it requires the final tool while retaining the
same complete tool schema, preventing deterministic retry loops without
changing the model-visible capabilities mid-chain.

Batch actions that would only slightly cross a hard budget are clipped to the
remaining allowance rather than rejected wholesale. The Star profiles permit
80 logical searches. If the backend still emits non-final research requests on
two turns after the true hard cap, one independent external finalizer receives
the accumulated audited evidence and must return the standard final-action
schema. This bounded fallback is disclosed as part of the evaluation scaffold.
The same bounded finalization path triggers after 900 seconds if the main
research loop has still not produced a final action. When four external-call
slots remain, three concurrent reviewers build a candidate matrix, falsify
unsupported assumptions, and independently solve the task; a fourth call
adjudicates those reviews against direct evidence. This leaves half of the task
budget for finalization, transport retries, and grading.

`headline` enforces lower bounds rather than exact values to permit endpoint-specific output settings, but comparable campaigns should use identical values.

## Search contract

Freeze and report:

- provider and product tier;
- region/country;
- language;
- safe-search mode;
- results per call;
- evaluation dates and timezone;
- cache mode;
- provider-side answer or reranking features, if any; and
- any rate-limit-induced failures.

Search results are part of the evaluated agent system. A bare-model claim is valid only when every model uses the same external scaffold.
The Star campaign profiles use the Brave Search API and do not use a personal
Chrome session.

## Browser contract

Freeze and report:

- backend (`direct`, `playwright`, or `auto`);
- user agent;
- timeout;
- redirect limit;
- response byte limit;
- text extraction limit;
- link limit;
- cache mode; and
- private-network policy.

Changing from direct HTTP to rendered browsing may materially alter results.

## Final-answer contract

The model's final action is transformed to:

```text
Explanation: ...
Exact Answer: ...
Confidence: N%
```

This is passed to the semantic grader with the item question and reference answer. Citations are retained for auditing but are not independently fact-checked by the standard grader.

## Grading contract

Headline runs use the BrowseComp reference semantic-grader prompt. Fix:

- grader endpoint;
- grader model/version;
- temperature;
- output-token limit;
- custom request fields;
- evaluation date; and
- parser version.

The current Star profiles use `gpt-5.6` with
`max_completion_tokens: 16384`. They omit the grader `temperature` field
because the live endpoint accepts only its server default of `1`; this is
separate from the evaluated Star model, which runs at temperature `0.3`.

A missing or malformed `correct: yes|no` field defaults to incorrect.

The deterministic grader is a diagnostic only. It recognizes strict normalized string equivalence and narrow numeric equivalence; it is not a substitute for semantic grading.

## Denominator policy

Every attempted trial is included in headline accuracy. The following count as incorrect:

- wrong answer;
- empty answer;
- missing final action;
- task timeout;
- model API failure;
- search failure that prevents completion;
- browser failure that prevents completion;
- grader failure or unparsable grader result; and
- any other trial exception.

The report separately exposes `n_graded`, `n_ungraded`, and error counts.

## Statistical reporting

At minimum report:

- correct count and denominator;
- accuracy;
- 95% Wilson interval;
- bootstrap interval;
- answer rate;
- errors and timeouts;
- total/median latency;
- total input/output tokens;
- total cost and cost per correct answer; and
- search/page usage.

For paired model comparisons, use item-level paired outcomes and a paired bootstrap or McNemar-style analysis. Do not infer a significant lead solely because one point estimate is larger.

## Result label

Recommended:

> **BrowseComp-250 (seed-0 fixed subset, n=250; one attempt/item; common FrontierRL browsing harness; [search provider]; [browser backend]; evaluated YYYY-MM-DD to YYYY-MM-DD): X.X% (Y/250), 95% Wilson CI [L, U].**

Not recommended:

> BrowseComp: X.X%
