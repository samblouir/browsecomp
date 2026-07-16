# Fair comparison campaigns

## Objective

A campaign should distinguish three different claims:

1. **Bare-model capability** under an identical external agent scaffold.
2. **Agent-system capability** where each vendor's preferred agent is allowed.
3. **Efficiency frontier** under fixed cost, latency, token, or tool budgets.

This repository is primarily designed for the first and third claims.

## Common-harness model comparison

For Star, GPT-5.6, Fable 5, and Grok 4.5:

- use this same repository commit;
- use `model.protocol: json` for all models unless every endpoint has equivalent native-tool behavior;
- hold the agent prompt and budgets fixed;
- hold the search and browser stack fixed;
- hold the grader fixed;
- use one attempt per item for the headline;
- use the same source CSV digest;
- interleave evaluation periods; and
- report all failures as incorrect.

Each model gets its own config and run name. Do not overwrite one run directory with another.

## Accuracy reporting

Do not use completed-only accuracy as a headline while a batch is still in
flight. Report a fixed denominator:

- `correct / assigned`, with wrong, failed, and pending counts shown separately;
- `strict_first_terminal` for development history, where timeout, error, and
  no-final records are incorrect; and
- `best_observed_after_repair` only as a development ceiling, never as pass@1.

`graded_first_completion` and `graded_latest_completion` deliberately exclude
terminal failures. They are diagnostic views and must not be described as
benchmark accuracy. The campaign summary also reports correct coverage against
all 250 questions so an incomplete run cannot appear complete by shrinking its
denominator.

## Interleaving

Live web evidence changes. Prefer one of:

- round-robin chunks of 10–25 items per model;
- simultaneous runs with matched concurrency; or
- a tightly bounded same-day window.

Avoid completing all 250 items for one model and evaluating another weeks later.

## Provider fairness

Model endpoints can differ in:

- context limits;
- hidden reasoning tokens;
- maximum output;
- caching;
- retry behavior;
- rate limits;
- server-side tools;
- model alias mutability; and
- safety refusals.

Record these differences. Do not silently give one model a larger agent turn/output budget because its API is less reliable.

## Search fairness

Use the same provider account/tier where possible. Search subscriptions can differ in ranking, freshness, request quotas, and result fields.

Keep region and language fixed. A US-region search stack and a European-region stack may return materially different evidence.

## Quality profile

`configs/quality.yaml` uses four attempts and larger browsing budgets. It can estimate a higher-compute frontier, but it is not headline-comparable to the single-attempt profile.

Recommended labels:

- `BrowseComp-250 pass@1` for the headline profile;
- `BrowseComp-250 pass@4` for the quality profile; and
- `BrowseComp-250 accuracy per attempt` for raw multi-attempt trial accuracy.

## Matched-cost studies

To compare at a fixed budget:

1. set input/output prices for each endpoint;
2. run a pilot on 20–30 items;
3. choose budgets that target the same expected cost;
4. freeze those budgets before the full run;
5. report overruns and failed calls; and
6. compare accuracy and latency on the same items.

The harness currently enforces tool/step/time budgets, not a hard dollar cutoff. A hard cost controller can be added as a protocol-versioned extension.

## Matched-latency studies

Use the same task timeout and concurrency. Report both:

- per-task latency distribution; and
- campaign wall-clock time.

A fast model can exploit its speed to perform more sequential research within a wall-clock deadline. Decide whether that is part of the product claim or should be controlled by equal turn/tool budgets.

## Agent-system comparison

When comparing vendor-native systems—such as Claude Code, Cursor, or a proprietary research agent—label the full system:

> Fable 5 + Vendor Agent

Do not place those scores in a chart titled only by model name. Tool availability and orchestration can dominate BrowseComp.

## Campaign manifest

For each run preserve:

- model and endpoint identifier;
- run-lock replay hash;
- source and subset hashes;
- evaluation timestamps;
- search/provider tier;
- browser backend;
- grader model;
- pricing assumptions;
- concurrency;
- outage notes; and
- repository/archive hash.

Use `bc250 compare` only after verifying every run with `bc250 verify-run`.
