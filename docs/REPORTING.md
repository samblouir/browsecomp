# Metrics and reporting

## Primary metric

Headline accuracy is:

```text
number of trials graded correct / number of attempted trials
```

Every timeout, service error, missing final answer, or grader failure contributes zero. The report exposes `n_graded` and `n_ungraded` so infrastructure problems remain visible.

## Confidence intervals

The harness reports:

- a Wilson score interval for binomial accuracy; and
- a nonparametric bootstrap interval over trial outcomes.

For a one-attempt 250-item run, these intervals quantify sampling uncertainty over the fixed subset's observed outcomes. They do not capture:

- benchmark contamination;
- web/search drift;
- grader uncertainty;
- endpoint nondeterminism;
- subset selection bias; or
- measurement changes from the harness.

## Multiple attempts

For attempts greater than one:

- `accuracy` remains per-attempt accuracy over all trial records;
- `pass_at_k` reports the fraction of items with at least one correct result among attempts `1..k`.

Do not present pass@4 as if it were a one-attempt result. Name the added test-time compute.

## Secondary metrics

The report includes:

- answer rate;
- error count;
- input, output, and cached tokens when available;
- model and grader cost;
- cost per correct trial;
- total, mean, median, p90, and p95 trial duration;
- total and mean search calls;
- total and mean page opens;
- Brier score; and
- 10-bin expected calibration error.

Confidence is taken from the evaluated model's final action and bounded conceptually to 0–100%. The harness records unusual values; operators should inspect them.

## Public output

`public/summary.json` is the canonical machine-readable aggregate.

`public/trials.csv` contains only:

- stable item identity;
- subset/source position;
- attempt;
- model identifier;
- status;
- correctness;
- confidence;
- latency;
- tool counts;
- token counts; and
- cost.

It intentionally omits:

- question text;
- reference answer;
- predicted answer;
- explanation;
- citations and URLs;
- grader reasoning; and
- transcript.

## Recommended headline

> Star achieves **X.X% (Y/250)** on **BrowseComp-250**, a fixed seed-0 250-item subset evaluated with the common FrontierRL browsing harness, Brave Search, direct HTTP retrieval, one attempt per item, and a fixed GPT-5.6 semantic grader. The 95% Wilson interval is **[L%, U%]**. Evaluation ran from **DATE/TIME UTC** to **DATE/TIME UTC**.

Include a footnote:

> BrowseComp-250 is a custom subset, not an official OpenAI split, and is not directly comparable to published full-set BrowseComp scores. Search, browser, grader, and test-time-compute choices affect results.

## Cost/latency headline

For FrontierRL's likely differentiation, pair accuracy with:

- total cost;
- cost per attempted item;
- cost per correct item;
- median task latency;
- p95 task latency; and
- aggregate wall-clock duration at stated concurrency.

A Pareto plot is usually more informative than a single accuracy bar when systems trade additional searches, pages, or model calls for small gains.

## Comparing two runs

The CLI summary comparison is descriptive:

```bash
bc250 compare runs/model-a runs/model-b
```

For a formal paired comparison:

```bash
bc250 paired-compare runs/model-a runs/model-b --output paired.json
```

The command joins public trial rows on `(source_index, attempt)` and reports:

- model A correct / model B wrong;
- model A wrong / model B correct;
- exact two-sided McNemar p-value;
- paired bootstrap interval for the accuracy difference;
- cost/latency deltas on common items; and
- protocol mismatch warnings derived from both run locks.

Do not treat a larger point estimate as sufficient statistical evidence.
