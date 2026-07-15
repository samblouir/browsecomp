# Grading

## Headline semantic grader

The headline profile uses the semantic grading prompt from OpenAI's BrowseComp reference evaluator. It receives:

- the original question;
- the model's complete formatted response; and
- the precise reference answer.

It is instructed to extract the final answer, compare it only to the reference, explain meaningful differences, and emit:

```text
extracted_final_answer: ...
reasoning: ...
correct: yes|no
confidence: ...
```

The harness parses `correct: yes|no` case-insensitively on its own line. Missing or malformed output defaults to incorrect and records a parse error.

## Why an LLM grader is needed

BrowseComp answers can differ without being semantically different:

- alternate formatting;
- abbreviations;
- titles with or without articles;
- numeric formatting;
- reordered equivalent names;
- small numeric tolerances; and
- answers embedded in a longer final response.

A strict exact-match scorer would create avoidable false negatives. The semantic grader is therefore the primary score.

## Fixed-grader requirement

A model comparison must hold constant:

- grader provider;
- exact model identifier or snapshot;
- API endpoint;
- prompt template;
- temperature;
- output limit;
- request extras;
- parser version; and
- grading date/window.

A changed grader produces a changed measurement instrument. Regrade all compared outputs together when switching graders.

## Grader independence

For the cleanest comparison, the grader should not be the same endpoint instance as the model being evaluated. At minimum, do not allow a model to grade its own output with hidden state or access to its trajectory.

The grader receives no search transcript and no citations beyond those already present in the final response. It judges answer equivalence, not research quality.

## Deterministic diagnostic grader

The deterministic mode:

- extracts the `Exact Answer:` line;
- Unicode-normalizes and case-folds text;
- removes English articles;
- normalizes punctuation and whitespace;
- treats thousands separators consistently; and
- permits a narrow numeric tolerance.

Use it for:

- local smoke tests;
- scorer regression tests;
- identifying obvious semantic-grader disagreements; and
- low-cost development.

Do not use deterministic accuracy as the public BrowseComp-250 headline unless explicitly labeled as a nonstandard exact/normalized-match score.

## Dual mode

```yaml
grader:
  mode: both
```

Dual mode retains the semantic result as authoritative and appends the deterministic result as a diagnostic note. This is useful for auditing grader behavior.

## Grader API configuration

```yaml
grader:
  mode: official_llm
  api_base: https://api.openai.com/v1
  api_key: ${BC250_GRADER_API_KEY:-}
  model: gpt-5.6
  temperature: 0.0
  max_output_tokens: 1200
  timeout_seconds: 120
  max_retries: 4
  extra_headers_json: {}
  extra_body: {}
```

Token prices can be configured separately so grader cost is included in total run cost.

## Failure policy

If the evaluated model produced a response but grading fails, the trial record has no positive grade and counts as incorrect in aggregate accuracy. This conservative policy prevents service failures from increasing the score by shrinking the denominator.

Preserve grader outputs in private artifacts for later regrading. A future extension can regrade stored model outputs without rerunning browsing; the current CLI intentionally keeps the primary path simple and auditable.

## Grader audit

Before publication, manually inspect a stratified sample:

- at least 20 semantic positives;
- at least 20 semantic negatives;
- all deterministic/semantic disagreements;
- all grader parse errors;
- all very low-confidence positive answers; and
- all answers with unusual formatting.

Report the audit sample size and any adjudication policy. Do not silently hand-correct only the evaluated model while leaving competitor outputs untouched.
