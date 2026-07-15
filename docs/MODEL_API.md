# OpenAI-compatible model API

## Required endpoint

The default configuration expects:

```text
<api_base>/chat/completions
```

For example:

```text
http://127.0.0.1:8000/v1/chat/completions
```

`api_base` may be either the `/v1` base or the complete `/chat/completions` URL.

## Request shape

The client sends an OpenAI-style request:

```json
{
  "model": "star",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.0,
  "max_tokens": 8192
}
```

When `model.extra_body` contains `max_completion_tokens`, the client removes `max_tokens`. This supports servers that implement the newer field instead.

Optional native-tool mode additionally sends:

```json
{
  "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
  "tool_choice": "auto"
}
```

## Response shape

Minimum accepted response:

```json
{
  "model": "served-model-id",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "{\"action\":\"search\",\"query\":\"...\"}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1000,
    "completion_tokens": 100
  }
}
```

The parser also accepts content-block lists and `input_tokens`/`output_tokens` usage names. Cached prompt tokens are read from `prompt_tokens_details.cached_tokens` or `input_tokens_details.cached_tokens` when present.

## Authentication

Default:

```http
Authorization: Bearer <BC250_MODEL_API_KEY>
```

Additional headers:

```dotenv
BC250_MODEL_EXTRA_HEADERS={"X-Tenant":"frontierrl"}
```

An empty API key is rejected by the headline validator unless explicitly enabled:

```dotenv
BC250_ALLOW_EMPTY_MODEL_API_KEY=true
```

This prevents accidental unauthenticated production runs while supporting local vLLM-compatible servers.

## Provider-specific request fields

Copy a configuration and add fields under `extra_body`:

```yaml
model:
  extra_body:
    reasoning_effort: high
    top_p: 0.95
    repetition_penalty: 1.0
```

Fields are passed after the standard body and can override it. The resolved, redacted configuration is stored in `run.lock.json`.

Do not pass credentials through `extra_body`; use environment variables and headers.

## JSON-action protocol

Recommended for broad compatibility:

```yaml
model:
  protocol: json
```

The model must return exactly one of the action objects defined in `src/browsecomp250/llm/protocol.py`. The parser tolerates a fenced JSON object or a single embedded object, but invalid actions trigger a bounded correction loop.

The system prompt requires no prose outside the action object. This minimizes accidental schema violations while keeping the endpoint contract simple.

## Native tool protocol

```yaml
model:
  protocol: tools
```

Use only after validating:

- the endpoint accepts OpenAI `tools` and `tool_choice`;
- tool calls appear in `choices[0].message.tool_calls`;
- function arguments are valid JSON;
- assistant/tool message history is accepted; and
- the endpoint does not silently convert tool calls to text.

Only the first tool call in a response is executed. This keeps one model turn equivalent to one agent decision.

## Auto protocol

```yaml
model:
  protocol: auto
```

The agent tries native tools and falls back to JSON actions after a model API failure. This is useful during integration but should not be the default for cross-model headline comparisons because different models may receive different effective protocols.

## Retry policy

Retries occur for:

- timeouts;
- network errors;
- HTTP 408, 409, 425, and 429; and
- HTTP 5xx.

Nonretryable 4xx responses fail immediately. Exponential backoff respects numeric `Retry-After` when present.

Retries are transport retries for the same model turn; they are not counted as additional benchmark attempts. Report repeated infrastructure failure rates.

## Cost accounting

Set model list prices or internal marginal prices:

```yaml
model:
  input_price_per_million: 0.10
  output_price_per_million: 0.50
```

The client computes cost from returned usage. If the endpoint omits usage, token and cost figures will be zero and must be labeled unavailable rather than truly free.

## Endpoint preflight

```bash
.venv/bin/bc250 doctor --config configs/smoke.yaml --live
```

Also inspect `/models` manually when supported:

```bash
curl -sS \
  -H "Authorization: Bearer $BC250_MODEL_API_KEY" \
  "$BC250_MODEL_API_BASE/models"
```

The benchmark itself does not require `/models`.

## Common vLLM setup

A typical vLLM endpoint exposes `/v1/chat/completions`. Ensure the served model name exactly matches `BC250_MODEL_NAME`, or use a router alias. The model must follow the JSON-action schema reliably; a correct chat template and reasoning-output handling are usually more consequential than sampling microparameters.
