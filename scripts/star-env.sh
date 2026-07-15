#!/usr/bin/env bash

set -euo pipefail

export BC250_MODEL_API_BASE="${BC250_MODEL_API_BASE:-http://127.0.0.1:8000/agent/v1}"
export BC250_MODEL_API_KEY="${BC250_MODEL_API_KEY:-frontierrl-public}"
export BC250_MODEL_NAME="${BC250_MODEL_NAME:-frontierrl/star-7}"
# Star benchmark campaigns are API-search-only. Personal Chrome is not part of
# the evaluation path because search-engine challenges make it non-reproducible.
export BC250_SEARCH_PROVIDER="brave"
export BC250_BRAVE_API_KEY="${BC250_BRAVE_API_KEY:-${BRAVE_API_KEY:-}}"
export BC250_TAVILY_API_KEY="${BC250_TAVILY_API_KEY:-${TAVILY_API_KEY:-}}"
export BC250_SERPER_API_KEY="${BC250_SERPER_API_KEY:-${SERPER_API_KEY:-}}"
export BC250_GOOGLE_CHROME_HOST="${BC250_GOOGLE_CHROME_HOST:-sam-mbp-rev}"
export BC250_EXTERNAL_MODEL_ENABLED="${BC250_EXTERNAL_MODEL_ENABLED:-true}"
export BC250_EXTERNAL_MODEL_API_URL="${BC250_EXTERNAL_MODEL_API_URL:-http://127.0.0.1:8000/api/external-model-requests}"
export BC250_EXTERNAL_MODEL_ADMIN_TOKEN="${BC250_EXTERNAL_MODEL_ADMIN_TOKEN:-${STATUS_ADMIN_TOKEN:-${ADMIN_TOKEN:-}}}"
export BC250_EXTERNAL_MODEL_PROVIDER="${BC250_EXTERNAL_MODEL_PROVIDER:-chatgpt}"
export BC250_GRADER_API_KEY="${BC250_GRADER_API_KEY:-${OPENAI_API_KEY:-}}"
export BC250_GRADER_MODEL="${BC250_GRADER_MODEL:-gpt-5.6}"

if [[ "$BC250_SEARCH_PROVIDER" == "brave" && -z "$BC250_BRAVE_API_KEY" ]]; then
  echo "BC250_BRAVE_API_KEY or BRAVE_API_KEY is required for the Brave search profile" >&2
  return 2 2>/dev/null || exit 2
fi
