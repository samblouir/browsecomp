#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LIMIT="${BC250_SMOKE_LIMIT:-1}"
exec .venv/bin/bc250 run --config "${BC250_CONFIG:-configs/smoke.yaml}" --limit "$LIMIT" "$@"
