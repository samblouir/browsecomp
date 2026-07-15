#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec .venv/bin/bc250 doctor --config "${BC250_CONFIG:-configs/smoke.yaml}" "$@"
