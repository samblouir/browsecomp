#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/star-env.sh

limit="${1:-1}"
mkdir -p logs
exec .venv/bin/bc250 run --config configs/star-smoke.yaml --limit "$limit"
