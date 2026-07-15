#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/star-env.sh

mkdir -p logs
exec .venv/bin/bc250 headline --config configs/star-headline.yaml --yes
