#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/star-env.sh
exec .venv/bin/bc250 prepare --config configs/star-headline.yaml
