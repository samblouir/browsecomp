#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="${1:-$ROOT/../browsecomp-250-openai-compatible.zip}"
PYTHON_BIN="${BC250_PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.12+ environment not found at $PYTHON_BIN; run ./scripts/bootstrap.sh" >&2
  exit 2
fi
PYTHONPATH=src "$PYTHON_BIN" scripts/package_repo.py --output "$OUTPUT"
"$PYTHON_BIN" scripts/verify_archive.py "$OUTPUT"
