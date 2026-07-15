#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(f"Python 3.12+ is required; found {sys.version.split()[0]}")
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ ! -d .venv ]]; then
  if ! "$PYTHON_BIN" -m venv .venv; then
    echo "Standard venv seeding failed; retrying with uv." >&2
    rm -rf .venv
    if ! command -v uv >/dev/null 2>&1; then
      echo "uv is required when Python ensurepip cannot seed the environment." >&2
      exit 2
    fi
    uv venv --seed --python "$PYTHON_BIN" .venv
  fi
fi

if [[ "${BC250_SKIP_PIP_UPGRADE:-0}" != "1" ]]; then
  .venv/bin/python -m pip install --upgrade pip setuptools
fi
.venv/bin/pip install -e '.[dev]'

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example; edit credentials before live checks."
fi

.venv/bin/bc250 --version
.venv/bin/bc250 subset
.venv/bin/python -m compileall -q src tests
.venv/bin/pytest

echo "Bootstrap complete. Next: edit .env, then run ./scripts/prepare.sh"
