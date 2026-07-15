#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${1:?Usage: verify_archive.sh path/to/archive.zip}"
exec python "$ROOT/scripts/verify_archive.py" "$ARCHIVE"
