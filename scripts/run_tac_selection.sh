#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUDGET="${1:-150}"
TARGET="${2:-all}"

python "${ROOT}/scripts/select_tac_sources.py" --budget "${BUDGET}" --target "${TARGET}"
