#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv-prod}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  echo "Install Homebrew python@3.12 first, or override PYTHON_BIN." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install torch transformers sentencepiece accelerate

echo "Production environment ready at $VENV_DIR"
echo "Run:"
echo "  $VENV_DIR/bin/python scripts/check_experiment_readiness.py --deep-runtime-probe --probe-timeout-seconds 20"
