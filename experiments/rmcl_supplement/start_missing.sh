#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$PYTHON" -B run.py preflight
"$PYTHON" -u -B run.py train --plan missing
"$PYTHON" -B run.py analyze
"$PYTHON" -B run.py representations
