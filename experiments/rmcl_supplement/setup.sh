#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Keep the server's working CUDA/PyTorch installation. No training is started here.
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B run.py preflight
