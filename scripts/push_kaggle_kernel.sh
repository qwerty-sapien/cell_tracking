#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

source .venv/bin/activate
python scripts/build_kaggle_kernel.py
kaggle kernels push -p kaggle_kernel
