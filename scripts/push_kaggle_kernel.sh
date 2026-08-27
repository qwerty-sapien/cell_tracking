#!/usr/bin/env bash
set -euo pipefail

echo >&2 "REFUSED: direct 'kaggle kernels push' bypasses the real prelaunch replay, shared lock, quota reconciliation, and bracketed cmux_ops boundary."
echo >&2 "Use scripts/cmux_ops.py --bracketed with a fresh real guard predicate and current LMT bindings."
exit 64
