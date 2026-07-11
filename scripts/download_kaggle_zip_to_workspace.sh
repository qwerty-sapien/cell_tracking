#!/usr/bin/env bash
set -euo pipefail

COMPETITION="${COMPETITION:-biohub-cell-tracking-during-development}"
OUT_DIR="${OUT_DIR:-data/kaggle/raw}"

mkdir -p "$OUT_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  cat >&2 <<EOF
Kaggle CLI was not found in this shell.

Run from an environment where 'kaggle' is available, or install/configure it.
The intended command is:

  kaggle competitions download -c "$COMPETITION" -p "$OUT_DIR"

This script downloads the ZIP only; it does not extract it.
EOF
  exit 127
fi

echo "Downloading Kaggle competition ZIP into $OUT_DIR"
kaggle competitions download -c "$COMPETITION" -p "$OUT_DIR"
echo "Done. Do not unzip; use scripts/inspect_zip_archives.py for metadata inspection."

