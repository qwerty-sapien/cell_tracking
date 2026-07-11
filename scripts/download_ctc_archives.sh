#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-data/ctc/raw}"
mkdir -p "$OUT_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/download_ctc_archives.sh DATASET [DATASET ...]
  scripts/download_ctc_archives.sh --smoke
  scripts/download_ctc_archives.sh --recommended

Datasets are downloaded as ZIP archives and are never extracted.

Known dataset keys:
  Fluo-N3DH-CHO       98 MB, small nuclei smoke test
  Fluo-C3DL-MDA231    182 MB, small real 3D microscopy smoke test
  Fluo-N3DH-CE        3.1 GB, C. elegans embryo nuclei
  Fluo-N3DL-DRO       5.8 GB, Drosophila embryo light-sheet
  Fluo-N3DH-SIM+      3.1 GB, simulated nuclei with exact annotations
  Fluo-N3DL-TRIC      20.6 GB, Tribolium embryo cartographic projection
EOF
}

url_for() {
  case "$1" in
    Fluo-N3DH-CHO) echo "https://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CHO.zip" ;;
    Fluo-C3DL-MDA231) echo "https://data.celltrackingchallenge.net/training-datasets/Fluo-C3DL-MDA231.zip" ;;
    Fluo-N3DH-CE) echo "https://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CE.zip" ;;
    Fluo-N3DL-DRO) echo "https://data.celltrackingchallenge.net/training-datasets/Fluo-N3DL-DRO.zip" ;;
    Fluo-N3DH-SIM+) echo "https://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-SIM+.zip" ;;
    Fluo-N3DL-TRIC) echo "https://data.celltrackingchallenge.net/training-datasets/Fluo-N3DL-TRIC.zip" ;;
    *) return 1 ;;
  esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -eq 0 ]]; then
  usage
  exit 0
fi

datasets=()
case "${1:-}" in
  --smoke)
    datasets=(Fluo-N3DH-CHO Fluo-C3DL-MDA231)
    ;;
  --recommended)
    datasets=(Fluo-N3DH-CHO Fluo-C3DL-MDA231 Fluo-N3DH-CE Fluo-N3DL-DRO Fluo-N3DH-SIM+)
    ;;
  *)
    datasets=("$@")
    ;;
esac

for dataset in "${datasets[@]}"; do
  url="$(url_for "$dataset")" || {
    echo "Unknown dataset key: $dataset" >&2
    usage >&2
    exit 2
  }
  out="$OUT_DIR/${dataset}.zip"
  if [[ -s "$out" ]] && zipinfo -t "$out" >/dev/null 2>&1; then
    echo "Skipping $dataset; complete ZIP already exists at $out"
    continue
  fi
  echo "Downloading $dataset -> $out"
  curl -L --fail --retry 5 --retry-delay 10 --continue-at - \
    --output "$out" "$url"
done
