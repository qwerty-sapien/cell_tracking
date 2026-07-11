# Ultrack Prototype Runbook

This branch keeps the Kaggle archive zipped. Do not extract raw zarrs; the CV
runner streams small clips from `data/kaggle/raw/biohub-cell-tracking-during-development.zip`
and writes only derived metadata, optional input clips, and Ultrack sqlite files
under `data/derived/ultrack/{run_id}/`.

## 1. Validate Gurobi

```bash
.venv/bin/python scripts/validate_gurobi.py \
  --output reports/gurobi_validation.json
```

The report never stores secret values. It only records whether credential-like
environment variables exist, whether `gurobipy` imports, whether a tiny binary
MILP solves, and whether `ultrack check_gurobi` passes.
By default the validator attempts `pip install gurobipy` and `pip install ultrack`
if they are missing; use `--no-install-missing` for a metadata-only dry run.
The current workspace venv is Python 3.14; at the time of this implementation,
published `ultrack` wheels require Python `<3.13`, so full local Ultrack
validation needs a Python 3.11/3.12 environment or a compatible Kaggle runtime.

If the report says `GUROBI_API_KEY` is insufficient, provide standard Gurobi WLS
credentials as `WLSACCESSID`, `WLSSECRET`, and `LICENSEID`, or set
`GRB_LICENSE_FILE` to a valid `gurobi.lic`.

## 2. IO/Input Smoke

```bash
.venv/bin/python scripts/run_ultrack_zip_cv.py \
  --io-only \
  --max-datasets 1 \
  --clip-length 3 \
  --foreground-quantiles 0.999 \
  --contour-presets low \
  --run-id io-smoke \
  --cache-root data/derived/ultrack-smoke \
  --metrics-output reports/ultrack_io_smoke_metrics.json \
  --output reports/ultrack_io_smoke_submission.csv
```

This verifies ZIP-backed reads and foreground/contour generation without
requiring Ultrack or a solver.

## 3. Default CV Clip

```bash
.venv/bin/python scripts/run_ultrack_zip_cv.py \
  --split train \
  --run-id first-ultrack-cv \
  --metrics-output reports/ultrack_cv_metrics.json \
  --output reports/ultrack_cv_submission.csv
```

With a passing Gurobi report, `--solver auto` uses `GUROBI` and a 16-frame clip.
Otherwise it falls back to `CBC` and a 12-frame clip. Default datasets are the
first available `44b6_*` and `6bba_*` training samples.

To run the planned first grid:

```bash
.venv/bin/python scripts/run_ultrack_zip_cv.py \
  --foreground-quantiles 0.995,0.9975,0.999 \
  --contour-presets low,medium,high \
  --max-link-um 8.0 \
  --plot-overlays \
  --metrics-output reports/ultrack_cv_grid_metrics.json
```
