# Scenario-Routed Alpha Runbook

Status: implemented first low-compute alpha, 2026-07-09.

## Direction

The active G-01 low-compute path is the non-duplicate/general alpha route:

1. Use the locked non-duplicate split in `reports/general_model_split_manifest.json`.
2. Exclude train/test duplicate IDs and GEFF-copy outputs from general-model score evidence.
3. Run multi-source candidate generation plus constrained lineage graph selection.

The alpha implements this without requiring Ultrack, Gurobi, Trackastra, or model training:

- classify each dataset/window into a scenario route;
- generate fused candidate centers from multiple proposal sources;
- select directed edges under biological constraints;
- optionally test a precision-first division route;
- serialize the required Kaggle node/edge CSV.

DoG is retained only as one proposal source. The reusable interfaces are under
`src/celltrack_alpha/`, so Ultrack, learned edge scorers, sparse U-Net/vector
models, or Trackastra-style association can later replace individual stages.

## Routes

### Historical Duplicate GEFF Route

The earlier public-ZIP route copied train GEFF graphs when train/test Zarrs had
matching member names, sizes, and CRC32 values. That route produced the
historical `reports/duplicate_geff_submission_report.json` score, but it is a
data-identity shortcut rather than a general tracker. G-01 removed the executable
GEFF-copy source and script from the active code path; the historical report and
submission artifact remain only as leakage evidence.

### General Alpha Routes

Current route labels:

- `clean_point`: fewer high-confidence candidates, no divisions, fast.
  Historical `44b6` post-link offsets are diagnostic only; any offset used for
  held-out scoring must come from a fold-safe calibration JSON.
- `dense_fusion`: high-recall multi-source proposals for crowded windows.
- `low_signal`: lower thresholds and contrast-normalized/dark proposals.
- `high_motion_gap`: wider linking and conservative 2-frame gap edges.
- `division_likely`: experimental fork-capable route; use only in CV until
  division FP is controlled. Auto-routing will not select it unless
  `--allow-division-route` is passed.

The router is heuristic today. `--compare-routes` creates the training table for
replacing it with a shallow classifier: each dataset/window can be labeled by
the best held-out route under the metric.

## Smoke Tests

Compile:

```bash
.venv/bin/python -m py_compile src/celltrack_alpha/*.py scripts/run_alpha_routed_zip.py
```

Fast clean route smoke:

```bash
.venv/bin/python scripts/run_alpha_routed_zip.py \
  --split train \
  --datasets 44b6_0113de3b \
  --clip-start 0 \
  --clip-length 2 \
  --evaluate \
  --route clean_point \
  --output reports/alpha_routed_clean_smoke_submission.csv \
  --metrics-output reports/alpha_routed_clean_smoke_metrics.json
```

Auto route smoke:

```bash
.venv/bin/python scripts/run_alpha_routed_zip.py \
  --split train \
  --datasets 44b6_0113de3b \
  --clip-start 0 \
  --clip-length 2 \
  --evaluate \
  --route auto \
  --output reports/alpha_routed_smoke_submission.csv \
  --metrics-output reports/alpha_routed_smoke_metrics.json
```

Route comparison CV smoke:

```bash
.venv/bin/python scripts/run_alpha_routed_zip.py \
  --split train \
  --datasets 44b6_0113de3b \
  --clip-start 0 \
  --clip-length 2 \
  --evaluate \
  --compare-routes \
  --route auto \
  --metrics-output reports/alpha_routed_compare_metrics.json
```

## Official Scoring

The venv now supports the official metric through `tracksdata` and `polars`.
Evaluate an existing train prediction CSV without extracting the Kaggle ZIP:

```bash
.venv/bin/python scripts/evaluate_alpha_official_zip.py \
  --prediction-csv reports/alpha_routed_full_44b6_0113de3b_offset_submission.csv \
  --datasets 44b6_0113de3b \
  --gt-self-check \
  --output reports/alpha_routed_full_44b6_0113de3b_offset_official_eval.json
```

Tune post-link coordinate corrections on an existing prediction:

```bash
.venv/bin/python scripts/tune_alpha_coordinate_offsets.py \
  --prediction-csv reports/alpha_routed_full_44b6_0113de3b_submission.csv \
  --dataset 44b6_0113de3b \
  --output reports/alpha_coordinate_offset_tuning.json
```

## Rigorous Testing

Use video-held-out and time-window-held-out route comparisons. For each route,
record:

- adjusted edge Jaccard, edge Jaccard, node recall, node-count ratio;
- strict approximate division TP/FP/FN/Jaccard;
- number of nodes, edges, divisions, merges, and out-degree >2 violations;
- runtime and failure modes.

The local environment now has the official metric dependencies. Use official
metrics for score-gate decisions; keep approximate metrics only as a cheap
debugging signal.

Fallback benchmark status is documented in `reports/alpha_reassessment.md`: the
corrected full-video `44b6_0113de3b` score is `0.6283`, below the objective's
`0.8` gate. The fallback needs a detector/localizer revamp for non-duplicate
private data.

## Compute Policy

Use local/Kaggle free compute for:

- route comparison tables;
- proposal grids;
- graph constraints and threshold tuning;
- small learned edge/fork scorers.

Paid compute is allowed only for a timed 30-60 minute pilot whose extrapolated
full run remains under `$25` with a 1.5x overhead factor. Spend compute first on
proposal diversity and small learned scoring, not large end-to-end 3D training.

## Submission

Test submission generation:

```bash
.venv/bin/python scripts/run_alpha_routed_zip.py \
  --split test \
  --route auto \
  --output data/kaggle/predictions/alpha_routed_submission.csv \
  --metrics-output reports/alpha_routed_test_metrics.json
```

The script validates schema, endpoint existence, duplicate edges, merge counts,
out-degree >2 counts, and that every requested dataset appears in the CSV.
