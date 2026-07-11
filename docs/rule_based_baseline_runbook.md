# Rule-Based Baseline Runbook

Budget policy: local compute plus Kaggle free compute only. Runpod balance remains unused until a specific bottleneck justifies a `$5` top-up.

## Local ZIP Smoke Tests

Activate the venv before running scripts:

```bash
source .venv/bin/activate
```

Run a small train smoke test from the compressed Kaggle archive:

```bash
python scripts/run_rule_based_baseline_zip.py \
  --split train \
  --datasets 44b6_0113de3b \
  --max-timepoints 5 \
  --evaluate \
  --output reports/rule_based_train_smoke_submission.csv \
  --metrics-output reports/rule_based_train_smoke_metrics.json
```

Run a small test-submission smoke test:

```bash
python scripts/run_rule_based_baseline_zip.py \
  --split test \
  --max-datasets 1 \
  --max-timepoints 2 \
  --output reports/rule_based_test_smoke_submission.csv
```

Create a visual failure overlay:

```bash
python scripts/plot_rule_based_overlay.py \
  --prediction-csv reports/rule_based_train_smoke2_submission.csv \
  --dataset 44b6_0c582fdc \
  --t 20 \
  --output reports/plots/rule_based_overlay_44b6_0c582fdc_t20.png
```

## Kaggle Free Compute

Build and push the self-contained Kaggle kernel:

```bash
scripts/push_kaggle_kernel.sh
```

Check status and logs:

```bash
kaggle kernels status onebitbrain/biohub-rule-based-dog-hungarian
kaggle kernels logs onebitbrain/biohub-rule-based-dog-hungarian
```

Download outputs after the run completes:

```bash
kaggle kernels output onebitbrain/biohub-rule-based-dog-hungarian -p data/kaggle/kernel_output
```

The kernel writes `/kaggle/working/submission.csv` in default `MODE=submit`.

## Current Technical Notes

- Local ZIP-backed zarr reads work without extracting the archive.
- Kaggle's base image lacked `zarr`, so the kernel enables internet and installs missing packages at startup.
- The first bright DoG detector produces valid graph CSVs but has zero approximate sparse recall on two tiny train smoke tests. Visual overlays suggest the sparse labels are not always strong bright local maxima, so the next tuning pass should sweep polarity, output coordinate transform, threshold, and scale before any paid compute.
- Division prediction is intentionally disabled in this baseline.
