# Retired experiment archive

Agents must ignore this directory unless a user explicitly requests archived material.

- `conflict_residual/`: retired after the full 57-video measurement showed only about +0.0016 and the line was explicitly closed.
- `pseudo_ssl/`: never run; the supervised-scaling gate answered the intended question first.
- `compact_head/`: compact-head detector/checkpoint experiments have no submission path and were rejected by `f4cmp`.
- `scored_partial/`: retained evidence for the measured -0.0746 scored-partial topology regression.
- `unused_ultrack/`: `celltrack_ultrack` was never installed and is the wrong fit for these sparse points.
- `general_model/`: roughly 800 G-01 probe outputs superseded by the later grouped-split work.
- `failed_bh9/`: `bh9o6` and `bh9o7` are large failed-run outputs with no active path.
- `cruise_staging/`: duplicate drift blobs and stale integrity-script copies that could mislead future work.
- `organizer_unet_transformer_f0/`: retired organiser TemporalUNet3D +
  SimpleNodeTransformer experiment. Its epoch-7 fit-only checkpoint is preserved, but
  convergence projected far beyond the locked 30 cumulative Kaggle T4-hour per-model
  ceiling. No holdout evaluation was performed.

The load-bearing `reports/g02/f4gtr.json`,
`reports/g02/division_failure_fold0.json`,
`reports/g02/source_softmax_edge_f0_fit117_holdout_prefix52_v1/error_budget.md`,
`reports/g02/source_softmax_edge_f0_fit117_holdout_complete57_v2/fragmentation.md`,
and `reports/g02/t04_v45f4_fold0_fixed_node_oracle/` remain outside this archive.
