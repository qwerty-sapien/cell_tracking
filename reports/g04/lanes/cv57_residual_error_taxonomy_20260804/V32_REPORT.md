# CV57 residual error taxonomy (exclusive, immutable public TemporalUNet V32 CV57)

The current scorer replay exactly reproduces the frozen APNCC CV57 score:
`0.914358473020729`.  The four edge/division categories are disjoint:
each scorer-valid edge FP is ordinary association; each official edge FN is
exactly one of detection, matched-lineage fragmentation, or ordinary
association; division FP/FN events are separate.

The full score-accounting repair ceiling is `1.100000000000000`
(`+0.185641526979271` from baseline).  This is the
theoretical 1.1 ceiling for a split that contains division events: perfect
edge score 1.0 plus perfect division Jaccard weighted by 0.1.  It is not a
claim that the independent graph edits are jointly realizable without side
effects.

Dominant category by exact nonlinear-score Shapley attribution:
**division**
(`+0.097297297297`).

| Category | Exclusive scorer events | Exact Shapley score impact | Score if only fixed |
| --- | ---: | ---: | ---: |
| detection | 538 | +0.010373258551 | 0.924544552935 |
| ordinary_association | 1,931 | +0.049586130139 | 0.962950511564 |
| track_fragmentation | 1,053 | +0.028384840992 | 0.942091616848 |
| division | 108 | +0.097297297297 | 1.011655770318 |

`score_if_only_fixed` is deliberately shown alongside the additive Shapley
attribution because the official Jaccard and node adjustment are nonlinear.
The four Shapley values sum exactly (within floating-point tolerance) to the
full-ceiling delta.

## Worst videos in visualizer-compatible order

Sorted by `edge_fp + edge_fn` descending, then identity ascending; `rank`,
`residual`, and contribution fields have the same ordering contract as
`scripts/build_track_visualizer.py`.

| Rank | Video | Edge residual | Detection FN | Ordinary edge events | Fragmentation FN | Division errors |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | train:6bba_786893ac | 457 | 44 | 273 | 140 | 9 |
| 2 | train:6bba_7f87b3d8 | 245 | 45 | 131 | 69 | 4 |
| 3 | train:6bba_f17befbc | 214 | 21 | 122 | 71 | 2 |
| 4 | train:6bba_3abfe10a | 192 | 20 | 112 | 60 | 6 |
| 5 | train:6bba_bb9f20c3 | 178 | 29 | 95 | 54 | 6 |
| 6 | train:6bba_0c7fa718 | 164 | 12 | 98 | 54 | 0 |
| 7 | train:44b6_d5e7d891 | 151 | 25 | 87 | 39 | 8 |
| 8 | train:6bba_ebdf3b34 | 126 | 6 | 76 | 44 | 3 |
| 9 | train:6bba_acd782a8 | 92 | 5 | 57 | 30 | 0 |
| 10 | train:6bba_3a1849c2 | 90 | 15 | 52 | 23 | 2 |

The full 57-video stable index and all one-way/Shapley counterfactuals are in
`CV57_RESIDUAL_ERROR_TAXONOMY.json`.

## Inputs and validation

- Current scorer parity: `reports/g02/scorer_parity_v2.json`
  (`47338fe959b7fd7c23ee2a973d7a723dfae3999eb0f20fe1e5d221af93159493`).
- Frozen `immutable public TemporalUNet V32 CV57` control: `reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_UNSHARDED_V2_CV57_SCORE.json`
  (`8827cd15ae3cc7d95dee95c773be67ed46480ba3e464aa48462b79664fed93c2`).
- Submission: `/Users/agent/Desktop/cell_tracking/reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/full_public_v32_unsharded_v2_output_v1/submission.csv`
  (`c61784c329b617b7aef4df7926642e3af60d539e1085a0d7d8655486b5aab446`).
- Raw archive: `data/kaggle/raw/biohub-cell-tracking-during-development.zip`
  (`538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19`).
- Replay validation: 57/57 per-video official count/metric rows and all five
  aggregate metric fields match the frozen control; edge partition and Shapley
  conservation checks passed (`PASS`).

## Reproduction command

```sh
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python reports/g04/lanes/cv57_residual_error_taxonomy_20260804/build_taxonomy.py
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv313/bin/python -m pytest -q tests/test_public_temporalunet_baseline.py tests/test_track_visualizer.py
```
