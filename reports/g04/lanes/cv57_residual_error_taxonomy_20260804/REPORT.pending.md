# CV57 residual error taxonomy (exclusive, APNCC control)

The current scorer replay exactly reproduces the frozen APNCC CV57 score:
`0.881522069959731`.  The four edge/division categories are disjoint:
each scorer-valid edge FP is ordinary association; each official edge FN is
exactly one of detection, matched-lineage fragmentation, or ordinary
association; division FP/FN events are separate.

The full score-accounting repair ceiling is `1.100000000000000`
(`+0.218477930040269` from baseline).  This is the
theoretical 1.1 ceiling for a split that contains division events: perfect
edge score 1.0 plus perfect division Jaccard weighted by 0.1.  It is not a
claim that the independent graph edits are jointly realizable without side
effects.

Dominant category by exact nonlinear-score Shapley attribution:
**division**
(`+0.100000000000`).

| Category | Exclusive scorer events | Exact Shapley score impact | Score if only fixed |
| --- | ---: | ---: | ---: |
| detection | 466 | +0.017446063106 | 0.898236372557 |
| ordinary_association | 2,459 | +0.061628263275 | 0.941289985754 |
| track_fragmentation | 1,482 | +0.039403603659 | 0.919473950549 |
| division | 42 | +0.100000000000 | 0.981522069960 |

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
| 1 | train:6bba_786893ac | 548 | 47 | 325 | 176 | 3 |
| 2 | train:6bba_f17befbc | 284 | 9 | 174 | 101 | 0 |
| 3 | train:6bba_7f87b3d8 | 280 | 45 | 151 | 84 | 3 |
| 4 | train:6bba_3abfe10a | 256 | 13 | 151 | 92 | 1 |
| 5 | train:6bba_0c7fa718 | 181 | 19 | 106 | 56 | 0 |
| 6 | train:6bba_bb9f20c3 | 174 | 18 | 94 | 62 | 3 |
| 7 | train:6bba_7af54fde | 173 | 8 | 105 | 60 | 2 |
| 8 | train:44b6_d5e7d891 | 163 | 5 | 98 | 60 | 2 |
| 9 | train:6bba_3a1849c2 | 126 | 19 | 70 | 37 | 0 |
| 10 | train:6bba_ebdf3b34 | 121 | 15 | 65 | 41 | 2 |

The full 57-video stable index and all one-way/Shapley counterfactuals are in
`CV57_RESIDUAL_ERROR_TAXONOMY.json`.

## Inputs and validation

- Current scorer parity: `/Users/agent/Desktop/cell_tracking/reports/g02/scorer_parity_v2.json`
  (`47338fe959b7fd7c23ee2a973d7a723dfae3999eb0f20fe1e5d221af93159493`).
- Frozen APNCC control: `/Users/agent/Desktop/cell_tracking/reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/control/APNCC_CV57_SCORE_WITH_FRAGMENTATION.json`
  (`60a1f2830e44e2fe5d641bb4e0568654f435f5111ce330794a8d9d70b5c1b5d4`).
- Submission: `/Users/agent/Desktop/cell_tracking/reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/control/apncc_union.csv`
  (`a8cd423a610418ad5463a0ccd90b80f5a0af52bd1559e4cf97114bdbe7dacafb`).
- Raw archive: `/Users/agent/Desktop/cell_tracking/data/kaggle/raw/biohub-cell-tracking-during-development.zip`
  (`538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19`).
- Replay validation: 57/57 per-video official count/metric rows and all five
  aggregate metric fields match the frozen control; edge partition and Shapley
  conservation checks passed (`PASS`).

## Reproduction command

```sh
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python reports/g04/lanes/cv57_residual_error_taxonomy_20260804/build_taxonomy.py
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 pytest -q tests/test_public_temporalunet_baseline.py tests/test_track_visualizer.py
```
