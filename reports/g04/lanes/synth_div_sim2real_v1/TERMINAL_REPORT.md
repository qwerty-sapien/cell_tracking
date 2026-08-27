# Synthetic Division Sim-to-Real v2 — Terminal Report

Date: 2026-08-04  
Decision: **PARK-C on real-transfer evidence; no graph integration**

## Outcome

The global preregistered ranking gate passed: beneficial-versus-harmful action
AUROC is `0.8893005804713272` against the frozen `0.85` pass line. The exact
candidate population contains 65 beneficial and 80,013,652 harmful actions.
The real candidate universe reaches 65/83 annotated divisions.

The frozen deployment grid nevertheless triggers the contract's zero-real-TP
hard failure. All 80,013,717 scores are below the minimum declared threshold
`0.0005`. Consequently all 19 thresholds × 6 per-video caps produce the same
frontier row:

| Rows | Predicted forks | TP | FP | FN | Precision | Recall | Division Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0 | 0 | 0 | 83 | undefined | 0.0 | 0.0 |

There is no maximum-J policy with a nonzero action, no recall policy with a
nonzero TP, and no policy with at least one TP in both 44b6 and 6bba. The full
114-row evidence remains in `REAL_EVALUATION_V3.json`.

## Diagnostics

- Global AUROC / AP: `0.8893005804713272` / `0.00001909311504527114`
- 44b6 AUROC / AP: `0.8434824934244854` / `0.0000041565408549718095`
- 6bba AUROC / AP: `0.8952174314380125` / `0.00002820538570045679`
- Reliability bin `[0,0.0005)`: 80,013,717 actions, mean score
  `0.00011840934363346956`, observed positive rate
  `0.0000008123607106016584`

## Governance disposition

The model ranked real forks better than chance, but its frozen calibrated
posterior-product scale is incompatible with every preregistered deployment
threshold. Lowering the grid, shifting logits, rescaling scores, or fitting a
threshold now would use real labels after outcome observation and is forbidden.

No V32 graph was read or modified, no integration artifact was written, no
combined-score frontier was run, and the visual-review bundle was not refreshed.
The visualizer's native V32 division HUD check remains owned by the V32 lane;
there is no synthetic-correction graph to visualize from this instantiation.

Authoritative adjudication: `REAL_GATE_ADJUDICATION_V3.json`.
