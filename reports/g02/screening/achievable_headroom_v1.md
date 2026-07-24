# Achievable headroom diagnostic

- In-pool net-correct swaps at labeled precision >=0.95: **2798**.
- Per-video p95 threshold IQR/range: **0.237329 / 0.897502**; stable = **False**.
- Fit-only recall allocation of the 378 target swaps: **290 in pool / 88 excluded**.
- Current-pool perfect-ranker oracle score: **0.895677948** versus banked **0.881522070**.
- Recommendation: **REDIRECT_DETERMINISTIC_AND_RECALL** — per-video thresholds exceed the preregistered IQR/range stability limits. Preserve the banked swap and prioritize deterministic calibration plus recall expansion.

The p95 precision is computed only over known labeled proposals. Unlabeled
background participates in the deployed cap but is not treated as negative.
At the selected point it consumes 4711 cap slots.
