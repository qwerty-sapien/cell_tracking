# P0A — Official scorer semantic parity

> First load `_shared_context.md`. This node gates local score claims. Transfer
> calibration is handled separately by P0R/P0B.

## Why this is the local root
If the local patched scorer implements stale semantics, every local screen measures the
wrong target. Prove source and fixture parity before local work. Do not require monotone
local-to-public movement here; that is an empirical transfer question for P0B.

## Tasks
1. Diff our patched metric implementation against the official scorer in
   `external/kaggle-cell-tracking-competition/` after the rescoring. Enumerate any divergence
   (division credit, gap-closing, node-vs-edge weighting, normalization). Confirm
   `src/celltrack_alpha/scorer_parity.py` matches the current official code, not a stale copy.
2. Run the deterministic official semantic fixtures and freeze the scorer/source hashes.
3. Record wrapper-only divergences without conflating them with public transfer residuals.

## Outputs
- `reports/g02/screening/p0_parity.md` with separate `P0A` and `P0B` verdicts.
- If P0A fails: a concrete scorer divergence and halt local score claims.

## Decision
- P0A PASS → local decomposition, diagnostics, and model development may proceed.
- P0A FAIL → halt local scoring until repaired.
- P0B remains independent and gates public inference/S1.
