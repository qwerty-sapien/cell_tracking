# E1 — Error taxonomy: detection-bound vs association-bound

> First load `_shared_context.md`. Parallel-eligible in Wave 1. **This node decides whether any GPU
> spend can help.**

## Why
Before rationing $5–10 of GPU on a learned association model, we must know whether the residual error
is even *reachable* by better association. If most FN/FP come from missing or wrong detections, no
edge model fixes them and GPU spend is wasted; if they come from association mistakes on correct
detections, a learned scorer has headroom.

## Tasks
1. Take the baseline 2,736 FP + 2,094 FN and bucket each into a mechanistic cause:
   - **detection-bound**: node missing / spurious / mis-segmented → the correct edge was impossible.
   - **association-bound**: nodes correct on both frames but the edge was wrong (gap-close failure,
     over-merge, ID switch, wrong parent at division).
2. Quantify the split. Report the association-bound fraction with a bound on classification error.
3. For the association-bound errors, characterize them (motion magnitude, local density, frame gap)
   so T1/M1 know what a model would need to fix.
4. Emit a one-word **verdict** for the GPU-gate: `ASSOCIATION-BOUND` (GPU plausibly helps) or
   `DETECTION-BOUND` (GPU parked; redirect effort to the detection front-end / B3).

## Output
`reports/g02/screening/e1_taxonomy.md` with the bucket counts, the association-bound fraction, and
the verdict. Feeds B3 and is condition (c) of the GPU-gate in P1.
