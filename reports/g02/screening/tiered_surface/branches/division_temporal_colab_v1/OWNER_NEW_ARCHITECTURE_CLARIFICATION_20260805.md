# Owner scope clarification: longer-context detector is a new architecture

Status: **OWNER CLARIFICATION RECORDED — NO WEIGHT-REFINEMENT WAIVER REQUIRED**.

The proposed division parent-event detector is outside the existing “no weight refinement below local `0.916`” rule on its merits:

- V32 consumes two frames. The proposed detector must consume more than two frames, changing the input tensor contract and temporal architecture.
- Only shape-compatible spatial features may be partially initialized. The temporal stem, parent-event head, division-weighted objective, optimizer state, and training trajectory are new.
- The output estimand is parent-event probability for reuse by the already-frozen conditional daughter-pair head. It is not a V32 score-tuning or daughter-assignment refit.
- V32 remains immutable and the rule continues to prohibit fine-tuning or weight refinement of V32 itself below the existing threshold.

Therefore this lane is classified as `NEW_LONGER_CONTEXT_ARCHITECTURE`, not `WEIGHT_REFINEMENT`. The former amendment request is superseded by this narrower scope clarification.

This clarification does **not** authorize training before the APNCC-vs-V32 taxonomy, frozen objective, measured capacity projection, and all deployment gates exist. It also does not authorize a RunPod launch; RunPod remains explicitly approval-gated per run.
