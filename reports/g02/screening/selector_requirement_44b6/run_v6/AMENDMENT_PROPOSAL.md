# Proposed separation-gate threshold amendment — human decision required

This proposal is evidence only. It does not modify the frozen gate.

The following feasible action classes exceeded at least one current floor (AUC 0.65, retention 0.15) at the minimum tested selector setting whose mean exact 44b6 adjusted-edge-Jaccard delta reached +0.005:

- `delete`: tested AUC 0.85; mean CG0-budget retention 0.020394; support prefix 16384.
- `occupied_swap`: tested AUC 0.90; mean CG0-budget retention 0.067910; support prefix 1024.

Proposed human decision: retain the current permissive screen as an unlock-only diagnostic with an explicit warning that it is not graph-sufficient, or raise class-specific floors no lower than the measured grid requirements after reviewing the complete seed distributions.

**HALT: HUMAN_REVIEW_REQUIRED.**
