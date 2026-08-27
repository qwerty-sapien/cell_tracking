# E1 error taxonomy result

Verdict: **ASSOCIATION-BOUND**
Decision: **BANK mechanism evidence; GPU remains gated**

On the exact complete-57 source-softmax fragmentation artifact:

- 1,319/2,161 FNs are verified occupied-endpoint association conflicts;
- 19/2,161 are verified degree/solver conflicts;
- 308/2,161 have an unmatched GT endpoint;
- 515/2,161 are mixed or unresolved from persisted output alone.

This gives a conservative verified association-bound lower bound of **61.92%** and a
detection/localization lower bound of **14.25%**. The fixed-node diagnostic oracle on the
same prediction nodes scores `0.9749936` versus `0.8495037`, proving large association
headroom on that identity.

These are not silently substituted for current `c=0.25` counts. They prioritize a cached
full association-cost replacement locally. GA still requires a preregistered `>= +0.005`
affected-slice proxy gain, runtime/budget fit, and the applicable launch approval.

Full consolidation: `reports/g02/screening/evidence_consolidation_v1.md`.
