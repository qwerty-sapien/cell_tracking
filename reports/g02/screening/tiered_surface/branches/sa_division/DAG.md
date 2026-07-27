# SA-DIVISION Phase-1 DAG

Date: 2026-07-28  
Owner: `SA-DIVISION`  
Write scope: `reports/g02/screening/tiered_surface/branches/sa_division/` only  
Status: dependency audit active; division fitting closed

## Objective and immutable order

Audit the N4 division dependency, preserving the existing persistent-fork decoder
evidence and validating only frozen N1 atomic fork/no-op and official scorer fixture
semantics unless the dependency opens.

```text
authenticate frozen root pins
  -> freeze dependency preregistration
  -> inventory banked association-continuity arms
       -> zero qualifying arms: PARK_DEPENDENCY
          -> N1 fork/no-op synthetic fixtures
          -> immutable official scorer semantics fixtures
          -> terminal fragment and proposed ledger row
       -> at least one qualifying arm: STOP_AND_REPORT_ROOT
          -> no fitting until root acknowledges the banked path/SHA/vectors
```

The dependency opens only when one *single banked association-continuity arm* has
immutable results on both `44b6` and `6bba`, with each family satisfying:

- adjusted edge-Jaccard delta `>= +0.005`;
- raw edge-Jaccard delta `>= 0`;
- node-recall delta `>= 0`.

The qualifying arm must be cited by resolvable path and SHA-256 with exact
`adjusted/raw/node` vectors for both families. Results may not be joined across
different arms. P0 division-oracle headroom, proposal metrics, CV57-only evidence,
the external memo's C2/K-best values, and its Branch-B `66/223` audit cannot satisfy
the dependency.

## Hard stops

- No division fitting, model training, held-out candidate scoring, `44b6`, `6bba`,
  or CV57 execution without the dependency.
- If a qualifying banked arm is found, stop and report it to the root coordinator
  before any fit.
- No D1 refit or decoder redesign; preserve its frozen evidence.
- No DEAD mechanism rescreen, plan amendment, GPU, Kaggle, submission, RunPod,
  external service, or sibling live-output read.
- Every local run requires a fresh 45-second idle preflight and zero-swap guard.

