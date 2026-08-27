# C1 N2 — build and deterministic dry validation

Date: 2026-07-27  
Verdict: **GO**  
Evidence SHA-256: `28b544de327ac273ce3f2d42207f853328573668c9e1940f080684d81df7f398`

## Built implementation

- Model/component/training primitives:
  `src/celltrack_alpha/c1_conflict_set_attention.py`
  (`0fa91569…19e0d`).
- Fail-closed fit-only identity audit:
  `scripts/audit_c1_hoct_lite.py` (`3513aff7…664b`).
- Deterministic dry runner:
  `scripts/run_c1_hoct_lite_dry.py` (`85fc141b…2750`).
- Focused tests:
  `tests/test_c1_conflict_set_attention.py` (`179f9a0f…e53e`).

The implementation contains only the frozen 20,097-parameter model and deterministic
component/fold/fit/predict functions. No Trackastra weight is loaded or trainable. It
has no graph adjacency input, cross-component call, division state, family feature,
architecture search, threshold sweep, or GPU path.

## Validation results

The machine-readable report
[`c1_n2_build_validation.json`](c1_n2_build_validation.json) records:

- **8/8 focused tests PASS:** exact parameter count, permutation equivariance,
  padding/mask invariance, cross-component isolation, explicit no-link subtraction,
  deterministic component construction, video-disjoint folds, and byte-deterministic
  fitting.
- All **105** feature payload hashes replayed: **195,354** proposals,
  **280,095** rows, **160,490** components.
- Exact C0 ridge construction was repeated twice across all five actual grouped folds.
  All **195,354** OOF prediction values were byte-identical:
  `1e88d6b2…840b5`.
- The literal C1 12-epoch grouped-OOF trainer ran twice on a deterministic real
  fit-only dry slice of ten videos (two per fold). Prediction bytes were identical:
  `ed15cfc8…b0ab`.
- Zero-label videos remained inference-only. The runner accepts no validation12,
  CV57, test, scorer, or arbitrary data-root argument.

The official scorer was never imported or called, no graph outcome was opened, and
APNCC/C0 input arrays were not rewritten. RSS/swap-intensive inference was unnecessary;
the dry validation completed in about 3.3 seconds.

## Contract comparison

The runner independently reconstructed N1's canonical hash
`d4a3651a…6f5e9`, asserted the 20,097-parameter ceiling, and failed closed before
work on any mismatch. The implementation therefore matches the single preregistered
design and is ready for a future outcome-opening gate.

## Verdict rationale

**GO.** Every deterministic implementation gate passed twice, exact C0 inputs and
predictions replayed, and no forbidden scope was reachable. Per the shared contract,
this GO cascades to N3's dependency check; it does not itself authorize a graph run.

## Proposed LEDGER row

`C1-TA-HOCT-LITE-N2 | IMPLEMENT | 8/8 tests; C0 195354/195354 repeat; C1 real
fit-only dry OOF byte-identical | 28b544de…f398; no graph score | 0 external usage |
GO | literal 20097-param implementation matches self-hashed N1`
