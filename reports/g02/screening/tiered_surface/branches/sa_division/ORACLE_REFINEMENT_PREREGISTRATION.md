# SA-DIVISION division-oracle refinement preregistration

Frozen: 2026-07-28, before executing the refinement script.

## Decision boundary

This refinement is a diagnostic read of the already-banked P0 `division_fork`
action. It does not fit a division model, execute a held-family graph, change
the D1 arm, open CV57, or propose a tier move.

Division fitting remains fail-closed unless the existing dependency audit
contains one immutable association-continuity arm that passes both `44b6` and
`6bba` at:

- adjusted edge-Jaccard delta `>= +0.005`;
- raw edge-Jaccard delta `>= 0`;
- node-recall delta `>= 0`;
- a positive connected-fork opportunity delta under the frozen D1 grammar.

The current banked dependency audit has zero qualifying arms. The refinement
must assert that state before reading P0 outcome fields and terminate without
fit if it is not exactly true.

## Frozen inputs

All shared inputs are read-only absolute paths under
`/Users/agent/Desktop/cell_tracking`.

| Input | SHA-256 |
|---|---|
| N1 interface | `2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865` |
| N1 implementation | `457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5` |
| grouped folds | `ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769` |
| P0 action-class oracle | `8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06` |
| D1 preregistration | `34b2aa020099e14ad7a73820d97e9baefadec386e0e2d9acbd7ab5c1cb07620f` |
| D1 arm manifest | `327af0d29dbde19172a61bf396b951d150040bb357c3fbba90e79249035e4776` |
| banked SA-DIVISION dependency audit | `38990d30d4b30ea7afbf99e0a655b0150c34a73fdd4918f93178316d9d9e3c76` |

## Predetermined calculations

For each family, take only P0's frozen control and `division_fork` rows and:

1. Recompute each reported delta as `oracle - control`.
2. Recompute the official combined score as
   `adjusted_edge_jaccard + 0.1 * division_jaccard`.
3. Record positive exact actions and non-GT daughter-pair background support.
   Their ratio is a population diagnostic only, never a promotion gate or a
   calibrated false-positive probability.
4. Compare P0's division Jaccard with the unchanged D1 thresholds:
   `GO >= 0.30`, `null <= 0.15`.
5. Holding oracle division TP and FN fixed, compute the largest integer division
   FP count compatible with Jaccard `>= 0.30`:

   `floor(TP / 0.30 - TP - FN)`.

   Report the minimum FP reduction from P0's observed division FP needed to
   reach that envelope. This counterfactual is diagnostic and does not assert
   that any policy can realize the removals.

## Fail-closed invariants

- Exact hashes above must match before P0 outcomes are consumed.
- Dependency qualifying-arm count must equal zero and its verdict must remain
  `PARK_DEPENDENCY`.
- D1's four arms, grammar, thresholds, stable order, and dry-validated manifest
  remain unmodified.
- P0 must identify the common fit117 CG0 control, not APNCC.
- Both family identities must be exact and complete.
- No fitting, training, graph execution, CV57, GPU, Kaggle, submission, RunPod,
  external action, plan amendment, DEAD action, or tier move is permitted.
- Memo C2/K-best and Branch-B `66/223` values remain quarantined.

The only successful terminal status is
`PASS_DIAGNOSTIC_PARK_DEPENDENCY`. Any mismatch is an error and opens no
scientific surface.

## Frozen errata

The initial commit used `f` instead of `c` at zero-based digest index 53. The
first guarded attempt failed at authentication before loading the P0 JSON. The
first repair misdiagnosed the visual difference as a duplicated `a`; retry 1
also failed at authentication before loading P0. `ORACLE_REFINEMENT_REPAIR_V2.md`
freezes the exact indexed correction to the already-banked digest above. No
calculation, threshold, input path, or scientific rule changed.
