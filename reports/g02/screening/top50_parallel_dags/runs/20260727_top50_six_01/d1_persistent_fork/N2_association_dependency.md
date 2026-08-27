# D1 N2 — association prerequisite

Date: 2026-07-27  
Verdict: **PARK**

## Gate result

No completed named association arm satisfies the indivisible prerequisite:

`adjusted graph delta >=+0.005 on 44b6 AND 6bba`

with raw edge and node-recall deltas nonnegative on both, plus a positive
increase in distinct connected two-branch opportunities under the frozen D1
grammar.

| Candidate evidence | Authenticated state | 44b6 | 6bba | Fork-opportunity delta | Eligible |
|---|---|---:|---:|---:|---|
| old H1-ZWRFEAT | DEAD structural M5↔T4 drift; graph never ran | null | null | null | no |
| U0-T4CANON | governance N0 GO; no terminal association graph | null | null | null | no |
| C1-TA-HOCT-LITE | mechanism manifest only; branch still active | null | null | null | no |
| C2-KBEST-STABILITY | PARK at C2-N0 dependency | null | null | null | no |

APNCC is the unchanged baseline, not an incremental qualifying arm. Proposal,
motif, cache, parity, and dry-validation diagnostics cannot satisfy this gate.
Partial evidence from different arms is not combined.

Because no qualifying emitted graph exists, the fork-opportunity recount is
undefined and was not fabricated from the cache. N3 44b6 graph execution, N4
6bba, and N5 CV57 remain closed.

## Authenticated evidence at decision time

- `LIVE_STATE.md`: SHA-256 `a960346d…f0da61`
- C2 terminal report: SHA-256 `259edb51…0b0a6`
- U0 N0 contract report: SHA-256 `b5bb9747…e5592`
- U0 draft contract: SHA-256 `08cd90e5…016c1`
- C1 N0 mechanism manifest: SHA-256 `f1227f3d…e770e`
- old H1 cache attestation: SHA-256 `e7c0ca58…1ce3b`

These hashes authenticate the available status inputs; they are not graph
success evidence.

## Resume condition

Retain the frozen N0/N1 foundation. Resume from N2 only after one single named
association arm publishes authenticated terminal 44b6 and 6bba graph vectors
meeting every threshold and its frozen emitted graphs permit a label-free
recount showing a strictly positive connected-fork opportunity delta. Do not
change D1 arms, persistence, suppression, folds, or kill lines on resume.

## Proposed ledger row

`D1-PERSISTENT-FORK-N2 | both-family association dependency | ALPHA DEPENDENCY |
0 qualifying completed arms; old H1 graph null; U0 governance-only; C1 active;
C2 PARK | no D1 graph score | 0 submissions | 0 GPU-h | PARK | retain N0/N1;
resume only on one arm's full both-family graph + opportunity pass |
N2_association_dependency.md`
