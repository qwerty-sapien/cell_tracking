# U0-N0 — T4-native contract preregistration

Date: 2026-07-27  
Verdict: **GO**  
Evidence class: beta/control governance enabler; not alpha; `NOT SCORED`

## Falsifiable question

Can `U0-T4CANON` be separated from the failed M5↔T4 parity mechanism so that every
fit, calibration, route, graph-mutation reconstruction, and later evaluation decision
is a function only of authenticated T4-domain inputs?

The null was frozen before implementation: any non-exact T4 replay, provenance
mismatch, forbidden-split exposure, M5 decision dependency, or disabled-challenger
APNCC mismatch halts before scoring.

## Result

**GO.** The draft contract defines exactly 12 machine-checkable invariants covering
identity, atomic integrity, conflict membership, offsets, label scope, schema,
candidate rows, grouped folds, reconstruction, scorer identity, output identity, and
exact replay. The scope is exactly 105 fit videos (`36` 44b6 + `69` 6bba),
`195,354` proposals, and `280,095` rows.

M5 has no executable input and cannot select a feature, row, threshold, tolerance,
arm, route, graph mutation, or verdict. It remains optional diagnostic evidence only.
No tolerance is changed. The old node remains
`H1-ZWRFEAT-CACHE-ATTEST = DEAD_STRUCTURAL_M5_T4_DRIFT`.

The production evidence is pinned to the authenticated T4 base report
`80b3d8df…c75c`, T4 feature report `d52a4a42…c0cf`, fit manifest
`ff7b205f…8f28`, cache contract `4da06d48…34b`, APNCC control
`cee547ed…8cf7`, graph-gate contract `d35e2d3d…7f48`, and scorer parity
`d8cfe66b…3cb6`.

U0 is deliberately labeled **beta/control**. Device-pinning is competent governance
that many teams could reproduce; it is only an enabler for the downstream C1/C2
alpha mechanisms.

## Frozen artifact

`results/u0_t4canon_contract_draft.json`

Its SHA-256 is recorded by U0-N1 after the interface is implemented.

## Proposed ledger row

`U0-T4CANON-PREREG | separate T4-native contract | CONTROL / GOVERNANCE | exactly 12 machine-checkable invariants; M5 decision-unreachable; old N1 unchanged DEAD | NOT SCORED | 0 submissions | 0 GPU-h | 0 RunPod | GO | proceed to fail-closed interface | U0_N0_contract_preregistration.md`

## Cascade

GO → `N1_implement_interface.md`.
