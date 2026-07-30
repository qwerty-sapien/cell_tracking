# MAIN N2 result — cheap falsifier design

**Verdict: PARK.** The hypothesis is grounded but no executable, objective-linked
falsifier is currently available in the local environment. No external compute was used.

## Preregistered falsifier

| Item | Contract |
|---|---|
| Input | A synthetic two-candidate association fixture with identical geometry/appearance and one candidate that violates a declared mitotic-duration transition. |
| Null | Shuffle duration labels while retaining all candidate edges; the hazard should lose discrimination. |
| Positive criterion | Hazard ranks the duration-compatible candidate strictly above the incompatible candidate in every declared fixture. |
| Kill line | Any tied/reversed valid-vs-incompatible pair, or nonzero signal after label shuffle, PARKs the mechanism. |
| Integrity | Synthetic inputs only; no videos, supplier weights, cache, 44b6 score, graph action, or campaign data. |
| Resource cap | ≤15 CPU minutes, ≤1 GiB RSS, 0 MiB swap growth. |

## Feasibility check

Static inspection confirms deterministic graph-action fixtures exist in
`tests/test_action_compiler.py`, but it has no phase-duration fixture or phase-template
interface. The available project `pytest` launcher is stale (Python 3.13 missing); the
system Python lacks `pytest`, and its bundled Python lacks `scipy` because importing
`celltrack_alpha` eagerly loads the pipeline. Thus the existing action-compiler test
cannot be executed in this workspace without environment repair, and it would not test
the duration mechanism anyway.

This is a tooling/contract PARK, not evidence against the hypothesis. A new fixture
would need a separate human-approved implementation request before it could earn a
Kaggle protocol.

## Proposed LEDGER row (not written)

`AE-P7-DURATION-HAZARD | PARK | no isolated duration fixture / local test runner unavailable | no compute request`
