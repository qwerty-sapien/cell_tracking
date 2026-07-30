# U0-T4CANON terminal report — Run 20260727_top50_six_01

Terminal verdict: **HUMAN-DECISION** at U0-N3  
Old H1 N1: unchanged `DEAD_STRUCTURAL_M5_T4_DRIFT`  
External usage: zero

## Node results

| Node | Verdict | Decisive result |
|---|---|---|
| N0 | GO | exactly 12 machine-checkable T4-only invariants |
| N1 | GO | 12/12 tests; 105+105 payload/sidecar coverage |
| N2 | GO | two byte-identical 105-video replays; exact APNCC disabled control |
| N3 | HUMAN-DECISION | owner must lock, reject, or request revision |

The exact replay hash is
`1d1a8ac2273acb22aa68f097cd3645fce4929f71faf7fba9f029aaab932b4758`.
All five 44b6 folds have no admissible threshold under the unchanged calibration
rule; this is retained as negative evidence, not tuned away.

Owner dossier:
`../../../branches/u0_t4canon/results/U0_N3_owner_dossier.md`.

## Proposed ledger rows

1. `U0-T4CANON-PREREG` — GO, `NOT SCORED`, zero external usage.
2. `U0-T4CANON-IMPLEMENT` — GO, 12/12 tests, `NOT SCORED`, zero external usage.
3. `U0-T4CANON-REPLAY` — GO, exact replay, `NOT SCORED`, zero external usage.
4. `U0-T4CANON-LOCK-GATE` — HUMAN-DECISION, zero external usage.

Only the root coordinator may append them.
