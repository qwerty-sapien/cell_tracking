# Proposed ledger rows

Root coordinator only; these rows are proposals and do not edit `LEDGER.md`.

| Date | Node | Tier | Decision | Evidence | Rationale |
|---|---|---|---|---|---|
| 2026-07-28 | SA-SWAP-W1-EXACT-REGRET preregistration | BANKED | LOCKED | `PREREGISTRATION.json` (`00e6036e…4628`) | Frozen before any valid held outcome; 44b6-first graph gate, no proposal-metric gate. |
| 2026-07-28 | SA-SWAP execution repair | BANKED | PASS | `POST_REPAIR_EXECUTION_MANIFEST.json` (`0f02903f…d2282a`) | 18/18 guarded tests, bitwise streamed parity, exact N1 locality proof; zero swap growth. |
| 2026-07-28 | SA-SWAP-W1-EXACT-REGRET / 44b6 | PARK-C | KILL_44B6 | `result_44b6_postrepair_v1/report.json` (`62953d27…4864`) | Adjusted delta `-0.0003102875275311501`, raw delta `-0.00029659222678224406`, node delta `0`; 45,961/45,961 N1 witness compiles, zero rejects. Exact instantiation must not be rerun. |
| 2026-07-28 | SA-SWAP-W1-EXACT-REGRET / 6bba | CLOSED | NOT_RUN | same terminal report | 44b6 failed the required GO gate, so 6bba was not authorized. |
