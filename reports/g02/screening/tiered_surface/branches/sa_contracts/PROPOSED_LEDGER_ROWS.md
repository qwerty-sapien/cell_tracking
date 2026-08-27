# SA-CONTRACTS Wave-3 proposed ledger rows

| Node | Proposal | Evidence | Bounded interpretation |
|---|---|---|---|
| AE-P5 | PARK-B → PARK-A | `AE_P5_C0_ASSIGNMENT_CONTRACT.json` (`930a3be1…3767`) and P0 (`8ad548c1…9c06`) | Exact C0/C2 contract is committed and the occupied-swap class has +0.075827891 44b6 P0 headroom. Implementation and rank-1 output remain closed. |
| AE-P7 | PARK-B → PARK-A | `AE_P7_PHASE_TEMPLATE_CONTRACT.json` (`ac53b228…cbeb`) and P0 (`8ad548c1…9c06`) | Exact phase-template contract is committed and the occupied-swap class has +0.075827891 44b6 P0 headroom. Fit-only identity/runtime smoke remains. |
| AE-P8 | PARK-B unchanged | `SUPPORT_IDENTITY_RESOLUTION.json` | Adapter identity is committed; supplier authentication and node-specific smoke are not supplied by this lane. |
| AE-P9 | PARK-B unchanged | `SUPPORT_IDENTITY_RESOLUTION.json` | Trackastra and a generic LoRA candidate are authentic, but the exact adapter target/cap/join and fitting authority remain unspecified. |
| G2 / SA-SWAP-W1-EXACT-REGRET | PARK-B → PARK-C for this exact instantiation only | 44b6 report (`62953d27…4864`) | Adjusted −0.000310288, raw −0.000296592, node 0. Broader decision-focused training and AE-P1/P2 are not demoted. |

These are proposals only. The root orchestrator is the sole writer of the census, LIVE_STATE, and LEDGER.
