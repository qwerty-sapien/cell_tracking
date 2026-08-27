# G0 dispatch manifest — 20260727_top50_six_01

Date: 2026-07-27  
Verdict: **GO**  
Authority: user message `Trigger all six branches now using TRIGGER_ALL_SIX.md.`

## Shared contract

- Objective and gates: `../../SHARED_CONTRACT.md`
- Orchestrator: `../../DAG.md`
- Trigger scope: local/cache-only work and written post-both-family CV57 nodes;
  no GPU, Kaggle, RunPod, paid compute, submission, push, or outward mutation.
- Ownership is disjoint. Only the root coordinator edits `../../../LEDGER.md`.
- Old H1-ZWRFEAT N1 remains DEAD.

## Dispatch records

| Branch | Task owner | State at G0 | Exclusive run output |
|---|---|---|---|
| U0-T4CANON | `/root/run_u0_t4canon` | RUNNING | `u0_t4canon/` |
| C1-TA-HOCT-LITE | `/root/run_c1_hoct_lite` | RUNNING | `c1_ta_hoct_lite/` |
| C2-KBEST-STABILITY | `/root/run_c2_kbest` | RUNNING | `c2_kbest_stability/` |
| D1-PERSISTENT-FORK | `/root/run_d1_fork` | RUNNING (backfilled after C2 terminal) | `d1_persistent_fork/` |
| R0-TA-PU-SHELL | `/root/run_r0_pu_shell` | RUNNING (backfilled after D1 terminal) | `r0_ta_pu_shell/` |
| F0-TRIVIEW-DENSE-TA | `/root/run_f0_triview` | RUNNING (backfilled after R0 terminal) | `f0_triview_dense_ta/` |

The thread exposes three worker slots. Queued branches are immutable dispatch records,
not omitted work; each starts immediately when a slot frees. No branch may merge with
another or inherit a different gate because of queuing.

Backfill log:

- C2 returned terminal `PARK_DEPENDENCY`; its freed slot immediately started D1.
- D1 returned terminal `PARK_DEPENDENCY`; its freed slot immediately started R0.
- R0 returned terminal `PARK_DEPENDENCY`; its freed slot immediately started F0.
- C1 returned terminal `PARK_DEPENDENCY` after N0/N1/N2 GO; no further branch was
  queued, so the slot was released.
- F0 returned terminal `PARK_MISSING_LOCAL_ENCODER_WEIGHTS` at N0; all six roots
  have now been physically dispatched.
- U0 returned `N0/N1/N2 GO → N3 HUMAN-DECISION`; its exact owner lock/reject gate is
  now the global critical path.

## Proposed ledger row

`CTRL-TOP50-G0-20260727 | six-branch parallel dispatch | CONTROL / DISPATCH |
six roots with disjoint ownership | 3 RUNNING + 3 deterministic backfill QUEUED |
no scorer outcome | 0 submissions | 0 GPU-h | 0 RunPod | GO | trigger authenticated;
all dependency and permission locks acknowledged | G0_dispatch_manifest.md`
