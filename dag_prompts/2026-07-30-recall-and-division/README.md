# 2026-07-30 owner-directed DAGs — recall supply + division checkpoint pilot

Two independently dispatchable lanes, created on owner instruction (2026-07-30):

1. **`recall/`** — reopen the orphaned recall/proposal-supply question. `H2` in the
   human-locked plan ("a learned no-link model makes the recall shell reachable") was
   explicitly conditioned on `H1`'s frozen-Trackastra features. `H1` is DEAD at the
   44b6 graph gate, so `H2` lost its enabler and nothing replaced it. Every retained
   branch is now a selector over a fixed candidate pool whose in-pool re-ranking
   projection is `~0.896` local, while the public `0.910` bar needs roughly
   `+0.044–0.061` local over banked APNCC. On the frozen pool the free-add and
   node-add oracles are exactly `0`, so selection cannot supply the gap. This lane
   asks whether **candidate supply** can.
2. **`division_ckpt/`** — the owner-approved one-week pilot on a *frozen external*
   division representation checkpoint. This is the only division route that avoids
   the `5 × 8 h = 40 T4-h` fit-only training lower bound established in
   `../2026-07-30-continuation/division/N3_THREE_STAGE_WEEKLY_PLAN.md`.

Dispatch starts at `N0` in each lane and auto-cascades **only** across `GO`. Pasteable
dispatch text is in `TRIGGER.md`.

| Lane | Dispatch root | Week-1 compute |
|---|---|---|
| recall supply | `recall/N0_pool_recall_ceiling.md` | 0 GPU-h through N2; N3 is a HUMAN-GATE |
| division checkpoint pilot | `division_ckpt/N0_checkpoint_provenance.md` | 0 GPU-h through N1; N2/N3 are HUMAN-GATEs |

The division lane's owner continuation gate is preregistered in
`division_ckpt/N3_efficacy_gate.md`: `>= 4` zero-FP division TP on CV57
(`division_jaccard >= 0.08022`, local `>= 0.889544`), layered on the unchanged
`+0.005` adjusted-edge floors on 44b6 and 6bba.

## Marginal-value gate (mandatory before any repeat-compute request)

Before requesting a second or later round of compute in either lane, read
`reports/g02/screening/MARGINAL_VALUE_GATE.md` and answer its checklist in writing. State
the realistic (not best-case) ceiling using any applicable campaign precedent collapse
ratio, compare it to the actionability floor, and STOP at the mechanism-*class* level
(not just the tested instance) if the limiting factor is structural rather than something
the next iteration could move. The recall lane's `N0B_RECONCILIATION.md` is the worked
example: it found the pool-absence bottleneck is action-vocabulary legality, not
candidate supply, and closed the whole candidate-generation class on that basis rather
than funding a second generator.

## Standing rules (inherited; no prompt here relaxes them)

- Read `reports/g02/screening/LIVE_STATE.md` first; its no-retry and terminal rules
  override every prompt in this tree.
- The objective is always the **frozen graph-level scorer**, 44b6 first, then 6bba,
  then CV57. Proposal precision / AUC / utility gain are proven not to predict it and
  may be reported, never gated on.
- Full frozen 44b6 is mandatory. No `SUBSAMPLE_CONTRACT` exists (0/12 usable cells).
- Only the root coordinator writes `reports/g02/screening/LEDGER.md`; nodes propose
  rows, they do not write them.
- No DEAD mechanism may be re-screened without a new named mechanism (LIVE_STATE DEAD
  list / DAG §6.1). For the recall lane this specifically bars `R12SHELL`, `MGSHELL`,
  `TEMPBG`, `POINTCTX`, `CROPLINK`, `TEMPRANK`, `TEMPHYBRID`, `TRACKCTX`, scalar
  recalibration/thresholding, hand-tuned graph-cost knobs, and organizer
  `TemporalUNet3D` (~478 T4-h).
- No Kaggle asset, kernel, GPU debit, submission, or push without explicit owner
  approval. RunPod is forbidden. Local runs keep the 45-second idle preflight and the
  zero-swap rule.
- Launch capacity is the lower owner-authenticated balance, **25.47 GPU-h**, under a
  30 h/week cumulative cap. Do not plan against the repo subtotal.
- No parallel GPU bets. At most one authorized route is in flight at a time.
