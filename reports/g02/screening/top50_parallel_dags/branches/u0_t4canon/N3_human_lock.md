# U0-N3 — Owner lock or reject decision

> Load `DAG.md`, all U0-N0..N2 outputs, `../../../LIVE_STATE.md`, and
> `../../../top50_seeking_alpha_research_20260727.md`. **HUMAN-GATE:** halt even when
> every analytical check passes.

## Objective / gate

Is the U0 dossier complete enough for the owner to lock a new T4-canonical experiment
without amending old N1? Analytical readiness requires all prior nodes GO, exact replay,
zero forbidden-split access, a frozen contract hash, and explicit downstream
44b6-first graph thresholds.

## Tasks

1. Audit that the dossier contains the objective, null, 12 invariants, implementation
   hashes, negative tests, repeat hashes, resource bounds, and all non-GO routes.
2. Re-state downstream gates: 44b6 first; kill at adjusted delta `<=0`; no 6bba unless
   the branch-specific 44b6 threshold passes; raw/node deltas must be nonnegative.
3. State what approval does and does not authorize. It may lock U0 for later local
   cache-only experimentation; it does not authorize GPU, RunPod, CV57 before family
   gates, Kaggle mutation, submission, or resurrection of old N1.
4. Present exact options: `LOCK U0-T4CANON`, `REJECT`, or `REQUEST REVISION`.

## Output

Write `results/U0_N3_owner_dossier.md` with a proposed `U0-T4CANON-LOCK-GATE` ledger
row (`HUMAN-GATE`, `NOT SCORED`, zero external usage). The root coordinator alone
updates the ledger/live decision after the owner acts.

## Verdict

- **HUMAN-DECISION** in all analytically complete cases; halt and ask the owner to
  lock, reject, or revise.
- PARK if evidence is incomplete but repairable.
- DEAD if old N1 would need reinterpretation or any T4 replay failed structurally.
- ESCALATE if the requested authority extends beyond local/cache-only work.

## Cascade

- On HUMAN-DECISION / PARK / DEAD / ESCALATE → halt and report; do not advance.
- If the owner later records `LOCK U0-T4CANON`, return control to the global
  orchestrator; do not start a graph experiment from this node.
