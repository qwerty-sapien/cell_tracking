# G2 — owner portfolio decision

> Load `DAG.md`, `SHARED_CONTRACT.md`, `../LIVE_STATE.md`, and
> `runs/<run-id>/G1_portfolio_synthesis.md`. **HUMAN-GATE.**

## Objective / gate

Which smallest set of next actions has earned execution, and what exact resources
would each consume? This node always returns **HUMAN-DECISION**; it never launches
compute, opens CV57, submits, pushes, or mutates an external service.

## Tasks

1. Present the recommended beta control, differentiating alpha(s), expected objective
   leverage, kill line, dependency, runtime/RAM/GPU estimate, and opportunity cost.
2. Separate approvals: local/cache-only graph execution, CV57 confirmation, GPU
   feature generation, and public submission are distinct decisions.
3. Show the no-action alternative and what evidence would reopen any PARKed lane.
4. Draft exact, scoped approval language; do not treat a general desire to improve
   rank as authorization for external effects.

## Output

Write `runs/<run-id>/G2_owner_decision.md` and a proposed decision row. The root
coordinator updates `../LEDGER.md` and `../LIVE_STATE.md` only after the owner decides.

## Verdict

- **HUMAN-DECISION** — always; report the scoped choices and halt.

## Cascade

- On HUMAN-DECISION → halt and ask the owner. No outcome auto-advances.

