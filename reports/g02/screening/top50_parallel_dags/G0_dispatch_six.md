# G0 — dispatch all six branch roots

> Load `DAG.md`, `SHARED_CONTRACT.md`, `../LIVE_STATE.md`, all six branch `DAG.md`
> files, `.agents/skills/seeking-alpha/SKILL.md`, and
> `.agents/skills/dag-prompts/SKILL.md`.

## Objective / gate

Can all six named branch roots be dispatched with disjoint ownership, the shared
scientific contract, and every dependency/permission stop intact? **GO** requires six
recorded dispatches, six owned output paths, and explicit acknowledgement of all
locks; any missing or conflicting ownership is **ESCALATE**.

## Tasks

1. Reconcile `LIVE_STATE.md` with each branch root; never restart a completed or DEAD
   node.
2. Assign exactly one worker/task to each branch. If fewer than six worker slots are
   available, enqueue the remainder immediately without merging branches or changing
   their contracts.
3. Give every worker `SHARED_CONTRACT.md`, its branch `DAG.md`, and an exclusive
   output directory.
4. Record task identifiers, start/queue state, and the first dispatchable node.
5. Do not execute a branch's downstream experiment merely because its root was
   dispatched.

## Output

Write `runs/<run-id>/G0_dispatch_manifest.md` with all six task records and return one
proposed `LEDGER.md` row for the root coordinator.

## Verdict

- **GO** — six branch roots are dispatched or deterministically queued with no
  ownership collision and every lock acknowledged.
- **ESCALATE** — a branch is missing, ownership overlaps, or a required contract
  conflicts with live state.
- **HUMAN-DECISION** — dispatch itself would require a new outward-facing resource.

## Cascade

- On **GO** → proceed directly to `G1_join_and_synthesize.md` in the same run; wait
  for all six terminal artifacts without asking the human between branch nodes.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.

