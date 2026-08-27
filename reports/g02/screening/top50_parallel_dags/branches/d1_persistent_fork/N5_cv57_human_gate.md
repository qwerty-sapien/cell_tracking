# N5 — CV57 confirmation boundary

> Load `DAG.md`, `runs/<run-id>/N4_6bba_replication.md`, `../../../LIVE_STATE.md`,
> and the global portfolio synthesis. **HUMAN-GATE.**

## Objective / gate

Has D1 earned a separately approved, one-shot CV57 confirmation with no remaining
selection degrees of freedom? This node always returns **HUMAN-DECISION** unless an
explicit approval naming D1 and CV57 is present. It never treats the six-branch trigger
as CV57 authority.

## Tasks

1. Present the fixed arm hash, both-family minimum vector, expected division leverage,
   exact one-shot CV57 command, runtime/RAM estimate, and no-retuning declaration.
2. State the preregistered confirmatory gate: division Jaccard `>=0.30`, adjusted/raw
   edge and node-recall deltas `>=0`, combined score `>0`, and no duplicate fork.
3. Ask for a separate approve/reject decision. Do not execute CV57 in this node.

## Output

Write `runs/<run-id>/N5_cv57_request.md` and a proposed decision row for the root.

## Verdict

- **HUMAN-DECISION** — always; CV57 remains closed pending explicit approval.

## Cascade

- On HUMAN-DECISION → halt and report. No automatic next node.

