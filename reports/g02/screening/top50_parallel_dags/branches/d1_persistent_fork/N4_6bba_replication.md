# N4 — 6bba held-fit replication

> Load `DAG.md`, `runs/<run-id>/N3_44b6_graph.md`, and the frozen 6bba held-fit
> manifest. Do not change the decoder, folds, thresholds, or suppression rule.

## Objective / gate

Does the unchanged D1 mechanism replicate on 6bba? **GO** requires 6bba division
Jaccard `>=0.30`, adjusted/raw edge and node-recall deltas all `>=0`, combined-score
delta `>0`, zero retained duplicate fork, and D1 combined score no worse than D0.
Division Jaccard `<=0.15` or any protected-endpoint regression is **DEAD**.

## Tasks

1. Re-run the four frozen arms on 6bba with the authenticated graph/scorer.
2. Report per-family and minimum-family vectors, TP/FP/FN, duplicate audit, and the
   preregistered stretch diagnostic at division Jaccard `>=0.50`.
3. Do not pool families to conceal a 6bba failure and do not inspect CV57.

## Output

Write `runs/<run-id>/N4_6bba_replication.md`, exact scorer artifacts and hashes, a
terminal D1 evidence classification, and a proposed root ledger row.

## Verdict

- **GO** — every 6bba and minimum-family requirement passes.
- **PARK** — division Jaccard is `(0.15, 0.30)` or alpha attribution is absent while
  the D0 beta remains viable.
- **DEAD** — division Jaccard `<=0.15`, any protected endpoint regresses, a duplicate
  survives, or provenance fails.

## Cascade

- On **GO** → proceed directly to `N5_cv57_human_gate.md`.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report.

