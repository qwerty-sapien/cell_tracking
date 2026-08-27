# N5 — 6bba held-fit replication

> Load `DAG.md`, `runs/<run-id>/N4_44b6_graph.md`, and the frozen 6bba held-fit
> manifest. Do not change any representation or fitting choice and do not run CV57.

## Objective / gate

Does the unchanged tri-view alpha replicate on 6bba? **GO** requires adjusted graph
delta `>=+0.005`, raw edge and node-recall deltas `>=0`, and tri-view adjusted score
strictly greater than the XY-only beta. Any adjusted delta `<=0` or protected endpoint
regression is **DEAD** for F0.

## Tasks

1. Score the unchanged four arms on 6bba with authenticated graph/scorer hashes.
2. Report both-family and minimum-family graph vectors, event counts, ablation
   attribution, wall/RAM use, and every negative.
3. Classify F0 as plan/research bet/guess. Do not inspect CV57 or use aggregate
   pooling to hide a family failure.

## Output

Write `runs/<run-id>/N5_6bba_replication.md`, machine-readable scorer artifacts and
hashes, terminal branch classification, exact CV57 approval request if GO, and a
proposed root ledger row.

## Verdict

- **GO** — every 6bba and minimum-family absolute/attribution gate passes; return to
  the global HUMAN-GATE for separate CV57 consideration.
- **PARK** — the absolute gate passes but tri-view fails to beat the XY beta.
- **DEAD** — adjusted delta `<=0`, raw/node regresses, the ablation explains the
  effect, or provenance/leakage fails.

## Cascade

- On **GO** → return the terminal artifact to the global join; do not run CV57.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report.

