# N3 — 44b6 held-fit graph kill

> Load `DAG.md`, `runs/<run-id>/N0_metric_hypothesis.md`,
> `runs/<run-id>/N1_decoder_validation.md`, and
> `runs/<run-id>/N2_association_dependency.md`.

## Objective / gate

Does fixed D1 earn meaningful division credit on the hardest family without damaging
association or nodes? **GO** requires 44b6 division Jaccard `>=0.30`, adjusted edge
delta `>=0`, raw edge delta `>=0`, node-recall delta `>=0`, combined-score delta `>0`,
and zero retained duplicate fork. Division Jaccard `<=0.15`, any negative protected
endpoint, or any retained duplicate is **DEAD**.

## Tasks

1. Run unchanged, D0 beta, D1 alpha, and persistence-ablation with identical manifests
   and seeds on the locked 44b6 held-fit graph gate.
2. Report the full graph vector and per-video event counts; never promote on motif
   precision or a proposal metric.
3. Attribute the alpha only if D1 strictly exceeds D0's combined score and the
   persistence ablation does not explain the effect; otherwise retain any valid D0
   result separately and PARK D1.

## Output

Write `runs/<run-id>/N3_44b6_graph.md`, machine-readable scorer outputs, emitted and
suppressed fork audit, hashes, and a proposed ledger row.

## Verdict

- **GO** — every absolute threshold passes and the D1-vs-D0 attribution test passes.
- **PARK** — division Jaccard is `(0.15, 0.30)`, or D1 passes the absolute gate but
  does not beat D0; retain the beta result separately.
- **DEAD** — division Jaccard `<=0.15`, a protected endpoint is negative, a duplicate
  survives, or provenance/leakage fails.

## Cascade

- On **GO** → proceed directly to `N4_6bba_replication.md`.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not run 6bba.

