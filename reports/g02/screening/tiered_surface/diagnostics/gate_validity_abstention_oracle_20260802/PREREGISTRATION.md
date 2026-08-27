# Gate validity and abstention-oracle diagnostic

Status: **FROZEN BEFORE D1 CONTROL SCORE OR D2 ORACLE READ**  
Date: 2026-08-02  
Scope: local CPU and existing caches only; no GPU, Kaggle, RunPod, submission,
`6bba`, or CV57 score access.

## D1 — retrospective 44b6 gate validity

- Challenger: the banked `GA-APNCC-DIRECT-20260724` local, node-disjoint direct
  swap realization, restricted to the same 36 held-fit `44b6` videos.
- Control: the exact pre-APNCC backward-motion, divisions-off `c=0.25` graph on
  those same identities. A control assembled from a different mechanism, family,
  partition, or video set is invalid.
- Scorer: the frozen official scorer and its campaign node-adjustment rule. Reuse
  exact banked per-video fragments when authenticated; otherwise do not infer a
  score from edge counts or the CV57 aggregate.
- Report: absolute challenger and control adjusted/raw/node vectors and their
  componentwise delta.
- Counterfactual decision: `PASS` iff adjusted delta is at least `+0.005`, raw
  delta is nonnegative, and node-recall delta is nonnegative. Otherwise `FAIL`.
- Interpretation frozen in advance: if the sole confirmed CV57 winner fails this
  retrospective gate, the `44b6`-first rule is not a valid necessary filter for
  CV57 improvement on the banked evidence. This is a methodological finding, not
  authority to change the human-locked gate.

## D2 — opportunity support versus selector abstention

An action is beneficial only when its exact individual frozen-scorer delta
`S(G_action) - S(G_incumbent)` is strictly positive. If an existing population
label is used, its construction must be authenticated as exactly this predicate;
classification correctness alone is not a substitute.

The following candidate universes are fixed before their oracle labels are read:

1. **A7:** all 45,961 frozen N1-valid `44b6` occupied-endpoint swap actions named
   by A7's preregistration, before A7's nested phase/template threshold. Also
   report its post-threshold and applied support (`0` expected from the terminal
   selection receipt) to identify complete abstention.
2. **Reverse consensus:** report all three nested supports: the common 45,961
   occupied-swap opportunity population; the 268 actions passing the frozen
   authentic bidirectional evidence predicate before caps/ownership; and the 53
   canonical node-disjoint selected actions. The 268-action set is the mechanism's
   primary evidence-qualified candidate set for the D2 verdict.
3. **R0:** all 38,081 `topology_code == 0` free/free 10–12 micrometre shell actions
   in the 36 label-blind shards, before the sealed B=4 ranking. Also report the
   selected four.

Verdict language is frozen: a nonempty beneficial candidate universe followed by
zero beneficial accepted edits is a selector/abstention failure and cannot justify
an evidence-park for the underlying opportunity population. Zero beneficial actions
in the fixed evidence-qualified universe is an evidence-negative result. No tier or
gate changes are authorized by this diagnostic.

## D3 — near-zero 44b6 kill review

- Read only banked exact `44b6` association-candidate results in campaign artifacts
  and the coordinator ledger.
- Include every valid candidate killed at the `44b6` gate with
  `abs(adjusted_delta) < 0.001`; exclude controls, invalid/parity-failed runs,
  division-specific gates, and exact no-op abstentions from D2.
- Rank by descending adjusted delta (closest positive first), then ascending
  absolute delta.
- A proposed CV57 read means one irreversible full-CV57 outcome exposure using
  existing candidate fragments/scorer inputs; estimate CPU/materialization cost
  from banked receipts where available. Do not open or score CV57.
- Recommendation depends on D1: if D1 fails, near-zero candidates are not ruled out
  by 44b6 direction alone, but a request is worthwhile only when the candidate is
  observation-distinct, its CV57 materialization is already cached/cheap, and its
  mechanism is compatible with local, narrow, node-disjoint, recall-preserving
  edits. This diagnostic itself authorizes nothing.
