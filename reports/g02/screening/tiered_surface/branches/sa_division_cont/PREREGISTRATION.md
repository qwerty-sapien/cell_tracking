# SA-DIVISION-CONTINUITY — frozen oracle target derivation

Date: 2026-07-29  
Status: **FROZEN BEFORE TARGET DERIVATION / FITTING LOCKED**

This lane converts the banked Wave-2 classification of 38/42 CV57 division
events as `PRESENT_FRAGMENTED` into a per-event target specification for a
future association-continuity arm. It does not fit a model, score a graph,
change a tier, or open CV57 as a tuning set.

Frozen inputs:

- Wave-2 audit
  `reports/g02/screening/tiered_surface/branches/sa_division_deep/oracle_audit_v2.json`,
  SHA-256 `6d1a2754bbe597d53c29863290cba51225d3138b002b83ac4d4ad9c5d306c119`.
- Frozen N1 interface SHA-256
  `2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865`.
- Grouped folds SHA-256
  `ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769`.
- P0 action oracle SHA-256
  `8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06`.

For every fragmented event the derivation must record:

1. the exact parent and two child endpoint identities;
2. required, already-correct, net-add, net-delete, and owned-conflict edges;
3. a deterministic direct-repair motif;
4. whether a one-frame-late scorer-equivalent fork is supported by two
   distinct grandchild-stage persistence nodes;
5. the exact preferred late bundle and deterministic tie-break if available.

The diagnostic may use frozen official match identities only to enumerate
oracle target classes. A future fitted association arm must recover an
analogue label-free. The target specification is not a feature, training
label, or permission to fit D1.

The dependency remains unchanged: division fitting stays closed until one
single named association-continuity arm passes adjusted edge-Jaccard
`>= +0.005` with raw edge and node recall deltas `>= 0` on both 44b6 and
6bba and produces a positive label-free connected-fork-opportunity delta.
No family splicing is allowed.
