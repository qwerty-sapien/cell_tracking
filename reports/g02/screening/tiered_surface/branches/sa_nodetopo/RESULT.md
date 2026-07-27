# NODETOPO-O0 prerequisite result

Verdict: **PARK — no complete authentic candidate universe.**

The candidate-origin procedure is real and label-independent at generation time:
`generate_candidates` runs on raw images before the legacy builder joins sparse GEFF
labels. That is not sufficient to open N4. The only persisted candidate tables are
fold-0 training artifacts with GT-derived columns, one three-frame clip per dataset,
and partial fit105 coverage: **4/36 44b6** datasets and **3/69 6bba** datasets. The
label columns were audited as contamination and were never used to select candidates.

The local extracted raw evidence is also incomplete: 12 training Zarrs, of which 10
overlap fit105. A local raw archive registry proves that more raw videos exist, but it
does not prove that a frozen full candidate bank exists. Generating a new pseudo-mask,
inventing coordinates, or substituting P0's GT restore oracle would violate the frozen
contract.

Two prerequisites are absent:

1. An exact deployment-population background. P0 itself records node-restore
   background as `UNDEFINED` with a null count.
2. An immutable join from each candidate identity to the frozen baseline node IDs,
   all added incident edges, all displaced incoming/outgoing conflicts, and the exact
   N1 expected-conflict boundary.

Therefore zero real actions were compiled and no 44b6 graph was scored. The synthetic
interface test confirms the semantics: one complete `restore_topology` action compiles
and changes the graph hash, while an isolated coordinate-only action is rejected.

`NODETOPO-O0` retains its banked P0 PARK-A headroom; this audit proposes no tier move.
`NODERESTORE-O0` remains PARK-B. Restoration fitting, 6bba, CV57, GPU, submission, and
external actions remain closed.

Resume only when root freezes a full fit105 label-independent candidate bank, exact
unlabeled background, and candidate-to-baseline incident-topology join by hash.

Direct metric source/transfer domain: the competition's official
[metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
is the exact task domain. Ultrack's multi-hypothesis selection is an adjacent
microscopy rationale only; no Ultrack hypothesis or segmentation mask was produced
or treated as benchmark evidence here.
