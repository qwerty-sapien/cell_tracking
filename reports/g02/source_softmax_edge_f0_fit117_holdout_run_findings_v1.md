# Source-softmax edge fold-0 holdout findings

Scope: frozen fold-0 held-out videos from
`reports/g02/grouped_3fold_audit20_v1.json`, evaluated by the frozen official
scorer. Kaggle is read-only for this work; no submission is authorized.

## Finding F1 — the 0.90 gate is arithmetically unreachable and closed

Status: **verified from the 52 sample rows** in
`reports/g02/source_softmax_edge_f0_fit117_holdout_prefix52_v1/scores.json`.

For each scored video, define
`w_i = edge_tp_i + edge_fp_i + edge_fn_i`. Direct aggregation gives:

- `n = 52`
- `W52 = sum(w_i) = 34,146`
- `sum(edge_tp) = 29,676`
- `sum(edge_fp) = 2,660`
- `sum(edge_fn) = 1,810`
- `sum(w_i * adj_edge_jaccard_i) / W52 = 0.8550656350917426`, exactly
  reproducing the reported summary

The 31 scored `6bba` videos have `sum(w_i) = 28,037`, hence mean
`w_i = 904.4193548387096`. Using that observed scenario mean for the six
unscored `6bba` videos gives inferred `W6 = 5,426.5161290322585`.

With `s52 = 0.8550656350917426`, the optimistic combined-score ceiling is

`(s52 * W52 + cap * W6) / (W52 + W6)`.

- At `cap = 1.00`, the ceiling is `0.874940253785718`.
- At `cap = 1.07`, the ceiling is `0.8845392423310426`.

Both are below 0.90. Conversely, the missing weight required for 0.90 is

`W6_required = (0.90 - s52) * W52 / (cap - 0.90)`.

- At `cap = 1.00`, `W6_required = 15,343.28824157358`.
- At `cap = 1.07`, `W6_required = 9,025.463671513868`.

These requirements exceed the scenario-mean estimate of 5,426.52. Therefore
the `>= 0.90` pass/fail gate is closed for this run on arithmetic evidence.
The remaining videos are still evaluated to obtain the complete citable
58-video number and to test whether division behavior changes by scope; they
are not evaluated as a 0.90 gate check.

## Finding F2 — prefix-52 edge error budget

Status: **verified from the same 52 sample rows and the frozen metric
formula**. Full derivation and formula citations are recorded in
`reports/g02/source_softmax_edge_f0_fit117_holdout_prefix52_v1/error_budget.md`.

The exact loss from an ideal adjusted edge score of 1.0 decomposes as:

- edge FP: `2,660 / 34,146 = 0.0779007790` score loss (`53.7490%` of
  total loss);
- edge FN: `1,810 / 34,146 = 0.0530076729` (`36.5736%`);
- signed node adjustment: `478.928824 / 34,146 = 0.0140259130`
  (`9.6774%`).

The node term consists of `0.0158605830` gross over-prediction penalty minus
`0.0018346700` under-prediction bonus. Removing the entire net node loss alone
would raise the score only to the unadjusted edge score `0.8690915481`.

The adjusted-numerator deficits are `1,534.328824` at 0.90 and
`2,217.248824` at 0.92: respectively 1,535 and 2,218 idealized unit-gain
score-equivalent events. Under the actual frozen formula, holding node ratios
fixed and optimally allocating available repairs, optimistic lower bounds are
1,558 corrected metric error labels for 0.90 and 2,292 for 0.92. These count
FP/FN labels, not graph-edit operations: one rewire may remove an FP and
recover an FN together.

## Finding F3 — the scored detector threshold is 0.970, not 0.985

Status: **verified from runtime code and the dry-run execution hash**. Full
evidence is recorded in `reports/g02/node_threshold_investigation_v1.md`.

The one-pass detector is locked to `0.970` in
`src/celltrack_alpha/bh_model.py`; `src/celltrack_alpha/v45_fast.py` applies
that detector without an override, and `scripts/v45feval.py` calls the fast
inference path without a threshold argument. Recomputing the one-view config
hash at 0.970 gives
`66009cdfc2cd74f4f9906073bdafe9201deba0eaf7fda1899cbd0aeb489bd575`,
exactly matching the frozen dry-run contract and the per-video reports.
Therefore the prefix-52 and completed-58 scores from this run are 0.970
results. The presumed frozen 0.985 value was not wired into this evaluator.

A prior fit-only 14-video grid selected 0.985 globally and for both scenarios
under a maximum per-video node-recall-loss guard of 0.002. Relative to 0.970,
the verified but provisional mean adjusted-edge changes on that grid are:

- all 14 videos: `0.824831419 -> 0.829188086` (`+0.004356667`);
- five `44b6`: `0.753033073 -> 0.760315099` (`+0.007282026`);
- nine `6bba`: `0.864719389 -> 0.867450856` (`+0.002731468`).

The grid is not an official holdout comparison: it is 14/117 fit videos, uses
released sequential-D4 detections and inherited motion association, and stores
per-video means rather than edge-event-weighted aggregates. The frozen
holdout fragments omit node confidence, so 0.985 cannot be evaluated post hoc;
it requires fresh one-pass inference and official rescoring. No runtime
threshold change is made on this provisional evidence.

## Finding F4 — division failure is structural connectivity, not parameter drift

Status: **verified on all 37 GT division events in the prefix-52**. Full
metric/code trace and the known-GT replay are recorded in
`reports/g02/division_bug_investigation_holdout_prefix52_v1.md` and
`tests/test_bh_division_holdout.py`.

Post-patch revalidation at official checkout `075fc5f` preserves the core
finding: the held-out diagnostic bridge still changes divider 3 from FN to TP
because it forms a valid local matched-parent fork with two distinct matched
daughter branches. The former `35/37` weak-component classification remains a
pre-patch structural measurement and is not the patched scorer's decision
procedure. On the current division-off D4 57-video output, patched
classification is instead 39/42 strict-role-covered events with no local
parent fork and 3/42 with a missing matched role; see
`reports/g02/division_failure_holdout57_postpatch_v1.json`.

The scorer requires matched prediction nodes covering the pre-split stage and
both daughter lineages in one weakly connected component that contains a
prediction fork. Exact replay classified 35/37 GT events as having the needed
matched stage coverage split across multiple prediction components; 2/37 miss
a required stage. No event has a spanning component or a scorer-eligible fork,
reproducing zero true positives before global fork pairing.

The focused `44b6_267148e4` division has one matched chain
`415(t2)->596(t3)->792(t4)->992(t5)` and a disconnected second daughter chain
`800(t4)->1008(t5)`. Adding the held-out diagnostic bridge `596->800` changes
that GT division from FN to TP, proving connectivity is sufficient, but this
oracle edge is not an eligible production rule.

The distance tests use the correct pairs. For that required bridge the
prediction-space parent/existing, parent/candidate, and sister distances are
`1.816805`, `9.951052`, and `11.170028` um: the latter two exceed the current
`4.66` and `8.5` um gates. Fit-only geometry shows the inherited envelope
covers only `19/83 = 22.891566%` of fit GT divisions. Postprocessing preserves
division components and does not remove the emitted forks.

The constants do match the scored V45 reference; the failure is the local
orphan-node objective versus the scorer's lineage-connectivity objective.
Widening geometry or applying the held-out oracle bridge is not supported.
Production divisions therefore default OFF, with explicit opt-in retained for
future fit-derived work. The known-GT test now freezes the post-patch baseline
failure and the diagnostic bridge TP; the bridge remains ineligible as a
held-out oracle. It is inferred, not individually edge-match-verified, that the
false forks also increase edge FPs.

This result must be read beside the fixed-node oracle at
`reports/g02/t04_v45f4_fold0_fixed_node_oracle/`. With its prediction nodes,
coordinates, and node penalty fixed, held-out-GT-supported association rewiring
scores adjusted edge `0.9749936254`, edge `TP/FP/FN = 36,909/0/324`, and
division `TP/FP/FN = 37/0/5` (`division_jaccard = 0.8809523810`). The oracle is
diagnostic and ineligible for selection or submission, but it establishes that
association fragmentation is the upstream cause shared by the edge deficit and
the zero division term. The 22.9% division-geometry envelope is therefore a
symptom of missing intermediate connectivity, not a reason to widen gates or
add re-parenting.

## Finding F5 — terminal result is complete at 57; association fragmentation dominates

Status: **verified by the frozen scorer on the prefix 52 plus five completed
suffix fragments**. Full evidence and break-level records are in
`reports/g02/source_softmax_edge_f0_fit117_holdout_complete57_v2/`.

By explicit direction, `train:6bba_f4ae811c` is excluded from the terminal
scope. Its evaluator finished while cancellation was being delivered, so the
fragment is preserved. A previously launched background recovery later
materialized a 58-video directory after the stop instruction; it is preserved
but is out of scope and is not used for the terminal result or fragmentation
analysis.

The 57-video result is:

- adjusted edge Jaccard `0.8495036625`;
- raw edge Jaccard `0.8635120766`;
- edge `TP/FP/FN = 34,107/3,230/2,161`;
- division `TP/FP/FN = 0/149/42`, division Jaccard `0.0`.

The prefix-52 pooled prediction component size is `23.3652` nodes versus
`31.8989` for GT weak lineages, reproducing the reported 23.4 versus 31.9.
`542/1,019 = 53.19%` of GT weak lineages are split across multiple prediction
components. On 57 the values are `23.7173` versus `30.7767`, with
`674/1,218 = 55.34%` fragmented.

All prefix FNs are assigned to exact observable classes: 271 unmatched
endpoints, 86 synthesized-gap endpoints, 336 final matched pairs above 10 um,
15 division-degree constraints, 1,094 occupied-endpoint conflicts, and 8 cases
with no observed endpoint conflict. No correct matched target ranks beyond 8
among surviving final nodes. The original raw candidate tensors and logits were
not persisted, so “gate exclusion” versus “out-ranked” cannot be split exactly;
the report keeps that inference separate from the verified topology.

The crowding claim reproduces on prefix 52 with Pearson `-0.701770` versus
`-0.083338` for node ratio. On 57 the raw Pearson is `-0.464043`, weakened by
three high-peak suffix outliers, but Spearman remains `-0.780080` and Pearson
against `log1p(peak)` is `-0.689825`. Event-level break risk is 28.125% above
the cap-8 final-node neighborhood versus 5.112% at or below it (`5.50x`).

The decision is to target crowded-component association continuity. Do not
widen division geometry, add a division head, tune division constants, or
apply generic re-parenting. Divisions now default OFF in production, with
explicit opt-in retained.
