# Frozen fold-0 association fragmentation audit (complete at 57)

## Scope and terminal run status

Status: **complete at 57 by explicit direction**. The selected scope is the
frozen fold-0 set minus `train:6bba_f4ae811c`; the exact exclusion is locked in
`scripts/analyze_holdout_fragmentation.py:54,738-740` and recorded in
`fragmentation.json:67576-67643`. The excluded evaluator finished while its
cancellation was being delivered, so its fragment is preserved. A previously
launched background recovery later materialized a 58-video directory after the
stop instruction; that artifact is also preserved but is explicitly outside
the terminal scope and is not used anywhere in this audit.

Before this analysis, the complete repository state was preserved on branch
`codex/fragmentation-57-rollback` at commit `7c3f56bc52a6`. It was not pushed.
No model training, Kaggle write, submission, paid compute, public artifact,
split change, or scorer change was performed.

The citable 57-video result is in `scores.json:2061-2073`:

- adjusted edge Jaccard: `0.8495036625176058`;
- raw edge Jaccard: `0.8635120765608385`;
- edge `TP/FP/FN = 34,107 / 3,230 / 2,161`;
- division `TP/FP/FN = 0 / 149 / 42`, division Jaccard `0.0`;
- combined score: `0.8495036625176058`.

This is a descriptive terminal result, not a pass/fail test against 0.90. The
arithmetic ceiling finding remains closed.

## Root-cause synthesis: F4 beside the fixed-node oracle

### Verified

The division replay in
`reports/g02/division_bug_investigation_holdout_prefix52_v1.md:22-65` found:

- 35/37 GT divisions have every required matched stage, but those stages are
  split across prediction components;
- 2/37 miss a required matched stage;
- 0/37 have a spanning component or scorer-eligible fork;
- the focused missing bridge is 9.951052 um from parent to candidate and
  11.170028 um sister-to-candidate, beyond the inherited 4.66/8.5 um gates;
- the fit-only inherited geometry envelope covers only `19/83 = 22.891566%`
  of real divisions.

The fixed-node oracle freezes its source prediction nodes, coordinates, and
node-count penalty, then replaces association edges with held-out-GT-supported
edges (`reports/g02/t04_v45f4_fold0_fixed_node_oracle/report.json:1`). Its full
58-video official score at
`reports/g02/t04_v45f4_fold0_fixed_node_oracle/scores.json:2626-2638` is:

- adjusted edge Jaccard `0.9749936253918458`;
- edge `TP/FP/FN = 36,909 / 0 / 324`;
- division `TP/FP/FN = 37 / 0 / 5`, division Jaccard
  `0.8809523809523809`.

The oracle is diagnostic only: it uses held-out labels and is ineligible for
candidate selection or submission. Nevertheless, together these two results
show that zero division credit and most edge loss share the same upstream
failure: association fragments matched lineage nodes into separate components.

### Inference and decision

The 22.9% division-geometry coverage is a downstream symptom, not evidence for
wider division gates. Once an intermediate lineage is absent, a late
re-parenting bridge must span the accumulated displacement. Widening geometry,
adding a division head, or tuning division constants would act after the
fragmentation and would expose many more false forks. The held-out oracle also
invalidates the suggested generic re-parenting fix: GT-supported rewiring works,
but the current inference signal does not identify those bridges safely.

Production divisions therefore remain default OFF at
`src/celltrack_alpha/bh_graph.py:1930`; explicit opt-in remains available. The
focused regression is at `tests/test_bhg.py:709-726`, and the known-GT true
positive remains an expected failure at `tests/test_bh_division_holdout.py:34-57`.
No geometry constant or rewiring rule changed.

## Component fragmentation

### Verified

The prefix-52 component statistic reproduces the motivating observation
exactly (`fragmentation.json:67385-67575`):

- pooled prediction weak-component size:
  `1,097,882 / 46,988 = 23.365157` nodes;
- pooled GT weak-lineage size: `32,505 / 1,019 = 31.898921` nodes;
- `542/1,019 = 53.1894%` of GT weak lineages have matched nodes spread across
  more than one prediction component.

On the complete 57 (`fragmentation.json:64143-64332`):

- pooled prediction component size:
  `1,235,932 / 52,111 = 23.717296` nodes;
- pooled GT weak-lineage size: `37,486 / 1,218 = 30.776683` nodes;
- `674/1,218 = 55.3366%` of GT weak lineages are fragmented.

These are pooled weak-component means. GT weak components may contain a
division lineage, so “GT track” here means the full weak lineage rather than a
single post-division arm.

## Where breaks occur

### Time and absolute space: verified

Breaks are not confined to one scope, time window, or image region. On 57:

- source-frame-decile break rates range from `5.12%` (frames 80-89) to `7.05%`
  (frames 20-29); the full exposure-adjusted table is at
  `fragmentation.json:64440-64500`;
- x-third rates are `6.10% / 5.90% / 5.90%`;
- y-third rates are `5.88% / 5.85% / 6.19%`;
- z-third rates are `6.00% / 5.52% / 6.56%`.

Thus there is a mild early/mid-sequence and high-z excess, but no spatial or
temporal slice explains the general fragmentation.

For the 1,853 complete-57 breaks with both GT endpoints matched, the final
matched-pair displacement is median `7.1065` um, mean `7.5710` um, p90
`12.0787` um, and max `22.7971` um. These are postprocessed output coordinates,
not raw candidate-gate coordinates.

### Crowding: verified direction, nonlinear full-57 effect

The original prefix-52 claim reproduces exactly:

- Pearson(edge Jaccard, peak component matrix dimension) =
  `-0.7017698817`, p=`6.79e-9`;
- Pearson(edge Jaccard, total node ratio) = `-0.0833377951`, p=`0.557`.

On all 57, three newly added 6bba videos have peak dimensions 464-563 without
proportionally low edge scores. Consequently the raw linear Pearson weakens to
`-0.4640425024` (p=`2.77e-4`), so the exact `-0.702` coefficient does **not**
generalize unchanged. The crowding relationship remains strong and nonlinear:

- Pearson against `log1p(peak)` = `-0.6898251718`, p=`2.94e-9`;
- Spearman rho against raw peak = `-0.7800799210`, p=`8.56e-13`;
- total-node-ratio Pearson remains negligible at `-0.0880786923`, p=`0.515`.

At event level, final-node next-frame neighborhoods above the cap of 8 have an
edge-break rate of `18/64 = 28.125%`; neighborhoods at or below 8 have
`1,835/35,896 = 5.112%`. That is a `5.50x` risk ratio. Recoverable FNs have
mean final-node radius-neighbor count `3.170`, versus `2.038` for TPs. These
figures confirm that crowding is the dominant observable correlate even though
the raw full-57 Pearson is outlier-sensitive. Complete tables are in
`fragmentation.json:64341-64596`.

## Break classification and edge error attribution

The classifier and its precedence are implemented at
`scripts/analyze_holdout_fragmentation.py:305-366`. Each FN record, including
matched node IDs, frame, final displacement, final-node neighbor count/rank,
occupied endpoints, and class, is retained in
`fragmentation.json:2-64127`. Exact accounting is asserted against the frozen
per-video scorer rows at `scripts/analyze_holdout_fragmentation.py:545-558`.

### Prefix-52 FN budget: verified observable classes

All `1,810` FNs are assigned exactly:

| Observable class | FNs | Share |
|---|---:|---:|
| source or target GT endpoint unmatched | 271 | 14.97% |
| matched endpoint is a synthesized gap node | 86 | 4.75% |
| final matched-pair geometry exceeds 10 um | 336 | 18.56% |
| final matched target ranks beyond cap 8 | 0 | 0.00% |
| another true daughter already consumes the source degree | 15 | 0.83% |
| correct pair absent and source or target is occupied elsewhere | 1,094 | 60.44% |
| both matched endpoints free; no output conflict observed | 8 | 0.44% |

The corresponding complete-57 counts are `308 / 99 / 407 / 0 / 19 / 1,319 /
9`, summing to all `2,161` FNs.

The most important number is `1,094/1,810`: the majority of breaks are not
unmatched detections or isolated long steps. They are explicit endpoint
competitions that select another association and split the GT lineage.

### What is verified versus inferred about the requested four causes

The final fragments persist nodes and selected edges only. Candidate scores are
constructed in memory at `scripts/v45feval.py:314-354`, and the writer at
`scripts/v45feval.py:562` stores no confidence, embedding, pre-prune detection,
candidate row, logit, motion cost, or rejection reason. Moreover, the KD-tree
gate runs on raw detector positions (`src/celltrack_alpha/bh_edge.py:328-397`),
while the CSV contains positions after line-fit smoothing
(`src/celltrack_alpha/bh_graph.py:1808-1888`). Therefore:

- **Candidate absent from KD-tree gate:** not exactly recoverable from the
  persisted artifact. No break ranks beyond 8 among surviving final nodes, but
  this is not proof that all correct pairs were in the original cap-8 tensor.
- **Candidate present but out-ranked at a conflict:** `1,094` prefix / `1,319`
  complete-57 occupied-endpoint cases are verified conflicts. Calling them
  “out-ranked” is a strong inference, not a logged score comparison.
- **Gate radius too tight:** `336` prefix / `407` complete-57 final pairs exceed
  10 um. The geometry is verified, but attributing it to the raw 10-um gate is
  inferred because smoothing occurs after association. Consistent with F4,
  many long final bridges can be consequences of prior fragmentation.
- **Solver degree/one-parent constraint:** `15` prefix / `19` complete-57
  division edges are verified cases where a correctly connected sister already
  consumes the base one-output degree. These are downstream of the same
  association problem, not support for a wider division heuristic.

This limitation is material. An exact split between gate absence and ranking
would require regenerating and persisting raw candidate tensors/logits; the
read-only artifacts do not contain them. The report does not invent that split.

### Prefix-52 FP budget

Direct scorer-valid FP endpoint classes are exact and sum to all `2,660`:

- matched GT source -> unmatched prediction target: `1,364`;
- unmatched prediction source -> matched GT target: `1,281`;
- wrong transition between two matched GT nodes: `15`.

For causal comparison with the FN classes, each valid FP was linked to missing
GT edge(s) incident on its matched endpoint. This allocation is **inferred**,
not an official scorer label:

| Linked break observation | FPs | Share |
|---|---:|---:|
| occupied-endpoint conflict | 1,736 | 65.26% |
| final geometry above 10 um | 504 | 18.95% |
| synthesized-gap endpoint | 165 | 6.20% |
| unmatched GT endpoint | 113 | 4.25% |
| solver division degree | 8 | 0.30% |
| mixed linked causes | 7 | 0.26% |
| no incident missing GT edge | 127 | 4.77% |

On 57 the corresponding inferred allocation is `2,112 / 630 / 189 / 135 /
10 / 11 / 143`, summing to all `3,230` FPs. The direct and linked tables are
recorded at `fragmentation.json:64143-64332,67385-67575`.

## Decision

Association fragmentation is the single lever large enough to address both the
edge deficit and the zero division term. The evidence supports work on
candidate/assignment continuity in crowded components. It does not support
division-constant tuning, geometry widening, generic re-parenting, a new
division head, or node-threshold tuning as the primary next step.

Divisions remain default OFF. No new model training or submission was started.
