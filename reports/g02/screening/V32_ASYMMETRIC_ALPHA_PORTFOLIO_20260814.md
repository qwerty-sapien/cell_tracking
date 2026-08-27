# V32 asymmetric alpha portfolio

Date: 2026-08-14  
Reference: immutable public TemporalUNet V32; public score `0.911`; frozen
CV57 score `0.914358473020729`.

## Decision after adversarial review

There is **no scientific GO**. Fixed-graph V32 replay and scalar/graph HPO are
poor uses of quota. The only defensible lead is a revised held-fit-only
preregistration for a structured V32 hypothesis lattice; it is `PARK-REVISE`,
not authorized compute.

V32 is not proven saturated. Its exact CV57 taxonomy contains `538`
detection-attributed edge FNs caused by `422` unmatched GT nodes, `1,053`
fragmentation edge FNs, `1,931` ordinary-association errors (`1,911` FP plus
`20` FN), and `108` division errors (`69` FP plus `39` FN over `42` GT
divisions). The exact nonlinear Shapley headroom is respectively `+0.01037`,
`+0.02838`, `+0.04959`, and `+0.09730`. These are counterfactual ceilings, not
jointly realizable gains. They permit a 5--10 point scenario; they do not make
one locally plausible.

## Priority 1: `V32-HYPOTHESIS-LATTICE-v1` — `PARK-REVISE`

**Mechanism.** Retain both frozen V32 seeds' and existing eight-view detector
evidence before blending and peak thresholding. Generate structured, mutually
exclusive node hypotheses with exact deduplication, split, and exclusion rules,
then solve selection and tracking jointly. Distinctness comes from preserving
seed/view disagreement as alternate nodes and actions; another transform sweep
or threshold grid is not a new mechanism. V32 weights remain immutable.

Ultrack supports the general idea of structured segmentation hypotheses and
joint segmentation/tracking optimization
([Ultrack, Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0)).
It does not validate arbitrary point peaks, so the point-hypothesis transfer
must be specified and falsified locally.

**Current dependency.** No authenticated heatmap artifact exists. The current
V32 replay cache deliberately omits dense logits and per-view tensors. CV57
identities may remain diagnostic visual stress cases, but cannot select
thresholds, exclusions, or resource caps. Those rules must be frozen on
held-fit data before any CV57 outcome is opened.

**Required held-fit-only falsifier before implementation/training.** First
preregister the structured generator and an outcome-free heatmap exporter.
Then compile one exact feasible combined graph per family with the frozen
scorer. Independent-action sums and proxy recovery counts are report-only.

GO requires all of:

- node inflation at most `10%` and candidate-edge inflation at most `20%`;
- maximum conflict-component size at most `2x` exact V32 and solver wall time
  at most `1.5x` the same-identity V32 control;
- current LMT RSS/growth/workspace limits pass without relaxation;
- at least `25%` of unique unmatched-node scorer-equivalence events have a
  valid alternate hypothesis;
- at least `25%` of fragmentation equivalence events have a one-to-one
  realizable alternate node/edge path;
- one exact feasible combined graph reaches adjusted delta at least `+0.010`
  in each held-fit family, with raw-edge and node-recall deltas nonnegative.

The first outcome-free realization still needs adjusted delta `>=+0.005` on
held-fit 44b6 with nonnegative raw/node deltas before 6bba opens. CV57 remains a
separate gate. A Kaggle kernel would require strict exact local CV57
improvement, no family regression, and a PASS exact-candidate local visual
receipt.

**Assessment.** Highest-priority design dependency. Gain is unpredicted until
the feasible combined-graph oracle passes.

## Priority 2: `V32-POSTERIOR-SECOND-OPINION-v1` — `PARK-DEPENDENCY`

**Mechanism.** Enumerate near-optimal V32 solutions and measure solution
instability. Do not call it posterior uncertainty unless held-fit calibration
proves that interpretation. Use instability only as a router into raw-volume
second-opinion actions that can add, drop, split, or relink nodes. Selecting a
different existing V32 edge is insufficient.

OrganoidTracker 2.0 motivates calibrated local-configuration marginalization
and context-aware error prediction
([Nature Methods 2025](https://www.nature.com/articles/s41592-025-02845-6)).
It does not establish that gaps between generic V32 solutions with mixed
learned/hand costs are probabilities.

**Dependency and kill gate.** Historical final graphs and MP4s omit rejected
candidates and costs; no authenticated pre-ILP data product currently exists.
After one exists, preregister a fixed region partition, one-to-one error
attribution, minimum support, shuffled/null concentration, and held-fit-only
calibration. The top `10%` instability regions must contain at least `50%` of
ordinary-association plus fragmentation errors in each held-fit family. One
exact feasible raw-context action graph must then reach adjusted min-family
delta `+0.010` with nonnegative raw/node deltas. AUC and permissive independent
edits cannot promote it. Scientific M3 remains excluded without identity and
label-provenance disjointness.

**Assessment.** Second priority, blocked on authenticated outcome-free rejected-
candidate/cost data. Gain unpredicted.

## Priority 3: `V32-BACKWARD-FIELD-TREE-v1` — `DEAD-AS-WRITTEN`

The cited center-plus-backward-vector/global-ILP mechanism is real
([Malin-Mayor et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614077/)), but
the proposed small from-scratch seven-frame crop model is not supported by that
precedent. The paper used dense 4-D U-Nets, long training, and validation-grid-
searched ILP parameters. Here it collides with the DEAD from-scratch ranker
class and the no-weight-refinement-below-`0.916` lock.

It may return only as a new named frozen/pretrained-feature mechanism after the
refinement lock opens and after a current-LMT projection. Its binding target is
the complete frozen 38-event held-fit division population across both families,
not an outcome-selected CV57 subset. Any successor must report all-38 Jaccard,
exact combined delta `>0`, nonnegative adjusted/raw/node deltas, zero duplicates,
one candidate per scorer-equivalence event, and atomic forks. The historical
`5 TP / 17 FP / 22 forks` surface is report-only.

## Locked watchlist: `V32-TEMPORAL-PROMPT-TTA-v1` — `PARK-LOCKED`

Per-video prompt adaptation remains non-actionable under the no-weight-
refinement-below-`0.916` rule. VPTTA freezes the base model but still optimizes a
per-image prompt and was demonstrated for 2-D medical segmentation, not 4-D
tracking
([VPTTA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Each_Test_Image_Deserves_A_Specific_Prompt_Continual_Test-Time_Adaptation_CVPR_2024_paper.html)).
A time-consistency proxy may screen but cannot promote without exact held-fit
graph evidence.

Generalist suppliers such as
[CellSAM](https://www.nature.com/articles/s41592-025-02879-w) and
[µSAM](https://www.nature.com/articles/s41592-024-02580-4) need a frozen 3-D
adapter, disjoint provenance, and resource projection before any Priority-2
bake-off. They are not ready-made 3-D tracking action models.

## Exact V32 visual read

The validated 57-video catalogue is at
`reports/g04/v32_exact_visual_review_20260814/index.html`. Direct inspection of
high/low-score identities across both families shows:

- crowded `train:44b6_87bba6c4` contains distributed association and
  fragmentation failures rather than one scalar-motion defect;
- `train:6bba_786893ac` misses all three annotated divisions amid dense but
  broadly coherent motion;
- `train:44b6_aaf8b0ea` has clean ordinary links but still misses its annotated
  division;
- sparse `train:6bba_76db78c1` is a useful no-change control.

Ground truth is sparse, so unannotated predictions remain unknown rather than
false positives. This diagnosis supports information/action-universe changes;
it proves none of the proposed mechanisms. The recovered V32 output is
historical, not a fresh local execution receipt.

## Execution order

1. Keep every scientific lane parked. Freeze a held-fit-only structured-
   hypothesis/exclusion/resource preregistration for Priority 1.
2. Separately design and adversarially review the missing outcome-free seed/view
   heatmap exporter. Do not weaken or relabel the authenticated sparse cache.
3. Only after that data product exists, run the Priority-1 held-fit feasible-
   graph ceiling. CV57 stress videos remain diagnostic, not configuration data.
4. Keep Priority 2 blocked until rejected-candidate/cost data and calibration
   exist. Do not run it in parallel from historical final graphs.
5. Do not run Priority 3 as written. Package or launch nothing until exact local
   execution, prediction binding, MP4 review, scientific gates, and the central
   prelaunch guard all pass.

The V32 replay cache remains reusable infrastructure, not a scientific winner
and not a reason to launch a replay kernel.
