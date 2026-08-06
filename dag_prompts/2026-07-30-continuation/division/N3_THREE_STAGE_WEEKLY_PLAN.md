# TAP-SYM-DIV — adversarially honest three-stage weekly plan

**Disposition: NOT ELIGIBLE / HUMAN-DECISION.** This is a conditional amendment
design, not authorization to acquire an asset, train, infer, fit, score, or launch a
Kaggle kernel. N2 remains `ESCALATE`; it does not become a GO by distributing work
over stages.

## Bottom line

There is **no executable three-stage plan in the present one-week Kaggle-T4 envelope**
that both preserves strict five-fold isolation and supplies a legitimate
objective-level division efficacy result. The published TAP recipe is planned at about
8 GPU-h per model in the N2 evidence contract. Five independent fit-only encoders have
a training-only lower bound of `5 × 8 = 40 T4-h`, before held-side embedding,
fit-side-only selector fitting, or frozen-scorer evaluation. The requested weekly
envelope is `13 ± 3 = 10–16 T4-h`; 40 h is 2.5 times its maximum. Current
owner-authenticated launch capacity is also only 25.47 GPU-h.

More importantly, a shared self-supervised encoder trained on all raw competition
videos is **not** an economy: for fold *f*, it lets the representation learn from
fold *f*'s held raw frames. That is transductive leakage under the existing N2 contract.
Training a single encoder and calling the five later reads “CV” is therefore invalid.
The TAP source's default configuration also assumes an NVIDIA GPU with at least 24 GB;
T4 feasibility after a batch-size change has not been authenticated. Neither lower
batch size nor a different backbone/checkpoint is an execution repair; either needs a
named, pre-outcome amendment.

The campaign has a second independent block. Division fitting is locked and the
division lane is `PARK_DEPENDENCY`: there are no association arms satisfying the
both-family prerequisite. A raw-pixel representation record can establish a new
observation channel, but cannot honestly be presented as a graph-score gain until a
human amendment explicitly permits a fit-side-only selector and the association
dependency is resolved. The 38 repair bundles are oracle diagnostics, never a training,
threshold-setting, or selection population.

Consequently, the stages below are a **falsification and authorization sequence**. The
only current safe outcome is to stop before Stage 2. “Efficacy” in Stages 1–2 means
mechanism efficacy, not campaign efficacy; the sole objective efficacy metric is in
Stage 3.

## Non-negotiable isolation contract

Freeze these rules before any future source checkout, data mount, or GPU work:

1. Freeze the existing five outer group folds and their `fit_f` / `held_f` manifests
   byte-exactly. For each fold `f ∈ {0,…,4}`, TAP weights `W_f` may read only raw
   intensity frames in `fit_f`; no shared checkpoint, self-supervised statistic,
   augmentation cache, normalizer, early-stop signal, or hyperparameter choice may read
   `held_f`.
2. Freeze `W_f` before a one-way held-side representation pass. `held_f` can then be
   encoded once, append-only, to the evidence schema `A, S, U, Q=A×S×U` and order
   checksum. The record must contain no action ID, current-pool rank/score,
   continuity result, oracle-bundle ID, GT field, threshold, or graph edit.
3. If and only if a later amendment opens a selector, all selector/threshold parameters
   must be learned within nested groups of `fit_f` only. The held representation may be
   scored once by this frozen selector. No fold result may change an encoder, selector,
   crop, threshold, epoch, or seed for any other fold.
4. The only final graph read is the frozen official scorer applied once to all five
   sealed held folds. Do not choose a threshold, a subset of folds, or a division rule
   after inspecting that read. No “time-only” shared pretraining, held-frame
   normalization, oracle transduction, or threshold transfer is allowed.

## Pre-registered metrics and verdict vocabulary

`Q` remains evidence, not a production score. On the two fixed analytic fixtures from
N2, define the mechanical contract

`E_syn = [Q(split)>Q(non-split)] ∧ [A(forward)>A(reverse)] ∧ [|Q-Q_swap-reflect|≤1e-6] ∧ [forbidden-field schema rejection]`.

For a future external, balanced time-order test set of `n=10,000` fixed raw patch pairs,
define TAP mechanism efficacy `E_TAP` as the exact two-sided Clopper–Pearson 95% lower
bound for time-order accuracy being strictly greater than 0.50, evaluated separately
for each of two predeclared seeds. This shows a learned time-arrow signal; it says
nothing about division-selection value.

If the dependency and fitting lock are both explicitly amended, the *only* objective
efficacy statistic is, for each family `g`, the frozen-scorer adjusted-edge Jaccard
delta over its five-fold held predictions:

`Δ_g = J_adjusted(TAP-SYM-DIV, g) − J_adjusted(division-off baseline, g)`.

Raw score and node recall deltas are recorded as non-regression guards. Proposal AUC,
time-arrow accuracy, `Q`, number of apparent divisions, and oracle-bundle recovery are
not substitutes for `Δ_g`.

## Conditional stages

### Stage 1 — source, data-rights, and synthetic contract

| Item | Preregistered requirement |
|---|---|
| Inputs | Pinned `tarrow` source commit, BSD-3-Clause license receipt, exact offline dependency hashes, the two analytic split/non-split fixtures, and the five fold manifests. **No competition raw frame, label, scorer, action table, or oracle bundle is read.** |
| Scope | Verify source and manifest provenance; implement only the append-only evidence schema; exercise it on the analytic fixtures. Do not make a crop from campaign imagery, train a model, or materialize a candidate action. |
| Kaggle T4 estimate | **0 h.** Spending GPU merely to meet a weekly target would not be evidence. |
| Efficacy checkpoint | Mechanical `E_syn`, exactly the four conjuncts above. It is explicitly not a graph checkpoint. |
| GO | Valid source/license/hash receipts, pairwise-disjoint fold manifests, and `E_syn = true` on both fixed fixtures in deterministic repeat. |
| PARK | The contract is conceptually intact but a permitted source revision, weights, external-data rights, or T4-compatible configuration is absent. Do not improvise a replacement. |
| DEAD | With valid pinned inputs, any `E_syn` clause fails, an encoder/hash mismatch occurs, or the schema admits a forbidden field. The tested TAP-SYM-DIV implementation is dead; a different encoder/observable would be a new named mechanism. |
| Human gate | **Required before Stage 2:** approve the source pin, any source-published lower-memory configuration, external-public corpus and retention terms, and at most 16 T4-h. A Stage-1 GO never auto-launches. |

### Stage 2 — public-data T4 feasibility and time-arrow replication

| Item | Preregistered requirement |
|---|---|
| Inputs | Only the Stage-1-pinned code/config and two named, externally public raw 2D+time corpora with documented rights; 10,000 pre-enumerated held public patch pairs; no competition frame or campaign-derived crop. The TAP README names raw 2D+time input and gives HELA/MDCK as example public data, but exact corpus versions and hashes must be frozen before approval. |
| Scope | Run two independent, predeclared seeds of the same source-published low-memory configuration. No architecture, crop, delta, batch size, epoch count, seed, or early-stop rule may change between seeds or after results. This tests portability and the time-arrow mechanism only—not cell-tracking performance. |
| Kaggle T4 estimate | **16 h maximum** (`2 × 8 h` planning estimate), which is inside the 10–16 h weekly envelope. If either run projects above 8 h or breaches 16 GiB, abort that run before an outcome; do not borrow time from Stage 3. The estimate is not yet a measured T4 projection. |
| Efficacy checkpoint | Both seeds pass `E_syn`; both have exact Clopper–Pearson 95% lower bound `> 0.50` on their fixed 10,000-pair public held set; neither has an OOM, held-data read, or source/config hash drift. This is **mechanism efficacy only**. |
| GO | Every Stage-2 condition passes and actual cumulative debit is `≤16 T4-h`. This GO authorizes only a human review of Stage 3; it does not authorize competition-data processing. |
| PARK | Both seeds satisfy `E_syn` and `E_TAP`, but T4 memory/runtime or available capacity prevents a five-fold campaign execution. Park rather than downsize folds, pool raw frames, or quietly use one model across folds. |
| DEAD | Either valid seed fails `E_syn` or `E_TAP`, source/config integrity fails, or any campaign/held data is mounted. Treat a memory-only failure as PARK, not biological failure. |
| Human gate | **Required before Stage 3:** a written plan amendment must separately open division fitting, name the fit-only selector and its fixed decision rule, show a passing association-continuity prerequisite, allocate the full compute envelope, and approve a single Kaggle execution protocol. |

### Stage 3 — five independent encoders and the only graph-efficacy test

| Item | Preregistered requirement |
|---|---|
| Inputs | For each outer fold `f`, `fit_f` raw frames only to train `W_f`; frozen `W_f` then encodes `held_f` once. A human-approved selector may use only nested `fit_f` groups and representation fields `A,S,U,Q`; it cannot use current-pool scores/ranks, continuity outputs, oracle repairs, or held labels. The frozen division-off baseline, official scorer, and five fold manifests remain unchanged. |
| Scope | Produce five independent weight/input manifests and sealed held evidence manifests. After every `W_f` and selector is frozen, run exactly one held-side graph evaluation; aggregate only after all five results exist. There is no per-fold retuning, threshold rescue, or retry. |
| Kaggle T4 estimate | **At least 40 h for encoder training alone** (`5 × 8 h`), plus currently unmeasured held inference, selector, and frozen-scorer time. It cannot run in one 10–16 h week and exceeds the current 25.47 h authorization capacity before inference. Therefore Stage 3 is **not eligible now**. |
| Efficacy checkpoint | The sole campaign metric is `Δ_44b6` and `Δ_6bba` above, after all five sealed folds. Report raw-score and node-recall deltas, division TP/FP/FN, and exact manifests, but do not gate on them. |
| GO | `Δ_44b6 ≥ +0.005` **and** `Δ_6bba ≥ +0.005`, with neither raw-score nor node-recall regression, all five manifests disjoint, and no forbidden-input/threshold event. This still proceeds only to the existing human-controlled CV57/public-probe policy. |
| PARK | `Δ_44b6 > 0` but either family is `< +0.005`, or any non-regression guard fails without an isolation breach. Keep division output off; do not inspect oracle bundles for a repair. |
| DEAD | `Δ_44b6 ≤ 0` under the valid preregistered protocol, or any isolation/forbidden-input breach occurs. The latter invalidates the run and is not efficacy evidence; no retry with changed thresholds, transduction, or pooled training is allowed. |
| Human gate | **Required before any kernel:** confirm the Stage-2 receipts, a named association arm that passed both-family prerequisite, approved selector/fitting amendment, source/weight/rights pins, a measured ≤16 GiB configuration, and sufficient capacity for `40 h + measured inference/score margin`. A grant for one week is not a grant for Stage 3. |

## What a scientifically valid alternative looks like

1. **Keep TAP-SYM-DIV parked.** This is the only present-state option. Complete an
   association-continuity mechanism through its existing gates first; it may make the
   division objective legal to test, but does not by itself make TAP affordable.
2. **Fund the complete isolated study, not a disguised shortcut.** After the two
   amendments above, reserve at least 40 T4-h plus a measured inference/score margin.
   It would need several 10–16 h weekly windows (for example, two encoders in each of
   two weeks and one encoder plus measured inference in a third), with each fold's
   code fixed before its fit-side training. The present capacity cannot fund even the
   training lower bound. A multi-week schedule is a budget change, not a weekly plan.
3. **Use an authenticated frozen *external* TAP checkpoint only if one is actually
   available.** Its training provenance must exclude every competition movie; it must
   be source/weight/license pinned and pass Stage 1 plus a T4 inference preflight. This
   can remove self-supervised competition-frame leakage and training cost, but is a
   separate named frozen-external-representation mechanism. It still cannot open the
   locked division selector or claim graph efficacy without the association and fitting
   amendments. No checkpoint is currently authenticated, so this is not a launchable
   substitute.

Invalid alternatives are explicitly excluded: a single transductive encoder over all
five folds; any held-frame normalization or early stopping; selecting `Q`/a threshold
using oracle bundles or held outcomes; declaring a public time-arrow result to be a
division-score result; replacing the source's memory requirement with an unpinned
backbone; or treating a resource failure as permission to run a fewer-fold study.

## Evidence consulted

- Current campaign state and division lock: `reports/g02/screening/LIVE_STATE.md`.
- Locked governance, 44b6-first graph floor, and division dependency: 
  `reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md` and
  `reports/g02/screening/B1_division_tp.md`.
- Existing TAP N0–N2 findings and original 40-h feasibility calculation:
  `dag_prompts/2026-07-30-continuation/division/runs/N0_SCOPE.md`,
  `runs/N1_LITERATURE.md`, and `runs/N2_FALSIFIER.md`.
- TAP supports raw 2D+time self-supervised time-arrow representations but documents a
  default NVIDIA GPU requirement of at least 24 GB; it does not demonstrate this
  campaign's T4 configuration: [TAP source README](https://github.com/weigertlab/tarrow).
  The biological premise, not a graph-gain claim, is supported by
  [Gallusser, Stieber & Weigert (2023)](https://arxiv.org/abs/2305.05511) and
  [Gilad et al. (2019)](https://doi.org/10.1093/bioinformatics/bty1034).
