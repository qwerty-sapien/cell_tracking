# CADINO / Trackastra successor hypothesis audit

**Date:** 2026-08-17  
**Disposition:** `BANK_EXPLORATORY` for the method and audit; `PARK_DEPENDENCY` for every model-executing hypothesis  
**Validated audit plan:** `dag_prompts/2026-08-16-five-lane-alpha-swarm-v1/cadino_triplet/runs/auto_experiment_audit/CADINO_experiment_plan.json`  
**Plan SHA-256:** `5f6492ca836c1fcd4195273b7ef4b428ae0dc095982c4e33e52200b882837f00`

## Adversarial verdict

The earlier claim was too optimistic. Reaching `0.92` from the frozen local CV57 diagnostic score `0.9143584730207287` requires `+0.0056415269792713`. That is a plausible research target, but there is no evidence for calling it high-likelihood. An expected ceiling materially above `0.95` is not defensible. The `0.95` target requires `+0.0356415269792713`; residual-oracle arithmetic shows theoretical room in association and division errors, but an oracle is not a learnable mechanism or a forecast.

The closest local evidence is cautionary:

- Trackastra proposal ranking looked strong on a proxy, but the zero-wrfeat graph realization lost `-0.0111480766` adjusted score despite proposal precision `0.9565217391`.
- The earlier Trackastra PEFT realization gained only `+0.000133826958`, about `37.36x` below the binding `+0.005` gate.
- Historical proposal-to-graph transfer is weak: Spearman `0.211` and direction agreement `25%` in the cited local audit.
- Channel-Adaptive DINO and ExPLoRA validate adjacent mechanism classes, not this zebrafish lineage task, this candidate distribution, or this competition scorer.

The literature therefore supports testing these mechanisms, not assigning a success probability. Trackastra supports learned temporal association and division; Channel-Adaptive DINO supports microscopy/channel-aware representations; ExPLoRA supports parameter-efficient continued pretraining under domain shift; Ultrack supports competing segmentation hypotheses and global optimization, including embryo applications. None validates the complete proposed pipeline on this target.

## Why the two requested options did not launch

Both directions were advanced in parallel as hypotheses, plan design, literature review, and static/model-free falsification. They were not trained. The local Meta checkpoint is present at `1,403,278,520` bytes with SHA-256 `17293cad76b8e592db743f7d375ba8d4104261f3f465ff64e667da406855c85b`, but weights alone do not satisfy the launch contract:

- the bound README says Extended-CHAMMI while the model entrypoint docstring says HPA;
- the code-license file says CC-BY-4.0 while the bound README says CC-BY-NC, and model weights are separately FAIR Noncommercial;
- the direct upstream loader uses `torch.load` without `weights_only`;
- the required trusted runner, fixture suite, central registration, exact offline carrier, train-only manifest, biological group provenance, objective, optimization budget, output/retention rules, and stop criteria are not bound;
- the current score lock forbids checkpoint updates, warm starts, adapters, PEFT, and HPO below `0.931`; only an explicit owner amendment can change that rule for a named successor.

The detector-conditioned positive-unlabeled Trackastra option is H2. The continued-SSL/PEFT CADINO option is H4. H1 and H3 are deliberately frozen predecessors: jumping directly to H2/H4 would confound representation viability with training and selection.

## Frozen hypothesis family

No hypothesis below has a campaign p-value. `p = N/A` is deliberate: none has been executed against an authorized independent biological sample, and the deterministic graph improvement thresholds are the primary decisions.

| ID | H0 | Ha | Primary minimum effect | Current p-value | Status |
|---|---|---|---:|---|---|
| H1 | Frozen Trackastra with authentic detector/mask features fails to improve the complete 44b6 graph by `+0.005`, or regresses raw score/node recall. | It clears `+0.005` with nonnegative raw/node effects on 44b6 and repeats the gate on 6bba. | `+0.005` vs V32 | `N/A — unexecuted` | `PARK_DEPENDENCY` |
| H2 | Detector-conditioned positive-unlabeled Trackastra PEFT adds `<+0.002` over H1 or misses the absolute H1 gate. | A finite inner-selected PEFT procedure adds `>=+0.002` over H1 and clears `+0.005` vs V32 without raw/node regression. | `+0.002` incremental and `+0.005` absolute | `N/A — unexecuted` | `PARK_DEPENDENCY` |
| H3 | Frozen, identity-resolved CADINO lacks repeatable temporal signal or misses `+0.005` on the complete graph. | It passes temporal/block-null controls and the ordered `+0.005` graph gates without raw/node regression. | `+0.005` vs V32 | `N/A — unexecuted` | `PARK_DEPENDENCY` |
| H4 | CADINO continued SSL plus PEFT adds `<+0.002` over H3 or violates controls/firewall. | A finite ExPLoRA-style fit-only procedure adds `>=+0.002` over H3 and clears `+0.005` vs V32. | `+0.002` incremental and `+0.005` absolute | `N/A — unexecuted` | `PARK_DEPENDENCY` |
| H5 | Learned association/division evidence plus a structured multi-hypothesis global decoder adds `<+0.005` over its parent or `<+0.010` vs V32. | The fixed decoder clears both gates on ordered held-fit families with nonnegative raw/node effects. | `+0.005` incremental and `+0.010` absolute | `N/A — unexecuted` | `PARK_DEPENDENCY` |

The family is tested in fixed sequence only after authorization. A failure terminates the exact realization. A successor inspired by failure gets a new identity and cannot reuse the same confirmation outcome.

## Eight completed mini-experiments

These experiments were deliberately cross-cutting. They exercised statistics, selection, ceiling claims, local priors, loader safety, identity/license consistency, namespace leakage, and cached-compute scope without opening a checkpoint, pixels, labels, serialized features, or running the scorer. E3/E4 did read already-frozen historical CV57 score/residual summaries and terminal receipts solely for diagnostic arithmetic and heterogeneous evidence context; those summaries were not used to choose a candidate or parameter.

| Experiment | Result | Consequence |
|---|---|---|
| E1 pseudoreplication stress | With 12 embryos and 32 nested rows, naive row-level Type-I error was `5.21%`, `51.54%`, and `70.81%` at ICC `0`, `.25`, and `.75`; embryo-cluster analysis stayed `4.58–4.95%`. | Frames/cells/edges cannot be counted as independent n. |
| E2 null HPO winner stress | Selecting the best of 100 null trials produced an apparent development discovery `92.135%` of the time and mean selected z `2.509`; untouched confirmation stayed near the null (`2.59%` positive-CI rate). | Open-ended iteration is an optimism machine; freeze candidates and keep confirmation untouched. |
| E3 score/headroom arithmetic | `0.92` needs `+0.005642`; `0.95` needs `+0.035642`. A perfect detection-only or fragmentation-only fix is arithmetically insufficient for `0.95`; association/division oracle room exists. | The desired score is a target, not a forecast. |
| E4 local-prior stress | No receipt directly validates H1–H5. The strongest nearby graph and PEFT evidence is negative or far below gate. | Generic literature cannot override target-local evidence. |
| E5 loader safety lint | The upstream path constructs the model then calls `torch.load(..., map_location='cpu')` without `weights_only`. | No direct deserialization; a reviewed weights-only loader remains mandatory. |
| E6 identity/license matrix | Extended-CHAMMI conflicts with HPA; CC-BY-4.0 conflicts with README CC-BY-NC. | Identity and legal-use handling stay unresolved; no single clean assertion is allowed. |
| E7 namespace firewall | ZIP has `24,886` members: `24,477` train and `408` test; `test/44b6_...` proves family prefix is not a train-only firewall. | Future data access needs a hash-bound path allowlist, not family globbing. |
| E8 Trackastra cache audit | JSON reports cover 105 videos, 280,095 rows, 195,354 proposals, 9,363 calls, 83,615,281 bytes, and 609.55 s; feature payloads stayed sealed. | Cached scope/cost is known; efficacy is not. |

E1 and E2 are synthetic demonstrations, not biological inference. Their numerical Monte Carlo errors do not become p-values for CADINO.

## Future data and validation contract

The independent unit is the highest verified biological block—ideally clutch/acquisition day, otherwise embryo—not frame, slice, cell, track, edge, triplet, proposal, or overlapping window. Until provenance proves that unit, analysis is a deterministic case study and cannot make a population claim.

| Method | Permissible role | Restriction |
|---|---|---|
| Fixed grouped holdout | Final confirmation of one immutable pipeline | Open once; no debugging, thresholding, calibration, or selection after opening. |
| Nested stratified GroupKFold | Fit-only architecture/hyperparameter selection | All transforms, mining, early stopping, thresholds, and checkpoints stay inside the inner loop; outer folds evaluate the full procedure. |
| Leave-p-groups-out | Sensitivity to influential embryos/clutches | Preregister p and combinations; overlapping fits are not independent replications. |
| Grouped ShuffleSplit | Exploratory stability diagnostic | Never confirmatory and never counted as extra n. |
| Ordinary K-fold | Only if it is group-aware | Row/frame/cell K-fold is invalid. |
| Blocked time-series split | Within-group forward-extrapolation diagnostic | It does not replace left-out-embryo generalization. |
| Ordered campaign holdout | Exact 44b6 then 6bba then CV57 cascade | A miss stops the cascade; CV57 cannot select or repair a model. |

The future train manifest must enumerate allowed `train/` paths, verified biological group IDs, labels permitted for each stage, and forbidden test/holdout paths by digest. The archive audit proves that a `44b6*` glob is unsafe.

## Proposed finite parameter inventories

These are design proposals, not authorization and not yet hash-bound inventories. Each must be reduced to explicit candidate rows with code/environment/data/split digests and exact trial counts before execution.

### H1 — frozen authentic-feature Trackastra

Use the existing immutable Trackastra checkpoint. Replace zero-filled wrfeat with authentic detector/mask-derived geometry, intensity/morphology, and temporal features through a tested fixed adapter. No weight update, checkpoint choice, or threshold search. First require synthetic known-link/division recovery, determinism, temporal-shift negative control, action-supply accounting, and one complete 44b6 graph.

### H2 — detector-conditioned positive-unlabeled Trackastra PEFT

Train on the same candidate process used at inference. Known true links—including both verified daughters of a division—are positives. A candidate may be negative only when frozen event/horizon semantics prove it non-true; ambiguous candidates and all other unmatched detections remain unlabeled and must not silently become background negatives. Mere competition for the same parent is insufficient because a true parent may have two valid children.

A possible maximum-eight recipe inventory is the Cartesian product of:

- update scope: association head only; association head plus last attention block;
- learning rate: `1e-4`; `3e-4`;
- admissible hard-negative cap per positive: `1`; `3`.

Keep weight decay `0.01`, one frozen seed for inner selection, a fixed owner-authorized step/epoch budget, and one refit of the chosen procedure. Do not add retries after seeing results. This parameterization is motivated by controlling adaptation capacity; it is not literature-validated on this target.

### H3 — frozen CADINO triplet representation

Resolve checkpoint identity/license and run only through the trusted weights-only loader and fixture-bound wrapper. Hold weights fixed. Test temporal coherence against within-block temporal shifts and cross-group identity breaks; then expose a fixed representation-to-action interface to a complete graph scorer. This separates representation signal from adaptation.

### H4 — continued SSL plus CADINO PEFT

Only after an explicit owner amendment authorizes H4 and H3 clears its preregistered temporal/block-null controls plus the exact `+0.005` complete-graph gate without raw/node regression, a possible maximum-eight ExPLoRA-style inventory is:

- LoRA rank: `4`; `8`;
- unfrozen terminal blocks: `1`; `2`;
- paired learning rates `(LoRA, block)`: `(1e-4, 1e-5)`; `(3e-4, 3e-5)`.

Hold `alpha = 2 * rank`, dropout `0.05`, weight decay `0.05`, and warmup fraction `0.05` fixed. The owner amendment must specify the exact self-supervised objective, train-only channel/augmentation policy, optimization-step cap, checkpoint rule, resource estimate, and retention/deletion rules; execution must satisfy the then-current binding `LMT.md` guard, which remains the sole resource authority. Selection is entirely nested by biological group. ExPLoRA is rationale from another domain, not efficacy evidence here.

### H5 — learned evidence plus structured multi-hypothesis decoder

This is the ceiling-oriented alternative to pairwise-only ranking. Freeze a finite candidate generator that represents competing node, link, gap-closing, and division/fork hypotheses. Feed learned H1/H3 evidence into a fixed global optimization objective with conservation and temporal constraints. Test solver determinism, same-count placebo actions, action-supply ceilings, division TP/FP/FN, ordinary-association and fragmentation residuals, then the complete graph. Ultrack makes the mechanism defensible; it does not establish competition gain.

## P-values and anti-p-hacking safeguards

The primary campaign decisions are exact preregistered score deltas and safety gates, not `p < 0.05`. If enough independent embryos/blocks exist, uncertainty may use a cluster bootstrap that resamples whole blocks and recomputes the full nonlinear metric. Mechanism-null tests may use paired block-level sign/randomization or blocked temporal permutation only when exchangeability is justified.

For a future confirmatory family:

- freeze the five-hypothesis family, alpha `0.05`, order, one look, and stop at first failure;
- use fixed-sequence testing for the ordered cascade; use Holm only for a genuinely unordered small family;
- never treat folds, repeated ShuffleSplits, frames, cells, or edges as replicate n;
- bind every architecture, parameter, seed, checkpoint rule, and maximum trial count in a hashed inventory;
- preserve failed runs and forbid outcome-adaptive additions;
- keep protected confirmation inaccessible to hypothesis generation, debugging, and manual visual selection;
- report effect sizes and block-level sensitivity even if a p-value is available.

## Auto-experiment skill decision

The tangent is meaningful only after narrowing the idea. `$auto-experiment` is implemented as a fail-closed experiment compiler and adversarial auditor, not an autonomous system that repeatedly trains until something looks significant. It has three explicit modes—`exploratory`, `confirmatory`, and `audit_only`—and rejects pseudoreplication, ungrouped or repeated confirmatory splits, selection without nested grouped evaluation, incomplete finite candidate accounting, multiplicity gaps, holdout reuse, and unresolved authority/runtime/data identities.

The skill was immediately deployed here in `audit_only` mode. Its validator accepted the five-hypothesis/eight-experiment plan, and 17 focused regression tests passed. This deployment correctly refused model execution. That refusal is a successful scientific outcome: it prevented available weights from being mistaken for an authorized, identified, safely loadable, leakage-controlled experiment.

## Promotion sequence

1. Obtain a named owner successor amendment that defines the mechanism, data/labels, objective, exact finite optimization budget, validation/firewall, output retention, and stop rules, and explicitly reconciles the current score lock.
2. Resolve checkpoint identity and code/model license conflicts; bind a trusted weights-only loader, fixtures, independent runner review, central registration, exact offline carrier, and LMT guard.
3. Produce a hash-bound train-only biological-group manifest and prove denylisted test/holdout access fails.
4. Run H1 and H3 frozen branches first, in parallel only if outputs are disjoint and the serial/resource rules allow it.
5. Unlock H2 only through an explicit owner amendment and only after H1 clears its preregistered `+0.005` complete-graph gate with nonnegative raw/node effects; unlock H4 only through an explicit owner amendment and only after H3 clears that same frozen-representation gate plus its temporal/block-null controls. An H1/H3 held-family outcome used as an unlock cannot then select, tune, or confirm H2/H4. A future plan must either use outcome-free/fit-only prerequisites plus a fresh untouched confirmation allocation, or freeze the full fixed-sequence candidate inventory and tests before the first family score opens. Do not use training to rescue a failed frozen representation against the same outcome.
6. A future owner-authorized H5 plan must either pre-name its upstream parent before confirmation or select the parent entirely inside nested fit groups and evaluate it on a new untouched outer/confirmation population. It must also freeze a quantitative action-supply gate before outcomes; neither “best parent” nor “sufficient supply” may be chosen from 44b6/CV57 and then evaluated on that same outcome. The current audit-only H5 wording is not launch authorization.

Until those gates are met, local training remains parked.

## Primary literature anchors

- [Trackastra: Transformer-based cell tracking for live-cell microscopy](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf)
- [Channel-Adaptive DINO](https://openreview.net/pdf?id=pT8sgtRVAf)
- [ExPLoRA](https://proceedings.mlr.press/v267/khanna25a.html)
- [Ultrack](https://www.nature.com/articles/s41592-025-02778-0)
- [Carley, Validating Computational Models](http://reports-archive.adm.cs.cmu.edu/anon/isr2017/CMU-ISR-17-105.pdf)
- [Cawley and Talbot, selection bias](https://jmlr.org/papers/v11/cawley10a.html)
- [Varma and Simon, nested cross-validation](https://doi.org/10.1186/1471-2105-7-91)
