# G-02 Specification — Trained Hybrid Tracker to 0.92

- Planning state: Active; revised screening/calibration DAG adopted 2026-07-24
- Slice: G-02
- Origin: `.cruise/goal/ORIGIN.yaml`, raw-prompt SHA-256 `39a4581e041f6d78546723fdcf99e8c964f98a96a87dc3814b53bec5da3a55ae`
- Baseline: `c3b9b703b049fc25f10e2b42bb986bb94f137a72`; provided baseline digest `6837725a09e6aff4198a80b9bfb38bb864a639842c6de38654398c6fc4f30f21`

## Outcome

Deliver a low-compute trained hybrid tracker based on the released official
`TemporalUNet3D` checkpoint. It must produce sparse center, backward-displacement,
uncertainty, division, and candidate-edge outputs, score sparse physical-coordinate
candidates with a lightweight learned edge/division model, and solve the lineage graph
globally. The one shared backbone has three lightweight specialist adapters/heads:
`sparse_clean`, `dense_crowded`, and `low_snr_depth`. An out-of-fold router selects an
adapter; confidence below `0.60` selects the universal adapter. Prioritize a materially
stronger baseline above 0.88 and then robust performance in the 0.90s while retaining the
0.92 completion target.

Completion is deliberately conjunctive: official grouped local CV >= `0.92`, untouched stratified 20-video audit >= `0.92`, a valid package/submission and projected final hidden runtime <= six hours, and an actually displayed Kaggle site score >= `0.92`. Local-only success is insufficient.

## Observed evidence

- `reports/general_model_cv_audit_fold0_auto_n61_1d0d_minx32_offsetonly_12ds_12clips.json` records a historical local 12-clip rule-based alpha audit of `0.8713516075529171`, edge Jaccard `0.9119496855345912`, node recall `0.9497274492234169`, and zero scored divisions. It is not a Kaggle score.
- `reports/general_model_scorer_parity_after_n61_1d0d_minx32_offsetonly.json` contains parity cases for node adjustment, physical matching, edges, and division connected components. Existing local parity is evidence for a starting point, not calibration against an actual site score.
- The checked-in competition reference describes `TemporalUNet3D` detection plus dense `SimpleNodeTransformer` pair association. Its documented image scale is z/y/x = `1.625/0.40625/0.40625` um per voxel. The approved direction replaces the dense association stage, not the released checkpoint backbone.
- Existing alpha code has physical-scale parsing, sparse proposal/routing/solver paths and local graph validation. The previous G-01 kernel generated a valid four-dataset CSV (`57,004` nodes, `44,370` edges) but its direct-file and kernel submission creation calls returned Kaggle HTTP 403; no scored submission was created.
- The root has no `README.md` or `pyproject.toml`; the only inspected copies are under `external/kaggle-cell-tracking-competition/`. Planning therefore treats project packaging as a Start-time manifest/configuration task rather than assuming a root package contract.
- The worktree was already materially dirty before this packet. The plan must preserve all unrelated changes.

## Inferences and constraints

- The historical local result leaves a material performance gap, but cannot estimate the Kaggle gap without a successful qualifying scored submission.
- Sparse supervision and dense 3D scenes make frame-level random splits and dense all-pairs scoring unacceptable for final validation/runtime claims.
- Cache reuse is essential: model output and maximal candidate graph materialization must precede threshold/cost sweeps, so HPO changes do not repeatedly run the backbone.
- G-01's submit-permission 403 is an external operational risk. It does not authorize a workaround outside the Kaggle scope; it is a stop condition for the controlled-experiment task.

## Scientific basis

- Malin-Mayor et al., *Nature Biotechnology* 2023: sparse point supervision, learned center/motion outputs, and global binary-forest optimization. https://www.nature.com/articles/s41587-022-01427-7
- Trackastra: learned temporal association with explicit division handling. https://arxiv.org/abs/2405.15700
- Ultrack: multi-hypothesis detection/segmentation coupled to global temporal optimization in dense 3D tissues. https://www.nature.com/articles/s41592-025-02778-0

These sources motivate the approved compact learned center/motion-plus-global-solver path; they do not relax the locked checkpoint, sparse association, low-compute, or official-score requirements.

## Requirements

### R-01 — Official sparse-label scorer semantics

Use the official semantics: 7.0 um bipartite node matching; z/y/x physical scales `1.625/0.40625/0.40625` um per voxel; node-overprediction adjustment; edge Jaccard aggregation; sparse-label-aware division connected-component Jaccard; and the combined score. Official semantic fixtures must pass exactly. Local-to-Kaggle transfer is a separate, interval-aware gate: any transfer model used for public-score decisions must meet its preregistered out-of-sample error bound on frozen calibration pairs.

Scorer-code/semantic parity and empirical local-to-Kaggle transfer are separate gates.
Semantic parity authorizes leakage-safe local scoring, decomposition, error taxonomy, and
candidate development. Unresolved transfer calibration blocks fine public-score forecasts,
interpretation of public probes, and final portfolio selection; it does not halt local
model development.

### R-02 — Leakage-safe validation and audit

Create a reproducible hash-group near-duplicate dataset registry, stratify scenario groups, reserve exactly 20 videos as untouched audit, and use grouped three-fold CV for promoted candidates. Training-derived calibration, normalization, route thresholds, and specialist labels may not cross a validation/audit group boundary.

### R-03 — Compact checkpoint-based learned core

Fine-tune only the released official `TemporalUNet3D` checkpoint with a shared backbone and lightweight adapters/heads for sparse centers, backward displacement, uncertainty, division, and scenario specialization. Search only the supplied learning-rate, sparse-negative-weight, threshold, radius, neighbor, cost, gap, division, and unfreeze-depth space. No large from-scratch model is permitted.

### R-04 — Sparse association and valid global lineage

Replace dense NxN association with physical-coordinate KD-tree gating, a bounded sparse learned edge/division scorer, and the constrained global lineage solver. Outputs must have unique node IDs, time-forward edges, at most one parent, at most two children, no merges, valid division motifs, and complete dataset coverage.

### R-05 — Specialist routing and fallback

Evaluate the three specialists out of fold. Train the router only from image statistics and model uncertainty available without labels at inference. Its low-confidence (< `0.60`) universal-adapter path must be measured. Mean regret against the specialist oracle must be <= `0.01`, and worst-scenario regret <= `0.03`.

### R-06 — Budgeted optimization, runtime, and reproducibility

Cache backbone outputs and maximal sparse graphs; use 10/30/100 percent successive halving. Before every paid trial, record a runtime/spend prediction and reject it if its maximum charge could exceed a cap. Run at most two paid trials concurrently, spend at most `$8/day` and `$80/month` on development, and retain `$2/day` and `$20/month` for final validation. Show a public four-clip time <= seven minutes and a measured 199-video/final-workload proxy projected <= six hours including preprocessing, inference, solve, and CSV generation. Manifest every checkpoint/source/fold/package/runtime/score/spend identity.

Initial training is capped at 30 cumulative Kaggle T4 GPU-hours per model; archive any
model that cannot become viable inside that ceiling. RunPod has a `$10` total campaign
cap and is considered only after an accepted score reaches `0.89`, for a
higher-quality-GPU-dependent optimization with literature or theoretical support for an
expected score improvement greater than `5%` (ordinary hard improvement gate `3.5%`).
Every launch still requires explicit human approval after its maximum cost is recorded.

Before any new learned candidate is implemented or trained, require a literature-gate
manifest with primary peer-reviewed evidence, a measured fit-only bottleneck-to-mechanism
mapping, expected improvement of at least `3.5%`, and explicit falsification, stop, and
promotion criteria. Prefer lightweight candidates that can fail quickly on the M5.
Maintain at most one fit-only local training lane in parallel with eligible external
work: cap the process tree at `18 GiB`, reserve at least `6 GiB` for background work,
forbid holdout access and swap growth, and end an unattended stage within seven hours.
Passing this local funnel does not relax the Kaggle or RunPod gates.

### R-07 — Controlled Kaggle evidence and truthful calibration

Use only private training/inference notebooks, directly passing private training outputs to private inference; do not create or publish a Kaggle dataset or model. Make at most two paid trials concurrent, at most five Kaggle submissions per competition day, and select at most two final submissions. Each scored submission needs a frozen artifact hash, local official evidence, valid graph/package evidence, runtime/spend prediction, and ledger entry. Kaggle site evidence is never inferred from local CV.

While transfer calibration is unresolved, every submitted probe consumes this cap and must
be preregistered, observation-distinct, and measured sequentially so later probes can be
redesigned from earlier results. Displayed Kaggle scores are rounded intervals, not exact
responses.

### R-08 — Gated division policy and completion integrity

Do not activate the division branch until edge-only grouped CV >= `0.90`. Retain it only if combined score improves and adjusted edge Jaccard falls by no more than `0.005`. Complete only after R-01 through R-07 and all acceptance criteria pass; preserve the alpha teacher/fallback and prior G-01 evidence throughout.

## Acceptance criteria

| ID | Criterion | Required proof |
| --- | --- | --- |
| AC-01 | Scorer semantics and transfer | Official semantic fixtures and source hashes pass exactly; separately, any local-to-Kaggle transfer model used for public-score decisions meets its preregistered interval-aware out-of-sample bound on frozen calibration pairs. |
| AC-02 | Model quality | Grouped three-fold and untouched 20-video audit combined score each >= `0.92`; complete per-sample node/edge/division diagnostics. |
| AC-03 | Routing | Mean regret <= `0.01`, worst-scenario regret <= `0.03`, and universal fallback test. |
| AC-04 | Graph validity | Every output graph has all required IDs, forward/time/legal degree constraints, valid divisions, and full dataset coverage. |
| AC-05 | Runtime | Public four-clip <= seven minutes and measured final-workload proxy projects <= six hours end to end. |
| AC-06 | External endpoint | A qualifying private Kaggle submission is complete and its displayed score is >= `0.92`. |
| AC-07 | Reproducibility and spend | Immutable manifest joins source/checkpoint/fold/config/package/runtime/scores and pre-/post-trial spend ledger. |

## Scope and protected behavior

In scope: task-owned paths under `src/celltrack_alpha`, `scripts`, `tests`,
`kaggle_kernel`, `reports/g02`, and narrowly necessary configuration/manifests. The
rule-based alpha remains readable and executable as teacher/fallback/regression baseline.
Official CSV schema and sparse-label metric semantics remain unchanged. Ordinary commits
and pushes to this private repository are permitted.

Out of scope: public releases; new Kaggle datasets; account/profile/credential actions;
unrelated Kaggle activity; destructive git-history rewrite; a large from-scratch model;
and restoring dense NxN association. RunPod is in scope only after the `0.89`, `>5%`
evidence, and `$10` gates pass, explicit human approval is given for the particular
launch, and pre-launch hours/$/artifact accounting is complete.

## Completion and block rules

- A score below `0.92` on any local or site check is evidence, not success.
- Stop Blocked if scorer parity cannot be maintained, split leakage is found, graph validity fails after bounded remediation, a predicted charge breaches a cap, Kaggle authentication/submission permission is unavailable, quota prevents the required experiment, or a credible next step needs a forbidden action.
- Never lower score/runtime/parity gates, relabel the historical `0.8713516075529171` local audit as site evidence, or claim an unobserved Kaggle score.
