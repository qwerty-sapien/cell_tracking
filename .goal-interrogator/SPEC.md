# Specification: Low-Compute General Cell Tracking Improvement

- Planning state: Prepared (inactive)
- Mode: New
- Current Goal slice: G-01
- Last reconciled: 2026-07-12
- Repository evidence baseline: unborn `main`; no commit/tree identity. Relevant untracked evidence hashes recorded below; `.env` and `gurobi.lic` were not read. Kaggle ZIP path `data/kaggle/raw/biohub-cell-tracking-during-development.zip`, size `87393127165` bytes, mtime `Jun 26 21:36:15 2026`. Official external checkout `016845facc82e7f5f0c7a37f8da2a9432f51076b`.

## Outcome

Develop and verify a non-duplicate/general Biohub cell-tracking route that targets `>=0.85` official local score under a leakage-safe validation protocol, while staying within a low-compute budget and deleting or disabling the duplicate-GEFF copy path from general-model execution and scoring evidence.

## Repository Evidence

| Evidence | Classification | Source | Consequence |
| --- | --- | --- | --- |
| Current public/test ZIP duplicate route scores `1.1936358875688189` by copying matched train GEFFs for four byte-identical train/test image Zarrs. | Observed | `reports/duplicate_geff_submission_report.json` sha256 `a9c7cda0b96564e2bce38b0457624d9c4dab12e8db664e0557cf7a8a69071a53` | Duplicate route is out of scope for general-model score evidence. R-03, AC-03, D-02. |
| Best current non-duplicate/general fallback evidence is full-video `44b6_0113de3b` official score `0.6283480979676915`, edge Jaccard `0.5964912280701754`, node recall `0.8269230769230769`, no divisions. | Observed | `reports/alpha_routed_full_44b6_0113de3b_offset_metrics.json` sha256 `24561d7258ae5dab1761c52e0eaf5da47a1c0e05c1d40719f2b30d2283e1ad8b` | `>=0.85` requires material model improvement; localization/candidate quality is the first bottleneck. R-01, R-04, AC-01. |
| The full-video score improved from `0.2107` to `0.6283` after a prefix output offset `(z,y,x)=(0,-16,+8)`. | Observed | `reports/alpha_reassessment.md` sha256 `e04c85248cb0c4956dc1f1566dd53d6309c3d48f245cf3da6fb774bd30810272` | Hard-coded offsets are useful diagnostics but leakage-prone; fold-safe localizer/calibration is required. R-05, D-03. |
| Rule-based downloaded Kaggle kernel used bright DoG, `threshold_quantile=0.99955`, `min_distance_um=2.8`, `max_link_um=8.0`, `max_points_per_t=180`, and generated a valid 12,144-node, 9,399-edge CSV in about 758s. | Observed | `data/kaggle/kernel_output/biohub-rule-based-dog-hungarian.log`; local official check to `/private/tmp/kernel_submission_official_eval.json` scored `0.06376641021151673` against duplicate train GT. | Treat DoG/Hungarian as baseline/proposal source, not evidence for final direction. R-04, D-01. |
| Official metric uses 7 micrometer node matching, sparse edge Jaccard, adjusted node-count penalty, division Jaccard, and score `adjusted_edge_jaccard + 0.1 * division_jaccard`. | Observed | `external/kaggle-cell-tracking-competition/metrics.md` sha256 `41a6cf2f6113fc5b169539e8b67e7edce9a92b33d1333e97c212df2a3fede88e`; GitHub official repo lines 256-262, 377-378 | Completion must use official metric wrappers, not approximate metrics alone; local scorer parity is mandatory. R-02, R-11, AC-02, AC-10, D-08. |
| Official baseline uses 3D U-Net detection plus cross-attention transformer linking under sparse supervision. | Observed | `external/kaggle-cell-tracking-competition/README.md` sha256 `463fae4c7c80e2665a958353fdc73f5ebade5231699190f3dc8704cd923fecaf`; GitHub official repo lines 257-262 | Literature/official direction supports sparse point detector/localizer plus learned association as a credible low-compute target if scoped. R-04, R-06. |
| Current alpha seams exist: pipeline, proposals, routing, solver, and official metric wrapper are modular. | Observed | `src/celltrack_alpha/pipeline.py` sha256 `d812867e50c2f74d7599750b571c2aca28562e0dfd13eaf031e5fc4e4c9101fd`; `proposals.py` sha256 `15420ef1a32ec34db0a047583abaed74f5cf0b1784b33779e1d86a7eb992a397`; `solver.py` sha256 `a68db6d28d1edc549a779f9ec12576c02a74ef741767afd7167f88f4efbd06b7`; `official_eval.py` sha256 `07eff2f101aa7f72a27c7e4ca6f302eb5c22314fe1b97c613836c4bc582670f0` | Plan should upgrade stages independently rather than rewrite whole stack. R-07. |
| Local `.venv` is Python `3.14.6`; official external pyproject supports `<3.14`, and Ultrack is optional/incompatible in current environment. | Observed | `.venv/bin/python --version`; `doc_cell_tracking/ARCHITECTURE.md` sha256 `9e3c5ae5c2e66a27c81bf68d8558585a60ec685283ac781c13b57243f32b2df6` | External deep/Ultrack branches need an explicit environment decision and cannot be assumed local. R-08, D-05. |
| Trackastra learns pairwise associations from detections in temporal windows and accounts for divisions without dense image processing. | Observed | https://arxiv.org/abs/2405.15700 | Use learned edge/division scoring after node recall is strong. R-06. |
| Linajea combines weakly supervised center-point evidence, learned division/polar-body detectors, and ILP lineage extraction for dense embryo tracking. | Observed | https://arxiv.org/abs/2208.11467 | Supports sparse center/localizer plus global graph solver as final general direction. R-04, R-06. |
| Ultrack-style multi-hypothesis selection can combine off-the-shelf segmentations and works at large 3D+t scale with ILP. | Observed | https://arxiv.org/abs/2308.04526 | Use Ultrack/multi-hypothesis as optional proposal branch, not first blocking dependency. R-06, D-04. |
| bioRxiv API sampled metadata for 2024-05-20 to 2024-05-31 did not return targeted cell-tracking method records in first 10 of 30; BioStudies search returned mostly non-method imaging studies. | Observed | bioRxiv and BioStudies API calls via named skills on 2026-07-11 | Do not overstate targeted preprint/database evidence; rely on official repo, arXiv method papers, and existing repo reports. R-09. |

## Scope

### In Scope

- Build a leakage-safe, non-duplicate/general validation protocol and run log.
- Double down on metric-aware center localization/candidate generation as the primary direction.
- Add three synergistic branches behind the existing alpha interfaces:
  - deterministic/fold-safe center localizers and candidate fusion;
  - optional Ultrack-style or reduced multi-hypothesis proposal selection;
  - low-compute learned edge/division scoring over candidate graphs.
- Tune and verify with official local metrics, Kaggle-scorer-parity checks, full graph validators, runtime/cost logs, and ablations.
- Produce a Kaggle-compatible low-compute submission path only after the local official gate is met or a blocked stop is reached.

### Non-Goals

- Claiming success from the duplicate-GEFF public-ZIP route.
- Training a large model from scratch or running paid GPU jobs without a separate locked budget approval.
- Using random-frame CV as the decisive success criterion.
- Treating DoG/Hungarian or approximate metrics as sufficient final evidence.
- Production Kaggle submission, Kaggle push, paid compute, destructive cleanup, or secret handling without separate explicit approval. Package installs or environment creation are allowed only inside the D-05 separate-environment branch and still require execution-time approval.

## Requirements

| ID | Requirement | Kind | Source | Decision Links | State |
| --- | --- | --- | --- | --- | --- |
| R-01 | The general route must target official local score `>=0.85` under a predeclared non-duplicate validation protocol. | Behavior | User goal; reports showing `0.6283` current fallback | D-01 | Accepted |
| R-02 | Completion evidence must use the official metric wrapper and include edge, adjusted edge, node recall, division, row-count, and graph-validity diagnostics. | Constraint | Official metric docs and `official_eval.py` | D-01, D-08 | Accepted |
| R-03 | The duplicate-GEFF copy route must be deleted or disabled from general-model execution and excluded from score evidence. | Invariant | Duplicate report score `1.1936`; user deletion preference | D-02 | Accepted |
| R-04 | The first doubled-down modeling direction must improve metric-aware center localization and high-recall candidate generation before solver or division complexity is treated as the main bottleneck. | Behavior | `alpha_reassessment.md`; planners; Linajea/official baseline evidence | D-01, D-03, D-04 | Accepted |
| R-05 | Dataset/prefix output offsets may not be used as held-out/test leakage; any calibration must be fold-safe, replayable from training folds, and recorded in evidence. | Invariant | `44b6` offset risk in `routing.py` and `pipeline.py` | D-03 | Accepted |
| R-06 | Ultrack/multi-hypothesis and learned association/division scoring must be incorporated as optional, measurable branches that feed common candidate/edge abstractions and are enabled only when ablations improve official CV. | Behavior | User hybridization preference; Trackastra/Ultrack/Linajea evidence | D-04, D-06 | Accepted |
| R-07 | Implementation must preserve current alpha modular boundaries while removing/disabling duplicate-copy routing from the general model path. | Constraint | `celltrack_alpha` architecture; D-02 | D-02 | Accepted |
| R-08 | The main path must stay local/Kaggle-free and low-compute; an independent Python 3.11/3.12 environment path may test Ultrack/official-baseline compatibility; paid GPU and Kaggle submission remain unapproved. | Constraint | User budget intent; D-05 | D-05 | Accepted |
| R-09 | Literature and Kaggle claims must separate observed local evidence, live-source evidence, and unavailable/private leaderboard evidence. | Deliverable | Kaggle fetch limitations; bioRxiv/BioStudies API results | None | Accepted |
| R-10 | The result must be reproducible from recorded code/config/data identities despite the unborn Git state and untracked workspace. | Invariant | `doc_cell_tracking/EVALUATION.md`; git status | D-07 | Accepted |
| R-11 | The local scorer/evaluation path must match Kaggle competition scoring semantics closely enough for iterative local testing. | Constraint | User scorer-parity instruction; official metric source | D-08 | Accepted |

## Acceptance Criteria

| ID | Protects | Observable Pass Condition | Required Evidence | Mandatory? |
| --- | --- | --- | --- | --- |
| AC-01 | R-01, R-03 | Frozen general route reaches `>=0.85` official score on the locked non-duplicate CV protocol, with duplicate route disabled/deleted and leakage report clean. | Official JSON score table; command lines; split manifest; leakage report; config hash. | Yes |
| AC-02 | R-02 | GT-as-prediction self-check returns edge Jaccard `1.0`; candidate route evaluation uses `metric_source=official`, not approximate-only metrics. | `scripts/evaluate_alpha_official_zip.py --gt-self-check` output and scorer version identity. | Yes |
| AC-03 | R-03, R-05 | No GEFF-copy route, train/test duplicate route, held-out-label-derived offset, or dataset-ID-specific held-out calibration is used in claimed general score. | Static/command inspection plus run config manifest and calibration provenance. | Yes |
| AC-04 | R-04 | Candidate/localizer stage meets a locked pre-solver gate for 7um node recall and node-count ratio, or records a blocked stop proving localization remains the bottleneck. | Candidate recall report across both `44b6` and `6bba` held-out folds. | Yes |
| AC-05 | R-06 | Ultrack/multi-hypothesis branch and learned edge/division scoring are each ablated; enabled branches improve official CV or remain disabled with evidence. | Ablation JSON; route comparison table; learned scorer calibration report if used. | Yes |
| AC-06 | R-07 | Output graphs remain valid: no duplicate node IDs, missing endpoints, duplicate edges, merges, out-degree `>2`, invalid times, or missing requested datasets. | Submission validator output and graph diagnostics. | Yes |
| AC-07 | R-08 | Runtime, memory proxy, storage output size, compute source, and optional environment provenance are recorded; no unapproved paid/Kaggle submission side effect occurred. | Compute log; side-effect ledger; approval record if any network/env action is used. | Yes |
| AC-08 | R-09 | Final report cites Kaggle best-solution evidence only when directly observed; inaccessible Kaggle discussion/leaderboard pages are marked unavailable. | Source ledger with local, web, and unavailable evidence categories. | Yes |
| AC-09 | R-10 | Reproduction manifest records relevant file hashes, ZIP identity, external checkout, command lines, configs, output hashes, and dependency versions. | Manifest under `reports/` or `.goal-interrogator/CLOSEOUT.md` after activation. | Yes |
| AC-10 | R-02, R-11 | Local scorer parity tests demonstrate the local iterative scorer matches official Kaggle metric behavior on GT-self, known predictions, adjusted node-count penalty, sparse edge matching, and division cases. | Scorer parity report, official metric version identity, comparison fixtures/results. | Yes |

## Protected Behavior and Invariants

- R-03: The duplicate-GEFF copy route must not count toward `>=0.85` general-model evidence and should be removed or disabled from the activated general path.
- R-05: Hard-coded offsets are diagnostics unless derived fold-safely; they cannot silently affect held-out claims.
- R-08: Activation does not authorize Kaggle submission, paid GPU use, Runpod use, or modifying/reading `.env`/`gurobi.lic`. A separate Python 3.11/3.12 path is allowed by D-05 only as an independent compatibility branch with provenance and execution-time command approval.
- R-10: Secrets are excluded from manifests and never printed.
- R-11: Approximate metrics may be retained as cheap diagnostics but cannot be the scorer used for completion or route selection evidence.

## Constraints and Side-Effect Boundaries

- No implementation or external side effects occur during this interrogation.
- Execution writes must be limited to approved source/script/test/report/prediction/derived paths in PLAN.
- The D-05 optional environment branch may use a separate Python `3.11` or `3.12` environment, but concrete network/package install commands still require execution-time approval and provenance.
- Kaggle CLI, Runpod, Gurobi, full-ZIP exhaustive runs, paid GPU, and secret reads remain unapproved unless a later locked decision names the boundary.
- The current local Python is `3.14.6`; Ultrack and official training stack may require Python `3.11` or `3.12`.

## Risks and Rollback Expectations

| Risk | Affected IDs | Prevention or Detection | Rollback / Stop Expectation |
| --- | --- | --- | --- |
| Duplicate route leakage gives a misleading high score. | R-01, R-03, AC-01, AC-03 | Explicit leak guard and static/config inspection. | Stop; rerun with duplicate disabled and calibration reset. |
| Hard-coded offsets overfit a prefix/video. | R-05, AC-03, AC-04 | Fold-safe calibration manifest and held-out split lock. | Disable offsets or convert to train-only localizer; rerun affected evidence. |
| Candidate recall remains below score ceiling. | R-04, AC-04 | Candidate recall/overcount gate before learned edge work. | Stop or reopen direction before training larger models. |
| Optional Ultrack/deep dependencies consume time or break Python environment. | R-06, R-08 | Optional branch, separate environment decision, no `.venv` mutation without approval. | Keep reduced in-repo multi-hypothesis branch; do not block main path. |
| Learned scorer improves local routes but overfits sparse labels. | R-01, R-06 | Video-held-out protocol, calibration curves, ablation. | Disable scorer or lower scope to decision-complete report. |
| Full ZIP runtime exceeds low-compute budget. | R-08, AC-07 | Early runtime extrapolation and resource log. | Stop before full run; ask for budget/strategy revision. |

## Assumptions and Open Material Questions

| Item | Decision Link | Validation or Reopening Trigger | Owner |
| --- | --- | --- | --- |
| Concrete package-install or environment-creation commands for the optional Python 3.11/3.12 branch still require execution-time approval. | D-05 | Executor reaches optional branch setup. | Execution owner / User |

## Traceability Summary

- Requirements without acceptance criteria: None.
- Acceptance criteria without deterministic evidence yet: AC-01, AC-04, AC-05, AC-07, AC-09, and AC-10 are execution evidence and remain pending by design.
- Locked decisions not reflected in requirements: None.
- Open blocker before `Prepared (inactive)`: None. The plan is prepared but inactive until the user explicitly activates implementation.
- Deferred work: paid GPU/external training and Kaggle submission/leaderboard operation remain future slices or later approvals.
