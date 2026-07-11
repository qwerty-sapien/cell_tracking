# Plan: G-01 - Leakage-Safe General Model Improvement

- Planning state: Draft
- Repository evidence baseline: unborn `main`; no commit/tree. Relevant hashes are recorded in SPEC. External official checkout `016845facc82e7f5f0c7a37f8da2a9432f51076b`.
- Source artifacts: `SPEC.md`, `DECISIONS.md`, `ROADMAP.md`, and this `PLAN.md` are Draft; D-01 through D-06 and D-08 are locked, D-07 remains Proposed, and no approval-bound final digests exist yet.
- Integration owner: Serialized integration owner.
- Independent planning: Complete. Three read-only reconnaissance lanes and two read-only planning passes ran through subagents. One implementation planner incorrectly disclosed reduced assurance despite subagent execution; synthesis treats it as an independent read-only proposal but does not adopt that reduced-assurance statement.

## Required Completion Evidence

| AC ID | Protects | Deterministic Final Evidence | Evidence Input | Producing Node | Coordinated Gate | Mandatory? |
| --- | --- | --- | --- | --- | --- | --- |
| AC-01 | R-01, R-03 | Official local CV summary score `>=0.85`, duplicate disabled, split locked before run. | Frozen prediction CSVs, split manifest, run configs. | N10 | VG-01, VG-02, VG-04 | Yes |
| AC-02 | R-02 | GT self-check edge Jaccard `1.0`; evaluated route rows show `metric_source=official`. | Official scorer output JSON. | N2, N10 | VG-01 | Yes |
| AC-03 | R-03, R-05 | Leakage report finds no GEFF-copy, duplicate route score use, held-out-label offsets, or dataset-ID calibration on held-out claims. | Configs, source/static inspection, calibration manifest. | N3, N10 | VG-02, VG-03 | Yes |
| AC-04 | R-04 | Candidate/localizer report meets locked 7um recall and node-count-ratio gate, or records blocked bottleneck evidence. | Candidate recall reports by fold/prefix. | N5 | VG-05 | Yes |
| AC-05 | R-06 | Ultrack/reduced multi-hypothesis and learned scorer branches are ablated; only official-CV-improving branches are enabled. | Ablation and route comparison reports. | N8, N9, N10 | VG-06 | Yes |
| AC-06 | R-07 | Graph validation reports no duplicate IDs/edges, missing endpoints, merges, out-degree `>2`, invalid time, or dataset omissions. | Submission CSVs and diagnostics. | N10 | VG-07 | Yes |
| AC-07 | R-08 | Runtime/cost/source log shows low-compute compliance or approved exception. | Compute log and side-effect ledger. | N10 | VG-08 | Yes |
| AC-08 | R-09 | Source ledger separates observed local evidence, live web/literature evidence, and unavailable Kaggle/bioRxiv/BioStudies evidence. | Final report/source ledger. | N10 | VG-09 | Yes |
| AC-09 | R-10 | Reproduction manifest includes file/data/config/output/dependency identities and excludes secrets. | Manifest and output hashes. | N0, N10 | VG-10 | Yes |
| AC-10 | R-02, R-11 | Local scorer parity report verifies iterative scoring matches official Kaggle metric behavior. | Parity fixtures/results and official metric version identity. | N2, N10 | VG-11 | Yes |

## Proposal Reconciliation

- Implementation proposal strengths: Focused on concrete localizer interface, deterministic refiners, candidate recall reports, edge scorer tables, and route ablations mapped to current alpha seams.
- Architecture/risk proposal strengths: Strong leakage controls, duplicate-route quarantine, offset leakage gate, environment/compute boundaries, and official score protocol.
- Critic findings applied by main interrogator: Current plan makes duplicate and offset leakage blocking gates, prioritizes localization before solver/division, treats Ultrack/deep branches as optional, and requires official held-out evidence.
- Rejected alternatives: Using duplicate public-ZIP score as success evidence; global prefix offsets without fold-safe provenance; making Ultrack or deep training a blocking first step; using public leaderboard as the only target before local CV is defined.
- Remaining uncertainty: D-07 remains Proposed. This PLAN stays Draft until the reproducibility baseline is locked or safely revised.

## Initial Inputs

| Input ID | Existing Source And Version | Consumers | Validation |
| --- | --- | --- | --- |
| I-01 | Kaggle ZIP `data/kaggle/raw/biohub-cell-tracking-during-development.zip`, size `87393127165` bytes. | N0, N2, N5, N10 | Stat/ZIP inspection and existing `reports/kaggle_zip_inspection.md`. |
| I-02 | Official metric repo checkout `016845facc82e7f5f0c7a37f8da2a9432f51076b`. | N2, N10, T1 | `git -C external/kaggle-cell-tracking-competition rev-parse HEAD`. |
| I-03 | Current alpha pipeline files listed in SPEC evidence. | N3-N10 | SHA-256 hashes recorded in SPEC. |
| I-04 | Current fallback score `0.6283` and duplicate route score `1.1936`. | N1, N10 | Existing reports hashed in SPEC. |
| I-05 | Literature/source evidence: official baseline, Linajea, Trackastra, Ultrack, and local research docs. | N4, N8, N9, N10 | Source ledger in SPEC and final report. |

## DAG Nodes

### N0 - Freeze Baseline And Run Ledger

- **Objective:** Create a reproducibility baseline and run-log schema for the activated experiment.
- **Owner:** Integration owner.
- **Depends on:** None.
- **Inputs:** I-01, I-02, I-03, I-04.
- **Outputs provided:** `baseline_manifest`, `general_model_run_log_schema`.
- **Read scope:** Repository metadata, relevant source/report/docs files, ZIP metadata; not `.env` or `gurobi.lic`.
- **Write ownership:** `reports/general_model_run_log.csv`, `reports/general_model_baseline_manifest.json`, `.goal-interrogator/CLOSEOUT.md` after activation.
- **Remediation ownership handoff:** None; integration owner already owns.
- **Requirements:** R-10.
- **Acceptance criteria:** AC-09.
- **Decisions:** D-07.
- **Actions:** Hash relevant files/configs; record environment versions and ZIP identity; define run-log fields for config hash, split, route, score, runtime, and side effects.
- **Node completion evidence:** Manifest exists, excludes secrets, and records command/file identities.
- **Gates triggered:** VG-10.
- **Side effects / approvals:** None if using local read-only hashing and report writes after activation.
- **Failure / rollback / stop:** Stop if relevant inputs cannot be identified without reading secrets.
- **Parallel-safe with:** None; baseline must precede evidence-producing nodes.

### N1 - Lock Non-Duplicate Split And Leakage Policy

- **Objective:** Define the validation protocol, duplicate-route delete/disable rule, and leakage checks before model tuning.
- **Owner:** Integration owner.
- **Depends on:** N0.
- **Inputs:** `N0:baseline_manifest`, I-04.
- **Outputs provided:** `split_manifest`, `leakage_policy`, `duplicate_delete_or_disable_rule`.
- **Read scope:** `scripts/run_alpha_routed_zip.py`, `scripts/build_duplicate_geff_submission.py`, `src/celltrack_alpha/duplicate.py`, current reports.
- **Write ownership:** `reports/general_model_split_manifest.json`, `reports/general_model_leakage_policy.md`.
- **Remediation ownership handoff:** None.
- **Requirements:** R-01, R-03, R-05.
- **Acceptance criteria:** AC-01, AC-03.
- **Decisions:** D-01, D-02, D-03.
- **Actions:** Encode held-out folds; remove or disable duplicate-GEFF routing in general CV; define offset/calibration provenance requirements.
- **Node completion evidence:** Split manifest and leakage policy are present and referenced by later commands.
- **Gates triggered:** VG-02, VG-03, VG-04.
- **Side effects / approvals:** None.
- **Failure / rollback / stop:** Stop if D-01 through D-03 are not locked or if duplicate-copy behavior remains reachable from the general route.
- **Parallel-safe with:** None; downstream nodes depend on policy.

### N2 - Confirm Official Metric Harness And Scorer Parity

- **Objective:** Prove the scorer and validation commands are current, official-metric based, and suitable for iterative Kaggle-score-aligned testing.
- **Owner:** Evaluation owner.
- **Depends on:** N1.
- **Inputs:** I-01, I-02, `N1:split_manifest`.
- **Outputs provided:** `official_metric_self_check`, `scorer_parity_report`, `evaluation_command_template`.
- **Read scope:** `src/celltrack_alpha/official_eval.py`, `scripts/evaluate_alpha_official_zip.py`, official metric source.
- **Write ownership:** `src/celltrack_alpha/evaluation.py`, `src/celltrack_alpha/official_eval.py`, `scripts/evaluate_alpha_official_zip.py`, scorer parity tests/fixtures, `reports/general_model_official_metric_self_check.json`, `reports/*official_eval*.json`, `reports/*scorer_parity*.json`.
- **Remediation ownership handoff:** Integration owner after any scorer change.
- **Requirements:** R-02, R-11.
- **Acceptance criteria:** AC-02, AC-10.
- **Decisions:** D-01, D-08.
- **Actions:** Run GT-as-prediction self-check on representative folds; add/verify scorer parity fixtures for sparse matching, adjusted node-count penalty, division scoring, and known predictions; document exact official scorer version and command template.
- **Node completion evidence:** Self-check edge Jaccard `1.0`, official metric available, and scorer parity report passes.
- **Gates triggered:** VG-01, VG-11.
- **Side effects / approvals:** None for local scoring; full ZIP scoring governed by D-05.
- **Failure / rollback / stop:** Stop if official scorer unavailable, scorer parity diverges, or approximate-only metrics are required for route selection/completion.
- **Parallel-safe with:** None until split policy exists; later read-only evaluation may overlap after outputs stabilize.

### N3 - Replace Hard-Coded Offset Use With Fold-Safe Localizer Contract

- **Objective:** Create a localizer/refiner interface and remove unsafe reliance on hard-coded held-out offsets.
- **Owner:** Localizer owner.
- **Depends on:** N1, N2.
- **Inputs:** `N1:leakage_policy`, current `CandidateNode`/proposal interfaces.
- **Outputs provided:** `localizer_interface`, `offset_provenance_manifest`.
- **Read scope:** `src/celltrack_alpha/types.py`, `proposals.py`, `pipeline.py`, `routing.py`, calibration scripts.
- **Write ownership:** `src/celltrack_alpha/localization.py`, scoped edits in `src/celltrack_alpha/types.py`, `src/celltrack_alpha/pipeline.py`, `src/celltrack_alpha/proposals.py`, `scripts/calibrate_alpha_offsets.py` or a new calibration script, localizer tests.
- **Remediation ownership handoff:** Integration owner owns all modified alpha interfaces after N4 integration begins.
- **Requirements:** R-04, R-05, R-07.
- **Acceptance criteria:** AC-03, AC-04, AC-06.
- **Decisions:** D-03, D-04.
- **Actions:** Define deterministic patch/refiner interface; encode calibration source/fold provenance; make unsafe prefix offsets detectable.
- **Node completion evidence:** Unit tests or deterministic checks for localizer provenance and no held-out leakage.
- **Gates triggered:** VG-03, VG-05.
- **Side effects / approvals:** None.
- **Failure / rollback / stop:** Stop if localizer cannot be made fold-safe under D-03.
- **Parallel-safe with:** N6 only after interface output is stable; otherwise no.

### N4 - Implement Candidate Localization And Fusion Improvements

- **Objective:** Improve center localization and high-recall candidate generation with low-compute deterministic methods first.
- **Owner:** Localizer/proposal owner.
- **Depends on:** N3.
- **Inputs:** `N3:localizer_interface`, I-03.
- **Outputs provided:** `candidate_generation_route`, `candidate_recall_inputs`.
- **Read scope:** Alpha proposal/routing code and selected train folds.
- **Write ownership:** `src/celltrack_alpha/proposals.py`, `src/celltrack_alpha/routing.py`, `src/celltrack_alpha/localization.py`, focused tests.
- **Remediation ownership handoff:** Integration owner before N10.
- **Requirements:** R-04, R-07.
- **Acceptance criteria:** AC-04, AC-06.
- **Decisions:** D-03, D-04.
- **Actions:** Add local patch centroid, dark-center/refined-center, component-center, and fold-safe calibration options; keep DoG as one source; preserve physical-unit NMS.
- **Node completion evidence:** Candidate generation runs on small locked clips and produces valid candidate tables.
- **Gates triggered:** VG-05, VG-07.
- **Side effects / approvals:** None beyond local report/source writes.
- **Failure / rollback / stop:** If candidate generation overproduces or cannot improve 7um recall, stop before learned scorer.
- **Parallel-safe with:** N6 after N3 if write scopes are separated; final integration serialized.

### N5 - Candidate Recall And Overcount Gate

- **Objective:** Quantify node recall, edge-candidate recall, overcount ratio, and localization failures before heavier modeling.
- **Owner:** Evaluation owner.
- **Depends on:** N4.
- **Inputs:** `N4:candidate_generation_route`, `N2:evaluation_command_template`.
- **Outputs provided:** `candidate_recall_report`, `candidate_gate_decision`.
- **Read scope:** Candidate outputs, train GEFFs, official metric wrapper.
- **Write ownership:** `scripts/diagnose_alpha_detection.py`, `scripts/run_alpha_routed_zip.py` scoped metric-output changes, `reports/*candidate_recall*.json`.
- **Remediation ownership handoff:** Integration owner after report is consumed by N8/N9.
- **Requirements:** R-04.
- **Acceptance criteria:** AC-04.
- **Decisions:** D-01, D-04.
- **Actions:** Run representative held-out clips/folds; record recall at 7um, node count ratio, failure clusters, runtime.
- **Node completion evidence:** Candidate recall report meets the locked gate or declares localization blocked.
- **Gates triggered:** VG-05.
- **Side effects / approvals:** Full ZIP run requires D-05 if considered large; otherwise use bounded clips/folds.
- **Failure / rollback / stop:** Stop if candidate recall is below the locked threshold and no localizer branch remains.
- **Parallel-safe with:** N7 after candidate report exists.

### N6 - Optional Multi-Hypothesis / Ultrack-Compatible Proposal Branch

- **Objective:** Add a reduced multi-hypothesis proposal branch or Ultrack-compatible adapter without making optional dependencies mandatory.
- **Owner:** Multi-hypothesis owner.
- **Depends on:** N3, N5.
- **Inputs:** `N3:localizer_interface`, `N5:candidate_recall_report`.
- **Outputs provided:** `multi_hypothesis_candidates`, `ultrack_branch_status`.
- **Read scope:** `src/celltrack_ultrack/core.py`, alpha proposal interfaces, docs.
- **Write ownership:** `src/celltrack_ultrack/core.py`, optional adapter files under `src/celltrack_alpha/`, `scripts/run_ultrack_zip_cv.py`, reports under `reports/*ultrack*`.
- **Remediation ownership handoff:** Integration owner before N10.
- **Requirements:** R-06, R-08.
- **Acceptance criteria:** AC-05, AC-07.
- **Decisions:** D-04, D-05.
- **Actions:** Prefer reduced in-repo multi-threshold/hypothesis candidates; run true Ultrack only if environment approval exists.
- **Node completion evidence:** Ablation shows added recall/score or branch remains disabled with reason.
- **Gates triggered:** VG-06, VG-08.
- **Side effects / approvals:** A separate Python 3.11/3.12 environment is allowed by D-05 for this optional branch; concrete network/package install commands still require execution-time approval and provenance.
- **Failure / rollback / stop:** Disable branch if dependency/runtime risk exceeds benefit.
- **Parallel-safe with:** N7/N8 only after candidate table contract is stable and write scopes are disjoint.

### N7 - Build Candidate Graph Training Table

- **Objective:** Build a fold-safe table of candidate nodes, candidate edges, GT matches, and hard negatives for low-compute scoring.
- **Owner:** Scoring-data owner.
- **Depends on:** N5.
- **Inputs:** `N5:candidate_recall_report`, candidate graph outputs, train GEFFs.
- **Outputs provided:** `edge_training_table`, `division_training_table`.
- **Read scope:** Candidate outputs, GT arrays, `src/celltrack_alpha/types.py`, `solver.py`.
- **Write ownership:** `src/celltrack_alpha/scoring.py`, `scripts/build_edge_training_table.py`, `data/derived/edge_scorer/`, `reports/*edge_training_table*.json`.
- **Remediation ownership handoff:** Integration owner before N9.
- **Requirements:** R-06, R-10.
- **Acceptance criteria:** AC-05, AC-09.
- **Decisions:** D-04, D-06, D-07.
- **Actions:** Generate positives/negatives from fold-safe matches; add engineered features; record table schema/hash.
- **Node completion evidence:** Training table manifest with no held-out leakage.
- **Gates triggered:** VG-02, VG-06, VG-10.
- **Side effects / approvals:** Derived data writes only to `data/derived/edge_scorer/`.
- **Failure / rollback / stop:** Stop if sparse labels cannot produce meaningful fold-safe training examples.
- **Parallel-safe with:** N6 after N5 if write scopes remain disjoint.

### N8 - Train Low-Compute Edge Scorer

- **Objective:** Train and calibrate a small engineered-feature scorer before any large neural model.
- **Owner:** Scoring owner.
- **Depends on:** N7.
- **Inputs:** `N7:edge_training_table`.
- **Outputs provided:** `edge_scorer_model`, `scorer_calibration_report`.
- **Read scope:** Training table and scoring module.
- **Write ownership:** `src/celltrack_alpha/scoring.py`, `scripts/train_edge_scorer.py`, `data/derived/edge_scorer/`, `reports/*edge_scorer*.json`.
- **Remediation ownership handoff:** Integration owner before N9.
- **Requirements:** R-06, R-08.
- **Acceptance criteria:** AC-05, AC-07.
- **Decisions:** D-04, D-05.
- **Actions:** Start with logistic/gradient boosting/small MLP if supported; record calibration and ranking metrics; avoid large 3D training.
- **Node completion evidence:** Scorer improves true-edge ranking over distance baseline or remains disabled.
- **Gates triggered:** VG-06, VG-08.
- **Side effects / approvals:** Package installs or GPU use require locked approval.
- **Failure / rollback / stop:** Disable scorer if it overfits or fails held-out calibration.
- **Parallel-safe with:** None for shared scoring outputs.

### N9 - Integrate Scorer And Precision-First Division Logic

- **Objective:** Feed calibrated edge/division probabilities into solver costs and enable divisions only if official CV improves.
- **Owner:** Solver owner.
- **Depends on:** N8; N6 optional.
- **Inputs:** `N8:edge_scorer_model`, optional `N6:multi_hypothesis_candidates`.
- **Outputs provided:** `scored_solver_route`, `division_route_status`.
- **Read scope:** Solver, types, route configs, scorer outputs.
- **Write ownership:** `src/celltrack_alpha/solver.py`, `src/celltrack_alpha/types.py`, `src/celltrack_alpha/routing.py`, solver/scorer tests.
- **Remediation ownership handoff:** Integration owner before N10.
- **Requirements:** R-06, R-07.
- **Acceptance criteria:** AC-05, AC-06.
- **Decisions:** D-04, D-06.
- **Actions:** Add scorer cost hooks; calibrate birth/death/link/division thresholds; enable divisions only with official-CV benefit.
- **Node completion evidence:** Valid graph outputs and ablation showing scorer/division effect.
- **Gates triggered:** VG-06, VG-07.
- **Side effects / approvals:** None.
- **Failure / rollback / stop:** Disable scorer/division route if graph validity or official score regresses.
- **Parallel-safe with:** None; solver is shared integration surface.

### N10 - Final Integration, Ablation, Packaging Dry Run

- **Objective:** Select frozen low-compute route, run official CV, validate graph/submission, and record complete evidence.
- **Owner:** Serialized integration owner.
- **Depends on:** N2, N5, N6 optional, N9.
- **Inputs:** All prior outputs.
- **Outputs provided:** `frozen_general_route`, `official_cv_report`, `ablation_report`, `submission_dry_run`, `final_manifest`.
- **Read scope:** Entire affected alpha/scripts/tests/report surfaces, ZIP metadata, generated outputs.
- **Write ownership:** `scripts/run_alpha_routed_zip.py`, `scripts/evaluate_alpha_official_zip.py`, `kaggle_kernel/` only if D-05 allows packaging dry-run, `reports/*general_model*.json`, `data/kaggle/predictions/*general*.csv`, `.goal-interrogator/CLOSEOUT.md` after activation.
- **Remediation ownership handoff:** Already integration owner; earlier owners become read-only for touched scopes.
- **Requirements:** R-01 through R-11.
- **Acceptance criteria:** AC-01 through AC-10.
- **Decisions:** D-01 through D-08.
- **Actions:** Run official held-out CV, scorer parity checks, ablations, graph validation, compute logging, source ledger, and optional packaging dry run.
- **Node completion evidence:** All mandatory ACs pass or an explicit blocked condition is recorded.
- **Gates triggered:** VG-01 through VG-11.
- **Side effects / approvals:** Kaggle push/submission, paid compute, Runpod, secret reads, or full ZIP run require separate locked approval. The optional Python 3.11/3.12 branch is allowed by D-05 but concrete install/network commands still need execution-time approval.
- **Failure / rollback / stop:** Stop if `>=0.85` not reached after in-scope iteration or if mandatory evidence is unavailable.
- **Parallel-safe with:** T1 read-only after stable outputs.

### T1 - Verification Coordination

- **Objective:** Coordinate selected adversarial gates and report evidence/status without remediation writes.
- **Owner:** Read-only verification coordinator.
- **Depends on:** N10.
- **Inputs:** `N10` outputs and gate definitions.
- **Outputs provided:** Gate report returned to integration owner for `CLOSEOUT.md`.
- **Read scope:** Source diffs, configs, reports, prediction CSVs, manifests, logs, official metric outputs.
- **Write ownership:** None.
- **Remediation ownership handoff:** None; T1 never writes or remediates.
- **Requirements:** R-01 through R-11.
- **Acceptance criteria:** AC-01 through AC-10.
- **Decisions:** D-01 through D-08.
- **Actions:** Inspect evidence freshness, run read-only checks where safe, challenge leakage, score, graph, compute, and provenance claims.
- **Node completion evidence:** Gate report with PASS/BLOCK/ADVISORY/NOT_RUN statuses.
- **Gates triggered:** VG-01 through VG-11.
- **Side effects / approvals:** None; privileged checks must be run by authorized integration owner.
- **Failure / rollback / stop:** Any mandatory BLOCK or unwaived NOT_RUN prevents completion.
- **Parallel-safe with:** Read-only review after N10 outputs stabilize.

## Ownership Ledger

| Scope Or Resource | Kind | Initial Sole Writer | Remediation Sole Writer | Other Readers | Handoff / Serialization Point |
| --- | --- | --- | --- | --- | --- |
| `.goal-interrogator/CLOSEOUT.md` | Execution record | Integration owner | Integration owner | T1 | After activation only. |
| `src/celltrack_alpha/localization.py` | Source | N3/N4 localizer owner | Integration owner | N5-N10, T1 | Handoff before N10. |
| `src/celltrack_alpha/proposals.py`, `pipeline.py`, `routing.py` | Shared source | N3/N4 localizer owner | Integration owner | N5-N10, T1 | Handoff before solver/package integration. |
| `src/celltrack_alpha/solver.py`, `types.py` | Shared source | N9 solver owner | Integration owner | N7-N10, T1 | Handoff before N10. |
| `src/celltrack_alpha/scoring.py` | Source/model glue | N7/N8 scoring owner | Integration owner | N9-N10, T1 | Handoff before N9 integration. |
| `src/celltrack_ultrack/core.py` | Optional branch source | N6 multi-hypothesis owner | Integration owner | N10, T1 | Handoff before N10 if enabled. |
| `scripts/run_alpha_routed_zip.py`, eval scripts, scorer parity fixtures | Shared scripts/tests | N2/N5 evaluation owner, then N10 integration owner | Integration owner | T1 | Serialized at N10. |
| `reports/general_model_*`, `reports/*candidate*`, `reports/*ablation*` | Evidence | Producing node owner | Integration owner | T1 | Each report consumed after producer completes. |
| `data/derived/edge_scorer/` | Derived cache/model | N7/N8 scoring owner | Integration owner | N9-N10, T1 | Handoff before N9. |
| `data/kaggle/predictions/*general*.csv` | Prediction artifact | N10 integration owner | Integration owner | T1 | N10 only. |
| `src/celltrack_alpha/duplicate.py`, duplicate build script path | Duplicate shortcut | N1 integration owner | Integration owner | T1 | Remove or disable from general path; preserve historical reports only. |
| `.env`, `gurobi.lic` | Secrets/licenses | None | None | None | Never read/print/write unless separate approval. |

## Verification Gates

### VG-01 - Official Metric Fidelity

- **Class:** Mandatory
- **Threat / failure hypothesis:** Approximate metrics or stale scorer output are used to claim `>=0.85`.
- **Protected requirements:** R-01, R-02, R-11.
- **Protected acceptance criteria:** AC-01, AC-02, AC-10.
- **Protected decisions:** D-01, D-08.
- **Preconditions:** N2/N10 outputs stable.
- **Deterministic checks:**
  1. Run `scripts/evaluate_alpha_official_zip.py` with `--gt-self-check` on locked folds.
     - Expected: GT self edge Jaccard `1.0`; route rows show official scorer and summary score.
  2. Inspect metrics JSON.
     - Expected: no approximate-only score used for completion.
- **Reviewer duty:** Confirm command inputs match locked split and current predictions.
- **Evidence record:** Pending execution.
- **Status:** NOT_RUN
- **Status rationale:** Draft plan only.
- **Remediation owner:** Integration owner after handoff.
- **Retest requirements:** Full VG-01 after scorer/prediction changes.
- **Waiver decision:** None.

### VG-02 - Duplicate Leakage

- **Class:** Mandatory
- **Threat / failure hypothesis:** GEFF-copy route or train/test duplicate identity boosts the claimed general score.
- **Protected requirements:** R-03.
- **Protected acceptance criteria:** AC-01, AC-03.
- **Protected decisions:** D-02.
- **Preconditions:** Split/config/prediction outputs stable.
- **Deterministic checks:**
  1. Inspect route config/run manifest and prediction provenance.
     - Expected: duplicate route deleted, disabled, or unreachable for general CV.
  2. Inspect source/static references.
     - Expected: no `build_duplicate_geff_submission.py` output is used as general prediction evidence.
- **Reviewer duty:** Challenge any score that includes public duplicate datasets as model evidence.
- **Evidence record:** Pending execution.
- **Status:** NOT_RUN
- **Status rationale:** Draft plan only.
- **Remediation owner:** Integration owner.
- **Retest requirements:** Repeat after any routing or submission-path change.
- **Waiver decision:** None.

### VG-03 - Offset Leakage

- **Class:** Mandatory
- **Threat / failure hypothesis:** Hard-coded prefix/dataset offsets leak held-out labels or overfit public examples.
- **Protected requirements:** R-05.
- **Protected acceptance criteria:** AC-03, AC-04.
- **Protected decisions:** D-03.
- **Preconditions:** Calibration/localizer manifest available.
- **Deterministic checks:**
  1. Inspect calibration manifest.
     - Expected: held-out/test labels not used; offsets/refiners derived from allowed training folds only.
  2. Inspect route config for hard-coded prefix offsets affecting held-out claim.
     - Expected: either disabled or fold-safe provenance linked.
- **Reviewer duty:** Verify a passing score is not caused by target-specific manual offsets.
- **Evidence record:** Pending execution.
- **Status:** NOT_RUN
- **Status rationale:** Draft plan only.
- **Remediation owner:** Integration owner.
- **Retest requirements:** Repeat after any localizer or route-config change.
- **Waiver decision:** None.

### VG-04 - Split Integrity

- **Class:** Mandatory
- **Threat / failure hypothesis:** Random-frame or post-hoc split selection inflates score.
- **Protected requirements:** R-01, R-10.
- **Protected acceptance criteria:** AC-01, AC-09.
- **Protected decisions:** D-01, D-07.
- **Preconditions:** Split manifest locked before tuning/evaluation.
- **Deterministic checks:**
  1. Inspect split manifest timestamp/hash relative to runs.
     - Expected: split predates score-producing runs.
  2. Inspect folds.
     - Expected: video/dataset-level or explicitly locked protocol, not random-frame final gate.
- **Reviewer duty:** Check route selection did not use held-out metric feedback outside allowed ablations.
- **Evidence record:** Pending execution.
- **Status:** NOT_RUN
- **Status rationale:** Draft plan only.
- **Remediation owner:** Integration owner.
- **Retest requirements:** Rerun full CV if split changes.
- **Waiver decision:** None.

### VG-05 - Candidate Localizer Gate

- **Class:** Mandatory
- **Threat / failure hypothesis:** Solver/scorer work proceeds when missing/mislocalized nodes cap the score below target.
- **Protected requirements:** R-04.
- **Protected acceptance criteria:** AC-04.
- **Protected decisions:** D-03, D-04.
- **Preconditions:** Candidate route stable.
- **Deterministic checks:**
  1. Run candidate recall diagnostic on locked folds.
     - Expected: locked recall/overcount threshold met, or explicit blocked bottleneck recorded.
  2. Inspect per-prefix/per-depth failures.
     - Expected: both `44b6` and `6bba` represented.
- **Reviewer duty:** Confirm node recall and node-count ratio both support the claimed route.
- **Evidence record:** Pending execution.
- **Status:** NOT_RUN
- **Status rationale:** Draft plan only.
- **Remediation owner:** Integration owner.
- **Retest requirements:** Repeat after proposal/localizer changes.
- **Waiver decision:** None.

### VG-06 - Branch Ablation

- **Class:** Mandatory
- **Threat / failure hypothesis:** Optional Ultrack/multi-hypothesis/scorer/division branches add complexity without official score gain.
- **Protected requirements:** R-06.
- **Protected acceptance criteria:** AC-05.
- **Protected decisions:** D-04, D-06.
- **Preconditions:** Candidate and solver branches stable.
- **Deterministic checks:**
  1. Compare base localizer route, multi-hypothesis branch, scorer branch, and division branch on locked folds.
     - Expected: enabled branch improves official CV or remains disabled.
  2. Inspect runtime/cost changes.
     - Expected: enabled branch remains within D-05 boundary.
- **Reviewer duty:** Challenge improvements that appear only in approximate metrics or single videos.
- **Evidence record:** Pending execution.
- **Status:** NOT_RUN
- **Status rationale:** Draft plan only.
- **Remediation owner:** Integration owner.
- **Retest requirements:** Repeat affected ablations after any branch change.
- **Waiver decision:** None.

### VG-07 - Graph And Submission Validity

- **Class:** Mandatory
- **Threat / failure hypothesis:** A high score claim comes from invalid or non-submittable graphs.
- **Protected requirements:** R-02, R-07.
- **Protected acceptance criteria:** AC-06.
- **Protected decisions:** D-02, D-06.
- **Preconditions:** Prediction CSV stable.
- **Deterministic checks:**
  1. Run existing submission validator and graph diagnostics.
     - Expected: no duplicate node IDs, duplicate edges, missing endpoints, merges, out-degree `>2`, invalid times, or missing datasets.
  2. Inspect row counts and schema.
     - Expected: exact required columns and stable ordering.
- **Reviewer duty:** Confirm validation covers every dataset in the run, not just one smoke clip.
- **Evidence record:** Pending execution.
- **Status:** NOT_RUN
- **Status rationale:** Draft plan only.
- **Remediation owner:** Integration owner.
- **Retest requirements:** Repeat after any CSV/graph writer change.
- **Waiver decision:** None.

### VG-08 - Compute And Side-Effect Boundary

- **Class:** Mandatory
- **Threat / failure hypothesis:** The route exceeds low-compute intent or uses unapproved paid/network/external effects.
- **Protected requirements:** R-08.
- **Protected acceptance criteria:** AC-07.
- **Protected decisions:** D-05.
- **Preconditions:** Runtime/cost log stable.
- **Deterministic checks:**
  1. Inspect compute log and side-effect ledger.
     - Expected: local/Kaggle-free only unless locked approval exists.
  2. Inspect commands.
     - Expected: no Kaggle push/submission, paid GPU, Runpod, secret read, or full 81GB run beyond approved boundary; optional Python 3.11/3.12 branch has provenance and execution-time approval for any install/network command.
- **Reviewer duty:** Treat missing compute evidence as blocking, not advisory.
- **Evidence record:** Pending execution.
- **Status:** NOT_RUN
- **Status rationale:** Draft plan only.
- **Remediation owner:** Integration owner.
- **Retest requirements:** Repeat after any environment/compute change.
- **Waiver decision:** None.

### VG-09 - Source Claim Discipline

- **Class:** Mandatory
- **Threat / failure hypothesis:** Plan or closeout overstates Kaggle best-solution/literature evidence not directly observed.
- **Protected requirements:** R-09.
- **Protected acceptance criteria:** AC-08.
- **Protected decisions:** None.
- **Preconditions:** Final report/source ledger stable.
- **Deterministic checks:**
  1. Inspect source ledger.
     - Expected: local evidence, live web evidence, and unavailable evidence are separated.
  2. Inspect Kaggle/literature claims.
     - Expected: Kaggle discussion/leaderboard unavailable status is explicit; no unsupported top-solution claim.
- **Reviewer duty:** Challenge any "best solution" statement without direct source or local artifact.
- **Evidence record:** Pending execution.
- **Status:** NOT_RUN
- **Status rationale:** Draft plan only.
- **Remediation owner:** Integration owner.
- **Retest requirements:** Repeat after report changes.
- **Waiver decision:** None.

### VG-10 - Reproducibility And Secret Exclusion

- **Class:** Mandatory
- **Threat / failure hypothesis:** Results cannot be reproduced because repository is uncommitted, or secrets are accidentally read/recorded.
- **Protected requirements:** R-10.
- **Protected acceptance criteria:** AC-09.
- **Protected decisions:** D-07.
- **Preconditions:** Manifest stable.
- **Deterministic checks:**
  1. Inspect manifest.
     - Expected: relevant source/config/data/output hashes and dependency versions recorded.
  2. Inspect manifest paths.
     - Expected: `.env` and `gurobi.lic` excluded; no secret values printed.
- **Reviewer duty:** Confirm output hashes match current evidence files.
- **Evidence record:** Pending execution.
- **Status:** NOT_RUN
- **Status rationale:** Draft plan only.
- **Remediation owner:** Integration owner.
- **Retest requirements:** Repeat after any source/config/output change.
- **Waiver decision:** None.


### VG-11 - Kaggle Scorer Parity

- **Class:** Mandatory
- **Threat / failure hypothesis:** Local iteration optimizes a metric that diverges from the Kaggle competition scorer.
- **Protected requirements:** R-02, R-11.
- **Protected acceptance criteria:** AC-02, AC-10.
- **Protected decisions:** D-01, D-08.
- **Preconditions:** N2 scorer parity outputs stable.
- **Deterministic checks:**
  1. Run scorer parity fixtures against the local scorer and official metric wrapper.
     - Expected: matching score components for GT-self, known prediction rows, sparse edge matching, adjusted node-count penalty, and division cases.
  2. Inspect iterative run outputs.
     - Expected: route selection and completion use official/parity scorer, not approximate-only metrics.
- **Reviewer duty:** Challenge any local score table that lacks official metric version identity or parity fixture evidence.
- **Evidence record:** Pending execution.
- **Status:** NOT_RUN
- **Status rationale:** Draft plan only.
- **Remediation owner:** Integration owner.
- **Retest requirements:** Repeat after scorer, metric wrapper, CSV parser, or official dependency changes.
- **Waiver decision:** None.

## Traceability Matrix

| Source ID | Planned Nodes | Completion Evidence | Gates | Coverage State |
| --- | --- | --- | --- | --- |
| R-01 / AC-01 / D-01 | N1, N2, N10, T1 | Official CV score report | VG-01, VG-04 | Covered |
| R-02 / AC-02 / D-08 | N2, N10, T1 | Official self-check and metric-source report | VG-01, VG-11 | Covered |
| R-03 / AC-03 / D-02 | N1, N10, T1 | Leakage report | VG-02 | Covered |
| R-04 / AC-04 / D-03/D-04 | N3, N4, N5, T1 | Candidate recall/localizer report | VG-03, VG-05 | Covered |
| R-05 / AC-03 / D-03 | N1, N3, N10, T1 | Offset provenance report | VG-03 | Covered |
| R-06 / AC-05 / D-04/D-06 | N6, N7, N8, N9, N10, T1 | Ablation reports | VG-06 | Covered |
| R-07 / AC-06 / D-02 | N1, N3, N4, N9, N10, T1 | Graph validation and duplicate-disable report | VG-02, VG-07 | Covered |
| R-08 / AC-07 / D-05 | N6, N8, N10, T1 | Compute and side-effect ledger | VG-08 | Covered |
| R-09 / AC-08 | N10, T1 | Source ledger | VG-09 | Covered |
| R-10 / AC-09 / D-07 | N0, N7, N10, T1 | Reproduction manifest | VG-10 | Covered but D-07 unresolved |
| R-11 / AC-10 / D-08 | N2, N10, T1 | Scorer parity report | VG-11 | Covered |

## DAG Validation

- Unique node IDs: PASS - N0 through N10 plus T1 appear once.
- Existing dependencies: PASS - all dependency references exist.
- Acyclic graph: PASS - topological order `N0 -> N1 -> N2 -> N3 -> N4 -> N5 -> N6/N7 -> N8 -> N9 -> N10 -> T1`.
- Ancestor-provided inputs: PASS - every node consumes initial inputs or ancestor outputs.
- Requirement/criterion/decision coverage: BLOCK - D-07 remains Proposed.
- Non-overlapping write ownership: PASS for planned serialized handoffs; N10 integration owner serializes shared scopes.
- External-effect approvals: PASS - D-05 locks local/Kaggle-free main path plus optional separate Python 3.11/3.12 branch; paid GPU, Kaggle submission, Runpod, secret reads, and unbounded full-ZIP jobs remain prohibited without later approval.
- Current-slice-only scope: PASS - G-02/G-03 excluded.

## Verification Attempt Model

T1 returns attempt reports; the integration owner records them in `CLOSEOUT.md`. On a block, record the temporal ownership handoff, make prior writers read-only for affected scopes, invalidate changed outputs and descendants, replay the topological suffix, and rerun affected gates. This does not create cyclic DAG edges.
