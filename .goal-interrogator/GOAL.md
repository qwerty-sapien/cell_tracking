# Goal G-01 - Leakage-Safe General Model Improvement

- Contract state: Prepared (inactive)
- Prepared from repository baseline: unborn `main`; no commit/tree. Relevant hashes are in `SPEC.md`; `.env` and `gurobi.lic` excluded.
- Integration owner: Serialized integration owner
- T1 verification coordinator: Read-only coordinator

## End State

The repository contains a low-compute, non-duplicate/general Biohub tracking route that either reaches `>=0.85` official local CV under locked leakage-safe and Kaggle-scorer-parity evidence, or stops with a decisive blocked report identifying the next bottleneck.

## Current Slice

- **G-01 result:** Improve and verify the general route while deleting or disabling duplicate-GEFF copy behavior from general-model execution and score evidence.
- **Included IDs:** R-01 through R-11; AC-01 through AC-10; D-01 through D-08 locked.
- **Transition value:** This slice retires the main uncertainty: whether the current alpha stack can reach the target with metric-aware localization, optional multi-hypothesis proposals, and low-compute edge/division scoring before paying for larger training.

## Approved Artifact References

These artifacts are prepared but inactive. Before activation, read the finalized versions:

1. `.goal-interrogator/SPEC.md` - Prepared (inactive).
2. `.goal-interrogator/DECISIONS.md` - Prepared (inactive); D-01 through D-08 locked.
3. `.goal-interrogator/ROADMAP.md` - Prepared (inactive); current slice G-01.
4. `.goal-interrogator/PLAN.md` - Prepared (inactive); DAG coverage passes.

If a referenced artifact is missing, stale, or inconsistent, stop before implementation.

## Context To Read First

1. `doc_cell_tracking/EVALUATION.md`
2. `doc_cell_tracking/ARCHITECTURE.md`
3. `reports/alpha_reassessment.md`
4. `reports/alpha_routed_full_44b6_0113de3b_offset_metrics.json`
5. `reports/duplicate_geff_submission_report.json`
6. `docs/alpha_routed_runbook.md`
7. `src/celltrack_alpha/pipeline.py`
8. `src/celltrack_alpha/proposals.py`
9. `src/celltrack_alpha/solver.py`
10. `src/celltrack_alpha/official_eval.py`
11. `external/kaggle-cell-tracking-competition/metrics.md`

Confirm repository baseline and unrelated changes before editing.

## Scope And Non-Goals

### In Scope

- Lock a non-duplicate CV protocol and leakage guard.
- Build fold-safe metric-aware localization/candidate improvements first.
- Add optional multi-hypothesis and learned scoring branches only through common alpha interfaces.
- Use official metrics, candidate recall gates, ablations, graph validation, compute logs, and reproduction manifests.

### Non-Goals

- Claim success from duplicate-GEFF copy score, or leave duplicate-copy routing reachable from the general model path.
- Run Kaggle push/submission.
- Run paid GPU, Kaggle submission, Runpod, or full external deep training without separate locked approval. A separate Python 3.11/3.12 compatibility branch is allowed by D-05, but concrete install/network commands still require execution-time approval and provenance.
- Treat approximate metrics, one smoke clip, or public leaderboard guesswork as sufficient final evidence.

## Protected Behavior And Decisions

- **R-03 / D-02 locked:** Duplicate-copy routing must be deleted or disabled from the general model path and cannot be score evidence.
- **R-05 / D-03 locked:** Held-out/test-derived offsets are forbidden; calibration must be fold-safe and provenance-recorded.
- **R-08 / D-05 locked:** Main path stays local/Kaggle-free; a separate Python 3.11/3.12 branch may test optional compatibility; paid compute, Kaggle submission, Runpod, and secret reads remain unapproved.
- **R-10 / D-07 locked:** Reproducibility requires a per-run manifest of file/config/data/output/dependency/command identities and secret exclusion; no Git commit is required.
- **R-11 / D-08 locked:** Local iterative scorer must match Kaggle competition scoring semantics.

Do not weaken an acceptance criterion, fixture, expected output, split policy, or leakage gate to obtain a pass.

## DAG Allocation And Ownership

Follow `.goal-interrogator/PLAN.md` as the canonical graph after decisions are locked.

| Node | Outcome | Owner | Depends On | Write Ownership | Required Evidence / Gates |
| --- | --- | --- | --- | --- | --- |
| N0 | Baseline and run ledger | Integration owner | None | `reports/general_model_*`, future `CLOSEOUT.md` | AC-09; VG-10 |
| N1 | Non-duplicate split and leakage policy | Integration owner | N0 | `reports/general_model_split_manifest.json`, leakage policy report | AC-01, AC-03; VG-02, VG-03, VG-04 |
| N2 | Official metric self-check and scorer parity | Evaluation owner | N1 | scorer/eval source, parity fixtures, `reports/*official_eval*.json`, `reports/*scorer_parity*.json` | AC-02, AC-10; VG-01, VG-11 |
| N3 | Fold-safe localizer contract | Localizer owner | N1, N2 | `src/celltrack_alpha/localization.py`, scoped alpha edits, tests | AC-03, AC-04; VG-03, VG-05 |
| N4 | Candidate localization/fusion improvements | Proposal owner | N3 | proposal/routing/localization source scopes | AC-04, AC-06; VG-05, VG-07 |
| N5 | Candidate recall/overcount report | Evaluation owner | N4 | candidate diagnostic scripts/reports | AC-04; VG-05 |
| N6 | Optional multi-hypothesis branch | Branch owner | N3, N5 | Ultrack/reduced branch source and reports | AC-05, AC-07; VG-06, VG-08 |
| N7 | Candidate graph training table | Scoring-data owner | N5 | `src/celltrack_alpha/scoring.py`, `scripts/build_edge_training_table.py`, `data/derived/edge_scorer/` | AC-05, AC-09; VG-06, VG-10 |
| N8 | Low-compute edge scorer | Scoring owner | N7 | scoring source/scripts/derived model | AC-05, AC-07; VG-06, VG-08 |
| N9 | Solver/scorer/division integration | Solver owner | N8; optional N6 | solver/types/routing/scoring scopes | AC-05, AC-06; VG-06, VG-07 |
| N10 | Final route, CV, ablation, package dry run | Integration owner | N2, N5, optional N6, N9 | shared scripts, reports, predictions, closeout | AC-01 through AC-10; VG-01 through VG-11 |
| T1 | Verification coordination | Read-only coordinator | N10 | None | All gates |

Use write-capable subagents only for disjoint ownership. Route shared alpha interfaces, route configs, solver, generated outputs, and final reports through the serialized integration owner.

## Mandatory Completion Evidence

| AC ID | Required Check / Observation | Expected Result | Evidence To Record | Gate |
| --- | --- | --- | --- | --- |
| AC-01 | Official CV on locked non-duplicate split | Score `>=0.85` or explicit blocked stop | Metrics JSON, split manifest, config hash | VG-01, VG-02, VG-04 |
| AC-02 | Official scorer self-check | GT self edge Jaccard `1.0`; official metric source | Self-check JSON | VG-01 |
| AC-03 | Leakage inspection | No duplicate route or held-out offset leakage | Leakage report | VG-02, VG-03 |
| AC-04 | Candidate/localizer gate | Locked recall/overcount threshold met or blocked | Candidate report | VG-05 |
| AC-05 | Branch ablation | Enabled branches improve official CV; others disabled | Ablation table | VG-06 |
| AC-06 | Graph/submission validation | No invalid endpoints/degrees/duplicates/missing datasets | Validator report | VG-07 |
| AC-07 | Compute/side-effect log | Low-compute compliant; no unapproved effects | Compute ledger | VG-08 |
| AC-08 | Source ledger | Local/web/unavailable claims separated | Source report | VG-09 |
| AC-09 | Reproduction manifest | Relevant identities recorded; secrets excluded | Manifest | VG-10 |
| AC-10 | Scorer parity | Local scorer matches official Kaggle metric behavior on parity fixtures | Scorer parity report | VG-11 |

## Adversarial Gates

T1 coordinates the selected gate definitions in PLAN and returns `PASS`, `BLOCK`, `ADVISORY`, or `NOT_RUN` with evidence. Mandatory `NOT_RUN` blocks unless a linked locked waiver explicitly accepts residual risk. T1 never remediates.

- **VG-01 - Official Metric Fidelity:** protects R-01/R-02; mandatory.
- **VG-02 - Duplicate Leakage:** protects R-03; mandatory.
- **VG-03 - Offset Leakage:** protects R-05; mandatory.
- **VG-04 - Split Integrity:** protects R-01/R-10; mandatory.
- **VG-05 - Candidate Localizer Gate:** protects R-04; mandatory.
- **VG-06 - Branch Ablation:** protects R-06; mandatory.
- **VG-07 - Graph And Submission Validity:** protects R-02/R-07; mandatory.
- **VG-08 - Compute And Side-Effect Boundary:** protects R-08; mandatory.
- **VG-09 - Source Claim Discipline:** protects R-09; mandatory.
- **VG-10 - Reproducibility And Secret Exclusion:** protects R-10; mandatory.
- **VG-11 - Kaggle Scorer Parity:** protects R-02/R-11; mandatory.

## Iteration Policy

1. Execute only ready DAG nodes within their ownership.
2. Run focused node evidence after each material change.
3. Stabilize integration-owner outputs before T1 evaluates dependent gates.
4. When a gate blocks, preserve the attempt, transfer affected scopes exclusively to the integration owner, and make prior writers read-only.
5. Invalidate changed nodes plus transitively affected descendant outputs/evidence, replay the DAG suffix topologically, then rerun affected gates.
6. Record attempts in `CLOSEOUT.md`; do not add cyclic DAG edges.
7. Stop when mandatory evidence passes, a blocked condition occurs, or an approved iteration/resource limit is reached.

## Side-Effect Approvals

| Effect | Allowed Boundary | Locked Decision | Authorized Owner | Evidence |
| --- | --- | --- | --- | --- |
| Kaggle push/submission | None approved | None | None | Not allowed |
| Paid GPU/Runpod | None approved | None | None | Not allowed |
| Separate Python 3.11/3.12 compatibility environment | Optional branch only; no paid compute; provenance required; concrete install/network commands need execution-time approval | D-05 | Branch owner / integration owner | Environment manifest and command evidence |
| `.env` or `gurobi.lic` reads | None approved | None | None | Not allowed |
| Full 81GB ZIP exhaustive run | None approved unless later scoped by runtime boundary | None | None | Not allowed as completion evidence |

Activation approval does not expand this table.

## Explicit Blocked Stops

Stop and report `Blocked` without improvising when:

- The prepared artifact bundle is missing, stale, or inconsistent before execution.
- A required artifact or repository premise is missing or stale.
- The official scorer cannot run, scorer parity fails, and no locked waiver exists.
- Duplicate-copy route remains reachable from the general model path, or held-out-label-derived offset appears in general score evidence.
- Candidate recall/overcount gate fails and no in-scope localizer branch remains.
- A branch requires paid/Kaggle/Runpod/secret effects without locked approval, or package/environment commands exceed D-05 plus execution-time approval.
- Graph validation fails after bounded remediation.
- Runtime/cost exceeds the locked low-compute boundary.
- Reproduction manifest cannot exclude secrets or bind relevant evidence.

## Progress And Closeout

Create `.goal-interrogator/CLOSEOUT.md` only after explicit activation. Record approval-bound artifact identities, activation method/time, actual changes, node progress, verification attempts, command lines, outputs, deviations, side effects, remaining risk, and final `Completed` or `Blocked` state.

## Completion Statement

Claim completion only when every mandatory evidence item and gate satisfies this contract. Otherwise report the explicit blocked condition and evidence needed to resume.
