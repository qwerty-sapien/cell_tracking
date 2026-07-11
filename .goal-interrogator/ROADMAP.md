# Roadmap: General Low-Compute Cell Tracking Program

- Planning state: Draft
- Current slice: G-01
- Last reconciled: 2026-07-11

## Program End State

A leakage-safe, low-compute cell tracking system that removes or disables duplicate-GEFF copy behavior from general-model evidence and uses a general candidate-localizer plus graph-solver/scorer path that can be validated with official/Kaggle-parity local metrics and packaged for Kaggle submission after separate approval.

## Slice Index

| ID | Strategically Complete Result | Requirement / Criterion Coverage | Depends On | Completion Proof | Status |
| --- | --- | --- | --- | --- | --- |
| G-01 | Non-duplicate/general route reaches or honestly blocks against `>=0.85` official local CV under low-compute, scorer-parity, and leakage controls. | R-01 through R-11; AC-01 through AC-10 | None | Official CV report, scorer-parity report, leakage report, candidate/localizer gate, ablation table, graph validation, compute log, reproduction manifest | Current |
| G-02 | Optional external/deep branch: Python-compatible Ultrack/official sparse U-Net/Trackastra-style branch or paid GPU pilot, only if G-01 evidence justifies it. | R-06, R-08 plus future criteria | G-01 blocked or plateau evidence; locked compute/env decision | Separate environment/budget approval and branch-specific CV improvement | Deferred |
| G-03 | Kaggle submission/leaderboard operation with selected route and production evidence. | R-02, R-03, R-08, R-10 plus future criteria | G-01 success and locked Kaggle effect approval | Valid submission artifact, Kaggle log/output, optional leaderboard record | Deferred |

## G-01 - Leakage-Safe General Model Improvement

- **Result:** A general, non-duplicate route either reaches `>=0.85` official local CV or produces a decisive blocked report showing the next bottleneck.
- **Included requirements:** R-01, R-02, R-03, R-04, R-05, R-06, R-07, R-08, R-09, R-10, R-11.
- **Acceptance criteria:** AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09, AC-10.
- **Locked decisions:** D-01, D-02, D-03, D-04, D-05, D-06, and D-08. D-07 remains Proposed and must be resolved before activation.
- **Dependencies:** Existing official metric harness, zipped Kaggle data, alpha pipeline, route/report evidence.
- **Explicit exclusions:** Paid GPU, Kaggle submission/push, new Python environment, full external deep training, and duplicate route score claims.
- **Why this is strategically complete:** It directly tests the main uncertainty: whether a low-compute, general path can reach the target without duplicate-copy leakage or large training-from-scratch.
- **Completion and transition evidence:** Official CV score, scorer-parity report, candidate recall/overcount report, ablation table, graph validation, compute log, and reproduction manifest.
- **Next-slice trigger:** G-01 reaches `>=0.85` and user approves Kaggle operation, or G-01 blocks with evidence that requires a new environment/paid compute/deep branch.

## G-02 - Optional External Or Deep Branch

- **Result:** Evaluate a separately approved Ultrack, official sparse U-Net, Trackastra-style, or paid GPU pilot branch.
- **Included requirements:** Future refined requirements from G-01 evidence.
- **Acceptance criteria:** Future criteria; at minimum official CV improvement and budget/environment evidence.
- **Locked decisions:** Requires new locked environment and/or budget approvals.
- **Dependencies:** G-01 evidence showing local deterministic/tabular path is insufficient or branch is clearly promising.
- **Explicit exclusions:** Not part of current GOAL.
- **Why this is strategically complete:** Separates risky dependency/compute work from low-compute alpha work.
- **Completion and transition evidence:** Branch-specific CV, runtime/cost, reproducibility, and integration decision.
- **Next-slice trigger:** Approved need for advanced branch.

## G-03 - Kaggle Submission Operation

- **Result:** Package and submit the selected route to Kaggle or produce a final submission artifact under an approved effect boundary.
- **Included requirements:** Future operational requirements.
- **Acceptance criteria:** Valid Kaggle output/log/leaderboard evidence if submission is authorized.
- **Locked decisions:** Requires separate Kaggle push/submission approval and credential boundary.
- **Dependencies:** G-01 success or user-approved external score test.
- **Explicit exclusions:** Not part of current GOAL.
- **Why this is strategically complete:** Separates implementation proof from external competition operation.
- **Completion and transition evidence:** Submission artifact, logs, score if available, and rollback/no-secret evidence.
- **Next-slice trigger:** User approves Kaggle operation.

## Deferred Work

| Item | Intended Slice | Reason Deferred | Revisit Trigger |
| --- | --- | --- | --- |
| Paid GPU or Runpod pilot | G-02 | D-05 authorizes local/Kaggle-free main work and an optional separate Python environment only; paid compute remains unapproved. | G-01 blocks below target and branch evidence justifies cost. |
| Full official sparse U-Net or Trackastra-style dense training | G-02 | Requires environment/dependency/compute decisions. | Candidate/localizer path is insufficient or user explicitly approves. |
| Kaggle push/submission | G-03 | External effect and credentials; not authorized by planning. | G-01 success plus user approval. |
| Duplicate route exploitation | G-03 or separate | Already solved for current ZIP but not general. | User asks for leaderboard-only operation. |

## Coverage Audit

- Requirements assigned to more than one active slice: None.
- Requirements assigned to no slice: None for current SPEC; G-02/G-03 will receive future refined requirements.
- Current GOAL slice: G-01.
