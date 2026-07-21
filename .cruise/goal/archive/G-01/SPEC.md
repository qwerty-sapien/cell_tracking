# Specification

- Planning state: Draft
- Mode/current slice: New / G-01
- Last reconciled: 2026-07-15
- Repository identity: `c3b9b703b049fc25f10e2b42bb986bb94f137a72` / tree `6545134fe081bf5a1e5b8e7b44b07ea8b0b66935`
- Origin: `.cruise/goal/ORIGIN.yaml` (`cf050ce8f0385c4df44dab480f1f2cec4018aa4dcab54b4a189e16be56ff6aa3`)

## Outcome

G-01 will verify the current Biohub cell-tracking model against the Kaggle site, then iterate with low-compute changes until the authoritative Kaggle site score reaches at least `0.92` or the approved bounded submission budget is exhausted.

The beneficiary is the model-development loop: it must stop treating local locked-CV scores as site truth, package the correct `celltrack_alpha` model rather than the older baseline, and produce evidence strong enough to guide the next competition iteration.

## Evidence

### Observed

- `.cruise/runs/prepare-g01/T-00_repo_submission_probe.report.yaml` reports no repository evidence that the current model was submitted to Kaggle.
- The current local locked-12 audit score is `0.8713516075529171`, with edge Jaccard `0.9119496855345912` and node recall `0.9497274492234169`, from `reports/general_model_cv_audit_fold0_auto_n61_1d0d_minx32_offsetonly_12ds_12clips.json`.
- That audit is local train-label evaluation, not a Kaggle site score.
- `reports/general_model_scorer_parity_after_n61_1d0d_minx32_offsetonly.json` records official-wrapper scorer parity, which supports local metric fidelity but does not prove remote scoring.
- `scripts/build_kaggle_kernel.py`, `scripts/push_kaggle_kernel.sh`, and `kaggle_kernel/rule_based_submission.py` currently package the older `celltrack_baseline` path, not the current `celltrack_alpha` implementation.
- `scripts/run_alpha_routed_zip.py` is the existing local alpha execution path and can generate split-specific CSVs from the Kaggle ZIP.

### Inferred

- A Kaggle verification submission must first package or submit the current alpha route, not the existing baseline kernel.
- The hidden/public Kaggle score may differ from local locked-CV due to split differences, sparse labels, test distribution, packaging/runtime behavior, or submission-format defects.

### Conflicts

- Previous local completion target was `>=0.85`; the new user endpoint is a Kaggle site score of `>=0.92`.
- The repository has Kaggle automation, but that automation is not aligned to the current alpha model.

### Unknown

- Remote Kaggle submission history, submission quota, and site score are unknown until authenticated Kaggle inspection/submission runs.
- The first site score may reveal that local CV is optimistic or that packaging/runtime needs repair.

## Scope

### In Scope

- Preserve origin, permissions, and evidence under `.cruise/goal/` and `.cruise/runs/`.
- Package or adapt the Kaggle submission path so it runs the current `celltrack_alpha` route that produced the local N61 audit, not the older baseline.
- Generate a complete `submission.csv` for the test split and validate schema, dataset coverage, graph constraints, and local reproducibility.
- Submit to the Biohub Kaggle competition through the approved logged-in Kaggle/Chrome path or an approved Kaggle CLI/API path if available without exposing secret values.
- Capture the authoritative Kaggle submission ID, timestamp, status, and site score.
- If score is below `0.92`, perform bounded low-compute iteration using existing heuristics, route gating, scorer calibration, packaging fixes, and lightweight learned scorers already present in the repository.
- Run local scorer-parity, synthetic, unit, schema, leakage, and reproducibility checks before each scored submission.

### Non-goals

- No paid compute.
- No training large models from scratch.
- No unrelated Kaggle account changes, profile changes, purchases, or submissions to other competitions.
- No claims that local CV equals Kaggle site score.
- No destructive cleanup of user changes outside approved generated outputs.

## Requirements

### R-01 - Truthful Score Provenance

The workflow must distinguish local CV, local parity, and Kaggle site scores in every report and final claim.

- Source: Origin prompt, D-03, worker report.
- Acceptance: AC-01, AC-05.

### R-02 - Correct Alpha Submission Artifact

The Kaggle artifact or uploaded CSV must run or represent the current `celltrack_alpha` route, including the N61 promoted configuration, not the older `celltrack_baseline` kernel.

- Source: Worker report conflict.
- Acceptance: AC-02, AC-03.

### R-03 - Valid Full Test Submission

Every Kaggle submission must contain all required test datasets and valid node/edge rows in the required CSV format.

- Source: Competition format and local validation requirements.
- Acceptance: AC-03.

### R-04 - Site-Score Endpoint

G-01 completes only when the Kaggle site score is at least `0.92`. If the approved budget is exhausted below `0.92`, the goal is Blocked, not complete.

- Source: Origin prompt, D-03, D-04.
- Acceptance: AC-05.

### R-05 - Low-Compute Iteration

Iteration must prioritize low-compute model/algorithm hybridization: route gating, proposal fusion, graph solver costs, learned lightweight edge/candidate scorers, calibration, and packaging/runtime fixes. It must not train large models from scratch.

- Source: Prior user direction and D-04.
- Acceptance: AC-04, AC-06.

### R-06 - Permission Compliance

Kaggle, Chrome, network, submissions, file deletion/replacement, and worker data transfer must remain inside D-01, D-02, and D-04.

- Source: D-01, D-02, D-04.
- Acceptance: AC-06.

### R-07 - Reproducibility And Leakage Guard

Each submitted artifact must be reproducible from recorded source/config/report identities and must avoid held-out/test-label leakage or duplicate-GEFF shortcuts.

- Source: Existing reproducibility/scorer-parity work and competition integrity.
- Acceptance: AC-04, AC-06.

## Acceptance Criteria

### AC-01 - Initial Truth Check

The closeout states that no Kaggle submission had been verified before G-01 unless remote Kaggle evidence proves otherwise, and identifies `0.8713516075529171` as local locked-CV only.

### AC-02 - Package Parity

A package or direct submission-generation artifact is inspected and recorded as containing the current `celltrack_alpha` code/config path. The older `celltrack_baseline` kernel path is not submitted as the current model.

### AC-03 - Submission Validity

The generated `submission.csv` passes local schema, dataset coverage, row identity, node/edge reference, graph out-degree, merge, and reproducibility checks before upload.

### AC-04 - Local Evidence Before Site Submission

Before each scored Kaggle submission, the run records current local validation evidence: scorer parity still passes, focused unit/synthetic tests pass, no leakage guard fails, and local CV/regression checks do not show an obvious degradation unless the submission is a packaging-only repair.

### AC-05 - Authoritative Site Score

The run records Kaggle submission ID, timestamp, status, and site score. Completion requires site score `>=0.92`; otherwise, after the D-04 cap, the run is Blocked with exact score evidence.

### AC-06 - Permission And Budget Evidence

The closeout records all Kaggle submissions used, browser/API actions, generated/deleted/replaced files, worker launches, command evidence, cleanup, and confirmation that no paid compute or out-of-envelope action occurred.

## Protected Behavior

- Preserve the N61 local audit evidence and do not relabel it as a Kaggle score.
- Preserve official scorer parity and graph validation behavior.
- Preserve existing user/unrelated worktree changes; do not revert or clean unrelated files.
- Keep generated artifacts isolated under clearly named report/submission paths where feasible.
- Do not exceed three scored Kaggle submissions without a new explicit decision.

## Risks

- Public/private leaderboard score may be materially lower than local CV.
- The current Kaggle kernel packaging may need nontrivial adaptation from baseline to alpha.
- Kaggle runtime or account/session state may block submission.
- Daily submission quota may be lower than the D-04 cap.
- Lightweight iteration may not reach `0.92` without larger modeling or more data.

## Rollback And Cleanup

- Repository changes remain in the working tree; unrelated user changes are preserved.
- Generated local `submission.csv` artifacts may be replaced or deleted only when owned by G-01.
- Kaggle submissions cannot necessarily be undone; closeout records every remote mutation and score.
- If Chrome/API state changes outside the envelope, stop and report before proceeding.
