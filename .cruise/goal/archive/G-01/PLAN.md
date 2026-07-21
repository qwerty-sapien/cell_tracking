# Plan

- Planning state: Draft
- Mode/current slice: New / G-01
- Last reconciled: 2026-07-15
- Repository identity: `c3b9b703b049fc25f10e2b42bb986bb94f137a72` / tree `6545134fe081bf5a1e5b8e7b44b07ea8b0b66935`
- Origin: `.cruise/goal/ORIGIN.yaml` (`cf050ce8f0385c4df44dab480f1f2cec4018aa4dcab54b4a189e16be56ff6aa3`)
- Alignment: `.cruise/goal/ALIGNMENT.md`

## Required Completion Evidence

| AC | Final check | Expected result | Evidence identity/location | Producing nodes | Gate |
| --- | --- | --- | --- | --- | --- |
| AC-01 | Review closeout and remote evidence | Prior `0.8713516075529171` labeled local-only; remote Kaggle history/site score recorded if found | `.cruise/goal/CLOSEOUT.md` | T-04, T-07 | E-01 |
| AC-02 | Inspect Kaggle package/submission path | Artifact uses `celltrack_alpha` current route/config, not baseline-only kernel | package manifest/report under `reports/kaggle_g01/` | T-02 | E-02 |
| AC-03 | Validate generated `submission.csv` | All required datasets and rows valid; graph constraints pass | validation report under `reports/kaggle_g01/` | T-03 | E-03 |
| AC-04 | Run local pre-submit checks | Scorer parity, focused tests, leakage checks, and relevant CV/regression evidence pass | check reports under `reports/kaggle_g01/` | T-03, T-06 | E-04 |
| AC-05 | Read Kaggle site result | Submission ID/status/score captured; completion only if score `>=0.92` | Kaggle evidence report/screenshots/API output under `reports/kaggle_g01/` | T-04, T-07 | E-05 |
| AC-06 | Audit permissions/effects | <=5 valid scored submissions in this phase; no paid compute; all browser/API/file actions recorded | `.cruise/goal/CLOSEOUT.md` | T-04, T-07, T-01 | E-06 |
| AC-07 | Calibrate local scorer against Kaggle | Every frozen diagnostic holdout has absolute local-versus-Kaggle error <=0.02; calibration and holdout definitions were frozen before scores | scorer calibration report under `reports/kaggle_g01/` | T-05, T-06, T-07 | E-07 |

## Baseline Inputs

| Input | Identity | Source | Consumers |
| --- | --- | --- | --- |
| I-01 | SHA `cf050ce8f0385c4df44dab480f1f2cec4018aa4dcab54b4a189e16be56ff6aa3` | `.cruise/goal/ORIGIN.yaml` | All nodes |
| I-02 | Commit `c3b9b703b049fc25f10e2b42bb986bb94f137a72`, tree `6545134fe081bf5a1e5b8e7b44b07ea8b0b66935`, dirty tracked count 28, untracked count 980 | Repository baseline | All nodes |
| I-03 | Local score `0.8713516075529171` | `reports/general_model_cv_audit_fold0_auto_n61_1d0d_minx32_offsetonly_12ds_12clips.json` | T-02, T-03, T-06, T-01 |
| I-04 | Scorer parity report | `reports/general_model_scorer_parity_after_n61_1d0d_minx32_offsetonly.json` | T-03, T-06, T-01 |
| I-05 | Worker path report | `.cruise/runs/prepare-g01/T-00_repo_submission_probe.report.yaml` | T-02, T-04, T-01 |
| I-06 | Existing alpha source and tests | `src/celltrack_alpha/`, `scripts/run_alpha_routed_zip.py`, `tests/` | T-02, T-03, T-06 |
| I-07 | Existing Kaggle baseline package path | `scripts/build_kaggle_kernel.py`, `scripts/push_kaggle_kernel.sh`, `kaggle_kernel/` | T-02, T-04 |

## Execution DAG

Topological order: T-02 -> T-03 -> T-04 -> T-05 -> T-06 -> T-07 -> T-01.

### T-02 - Package Current Alpha Submission Path

- **State:** Not started
- **Owner:** implementation worker
- **Depends on:** None
- **Inputs:** I-01, I-02, I-05, I-06, I-07
- **Outputs:** Alpha-compatible Kaggle package or direct submission-generation path; package parity manifest under `reports/kaggle_g01/`; changed source/scripts if required.
- **Satisfies:** R-02, AC-02
- **Preserves:** R-01, R-06, R-07
- **Verification:** E-02
- **Decisions and permissions:** D-01, D-02, D-03, D-04
- **Write ownership:** `scripts/build_kaggle_kernel.py`, `scripts/push_kaggle_kernel.sh`, `scripts/run_alpha_routed_zip.py`, `kaggle_kernel/**`, `reports/kaggle_g01/**`, any new alpha package helper under `scripts/` or `src/celltrack_alpha/`
- **Shared-resource interaction:** Kaggle package directory; no remote Kaggle action in this node.
- **Micro-plan constraint:** Prefer adapting existing package/build scripts and copying the already-promoted alpha implementation/config. Do not train a new model or replace the alpha route with baseline code.
- **Focused evidence:** Package manifest proves `celltrack_alpha` files/config are included and `celltrack_baseline` is not the submitted model.
- **Attempt/resource limit:** Two package attempts; stop if Kaggle runtime cannot support required dependencies without paid compute or large downloads.
- **Evidence return:** Worker attempt report plus package parity manifest.
- **Failure and rollback:** Leave changed files visible; stop if packaging would require broad architecture or secret access.
- **Parallel-safe with:** None; owns shared package path.

### T-03 - Generate And Locally Validate Full Test Submission

- **State:** Not started
- **Owner:** implementation worker
- **Depends on:** T-02
- **Inputs:** I-01, I-02, I-03, I-04, T-02 outputs
- **Outputs:** Candidate `submission.csv`; local validation report; command log; updated reproducibility manifest.
- **Satisfies:** R-03, R-07, AC-03, AC-04
- **Preserves:** R-01, R-02, R-06
- **Verification:** E-03, E-04
- **Decisions and permissions:** D-02, D-03, D-04
- **Write ownership:** `reports/kaggle_g01/**`, generated `submission.csv` paths, local temp/cache outputs created by the command.
- **Shared-resource interaction:** Reads Kaggle ZIP/test data already in workspace; no remote Kaggle mutation.
- **Micro-plan constraint:** Use the alpha path to produce `--split test` CSV, then validate exact Kaggle schema and graph constraints before upload.
- **Focused evidence:** Validation report lists datasets, row counts, node/edge constraints, and reproducible command.
- **Attempt/resource limit:** Two generation attempts; stop on missing data or runtime exceeding practical local budget.
- **Evidence return:** Worker attempt report and validation artifacts.
- **Failure and rollback:** Replace only generated G-01 candidate CSV/report files.
- **Parallel-safe with:** None; depends on package path and owns generated CSV.

### T-04 - Submit Baseline Alpha And Capture Site Score

- **State:** Not started
- **Owner:** Kaggle submission worker
- **Depends on:** T-03
- **Inputs:** I-01, I-02, T-02 outputs, T-03 candidate CSV and validation report
- **Outputs:** Kaggle submission evidence report containing method, submission ID, timestamp, status, and returned site score; screenshots/API logs as appropriate.
- **Satisfies:** R-01, R-04, R-06, AC-01, AC-05, AC-06
- **Preserves:** R-02, R-03, R-07
- **Verification:** E-01, E-05, E-06
- **Decisions and permissions:** D-02, D-03, D-04
- **Write ownership:** `reports/kaggle_g01/**`; browser/API session state only within approved Kaggle scope.
- **Shared-resource interaction:** Kaggle account, competition submission quota, logged-in Chrome or Kaggle CLI/API.
- **Micro-plan constraint:** Prefer the narrowest reliable submission path. If using Chrome, stay on Kaggle competition/submission pages. If using CLI/API, do not print secret values. Capture the score from the site/API.
- **Focused evidence:** One scored Kaggle submission or a precise block reason such as auth/quota/rules/runtime failure.
- **Attempt/resource limit:** Consumes at most one scored submission; one packaging-only retry is allowed only if the first upload fails before a scored submission is created.
- **Evidence return:** Worker attempt report, score evidence, cleanup/state notes.
- **Failure and rollback:** Kaggle submissions may not be reversible; record remote state and stop if unexpected account prompts or unrelated pages appear.
- **Parallel-safe with:** None; external account mutation must serialize.

### T-05 - Score Gap Diagnosis

- **State:** Not started
- **Owner:** analysis worker
- **Depends on:** T-04
- **Inputs:** I-01, I-02, I-03, I-04, T-04 site-score evidence
- **Outputs:** Gap diagnosis and frozen scorer-calibration design under `reports/kaggle_g01/`, including pre-score local predictions for four single-axis diagnostic variants and a calibration/holdout split.
- **Satisfies:** R-04, R-05
- **Preserves:** R-01, R-06, R-07
- **Verification:** E-04, E-05
- **Decisions and permissions:** D-03, D-04
- **Write ownership:** `reports/kaggle_g01/**`; read-only over source unless handed off to T-06.
- **Shared-resource interaction:** None.
- **Micro-plan constraint:** Diagnose from site score, local CV by route/dataset, scorer parity, validation logs, and package evidence. Do not propose large training-from-scratch.
- **Focused evidence:** Report records the baseline residual, freezes each diagnostic transformation and artifact identity before submission, and recommends calibrate, stop complete, improve, or block.
- **Attempt/resource limit:** One diagnosis pass per site score.
- **Evidence return:** Worker attempt report and diagnosis report.
- **Failure and rollback:** Stop if site score unavailable or evidence conflicts.
- **Parallel-safe with:** None; depends on score.

### T-06 - Bounded Low-Compute Improvement

- **State:** Not started
- **Owner:** implementation worker
- **Depends on:** T-05
- **Inputs:** I-01, I-02, I-03, I-04, T-05 diagnosis
- **Outputs:** Up to four bounded diagnostic candidates and local scorer revisions, with refreshed validation, scorer tests, local CV/regression evidence, and frozen pre-score predictions.
- **Satisfies:** R-05, R-07, AC-04
- **Preserves:** R-01, R-02, R-03, R-06
- **Verification:** E-03, E-04
- **Decisions and permissions:** D-01, D-02, D-03, D-04
- **Write ownership:** `src/celltrack_alpha/**`, `scripts/**` for alpha model tooling, `tests/**` for focused tests, `reports/kaggle_g01/**`, generated candidate CSVs.
- **Shared-resource interaction:** Local reports/test caches; no Kaggle mutation in this node.
- **Micro-plan constraint:** Use only low-compute changes. For scorer calibration, each CSV must isolate one predeclared axis (division edges, long association edges, node density/confidence, or association pruning), and all definitions must be frozen before their Kaggle scores are observed. No large model training from scratch.
- **Focused evidence:** Every diagnostic candidate passes schema/graph checks and has a local score, baseline delta, transformation manifest, and SHA-256 recorded before T-07 submits it.
- **Attempt/resource limit:** At most four diagnostic candidates after the baseline; no minimum local-score threshold applies to these scorer-identification submissions.
- **Evidence return:** Worker attempt report, diff summary, local validation artifacts.
- **Failure and rollback:** Preserve failed attempts; transfer affected ownership to integration owner for remediation or block.
- **Parallel-safe with:** None; owns shared source and generated outputs.

### T-07 - Resubmit Improved Candidate And Capture Site Score

- **State:** Not started
- **Owner:** Kaggle submission worker
- **Depends on:** T-06
- **Inputs:** T-06 candidate package/CSV and local evidence
- **Outputs:** Updated Kaggle submission evidence; cumulative submission budget ledger.
- **Satisfies:** R-01, R-04, R-06, AC-05, AC-06
- **Preserves:** R-02, R-03, R-07
- **Verification:** E-05, E-06
- **Decisions and permissions:** D-02, D-03, D-04
- **Write ownership:** `reports/kaggle_g01/**`; Kaggle account interaction within D-02/D-04.
- **Shared-resource interaction:** Kaggle account and submission quota.
- **Micro-plan constraint:** Submit diagnostic candidates only in their frozen order after validation, regardless of expected leaderboard improvement. After each score, compare against the pre-recorded local prediction. Fit scorer changes only on calibration pairs; do not change holdout definitions after observing Kaggle results.
- **Focused evidence:** Submission ID/status/score, artifact hash, local prediction, absolute residual, baseline delta residual, calibration/holdout role, and budget ledger.
- **Attempt/resource limit:** Up to four scored diagnostic submissions, for a maximum of five valid scored G-01 submissions including T-04.
- **Evidence return:** Worker attempt report and updated score evidence.
- **Failure and rollback:** Record remote state; stop on auth/quota/unexpected account prompts.
- **Parallel-safe with:** None; external account mutation must serialize.

### T-01 - Final Read-Only Verification

- **State:** Not started
- **Owner:** verification worker
- **Depends on:** T-02, T-03, T-04, T-05, T-06, T-07
- **Inputs:** I-01, all node reports, final diff, final local validation, final Kaggle score evidence, `.cruise/goal/CLOSEOUT.md`
- **Outputs:** Read-only verification report with PASS/BLOCK/NOT_RUN for E-01 through E-06.
- **Satisfies:** R-01, R-02, R-03, R-04, R-05, R-06, R-07, AC-01, AC-02, AC-03, AC-04, AC-05, AC-06
- **Preserves:** All protected behavior in ALIGNMENT.
- **Verification:** E-01, E-02, E-03, E-04, E-05, E-06
- **Decisions and permissions:** D-01, D-02, D-03, D-04
- **Write ownership:** None. T-01 returns an immutable report; it does not edit source, fixtures, submissions, or closeout.
- **Shared-resource interaction:** Read-only inspection of local evidence; no Kaggle mutation.
- **Micro-plan constraint:** Validate ORIGIN hash, decisions, package parity, local evidence freshness, Kaggle score source, permission budget, and closeout consistency.
- **Focused evidence:** PASS only if all mandatory gates pass and site score is `>=0.92`; otherwise BLOCK or NOT_RUN with exact remediation.
- **Attempt/resource limit:** One final verification pass plus one retest after integration remediation.
- **Evidence return:** Immutable verification report to coordinator.
- **Failure and rollback:** Integration owner remediates or closes Blocked.
- **Parallel-safe with:** None; final verifier runs after all producers.

## Gates

### E-01 - Score Provenance

- **Class:** Mandatory
- **Threat:** Local CV is misrepresented as Kaggle site score, or prior submission status is invented.
- **Protects:** R-01, AC-01
- **Preconditions:** T-04/T-07 score evidence or block reason.
- **Checks:** Review closeout, local audit references, and Kaggle evidence. Expected: local `0.8713516075529171` remains local-only; any site score has submission ID/timestamp/source.
- **Reviewer duty:** Challenge stale screenshots, missing submission IDs, or ambiguous "score" labels.
- **Evidence:** Pending.
- **Status:** NOT_RUN
- **Rationale:** Execution inactive.
- **Remediation owner:** integration owner
- **Retest:** Full E-01 after evidence correction.
- **Waiver:** None.

### E-02 - Alpha Package Parity

- **Class:** Mandatory
- **Threat:** Existing Kaggle automation submits the older baseline instead of the current alpha route.
- **Protects:** R-02, AC-02
- **Preconditions:** T-02 package manifest.
- **Checks:** Inspect package/source manifest and entrypoint. Expected: current `celltrack_alpha` route/config is included and used.
- **Reviewer duty:** Verify the artifact was generated after T-02 and is the one submitted.
- **Evidence:** Pending.
- **Status:** NOT_RUN
- **Rationale:** Execution inactive.
- **Remediation owner:** T-02/integration owner
- **Retest:** Reinspect package and submitted artifact.
- **Waiver:** None.

### E-03 - Submission Schema And Graph Validity

- **Class:** Mandatory
- **Threat:** Kaggle rejects the CSV or scores a malformed graph.
- **Protects:** R-03, AC-03
- **Preconditions:** T-03/T-06 candidate CSV and validation report.
- **Checks:** Validate header, required datasets, consecutive `id`, node/edge row fields, references, out-degree, merges, and generated row counts.
- **Reviewer duty:** Challenge partial test splits and stale CSV paths.
- **Evidence:** Pending.
- **Status:** NOT_RUN
- **Rationale:** Execution inactive.
- **Remediation owner:** T-03/T-06
- **Retest:** Regenerate and revalidate CSV.
- **Waiver:** None.

### E-04 - Local Evidence And Leakage Guard

- **Class:** Mandatory
- **Threat:** A site submission is made from degraded, untested, or leakage-prone code.
- **Protects:** R-05, R-07, AC-04
- **Preconditions:** T-03/T-06 local evidence.
- **Checks:** Run focused tests, scorer parity, synthetic benchmark, reproducibility/leakage checks, and relevant local CV/regression checks. Expected: PASS or documented packaging-only exception before site submission.
- **Reviewer duty:** Ensure skipped tests or stale reports do not count as PASS.
- **Evidence:** Pending.
- **Status:** NOT_RUN
- **Rationale:** Execution inactive.
- **Remediation owner:** T-03/T-06
- **Retest:** Re-run affected local checks.
- **Waiver:** None.

### E-05 - Kaggle 0.92 Endpoint

- **Class:** Mandatory
- **Threat:** Goal completes without reaching the actual site score target.
- **Protects:** R-04, AC-05
- **Preconditions:** T-04/T-07 scored Kaggle submissions.
- **Checks:** Read Kaggle submission result. Expected for completion: score `>=0.92`. Expected for block: score below `0.92` after D-04 cap or exact Kaggle blocker.
- **Reviewer duty:** Distinguish public/private/visible score labels and confirm the score belongs to this artifact.
- **Evidence:** Pending.
- **Status:** NOT_RUN
- **Rationale:** Execution inactive.
- **Remediation owner:** T-07/integration owner
- **Retest:** Re-check Kaggle result after submission/score availability.
- **Waiver:** None.

### E-06 - Permission And Budget Compliance

- **Class:** Mandatory
- **Threat:** Run exceeds user-approved Kaggle, worker, browser, file, or compute boundaries.
- **Protects:** R-06, AC-06
- **Preconditions:** Closeout and node reports.
- **Checks:** Count scored submissions, inspect browser/API actions, command logs, generated/deleted files, worker launches, and compute use. Expected: within D-01/D-02/D-04; no paid compute.
- **Reviewer duty:** Challenge hidden external effects, unrelated Kaggle actions, and missing cleanup records.
- **Evidence:** Pending.
- **Status:** NOT_RUN
- **Rationale:** Execution inactive.
- **Remediation owner:** coordinator/integration owner
- **Retest:** Reaudit closeout and effect logs.
- **Waiver:** None.

### E-07 - Kaggle-To-Local Scorer Calibration

- **Class:** Mandatory
- **Threat:** Local CV is treated as a faithful Kaggle proxy despite sparse-label, aggregation, node-penalty, division, or test-distribution differences.
- **Protects:** AC-07 and the integrity of subsequent low-compute iteration.
- **Preconditions:** Baseline plus frozen diagnostic submissions with pre-score local predictions, artifact hashes, Kaggle submission IDs, and returned site scores.
- **Checks:** Confirm variant definitions and calibration/holdout roles were frozen before score observation. Recompute local scores with the revised scorer. Expected: every frozen holdout satisfies `abs(local_score - kaggle_score) <= 0.02`; calibration-point fit alone is insufficient.
- **Reviewer duty:** Challenge post-hoc holdout changes, multi-axis variants, missing artifact identity, and score comparisons that mix public/private or per-sample/aggregate values.
- **Evidence:** Pending.
- **Status:** NOT_RUN
- **Rationale:** No valid Kaggle submission or site score exists yet.
- **Remediation owner:** T-05/T-06/integration owner
- **Retest:** Recompute all frozen paired observations after each scorer revision without changing the holdout set.
- **Waiver:** None.

## Permission Mapping

| Capability | Decision | Scope |
| --- | --- | --- |
| Cruise worker external packet/evidence transfer | D-01 | Bounded Cruise worker packets/reports for G-01; no secret values |
| Repository writes | Plan/Start only | Declared node ownership paths |
| Tests/builds/local generation | Plan/Start only | Local workspace/temp paths and generated reports |
| Network/Kaggle read | D-02 | Kaggle competition/submission/leaderboard content for this competition |
| Kaggle external mutation | D-02, D-04, D-07 | Up to 5 valid scored submissions in this calibration phase for this competition |
| Chrome control | D-02 | Logged-in Kaggle pages only, if needed |
| Credential handling | D-02 | Use existing session/API without printing or exposing secret values |
| Paid compute | None approved | Prohibited |
| Git commit/push/PR | None approved | Prohibited unless separately requested |

## Shared Resources

| Resource | Mutable? | Owner | Ordering/lock | Permission D-ID | Rollback/cleanup |
| --- | --- | --- | --- | --- | --- |
| `kaggle_kernel/**` | Yes | T-02 then integration owner | Serialized before submissions | D-02 | Preserve diff; do not revert unrelated files |
| `reports/kaggle_g01/**` | Yes | Current active node; coordinator owns closeout | Serialized ledger writes | D-02, D-04 | Generated files may be replaced if owned |
| Kaggle account/submissions | Yes | T-04/T-07 only | Serialized, max 5 valid scored submissions in this phase | D-02, D-04, D-07 | Remote submissions recorded; no unrelated cleanup |
| Chrome session | Yes | T-04/T-07 only | Kaggle pages only | D-02 | Stop on unrelated/sensitive prompts |
| `.cruise/goal/CLOSEOUT.md` | Yes | Coordinator/integration owner only | Created after Start | D-01/D-02 | Append accurate actual state only |

## Iteration Policy

- First submit only after T-02/T-03 local package and validation gates are ready.
- After the first site score, run T-05 even if it is `>=0.92` so scorer calibration remains independently verified.
- Freeze four single-axis diagnostic definitions and their local predictions before observing their Kaggle scores; designate the final two as holdouts.
- Diagnostic submissions require schema/graph validity and attribution evidence, but no minimum expected leaderboard score.
- Fit scorer revisions only on the baseline and calibration diagnostics; accept parity only when every frozen holdout is within absolute `0.02`.
- Stop Blocked when the five-submission D-04 cap is reached without scorer parity and the `0.92` endpoint, Kaggle scoring/auth/quota blocks, or the next credible path requires paid compute/large training/new permission.

## Coverage Validation

| ID | Implement/protect node | Evidence | Gate | Notes |
| --- | --- | --- | --- | --- |
| R-01 | T-04, T-07, T-01 | AC-01, AC-05 | E-01, E-05 | Score provenance |
| R-02 | T-02, T-01 | AC-02 | E-02 | Alpha package parity |
| R-03 | T-03, T-06, T-01 | AC-03 | E-03 | CSV validity |
| R-04 | T-04, T-05, T-07, T-01 | AC-05 | E-05 | 0.92 site endpoint |
| R-05 | T-05, T-06 | AC-04 | E-04 | Low-compute iteration |
| R-06 | T-04, T-07, T-01 | AC-06 | E-06 | Permissions and budget |
| R-07 | T-03, T-06, T-01 | AC-04 | E-04 | Reproducibility/leakage |
| Scorer calibration | T-05, T-06, T-07, T-01 | AC-07 | E-07 | Frozen Kaggle/local holdout parity |
| D-01 | All worker nodes | Worker launcher reports | E-06 | External worker data transfer |
| D-02 | T-04, T-07 | Kaggle evidence logs | E-06 | Kaggle/Chrome account use |
| D-03 | T-04, T-07, T-01 | Site score evidence | E-05 | Endpoint source |
| D-04 | T-04, T-07, T-01 | Submission ledger | E-06 | Max 5 valid scored submissions in this phase |
| D-07 | T-05, T-06, T-07, T-01 | Calibration manifest and paired scores | E-07 | Absolute holdout error <=0.02 |

## Mechanical Validation

- T-01 appears exactly once and is read-only.
- Dependencies are acyclic: T-02 -> T-03 -> T-04 -> T-05 -> T-06 -> T-07 -> T-01.
- Every mutable output has one active owner at a time.
- External Kaggle effects map to D-02 and D-04.
- Every mandatory requirement maps to at least one node, AC, and gate.
- Future G-02/G-03 work is excluded from G-01 except as closeout recommendations.

## Closeout Requirements

After Start, `.cruise/goal/CLOSEOUT.md` must record:

- Activation method and approved artifact digests.
- Actual commands, browser/API steps, and worker attempts.
- Generated package/submission identities.
- Kaggle submission IDs, statuses, timestamps, and scores.
- Local validation and scorer-parity evidence for every submission.
- Frozen diagnostic definitions, local predictions, Kaggle scores, residuals, calibration changes, and untouched holdout identities.
- All file replacements/deletions and cleanup.
- Final determination: Completed only if mandatory gates pass and Kaggle site score `>=0.92`; otherwise Blocked with exact blocker.
