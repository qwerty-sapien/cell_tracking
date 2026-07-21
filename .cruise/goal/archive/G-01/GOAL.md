# Goal Contract

- Planning state: Plan approved (inactive)
- Mode/current slice: New / G-01
- Last reconciled: 2026-07-15
- Repository identity: `c3b9b703b049fc25f10e2b42bb986bb94f137a72` / tree `6545134fe081bf5a1e5b8e7b44b07ea8b0b66935`
- Plan approval: user selected `1`, mapped to **Approve Plan**, on 2026-07-15
- Origin: `.cruise/goal/ORIGIN.yaml` (`11dc2f88f163807c46437833842d4e18af861f547c8f94ccfe946ff9204022d2`; raw prompt SHA `cf050ce8f0385c4df44dab480f1f2cec4018aa4dcab54b4a189e16be56ff6aa3`)

## Objective

Complete G-01: verify the current Biohub cell-tracking model on Kaggle, calibrate the local scorer to within an absolute `0.02` of frozen Kaggle diagnostic holdouts, and iterate under low-compute constraints until the authoritative Kaggle site score is at least `0.92`, or stop blocked with exact score, permission, parity, and submission-budget evidence.

## Approved Sources

Read and obey these files before editing or submitting:

| Artifact | SHA-256 |
| --- | --- |
| `.cruise/goal/ORIGIN.yaml` | `11dc2f88f163807c46437833842d4e18af861f547c8f94ccfe946ff9204022d2` |
| `.cruise/goal/SPEC.md` | `09d53779bfea615800a615757f754b3ff80651653dfe203a1d62259075f14690` |
| `.cruise/goal/DECISIONS.md` | `058864b4a53b5d62534c2bf1b84327d7d5ac08b71a9b492e7024d5d7dcf2aa45` |
| `.cruise/goal/ALIGNMENT.md` | `09b2a3eb4fc0a05597fa41b7b2e6cf0adb7d09cfcff4cb3f2c26261658152529` |
| `.cruise/goal/ROADMAP.md` | `f0d40f2f25edb8b4cc9ce954effde47564a380059298532de93f5a56e4b97607` |
| `.cruise/goal/PLAN.md` | `c0b79fe571b0b8f713e47bbc9a3a000e35c3759760791250ec4d1266c668f7d1` |

Worker evidence to preserve:

- `.cruise/runs/prepare-g01/T-00_repo_submission_probe.report.yaml`
- `.cruise/runs/prepare-g01/T-00_launcher_block.yaml`

Current local evidence to preserve:

- `reports/general_model_cv_audit_fold0_auto_n61_1d0d_minx32_offsetonly_12ds_12clips.json`
- `reports/general_model_scorer_parity_after_n61_1d0d_minx32_offsetonly.json`
- `reports/synthetic_benchmark_after_n61_1d0d_minx32_offsetonly.json`

## Alignment Stem

Verify the current Biohub cell-tracking model on Kaggle and iterate under low-compute constraints until the authoritative Kaggle site score is at least `0.92`, or stop blocked with exact score and budget evidence.

## Scope

In scope:

- Package or adapt the submission path so Kaggle receives the current `celltrack_alpha` model/config, not the older baseline-only kernel.
- Generate and locally validate a full test `submission.csv`.
- Submit through the approved Kaggle/Chrome or narrow Kaggle CLI/API path.
- Capture Kaggle submission ID, timestamp, status, and site score.
- Iterate with low-compute changes only: route gating, proposal fusion, graph solver costs, lightweight existing scorers/pruners, calibration, validation, and packaging/runtime fixes.

Non-goals:

- No paid compute.
- No large model training from scratch.
- No unrelated Kaggle account actions, purchases, profile/account setting changes, or submissions to other competitions.
- No claim that local CV is the Kaggle site score.
- No git commit, push, PR, or broad cleanup unless separately requested.

Protected behavior:

- Keep the N61 `0.8713516075529171` score labeled as local locked-CV only.
- Preserve official scorer parity, graph validation, leakage/reproducibility checks, and unrelated worktree changes.
- Do not exceed five valid scored Kaggle submissions in the current calibration phase without a new explicit user decision.

## DAG Summary

Execute the current slice from `.cruise/goal/PLAN.md`:

1. T-02: package the current alpha submission path and prove package parity.
2. T-03: generate and locally validate full test submission.
3. T-04: submit baseline alpha and capture the site score.
4. T-05: diagnose score gap.
5. T-06: implement bounded low-compute improvement when score is below `0.92`.
6. T-07: submit the frozen diagnostic candidates, calibrate the local scorer, and capture site scores within the cap.
7. T-01: run final read-only verification.

Remain the Cruise coordinator. Assign a `recommended_level` and launch bounded `cruise worker --level <2-7>` packets for substantive implementation, Kaggle interaction, diagnosis, and verification. Workers must not ask the user questions, expand permissions, integrate sibling changes, or write `.cruise/goal/CLOSEOUT.md`.

## Scorecard And Stop Condition

Completion scorecard:

- AC-01 through AC-07 pass.
- E-01 through E-07 pass.
- Kaggle site score for the submitted G-01 artifact is `>=0.92`.
- Frozen diagnostic holdouts satisfy `abs(local_score - kaggle_score) <= 0.02`.
- Submission budget remains within D-04.
- No permission or protected-behavior violation remains.

Stop Blocked when:

- Five valid scored G-01 submissions are consumed without reaching the endpoint and no new submission budget is approved.
- Kaggle authentication, quota, rules, or scoring availability prevents AC-05.
- Reaching `0.92` plausibly requires paid compute, training large models from scratch, more submissions, or broader account actions.
- Package parity cannot be achieved without a larger architecture decision.
- Mandatory evidence is NOT_RUN or BLOCK without a locked waiver.

## Feedback Loop

Fast loop:

- After T-02/T-03, inspect package parity and CSV validation before any Kaggle mutation.
- After every Kaggle score, compare it with the pre-recorded local score, update the residual and submission ledgers, and decide calibrate, validate, improve, complete, or blocked.
- After every improvement, run focused local checks before another scored submission.

Slower final check:

- T-01 rereads ORIGIN, decisions, PLAN, local evidence, package evidence, Kaggle score evidence, final diff, and CLOSEOUT. It returns PASS/BLOCK/NOT_RUN for E-01 through E-06.

## Permissions

Approved:

- D-01: bounded Cruise worker packet/evidence transfer for this G-01 workflow, excluding secret values.
- D-02: Kaggle account interaction in the logged-in Chrome session or narrow Kaggle CLI/API flow for this Biohub competition, including reading pages, uploading generated `submission.csv`, checking scores, and deleting/replacing generated submission artifacts where necessary.
- D-04/D-07: up to five valid scored Kaggle competition submissions in this calibration phase: one baseline plus up to four frozen diagnostic variants, with no minimum score threshold for the diagnostics.

Not approved:

- Paid compute.
- Secret value disclosure.
- Account/profile/credential changes.
- Unrelated submissions or external mutations.
- Git commit/push/PR.

Runtime approvals from sandbox, browser, Chrome, Kaggle, or OS prompts still apply and do not expand this contract.

## UI And Kaggle Verification Plan

Use the narrowest reliable path:

- Prefer Kaggle CLI/API only if credentials are already configured and no secret values need to be printed.
- Use Chrome only within Kaggle competition/submission pages when logged-in browser state is needed.
- Capture deterministic evidence where possible: submission ID, status, timestamp, score, artifact path/digest, and screenshots/API output without exposing secrets.
- Stop on unexpected account prompts, unrelated sensitive pages, payment prompts, credential-change prompts, or requests to interact outside this competition.

## Checkpoints

Create or update `.cruise/goal/CLOSEOUT.md` only after Start. Keep it current after:

- T-02 package parity evidence.
- T-03 local validation evidence.
- Every Kaggle submission/score.
- Every local/Kaggle scorer residual and calibration revision.
- Every improvement attempt.
- Final T-01 verification.

Each checkpoint records commands or browser/API steps, worker report paths, generated artifact identities, remote mutations, and remaining budget.

## Closeout

The final closeout must state one of:

- Completed: Kaggle site score `>=0.92`, frozen scorer holdouts are within absolute `0.02`, every mandatory AC/gate passed, and no permission violation remains.
- Blocked: exact blocker, current site score, submissions consumed, required next permission/decision/evidence, and whether local evidence still suggests a path.
- Completed (waiver-qualified): only if a separate locked waiver exists for a mandatory NOT_RUN gate.

Do not close as complete from local CV alone.
