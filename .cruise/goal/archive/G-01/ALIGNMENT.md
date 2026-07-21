# Alignment

- Planning state: Draft
- Mode/current slice: New / G-01
- Last reconciled: 2026-07-15
- Repository identity: `c3b9b703b049fc25f10e2b42bb986bb94f137a72` / tree `6545134fe081bf5a1e5b8e7b44b07ea8b0b66935`
- Origin: `.cruise/goal/ORIGIN.yaml` (`cf050ce8f0385c4df44dab480f1f2cec4018aa4dcab54b4a189e16be56ff6aa3`)

## Stem

Verify the current Biohub cell-tracking model on Kaggle and iterate under low-compute constraints until the authoritative Kaggle site score is at least `0.92`, or stop blocked with exact score and budget evidence.

## Fence

- In scope: correct alpha packaging, full test `submission.csv`, local validation, Kaggle submission/score capture, and bounded low-compute iteration.
- Out of scope: paid compute, large training-from-scratch, unrelated Kaggle actions, and claiming local CV as site truth.
- Protected behavior: local N61 evidence remains local-only; scorer parity and graph validation remain mandatory; unrelated worktree changes are not reverted.
- Done only when: AC-01 through AC-06 pass, including Kaggle site score `>=0.92`, or the run closes Blocked after the D-04 cap with current evidence.

## Sources Of Truth

- Origin: `.cruise/goal/ORIGIN.yaml`
- Decisions: `.cruise/goal/DECISIONS.md`
- Specification: `.cruise/goal/SPEC.md`
- Plan: `.cruise/goal/PLAN.md`
- Worker evidence: `.cruise/runs/prepare-g01/T-00_repo_submission_probe.report.yaml`
- Current local audit: `reports/general_model_cv_audit_fold0_auto_n61_1d0d_minx32_offsetonly_12ds_12clips.json`
- Scorer parity: `reports/general_model_scorer_parity_after_n61_1d0d_minx32_offsetonly.json`

## Reopen Triggers

- User changes the endpoint away from Kaggle site score `>=0.92`.
- Kaggle quota, rules, authentication, or scoring availability prevents AC-05.
- Any plan step needs paid compute, large training from scratch, secret-value disclosure, or more than three scored submissions.
- Package inspection shows the current alpha implementation cannot be made runnable in the Kaggle path without a larger architecture decision.
