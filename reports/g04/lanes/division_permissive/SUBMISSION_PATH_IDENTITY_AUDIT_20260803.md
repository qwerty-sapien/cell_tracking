# G04 submission-path runtime identity audit — 2026-08-03

## Verdict

**PASS — no additional live code-submission path hard-codes the mounted test
population.** The sole match is the already-invalidated historical copy
transport. It remains immutable evidence and is registry-quarantined; it is not
repaired in place or eligible for another launch.

The repaired `onebitbrain/g04divcap5-v1` path has already completed one exact
Kaggle code-output submission (`ref 55203439`, Version 1) and received public
score `0.881`. That is the required end-to-end blind-rerun canary evidence: a
Kaggle code submission cannot exercise the blind rerun without also creating a
normal competition submission. Launching an identical second one would violate
the frozen one-shot/no-retry policy, so no duplicate remote mutation was made.

## Audited execution paths

| Path | Runtime test identity source | Result |
|---|---|---|
| `kaggle_kernel/g04-division-permissive-submit-v1/g04_division_permissive_submit_v1.py` | `EXPECTED_DATASETS` is a frozen four-public-dataset registry; it never reads competition input | Historical KFG-25 defect. `TRANSPORT_INVALIDATION.json` marks it `TERMINAL_INVALIDATED_NO_RETRY`, and the ownership registry limits it to evidence only. |
| `kaggle_kernel/g04divcap5-v1/main.py` + embedded APNCC runner | Submit mode passes no `--datasets`; `run_bh.py:564` discovers every `test/*.zarr` at the mounted runtime input and rejects an explicit submit-time list | PASS. |
| `scripts/g04_division_hidden_postprocess.py` | Reads fresh parent-CSV dataset names and independently enumerates mounted `test/*.zarr`; requires the two sets to match exactly | PASS. Any missing, extra, or non-Zarr mount fails closed. |
| `kaggle_kernel/bhap-submit/bh45.py` | APNCC submit-mode runner dynamically discovers the complete mounted test split rather than using its public registry | PASS. This is the parent generator used by the repaired G04 pipeline. |
| `kaggle_kernel/alpha_submission.py` | Enumerates `test/*.zarr` at runtime | PASS. |
| `kaggle_kernel/g02_division_lane1/division_submission_kernel.py` | Iterates datasets parsed from its supplied full base submission; it does not embed test identities or perform a code-output submission | PASS for this anti-pattern. |
| `scripts/g04_hidden_kaggle_code_submit.py`, `scripts/kaggle_submission_lifecycle.py`, `scripts/submit_kaggle_alpha.py` | Submission/lifecycle wrappers; they do not generate test identities | PASS. |

`PUBLIC_DATASETS` in `scripts/g04_division_hidden_guard.py` and the fixed set in
`scripts/g04_division_permissive_sweep.py` are public-parity validators run
locally after the public kernel result. They are not shipped into the
blind-rerun execution path and must not be mistaken for runtime membership
sources.

## Evidence

- `reports/g04/lanes/division_permissive/code_transport_v1/TRANSPORT_INVALIDATION.json`
  records the legacy `55201829` blank/zero-byte failure and its permanent
  quarantine.
- `reports/g04/lanes/division_permissive/hidden_transport_v1/cmux_receipts/competition_code_submit_2026-08-03T053925046519Z.json`
  binds the repaired Version 1 to its code-output submission.
- `reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/G04_CAP5_PUBLIC_CALIBRATION_20260803.json`
  records that ref `55203439` terminalized `COMPLETE` with public `0.881`.
- Focused dynamic-discovery/guard/lifecycle regression: 21 tests passed on
  2026-08-03. The suite includes an unseen hidden dataset name, a mounted
  inventory mismatch refusal, exact-version binding, and one-shot submission
  lifecycle checks.

## Scope and no-retry conclusion

The documented terminal result is a **transport pass**, not evidence that the
candidate improves the leaderboard: it tied APNCC at displayed precision. The
next score-bearing candidate must still be observation-distinct and meet its
own scientific gate. If a future code-output canary ever errors, the KFG-25
diagnosis must be reopened rather than patched and resubmitted blindly.
