# CELLECT-R7 separation sample v2 — terminal report

Status: **INVALID / infrastructure failure before feature computation**.

The human-approved packaging-only retry was frozen in
`CELLECT_R7_EXECUTION_CONTRACT_v3_packaging_retry.json`
(`59077286ae6b0dad94b59a6055ffb033e1db6a46c357669bc1f3b37ff524d6db`).
The private asset dataset version containing the three hash-pinned offline wheels became
visible, and private T4 kernel version 2 was launched under authorization
`b93fa2ed66d9f2a23781b6459209f4b18f49471b360c4e2f05f784cfc2463f8b`.

The offline Zarr runtime installed successfully. The kernel then stopped before model
inference with:

`FeatureError: expected one U-ext+-x3rd-149.0-4.6540.pth, found 0`

The terminal log is
`kaggle_v2_error/g02-cellect-r7-separation-v1.log`
(`0405857f6197eac6c9a418bc8d9db5dc860d7b35520d393b1d6aaec18ab53fe7`).
Elapsed metered runtime was 982.946 seconds; the conservative booked debit is 0.2731
kernel-hours. Together with v1, the two infrastructure attempts cost 0.2766 kernel-hours.

No complete feature manifest, B0/A1/A0 metric, per-class PASS/FAIL, graph vector, or
scientific outcome was produced. R7 therefore remains PARK-B. No full-pool cache, graph,
6bba, CV57, submission, RunPod action, DEAD promotion, plan amendment, or tier move
occurred. `uncited_tier_moves` remains zero.

The v2 contract forbids another automatic retry. Resumption requires a new explicit human
authorization and a new pre-outcome packaging contract that authenticates the mounted
CELLECT weight path without altering the frozen populations, labels, folds, metrics, or
thresholds.
