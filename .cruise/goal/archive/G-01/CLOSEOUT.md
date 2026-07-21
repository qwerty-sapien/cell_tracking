# Closeout: G-01 - Kaggle Site Verification And Low-Compute Push To 0.92

- Execution state: Active; Kaggle scored submission still blocked by Kaggle submit permission.
- Last updated: 2026-07-15.
- Endpoint: Kaggle site score `>=0.92`, not local CV.
- Scored submissions consumed in this G-01 flow: `0/5` in the current calibration phase.

## Outcome

The alpha package now generates a full Kaggle test `submission.csv` through the private unscored kernel, and the CSV passes local schema/graph validation plus the aggregated pre-submit gate. Direct CSV upload and kernel-output submission were both attempted against Kaggle. Both routes fail at Kaggle's submission-creation endpoints with HTTP 403, and the submissions endpoint still reports `No submissions found`. This is not a model-output or CSV-format failure; it is a Kaggle account/team/rules submit-permission block.

## Current Candidate

| Artifact | Value |
| --- | --- |
| Kernel | `onebitbrain/biohub-celltrack-alpha` |
| Kernel version | `4` completed |
| Package identity | `8335c7d79d77425cedde8fca4a0d4b28d4419b8d82719427ed9add9034bd5730` |
| CSV | `reports/kaggle_g01/kernel_output/submission.csv` |
| CSV SHA-256 | `2c45439aa299c27e42c2b2eef97a743e5db3761ef8602f88505186910b48031d` |
| Rows | `101,374` data rows |
| Nodes / edges | `57,004` nodes / `44,370` edges |
| Test datasets | `4/4` present |
| Gate | `reports/kaggle_g01/pre_submit_gate_report.json`, status `PASS` |

## Node Attempts

| Node | State | Result |
| --- | --- | --- |
| T-00 | Complete | Repository/submission path probe completed. |
| T-02 | Complete | Alpha Kaggle package built with `alpha_submission.py` and `celltrack_alpha` route `auto`. |
| T-03 | Blocked/superseded | Full local generation did not complete on the local 87.4 GB ZIP. |
| T-03R | Complete | Diagnosed local resource/runtime block and recommended unscored Kaggle kernel generation. |
| D-05 | Complete | User approved unscored private Kaggle kernel generation. |
| T-03B v1 | Failed before CSV | Missing `zarr` in Kaggle image. |
| T-03B v2 | Failed before CSV | `celltrack_alpha` not importable in Kaggle runtime. |
| D-06 | Complete | User approved continued unscored packaging/runtime repair attempts. |
| T-03B v3 | Failed before CSV | Missing `/kaggle/src/alpha_submission_config.json`. |
| T-03B v4 | Complete | Embedded config and bundled source worked; full `submission.csv` downloaded. |
| T-03C | Complete/PASS | CSV validation passed: all datasets, no missing endpoints, no duplicate edges, no merges, no outdegree >2, no warnings. |
| T-04 file | Blocked | Direct CSV upload reached 100%, then `CreateSubmission` returned 403; no submission created. |
| T-04 kernel | Blocked | Kernel-output submission returned 403 at `CreateCodeSubmission`; no submission created. |
| T-04 after Chrome/plugin reinstall | Blocked | Chrome updated to `150.0.7871.125`; extension/native-host checks pass, but live bridge discovery is empty and recovery still fails with `kLSNoExecutableErr`. Fresh direct-file and kernel-output submission attempts again returned 403; no submission was created. |

## Changes And Evidence

| Scope | Change/evidence |
| --- | --- |
| `scripts/build_kaggle_kernel.py`, `kaggle_kernel/alpha_submission.py` | Added Kaggle-runtime repairs: fallback Zarr v3 reader, single-file bundled modules, embedded alpha config. |
| `scripts/validate_remote_submission_csv.py` | Local no-network validator for downloaded Kaggle CSVs. |
| `scripts/submit_kaggle_alpha.py` | Guarded submit wrapper added; validates CSV/gate, records before/after Kaggle submissions, supports file and kernel modes, classifies 403 submit blocks. |
| `tests/test_submit_kaggle_alpha.py` | Unit tests for submit-result classification. |
| `reports/kaggle_g01/t04_pre_submit_validation_report.json` | Latest local validation report for the candidate CSV. |
| `reports/kaggle_g01/t04_submission_attempt.json` | Guarded direct-file submit attempt; status `blocked_kaggle_file_submission_permission_403`. |
| `reports/kaggle_g01/t04_code_submission_attempt.json` | Guarded kernel-output submit attempt; status `blocked_kaggle_code_submission_permission_403`. |
| `src/celltrack_alpha/kaggle_calibration.py` | Deterministic single-axis graph probes: edge thinning, long-edge pruning, short-component pruning, and conservative birth-track division rescue. |
| `scripts/build_kaggle_calibration_probes.py` | Freezes probe order, roles, artifact hashes, and per-dataset transformations before any Kaggle score is observed. |
| `scripts/score_cv_submission.py` | Scores arbitrary submission-style CSVs against the exact locked CV clips with the official metric. |
| `src/celltrack_alpha/site_calibration.py`, `scripts/update_kaggle_calibration_ledger.py` | Fits identity/bias/affine correction on calibration pairs only and evaluates untouched holdouts at absolute error `<=0.02`. |
| `reports/kaggle_g01/calibration_probes/frozen_probe_manifest.json` | Baseline plus four validated, pre-registered Kaggle probe artifacts. |
| `reports/kaggle_g01/calibration_probes/calibration_ledger.json` | Pre-Kaggle local scores and deltas; status `awaiting_kaggle_scores`. |

## Completion Evidence

| AC | Expected check/result | Current evidence | Result |
| --- | --- | --- | --- |
| AC-01 | Prior local score labeled local-only; remote evidence recorded if found | Local score remains local-only; Kaggle leaderboard was readable but no own score exists. | PASS |
| AC-02 | Submitted artifact uses `celltrack_alpha` current route/config | Package parity manifest shows `alpha_submission.py`, `from celltrack_alpha import predict_dataset_window`, route `auto`, no baked submission rows. | PASS |
| AC-03 | Full `submission.csv` passes local validation | `t04_pre_submit_validation_report.json`: ready true, 4/4 datasets, 57,004 nodes, 44,370 edges, no warnings/errors. | PASS |
| AC-04 | Local pre-submit checks pass | `pre_submit_gate_report.json`: status `PASS`, candidate CSV hash matches. | PASS |
| AC-05 | Kaggle submission ID/status/score recorded; complete only at `>=0.92` | Both submit routes return 403 and `competitions submissions` says `No submissions found`. | BLOCK |
| AC-06 | Permission and budget ledger complete | D-04 remains `0/5` valid scored submissions consumed in this phase; no paid/GPU/internet compute used. | PASS |
| AC-07 | Frozen scorer holdouts within absolute `0.02` of Kaggle | No authoritative Kaggle score exists yet, so no calibration or holdout residual can be computed. | BLOCK |

## Verification Attempts

| Attempt | Status | Evidence/finding |
| --- | --- | --- |
| CSV dry-run gate | PASS | `python3 scripts/submit_kaggle_alpha.py --dry-run` passed validator and pre-submit gate. |
| Direct CSV scored submit | BLOCK | `python3 scripts/submit_kaggle_alpha.py --mode file` uploaded the CSV, then Kaggle returned 403 at `CreateSubmission`; before/after history: `No submissions found`. |
| Kernel-output scored submit | BLOCK | `python3 scripts/submit_kaggle_alpha.py --mode kernel --attempt-report reports/kaggle_g01/t04_code_submission_attempt.json` returned 403 at `CreateCodeSubmission`; before/after history: `No submissions found`. |
| Leaderboard/access probe | PASS | Kaggle files and public leaderboard are readable; current visible leaderboard top is `0.970` as of the probe on 2026-07-15. |
| Chrome capability retry | BLOCK | Chrome 150 is running; extension and native-host checks pass, but no extension browser binds. The approved recovery launcher fails with `NSOSStatusErrorDomain Code=-10827 (kLSNoExecutableErr)`. Cruise provisioning defect recorded as `qwerty-sapien/cruise#1`. |
| Post-reinstall Chrome retry | BLOCK | User reinstalled Chrome and the ChatGPT Chrome bridge and logged into Kaggle. Chrome `150.0.7871.125`, extension registration, and native host all pass; browser discovery remains empty and the recovery launcher repeats `kLSNoExecutableErr`. |
| Post-reinstall file/API retry | BLOCK | Exact baseline SHA `2c45439a...` revalidated and uploaded fully; Kaggle returned 403 at `CreateSubmission`; before/after history remained empty. |
| Post-reinstall kernel/API retry | BLOCK | Exact kernel v4 artifact revalidated; Kaggle returned 403 at `CreateCodeSubmission`; before/after history remained empty. |
| Locked CV regeneration | PASS | Exact promoted score reproduced at `0.8713516075529171`; 4,478 nodes, 2,745 edges, zero predicted/scored divisions. |
| Frozen scorer probes | PASS/WAIT | Four Kaggle CSVs pass schema/graph validation. Pre-score local predictions: edge-thin `0.7814388072`, long-edge prune `0.8044315247`, short-component prune `0.8574125098`, division rescue `0.8713516076`. Kaggle pairs remain unavailable. |

## Current Block

Kaggle accepts reads and unscored kernel execution, but rejects creation of scored submissions for the API account/team that owns `onebitbrain/biohub-celltrack-alpha`. Reinstalling Chrome and the ChatGPT Chrome bridge did not restore browser attachment: all static checks pass, but discovery remains empty and the profile-window recovery repeats `kLSNoExecutableErr`. The provisioning issue is recorded at `qwerty-sapien/cruise#1`. The remaining identity check is whether the logged-in Chrome Kaggle username is `onebitbrain`. If it is different, the API token and browser session address different accounts. If it is `onebitbrain`, Kaggle still has not granted that account/team submission permission despite the accepted web session.

```bash
python3 scripts/submit_kaggle_alpha.py --mode file
```

If file submissions remain blocked after web entry is confirmed, rerun the kernel-output route:

```bash
python3 scripts/submit_kaggle_alpha.py --mode kernel --attempt-report reports/kaggle_g01/t04_code_submission_attempt.json
```

## Final Determination

- Mandatory local evidence: PASS.
- Kaggle site score evidence: BLOCKED/NOT_AVAILABLE.
- Goal state: Active; cannot be completed until Kaggle creates a scored submission and returns a site score.
