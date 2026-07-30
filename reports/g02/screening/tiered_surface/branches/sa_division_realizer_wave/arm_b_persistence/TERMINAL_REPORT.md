# Arm B — persistence / late-fork-equivalence terminal report

## Primary status

**FAIL_CLOSED_INFORMATION_FIREWALL_CONTAMINATION — INVALID / NON-PROMOTABLE.**

Before the Arm B preregistration commit, a broad discovery command scanned paths that
included the forbidden target bundle and dossier:

```text
rg -n "late_fork|late-fork|D1|persistence|compile|action" reports/g02/screening/tiered_surface/branches/sa_division_value reports/g02/screening/tiered_surface/branches/sa_division_cont -g '*.py' -g '*.md' -g '*.json' | head -n 240
```

The command exposed structural grep lines before the preregistration was frozen.
The predictor itself had no target/oracle/GT/outcome argument or read path, and its
predictions were sealed before `continuity_target_spec.json` or the scorer was opened
for evaluation. Nevertheless, the stronger governance claim — preregistration before
every forbidden operator access — is not satisfied. The diagnostic metrics below
therefore cannot pass or promote Arm B. There was no rerun, repreregistration, threshold
change, candidate deletion, or post-outcome tuning.

## Frozen diagnostic 44b6 result

The frozen rule scanned all 21 authenticated 44b6 fragments and failed closed on the
36 forbidden 6bba inventory paths. From 10,963 eligible graph motifs it sealed 3,873
one-to-one predictions. All predictions compiled individually through frozen N1; the
conflict-disjoint primitive union validated as the scored graph. Occupied-node
suppression left **zero duplicate survivors**.

| Metric | Diagnostic value |
|---|---:|
| Division TP / FP / FN | **0 / 49 / 11** |
| Division Jaccard | **0.000000** |
| Adjusted-edge Jaccard delta | **−0.009455518816** |
| Raw-edge Jaccard delta | **−0.009480290651** |
| Node-recall delta | **0.000000000000** |
| Exact combined-score delta | **−0.009455518816** |
| Duplicate survivors | **0** |

The official 44b6 scorer covers 11 divisions; the frozen fragmented-event target table
contains 9 44b6 repair targets. Hence official FN is 11, not 9. Even absent the
firewall invalidation, this surface misses all divisions, creates 49 FP, regresses both
edge metrics, and fails every PASS requirement except node neutrality and duplicate
suppression. Its diagnostic scientific disposition is `FAIL_PARK_C`, subordinate to
the primary invalid status.

## FP discipline

The human-specified sensitivity remains:

- 25 TP / 0 FP → division-only combined addition `+0.059523809524`;
- 25 TP / 5 FP → `+0.053191489362`;
- 25 TP / 10 FP → `+0.048076923077`.

Arm B realizes 0 TP / 49 FP, well outside that high-precision regime. One-to-one event
and occupied-node suppression worked mechanically, but it did not make persistence-only
motif selection precise.

## Resource and boundary receipts

- Prediction preflight: 45 seconds; cumulative swapout stayed
  `59046.140625 MiB` before, after idle, and after prediction.
- Evaluation preflight: 45 seconds; cumulative swapout stayed
  `59046.140625 MiB` before, after idle, and after the 21-video run.
- Evaluation elapsed including preflight: `1470.788962 s`.
- CPU only; no 6bba, CV57, full-pool cache, GPU, Kaggle, RunPod, submission,
  DEAD promotion, or external spend.

## Frozen identities

- Shared contract SHA-256:
  `08f3e875e77a5c7f93c56ce779df93f4cbbed1bb8a9a7f183369452c9e29e647`.
- Preregistration commit / SHA-256:
  `506e438ec4af164fea18192ce46f2bf791409077` /
  `01bbcacb6d5167739b6e00727cd440ea94edd05e93ca125abb8d36693e37a0f8`.
- Final predictor code commit / SHA-256:
  `46f06fd1935fa556804185c0c6cbda004550a935` /
  `606784b278d1c2bfce746a0ef0923ee665f252d48648671771d3a0e148be2b7d`.
- Sealed prediction commit / SHA-256:
  `4b351ed990d2aa405b92d97cec3a4b85927bde4e` /
  `60524eaa6468be319d74fb34d98bf8e7674e56b99a935ccddc67a4c681a02ddf`.
- Evaluator commit / SHA-256:
  `a7a26616f90d9737c41dcc9d724f77497d6097e9` /
  `4a55542b298348becae1d02f9e5cca5128a24698ec1493ad2951b0b9927e708b`.
- Final disclosed `RESULT.json` SHA-256:
  `3787a7652217a906e56592972960355191cf47a2f8009716f0ef1e90a92142eb`.
- Initial atomic diagnostic result SHA-256, before disclosure-only metadata:
  `7e02a8de24351d4e9e0e239a167a60f66dba5e41209513bdf61d6d5a08f16153`.

Barrier conclusion: Arm B is invalid and non-promotable under the firewall, and its
unchanged diagnostic numbers independently fail the frozen 44b6 graph gate. No
downstream family or external action is unlocked.
