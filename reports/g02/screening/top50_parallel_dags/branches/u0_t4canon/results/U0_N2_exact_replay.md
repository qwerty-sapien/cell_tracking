# U0-N2 — exact T4-native replay

Date: 2026-07-27  
Verdict: **GO**  
Score: `NOT SCORED`

## Preregistered gate

Two fresh-process CPU replays from identical authenticated T4 inputs must be
byte-identical across fitted parameters, calibration decisions, selected proposal
IDs, graph-mutation payloads, and manifests. Disabling the U0 challenger must reproduce
the APNCC mutation payload exactly. No float tolerance is permitted.

## Result

**GO.** The two complete 105-video replays are byte-identical:

- run 1 SHA-256: `1d1a8ac2273acb22aa68f097cd3645fce4929f71faf7fba9f029aaab932b4758`
- run 2 SHA-256: `1d1a8ac2273acb22aa68f097cd3645fce4929f71faf7fba9f029aaab932b4758`
- `cmp`: exact equality
- disabled challenger: exact APNCC mutation payload on **105/105** videos
- held identity present in fit/calibration: **0**
- graph scorer calls: **0**

The canonical graph payload is the scorer-free reconstruction unit frozen at N1:
base selected-edge identity plus exact node-ID remove/add mutations. The replay
contains 11,799 APNCC control mutations and 6,391 active U0 mutations.

The unchanged calibration rule also produced important negative evidence: **all five
44b6 folds have `NO_ADMISSIBLE_THRESHOLD`** at the frozen ≥100 accepted / ≥0.95
precision requirement, while all five 6bba folds have admissible thresholds. The 44b6
decision is an exact fail-closed no-op; no threshold or tolerance was weakened. This
does not fail U0's governance/replay gate, but it means U0 by itself supplies no
44b6 Trackastra-linear beta signal and cannot be presented as model evidence.

The 45.58-second preflight had 0 MiB swap growth. Complete runs took 4.663 s and
4.667 s, peaked at 0.327 and 0.332 GiB RSS, and each had 0 MiB swap growth.

Two earlier implementation rehearsals failed closed before a complete comparison:
one exposed label-free video handling and one exposed the 44b6 no-threshold result.
Both repairs were made explicit and tested before the two final runs; neither changed
a threshold, tolerance, data scope, scorer, or old evidence.

Full pins and refinement audit are in
`results/u0_t4canon_exact_replay_attestation.json`.

Old `H1-ZWRFEAT-CACHE-ATTEST` remains
`DEAD_STRUCTURAL_M5_T4_DRIFT`.

## Proposed ledger row

`U0-T4CANON-REPLAY | exact two-process T4-native replay | CONFIRM / INTEGRITY |
105/105 byte-identical; disabled challenger exact APNCC; all 44b6 folds deterministically
NO_ADMISSIBLE_THRESHOLD under unchanged rule | NOT SCORED | 0 submissions | 0 GPU-h |
0 RunPod | GO to HUMAN-GATE | no scorer/CV57; 0 MiB swap growth |
U0_N2_exact_replay.md`

## Cascade

GO → `N3_human_lock.md`; halt there for `HUMAN-DECISION`.
