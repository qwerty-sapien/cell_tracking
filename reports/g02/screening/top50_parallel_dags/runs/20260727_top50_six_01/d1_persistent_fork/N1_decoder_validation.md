# D1 N1 — fixed decoder dry validation

Date: 2026-07-27  
Verdict: **GO**

## Result

- Pinned official scorer sandbox: **14/14** fixture `TP/FP/FN`
  classifications reproduced.
- Local D1 counterfactuals: **51/51** preserve in-degree `<=1`,
  out-degree `<=2`, time direction, and acyclicity.
- Equivalent duplicate survivors after suppression: **0**.
- Repeated canonical dry output: byte-identical, SHA-256
  `0b47ece998f99da56e435581bbe5c407b2faa75f9388d6651ab1781c51d753df`.
- Held-fit graph scoring, CV57, GPU, and external actions: **none**.

The reference implementation is `d1_decoder.py`; `validate_d1.py` writes the
full fixture/counterfactual audit to `N1_dry_validation.json`. It implements only
the preregistered motif grammar and stable pre-emission suppression. It does not
fit or select a head and does not inspect GT to choose events.

## Arm and feature feasibility

`N1_arm_manifest.json` fixes the unchanged, D0, D1, and
persistence-ablation semantics. A label-free schema audit on one authenticated
fit-cache video from each family confirms endpoint pairs, transition target
times, 49 frozen row features, geometry probabilities, and proposal/conflict
offsets. The selected upstream association graph is intentionally absent at
this stage: N2 must name and authenticate the qualifying arm's emitted graph.

No held identity affects normalization, weights, motif grammar, event ordering,
suppression, or thresholds. No training was needed to validate scorer semantics
and graph invariants.

## Exact identities

| Artifact | SHA-256 |
|---|---|
| preregistration | `34b2aa02…07620f` |
| decoder | `e780a398…bbcf6` |
| validator | `ae7fed45…00014` |
| full dry-validation output | `f6a96d86…deaf9` |

## Proposed ledger row

`D1-PERSISTENT-FORK-N1 | fixed scorer-semantic decoder dry validation | ALPHA
FOUNDATION | 14/14 official fixtures; 51/51 invariants; 0 duplicate survivors;
byte-identical repeat | no held graph score | 0 submissions | 0 GPU-h | GO |
association dependency remains closed | N1_decoder_validation.md`
