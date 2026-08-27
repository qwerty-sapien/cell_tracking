# Point-context OOF threshold-headroom diagnostic

Decision: **THRESHOLD-ONLY REPAIR NOT SUPPORTED.**

The compact point-context candidate PARKed at 1,950 correct and 118 harmful
held OOF proposals, precision `0.942940`. This diagnostic asked whether a
single stricter calibration rule could recover the locked `0.95` precision
floor without changing the representation.

The immutable diagnostic contract is
`reports/g02/point_context_oof_threshold_headroom_contract_v1.json`, SHA-256
`fc71f52e0f87286629863bd054f7c840af3a6c56b4eec6e86c360b79ec8308d8`.
It fixed seven calibration precision floors (`0.95` through `0.99`) before
replay and forbade graph reconstruction, validation12, CV57, Kaggle, GPU, and
RunPod actions. Every saved fold model and banked held metric had to reproduce
before analysis.

## Shared calibration-floor result

| calibration floor | held correct / harmful | precision | utility | weakest fold |
|---:|---:|---:|---:|---:|
| 0.950 | 1,950 / 118 | 0.942940 | 1,714 | 0.897500 |
| 0.970 | 1,727 / 107 | 0.941658 | 1,513 | 0.897500 |
| 0.980 | 1,361 / 84 | 0.941869 | 1,193 | 0.000000 |
| 0.985 | 1,174 / 62 | 0.949838 | 1,050 | 0.000000 |
| 0.990 | 1,025 / 49 | 0.954376 | 927 | 0.000000 |

No shared floor passed the preregistered combination of aggregate precision
`>=0.95`, every-fold precision `>=0.93`, accepted support `>=1,000`, and
conservative utility `>=1,500`. At floor `0.99`, fold 2 remained at
`0.925049`, fold 3 accepted no labeled proposal, and utility fell by 787 from
the banked rule.

## Non-authoritative oracle

Choosing a separate threshold from each held fold's labels—strictly an upper
bound—could produce 2,035 correct and 83 harmful proposals, precision
`0.960812` and utility `1,869`. Its thresholds ranged from `0.168591` to
`0.925360`. This shows that the representation contains useful separation,
but the calibration partition does not identify a transferable operating
point. The oracle cannot authorize graph replay or a new candidate.

## Consequence

Close the scalar-threshold repair branch for this model identity. Do not tune
against the five held folds or spend CV57 on it. The next candidate must change
the transferable representation or learn calibration from broader
pretraining; the existing shortlist points next to a provenance-verified
frozen pretrained Trackastra representation with a small fit-only head.

Machine report:
`reports/g02/point_context_oof_threshold_headroom_v1/report.json`, SHA-256
`18c16840276f1c6851a3bf1c094fd4d82c42d29019773a33275120bb6c09d683`.
The CPU run took 27.65 seconds, peaked at 0.544 GiB process-tree RSS, completed
with zero swap growth, and performed no external action.
