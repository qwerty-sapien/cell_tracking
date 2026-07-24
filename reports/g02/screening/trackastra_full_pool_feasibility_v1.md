# Frozen Trackastra complete-pool feasibility audit

Date: 2026-07-24  
Decision: **PASS — grouped-OOF frozen-head implementation authorized**

## Result

The label-free complete-pool audit covered all 69 fit-train `6bba` videos and
retained every deployed proposal identity:

| measure | result |
|---|---:|
| raw conflicting proposals before geometry gate | 2,309,298 |
| frozen-geometry-admissible proposals | 110,471 |
| nonempty transition windows / projected inference calls | 5,824 |
| transitions requiring a cropped 1,024-token window | 2,186 |
| largest full four-frame window | 4,157 detections |
| largest union of required challenger/incumbent endpoints | **870** |
| missing endpoints | **0** |
| projected frozen inference, including 1.5x safety factor | **1.213 h** |

Every challenger and every incumbent endpoint fits in one deterministic
transition-level association matrix. The scheme uses all required endpoints
first, then fills the remaining token budget with detections nearest their
coordinate centroid. No appearance threshold, proposal cap, sampled
background, label, validation12 or CV57 identity was used.

The densest transition leaves only 154 of 1,024 tokens for optional context.
This is valid endpoint coverage but may reduce representation quality. The
grouped-OOF candidate must therefore treat the exact packing rule as part of
its immutable model identity and may not infer that the five-video signal AUC
will transfer unchanged.

## Gate interpretation

This result proves deployment feasibility, not a score improvement. It
authorizes one fit-only grouped-OOF frozen-head implementation:

- generate one frozen Trackastra association matrix per nonempty transition;
- preserve every geometry-admissible challenger and incumbent identity;
- use Trackastra challenger-minus-incumbent margins as additional features;
- fit/calibrate only within nested grouped folds;
- require aggregate held precision at least 0.95 and every fold at least 0.93;
- reconstruct exact held-fit graphs only after the proposal prerequisite;
- require adjusted edge-Jaccard delta `>= +0.005`, nonnegative raw
  edge-Jaccard delta, nonnegative node-recall delta, and observation-distinct
  fragments before validation12 or CV57.

Any endpoint drift, token overflow, resource violation, or failure of the
held-fit graph gate parks this route before LoRA or external compute.

## Artifacts

- Contract:
  `reports/g02/trackastra_full_pool_feasibility_contract_v1.json`
  (`0bb92b8d…b761`)
- Report:
  `reports/g02/trackastra_full_pool_feasibility_v1/report.json`
  (`2296f6b9…2cc7`)
- Implementation:
  `scripts/audit_trackastra_full_pool_feasibility.py`
  (`cc25370a…4dcd`)

The audit took 1,014.79 seconds on CPU. No training, Kaggle, GPU, RunPod, or
submission action occurred.
