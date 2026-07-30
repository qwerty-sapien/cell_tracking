# D1-PERSISTENT-FORK terminal report

Date: 2026-07-27  
Terminal node: `D1-N2`  
Verdict: **PARK**

## Cascade trace

`D1-N0 GO → D1-N1 GO → D1-N2 PARK`

`D1-N3`, `D1-N4`, and `D1-N5` were not dispatched.

## Banked foundation

- Four arms and all graph thresholds were frozen before outcomes.
- The official scorer replay matched **14/14** sandbox classifications.
- The fixed D1 reference decoder passed **51/51** local graph-invariant
  counterfactuals.
- Duplicate fork survivors were **0**.
- Repeated canonical dry output was byte-identical.

The branch stopped for the preregistered causal reason: no one named association
arm has yet passed adjusted delta `>=+0.005` on both 44b6 and 6bba with raw/node
nonnegative and increased connected fork opportunities. This is a dependency
PARK, not a D1 mechanism failure.

## Usage and authority

- Held-fit D1 graph executions: **0**
- 6bba D1 graph access: **0**
- CV57 access: **0**
- GPU/Kaggle/RunPod/submission/push/external mutation: **none**
- `LEDGER.md`: not edited; proposed rows are in N0/N1/N2 reports

## Exact terminal artifact

`reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/d1_persistent_fork/N2_association_dependency.md`

Resume at D1-N2, without retuning, only after the complete dependency named in
that artifact is authenticated.
