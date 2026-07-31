# G-02 tiered-surface campaign — terminal report

**Frozen:** 2026-07-31

**Disposition:** CAMPAIGN CLOSED; no active experimental wave

**External actions:** no submission, no GPU launch, no RunPod launch
**Governing plan:** `campaign_narrowing_and_hypothesis_dag_v1.md`
(`a75786530df809df20f69ccf0d18a5458c78500a476e5d155746d4686ae73f77`)

## Executive disposition

The banked result remains **APNCC-DIRECT: local `0.881522069960`, public
`0.881`**. The local adjusted score is fixed by
`selected_graph_appearance_runtime_parity_cv57_v1/report.json`
(`a02d3439354d22614a223d5effbd32f9b6f3f794dd0a8db08915b41c874180e8`).
The authenticated public archive is `results-bhap.zip`
(`031aa5f64fd6eaf918213fc072a098657e2a6bc3d3e5a2d2ae0642d4037b7219`);
its public score and byte-exact submission identity are adjudicated in
`apncc_transport_smoke_v2/PUBLIC_ARCHIVE_PARITY_ADJUDICATION.json`
(`d0cf4c80ece8cb442e232531b80e908fc34ecd06603eadc670819294766e3e15`).

The tiered-surface campaign is terminal. The measured action-selector
requirement is materially above every tested representation. On DELETE,
independent appearance and assignment evidence converge at `0.670412` and
`0.670235` AUC, a difference of only `0.000176`, against the measured
requirement of `0.85`. The campaign therefore treats approximately `0.670` as
the empirical Bayes ceiling of the **current pointwise beneficial-action label
formulation**. This is a campaign inference from convergent evidence, not a
formal proof of a universal Bayes bound.

No tested supplier unlocks a graph-gate preregistration. No further wave,
submission, GPU use, or plan amendment is authorized by this report.

## Measured selector-requirement surface

DAG 2 run_v6 measured how synthetic selector quality transfers through the
unchanged held-fit graph machinery. Its authoritative surface is
`selector_requirement_44b6/run_v6/requirement_surface.json`
(`6e62a29ce3effdd2a3748716a9843f10f5f5954a322d10547f7a4cafc4d6c71b`);
the human-approved measured-gate amendment is
`SEPARATION_GATE_MEASURED_AMENDMENT_V4.json`
(`3526b54b29c6372488f15ef715ace4f76e4744efc1a7031215abc28cef64951d`).

The binding mean requirements are:

| Class | Required AUC | Required retention | Mean adjusted delta at binding point | Empirical p05 |
|---|---:|---:|---:|---:|
| DELETE | `0.85` | `0.020394` | `+0.006281` | `+0.004597` |
| occupied SWAP | `0.90` | `0.067910` | `+0.005563` | `+0.004413` |

DELETE remains primary because its measured prize is approximately `4.41×`
the occupied-swap prize (`+0.081897667` versus `+0.018571397` oracle ceiling).

### DELETE dose response

| Synthetic AUC | Prefix N | Mean retention | Mean adjusted delta | p05 adjusted delta |
|---:|---:|---:|---:|---:|
| `0.55` | 0 | `0.000447` | `0.000000` | `0.000000` |
| `0.65` | 32 | `0.001655` | `+0.000019` | `−0.000076` |
| `0.75` | 2,048 | `0.005501` | `+0.000402` | `−0.000276` |
| **`0.85`** | **16,384** | **`0.020394`** | **`+0.006281`** | **`+0.004597`** |
| `0.90` | 16,384 | `0.044946` | `+0.014959` | `+0.013113` |
| `0.95` | 16,384 | `0.115340` | `+0.030787` | `+0.029098` |
| `0.99` | 16,384 | `0.409258` | `+0.057932` | `+0.056379` |

### Occupied-SWAP dose response

| Synthetic AUC | Prefix N | Mean retention | Mean adjusted delta | p05 adjusted delta |
|---:|---:|---:|---:|---:|
| `0.55` | 0 | `0.001866` | `0.000000` | `0.000000` |
| `0.65` | 32 | `0.004851` | `+0.000026` | `−0.000248` |
| `0.75` | 512 | `0.009328` | `+0.000511` | `−0.000356` |
| `0.85` | 1,024 | `0.033955` | `+0.003035` | `+0.000639` |
| **`0.90`** | **1,024** | **`0.067910`** | **`+0.005563`** | **`+0.004413`** |
| `0.95` | 1,024 | `0.147388` | `+0.009163` | `+0.007748` |
| `0.99` | 512 | `0.455970` | `+0.014176` | `+0.013028` |

These are grid measurements, not interpolated claims. The robust reference
points remain DELETE `0.90 / 0.044946` and SWAP `0.95 / 0.147388`; the binding
campaign decision uses the mean requirements above.

## Representation ceiling

All comparable supplier screens used the frozen 44b6 action population, sealed
B0, grouped folds, probe, within-frame/video alignment control, and
per-video CG0 false-positive budgets.

| Evidence family | DELETE AUC | SWAP AUC | Terminal reading |
|---|---:|---:|---|
| incumbent B0 | `0.647271` | `0.681146` | below the measured surface |
| C1 TA-HOCT-LITE | `0.502034` | — | near-random; graph mechanism DEAD |
| DINOv2 ViT-S/14 endpoint appearance | `0.670412` | `0.682473` | scientific FAIL; PARK-C |
| C2 K-best assignment evidence | `0.670235` | `0.664616` | scientific FAIL; PARK-C |

C1 is pinned by
`top50_parallel_dags/branches/c1_ta_hoct_lite/outputs/c1_n3_44b6_graph.md`
(`6aee3c64f2789afb2bb87f4a7d551493ae1c40e47b8a5244492fc4cbd6ea938b`).
DINOv2 is pinned by
`sa_dinov2/local_bakeoff_v3/TERMINAL_REPORT.json`
(`538514df2a7b7554247543413c68209e5c413d866a9c229e2bb7ad1cafbab81e`)
and its full result
(`c58b1a58357c214728c1f2661932df0e8165a8422044af07c707a9ec9e0d41f2`).
Assignment evidence is pinned by
`sa_assignment_separation_v1/SEPARATION_RESULT.json`
(`264f318205d7a5a97d2a9eccde81966408dc46d5d2eebe2ce36d0cedd73e994e`)
and its measured verdict
(`8a204d9911ad6a9c4991decb8f3c7efa7de5dc91b84fa67b617b521308f57352`).

DINOv2's DELETE A1 mean exceeds its within-frame-permuted A0 by only
`+0.029939`; fold lifts are
`+0.049577/+0.037483/+0.003949/+0.018500/+0.040184`. It therefore does not
clear even `+0.03` in two of five folds and never clears the unchanged `+0.05`
B0-lift clause. For SWAP, A1−A0 averages only `+0.010646` and is negative in
two folds. The appearance signal is weak at the frozen alignment control, not
merely below an arbitrary absolute threshold.

The assignment supplier is orthogonal to endpoint appearance: its features are
LAP reduced-cost/min-marginal, force/forbid constrained-solve, Murty K-best
membership/rank-stability, and component-competition quantities. Yet its
DELETE AUC lands within `0.000176` of DINOv2, and its SWAP arm regresses B0 in
all five folds. Convergence of local appearance and global-assignment evidence
supports closure of the current pointwise action-labeling formulation rather
than another appearance-encoder wave.

## Division value and realizability

The exact frozen-N1 replay of all 38 repair bundles establishes the
**GT-dependent ceiling**, not a deployable result:

| Slice | Events | Exact combined-score delta | Deployment reading |
|---|---:|---:|---|
| all exact GT bundles | 38/38 | `+0.092067842` | GT-dependent ceiling |
| tier (a): inferable now | 1/38 | `+0.002406490` | available diagnostic; below `+0.005` materiality |
| tier (b): stronger representation | 37/38 | `+0.089661334` | unavailable; representation-blocked |
| tier (c): GT-only information | 0/38 | `0` | **EMPTY** |

The ceiling's components are adjusted/raw/node
`+0.001591651/+0.001607525/0` plus `0.1 ×` division Jaccard; division is
`38/0/4`, Jaccard `0.904762`. The prize is therefore **not GT-locked**:
37/38 events are blocked by representation, while no event requires
information available only from GT.

The banked FP sensitivity at 25 true positives is:

| False positives | Combined-score delta |
|---:|---:|
| 0 | `+0.059523810` |
| 5 | `+0.053191489` |
| 10 | `+0.048076923` |

Authoritative artifacts are
`sa_division_value/REPRESENTATION_CLOSURE_RECONCILIATION.json`
(`305dc18ad62b7b9791b3d12144b6b5c1feda025690e0c4cf5b0b0c495723899e`),
the exact N0–N3 result
(`6f9cad08202bec5d46554f7aad668ff9fd44005e608712e7932ce2ccd601ff10`),
dependency table
(`d83e32fb610f3be3236de6abb70d88d5101da1776c3542e034a83cb5600ec06c`),
and FP-sensitivity artifact
(`6015a1a6440af4d4da705a416472f3f6b3dab7be764549a969bf507b4e1ddd76`).

## CELLECT closeout

CELLECT is **permanently PARK-C and scientifically UNTESTED**. Six authorized
campaign attempts ended in packaging/environment failures and produced zero
scientific gate results. The final self-contained closeout obeyed the
first-action recursive mount inventory but proved the mounted carrier lacked
both CELLECT loader and checkpoint content hashes. No model, feature, A1, or
A0 output was created. Infrastructure invalidity is not recorded as a
scientific negative.

The terminal diagnosis is
`sa_repr/phase2_separation_gate/cellect_closeout_final_v1/TERMINAL_DIAGNOSIS.json`
(`4a9e2088e7b1f97237e04d3b55ab60be45541dcd9f4b1f56976df9852032c855`);
the permanent PARK-C decision is
`c973397b2b2d965d13a26ad71899241d62b93f6d4cf94b7effe466ba7c6a2fb6`.
The hard no-further-spend rule remains binding.

## Final tier census and registry

The authoritative 45-node census is
`tiered_surface/tier_census.json`
(`29fa828f27e390034570e953ffa0dd14cbd7cd754ebbc879237905f28a8f3f5f`).
There are no LIVE nodes.

| Tier | Count | Complete node registry |
|---|---:|---|
| DEAD | 1 | `R0` |
| PARK-A | 14 | `AE-P0`, `AE-P1`, `AE-P2`, `AE-P3`, `AE-P5`, `AE-P7`, `DIV-O0`, `G0`, `G1`, `G3`, `G4`, `G5`, `SWAP-O0`, `NODETOPO-O0` |
| PARK-B | 18 | `AE-P6`, `AE-P8`, `AE-P9`, `DIV-F2`, `GAP-F1`, `NODERESTORE-O0`, `DET-F0`, `DET-F2`, `DET-F3`, `R2`, `R3`, `R4`, `R5`, `R6`, `R8`, `R9`, `R10`, `G6` |
| PARK-C | 12 | `AE-P4`, `AE-P10`, `DIV-F0`, `DIV-F1`, `GAP-O0`, `NODEADD-O0`, `DET-F1`, `R1`, `R7`, `G2`, `ADD-O0`, `DELETE-O0` |

The machine census records parent nodes. Exact supplier instantiations add the
following terminal overlay without rewriting parent-node history:

| Instantiation | Terminal state |
|---|---|
| C1 TA-HOCT-LITE | DEAD after its 44b6 graph kill |
| Trackastra frozen endpoint supplier | scientific FAIL / PARK-C instantiation |
| DINOv2 DELETE and SWAP | scientific FAIL / PARK-C instantiations |
| C2 assignment-evidence DELETE and SWAP | scientific FAIL / PARK-C instantiations |
| CELLECT DELETE and SWAP | permanent PARK-C / scientifically UNTESTED |

The complete named DEAD mechanism-class register from plan §6.1 remains:

- scalar recalibration, thresholding, and distribution normalization:
  `SCENCAL`, `POINTCTX-THRESH`, `APNCC-POLICY`;
- hand-tuned graph-cost and motion knobs: historical sweeps, `GA-SP`,
  `LC-P4CTRL`;
- geometry/NCC/constant-velocity shell variants: `R12SHELL`, `MGSHELL`,
  `TEMPBG`;
- from-scratch rankers without pretraining or domain adaptation:
  `POINTCTX`, `CROPLINK`, `TEMPRANK`;
- full organizer TemporalUNet3D to convergence: `CTRL-COMPUTE`.

No item in this DEAD register may be rescreened without a new named mechanism.
The remaining plan-level conditional parks also remain closed: a detection
front end only after association is solved; division post-processing only
under its explicit fit gate; and GT-conflict gates only with
unlabeled-abstention regularization. The full-38 division realization arms are
PARK-C at their tested instantiations; the oracle ceiling is not an unlock.

## Reopening condition

**This surface reopens only if at least one of the following is supplied:**

1. a **new action-labeling formulation** that changes the estimand—for example,
   batch or sequential action valuation that assigns value in the context of
   the global edit set; or
2. a representation with **independent evidence of AUC `>=0.85` on this exact
   frozen beneficial-versus-harmful action discrimination task**.

Another endpoint-appearance encoder, architectural substitution, upload-path
repair, or uncalibrated proxy does not satisfy the condition. Reopening would
still require a new hash-pinned preregistration and human review. It does not
authorize a graph run, 6bba, CV57, GPU, submission, RunPod, or plan amendment.

## Terminal barrier

The campaign halts here. APNCC-DIRECT remains banked at local
`0.881522069960` / public `0.881`; all supplier-specific graph gates remain
closed; no further waves are active or authorized.
