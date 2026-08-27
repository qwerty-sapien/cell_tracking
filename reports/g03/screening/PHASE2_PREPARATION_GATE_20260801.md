# G-03 Phase 2 preparation gate

Date: 2026-08-01  
Updated: 2026-08-03  
Status: **H000 COMPLETE / BRANCH B DOSSIER FAIL / SCIENTIFIC GPU CLOSED**

Phase 1 passed. This document turns the selected
`G03-TA-CROPADAPTER-E2E-V1` direction into a fail-closed preparation queue.
It is not a training preregistration and authorizes no graph read, CV57,
submission, RunPod action, or scientific GPU trajectory.

## Frozen facts

- Existing detections and the Trackastra association-to-graph compiler remain unchanged.
- Each detection will receive one deterministic intensity tensor with axis order `t,z,y,x` and shape `(3,16,32,32)`.
- A trainable 3-D crop adapter must emit exactly twelve real-valued `wrfeat`
  channels after training. **Owner correction dated 2026-08-03:** its final
  `Linear(48,12)` projection has both weight and bias initialized to exact zero,
  followed by `tanh`, so the immutable h000 checkpoint emits exact zeros and
  reproduces the established zero-feature interface. This initialization is
  required, not a forbidden fallback; nonzero post-training outputs remain the
  scientific mechanism.
- Adapter, Trackastra transformer, and association head train end-to-end; all trainable state and optimizer groups must be checkpointed.
- The 105-video fit universe is the maximum preparation envelope, not a statement that every member trains the model. Its exact train/evaluation partition must be frozen without leakage; held-fit 44b6 is evaluation-only, and all audit/CV57 data remain inaccessible until their separate gates.
- Graph checkpoints at cumulative 0, approximately 5, 30, 60, and 90 T4-h are
  fixed before objective access. H000 must first pass exact zero-feature graph
  parity. H005 is a fixed early-slope read: delta `<=1e-6` from h000 stops and
  preserves the remaining approximately 25 T4-h; a larger positive delta may
  continue only under every unchanged gate. Continue only at held-fit
  adjusted-graph deltas `>=+0.010` at 30 hours and `>=+0.035` at 90 hours. The
  60-hour read is diagnostic and cannot select or continue a trajectory unless
  the owner separately adds a numerical gate.
- One seed and no sweep are allowed before the 30-hour result.

## Required closure receipts before any scientific launch

| gate | required immutable receipt | current state |
|---|---|---|
| raw-volume provenance | exact file identities, axis/voxel interpretation, detection-to-volume join, missing/duplicate policy | **OPEN** |
| crop semantics | center rule, physical alignment, even-size convention, padding value/mode, normalization statistics and fit-only provenance | **OPEN** |
| augmentation | stateless transform keyed by `(run_seed, global_step, sample_id)`, with ranges and disabled evaluation path | **OPEN** |
| adapter census | layer-by-layer architecture, parameter count `<=1,000,000`, output width 12, exact-zero h000 output, nonzero hidden crop-dependence and projection-gradient tests | **CLOSED BY `phase2_zero_init_v1/PHASE2_ZERO_INIT_CONTRACT.{json,md}`: 22,684 parameters; exact `Linear(48,12)` zero weight/bias; bitwise-zero output; nonzero hidden activations and projection gradients** |
| trainability census | every trainable/frozen tensor and optimizer group, with pretrained-weight identity | **OPEN** |
| resource calibration package | exact package hashes plus T4 peak VRAM, examples/s, checkpoint bytes, and projected 30/90-hour costs | **OPEN; small T4 calibration may be proposed only after the preceding bytes freeze** |
| output inventory | declared-only checkpoint, metrics, logs, handoff tree, graph-emission and terminal-validation files; undeclared or partial output fails | **OPEN** |
| checkpoint schedule | precommitted h000/h005/30/60/90-hour graph-checkpoint identities and access order; the 60-hour graph read is diagnostic and has no performance decision line | **PARTIALLY CLOSED BY `phase2_zero_init_v1/PHASE2_ZERO_INIT_CONTRACT.{json,md}`; h000 and fixed h005 rules are sealed, scientific training remains closed on the Branch-B dossier and remaining preparation receipts** |
| cost ceiling | worst-case full campaign projection `<=200 T4-h` and approximately 25 T4-h reserve retained | **OPEN** |
| prelaunch validation | static/runtime proof that no held-fit/audit/CV57 bytes are read during training or calibration | **OPEN** |

## 2026-08-03 h000 and allocation disposition

- Immutable h000 checkpoint SHA-256:
  `603e02f8f152db642a177d57763a07bebf84518f868abbbe1e0246187b018f95`.
- Zero interface: projection weight/bias and emitted `wrfeat` are bitwise zero;
  all `36/36` canonical held-fit36 graphs are byte-identical to the same-runtime
  zero-feature control.
- Corrected held-fit36 score: `0.7630876414338513`; Branch B. The earlier
  `0.829194058079541` native result is CV57 and is not a same-population h000
  baseline.
- Historical random-adapter R1 scored `0.7637338804169099` on held-fit36, so
  its controlled delta over zero-init h000 is `+0.0006462389830586`. The
  proposed `~0.066` initialization penalty is not supported by the matched
  comparison.
- The owner-required effect-size dossier verdict is **FAIL**: no audited source
  supplies a target-metric, target-domain `<=30 T4-h` curve or calibrated path
  to `0.915` and `0.945`. Allocation remains `0/30 T4-h`.
- H005 stays preregistered at `4.75–5.25` cumulative T4-hours but is
  **NOT FUNDED / NOT MEASURED**. No slope exists; all 30 scientific T4-hours
  are preserved. H030 remains unauthorized, and its `>=+0.010` gate is
  unchanged.

The crop normalization, padding, raw-source join, adapter topology, and T4
throughput are deliberately not invented here. They require repo-backed source
evidence or a preregistered calibration. Filling those fields is this week's
primary engineering work; launching before they are exact would invalidate the
selected architecture's own boundary.

## Portfolio boundary

Reuse AE-P7's passed three-sample raw-crop interface smoke as evidence while
building the crop contract. Do not reopen the archived G-02 pointwise
DINOv2/PEFT/selector mechanisms. Keep `G03-CROPBANK-CALIB-V1` and
`G03-DENSEEMB-ASSOC-PROBE-V1` parked until a new frozen package and gate make
one observation-distinct; available quota alone is not justification.
