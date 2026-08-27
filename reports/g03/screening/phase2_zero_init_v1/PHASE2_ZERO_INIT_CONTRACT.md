# G-03 Phase-2 zero-initialization and fixed-checkpoint contract

Date: 2026-08-03  
Status: **SEALED BEFORE H000 INFERENCE / SCIENTIFIC GPU CLOSED**

## Blocking adapter correction

The Phase-2 adapter is versioned separately from the historical Wave-R R1
adapter. Its last layer is exactly `Linear(48,12,bias=True)` followed by
`tanh`. Both the projection weight and bias are initialized with
`torch.nn.init.zeros_`. Therefore every finite h000 crop emits twelve bitwise
float32 zeros and Trackastra initially sees its established zero-feature
interface.

The adapter has 22,684 trainable parameters. Its nonconstant-crop hidden
activations are nonzero. On the first backward pass, the projection weight and
bias receive nonzero gradients. As with other zero-initialized residual
branches, the zero projection blocks gradients to the earlier convolutional
layers for that first backward pass; once the projection takes its first
update, convolutional gradients are nonzero. This is the precise trainability
claim and avoids overstating the LoRA/ControlNet analogy.

The historical R1 implementation and its `0.7637338804169099` result are not
modified. Their source and contract hashes remain pinned in the machine
contract.

## H000 parity protocol

H000 is an immutable, zero-hour checkpoint. It binds the authentic Trackastra
model and archive hashes, the Phase-2 adapter state, its exact-zero tests, the
Phase-2 contract hash, and the zero scientific-hour state. It contains no
fabricated optimizer/scheduler/scaler state: those objects remain explicitly
uncreated unless the Branch-B funding gate opens.

The h000 graph read uses the exact R1 held-fit36 population, model,
association-bank construction, compiler, and scorer. Before ground truth or a
score is opened, every one of the 36 emitted h000 fragments must be
byte-identical to the established native zero-`wrfeat` fragment with the same
identity. Any mismatch is classified as a decoding-path discrepancy and closes
the scorer until diagnosed.

The previously reported Trackastra native score `0.829194058079541` is a CV57
measurement. It is a useful descriptive reference but cannot be the numerical
parity target for a held-fit36 h000 read. Allocation is reclassified from the
corrected same-protocol held-fit36 score against the binding floor
`0.868299238910600`.

## Fixed early checkpoint

If and only if the Branch-B literature gate passes, h005 is fixed in advance at
`5.00` cumulative T4-hours, with an operational publication window of
`4.75–5.25` hours. It uses the same population, compiler, and scorer as h000
and h030. This is a fixed checkpoint measurement, not checkpoint selection.

The reported slope is
`(h005_score - h000_score) / actual_cumulative_T4_hours`. A delta at or below
`1e-6` is flat/negative at numerical tolerance and stops the trajectory,
preserving approximately 25 T4-hours. A larger positive delta may continue
toward h030, but it bypasses no other gate. The owner’s h030 `>=+0.010` and
h090 `>=+0.035` fixed held-fit improvement gates remain unchanged; h060 remains
diagnostic-only with no inferred decision threshold.

## Funding boundary

No scientific GPU hour may be spent before both h000 parity and an audited
Branch-B dossier PASS. The dossier must report source-native effect sizes,
datasets, controls, uncertainty where available, and disclosed training
budgets; directional plausibility is not a PASS. Shared serial launch,
authoritative quota reconciliation, hard runtime kill, swap abort, exact
version accounting, atomic publication, no-CV57, no-submission, and no-RunPod
controls remain binding.
