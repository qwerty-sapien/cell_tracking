# Relative-geometry swap gate v1

Status: **PARK; full CV57 not authorized**

This CPU-only bet tested whether a competitor-relative, abstaining gate could
repair association decisions without globally replacing the graph solver.  It
used the exact 105/12 fit-only split, the frozen cap-8 candidate graph, the
attested edge scorer, and no audit/test/public data during fitting or threshold
selection.

## What was new

The archived conflict residual learned an additive score for every edge from the
same 16 per-edge features.  It attained 93.35% pairwise validation accuracy but
collapsed to 0.848876 on its full held-out replay because a local ranking loss
globally perturbed otherwise-correct assignments.

This bet instead:

- represented each challenger relative to the current edge or dummy winner;
- included signed/absolute feature deltas, relative physical displacement,
  direction, distance, logit margin, dummy margins, and candidate count;
- trained correct-winner controls as negative decisions;
- excluded unknown and division sources;
- allowed at most one source-local promotion above a frozen threshold.

The expanded fit preparation contained 37,400 fully known sources:
37,363 edge targets, 21 true-terminal no-link targets, 16 cap-8-unreachable
no-link targets, and 33 excluded division sources.  The frozen gate used 33,315
training and 4,085 validation source decisions.  Its fit-domain validation
result was 3,776 correct rescues, one harmful swap, threshold 0.70.

## Registered top-10 result

The deployment replay exposed that the fit target was still misaligned with the
actual graph decision.  The pairwise dummy won 382,784/432,221 source decisions,
so the gate issued 233,063 promotions.  It had learned broad source-softmax
recalibration, not sparse graph-conflict repair.

| metric | banked `c=0.25` | swap gate | delta |
|---|---:|---:|---:|
| adjusted edge Jaccard | 0.723298354 | 0.721664744 | **-0.001633610** |
| edge TP | 5,122 | 5,110 | -12 |
| edge FP | 1,125 | 1,125 | 0 |
| edge FN | 725 | 737 | +12 |
| node recall | 0.990384 | 0.987907 | -0.002476 |

The candidate is observation-distinct from control (node Jaccard 0.977903,
edge Jaccard 0.930385, 3.61% eligible-edge disagreement), so this is a real
negative result rather than an identity failure.

## Decision

**PARK.** The registered `+0.005` affected-slice gate failed, so a full CV57 run,
GPU escalation, packaging, and public submission are not authorized.

The next association proxy must define its winner from the **actual frozen graph
selection**, not from pairwise logits.  It should train:

1. positive examples only where a verified GT continuation is blocked by the
   selected source or target edge; and
2. negative controls where the selected graph edge is GT-correct but a feasible
   challenger exists.

That makes intervention sparse and directly aligned with the occupied-endpoint
failure mechanism.

Artifacts:

- `reports/g02/relative_geometry_swap_prepare_v1/prepare.json`
- `reports/g02/relative_geometry_swap_gate_v2/report.json`
- `reports/g02/relative_geometry_swap_screen10_v2/report.json`
- `reports/g02/screening/obs_diff_relative_geometry_swap_screen10_v1.json`
