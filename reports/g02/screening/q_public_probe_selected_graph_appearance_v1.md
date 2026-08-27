# Q — preregistered public probe: selected-graph appearance direct swap

**State: READY-FOR-HUMAN-APPROVAL; launch and submission require explicit
human approval.**

## Why this is the next probe

This candidate is the first frozen-CV57 result in the previously unconstrained
high-score range:

- local adjusted edge Jaccard: `0.8815220699597315`;
- exact local control: `0.8731495935289003`;
- local delta: `+0.008372476430831166`;
- edge-count change: `+115 TP / -226 FP / -115 FN`;
- observation class: `DISTINCT`;
- node Jaccard: `0.9929723439138748`;
- edge Jaccard: `0.9847360366548744`.

It therefore provides more decision value than another near-tied learned-bonus
probe. It both tests whether a material local association improvement transfers
and supplies the first local/public anchor above `0.88`.

This is tagged as:

- detector/front-end: unchanged;
- association scorer: frozen learned edge scorer;
- graph solver/cost: legacy `c=0.25` parent;
- association post-solver: full 3-D appearance direct swap;
- lineage/division postprocessor: divisions OFF;
- learned architecture: unchanged.

The result is mechanism-distinct. A single submission must not be interpreted as
proof of a universal “model-family effect.”

## Submission-cap accounting

| item | registered value |
|---|---:|
| screening submissions consumed by this proposal | 1, only if approved and submitted |
| maximum per competition day | 5 |
| configs submitted from this decision batch | 1 before redesign |
| final selections reserved | 2 |

The old `bhp0-lb*` batch is not ahead of this candidate in the order. Those
near-tied probes offer less leverage and must be reconsidered only after the
first high-range result updates P0B.

## Predeclared interpretation

Kaggle's displayed score is treated as a rounded interval, not an exact
response. No point public-score forecast is registered because all historical
calibration anchors lie below this candidate and fine-ranking transfer is
unresolved.

After one approved score is observed:

1. record the exact submission/package identity and displayed-score interval;
2. append the local/public pair to P0B;
3. compare the observation vector and mechanism tag with existing anchors;
4. refit only statistically identifiable interval-aware calibration models;
5. redesign the next probe before spending another submission.

The result may validate this candidate as a stronger public baseline, show a
mechanism-specific transfer gap, or leave the high range unresolved. It cannot
by itself establish global calibration or a per-family slope.

## Fail-closed package contract

The package must abort unless every registered identity is present and exact:

- local report score `0.8815220699597315` with tuple
  `34,320 / 2,459 / 1,948`;
- exact `c=0.25` parent configuration and frozen learned-edge artifacts;
- geometry gate threshold `0.605`;
- full 3-D NCC threshold `-0.04768145084381103`;
- patch shape `16 × 32 × 32`;
- node-disjoint deterministic ordering by NCC margin and challenger identity;
- per-video cap `max(16, ceil(0.005 * selected_raw_edges))`;
- direct swaps after unchanged gap closure and before unchanged
  pruning/smoothing;
- divisions OFF and no GNN;
- no unset knob may fall back to a default.

Before requesting submission approval, the package must:

1. reproduce the exact full CV57 score and confusion tuple through the frozen
   scorer;
2. emit its source/config/checkpoint/package hashes;
3. exercise the packaged mechanism over the complete frozen CV57 runtime path;
4. compare that output against the locally banked candidate using the observation
   tool;
5. log expected runtime and the exact GPU quota requested for hidden-test
   generation.

These checks now pass. Package-path replay is exact on 57/57 videos, reproduces
`0.8815220699597315` and `34,320 / 2,459 / 1,948`, and yields a fragment tree
byte-identical to the banked candidate. The private package identity and quota
estimate are recorded in `appearance_probe_package_v1.md`. Generating the
hidden-test `submission.csv` necessarily uses Kaggle's mounted test data and
T4x2 runtime, so it remains behind the human launch gate.

## Human gate

No Kaggle launch or submission is authorized by this document. The immutable
package is ready: manifest identity
`779aa8d3cefd701024cfb1870d56f3fbb0b29ad1b72f120838951932017d606b`,
estimated wall time 7 minutes, budgeted use at most 0.25 Kaggle GPU-hours, and
one competition screening submission after output validation. Two final slots
remain reserved.
