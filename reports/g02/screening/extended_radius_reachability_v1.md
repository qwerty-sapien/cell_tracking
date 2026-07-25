# Fit-only extended-radius reachability

**Decision: BANK the radius headroom; do not expand the runtime tensor without
an explicit background-abstention gate.**

The exact frozen learned-association tensor (`10 µm`, 8 nearest targets)
contains `7,874,673` candidate pairs across the 117 fit-only videos, exactly
reconciling the prepared selected-graph artifact. Official 7 µm one-to-one
detection matching finds 74,718 adjacent, one-to-one GT edges with both
endpoints detected. Of these:

- 73,999 are reachable by the current tensor;
- 719 are excluded;
- 718/719 are excluded by distance above 10 µm;
- only 1/719 is excluded by cap-8 crowding within 10 µm.

The missing-candidate mechanism is therefore radius, not neighbor cap.

## Radius/cost trade-off

| policy | extra correct edges reachable | share of 719 recovered | candidate pairs | pool multiplier |
|---|---:|---:|---:|---:|
| 10 µm / cap 16 | 1 | 0.14% | 7,877,697 | 1.000× |
| **12 µm / cap 8** | **456** | **63.42%** | 13,216,744 | **1.678×** |
| 12 µm / cap 16 | 471 | 65.51% | 13,525,760 | 1.718× |
| 15 µm / cap 8 | 606 | 84.28% | 18,828,374 | 2.391× |
| 20 µm / cap 16 | 701 | 97.50% | 38,052,534 | 4.832× |

The 12 µm/cap-8 result replicates across the held-out fit-validation partition:
it recovers 51/70 excluded correct edges (72.86%) at a 1.631× candidate-pool
multiplier. On fit-training it recovers 405/649 (62.40%) at 1.682×.

Cap 16 adds only 15 correct edges over 12 µm/cap 8 across all fit videos, so a
cap expansion is not justified.

## Why this is not yet a candidate

The 12 µm shell adds 5,342,071 raw candidate pairs for 456 newly reachable
one-to-one correct edges—roughly one labeled positive per 11,715 added pairs
before graph-conflict filtering. A broad radius expansion would overwhelm the
solver and invalidate the successful sparse appearance rule.

The next SCREEN must therefore:

1. enumerate only actual occupied-endpoint conflicts in the 10–12 µm shell;
2. use full selected-graph predecessor/successor context, not the conservative
   partial context that just failed;
3. train with explicit shell-background/no-link examples;
4. require out-of-video precision at least 0.95 and a sparse per-video support
   cap before any prediction replay;
5. confirm on the 12 fit-validation videos before CV57.

The reachability scale is material: 456 fit-only correct edges are in the same
order as the 378 ideal wrong-to-right swaps required to move the current local
candidate from 0.8815 to 0.90. This is headroom, not a projected score.

Machine report:
`reports/g02/extended_radius_reachability_v1.json`
(`sha256 22a70e0b750d3d59bdd92ca5a82967d35c5d866fafe1c867dbb09b936efc2803`).
No predictions, GPU action, external action, or submission were performed.
