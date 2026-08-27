# VELOCITY-AWARE-CANDIDATE-WINDOW-v1 — repaired exact result

## Verdict

**Integrity repaired; reachability material; exact score negative.** The legacy
attempt mixed the 57-video CV57 replay with the 117-video GA shell and failed
conservation at 228 rows. That attempt remains excluded. This repair rebuilds
the actual 117-video G1-selected graph and does not substitute CV57
predecessors or relax the `456−90=366` guard.

The fixed-r10/cap8 motion-predicted recentering cleared every prescorer gate,
so the exact frozen scorer was run on the global-jump-corrected APNCC base. It
regressed decisively:

| Vector | Adjusted edge Jaccard | Raw edge Jaccard | Node recall |
| --- | ---: | ---: | ---: |
| G1 base | 0.886764420260 | 0.891461012028 | 0.990377056057 |
| Velocity recenter | 0.880534473728 | 0.885184801739 | 0.990377056057 |
| Delta | **−0.006229946531** | **−0.006276210289** | **0** |

Exact recentered edge TP/FP/FN is `34,200 / 2,368 / 2,068`, versus G1
`34,389 / 2,308 / 1,879`. The named realization is therefore **PARK-C / no
submission**. This is a scientific negative, not an integrity block. G1 remains
the banked positive.

## Exact fit117 cohort repair

`fit117_g1_selected_v1/SEALED_COHORT.json` seals exactly 117 identities in the
fit117 prepare order (105 train + 12 validation; 41 44b6 + 76 6bba), with
per-video hashes for the source graph fragment, BHC cache, cache metadata,
shell example, registration census, and output graph.

- actual G1-selected graph: 2,878,849 nodes / 2,758,880 edges;
- 122 pool-breaking transitions in 46 videos;
- G1 local edits: 18,548 raw edges removed, 15,987 registered edges added,
  958 conflicting synthetic edges removed, 2,369 pruned pairs skipped;
- GT archive opened during construction: **false**;
- CV57 substitution: **false**.

## Prescorer reachability ceiling

The sealed input table `fit117_recenter_v1/SEALED_INPUTS_366.csv` contains all
366 residual shell records and explicitly carries selected predecessor IDs and
vectors, the prior/source/target cumulative G1 shifts, graph-endpoint presence,
and the recentered cap-eight rank.

Strict conservation passes exactly: 456 original GA shell positives minus 90
G1 distance rescues equals 366 residual edges. The guard was not relaxed.

| Gate | Observed | Required | Result |
| --- | ---: | ---: | --- |
| Newly reachable true continuations | 67 | >=20 | PASS |
| Optimistic no-new-FP raw ceiling | 0.001736831190 | >=0.0005 | PASS |
| Background burden | 0.993271735102x | <=1.05x | PASS |
| Residual shell retained after cap8 | 98 / 366 | >=20 | PASS |

Across the deployment population, 61 previously reachable continuations are
also lost, so the net reachability change is only +6. The 366 residual records
partition as 98 reachable, 211 outside recentered r10, 28 with source absent
from the actual G1 graph, and 29 with target absent as the primary reason (one
additional target absence co-occurs with a missing source). No residual edge is
lost solely to cap8 after entering recentered r10.

## Exact-score integrity

The scorer implementation was hash-frozen before the outcome in
`EXACT_SCORER_IMPLEMENTATION_FREEZE.json`. It changes only the per-source query
coordinate to

`qhat[t+1] = q[t] + 0.30 * (q[t] - q[G1-selected-parent,t-1])`,

with exact fallback `qhat=q`, fixed radius 10 um, fixed cap eight, frozen edge
weights, unchanged image/checkpoint features, and the unchanged legacy graph
matcher. For frame 0 of all 57 CV videos—where no predecessor can exist—the
candidate membership and both pair/no-link probabilities are bit-exact against
the stock scorer. Exact fragment coverage is 57/57 with no missing, duplicate,
extra, or invalid fragment.

## Authoritative artifacts

| Artifact | SHA-256 |
| --- | --- |
| fit117 sealed cohort | `b8967e6104bee453a56cdd28b08f1bdec3aaaa5e1e3bffc0524d29ca5912bc08` |
| 366-row sealed inputs | `17721b0e38abff7542fcfd73ac018b34e39b642961934d60649ed148f688f42a` |
| reachability ceiling | `9b1d7f94c6580449fa3ed71f65486c8748bbe11dcbb0e54116730572a95bce7e` |
| exact-score implementation freeze | `4e17f2a3424bfc9d264ef6929b93247b7efbff7c7e2760954ba8cc98a91b5d52` |
| exact CV57 score | `5826c203f6f40c7acebb8bf937ae657cbad3413b632aa76fbb453fb5ff8f4872` |
| exact recentered fragment aggregate | `ff5618a63f6fbb5de59148ba9ccacf23ad5e813c31f8592ea9b6c73ed7e1a524` |

No refit, threshold search, GPU, Kaggle action, RunPod action, radius widening,
or DEAD-mechanism rescreen occurred in Lane A.
