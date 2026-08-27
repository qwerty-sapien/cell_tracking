# Native pretrained-tracker base wave — preregistration

Recorded before any native-base prediction or score outcome on 2026-07-31.

## Authority and objective

The owner explicitly opens this observation-distinct wave for exhaustive use of the
`24.94` authenticated remaining Kaggle GPU-hours before the 2026-08-01 refresh. The
per-launch `>=5 h` human gate is granted for this window; the weekly `30 h` cap,
per-kernel hard wall-clock kill, exact quota reconciliation, frozen official scorer,
serial push fence, and no-public-submission gate remain binding. RunPod is forbidden.

The scalar objective is the exact frozen competition score

`S(G) = adjusted_edge_Jaccard(G) + 0.1 * division_Jaccard(G)`.

Every candidate is reported against APNCC-DIRECT (`0.881522069960` on CV57) with raw
edge Jaccard, adjusted edge Jaccard, node recall, division TP/FP/FN/Jaccard, and the
combined score separated. A measured CV57 score above `0.881522069960` halts the wave
for immediate owner review. It does not authorize a public submission.

## New mechanism and crowding verdict

This is not another correction head. It replaces the complete APNCC edge solution with
Trackastra's pretrained association matrix plus Trackastra's global greedy feasibility
solver over the fixed V45/APNCC detection registry. The model is evaluated with the
same authenticated zero-wrfeat point interface that produced the completed H1 cache.
The alpha node is the untouched division term: native forks are evaluated as part of a
complete graph, and the APNCC-edge/native-fork hybrid measures both the edge perturbation
and the division gain. Trackastra itself is crowded beta; treating its native graph and
native fork topology as the base generator is observation-distinct in this campaign.

## Lane A contract

Fit scope is exactly the previously frozen fit105 manifest: 36 `44b6` and 69 `6bba`
videos, with no validation12 or CV57 access during A1/A2. Nodes are the authenticated
V45 sparse detections. For tractable inference at the pretrained `1024`-token limit,
each source node is evaluated exactly once in a deterministic spatial chunk. The chunk
contains all of that source's adjacent-frame candidates within `12 um` (up to 16 nearest,
banked before score selection) and nearest four-frame context up to 1024 tokens. This
tiling is fixed before outcomes and is reported as `trackastra-native-spatial-chunk-v1`.

A1 emits two frozen defaults from the same bank:

- `greedy_nodiv`: link radius `12 um`, neighbor cap `10`, association threshold `0.50`,
  maximum outdegree 1.
- `greedy_div`: the same, maximum outdegree 2, second-child/division threshold `0.50`.

A2 is the Cartesian fit-only sweep:

- link radius: `8, 10, 12 um`;
- neighbor cap: `8, 10, 16`;
- association threshold: `0.35, 0.50, 0.65`;
- division threshold: `0.50, 0.65, 0.80`, plus divisions off.

All variants use the native global descending-weight greedy constraints: at most one
incoming edge and at most one/two outgoing edges. No GT field enters inference or graph
construction. Selection is fixed as the lexicographic maximum of: exact fit105 combined
`S(G)`, division Jaccard, adjusted edge Jaccard, raw edge Jaccard, node recall, then the
lexicographically smallest configuration name. No family-specific post-selection is
allowed. Exactly one selected configuration receives exactly one CV57 measurement.

A4 overlays only native predicted forks whose two outgoing native edges are both
present in the selected native graph onto the APNCC edge solution. It removes any
APNCC outgoing edge from that parent and any incoming APNCC edge into either child
before atomically inserting the fork. The complete hybrid is rescored; edge preservation
is not assumed.

## Lane B contract

CELLECT is reopened only as a distinct native-tracker diagnostic, not as the terminal
PARK-C feature-supplier realization. Before a new spend, remote dataset version and
content identity must differ from the terminal closeout that proved zero safe-loader and
checkpoint matches. If unchanged, the standing same-signature rule makes B0 immediately
red without a redundant launch. If changed, one diagnostic kernel has a 30-minute hard
kill, recursively inventories `/kaggle/input` with SHA-256 before resolution, catches
every stage, and always writes a report. Any non-green result permanently abandons B.

## Runtime and stop rules

- Family jobs are separate kernels so their pushes can be serialized while runtimes
  overlap. Each kernel has a `17,100 s` internal deadline and is externally killed by
  `18,000 s`.
- Every video writes an atomic compressed edge-bank checkpoint plus a journal entry.
  A terminal or killed run retains completed checkpoints as durable evidence.
- A CUDA OOM is caught once per chunk, the source chunk is halved deterministically,
  and the retry is recorded. A one-source chunk that still OOMs terminates that video
  and the kernel report; there is no unchanged retry.
- Infrastructure receives at most 30 minutes per lane. A repeated registered failure
  signature is terminal.
- Quota is queried before each push and after each terminal state; the authenticated
  pre/post delta, not internal timing, is the debit.

