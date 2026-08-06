# Lane 1 dense deformable motion — terminal ceiling screen

Date: 2026-08-03  
Verdict: **DELETE FAIL_PARK_C / OCCUPIED_SWAP FAIL_PARK_C / LEARNED DISPLACEMENT CLOSED**

This is a genuinely different evidence type from A7. A7 supplied one rigid
global phase-correlation translation per frame pair. Lane 1 estimated a
spatially varying three-component TV-L1 field over every consecutive raw 3-D
volume pair, then sampled that field at each candidate source location to form
predicted-motion, residual-vector, residual-norm, and cosine features. It is
also distinct from DEAD hand-tuned motion/decode rules because the field was
estimated from voxel intensities before candidate geometry was consumed.

The amended synthetic-only precondition passed seven known 3-D translations
with maximum median interior endpoint error `0.80253` native voxel. The frozen
population run then processed all `41` 44b6 identities and `4,059` frame pairs,
emitting `1,391,458` action rows in `77` feature files. It took
`6,386.171 s` CPU wall time and peaked at `229.94 MiB` RSS. Oracle-label,
graph, 6bba, CV57, GPU, RunPod, and submission access were all false during
feature generation.

The existing evaluator SHA `548b6818…30ca` consumed the unchanged sealed B0,
then A1, then the within-identity/class/source-frame permuted A0. Results were:

| class | fold | B0 AUC | A1 AUC | A0 AUC | A1-B0 AUC | A1-A0 AUC | B0/A1/A0 retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| DELETE | 0 | 0.665039 | 0.722147 | 0.700146 | +0.057109 | +0.022001 | 0.003268 / 0.006536 / 0.003268 |
| DELETE | 1 | 0.695730 | 0.770644 | 0.707166 | +0.074914 | +0.063478 | 0 / 0.010101 / 0.010101 |
| DELETE | 2 | 0.596911 | 0.646992 | 0.578624 | +0.050081 | +0.068368 | 0 / 0 / 0 |
| DELETE | 3 | 0.665646 | 0.687785 | 0.644297 | +0.022140 | +0.043489 | 0 / 0 / 0.005128 |
| DELETE | 4 | 0.613032 | 0.707490 | 0.650503 | +0.094459 | +0.056988 | 0.002849 / 0.008547 / 0.002849 |
| occupied SWAP | 0 | 0.625542 | 0.658066 | 0.609446 | +0.032524 | +0.048621 | 0 / 0 / 0 |
| occupied SWAP | 1 | 0.837077 | 0.837961 | 0.836278 | +0.000883 | +0.001683 | 0 / 0.142857 / 0.142857 |
| occupied SWAP | 2 | 0.622301 | 0.612279 | 0.629869 | -0.010021 | -0.017590 | 0 / 0 / 0 |
| occupied SWAP | 3 | 0.664797 | 0.696030 | 0.655480 | +0.031233 | +0.040549 | 0 / 0 / 0 |
| occupied SWAP | 4 | 0.656014 | 0.667685 | 0.662121 | +0.011672 | +0.005565 | 0 / 0 / 0 |

Neither class passed every clause in any held fold. DELETE carried real AUC
signal, including A1-B0 lifts above `0.05` in four folds, but its A1-A0 lift
failed fold 0, its absolute AUC failed fold 2, and retention was at most
`0.010101`, far below the fixed `0.15` floor. Occupied SWAP had a negative A1
lift in fold 2 and near-zero retention outside one seven-positive fold; it also
failed the fixed conjunction. This is a measured ceiling below the machinery
gate, so no VoxelMorph-class training, sweep, graph candidate, or GPU action is
funded.

Primary literature supports only the mechanism class. TV-L1 is an L1 data-term
plus total-variation dense-flow estimator. VoxelMorph reports brain-MRI Dice
`0.753` versus ANTs `0.749` and NiftyReg `0.755`, `0.45 s` TitanX inference,
and a default `150,000` single-pair training iterations; that is brain MRI, not
zebrafish. Linajea operates on whole-embryo data including zebrafish and trains
cell-indicator and movement-vector networks for `400,000` iterations with batch
size one, but its learned detection/candidate-graph interface differs from this
frozen action supplier. Its largest published lineage-span effect is on mouse
(`75.8%` versus `31.8%` over one hour), not a transferable G-03 effect size.

Authoritative result: `run_v1/SEPARATION_RESULT.json` SHA
`8c5d9f8c…cecac9d`. Feature manifest SHA `f2fd6119…defceb`.

Primary sources:

- <https://www.ipol.im/pub/art/2013/26/>
- <https://arxiv.org/abs/1809.05231>
- <https://pmc.ncbi.nlm.nih.gov/articles/PMC7614077/>
