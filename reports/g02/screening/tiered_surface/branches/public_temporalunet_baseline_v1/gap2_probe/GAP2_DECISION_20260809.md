# GAP2 CV57 decision — 2026-08-09

## Verdict

**KILL / DO NOT SUBMIT.** The exact `public_v32_gap2` realization completed and
authenticated, but its frozen CV57 score is lower than immutable V32. The downloaded
CSV is also a 162-train-identity, `substrate_only` scoring artifact, not a hidden-test
submission artifact.

## Authentication

- Owner ZIP SHA-256: `0822d6fb546c8b1b96e31934e66f8384cf68d885044a54ffd993d208f9e96b03`.
- Terminal validation: `PASS`, 162 datasets, 3,211,269 nodes, 3,093,026 edges,
  6,304,295 rows; submission SHA-256
  `f487eec54b790020509d550dce1fe6a722e9b8d76a1c8e809fe9745b3cd353ab`.
- Persisted JSONL authenticated at
  `fa3610efadc7106f18f849829d57e5aaf96ff8ef13a8bec70ad2d9f0b73064b0`.
- Exact Kaggle slug is terminal `COMPLETE`. Fresh quota is `6.88/23.12 h`
  used/remaining; the `7.67 h` reservation was released and no active reservation remains.

## Frozen score comparison

| Metric | V32 | V32 + GAP2 | Delta |
|---|---:|---:|---:|
| Combined score | 0.914358473021 | 0.914250858131 | **-0.000107614890** |
| Adjusted edge Jaccard | 0.911655770318 | 0.911473080353 | -0.000182689965 |
| Raw pooled edge Jaccard | 0.907750333953 | 0.908010159994 | +0.000259826040 |
| Node recall | 0.988520839606 | 0.988865767073 | +0.000344927467 |
| Division Jaccard | 0.027027027027 | 0.027777777778 | +0.000750750751 |

GAP2 gains 19 ordinary-association TP, removes 19 FN, adds 10 FP, and emits 6,377
additional CV57 nodes. That small raw-edge gain does not survive the scorer's node-count
adjustment: the adjusted edge term falls. Division TP stays flat at 3 while division FP
drops from 69 to 66. Excess prediction segments improve by only 4, while the number of
fragmented matched GT lineages increases by 3.

The raw pooled edge-Jaccard lift `+0.000260` is far below R1's preregistered `+0.003`
kill line, and the actual frozen objective regresses. This exact enabled
`recover_strict_gap2` realization is therefore closed. Reopening requires a newly named,
mechanism-distinct stitcher rather than threshold or flag retuning.

## Same-run causal addendum

The V32 and GAP2 artifacts above came from independent stochastic inference runs, so
their delta is the correct candidate-versus-baseline comparison but not a clean estimate
of the GAP2 postlink action itself. A 2026-08-09 diagnostic authenticated GAP2's final
allocated two-node ID blocks and removed those nodes plus causal incident edges from the
completed GAP2 run, holding every other output—including later line-fit-smoothed
coordinates—fixed.

Under that within-run ablation, GAP2-minus-control adjusted/raw/node deltas are
`−0.000026127252/+0.000210275258/+0.000128194752` on held_fit105 and
`+0.000016582561/+0.000252577887/+0.000023352672` on CV57. The hard 44b6 family is
negative in both (`−0.000230778575` fit105, `−0.000244932863` CV57). Thus the raw-edge
benefit is causal in sign, but node inflation erases it in the adjusted objective and
the magnitude remains orders below the `+0.003` gate. This nuance does not change the
KILL/no-submit decision.

Receipts: `GAP2_SAME_RUN_ABLATION_HELD_FIT105.json` and
`GAP2_SAME_RUN_ABLATION_CV57.json`. The estimand is conditional on final smoothed
coordinates, not an exact rerun of the pre-GAP2 pipeline.

## Submission decision

Do not submit `output_v1/submission.csv`: its identities are the public train population,
so Kaggle's hidden scorer cannot use it. Do not build a hidden GAP2 package either: the
local objective gate failed. Immutable V32 remains the submitted public baseline (`0.911`).

Evidence: `TERMINAL_VALIDATION.json`, `CV57_SCORE.json`, and the retained `output_v1/`
directory beside this report.

## Appearance-selector follow-up

The owner-resumed `GAP-ID-TRANSPORT-v1` direction tested the mandatory ceiling before
extracting pixels. On held_fit105, the best exact label-informed selector over all
3,669 fixed GAP2 actions reaches minimum-family adjusted delta `+0.00053266`. A
nondeployable envelope that treats all 64 score-valid incident edges as TP, removes all
added FP/node cost, and forces node recall to one reaches only `+0.00078495`, still
3.82x below the `+0.003` gate. The fixed-action selector class is therefore DEAD; no
appearance features, Optuna, CV57, GPU, or kernel were run.

This leaves one precise reopening condition: a newly named mechanism must change the
graph action vocabulary before appearance/identity evidence is evaluated. Encoder or
threshold substitution over the same two-node bridges is not a new mechanism. Full
receipt: `../../../../../../../dag_prompts/2026-08-09-g02-three-way-alpha/GAP_ID_TRANSPORT_O0_20260809.md`.
