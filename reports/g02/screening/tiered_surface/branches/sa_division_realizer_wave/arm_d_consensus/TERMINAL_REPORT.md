# Arm D — 44b6 barrier report

**Verdict: FAIL_PARK_C.** The preregistered 3-of-3 high-precision consensus
selected no forks. Arm A supplied 12 sealed predictions, Arm B supplied 3,873,
and Arm C supplied no prediction because its existing-cache prerequisite failed
closed. The frozen `missing_arm_vote = ABSTAIN` rule therefore makes every
candidate an abstention.

The consensus prediction was sealed before target access at SHA-256
`f0b3e4a8e0e534d2107ae8f1fa5bad5cf6428647ae56f8a22558d5bf3539aaa4`
with zero duplicate survivors. The exact 38-event target's 44b6 subset scored
TP/FP/FN `0/0/9`, Jaccard `0`. The official frozen 44b6 scorer denominator
scored division TP/FP/FN `0/0/11`, Jaccard `0`; adjusted-edge, raw-edge,
node-recall, and exact combined deltas are all exactly `0`.

The arm fails because combined delta is not positive and division Jaccard is
below `0.30`. It has no edge or node regression and zero duplicates.

The required realized-FP sensitivity remains:

| TP | FP | division-only combined addition |
|---:|---:|---:|
| 25 | 0 | +0.0595 |
| 25 | 5 | +0.0532 |
| 25 | 10 | +0.0481 |

Both 45-second local preflights passed with zero swap growth. The successful
evaluation used 443.8 MiB peak RSS and ended 8 MiB below its starting swap
level. A first evaluation attempt produced no result because Zarr 3.2.1 tried
to close an unopened empty-action `ZipStore`; its zero-swap failure receipt is
preserved. The retry changed only the empty-action context manager. Predictions,
the target, thresholds, suppression, N1, and the scorer were unchanged.

No 6bba, full CV57, submission, Kaggle, GPU, RunPod, full-pool cache, or DEAD
action occurred. The campaign remains at the human 44b6 barrier.
