# Arm A — 44b6 barrier report

**Verdict: FAIL_PARK_C.** The preregistered geometry/motion arm enumerated
5,803,785 parent/birth pairs over all 21 authenticated 44b6 graphs, retained
26 fixed-rule candidates, and selected 12 after its one-per-video and occupied
node suppression. The prediction artifact was sealed before target access at
SHA-256
`55fdd22e8f171e2afa243791b90433f9ebf4ab8d9cf9ec3722f9ffc6f14c4108`.
Zero duplicate forks survived.

Against the exact 38-event target's 44b6 subset, TP/FP/FN are `0/12/9` and
Jaccard is `0`. Under the official frozen 44b6 scorer denominator, division
TP/FP/FN are `0/1/11`, Jaccard `0`. The score deltas are:

- adjusted edge Jaccard: `-0.000141522816`;
- raw edge Jaccard: `-0.000140538120`;
- node recall: `0`; and
- exact combined score: `-0.000141522816`.

The arm fails the positive-combined, `>=0.30` division-Jaccard, and both
edge-no-regression checks. Its node check and zero-duplicate check pass.

The fixed realized-FP sensitivity remains: 25 TP with 0/5/10 FP gives
`+0.0595/+0.0532/+0.0481` division-only combined addition. This arm's realized
precision is zero, so none of those positive-TP scenarios describes its
outcome.

Prediction and evaluation each followed a fresh 45-second zero-swap preflight.
The successful evaluator used 745.2 MiB peak RSS with zero swap growth. The
first evaluation attempt stopped before scoring because system Python lacked
the already-pinned `zarr` dependency; that packaging-only receipt is retained.
The retry used the repository `.venv` and did not change predictions, rules,
N1, target, or scorer.

No 6bba, full CV57, submission, Kaggle, GPU, RunPod, full-pool cache, or DEAD
action occurred. The campaign remains at the human 44b6 barrier.
