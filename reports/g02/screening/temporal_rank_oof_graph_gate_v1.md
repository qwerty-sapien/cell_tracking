# Held-fit surrogate-to-graph gate validation

Decision: **BANK the gate; PARK the temporal ranker**.

The already-PARKed `6bba` temporal family ranker was rerun as a retrospective
five-fold diagnostic on the 69 eligible `6bba` videos in fit-train105. Each
video's model weights, normalization, and operating threshold were fitted
without that video. The candidate and the banked APNCC control were then
reconstructed as complete graphs and scored through the attested official
scorer.

This is gate-validation evidence, not a new promotion test. The temporal
configuration had historically been selected using this fit pool, and its
CV57 failure was already known. Validation12 and CV57 were not opened by this
run.

| arm | fit69 adjusted edge Jaccard | raw edge Jaccard | node recall | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| exact banked APNCC control | 0.8593764824 | 0.8680442384 | 0.9858180775 | 54,863 | 4,647 | 3,693 |
| held-fit temporal ranker | 0.8587643913 | 0.8674034023 | 0.9862150426 | 54,865 | 4,696 | 3,691 |
| temporal minus APNCC | **-0.0006120910** | **-0.0006408361** | +0.0003969652 | +2 | +49 | -2 |

The proposal-only prerequisite reproduced exactly: 1,894 correct and 69
harmful accepted rescues, precision 0.9648497, conservative utility 1,756.
Nevertheless, exact graph realization lost to APNCC. Four of five graph folds
were negative; their adjusted deltas were `-0.000396`, `-0.000222`,
`+0.000059`, `-0.001288`, and `-0.001851`.

All 69 candidate fragments were observation-distinct from their controls.
Most comparisons used the same per-video proposal cap and therefore isolate
ranking quality; a few sparse videos also differed in accepted proposal count.
The aggregate loss came from 49 additional false-positive edges for only two
additional true-positive edges.

The preregistered graph gate required adjusted delta at least `+0.005`, no raw
edge-Jaccard regression, and no node-recall regression. It returned **PARK**,
correctly screening the temporal model that later lost `-0.001408` to APNCC
on CV57. This validates the new process rule:

> No learned association candidate may enter CV57 from proposal utility alone.
> It must first beat exact APNCC on leakage-safe held-fit reconstructed graphs.

Artifacts:

- report SHA-256: `470e4fdc37d037953eead6a46c59035e960122f807cce6472c15424207d8c764`
- guard SHA-256: `90226a407b59abc0ab0399fc4e1f2161db872b66b4ac4af9eafb8d21d71e3e04`
- contract SHA-256: `9e73a87c7427288d23d32715327eb0296a48b9dcf5f82e738fd2e100273b6e70`
- runtime: 3,604.8 seconds, 1.304 GiB peak process-tree RSS, zero swap growth,
  38% minimum observed system-free memory

No Kaggle submission, Kaggle GPU, RunPod, paid compute, or submission-budget
action occurred. APNCC `0.8815220699597315` remains the banked candidate.
