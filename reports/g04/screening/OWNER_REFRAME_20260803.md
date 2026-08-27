# G-04 owner reframe — division credit and computed segmentations

**Authority:** human owner, 2026-08-03.  This amendment supersedes campaign
conventions where they conflict with this document.  Historical measurements and
DEAD labels remain evidence; the conclusion that the achievable surface is exhausted
is withdrawn.

## Decisive observation

The banked APNCC candidate is local `0.881522069960` / public `0.881`, approximately
rank `1100/1800` and therefore below median.  The working hypothesis is now that a
standard competitive approach or configuration remains untested.  The historical
association-continuity dependency, 44b6-first filter, `+0.005` promotion threshold,
near-zero-FP division operating point, and local `0.89` submission floor are not
evidence about what is achievable and do not gate G-04.

## Exact objective and decision rule

The primary local objective is the frozen official combined score on CV57:

`S = 0.9 * adjusted_edge_jaccard + 0.1 * division_jaccard`.

The incumbent is `S0 = 0.881522069960`, with division counts `TP/FP/FN = 0/0/42`
and division Jaccard `0`.  Any observation-distinct candidate with exact local
`S > S0` is submission-eligible.  Lane winners are selected independently and may
each consume one submission; public outcomes are calibration data, not grounds for
retrofitting the local result.

Hard disqualifiers remain: frozen official scorer only; no GT-derived feature or
candidate generation at inference; no artifact identity/provenance ambiguity; no
computed segmentation may be represented as an organizer-provided or H3 authentic
mask; serial Kaggle launch/submission discipline and terminal monitoring; RunPod
still requires explicit owner approval.

## Parallel DAG

### Lane A — permissive division decoding

1. Recompute the exact combined-score curve over predicted-fork count.
2. Sweep permissive operating points, explicitly including substantial false-positive
   counts.
3. Report the full nondominated `(TP, FP, J, S)` frontier.
4. Submit the best observation-distinct candidate if `S > S0`.

Cheap falsifier: the frozen CV57 graph scorer itself.  Kill only a specific decoder or
ranking that cannot produce `S > S0`; do not infer that division as a class is dead
from a high-precision abstention result.

### Lane B — computed 3-D segmentation morphology

1. Authenticate a pretrained 3-D segmentation model, checkpoint SHA-256, source, and
   license.
2. Segment raw intensity Zarrs and label every result `COMPUTED_SEGMENTATIONS`.
3. Join detections to computed instances and derive an explicit 12-channel 3-D
   region/intensity feature vector.
4. Run Trackastra with nonzero computed `wrfeat` and divisions enabled on held-fit,
   then CV57.
5. Feed the same morphology to Lane A's fork ranker/decoder.
6. Submit the best observation-distinct Lane-B candidate if `S > S0`.

The pretrained segmenter is necessary beta.  The alpha node is the honest joint use
of computed 3-D morphology in both association/fork inference and exact graph-level
selection, an interface never evaluated in the campaign.

## Preregistered reporting

Every scored candidate reports adjusted/raw/node components, division
`TP/FP/FN/J`, combined `S`, delta from `S0`, exact population, graph identity, and
artifact hashes.  Negative held-fit or CV57 results prune only the named
configuration.  Submission is forbidden for byte-identical graphs or candidates not
strictly above `S0` locally.
