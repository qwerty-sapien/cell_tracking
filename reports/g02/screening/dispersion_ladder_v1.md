# P0B dispersion ladder v1

Decision: **BANK the six-rung observation ladder and the measured
APNCC-direct public anchor.**

P0A scorer semantics and P0B coarse high-range alignment are PASS.
APNCC-direct measured `0.881` publicly from `0.881522` local. Fine
local-to-Kaggle transfer remains **STILL-UNRESOLVED** for near-tie ranking.

No Kaggle submission, Kaggle GPU, RunPod, or external action occurred.

## Frozen rung table

All six rungs are exact full-CV57 fragment identities. TP/FP/FN use the attested
57-video scorer.

| Rung | Mechanism | Local CV57 | TP / FP / FN | Public evidence or planning band | Why it adds leverage |
|---|---|---:|---:|---|---|
| `c0` | graph cost/solver | 0.848394516 | 33,787 / 3,324 / 2,481 | measured `0.860`, interval `[0.8595, 0.8605)` | Existing distant low anchor; establishes broad-range slope but is structurally remote from the operating point. |
| `c2` | graph cost/solver | 0.872149947 | 34,178 / 2,699 / 2,090 | measured `0.878`, interval `[0.8775, 0.8785)` | Existing near-operating anchor and one side of the observed rank reversal. |
| `c0p25` | graph cost/solver | 0.873149594 | 34,205 / 2,685 / 2,063 | measured `0.877`, interval `[0.8765, 0.8775)` | Existing banked calibration baseline and the other side of the reversal. |
| `p4_no_attention` | learned association architecture/control | 0.874290046 | 34,227 / 2,658 / 2,041 | descriptive planning band `[0.876, 0.880]` | Cached mechanism-diverse anchor. It is not a reason to rebuild the dead P4 attention layer. |
| `apncc_global` | appearance association postprocessor + global solver | 0.877400878 | 34,281 / 2,584 / 1,987 | descriptive planning band `[0.878, 0.883]` | Appearance-mechanism bridge and solver control for APNCC-direct. |
| `apncc_direct` | appearance association postprocessor + exact local direct solver | **0.881522070** | **34,320 / 2,459 / 1,948** | **measured `0.881`, interval `[0.8805, 0.8815)`** | First local and public rung above 0.88; confirms coarse high-range alignment. |

The three unmeasured bands are deliberately broad planning intervals, not
decision-grade forecasts. Their centers use the current descriptive affine
mapping; their widths include the observed near-baseline reversal, Kaggle's
0.001 display bin, and extra extrapolation uncertainty above the measured
range. No affine line intersects all four public display bins.

## Pairwise observation vectors

The machine-readable source is
`dispersion_ladder_observation_matrix_v1.json`. It hash-validates every source
report and fragment tree, then applies `compare_submissions.py` one video pair
at a time. All 15 pairs are `DISTINCT`.

Each cell is `node Jaccard / raw edge Jaccard`. Scorer-eligible edge Jaccard is
identical to raw edge Jaccard for these predictions. Structural fork agreement
is 1.0 for every pair because all six emit no scorer-valid forks. The values
remain a named vector; they are not collapsed into an arbitrary scalar.

| A \\ B | `c0` | `c2` | `c0p25` | `p4_no_attention` | `apncc_global` | `apncc_direct` |
|---|---:|---:|---:|---:|---:|---:|
| `c0` | — | .961746 / .868372 | .963231 / .873289 | .963041 / .871987 | .963026 / .873198 | .960172 / .868541 |
| `c2` |  | — | .996059 / .989838 | .990444 / .973713 | .993386 / .982180 | .989977 / .976926 |
| `c0p25` |  |  | — | .991023 / .974946 | .996152 / .989217 | .992972 / .984736 |
| `p4_no_attention` |  |  |  | — | .990483 / .973725 | .986573 / .966958 |
| `apncc_global` |  |  |  |  | — | .994074 / .988688 |
| `apncc_direct` |  |  |  |  |  | — |

This resolves the original identity concern:

- The `c=2`/`c=0.25` public reversal occurred between predictions that differ
  on about 0.4% of nodes and 1.0% of edges. It is not a resubmission artifact.
- APNCC-direct versus `c=0.25` differs on about 0.7% of nodes and 1.5% of
  edges, so a public APNCC measurement is observationally informative.
- APNCC-global versus APNCC-direct differs on about 0.6% of nodes and 1.1% of
  edges, providing a useful solver-control branch if direct-swap transfer is
  surprising.

## Matched-pair admissibility

The current local resolution floor is the repository heuristic `0.004`. Exact
semantic equality alone is strictly vacuous; nonzero comparisons are described
by their observation vector.

| Pair | `abs(delta local)` | Node J | Edge J | Observation status | Admissible claim |
|---|---:|---:|---:|---|---|
| `c2` vs `c0p25` | 0.000999646 | .996059 | .989838 | DISTINCT | Admissible historical graph-cost interaction; the displayed public ordering reverses. It does not establish a family law. |
| `lb16` vs `c0` | 0.000492674 | .977009 | .923152 | DISTINCT | Not vacuous. Admissible learned-bonus/graph-cost interaction, but low-range and below APNCC submission priority. |
| `lb4` vs `c2` | 0.000058901 | .981162 | .945084 | DISTINCT | Not vacuous. Admissible learned-bonus/graph-cost interaction, but it adds no `>0.88` leverage. |
| `c0p25` vs `p4_no_attention` | 0.001140452 | .991023 | .974946 | DISTINCT | Mechanism-diverse matched-local anchor, but a single public pair would show only an interaction, not general family dependence. |
| `p4_no_attention` vs `apncc_global` | 0.003110832 | .990483 | .973725 | DISTINCT | Mechanism-diverse matched-local anchor; unmeasured and secondary to the high-range candidate. |
| `apncc_global` vs `apncc_direct` | 0.004121192 | .994074 | .988688 | DISTINCT | Just outside the current matched-local floor; useful as a score-separated solver comparison rather than a matched pair. |

The compact learned-bonus evidence is frozen in
`p0_lb_matched_pair_observation_v1.json`. Neither prepared pair is vacuous, but
both remain the same broad classical association/graph system and sit below the
unconstrained score range. They should not consume a submission before
APNCC-direct.

## Adaptive submission order

Previously measured rungs `c0`, `c2`, and `c0p25` consume no new slot.

1. **APNCC-direct — measured and BANKED at public `0.881`.**
   - The fail-closed output completed 4/4 test videos and its streamed
     submission hash matches `run.json`.
   - The archive was inspected in place and not expanded.
2. **Conserve the next slot for accuracy.**
   - Require an observation-distinct candidate targeting local CV57 at least
     `0.89`.
   - Submit one at a time only after explicit approval and update calibration
     immediately.
3. **Do not submit `lb4` or `lb16`.** They are observationally distinct,
   but their same-system, sub-0.88 information gain is dominated by
   APNCC-direct under the five-submission/day cap and performance objective.

While P0B is unresolved, submit at most one genuinely new configuration per
competition day, never exceed five submissions/day, feed the result back into
interval-aware calibration immediately, and preserve both final selections.

## Gate state

- P0A scorer semantic parity: **PASS**.
- P0B coarse high-range alignment: **PASS**.
- P0B fine transfer calibration: **STILL-UNRESOLVED**.
- Q next probe: **none; await a local `>=0.89` candidate**.
- S1 final selection: locked.
- Local candidate development: continues independently.
