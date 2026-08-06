# Lane 2 — authentic reverse supplier + literal endpoint consensus

Disposition: **TERMINAL FAIL — LITERAL CONSENSUS GRAPH NOT OPENED**

## What was and was not new

The authentic reverse supplier was already complete and terminal. It is genuine
reversed-time Trackastra inference over 36 fit-only 44b6 videos, 3,539 transitions,
84,883 proposals, and 121,534 comparison rows. For canonical `u_t -> v_t+1`, it
returns the separately inferred matrix entry `A_reverse[v,u]`. The checkpoint and
reverse manifest are hash-pinned, the repeat witness is array-identical, and no
forward-margin negation, matrix transpose, duplication, or recalibration was used.

The observation-distinct question was the literal abstention rule. The terminal
supplier screen had learned on reverse/cycle descriptors over the full population;
it had not tested a fixed "return to the same endpoints" rule. This audit fit and
ranked nothing.

## Frozen rule and support

The only candidate population was the exact 45,961 N1-valid occupied-endpoint
swaps. An action survived only when every exact edge it would remove satisfied:

- forward margin `>= 0.50`;
- authentic reverse margin `>= 0.50`;
- returned exact reverse pair minus returned no-link mass `>= 0.25`;
- agreeing forward/reverse signs.

The reverse comparison set also had to equal the action's exact removed-edge set.
Canonical identity/time/node order, exact endpoint ownership, two actions per video,
and a global cap of 64 replaced score ranking.

There were 268 literal-threshold candidates and **53 sealed edits across 29 videos**,
or `53/45,961 = 0.1153%` support. The disabled-reverse null selected zero. Frozen N1
accepted all 29 nonempty video batches: 53 swaps, 106 removed edges, no node-ownership
overlap, invariant node sets, and all edited graphs structurally distinct. No graph
score was read during this check.

## Label gate and exact graph disposition

After selection SHA `1ef8f3ed...e6c8` was frozen, only the 53 selected GT-derived
labels were opened. Result: **0 beneficial / 53 nonpositive**, with every individual
oracle delta exactly `0.0` (none negative, all neutral). Positive-delta precision is
therefore `0.0`, and the summed oracle delta is `0.0`.

The preregistered graph-opening gate required at least four beneficial edits and
zero nonpositive edits. It failed, so running the scorer would have violated the
preregistration. The deployed edit set is **0**, and the exact candidate is the
APNCC no-op:

| Vector | Adjusted | Raw | Node recall |
|---|---:|---:|---:|
| APNCC baseline | `0.8665345260361912` | `0.8625960574674240` | `0.9926180002844743` |
| Candidate | `0.8665345260361912` | `0.8625960574674240` | `0.9926180002844743` |
| Delta | **`0.0`** | **`0.0`** | **`0.0`** |

This is an exact identity vector, not an estimated graph result: the frozen gate
emits no edits, so scorer input remains the baseline byte-for-byte.

## Four required properties

- **Appearance-grounded: NO.** The authentic historical Trackastra realization uses
  twelve zero-filled `wrfeat` channels. It is learned bidirectional association
  evidence, but has no physical image-appearance input. The design's initial broad
  "appearance+motion" description is therefore corrected here.
- **Narrow: YES.** The sealed set was 53/45,961 (0.1153%), no ranking, hard cap 64;
  deployed support is zero.
- **Node-disjoint: YES.** Exact candidate and removed-edge endpoint ownership is
  disjoint; frozen N1 accepted every nonempty batch.
- **Recall-preserving: YES.** Only SWAP was eligible, with invariant node set and a
  replacement edge for occupied conflicts; the deployed no-op preserves APNCC.

## Downstream and scope

AE-P6, R8, and C3 are **not unblocked**. The fixed literal rule failed before the
graph, and the older full-population supplier failure remains terminal. There was no
reverse-inference rerun, learned selection, whole-pool reranking, 6bba, CV57, Kaggle,
GPU, submission, tier move, or shared `LIVE_STATE.md`/`LEDGER.md` write.
