# Pre-claim compiler repair R2 — 2026-08-14

The first authorized exact-winner scoring process was interrupted before the
one-shot claim after `10:56:53` wall time. At the atomic pause boundary,
`WINNER_GRAPH_SCORE_CLAIM.json` and `LOCAL_WINNER_GRAPH_SCORE.json` were both
absent. The official scorer therefore remains uncalled and the sole score
authorization remains unconsumed.

A two-second host stack sample localized the overrun to
`compile_replace_component`: the historical batch compiler was validating,
materializing, serializing, and hashing the complete baseline and candidate
graph once for each of 41,273 selected actions. Those individual graph hashes
are discarded by the batch result. This is an infrastructure complexity defect,
not a scientific outcome.

R2 leaves the frozen winner, selected action IDs, scoring script, score
contract, preregistered gate, official scorer, and LMT unchanged. Its wrapper
replaces only batch compilation. It performs the same per-action schema,
closure, endpoint, and cross-action ownership checks against a precomputed
baseline edge index, preserves the exact action hashes and retained anchors,
then materializes and hashes the final graph once per video. Outcome-free
differential tests require equality of all observable batch fields against the
historical implementation, including order independence and retained-anchor
behavior.

No score result was observed or used to design this repair. Any failure before
the claim remains an infrastructure failure; no second official score is
permitted after a claim exists.
