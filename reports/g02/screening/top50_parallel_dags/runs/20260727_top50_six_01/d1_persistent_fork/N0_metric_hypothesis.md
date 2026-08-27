# D1 N0 — scorer-semantic hypothesis frozen

Date: 2026-07-27  
Verdict: **GO**

The complete machine-readable hypothesis card is
`N0_preregistration.json`. It was written before any D1 held graph outcome was
opened. D1 is a grounded, locally falsifiable **research bet**; D0 remains the
necessary beta.

## Fixed claim

Once one named association arm has improved continuity by at least `+0.005`
adjusted graph delta on both 44b6 and 6bba, with raw edge and node recall
nonnegative and more connected two-branch opportunities, D1 will decode direct
or one-frame-late forks from frozen Trackastra evidence. Both branches must
persist through the grandchild evidence window, and equivalent/overlapping events
are suppressed before graph emission.

The four arms are fixed as unchanged graph, D0 parental-softmax/greedy beta, D1
active, and D1 with persistence removed. The persistence ablation is the
mechanism-negative control; it may not be replaced after outcomes.

## Literature and specification boundary

- Trackastra (ECCV 2024) grounds pretrained division-aware association and
  parental softmax: <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf>.
- The pinned official scorer commit
  `075fc5f5a52d11077f9dc2b074644618f26939e2` and division file SHA-256
  `0635c386…60f4dc9` define direct, late, grandchild, and one-to-one behavior.
- The local 14-fixture replay is documented in
  `reports/g02/division_sandbox_and_holdout_decomposition_v1.md`.

Neither source validates the exact D1 decoder. Its alpha is the non-obvious
competition-specific synthesis of scorer equivalence, persistence, and
pre-emission duplicate suppression.

## Leakage and stopping

Five outer folds use `sha256(video identity)[0:8] modulo 5`; videos and lineages,
not edge rows, are the units. Proposal/motif/calibration metrics are diagnostic.
D1 is DEAD if division Jaccard is `<=0.15`, any adjusted/raw/node endpoint is
negative, or a duplicate survives. It is GO only at division Jaccard `>=0.30`
with positive combined score and D1 attribution beyond D0. No D1 graph may be
scored before the both-family association dependency passes.

## Proposed ledger row

`D1-PERSISTENT-FORK-N0 | scorer-semantic persistent fork | ALPHA / RESEARCH BET |
four arms + exact scorer/cache/fold pins frozen before outcomes | no graph score |
0 submissions | 0 GPU-h | GO | dependency and D1 kill lines preregistered |
N0_metric_hypothesis.md`
