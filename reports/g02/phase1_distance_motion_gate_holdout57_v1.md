# Phase 1 E1/E2 — exact CV57 distance and motion gate

## Verified scope

- Exact divisions-OFF baseline: adjusted edge Jaccard `0.8711885823490648`.
- Official edge totals: TP `34174`, FP `2736`, FN `2094`.
- Newly derived raw-geometry cohort: `1549` eligible FNs = `1232` at <=10 um + `317` at >10 um.
- All official GT temporal edges: `36268`; rank-observable eligible edges: `35465`; candidate-absent/unrankable: `803`.

## E1 — distance rank

**PASS** against the stated joint >=90% rank<=10 rule over every official GT edge.

- All official GT edges: rank<=5 `35383/36268` (97.5598%); rank<=10 `35445/36268` (97.7308%); candidate-absent/unrankable `803`.
- Eligible-only secondary distribution: rank<=5 `35383/35465` (99.7688%); rank<=10 `35445/35465` (99.9436%).
- Above-10 um baseline FNs: rank<=5 `238/317` (75.0789%); rank<=10 `297/317` (93.6909%).
- Decision: PASS: replace the metric-only gate with a reviewed top-k standalone change.

## E2 — motion-predicted re-centring

**FAIL** under the explicit primary definition of large fraction = >=50%.

- Predicted distance <=10 um: `42/317` (13.2492%).
- Motion history present: `296/317`.
- Actually reachable by an unmatched-source-only pass: `72/317` sources; <=10 um after prediction `14/72` (19.4444%), equal to `4.4164%` of the full above-gate cohort.
- Decision: FAIL at the explicit primary >=50% convention for 'large fraction'; do not implement re-gating.

## Interpretation boundary

These are diagnostic gates only. No production top-k gate or motion re-gating was implemented. FN membership uses patched official matching on the exact baseline fragments; distances/ranks use raw cached detector coordinates, not final smoothed CSV coordinates.
