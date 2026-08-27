# B2 decomposition result

Decision: **BANK diagnostic**

On the banked `c=0.25` baseline (`0.8731495935`, TP/FP/FN
`34,205/2,685/2,063`):

- the worst video carries 12.02% of weighted residual;
- the worst 10 carry **52.00%**;
- 24/57 carry 80.48%;
- `44b6` is harder per weighted event (0.854173 versus 0.876647);
- `6bba` carries 82.11% of total residual through greater exposure.

The existing covariate audit identifies crowding/nearest-neighbor spacing and the tight
6 µm unmatched fraction as the useful screen selectors. Dataset prefix and depth are not
supported as primary routes.

Use the crowded/high-motion slice for the next association-cost screen. See
`reports/g02/screening/evidence_consolidation_v1.md` for the identity caveats and source
artifacts.
