# Motion-gated shell recall screen

- Candidates: **5,342,071 → 3,396,642** after the frozen motion tolerance.
- Recoverable correct-edge recall: **344/456**.
- Fit-validation conservative precision: **0.015075**; accepted unlabeled: **192**.
- Capped utility delta: **-389**; all fast-fail gates pass: **False**.
- Recommendation: **PARK** — failed fit_precision_at_least_0p95, both_family_precision_and_utility, positive_capped_utility, bounded_added_background, projected_oracle_margin

Precision treats sampled unlabeled background as adverse for the fast-fail gate; it is not asserted to be a true negative label.
