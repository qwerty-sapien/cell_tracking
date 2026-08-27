# Common-36 held-44b6 fidelity tables

Verdict: **SUBSAMPLING_UNSAFE**.

| k | scheme | median rho | sign agreement | max 95% width | usable |
|---:|---|---:|---:|---:|---|
| 3 | `simple_random` | 0.485714 | 0.682850 | 0.062611869 | NO |
| 3 | `residual_top10` | 0.485714 | 0.685950 | 0.058474723 | NO |
| 3 | `family_residual` | 0.485714 | 0.685783 | 0.057513626 | NO |
| 5 | `simple_random` | 0.600000 | 0.736083 | 0.044304594 | NO |
| 5 | `residual_top10` | 0.542857 | 0.717567 | 0.048024791 | NO |
| 5 | `family_residual` | 0.542857 | 0.714667 | 0.049040357 | NO |
| 8 | `simple_random` | 0.771429 | 0.777283 | 0.031277819 | NO |
| 8 | `residual_top10` | 0.771429 | 0.776333 | 0.031281439 | NO |
| 8 | `family_residual` | 0.771429 | 0.775533 | 0.031493854 | NO |
| 12 | `simple_random` | 0.828571 | 0.820683 | 0.023057323 | NO |
| 12 | `residual_top10` | 0.828571 | 0.813733 | 0.023017144 | NO |
| 12 | `family_residual` | 0.828571 | 0.814883 | 0.023308413 | NO |

All cells use six candidates, exactly 10,000 finite-population draws, the common 36-video held-44b6 universe, and a common CG0 reference. `family_residual` is a single-family degenerate form of `residual_top10` and cannot win a tie.

This is screening-only evidence. Every promotion still requires the full frozen 44b6-first graph gate and downstream campaign gates.
