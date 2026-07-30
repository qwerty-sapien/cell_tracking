# SA-DIVISION terminal report

Date: 2026-07-28  
Branch: `codex/sa-division`  
Verdict: **PARK_DEPENDENCY**  
Tier proposal: none; retain `DIV-F1` at dependency-blocked `PARK_A`

## Terminal decision

No single frozen banked association-continuity arm passes both required families at
adjusted edge-Jaccard delta `>= +0.005` with raw edge and node-recall deltas
nonnegative. Qualifying-arm count is exactly **0**.

| Arm | Banked artifact @ SHA-256 | `44b6` adjusted/raw/node | `6bba` adjusted/raw/node |
|---|---|---|---|
| `GA-H1-ZWRFEAT-GRAPH/points-only-zero-wrfeat-12` | `reports/g02/screening/ga_h1_zero_wrfeat_graph/report.json` @ `94a3cd006cd40aeb45f1773f6e198767133c3724eed720f573b1e3acd6851c9a` | `-0.011148076581064648 / -0.011057532129446046 / -0.00028790090722474737` | not run (`null/null/null`) |
| `C1-TA-HOCT-LITE` | `reports/g02/screening/top50_parallel_dags/branches/c1_ta_hoct_lite/outputs/c1_n3_44b6_graph.json` @ `6a202b209c9c1b7851ca1e43e4c67e5dc3772e78426fdc267f1c01cfe5953a10` | `-0.007602934501881209 / -0.007591069106931747 / -0.00018539307341891575` | not run (`null/null/null`) |
| `U0-T4CANON` | `reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/u0_t4canon/terminal_report.md` @ `c29d52abd4b48c349818b0e761d8fd2b338da3e23b49164d31c86e11486f0151` | governance only (`null/null/null`) | `null/null/null` |
| `C2-KBEST-STABILITY` | `reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/c2_kbest_stability/terminal_report.md` @ `259edb515125bd5616c3df8fcc07c4aa93a35afed895fbb61151612378a0b0a6` | not scored (`null/null/null`) | `null/null/null` |

APNCC is the unchanged beta, not an incremental arm. Missing-family vectors were not
imputed, arms were not spliced, and the external memo's C2/K-best and Branch-B
`66/223` values stayed quarantined.

## Allowed fixture result

After the dependency PARK, only the preregistered fixture audit ran:

- frozen N1 interface, implementation, folds, P0, and fit-provenance pins: all match;
- explicit no-op after-graph SHA-256:
  `fe992a9ed9abd31f615b8832a90d5cb4a64869d7cfe3d0eac53b317cc4e7a235`,
  exactly equal to the baseline;
- atomic division-fork after-graph SHA-256:
  `a55dabfad74baca7b765959a317542aad8958d9a645052a496d20fb6f2305857`;
- complete conflict ownership removed `(2,4)`, `(11,12)`, `(11,13)` before adding
  `(11,4)`, `(11,5)`;
- omitted conflict, nonbinary fork, and missing endpoint fixtures failed closed;
- **7/7** local tests passed;
- immutable D1 official scorer evidence remained **14/14** classifications,
  **51/51** invariant counterfactuals, **0** duplicate survivors, byte-identical
  repeat `0b47ece998f99da56e435581bbe5c407b2faa75f9388d6651ab1781c51d753df`;
- D1 was not rerun or modified.

Evidence: `guarded_results.json` @
`38990d30d4b30ea7afbf99e0a655b0150c34a73fdd4918f93178316d9d9e3c76`.

## Guard and scope

The successful retry followed a fresh 45.638-second preflight and completed in
1.833 seconds at 0.075 GiB peak process-tree RSS with **0 MiB swap growth**.
The earlier test-harness comparison error failed closed with zero swap growth and
opened no scientific surface.

Division model fitting, held-family graph scoring, `44b6`, `6bba`, CV57, GPU,
Kaggle, submission, RunPod, external service/mutation, plan amendment, DEAD rescreen,
and sibling live-output reads were all **false / not performed**.

Resume only when the root coordinator supplies one banked arm path+SHA with exact
passing vectors on both families. This terminal artifact does not grant fitting
authority.

## Follow-on division-oracle refinement

The banked dependency was reasserted at exactly zero qualifying arms before
the P0 diagnostic was consumed. The frozen D1 arm identities, grammar, stable
order, thresholds, and dry-validation boundary remain unchanged.

The exact fit117-CG0 P0 `division_fork` diagnostic recomputed:

| Family | Combined `ΔS` | Adjusted / raw / node delta | Division Jaccard | D1 GO `>=0.30` | Minimum FP reduction at fixed TP/FN |
|---|---:|---:|---:|---:|---:|
| `44b6` | `+0.014723031693` | `+0.001389698359 / +0.001375521786 / 0` | `0.146666666667` | no | 39 |
| `6bba` | `+0.021405055770` | `+0.001271006215 / +0.001287100359 / 0` | `0.215277777778` | no | 82 |

The FP reductions are counterfactual envelope calculations only. The P0 action
is a banked fit117-CG0 oracle, not APNCC and not a forecast. Neither family
meets D1's unchanged division-Jaccard GO threshold, and diagnostics cannot
promote or reopen fitting.

The successful guarded retry passed 10/10 checks after a fresh 45.674-second
preflight, then ran for 0.337 seconds at 0.0045 GiB peak process-tree RSS with
zero swap growth. Two earlier attempts failed closed at expected-hash
authentication before P0 parsing; both had zero swap growth and no result.

Evidence: `oracle_refinement_results.json` @
`aded494fd19aca50a9123dfdbcccfd37448709330f8449e76984359833c44c22`.

Final verdict remains **PARK_DEPENDENCY** with no tier move. Model fitting,
held-family execution, CV57, GPU/Kaggle, submission, RunPod, external mutation,
plan amendment, and DEAD action remain false.
