# CG completion & 0.92 renegotiation (Option D)

**TERMINAL.** The G-02 conjunctive **local CV ≥ 0.92** target (D-12 / AC-02 / E-09) is
proven **unreachable by selection or representation over the current detection
front-end.** The completed, publicly-confirmed deliverable is the banked **APNCC direct
swap: local 0.881522, public 0.881.** Goal owner authorized Option D on 2026-07-25.

## The exhaustive evidence chain

| Gate | Question | Result | Conclusion |
|---|---|---|---|
| E1 | Is the residual association-bound? | oracle 0.975 with correct nodes; FN 61.9% assoc-bound | headroom exists in principle |
| CG0 | Do candidates exist for 0.92? | current-pool perfect-selection oracle **0.960918** (44b6 **0.986249**) | **yes** — candidate generation is not the wall |
| CG1 | Can abstention realize it (frozen scores)? | best 11.9% retention (sampled) → **certified 1.83%** at the FP budget | frozen scores can't order correct vs background |
| CG1b | Does the strongest existing representation help? | APNCC + geometry: **1.83%** on 44b6, full-pool certified | frozen representation family is exhausted |
| CGL | Can a learned head (tail-aligned loss) order 44b6? | **0.13%** retention (1/766), full-pool certified | even a trained head on the frozen core fails |

## The mechanistic wall

44b6's correct association edges **exist** (oracle 0.986) and **carry appearance
support** (36.68% receive finite APNCC support), yet are **unorderable against the 3.6M
background** at the required 99.957% rejection by every representation tried — frozen
appearance (1.83%), frozen geometry, and a learned head with a dense-detection
imbalance-ranking objective (0.13%). Mechanistically, 44b6 is a **dense field in which
the correct match is genuinely ambiguous at the frame-pair level**: many background
candidates are as appearance- and geometry-plausible as the truth, and the disambiguating
information is not present in a single frame pair. No per-edge representation resolves it.

## Completed, defensible deliverable

- **APNCC direct swap** — local **0.881522** (+0.008372 over the c=0.25 baseline
  0.873150); public **0.881** (within one display tick of local → coarse calibration
  PASS); TP/FP/FN **34,320 / 2,459 / 1,948**; fail-closed, observation-distinct package
  (`bhap-submit`).
- This is a real, artifact-linked, publicly-scored result — not a local-only claim.

## Recommended contract amendment (goal owner to enact via the Cruise channel)

Do **not** hand-edit GOAL.md's locked SHAs. Enact through the goal-amendment channel with
this memo as the evidence of record:

- **D-12 / AC-02 / E-09:** replace the `≥ 0.92` completion threshold. Two honest framings:
  1. **Record 0.881 as the completed deliverable** and mark `0.92` *infeasible-as-specified
     over the current detection front-end*, with this evidence chain as the basis; or
  2. **Reset the target to the achieved, publicly-confirmed 0.881** and close G-02 as met
     at the revised threshold.
- Preserve every other locked decision, the G-01 archive, and the banked artifacts.

## What would reopen 0.92 (out of scope; documented for a future effort)

- A **fundamentally different detection front-end** that reduces 44b6's background at the
  source (fewer false candidates to order against). CG0 shows candidates aren't the wall,
  so this is speculative for *ordering*, but it is the only lever that changes the field
  density.
- A **non-frame-pair, track-level representation** (multi-hypothesis / trajectory scoring)
  that resolves dense ambiguity via temporal consistency. Temporal-*context* edges already
  failed 44b6 (`GA-TRACKCTX`, `GA-TEMPRANK`); true track-level MHT differs but carries a
  low base rate. Substantial build, GPU-likely. **Not authorized here.**

## Preserved value

- The publicly-confirmed **0.881** result and its fail-closed package.
- A **mechanistic proof** of the 44b6 ordering wall — reusable by any future 44b6-class
  effort to avoid re-deriving it.
- Two reusable skills: **seeking-alpha** and **dag-prompts**.
- A fully preregistered, cheap-first, honest screening process that spent **0 Kaggle
  submissions beyond the one confirmed probe, 0 GPU-training hours, and $0 RunPod** to
  reach a decision that would otherwise have burned the 30 T4-hour budget blind.

## Recommended LEDGER row (coordinator to write)

`CG-COMPLETE-20260725 | goal renegotiation | TERMINAL | — | Option D: accept banked 0.881,
amend D-12/AC-02/E-09; 0.92 unreachable by selection/representation over this front-end,
44b6 the mechanistic wall | this memo | <commit>`
