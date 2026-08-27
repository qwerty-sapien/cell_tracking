# Codex screening campaign — cell tracking

This folder contains the G-02 screening and transfer-calibration sub-DAG. The active
objective is a materially stronger baseline: first exceed 0.88, then pursue robust
performance in the competitive 0.90s, with 0.92 retained as the completion threshold.

## How to run it

1. Load `_shared_context.md`, then `DAG.md`.
2. `P1_framework_and_rubric.md` is the controller and runs first. The root coordinator
   alone writes `LEDGER.md`, shared gate state, and integrated calibration artifacts.
3. Confirm `P0A` scorer semantic identity before making local score claims.
4. Run transfer calibration (`P0B/CAL`) in parallel with existing-evidence consolidation
   and local candidate development.
5. Treat every Kaggle score as a rounded interval and every submitted calibration probe
   as part of the submission budget.
6. Submit probes sequentially: at most one newly approved, observation-distinct probe at
   a time, update the design from its result, and never exceed five submissions per
   competition day. Keep both final selections uncommitted.
7. Cap initial training at 30 cumulative Kaggle T4 GPU-hours per model and archive
   models that cannot become viable inside it. RunPod is unavailable before an accepted
   `0.89` score; later use needs >5% literature/theory-backed expected improvement, fits
   the `$10` total cap, and requires exact-launch approval. The ordinary hard improvement
   gate is 3.5%.
8. Prefer literature-validated lightweight candidates. Validate their manifest before
   implementation/training, then use the single guarded fit-only M5 lane for rapid
   falsification. It may run in parallel with eligible external work but is capped at
   18 GiB training memory, reserves 6 GiB for background work, forbids swap growth and
   holdout access, and stops within seven hours per unattended stage.

## Governing distinctions

- Scorer-code/semantic parity determines whether local scores are valid.
- Local-to-Kaggle transfer calibration determines whether those scores support public
  forecasting and final selection.
- Transfer uncertainty does not freeze free local decomposition, error taxonomy, graph
  diagnostics, model development, or mechanism-specific GPU gates.
- Detector/front-end, association scorer, graph cost/solver, lineage postprocessor, and
  learned architecture are separate mechanism labels. Do not call all structural changes
  different “model families.”

## Node index

| ID | File | Purpose |
|---|---|---|
| CTRL/P1 | `P1_framework_and_rubric.md` | Authority, rubric, ledger, identities, and gate controller. |
| P0A | `P0_metric_parity.md` | Attest official scorer semantics. |
| P0B/CAL | `P0R_recalibration_wide_dispersion.md` | Observation-aware, interval-aware transfer calibration and public-probe design. |
| B1/B2/E1 | Existing prompts plus prior reports | Consolidate division, decomposition, and error-mechanism evidence first. |
| B3/G1 | Existing prompts | Local scenario handling and generalization analysis. |
| T1 | `T1_training_opt.md` | Association-model optimization behind GA. |
| M1 | `M1_alt_model.md` | Detection or association model spike behind its matching gate. |
| S1 | `S1_submission_portfolio.md` | Submission accounting and final two-candidate hedge. |

Ordinary commits and pushes to this private repository are permitted. Kaggle submissions
and RunPod launches still require explicit human approval.
