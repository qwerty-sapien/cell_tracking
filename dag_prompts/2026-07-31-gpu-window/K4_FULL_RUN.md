# K4 — full nested-OOF PEFT run (≤4.8 or ≤8.0 T4-h per `H0`)

**Precondition: `K3` projection at or under the ceiling, quota reconciled, calibration
outputs quarantined.**

## Budget and kill

| Field | Option B (3 folds) | Option A (5 folds) |
|---|---:|---:|
| Projected T4-h | `4.8` | `8.0` |
| Hard wall-clock kill | `4 h 55 min` (`4.9166666667` h) | scale per the `H0` amendment; state it explicitly |
| Auto-ceiling | `<5 h` — auto-opens after guards | `>=5 h` — covered by the `H0` owner decision |

Re-query the owner-authenticated quota immediately before launch. Launch capacity uses
**that** number, not the local mirror. At `5.02` used / `24.98` remaining, either option
fits; verify anyway — the balance moved twice on 2026-07-31 alone.

The hard kill is a wall-clock guillotine, not a target. If it trips, the run is terminal
with quarantined partials and **no scientific verdict**. Do not restart into the remaining
quota; that is a same-signature retry of a resource failure.

## The preregistered kill line — unchanged, and the whole point

```
aggregate_adjusted_edge_delta_min   +0.005
aggregate_raw_edge_delta_min         0.0
aggregate_node_recall_delta_min      0.0
each_outer_fold_adjusted_edge_delta_min  0.0
information_leakage_exact            0
nonfinite_or_missing_fold            FAIL_CLOSED
failure_action  PARK-C this exact PEFT realization; no 6bba, CV57, submission,
                or graph continuation
```

All of these bind simultaneously. A positive aggregate with one negative outer fold is a
**FAIL**. Arm B of the division wave posted `+0.000029258` and still failed on its other
binding clause — the same discipline applies here.

## Run discipline

1. **One push. Serial.** Confirm zero in-flight G02 kernels. Read back the actual assigned
   kernel ID and assert it matches the launch guard's binding.
2. **Per-fold atomic checkpointing.** The DINOv2 local run completed 77 sequential shards
   with atomic checkpoints and zero swap growth — reuse that shape. A mid-run guillotine
   must leave verifiable per-fold state, never a half-written aggregate.
3. **Emit, per outer fold and seed:** the adjusted-edge, raw-edge, and node-recall deltas
   from the **frozen scorer**, the compiled edit set through the **frozen N1 compiler**,
   duplicate-survivor count (must be `0`), the leakage receipt, and the adapter hash.
4. **Report AUC and CG0-budget retention as secondary.** They are diagnostics. The gate is
   the graph delta. (For reference, the delete floors that the frozen suppliers missed:
   AUC `0.85`, retention `0.02039355992844365`, FP budget `261`.)
5. **Aggregate exactly as preregistered** — average fold/seed predictions across the two
   fixed seeds; do not select the better seed.
6. **Post-run, via cmux:** re-query quota, record the pre/post delta as the debit, update
   `kaggle_budget.json`, and close the authorization.

## Do not

- Do not extend epochs, add seeds, widen the adapter, or change the LR because an interim
  number looks weak. Every field is sealed.
- Do not open `6bba` or CV57 regardless of outcome. They sit behind a separate human barrier.
- Do not build a submission. `K6` is a human gate.
- Do not report a verdict from a partial fold set. `FAIL_CLOSED` means exactly that.

## Reporting

Write `sa_peft_adapter_v1/TERMINAL_REPORT.md` and `TERMINAL_RESULT.json` with: the verdict
token, every fold×seed metric, both guard receipt hashes, the authenticated pre/post quota
delta, the leakage receipt, duplicate-survivor counts, adapter parameter count and ratio,
and the exact runtime. Then append the dated line to `LIVE_STATE.md` — Current frontier and
Key numbers refreshed, newest first, short. Detail stays in the terminal report.

If the verdict is **PASS**, it unlocks exactly one supplier-class-specific 44b6 graph
contract (`K5`). It is not a promotion, not a tier move, not a submission.
If **FAIL**, PARK-C this exact realization and stop. That is a real result: it closes the
plan's own last named escalation, and the campaign should be told so plainly.
