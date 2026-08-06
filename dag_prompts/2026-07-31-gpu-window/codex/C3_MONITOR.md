# C3 — terminal monitoring, hard kill, and debit accounting (autonomous)

**Precondition: `C2` pushed one version and holds the serial lock.**
Monitoring is pre-authorized (`AGENTS.md`). Run it to terminal without asking.

## Poll loop

- Interval: `60 s`. Do not poll faster; `kernel_status` is a real API call and adds nothing
  at higher frequency.
- Each poll writes a receipt. Keep them; the timeline is evidence.
- Terminal states: `COMPLETE`, `ERROR`, `CANCELLED`. Anything else continues.

## Hard runtime kill — preregistered, non-negotiable

| Field | Value |
|---|---:|
| Sealed hard wall-clock kill | `4 h 55 min` (`4.9166666667` h) for the 3-fold design |
| Under H0 Option A | scale per `PREREGISTRATION_AMENDMENT_A.json`; **state the exact number in the receipt before the run starts** |

Measure elapsed from **your recorded push time**, not from Kaggle's reported start.

On trip:
1. Stop the kernel.
2. **Quarantine every partial artifact.** No verdict, no aggregate, no per-fold claim from a
   truncated run.
3. **Do not restart into the remaining quota.** A resource trip is not a new signature; a
   restart is a same-signature retry.
4. Record the trip, the elapsed time, and the debit.

`nonfinite_or_missing_fold: FAIL_CLOSED` is preregistered. A partial fold set is a
`FAIL_CLOSED`, not a partial result.

## On terminal

1. **Fetch output** — `kernel_output` into a lane-local directory. Bank the pushed code, the
   step-0 receipt, the terminal payload, and the logs, mirroring the existing
   `kaggle_gpu_bakeoff_v1/terminal_evidence/` layout.
2. **Quota post-query**, owner-authenticated. **The pre/post delta is the debit.** Not your
   elapsed time, not Kaggle's reported runtime. This is how the campaign already reconciled a
   `1.65` h accounting drift and a `0.74` h race.
3. **Update `kaggle_budget.json`** — append the debit entry with a stable `id`, the
   `accounting_basis: "owner-authenticated Kaggle pre/post quota difference"`, and the status
   string. Close the authorization (`active_authorizations: []`).
4. **Release the serial lock.**

## Adjudicating the science

Apply the amended kill line exactly. All clauses bind **simultaneously**:

```
aggregate_adjusted_edge_delta_min        +0.005
aggregate_raw_edge_delta_min              0.0
aggregate_node_recall_delta_min           0.0
each_outer_fold_adjusted_edge_delta_min   0.0   (all FIVE folds under Option A)
information_leakage_exact                 0
nonfinite_or_missing_fold                 FAIL_CLOSED
duplicate_survivors                       0
failure_action  PARK-C this exact PEFT realization; no 6bba, CV57,
                submission, or graph continuation
```

A positive aggregate with one negative outer fold is a **FAIL**. Division arm B posted
`+0.000029258` and still failed on its other binding clause; hold the same line here.

Report AUC and CG0-budget retention as **secondary diagnostics** (delete reference: AUC
`0.85`, retention `0.02039355992844365`, FP budget `261`). The gate is the graph delta.

## The caveat you must carry

Option A's five hash folds are **not naive** — the supplier separation gate already used
them, including for the DINOv2 and C2 `FAIL_PARK_C` results on this same population. This
must appear in the terminal report and in `LIVE_STATE.md`. The result may **not** be
presented as a clean-room nested-OOF estimate.

## Expected outcome, stated in advance

Two information-disjoint suppliers converged at DELETE AUC `0.670412` / `0.670235` — a
`0.000176` gap — against a `0.85` requirement. The campaign reads that as a label-side
ceiling. **A FAIL here is the most likely result and is a real, publishable outcome**: it
closes §6.2, the plan's last named escalation, by experiment rather than inference. Write it
that way. Do not soften it, and do not propose a variant to "rescue" it — that would be a new
mechanism needing its own preregistration and human review.

## Write-up

`sa_peft_adapter_v1/TERMINAL_REPORT.md` and `TERMINAL_RESULT.json`: verdict token, every
fold×seed metric, both guard receipt hashes, the authenticated pre/post quota delta, the
leakage receipt, duplicate-survivor counts, adapter parameter count and exact trainable
ratio, runtime, the hard-kill number, and the Option A caveat.

Then append the dated line to `LIVE_STATE.md` — Current frontier and Key numbers refreshed,
newest first, short.

## Done when

The run is terminal, output banked, quota reconciled and the debit recorded, the lock
released, the verdict adjudicated against all clauses, and `LIVE_STATE.md` updated.
On **PASS**, `K5` (independent replay) opens — not `C4`. On **FAIL**, PARK-C and stop.
