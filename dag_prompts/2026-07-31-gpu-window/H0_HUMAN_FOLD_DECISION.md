# H0 — BLOCKING human decision: the PEFT outer-fold contract

## ✅ RESOLVED 2026-07-31 — the owner decided both gates

> **OVERRIDE:** the `TERMINAL_CAMPAIGN_REPORT.md` closure is overridden **for the
> `sa_peft_adapter_v1` node only**. It does not reopen the tiered surface generally, any
> DEAD-register item, any PARK-C supplier, `6bba`, CV57, submission, or RunPod.
>
> **H0 DECISION: OPTION A** — amend `sa_peft_adapter_v1` to the five frozen hash folds over
> the unchanged `1,345,497`-action DELETE population. Five-outer-fold kill clause replaces
> the three-fold clause. All other preregistered fields unchanged. Projected `8.0` T4-h
> approved, above the `5` h auto-ceiling.

The decision was taken with the adjudication against it disclosed first
(see [`LANE_ADJUDICATION.md`](LANE_ADJUDICATION.md), "The PEFT reassessment"): the
`0.670412` / `0.670235` convergence across two information-disjoint suppliers points at a
label-side ceiling, so the most likely purchase is a fourth independent measurement of
`~0.670`. The owner elected to buy that measurement and close §6.2 by experiment rather than
by inference.

**Amendment receipt (pre-outcome, sealed preregistration left byte-unchanged):**
`sa_peft_adapter_v1/PREREGISTRATION_AMENDMENT_A.json`. It carries the permanent honesty
caveat that the five hash folds were already used by the supplier separation gate and are
therefore not naive to the campaign. That caveat must appear in the terminal report and in
`LIVE_STATE.md`.

`K1` may now start. `K0` is already complete.

---

*Original decision brief, retained for the record:*

**No agent may resolve this node.** `PRELAUNCH_AUDIT_20260731.json` sets
`required_human_decision.automatic_choice_forbidden: true`. Nothing downstream (`K1`..`K6`)
may start until the owner records a choice here.

## The defect, exactly

The frozen PEFT preregistration
(`sa_peft_adapter_v1/PREREGISTRATION.json`, SHA-256
`0d96b336f7ec5c8a18881be30e04aeba4c500f273f6376ec05a5228f0fe71ad8`) pins:

- `data_and_selection.folds: 3` over `reports/g02/grouped_3fold_audit20_v1.json`
  (`ddb313cb…8769`);
- `kill_line.each_outer_fold_adjusted_edge_delta_min: 0.0`;
- `kill_line.nonfinite_or_missing_fold: FAIL_CLOSED`.

The frozen DELETE population
(`sa_repr/phase2_separation_gate/run_v1/population_manifest.json`, `4ab7bb23…76f0c`)
is the **fit side of grouped outer fold 0**:

| Grouped outer fold | Videos | Actions | Beneficial |
|---:|---:|---:|---:|
| 0 | **0** | **0** | **0** |
| 1 | 21 | 648,405 | 576 |
| 2 | 20 | 697,092 | 542 |

Three outer-fold results cannot exist. Any nested-OOF claim from this population against
this preregistration would be invalid.

The same population also carries five **deterministic hash-fold** labels with nonzero
support everywhere:

| Hash fold | Videos | Actions | Beneficial |
|---:|---:|---:|---:|
| 0 | 8 | 244,465 | 306 |
| 1 | 5 | 173,496 | 99 |
| 2 | 9 | 311,859 | 167 |
| 3 | 7 | 238,772 | 195 |
| 4 | 12 | 376,905 | 351 |

These are the labels the supplier separation gate used. The PEFT preregistration explicitly
pinned the *other* artifact. Swapping them silently would change the experiment after the
target surface was known — which is why the audit refused to do it.

## Option A — amend to the five frozen hash folds

Replace the outer split with the five hash folds over **exactly** the unchanged
`1,345,497`-action population, and replace the three-outer-fold clause with a
five-outer-fold clause. Everything else in the preregistration stays byte-identical.

- **Compute:** 5 outer folds × 2 seeds ≈ **`8.0` T4-h** (the `4.8` h projection was
  3 folds × 2 seeds). Crosses the `<5 h` auto-ceiling → needs this same owner decision to
  cover it, which it does.
- **Wall clock:** fits the window with margin.
- **Honesty cost:** *real but bounded.* The hash folds were frozen before this node existed
  and were never selected on a PEFT outcome — but they **were** used by the supplier
  separation gate, so they are not naive to the campaign. The amendment must state this
  in writing, pre-outcome, and the node report must carry the caveat permanently.
- **Risk:** the amendment is recorded after the *population's* outcome surface is known,
  though before any PEFT outcome exists. Defensible; not pristine.

## Option B — preserve the grouped three-fold gate

Keep the preregistration untouched and **materialize** the missing authenticated action
population plus graph-scoring inputs for grouped outer fold 0 (the 21 held videos), so all
three outer folds exist.

- **Compute:** population materialization is CPU-only (the existing delete population built
  in `286.858 s` at `1.081 GiB` RSS, `0 MiB` swap). Then PEFT at 3 folds × 2 seeds ≈
  **`4.8` T4-h**, inside the auto-ceiling.
- **Wall clock:** materialization + B0 sealing + re-verification is the risk. It is a new
  data contract, and it must be sealed and hash-pinned before any PEFT fitting. Realistic
  estimate is several hours of CPU and review before K3 can start — **this is the option
  that may not finish inside the window.**
- **Honesty cost:** none. The sealed preregistration is honored exactly as written.
- **Risk:** schedule, not validity.

## What each option does to the window

| | Option A | Option B |
|---|---|---|
| GPU-h consumed | ~8.6 (calib + run) | ~5.1 (calib + run) |
| Finishes before `2026-08-01T00:00:00`? | very likely | uncertain |
| Preregistration integrity | amended, pre-outcome, with a permanent caveat | untouched |
| If it fails to finish | partial run, quarantined, no verdict | no launch at all |

## Recording the decision

The owner writes one of the following into this file and into `LIVE_STATE.md`, dated:

```
H0 DECISION: OPTION A — amend sa_peft_adapter_v1 to the five frozen hash folds over the
unchanged 1,345,497-action DELETE population. Five-outer-fold kill clause replaces the
three-fold clause. All other preregistered fields unchanged. Projected 8.0 T4-h approved,
above the 5 h auto-ceiling.
```

```
H0 DECISION: OPTION B — preserve grouped_3fold_audit20_v1.json. Materialize and seal the
grouped-fold-0 action population and graph-scoring inputs before any PEFT fitting.
Projected 4.8 T4-h.
```

```
H0 DECISION: NO LAUNCH — let the window expire; revisit after the 2026-08-01 quota refresh.
```

**Under Option A**, the amendment receipt must be written as
`sa_peft_adapter_v1/PREREGISTRATION_AMENDMENT_A.json` **before K1 starts**, must record the
pre-outcome timestamp, the exact fields changed, the reason, and the hash-folds-were-used-by-
the-separation-gate caveat, and must leave `PREREGISTRATION.json` byte-unchanged.

**Under Option B**, the new population contract must be sealed and hash-pinned, and the
sealed B0 (`a9ce8c0b…b4ff2`) must be extended to the new fold with the same frozen procedure,
before K1 starts.
