# GPU window 2026-07-31 → 2026-08-01 — scope, budget, and standing rules

**Read `reports/g02/screening/LIVE_STATE.md` first. This pack does not replace it.**

## The window

| Field | Value | Source |
|---|---:|---|
| Authoritative used / remaining T4-h | `5.02` / `24.98` | owner-authenticated `kaggle quota --format json`, `2026-07-31T03:56:43Z` |
| Weekly total | `30.0` | same |
| Quota `refresh_at` | `2026-08-01T00:00:00` | same |
| RunPod | **FORBIDDEN** | owner governance |
| Public submission | **EXPLICIT HUMAN APPROVAL** | owner governance |

Unused hours are lost at refresh, and a fresh 30 h opens immediately after. **Expiry is
not a reason to launch.** `MARGINAL_VALUE_GATE.md` and §6.3's own caveat both say the
same thing: sunk free quota does not justify an un-preregistered spend.

## ⚠ The campaign was declared terminal on 2026-07-31

While this pack was being written, the root coordinator froze
`reports/g02/screening/TERMINAL_CAMPAIGN_REPORT.md` (`9b0a7819…1a961`):
**CAMPAIGN CLOSED; no active experimental wave.** `LIVE_STATE.md`'s current frontier now
leads with it. Its reopening condition is narrow and explicit:

> 1. a **new action-labeling formulation** that changes the estimand — for example, batch
>    or sequential action valuation that assigns value in the context of the global edit
>    set; or
> 2. a representation with **independent evidence of AUC `>=0.85`** on this exact frozen
>    beneficial-versus-harmful action discrimination task.
>
> "Another endpoint-appearance encoder, architectural substitution, upload-path repair, or
> uncalibrated proxy does not satisfy the condition."

**This closes the PEFT lane too.** A LoRA adapter over the frozen Trackastra backbone is an
architectural substitution fit against the *same* pointwise label; it satisfies neither
clause. See [`LANE_ADJUDICATION.md`](LANE_ADJUDICATION.md) for why the evidence — not just
the governance — now points the same way.

The disposition is an agent-level coordinator decision under the human-locked plan. **The
owner can reopen it.** Nodes `H0`..`K6` below remain fully specified for exactly that case,
and require an explicit written override to start.

### ✅ Status 2026-07-31: the owner overrode, and chose H0 Option A

Both blocking gates are resolved — see [`H0_HUMAN_FOLD_DECISION.md`](H0_HUMAN_FOLD_DECISION.md).
The override is scoped to `sa_peft_adapter_v1` alone. The amendment receipt is
`sa_peft_adapter_v1/PREREGISTRATION_AMENDMENT_A.json` (pre-outcome; sealed preregistration
byte-unchanged; carries a permanent caveat that the five hash folds are not naive to the
campaign). Budget for the lane is **`8.0` T4-h projected**, above the `5` h auto-ceiling and
covered by that decision.

`K0` is **complete** — the Trackastra checkpoint is now at `assets/trackastra-authentic-v1/`
with both hashes verified. **`K1` is the next node.**

## The binding constraint is not compute

Honest accounting for this window: **~0 T4-h of the 24.98 is spendable under the campaign's
current disposition**, and ~5.4–8.6 T4-h if the owner overrides the terminal report to run
PEFT anyway. Every lane is terminal, PARK-C, or needs a new named mechanism. Do not invent
work to fill the window.

The one direction that satisfies the reopening condition on its merits is a **new
action-labeling formulation** — and it is design and CPU work, not GPU work. See
[`R0_REFORMULATION.md`](R0_REFORMULATION.md). That is what next week's quota should fund.

## The override lane

`sa_peft_adapter_v1` — the campaign plan's own §6.2 escalation (frozen backbone +
`<1%` adapter, selected on the 44b6 held-fit graph delta). Before the terminal report it was
the only node that was (a) preregistered, (b) never attempted, and (c) not a repeat of a
mechanism class that had already failed structurally.

It carries a **second, independent** blocker — **`PRELAUNCH_HARD_STOP_INVALID_EXPERIMENT`**
(`sa_peft_adapter_v1/PRELAUNCH_AUDIT_20260731.json`). The blocker is an evaluation-envelope
defect, not compute:

- The frozen DELETE population (41 videos, `1,345,497` actions, `1,118` beneficial) is the
  **fit side of grouped outer fold 0**. It contains grouped folds 1 and 2 and **zero**
  grouped-fold-0 videos.
- The preregistration pins `grouped_3fold_audit20_v1.json` as the outer split and sets
  `each_outer_fold_adjusted_edge_delta_min` plus `nonfinite_or_missing_fold: FAIL_CLOSED`.
- Three outer-fold results are therefore unobtainable from this population.

`required_human_decision.automatic_choice_forbidden` is `true`. **No node in this pack may
resolve it.** It is gated at [`H0_HUMAN_FOLD_DECISION.md`](H0_HUMAN_FOLD_DECISION.md).

So the PEFT lane needs **two** owner decisions before it can start: an override of the
terminal report, and the `H0` fold contract.

## Node order

```
OVERRIDE  owner reopens the terminal surface   BLOCKING, zero compute
H0  human fold decision            BLOCKING, zero compute
K0  asset rescue                   zero GPU, do immediately, independent of H0
K1  implementation + local tests    zero GPU, CPU-only, needs H0
K2  packaging + guard receipts      zero GPU, needs K0+K1
K3  calibration launch              ~0.3-0.6 GPU-h, needs K2
K4  full run                        <=4.8 or <=8.0 GPU-h per H0, needs K3
K5  44b6 graph gate                 zero GPU, needs K4 PASS
K6  submission gate                 HUMAN, needs K5 PASS
```

**Autonomous cmux execution of `K2`/`K3` and monitoring:** [`codex/`](codex/) — a Codex-facing
pack (`C0`..`C4`) that turns the owner-authenticated Kaggle session into a bounded,
receipted, serially-locked instrument. `C0` builds the ops module; `C1`..`C3` run the lane
unattended; `C4` evaluates the submission predicate. Read `codex/README.md` for the exact
authorization surface.

Aligned with the reopening condition, **CPU-first, no override needed to start designing**:
[`R0_REFORMULATION.md`](R0_REFORMULATION.md).

Optional, separately authorized, **not** part of the PEFT lane, and now doubly closed by the
terminal report: [`FILL_DPRC_O0.md`](FILL_DPRC_O0.md).

## Standing rules — every node, no exceptions

1. **Authentication.** Every Kaggle CLI `list`, `push`, `status`, `datasets`, or `submit`
   runs through **computer-use cmux**, where the owner's real session lives. The
   agent-shell token must never be used for those. A `403` is a token/scope problem, never
   "accept the competition rules".
2. **Guard receipt before every push.** Per `kaggle_prelaunch/RELIABILITY_RUNBOOK.md`:
   run `check_registry_freshness.py`, then `kaggle_prelaunch_guard.py --phase static`,
   then `--phase launch` bound to the exact kernel ID (`KFG-15`). Exit `0` and
   `"status": "PASS"` required. Cite the receipt path **and its SHA-256** in the node
   report. "The guard exists" is not evidence. **Do not fork the guard into a lane-local
   validator** — that is `KFG-18`, and it is how this campaign burned its week.
3. **Serial launches only.** One authorized run at a time. The
   `DINOV2-R5-SEPARATION-GPU-V1-V2-LAUNCH-RACE` cost `0.74` GPU-h to a concurrent push
   that created a byte-identical second version. Confirm zero in-flight G02 kernels
   before every push.
4. **Quota is re-queried immediately before and after every launch**, owner-authenticated.
   The local budget file is a derived mirror, never the authority. Record the pre/post
   delta as the debit.
5. **Same-signature retry is forbidden.** If a push fails with a signature already in
   `KAGGLE_FAILURE_REGISTRY.json`, stop. If the signature is new, add the `KFG-N` family
   **and** a regression fixture in the same change that fixes it, before retrying.
6. **Step 0 discipline.** Pure-stdlib, under 5 s, before any pip / import / model load /
   decode. It authenticates content by pinned hash and fails closed on missing or
   duplicate matches. Never recursively walk `/kaggle/input` — that traversal alone has
   burned three kernels (`KFG` lazy-mount family). Private datasets mount at
   `/kaggle/input/datasets/<owner>/<slug>`, not `/kaggle/input/<slug>`.
7. **Local runs** keep the 45-second clean idle preflight and the zero-swap rule.
8. **Preregistration is sealed.** Any change to a frozen contract must be recorded
   **before** the outcome is observed, as its own receipt, with a reason. Post-outcome
   edits are the failure this campaign exists to avoid.
9. **Scope.** `6bba`, CV57, graph continuation beyond the preregistered gate, submission,
   and RunPod stay closed unless a node here explicitly opens them.

## What "success" means in this window

A `PASS` at K5 unlocks **one supplier-class-specific 44b6 graph contract**. It is not a
promotion, not a submission, and not a tier move. A `FAIL` PARK-Cs this exact PEFT
realization. Both outcomes are worth the spend; neither is worth breaking a gate for.
