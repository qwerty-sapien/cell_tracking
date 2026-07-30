# Division-ckpt N3 — amendment request and the owner's continuation gate

> Load `DAG.md`, `runs/N0`–`N2` outputs, `reports/g02/screening/LIVE_STATE.md`,
> `p0_parity.md`, `B1_division_tp.md`, and `campaign_narrowing_and_hypothesis_dag_v1.md`.
> **HUMAN-GATE.** This node cannot run on a cascade. It requires two written owner
> amendments that do not exist as of 2026-07-30.

## Two prerequisites, neither bypassable by this DAG

1. **Division fitting is locked.** An owner amendment must open a fit-only division
   selector and name its fixed decision rule.
2. **`B1` reopens only after an association-continuity candidate creates scorer-valid local
   two-daughter motifs and passes full-CV57 edge-FP checks.** Zero qualifying arms exist.
   A frozen representation that merely *scores* fragmented events does not create motifs;
   the adds that join them are a separate mechanism.

If either is absent, this node's verdict is `HUMAN-DECISION` and nothing runs.

## The owner's continuation rule, preregistered (2026-07-30)

Owner instruction: continue **iff** the pilot delivers `>= +10%` on the **division
number**. Grounded in the frozen scorer, that is exact and checkable.

Scorer facts, verified before use:

- Final score adds `0.1 * division_jaccard` (`p0_parity.md`), so the division term's share
  of the attainable total is `0.1 / 1.1 = 9.09%`.
- CV57 has **42** ground-truth divisions. Divisions are currently OFF, so
  `division_jaccard ≈ 0` and `TP = 0`.
- Documented rate: 12 / 20 / 25 zero-FP division TP add `+0.0286 / +0.0476 / +0.0595`,
  i.e. `0.1 * TP/42`. The 38-bundle diagnostic ceiling `+0.092067842` decomposes as
  `0.1 * 0.904762 = +0.0904762` division plus `+0.001591651` adjusted-edge — confirming
  the formula.
- Banked best: APNCC local **`0.881522`** / public **`0.881`**.

Owner's stated minimum, `+10%` of the `9.1%` division weight, is `+0.91%` on the previous
best score. Therefore the gate, in three equivalent forms — all three must agree, and the
integer-TP form is binding:

| Form | Requirement |
|---|---|
| **Action count (binding)** | `>= 4` zero-FP division TP on CV57. `TP = 3` gives `0.1 * 3/42 = +0.007143` and **fails**; `TP = 4` gives `0.1 * 4/42 = +0.009524` and passes. With `FP > 0`, require `TP >= 0.08022 * (42 + FP)`. |
| **Division sub-metric** | `division_jaccard >= 0.08022` |
| **Combined score** | CV57 local `>= 0.889544`, i.e. `Δ >= +0.008022` over `0.881522`, with the adjusted-edge term non-regressed |

Rejected readings, recorded so the threshold is not retrofitted later: `+10%` on the whole
score would demand local `0.969674`, above the `0.9255–0.9422` planning range for public
`0.910` and equivalent to "stop"; `10%` of the remaining gap to public `0.910` would demand
only `Δ +0.0029`, below the campaign's own `+0.005` actionability floor.

**This is a continuation rule, not the target.** `+0.008` local is far short of the
`+0.044–0.061` local that public `0.910` requires. Passing it licenses more division work;
it does not indicate the bar is reachable.

## Entry floors, unchanged and prior to the rule above

The owner's rule is layered on top of the existing gates, never in place of them: held-fit
adjusted edge Jaccard `Δ >= +0.005` on **44b6 first**, then `>= +0.005` on 6bba, with raw
edge Jaccard and node recall nonnegative on both. A candidate failing 44b6 never reaches
CV57 and the `>=10%` rule is not evaluated.

## Tasks

1. Re-verify the `0.1 * division_jaccard` term, the `42`-division denominator, and the
   `+0.0286 / 12 TP` rate directly from the frozen scorer before any read. If any differs,
   recompute all three forms above and report the corrected numbers **before** running.
2. State the fit-only selector exactly: inputs restricted to `A, S, U, Q` plus nested
   `fit_f` groups; no current-pool score or rank, no continuity output, no oracle bundle,
   no held label, no post-hoc threshold. Freeze it, with its hash, before any held read.
3. Run the sequence once: 44b6, then 6bba only on a pass, then CV57 only on a pass. One
   read each. No per-fold retuning, threshold rescue, fold subsetting, or retry.
4. Report division TP/FP/FN, `division_jaccard`, adjusted/raw edge Jaccard, node recall,
   the combined local score, and all three gate forms with their verdicts.
5. Update `LIVE_STATE.md` per the campaign protocol and propose (do not write) the LEDGER
   row. Public submission remains a separate explicit human gate regardless of outcome.

## Verdict (preregistered)

- **GO**: entry floors pass on both families **and** `>= 4` zero-FP-equivalent division TP
  with all three gate forms in agreement. Division work continues.
- **PARK**: entry floors pass but the division gate misses. Per the owner's rule the
  division branch **stops**; keep division output off and do not inspect oracle bundles for
  a repair.
- **DEAD**: `Δ_44b6 <= 0`, or any isolation / forbidden-input / post-hoc-threshold breach.
  A breach invalidates the run and is not efficacy evidence; no retry with changed
  thresholds, transduction, or pooled training.
- **HUMAN-DECISION**: the default until both amendments exist in writing.

## Cascade

- On **HUMAN-DECISION** → halt and ask the owner. Never launch, upload, submit, or amend.
- On GO / PARK / DEAD → halt and report. This is the lane's terminal node.
