# CGL — D-14 learned-representation fast-fail (Option A resume)

> Load `.agents/skills/seeking-alpha/SKILL.md`, `.agents/skills/dag-prompts/SKILL.md`,
> `DAG.md`, `cg_escalation_v1`, and `../../../.cruise/goal/GOAL.md` (D-01/D-03/D-14, T-03).
> The goal owner chose **Option A** at the CG escalation. This is the cheap, fit-only
> gate that decides whether the 30 T4-hour learned path is worth launching. It does **not**
> authorize GPU training — it is the proxy that must pass first.

## Why a fast-fail before GPU

CG1/CG1b proved *frozen* representations cannot order 44b6 correct edges above background
at the operating tail (1.83% retention). CG0 proved the correct edges are present
(oracle 0.961) and CG1b proved 36.68% of them carry finite appearance support — so the
wall is **ordering under a frozen family**, not signal or candidates. A *learned* head can
in principle reshape that ordering, but 44b6 has walled every prior learned attempt. So
before spending the 30 T4-hour/model budget, cheaply falsify the one thing that must be
true for the learned path to reach 0.92: **a compact head, trained with a tail-aligned
objective on the frozen core, lifts 44b6 held-fit tail retention materially above 1.83%.**

## The alpha node (why this is not the DEAD ranker)

Every DEAD from-scratch ranker (§6.1) trained on a **proxy** — proposal precision / BCE —
which optimizes *average* separation (AUC 0.918 was already high, yet the tail failed).
The needle-in-3.6M-haystack at 99.957% rejection is an **extreme foreground/background
imbalance ranking** problem. Dense object detection solved exactly this with
imbalance-aware ranking losses — **focal loss, AP-loss, DR-loss** — that concentrate
capacity at the precision tail instead of the average. That transfer is the alpha:

> Train the compact head to optimize **retention at the fixed FP budget** (a soft
> top-k / AP-style / DR-style objective, or a cost-sensitive loss weighting a
> near-operating-point background FP at ~1/budget) — *the objective's own definition*,
> not BCE. Ground it in the dense-detection imbalance-loss literature.

## Scope (fit-only, cheap, 44b6-first)

- **Representation:** frozen released `TemporalUNet3D` core (reuse the existing epoch-5/7
  fit-only checkpoint from `M1-ORGCKPT` where valid) + a **compact head/adapter** (D-01/
  D-03 compliant). Core stays frozen; only the small head trains.
- **Objective:** the tail-aligned loss above, not generic classification.
- **Data:** fit-train105 folds only; **validation12, CV57, holdout, audit, test all
  closed.** Train on fit-fold proposals, evaluate on held **fit** folds. This is a proxy,
  not a score claim.
- **Metric:** held-fit **44b6** known-correct retention at CG0's per-video FP budget,
  certified on the **full** per-video background (never a small sample — CG1's mistake).
  Report 6bba too, but 44b6 gates.

## Preregistered gate (write before running)

- **GO** — held-fit 44b6 retention ≥ **15%** at the FP budget (the CG1b PARTIAL bar, ~8×
  the frozen 1.83%), with 6bba not collapsing → the learned path is worth its GPU. Proceed
  to the main G-02 **T-03** under its own gates (D-14 manifest, 30 T4-h/model cap, T-04/
  T-06 promotion). **T-03 launch is a HUMAN-GATE.**
- **PARK/DEAD** — 44b6 stays near the frozen floor → a trained head does not break the
  44b6 ordering wall either → **renegotiate the endpoint (Option D)**: reopen D-12/AC-02
  with the banked ~0.881 as the honest completion.

## Compute & gates

- Frozen-core feature extraction over the fit-only subset is bounded; if it needs GPU it
  is a **HUMAN-GATE** (bounded, well under the 30-h cap) — the head train/eval is CPU/free.
- No validation12/CV57/holdout access; no competition submission; no RunPod; no full
  T-03 training in this node. Reject holdout/CV57 paths before reading.

### Runtime guard — 12 GiB retry amended 2026-07-26 (goal-owner authorized)

The original D-14 **zero-swap** guard tripped on host-background paging (attempts 2/3 grew
swap by only +38.75 / +8.57 MiB at 0.21–0.86 GiB RSS — host noise, not CGL thrashing), so
the fast-fail never produced a verdict. The first bounded amendment to 1 GiB stopped at
1.1222 GiB after 90/105 feature videos; the second, to 1.5 GiB, stopped at 1.5796 GiB
after 91/105. Third retry contract:

- **Process-tree RSS ceiling: ≤12 GiB.**
- **Swap-growth ceiling remains ≤1.5 GiB.** The ≤7 h timeout is unchanged.
- The verdict remains **contract-valid**: the fast-fail's outputs are deterministic with
  respect to swap (swap changes speed, not the numbers), so a bounded swap allowance does
  not compromise the D-14 result — it only relaxes a runtime-hygiene guard.
- **Record this amendment in the D-14 contract** (`reports/g02/model_candidate_cgl_tail_rank_head_v1.json`)
  with a new content SHA before re-running, so the run is contract-honest. If swap growth
  exceeds 1.5 GiB or RSS/timeout is breached, still fail closed and report.

## Output & cascade

`reports/g02/screening/candgen/cgl_d14_fastfail_v1.{md,json}` + LEDGER rows: the head
identity, the tail-aligned objective, held-fit 44b6/6bba retention at budget, full-pool
certification, and the verdict.

- **GO** → hand to the main G-02 learned path at **T-03** (HUMAN-GATE for the GPU launch).
- **PARK/DEAD** → `CG_ESCALATE_goal_feasibility.md` Option D (renegotiate 0.92).
