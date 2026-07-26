# G-02 Live State — shared source of truth

**Update protocol.** Single live status doc for the G-02 cell-tracking campaign, shared by
**Claude and Codex**. READ IT FIRST each session. UPDATE IT after any state change (a node
result, a gate flip, a DEAD promotion, a decision). Keep it SHORT — current state only;
push detail to the linked reports. Append a dated line to the Log (newest first). This doc
points at the authoritative artifacts; it does not replace them.

## Authoritative artifacts
- Plan / frontier: `campaign_narrowing_and_hypothesis_dag_v1.md` (human-locked 2026-07-25).
- Bet ledger (root coordinator is the sole writer): `LEDGER.md`.
- Scorer / CV57→public transfer gate: `p0_parity.md`.

## Current frontier (as of 2026-07-27)
- **H0 DONE** — proposal-only screening **BANNED** (no proxy `|ρ|≥0.5`). Every candidate now
  routes straight to the **44b6-first held-fit graph gate**; proposal metrics may be reported, never gate.
- **H1-ZWRFEAT SMOKE3 PASS / TIMING ONLY** — both frozen T4 witnesses and T4-to-T4
  repeatability passed. Projected 105-video runtime with 35% margin is **5.538884 h**,
  below the 7 h gate. No full cache, graph result, CV run, or submission exists.
- **NEXT** — `H1-ZWRFEAT-CACHE-PREP`: pin a full-cache contract revision, then stop at the
  human Option-C dataset-version/kernel-push boundary. The resulting cache stays
  quarantined pending human M5↔T4 tolerance approval, then runs the 44b6 graph gate first.
  Do **not** add `submission.csv`; public submission remains after both graph gates + CV57.

## The one live lever
Frozen **pretrained Trackastra + tiny calibrated head**, tested **44b6-first at the graph
level** (H1). Recall shell (H2) and mask-feature ablation (H3) follow *only if* H1 shows
graph-level signal. Everything else is DEAD (below).

## Key numbers
- Baseline local `0.873150` → public `0.877`. Banked APNCC local `0.881522` → **public `0.881`**.
- In-pool re-ranking oracle `~0.896` (projection). Transfer slope `≈0.48–0.66` (soft, rounding-bounded).
- **Accepted (public) 0.89 needs local `≈0.895–0.900`** ⇒ requires H1→44b6 + H2 recall, not
  in-pool re-ranking (DAG §3.1). Kaggle cap 30 T4-h/model. RunPod locked until accepted `≥0.89`.
- H1 external usage: two human-created dataset v1s; two failed timing-kernel versions and
  one passing smoke; **0.0822 kernel-h** (≤0.1644 allocated T4-GPU-h), **0 submissions**.
  All family graph deltas remain unmeasured.
  Package covers exactly **36 44b6 + 69 6bba**.
- H1-ZWRFEAT pins: contract `a7e866c3…9404`, corrected kernel `5e67eff7…a88f`, graph
  contract `d35e2d3d…7f48`; focused fail-closed verification **22/22 PASS**, swap 0 MiB.

## DEAD — do not re-propose without a new named mechanism (register: DAG §6.1)
Deterministic recall / shell abstention (`R12SHELL`, `MGSHELL`, `TEMPBG`) · from-scratch
rankers (`POINTCTX`, `CROPLINK`, `TEMPRANK`, `TEMPHYBRID`, `TRACKCTX`) · scalar recal /
threshold (`SCENCAL`, `POINTCTX-THRESH`, `APNCC-POLICY`) · hand-tuned graph-cost / motion
knobs (`GA-SP`, `LC-P4CTRL`) · full organizer `TemporalUNet3D` (~478 T4-h).
Conditional-PARK (revisitable): detection front-end · divisions · GT-conflict gates.

## Log (newest first)
- 2026-07-27 — H1-ZWRFEAT-SMOKE3 PASS: authenticated T4 timing artifact projects
  **5.538884 h** with margin, with both-family T4 repeatability PASS. The notebook is
  intentionally non-submittable; it produced no `submission.csv`, full cache, or graph
  result. Next is full-cache contract preparation, not a public probe. (Codex)
- 2026-07-26 — H1-ZWRFEAT-SMOKE2 STOP after registration cleared: launcher omitted the
  pinned fit105 manifest from `/kaggle/working`, so Stage A failed before model work.
  Corrected the authenticated copy path locally; repush remains human-executed. (Codex)
- 2026-07-26 — H1-ZWRFEAT-SMOKE1 STOP at source registration: fit105 dataset is ready v1,
  but pulled remote kernel metadata omitted that source although local metadata declared it;
  code hashes match. Kernel failed in 6.7 s before CUDA/model/timing work. Added an exact
  remote-registration verifier; retry remains human-executed. (Codex)
- 2026-07-26 — H1-ZWRFEAT READY at the human upload boundary: decoupled points-only
  zero-wrfeat contract, fit105-only two-stage package, exact create/version-verify/smoke
  commands, and 44b6-first graph runner banked; 18/18 focused tests passed after a 45 s
  zero-swap preflight. H3-MASK-SOURCE is conditional and unstarted. No Kaggle/RunPod
  mutation or graph run occurred. (Codex)
- 2026-07-26 — H1 STOP / RESERVE QUOTA at launch-integrity gate: authenticated fit105
  manifests contain no real H3 mask/wrfeat artifact; authenticated Kaggle listings contain
  neither H1 dataset nor kernel. No external mutation or downstream gate ran. (Codex)
- 2026-07-26 — Human explicitly authorized the narrowly scoped H1 Kaggle timing smoke,
  conditional frozen-cache launch, and one conditional public probe; all preregistered
  runtime, identity, H3, 44b6-first, 6bba, and CV57 gates remain mandatory. (Codex)
- 2026-07-26 — Reconciled `R12SHELL`/`MGSHELL` ledger cells PARK→DEAD; added DAG §3.1
  public-target arithmetic; created this live-state doc + `AGENTS.md`/`CLAUDE.md` wiring. (Claude)
- 2026-07-26 — Cascade committed on `codex/h0-h1-campaign-cascade-v1`: H0 (`a3de734`) BAN
  verdict; H1-PREP (`0ed7983`) launch-ready but stopped at the human gate. (Codex)
