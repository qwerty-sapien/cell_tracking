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
- **H1-ZWRFEAT N1 DEAD — STRUCTURAL M5↔T4 DRIFT.** The full cache completed in
  **7,325.321653 s** and all 105 feature/base payloads plus 210 atomic manifests
  authenticated (36 44b6 + 69 6bba). Both family T4 repeats are exact. The locked
  cross-device gate failed: `geometry_confidences` and `geometry_probabilities`
  differ from the banked M5 reference (max `8.9406967e-7`), and eight float/neural
  columns are outside the preregistered `atol=1e-7, rtol=1e-5` probe.
- **NEXT / STOP** — do not run N2, 6bba, CV57, or build a submission kernel. No
  graph/local/public score exists. H3 remains closed because no 44b6 graph failure
  occurred. Revisiting parity requires a new human-locked node/contract; it is not
  an in-node tolerance approval.

## The one live lever
No lever is executable under the current DAG. Frozen **pretrained Trackastra + tiny
calibrated head** remains the mechanism of interest, but its zero-wrfeat cache is
quarantined by N1 structural cross-device drift. H2 still requires graph-level signal;
H3 still requires an actual 44b6 graph failure.

## Key numbers
- Baseline local `0.873150` → public `0.877`. Banked APNCC local `0.881522` → **public `0.881`**.
- In-pool re-ranking oracle `~0.896` (projection). Transfer slope `≈0.48–0.66` (soft, rounding-bounded).
- **Accepted (public) 0.89 needs local `≈0.895–0.900`** ⇒ requires H1→44b6 + H2 recall, not
  in-pool re-ranking (DAG §3.1). Kaggle cap 30 T4-h/model. RunPod locked until accepted `≥0.89`.
- H1 external usage: two human-created datasets; two failed timing-kernel versions, one
  passing smoke, and one completed full-cache kernel. Banked completed use is
  **2.1170 kernel-h** (≤**4.2340 allocated T4-GPU-h**), **0 submissions**. All family
  graph deltas and proposal precisions remain unmeasured.
  Package covers exactly **36 44b6 + 69 6bba**.
- H1-ZWRFEAT full-cache pins: contract `4da06d48…34b`, kernel `28a3801a…537`, source
  dependencies `a252e8ef…9e9`, graph contract `d35e2d3d…7f48`; focused verification
  **22/22 PASS**, staged v2 assets 55 files, swap 0 MiB.
- N1 pins: run manifest `548bbac4…8d10`, feature report `d52a4a42…c0cf`,
  cache attestation `e7c0ca58…ce3b`, tolerance evidence `8264bb9a…e621`;
  attestation preflight **45.152 s**, swap 0 MiB.

## DEAD — do not re-propose without a new named mechanism (register: DAG §6.1)
Deterministic recall / shell abstention (`R12SHELL`, `MGSHELL`, `TEMPBG`) · from-scratch
rankers (`POINTCTX`, `CROPLINK`, `TEMPRANK`, `TEMPHYBRID`, `TRACKCTX`) · scalar recal /
threshold (`SCENCAL`, `POINTCTX-THRESH`, `APNCC-POLICY`) · hand-tuned graph-cost / motion
knobs (`GA-SP`, `LC-P4CTRL`) · full organizer `TemporalUNet3D` (~478 T4-h).
Conditional-PARK (revisitable): detection front-end · divisions · GT-conflict gates.

## Log (newest first)
- 2026-07-27 — H1-ZWRFEAT-CACHE-ATTEST **DEAD**: full cache/atomic integrity and
  two-family T4 repeatability passed, but exact-only M5↔T4 geometry drift and eight
  out-of-probe float/neural columns triggered `STOP_EXACT_GEOMETRY_OR_IDENTITY_DRIFT`.
  N2/CV57/submission remain closed; H3 is not eligible. (Codex)
- 2026-07-27 — H1 full-cache monitor CONTINUE: remote kernel `RUNNING`; Stage A reached
  82/105 at 5,088 s with steady warm progress and projected total ~2.3–2.5 h. No abort
  or 7 h leeway is warranted now; recorded the conditional 7 h stop rule. (Codex)
- 2026-07-27 — H1-ZWRFEAT-CACHE-PREP package ready: timing evidence, full-cache
  contract, source overlay, kernel hashes, and a five-node auto-cascade were pinned.
  Per `dag-prompts`, halted before the human Kaggle dataset-version/GPU gate. (Codex)
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
