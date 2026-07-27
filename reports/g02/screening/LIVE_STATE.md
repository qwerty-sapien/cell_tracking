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
- **GA-H1-ZWRFEAT-GRAPH DEAD — 44b6 GRAPH KILL.** The human-authorized parity
  repair supersedes N1's false M5↔T4 fail: T4 is canonical, both T4 witnesses are
  array-exact, and the one-time M5 comparison is **BENIGN_FP**. The existing
  105-video cache was not regenerated. On 44b6, adjusted edge Jaccard
  **−0.01114808**, raw edge Jaccard **−0.01105753**, and node recall
  **−0.00028790**; held proposal precision was report-only **0.956522** (242/253).
- **NEXT / STOP** — H1 zero-wrfeat is falsified at the graph level. **6bba and
  CV57 were not run.** Do not build a submission kernel. `H3-MASK-SOURCE` is
  eligible only as a separately authorized provenance node for authentic
  per-detection masks; pseudo-masks remain forbidden and H3 is not started.

## The one live lever
No lever is executable under the current DAG. Frozen **pretrained Trackastra + tiny
calibrated head** remains the mechanism of interest, but zero-wrfeat H1 is now
falsified at the 44b6 graph gate. H2 still requires graph-level signal;
H3-MASK-SOURCE is eligible only as a separate authentic-mask provenance node.

## Key numbers
- Baseline local `0.873150` → public `0.877`. Banked APNCC local `0.881522` → **public `0.881`**.
- In-pool re-ranking oracle `~0.896` (projection). Transfer slope `≈0.48–0.66` (soft, rounding-bounded).
- **Accepted (public) 0.89 needs local `≈0.895–0.900`**; zero-wrfeat H1 cannot
  supply it after the 44b6 graph kill. Kaggle cap 30 T4-h/model. RunPod locked
  until accepted `≥0.89`.
- H1 external usage: two human-created datasets; two failed timing-kernel versions, one
  passing smoke, and one completed full-cache kernel. Banked completed use is
  **2.1170 kernel-h** (≤**4.2340 allocated T4-GPU-h**), **0 submissions**. The
  44b6 adjusted/raw/node vector is
  `−0.01114808/−0.01105753/−0.00028790`; 6bba remains unrun.
  Package covers exactly **36 44b6 + 69 6bba**.
- H1-ZWRFEAT full-cache pins: contract `4da06d48…34b`, kernel `28a3801a…537`, source
  dependencies `a252e8ef…9e9`, graph contract `d35e2d3d…7f48`; focused verification
  **22/22 PASS**, staged v2 assets 55 files, swap 0 MiB.
- N1 pins: run manifest `548bbac4…8d10`, feature report `d52a4a42…c0cf`,
  cache attestation `e7c0ca58…ce3b`, tolerance evidence `8264bb9a…e621`;
  attestation preflight **45.152 s**, swap 0 MiB.
- Parity/graph pins: repair contract `f0beba3c…aa24`, rotated graph contract
  `09b67cdd…4fb`, coordinator result `94a3cd00…c9a`, 44b6 report
  `53011727…19d`, guard `3538f97e…a71`; runtime **2,092.348 s**, peak RSS
  **1.119 GiB**, swap growth **0 MiB**, and all 36 fragments distinct.

## DEAD — do not re-propose without a new named mechanism (register: DAG §6.1)
Deterministic recall / shell abstention (`R12SHELL`, `MGSHELL`, `TEMPBG`) · from-scratch
rankers (`POINTCTX`, `CROPLINK`, `TEMPRANK`, `TEMPHYBRID`, `TRACKCTX`) · scalar recal /
threshold (`SCENCAL`, `POINTCTX-THRESH`, `APNCC-POLICY`) · hand-tuned graph-cost / motion
knobs (`GA-SP`, `LC-P4CTRL`) · full organizer `TemporalUNet3D` (~478 T4-h).
Conditional-PARK (revisitable): detection front-end · divisions · GT-conflict gates.
Frozen Trackastra zero-wrfeat H1 is DEAD at the 44b6 graph gate. Only the distinct,
separately authorized authentic-mask `H3-MASK-SOURCE` provenance question remains eligible.

## Log (newest first)
- 2026-07-27 — `GA-H1-ZWRFEAT-GRAPH` **DEAD / 44b6 KILL**:
  Option-C parity repair classified M5↔T4 drift BENIGN_FP and accepted the
  existing exact-repeat T4 cache. The fit-only graph gate then regressed
  adjusted/raw/node by `−0.01114808/−0.01105753/−0.00028790`; 6bba/CV57
  stayed closed. No GPU, Kaggle, RunPod, submission, or forbidden-scope
  access occurred. (Codex)
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
