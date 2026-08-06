# Parallel Codex dispatch — expiring-quota window, 2026-08-01

> **STATUS: PARK-B / NOT DISPATCHED (owner decision, 2026-08-01).** The owner parked both
> lanes without executing either. Nothing in this file is authorized; the §0 amendment was
> never granted and no quota was spent. Reopen only when the owner elects to expand the
> RunPod budget, at which point the T4-h arithmetic that motivated the lane caps no longer
> applies and both lanes must be re-costed and re-preregistered before use. Do not execute
> this file as written.

**Purpose.** Convert T4-hours that will otherwise expire at the next Kaggle refresh into
evidence that buys down the cost and the kill-risk of the only remaining route to the
adjusted **edge** Jaccard (≈91% of the combined score, `combined = adjusted_edge_jaccard +
0.1 * division_jaccard`, per `reports/g02/screening/p0_parity.md`).

**What this is not.** Not a G-02 reopening. Not an edit-selection (DELETE/SWAP/ADD)
experiment. Not a submission. Not a division experiment. Not a re-screen of any DEAD
mechanism. It does not touch the running G-02 native-base wave, its sealed fit selection,
or its one-shot CV57 authorization.

**Why the two lanes below and not something else.** Post-hoc graph editing is arithmetically
incapable of the required move. Banked is `0.881522069960`; `0.91` needs `+0.0285` adjusted
edge Jaccard. The *entire perfect-deletion oracle* on frozen 44b6 is `+0.081897667`, and
capturing any usable fraction of it was measured to require DELETE AUC `>=0.85` /
retention `>=0.020394` against a best-ever measured `0.677967` across DINOv2, C2, reverse
Trackastra, PEFT and raw-3D. Division is `FALSIFIED-BY-REALIZATION` at `0/38`. Therefore
only a **different base association model** can move the 91% term, and the only funded
candidate is `G03-TA-CROPADAPTER-E2E-V1` at 140–188 T4-h. Both lanes below reduce the cost
or the risk of exactly that.

---

## 0. Owner amendment being granted (must be recorded before any action)

This dispatch is void unless the owner has granted, and Codex has recorded verbatim in
`reports/g03/screening/LIVE_STATE.md` plus an `OWNER_AMENDMENT_20260801.json` receipt,
all five clauses:

1. **Concurrency.** The one-kernel serial lock is relaxed to **at most two additional
   concurrent GPU kernels** beyond G-03 Phase 1 A v2, for this window only. The shared
   quota bracket, prelaunch guard receipt, and per-lane serial push discipline still apply
   to each push individually. The lock reverts to serial at the window's end.
2. **Scope.** Lanes A and B below are authorized as **G-03 Phase 0/1 infrastructure and
   supplier-probe nodes**. They create no G-02 result and do not alter the G-02 archive.
3. **Graph reads.** Lane B may read the frozen **fit105 / 44b6 fit** GT and official graph
   score for the single preregistered configuration named in B1. No held-fold, 6bba, CV57,
   or public read is authorized in either lane.
4. **Spend.** Hard cap **6 T4-h per lane**, hard wall-clock kill at **5 h per kernel**.
   Unspent quota is allowed to expire; there is no instruction to consume it.
5. **No downstream unlock.** Nothing here authorizes Phase 2 training, a CV57 measurement,
   a dataset publication beyond the ones named, a submission, a final-slot change, or
   RunPod. A PASS produces a written request, not an action.

Absent any clause, emit `HUMAN-DECISION` and halt. Do not infer authorization from the
existence of quota.

---

## 1. Budget reality — state this back before launching

Quota is billed by **session wall-clock**, so remaining quota is not spendable faster than
concurrency allows. Before the first push, Codex must record, from an authoritative
authenticated query (not inference from wall time):

- exact `used / remaining` T4-h and the exact refresh timestamp;
- the account's actual concurrent-GPU-session limit, **measured or documented — not
  assumed**;
- the resulting hard ceiling: `min(remaining, hours_to_refresh × concurrent_limit)`.

If that ceiling is below the sum of the lane caps, cut Lane B first and say so. Do not
launch a kernel whose useful work cannot complete before the refresh.

---

## 2. Lane A — `G03-CROPBANK-CALIB-V1` (primary; launch first)

**Question.** Is the selected `G03-TA-CROPADAPTER-E2E-V1` architecture actually executable
on a T4 at the pinned contract, and what is its true measured cost?

**Run this as a kill test, not as phase 1 of a campaign.** The owner has *not* committed to
funding G-03's 140–188 T4-h envelope, and on the campaign's own gate structure
(`>=+0.005` adjusted delta at 90 T4-h) a full success lands near `0.8865` — well short of
the `0.91` / `0.9255` targets. The purpose of this lane is therefore to find a cheap,
defensible reason **not** to spend 140 hours. A measured `NO-GO` against the dossier's
200 T4-h invalidation clause is the **most valuable** outcome and must be reported as a
success, not a setback. Do not soften, re-scope, or re-measure a NO-GO into a GO. This lane
banks zero training progress by design; it retires part of the 16–20 T4-h proof line item
only.

**Why it is worth expiring quota.** `reports/g03/screening/ARCHITECTURE_DOSSIER_20260731.md`
names three conditions that *invalidate the architecture selection*: a missing raw source,
a 12-channel contract drift, and a measured worst-case projection above 200 T4-h. It also
charges the crop adapter a **conservative unmeasured 1.5× factor**. Discovering any of
these after Phase 2 begins costs 90–120 T4-h. Discovering them tonight costs quota that
disappears anyway. This is the definition of work worth parking until compute is free.

**Explicitly not.** No training toward the objective, no checkpoint that may later be
selected, no 44b6 graph read, no loss or AUC reported as a result.

**Measurements (all on the T4, all from the lazy-mounted competition source, no upload):**

1. Raw-volume reachability and exact array geometry for all 105 fit videos, authenticated
   by content hash, inside the step-0 envelope.
2. Crop extraction throughput at the pinned `(3, 16, 32, 32)` `t,z,y,x` contract with fixed
   boundary padding, over a fixed-seed sample of detections spanning **all 105 videos**
   (report crops/sec, and whether extraction is decode-bound or compute-bound).
3. Adapter + pinned Trackastra forward/backward: examples/sec, peak VRAM at the pinned
   batch schedule, and whether the pinned batch fits at all on 16 GB.
4. `state.pt` size, atomic handoff write + SHA-verify time, and the per-handoff overhead as
   a fraction of a 5-hour session.
5. **Crop-supply arithmetic, recorded explicitly.** `2,732,827` detections × `49,152`
   voxels is ~134 GB at uint8 and ~265 GB at float16 — a precomputed full bank is very
   likely infeasible as a Kaggle input. Decide and record, from measurement, whether crops
   must be extracted on-the-fly every epoch, and fold that cost into the projection.

**Deliverable.** `reports/g03/screening/CROPBANK_CALIBRATION_20260801.json` + a short
terminal report, containing every measurement above, a recomputed 140–188 T4-h envelope
from measured rates, and an explicit **GO / NO-GO against the dossier's 200 T4-h
invalidation clause**. A NO-GO is a full result and must be reported as one — it would save
the campaign more than a PASS.

**Fail-closed.** This lane dies terminally, with no retry, on any of: step-0 mount
resolution failure (the `KFG-15/16` family that killed DINOv2 R5 v1–v3 — use the evidenced
`/kaggle/input/...` mount form and the mandated prelaunch guard receipt), 12-channel
contract drift, or breach of the 5-hour wall kill.

**Cap.** 6 T4-h / 5 h wall.

---

## 3. Lane B — `G03-DENSEEMB-ASSOC-PROBE-V1` (conditional; higher risk, higher upside)

**Question.** Does a **dense-embedding** association supplier produce a better base graph
than APNCC-DIRECT on the frozen fit105 44b6 control?

**Why this is a new named mechanism and not a re-screen.** The G-02 terminal reopening
clause governs the *pointwise beneficial-vs-harmful edit* estimand; this lane does not
propose an edit at all — it regenerates the association graph. It is also not
`POINTCTX/CROPLINK/TEMPRANK` ("from-scratch rankers without pretraining or domain
adaptation"), because it uses a pretrained dense-feature association model. The
architecture dossier's own primary sources support it: Trackastra's authors expose a newer
model using learned SAM2 features, and Dalinghaus, Archit & Pape (MIDL 2026) report that
dense embeddings from self-supervision and foundation models **improve cell linking
performance** — independent evidence on the linking task, not another endpoint-appearance
encoder bolted onto edit selection.

### B0 — provenance gate (CPU, hard 45-minute box, no GPU)

Authenticate, by content hash, a released dense-embedding association checkpoint: source
revision, license SHA, weight byte size and SHA, and a source-published inference
configuration that fits one T4 at 16 GB. This lane has an unforgiving precedent —
`division/N0` found `0/>=1` qualifying releases, and CELLECT, DPRC and MCC each died on
exactly this step. **`0` qualifying releases, or the box expiring, is terminal: no launch,
no retry, no substitute checkpoint.** Report the negative as a result.

### B1 — one graph probe (GPU, only if B0 passes)

Preregister and hash-pin **before the push**, with no post-hoc adjustment:

- exactly **one** configuration — no sweep, no seed grid, no tuning, no second variant;
- scope: frozen **fit105 44b6** only, full coverage (no video subsampling — `0/12` cells
  were usable in the fidelity barrier);
- endpoint: the **exact frozen official scorer**, compared to the APNCC fit105 control
  `0.860516437125` combined/adjusted, `0.867176587961` raw, `0.988149479570` node recall;
- **kill line:** adjusted delta `<= 0` → PARK-C, terminal, no retry;
- **survive:** adjusted delta `> 0` with nonnegative raw-edge and node-recall deltas;
- **earn a written CV57 request:** adjusted delta `>= +0.005` on the same terms. Even then
  the lane **stops and asks**; it does not run CV57, 6bba, or a submission.

Run the marginal-value gate (`reports/g02/screening/MARGINAL_VALUE_GATE.md`) in writing
before the push: state the *realistic* precedent-adjusted ceiling, not the optimistic one.

**Cap.** 6 T4-h / 5 h wall.

---

## 4. Standing controls (unchanged, and they override everything above)

- Read `CLAUDE.md`, `reports/g02/screening/LIVE_STATE.md`, and
  `reports/g03/screening/LIVE_STATE.md` first. Their terminal and no-retry rules win.
- Do not touch the running G-02 native fit105 scoring, its sealed selection, or its
  one-shot CV57 authorization. Do not read its scores to inform anything here.
- Do not disturb G-03 Phase 1 A v2. If A v2 is still `RUNNING`, it holds its own lane; if
  it fails, Phase 1 still halts per its charter and **no v3 is authorized** — that is
  independent of this dispatch and must not be worked around.
- Every push: authenticated pre-quota → prelaunch guard receipt → push → post-quota, under
  `scripts/cmux_ops.py`; credentials never leave cmux. A 403 is a token/scope problem, never
  a rules-acceptance problem.
- No RunPod. No submission. No final-slot change. No dataset publication beyond the ones
  named. No threshold edited after seeing a result.
- Only the root coordinator writes `LEDGER.md`; nodes propose rows.
- Update both `LIVE_STATE.md` files after any real state change, per their update protocol.

## 5. Timeline and hard reconcile

| T+ | action |
|---|---|
| 0:00–0:45 | Record owner amendment; authoritative quota + concurrency ceiling; hash-pin both preregistrations; guard receipts. No GPU. |
| 0:45 | Push Lane A. |
| 0:45–1:30 | Lane B0 provenance box (CPU, parallel). |
| 1:30 | Push Lane B1 **only** if B0 passed and the ceiling still permits it. |
| −0:30 before refresh | Hard reconcile: authoritative post-quota for every kernel, terminal reports for both lanes regardless of outcome, both `LIVE_STATE.md` files updated, LEDGER rows proposed. |

Report every lane honestly, including `NO-GO`, `PARK-C`, and `terminal / no result`. A
measured negative that saves 90–120 T4-h is a better outcome than a launched kernel with
nothing to say.
