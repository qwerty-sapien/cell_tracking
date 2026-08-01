# `GLOBAL-REASSIGN-ORACLE-V1` — Codex dispatch

**Commander:** Claude (root). **Executor:** Codex. **Surface:** G-02 edit vocabulary,
reopened narrowly under reopening clause #1. **Cost:** local CPU only. **GPU / Kaggle /
RunPod: forbidden, zero exceptions.**

---

## Pasteable trigger

```text
Read CLAUDE.md, reports/g02/screening/LIVE_STATE.md, reports/g03/screening/LIVE_STATE.md,
and reports/g02/screening/MARGINAL_VALUE_GATE.md first. Their terminal, no-retry, and
stop-loss rules override everything in the dispatch.

Execute dag_prompts/2026-08-01-global-reassign-oracle/CODEX_DISPATCH.md as a single
preregistered node. Hash-pin the preregistration BEFORE reading any oracle value.

This is CPU-only and local. No Kaggle asset, dataset, kernel, push, GPU debit, CPU-kernel,
submission, or RunPod action is authorized under any circumstance, including if it looks
faster. Do not modify the frozen official scorer or the frozen N1 feasibility compiler.

Emit the tiers in the preregistered order with the preregistered early-stop. Report
STOP/ESCALATE verdicts exactly as measured; do not soften a stop into a continue.

Write artifacts to dag_prompts/2026-08-01-global-reassign-oracle/runs/, propose a LEDGER
row (do not write LEDGER.md), and update reports/g02/screening/LIVE_STATE.md per its
update protocol.
```

---

## 0. Owner amendment (granted 2026-08-01, this session)

The owner reopens the G-02 edit surface **for one diagnostic node only**, under
`TERMINAL_CAMPAIGN_REPORT.md`'s reopening clause #1 — *"a new action-labeling formulation
that changes the estimand … batch or sequential action valuation that assigns value in the
context of the global edit set."* The compound/chained conflict-swap vocabulary is exactly
that formulation, and `MARGINAL_VALUE_GATE.md` §5 already names it as the sanctioned
escalation path rather than an iteration.

Granted: read frozen 44b6 fit GT and official graph scores for the tiers below; produce a
GT-dependent ceiling.
**Not granted:** any selector, training, representation, AUC, threshold, 6bba, CV57,
held-fold, public read, tier move, DEAD/PARK promotion, plan amendment, or compute spend
of any kind beyond the local CPU. A PASS produces a written proposal, never an action.

---

## 1. Why this node exists (state it back before running)

Two facts in the banked evidence have never been placed side by side:

- `extended_radius_reachability_v1.md`: the r10/k8 candidate tensor already contains
  **99.04%** (`73,999/74,718`) of clean one-to-one GT edges.
- `recall/runs/N0B_RECONCILIATION.md`: yet `626/673` (93.02%) of held-44b6 edge FNs are
  absent from the deployed action population — **not** because the edge is missing, but
  because `free-add` requires both endpoints unclaimed (realized oracle exactly `0`) and
  `occupied-swap` is node-disjoint. Quote: *a correct edge whose fix requires bumping a
  chain of other wrong edges is legally unreachable under this contract even though it
  sits inside the tensor.*

So every selector failure since 2026-07-25 — DINOv2, C2, PEFT, raw-3D, reverse Trackastra —
was a hunt for a **representation** on a surface whose binding constraint was the **action
vocabulary**. This node prices that constraint directly, before anyone funds another
selector.

**It measures a ceiling. It does not measure a gain.** Banking an oracle as realized
headroom is failure pattern #1 in `MARGINAL_VALUE_GATE.md`; the collapse discipline in §4
is mandatory, not advisory.

## 2. Frozen inputs (pin every SHA in the preregistration)

- Frozen 44b6 fit population, **all 36 videos, no subsampling** — the subsample fidelity
  barrier measured `0/12` usable cells; partial coverage invalidates the node.
- The banked APNCC-DIRECT 44b6 baseline graph behind local `0.881522069960`.
- The frozen candidate tensor (r10/k8) and frozen action population `4ab7bb23…76f0f`.
- The **exact frozen official scorer**, unmodified.
- The frozen N1 feasibility compiler (`interface 2a72950c…f865`, `implementation
  457f6607…24f5`) — read and honor its structural constraints; **do not edit it**.

## 3. Tiers, in this exact order

### T3 — free global reassignment (cheap upper bound; run FIRST)

GT-optimal legal subgraph selected from the frozen candidate tensor subject only to
structural validity (degree, time ordering, no-merge, one-to-many parental but not
many-to-one), ignoring the baseline graph entirely. Score with the official scorer under
the **adjusted** metric so node errors are charged.

**Preregistered early stop:** if `T3 − baseline < +0.030` adjusted, the entire
better-assignment-over-same-candidates family cannot reach the `0.91` requirement even at
oracle. Emit `STOP_CLASS`, skip T0/T1/T2, and terminate the node. This saves the expensive
search and is a complete, reportable result.

### T0, T1 — integrity controls (must reproduce, or the node is INVALID)

- T0 node-disjoint single-hop occupied-swap oracle → must reproduce `+0.018571397`.
- T1 delete-class oracle at prefix 2,048 → must reproduce `+0.081897667`.

Agreement to `1e-9`. A mismatch means the harness is not wired to the same objects as
`run_v6`; emit `INVALID_HARNESS` and stop. Do not "fix" it by adjusting a tolerance.

### T2 — chained conflict-swap (the question)

Install a correct edge by displacing the incumbent edge(s) at its endpoints and recursively
re-homing the displaced endpoints. The chain is **valued as one unit** — that is the
estimand change. Preserve every structural constraint from §2; forbid duplicate edges.

Report the oracle at **every** preregistered maximum chain length `L ∈ {2, 3, 4, unbounded}`.
Do not select an `L` after seeing results; all four are reported.

For each tier report: exact official **adjusted / raw / node / combined** deltas versus the
banked 44b6 baseline, correct edges installed, FPs removed, chains compiled, and the
chain-length histogram.

## 4. Decision arithmetic — preregistered, applied before any follow-on

The informative quantity is `T2(unbounded) − T0`: what a new compiler action-kind buys over
today's vocabulary.

Convert oracle to expectation using the campaign's own measured collapse ratio
`0.018571397 / 0.075828 = 24.49%`:

- **Continue** (propose a selector node) only if
  `0.2449 × (T2(unbounded) − T0) >= +0.005` adjusted.
- Otherwise emit `STOP_CLASS`: the compound/chained action vocabulary is closed at the
  **mechanism-class** level per `MARGINAL_VALUE_GATE.md` §4, and no second generator,
  parameterization, or representation may be funded against it.

State the realistic, precedent-adjusted number in the report **before** the raw one, so the
raw ceiling cannot be quoted out of context later.

## 5. Hard constraints

- **Local CPU only.** No Kaggle GPU or CPU kernel. The Kaggle CPU transport record on this
  campaign is `0-for-6` (SELECTOR-REQUIREMENT CPU v1/v2/v3, DINOv2 R5 CPU v1/v2/v3, all
  dead in step 0 for `0.30` CPU-h and zero science). The local machine has 10 cores /
  24 GiB and is idle; a Kaggle CPU session gives 4 vCPU and is strictly worse.
- 45-second clean-idle preflight; sustained swapout `>=256 MiB` aborts; per-video fragments
  written atomically so the run is resumable without recomputation.
- Wall-clock cap **6 h**. Precedent from this machine: delete oracle `286.858 s`,
  occupied-swap `2,221.571 s`, restore-topology `349.107 s`, all at ~1.0–1.1 GiB RSS.
  If T2 projects past the cap, emit `ESCALATE` with the measured projection — do not
  silently reduce coverage.
- **Do not modify the frozen N1 compiler.** Compute the oracle in a separate, clearly
  labeled analysis harness, and report whether the frozen compiler could ever compile the
  winning chains. If chained actions are inexpressible in it, that is `ESCALATE` **and a
  finding** — it confirms the compiler is the binding constraint.
- Label every number `[GT-DEPENDENT]`. No AUC, no selector, no training, no representation.
- No re-screen of any DEAD mechanism; no reopening of any terminal/no-retry route.
- Only the root coordinator writes `LEDGER.md`; propose the row.

## 6. Deliverables

- `runs/N0_PREREGISTRATION.md` — hash-pinned, committed **before** any oracle read.
- `runs/N0_GLOBAL_REASSIGN_ORACLE.json` + `.md` — all tiers, both the precedent-adjusted
  and raw numbers, the §4 verdict, and the compiler-expressibility finding.
- A proposed `LEDGER.md` row and a dated `LIVE_STATE.md` log line.

`STOP_CLASS` is a complete and valuable outcome: it would close the last untried idea on
this surface for a few CPU-hours, and it must be reported without hedging.
