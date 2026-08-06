# DAG — GPU window 2026-07-31

**The tiered surface was declared TERMINAL on 2026-07-31
(`TERMINAL_CAMPAIGN_REPORT.md`, `9b0a7819…1a961`).** The `K*` chain below therefore requires
an explicit owner override to start. The recommended path is `R0`.

```
   ┌──────────────────────────────────────────────────────┐
   │ R0  batch/sequential action valuation                │  0 GPU · RECOMMENDED
   │ satisfies reopening clause 1 on its merits           │  design + CPU, fits 20 h
   │ output: diagnosis, gate verdict, maybe a prereg      │  no override needed to start
   └──────────────────────────────────────────────────────┘

   ══════════ everything below needs an owner override ══════════

                    ┌──────────────────────────────┐
                    │ OVERRIDE  reopen the surface │  BLOCKING · owner only
                    └───────────────┬──────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │ H0  human fold decision      │  BLOCKING · 0 GPU
                    │ automatic_choice_forbidden   │  owner only
                    └───────────────┬──────────────┘
                                    │
   ┌────────────────────┐           │
   │ K0  asset rescue   │           │   K0 is independent of H0 —
   │ 0 GPU · do NOW     │           │   run it immediately
   └─────────┬──────────┘           │
             │                      │
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │ K1  implementation   │  0 GPU · CPU-only · TDD
             │ freeze proof, <1%    │  45 s idle preflight, 0 swap
             │ leakage firewall     │
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │ K2  packaging+guard  │  0 GPU · dataset READY
             │ static guard PASS    │  registry entry first (KFG-17)
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │ K3  calibration      │  ~0.3-0.6 GPU-h
             │ 2 epochs x 2 folds   │  ABORT unless projection <= ceiling
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │ K4  full run         │  <=4.8 h (opt B) / <=8.0 h (opt A)
             │ nested OOF           │  hard kill 4h55m
             └──────────┬───────────┘
                   PASS │ FAIL ──▶ PARK-C this realization. STOP.
                        ▼
             ┌──────────────────────┐
             │ K5  44b6 graph gate  │  0 GPU · independent replay
             │ provenance closure   │  unlocks ONE 44b6 contract only
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │ K6  submission       │  HUMAN GATE · default NO
             │ needs local >= 0.89  │  banked is 0.881522
             └──────────────────────┘

   ┌───────────────────────────────────────────────┐
   │ FILL  DPRC-O0   authorized but gate-negative  │  only after K4,
   │ <=5.0 GPU-h ceiling, <=4.0 projection         │  only on owner override
   └───────────────────────────────────────────────┘
```

## Codex autonomous execution layer (`codex/`)

`K2`/`K3` are specified as *what must be true*. The `codex/` pack is *how it runs unattended*
against the owner's authenticated cmux session:

```
C0  cmux_ops.py: bounded command plan, serial lock, scrubbed receipts   no external calls
    └─ whitelist argv only · no shell=True · no retry logic
    └─ kernel_push refuses while a version is RUNNING/QUEUED  (the 0.74 GPU-h race)
    └─ competition_submit refuses without an eligible predicate receipt
         │
C1  asset upload ─────────────▶ verify by DOWNLOAD + HASH, never `datasets files` (403 by scope)
         │
C2  launch ───────────────────▶ guard chain · quota pre-query · one push · version read-back
         │                       C2a calibration (abort unless projection <= 8.0 T4-h)
         │                       C2b full run
         │
C3  monitor ──────────────────▶ 60 s poll · hard kill · output banked · quota post-query
         │                       debit = owner-authenticated pre/post delta
         │
C4  submission predicate ─────▶ P1..P7; for this lane it evaluates FALSE (0.881522 < 0.89)
```

Launch and monitoring are **pre-authorized** by `AGENTS.md` for this approved node — Codex
does not stop to ask. Submission is pre-authorized *under its predicate*, which this lane
does not satisfy.

## Budget ledger for the window

| Node | GPU-h | Cumulative |
|---|---:|---:|
| **R0 (recommended)** | `0` | `0` |
| K0, H0, K1, K2 | `0` | `0` |
| K3 calibration | `~0.3–0.6` | `~0.6` |
| K4 full run (opt B / opt A) | `4.8` / `8.0` | `~5.4` / `~8.6` |
| K5, K6 | `0` | unchanged |
| FILL DPRC-O0 (if overridden) | `<=5.0` | `~10.4` / `~13.6` |

Starting balance `24.98` h, refresh `2026-08-01T00:00:00`. Expected unspent under the
recommended path: **all `24.98` h.** That is the correct outcome, not a failure — the surface
is closed, and a fresh `30` h opens at the refresh. See `LANE_ADJUDICATION.md`.

## Kill conditions that end the whole window

- No owner override → the `K*` chain never opens; run `R0`.
- `H0` returns NO LAUNCH.
- `K3` projection exceeds the ceiling → terminal before score-bearing computation.
- `K4` trips the wall-clock kill → quarantined partials, no verdict, no restart.
- Any push fails with a signature already in `KAGGLE_FAILURE_REGISTRY.json` → stop, no retry.
- Guard receipt missing or stale at any push → refuse the compute request.
