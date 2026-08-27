# G-03 Live State — current source of truth

The root coordinator updates this file after a G-03 decision, launch, or terminal
result. Historical detail remains in `LEDGER.md` and immutable receipts.

## Current frontier — 2026-08-12

- No G-03 process, Kaggle launch, GPU reservation, scorer access, or submission is
  active.
- Phase 1 A-v2/B-v2 completed previously with authenticated handoff parity. Phase 2
  H000 completed exact same-runtime parity and selected Branch B; no further training
  or hidden/public action is authorized from that result alone.
- The h005 lane is closed after two pretraining infrastructure failures under its
  frozen no-V3 policy. h030, evaluation, CV57, hybridization, submission, and RunPod
  remain closed.
- Wave-R R1 is terminal below promotion; R4 is operationally killed before a sealed
  score. No score may be reconstructed from its partial SQLite/spool artifacts.
- Bulky R1/R4 generated artifacts and spools were moved to recoverable quarantine
  `/private/tmp/cell-tracking-stale.kooFaj`; compact contracts, terminal receipts, and
  `LEDGER.md` remain.
- Current operational/resource authority is repository-root `LMT.md`. Its local
  resource envelope applies only to `cell_tracking`-initiated local CPU runs and not
  to Kaggle packaging or compute.

## Recent log

- 2026-08-12 — Removed stale G-03 division/OOF scripts and tests through Git and moved
  approximately 35 GB of failed-run R4 caches/spools plus closed R1 artifacts to
  recoverable quarantine. No live process or accepted scientific result was removed.
