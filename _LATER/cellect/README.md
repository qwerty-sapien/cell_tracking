# CELLECT — Paused

This directory preserves the complete CELLECT investigation and clean-room
reimplementation work as of 2026-07-23. It is valid but deprioritised. Do not
resume, modify, run, package, or otherwise act on it unless explicitly asked.

Read [PAUSE_RATIONALE.md](PAUSE_RATIONALE.md) before resuming. The paper-derived
contract is consolidated in [PAPER_SPECIFICATION.md](PAPER_SPECIFICATION.md).

The prior repository layout is retained beneath this directory so the work can be
moved back without reconstruction:

- `src/`, `tests/`, and `scripts/`: implementation, tests, fixture generator, and
  experiment entry points;
- `tests/fixtures/`: frozen numerical fixtures;
- `reports/` and `artifacts/`: experiment notes, manifests, results, and
  checkpoints;
- `third_party_licenses/`: the preserved GPL notice and licence analysis;
- `tmp/pdfs/`: the downloaded supplement, extracted text, page splits, and page
  renders;
- `research/`: CELLECT-specific material extracted from broader active notes.

`external/CELLECT` deliberately remains at the repository root as an unmodified
reference-only Git submodule. It is not shipped code and must never be imported by
submission code.
