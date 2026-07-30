# DINOv2 N2 — fixture-only falsifier

> Load `skills/seeking-alpha/SKILL.md`, `DAG.md`, `runs/N1_LITERATURE.md`, DINO terminal reports, and LIVE_STATE.

## Objective / gate

Can the mechanism be killed or retained using only synthetic/existing unit fixtures,
with no weights, videos, feature generation, 44b6 score, or R5 payload access?

## Tasks

1. Preregister the fixture, null, expected direction, exact observable, and kill line.
2. Inspect repository interfaces and existing fixture tests; run only focused unit tests
   that do not load a model or campaign data.
3. Draft a Kaggle-first, fresh-supplier protocol: step-0 authentication, bounded subset,
   <5-hour projection estimate, 0-MiB-swap rule, and no-retry condition.

## Output

Write `runs/N2_FALSIFIER.md`, `runs/N2_KAGGLE_PROTOCOL_DRAFT.md`, and a proposed LEDGER row.

## Verdict

- **GO**: fixture contract is deterministic and the proposed Kaggle run is a new route.
- **PARK**: grounded mechanism lacks a faithful fixture.
- **DEAD**: fixture contradicts the required discrimination.
- **ESCALATE**: only terminal R5 infrastructure could test it.

## Cascade

- On **GO** → proceed directly to `N3_human_kaggle_gate.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
