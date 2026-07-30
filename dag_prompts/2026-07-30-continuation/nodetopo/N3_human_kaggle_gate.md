# NODETOPO N3 — event-local Kaggle authorization

> Load `DAG.md`, N2 outputs, NODETOPO terminal reports, and LIVE_STATE. **HUMAN-GATE.**

## Objective / gate

Can the owner review one new, bounded Kaggle evidence experiment with no ambiguity
about its separation from the terminal NODETOPO mechanism?

## Tasks

1. State the new information, input provenance, preflight, fixture result, resource
   estimate, stop conditions, outcome artifact, and no-retry semantics.
2. Explicitly exclude raw-universe construction and the 50-action graph gate.
3. Request approval only; do not create assets, launch, upload, submit, or push.

## Output

Write `runs/N3_HUMAN_REQUEST.md` and a proposed (not written) LEDGER row.

## Verdict

- **HUMAN-DECISION**: always — owner approval is required for any Kaggle action.

## Cascade

- On **HUMAN-DECISION** → halt and ask the owner; do not advance or launch.
