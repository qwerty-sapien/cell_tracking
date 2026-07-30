# MAIN N3 — Kaggle-first authorization request

> Load `DAG.md`, `runs/N2_FALSIFIER.md`, `runs/N2_KAGGLE_PROTOCOL_DRAFT.md`, and LIVE_STATE. **HUMAN-GATE.**

## Objective / gate

Is there a complete, bounded Kaggle-first protocol that an owner can approve or reject
without further design work?

## Tasks

1. Summarize the exact hypothesis, frozen inputs, preflight, estimate, stop rules,
   budget debit, result manifest, and no-retry semantics.
2. State why Kaggle is preferred over local or RunPod, without touching Kaggle.
3. Present the approval request only; do not launch, create assets, submit, or push.

## Output

Write `runs/N3_HUMAN_REQUEST.md` and a proposed (not written) LEDGER row.

## Verdict

- **HUMAN-DECISION**: always — owner approval is required before any Kaggle action.

## Cascade

- On **HUMAN-DECISION** → halt and ask the owner; do not advance or launch.
