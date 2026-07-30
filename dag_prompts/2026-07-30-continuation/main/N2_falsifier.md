# MAIN N2 — preregistered cheap falsifier

> Load `skills/seeking-alpha/SKILL.md`, `DAG.md`, `runs/N1_LITERATURE.md`, retained
> scorer/parity evidence, and LIVE_STATE.

## Objective / gate

Can the leading mechanism be tested on a frozen fit-only slice in **≤15 CPU minutes,
≤1 GiB RSS, and 0 MiB swap growth**, with an objective-linked preregistered kill line?

## Tasks

1. Specify exact inputs, code/module boundaries, holdout exclusions, null control,
   positive threshold, and `≤0` or null-envelope kill condition.
2. Verify feasibility only through static inspection and existing unit fixtures; do not
   access held 44b6 scoring, create features, or run a model.
3. If it survives, draft the smallest Kaggle-first iteration protocol and estimate
   requested GPU-hours against the 25.8111-hour capacity. No execution is permitted.

## Output

Write `runs/N2_FALSIFIER.md` and `runs/N2_KAGGLE_PROTOCOL_DRAFT.md`, plus a proposed
(not written) LEDGER row.

## Verdict

- **GO**: all resource, integrity, and preregistration fields are concrete and pass.
- **PARK**: a credible mechanism lacks a cheap objective-linked falsifier.
- **DEAD**: the cheapest faithful falsifier is nonpositive or violates a hard constraint.
- **ESCALATE**: the test needs a locked plan/amendment or external permission.

## Cascade

- On **GO** → proceed directly to `N3_human_kaggle_gate.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
