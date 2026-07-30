# MAIN N1 — literature and novelty audit

> Load `skills/seeking-alpha/SKILL.md`, `DAG.md`, `runs/N0_SCOPE.md`, and LIVE_STATE.

## Objective / gate

Does at least one N0 mechanism have **two independent primary sources** supporting its
causal claim, a named reason it is not crowded beta, and a testable link to 44b6?

## Tasks

1. Search primary literature and record persistent links, exact supporting findings,
   assumptions, and counterevidence for each candidate.
2. Reject evidence that supports only a generic embedding/model/tool rather than the
   candidate's mechanism.
3. Select at most two survivors and state their minimum effect and falsification slice.

## Output

Write `runs/N1_LITERATURE.md` with citations and a proposed (not written) LEDGER row.

## Verdict

- **GO**: ≥1 survivor has two independent primary sources plus a defensible novelty rationale.
- **PARK**: evidence is promising but incomplete or only secondary.
- **DEAD**: sources contradict the causal mechanism.
- **ESCALATE**: validating it would require an already forbidden route.

## Cascade

- On **GO** → proceed directly to `N2_falsifier.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
