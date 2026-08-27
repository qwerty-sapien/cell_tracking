# N0 — exact action-class oracle

> Load `seeking-alpha/SKILL.md`, `DAG.md`, the locked campaign plan, and the frozen
> scorer. Current banked output is
> `../../action_class_oracle/p0_action_class_oracle_v1.json`.

## Objective / gate

Compute exact family-separated graph-score headroom for add, delete, swap, gap,
division, node-add, and node-restore. GO iff at least one class has positive exact
headroom and every transformed graph is valid; a zero class is PARK-C without model
spend.

## Tasks

1. Use fit-only labels only for the oracle diagnostic.
2. Apply complete incident-conflict mutations under official matching.
3. Compare each action with an explicit per-video no-op.
4. Report score/adjusted/raw/node/division deltas, support, background, and hashes.

## Output

Immutable JSON/Markdown, guard report, LIVE_STATE update, and a proposed ledger row.

## Verdict & cascade

Banked verdict: **GO**. Delete, occupied swap, topology-complete restore, and division
have positive headroom; graph-invalid actions = 0.

- On GO -> proceed directly to `N1_FEASIBILITY_COMPILER.md`.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION -> halt and report.
