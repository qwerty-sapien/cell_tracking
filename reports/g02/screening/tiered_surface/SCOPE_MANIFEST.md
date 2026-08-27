# Sweep scope manifest — independent agent fencing

Status: **BINDING.** Read with `ALIGNMENT.md`. Defines the disjoint scope of every sweep
subagent so ≥5 can run concurrently without shared-mutable-state corruption.

## Phases

- **Phase 0 (serial, orchestrator or one setup agent):** build **N1 feasibility-exact
  compiler**; freeze its interface + validity test suite; hash-pin. This is a hard
  prerequisite — every Phase-1 agent loads it read-only. Do NOT dispatch Phase 1 until the
  N1 hash is banked.
- **Phase 1 (parallel, ≥5 subagents):** the disjoint lanes below.
- **Barrier:** orchestrator merges all `state_fragment.json` into `tier_census.json` +
  `LIVE_STATE.md`, appends LEDGER rows, freezes any new caches, then authorizes Phase 2
  (integration of frozen suppliers into the swap lane) if warranted.

## Agents (Phase 1)

| agent id | node | write-scope (ONLY) | frozen read-only inputs | GPU | mutable-state race guard |
|---|---|---|---|---|---|
| `SA-DELETE` | N2 beta | `branches/sa_delete/` | N1, folds, P0 | no | fragment-only |
| `SA-SWAP` | N2 alpha | `branches/sa_swap/` | N1, folds, P0, frozen APNCC/Trackastra cache | no (wave-1) | fragment-only |
| `SA-REPR` | N3 | `branches/sa_repr/` | N1, folds, encoder weights (if authenticated) | **yes** (requests via orchestrator) | fragment-only; sole producer of new caches |
| `SA-NODETOPO` | N4 restore | `branches/sa_nodetopo/` | N1, folds, P0, raw Zarrs (read-only) | maybe (via orchestrator) | fragment-only |
| `SA-DIVISION` | N4 division | `branches/sa_division/` | N1, folds, P0 | no | fragment-only; fitting stays continuity-gated |
| `SA-PARKB-SWEEP` (recommended 6th) | cross-cut | `branches/sa_parkb/` | branch registers (read), N1, folds | no | fragment-only; proposes tier moves, never writes census |

Rules:
- An agent that needs another agent's output waits for the barrier; it never reads a
  sibling's live subtree.
- `SA-SWAP` wave-1 uses ONLY the existing frozen feature cache. New `SA-REPR` suppliers are
  integrated by `SA-SWAP` in Phase 2, after the orchestrator freezes+hash-pins them.
- `SA-REPR` and `SA-NODETOPO` are the only GPU-eligible lanes. They must REQUEST a launch with
  an hours estimate; the orchestrator checks `kaggle_budget.json`, authorizes or defers, and
  debits after the run. No concurrent authorizations may jointly exceed remaining weekly hours.

## Harness enforcement (how the fence is actually held)

1. **Git worktree per agent — MANDATED, auto-approved.** Each Phase-1 subagent runs in its own
   git worktree; only its `branches/<agent>/` diff is merged back at the barrier. Shared inputs
   are read-only in the base. This is the required default; do not co-locate two agents in one tree.
2. **Path-allowlist assertion (belt-and-suspenders).** After each subagent, assert its diff
   touches ONLY `branches/<agent>/` (+ its `state_fragment.json`); abort and discard on any
   out-of-scope write. This runs even with worktrees, as a second check.
3. **Preflight input-hash check** — each agent asserts the SHA-256 of every frozen input
   (N1, folds, P0, caches) before use; mismatch = fail closed.
4. **Single-writer files** — `LIVE_STATE.md`, `LEDGER.md`, `tier_census.json`,
   `kaggle_budget.json` are orchestrator-only. Any subagent write to them is a violation.
5. **GPU launch gate** — the orchestrator authorizes a Kaggle run only if the AUTHENTICATED
   projection is `< 5 h` (auto) or has explicit human approval (`>= 5 h`), AND the remaining
   weekly budget covers it. Auto-approved runs get a hard 5 h wall-clock kill. Debit
   `kaggle_budget.json` after the run before authorizing the next.
6. **Reconcile barrier** — orchestrator runs the merge; it REJECTS any fragment whose tier
   change lacks a cited banked artifact, and quarantines any number matching the memo's
   known-fabricated C2/Branch-B values.
