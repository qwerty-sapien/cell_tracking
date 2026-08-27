# C1 N1 — freeze the single alpha arm

> Load `DAG.md` and `outputs/c1_n0_mechanism_manifest.{md,json}`.

## Objective / gate

Can one fully specified `<=100,000`-trainable-parameter conflict-set head be frozen
before any held graph outcome is opened? **GO** requires an exact tensor schema,
architecture, loss, seed, fold map, training schedule, inference rule, controls,
reproducibility null, and graph verdict table.

## Tasks

1. Freeze one permutation-equivariant design: frozen Trackastra/APNCC edge tokens, a
   shared projection, exactly one masked edge-to-edge self-attention block within each
   conflict component, one explicit learned no-link token, and a shared output head.
   Record dimensions and prove the trainable count is `<=100,000`.
2. Forbid topology messages: no adjacency propagation, node/track state update,
   multi-frame graph convolution, division head, or communication across conflict sets.
3. Freeze five video-grouped folds, one seed (`20260727`), optimizer/schedule, loss,
   stopping rule derived only inside training folds, and one deterministic inference
   rule. No hyperparameter or acceptance-threshold sweep is permitted.
4. Freeze comparators on identical rows: exact APNCC and `C0-T4TA-LINEAR`. Define
   `epsilon_null = max(1e-12, 2 * maximum absolute adjusted-score difference among
   exact identical-repeat reconstructions)` before model scoring.
5. Preregister N3/N4: adjusted delta vs APNCC `>=+0.005`, raw delta `>=0`, node-recall
   delta `>=0`, observation-distinct output; N3 also requires C1 adjusted score minus
   C0 adjusted score `>=max(0.001, epsilon_null)`. A 44b6 adjusted delta `<=0` is DEAD.

## Output

Write immutable `outputs/c1_n1_preregistration.md` and `.json` with a self-hash,
parameter-count derivation, fold IDs, seeds, exact commands to be implemented, and
proposed LEDGER row `C1-TA-HOCT-LITE-N1 | PREREGISTRATION | frozen contract | no score
opened | 0 external usage | GO/PARK/DEAD/ESCALATE | reason`. Do not edit LEDGER;
update LIVE_STATE after the node verdict.

## Verdict

- **GO** — every degree of freedom and every later outcome route is fixed and the
  trainable count is within budget.
- **PARK** — the grounded idea cannot yet be made deterministic without more fit-only
  evidence.
- **DEAD** — the design necessarily violates the no-GNN/no-search/frozen-body rules.
- **ESCALATE** — an upstream identity or scorer contract must change.

## Cascade

- On **GO** → proceed directly to `N2_build_dry_validate.md` in the same run; do not
  wait for a human turn.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
