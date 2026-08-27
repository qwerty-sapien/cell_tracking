# C1 N2 — build and deterministic dry validation

> Load `DAG.md`, `outputs/c1_n0_mechanism_manifest.{md,json}`, and the self-hashed
> `outputs/c1_n1_preregistration.{md,json}`. Do not open held graph outcomes.

## Objective / gate

Does the literal implementation match the frozen preregistration and reproduce inputs,
fold isolation, parameter count, output identity, and C0 exactly? **GO** requires all
specified tests to pass twice with byte-identical predictions and graph inputs.

## Tasks

1. Implement only the frozen C1 architecture and deterministic grouped-OOF runner.
2. Add tests for permutation equivariance within conflict sets, explicit no-link
   behavior, padding/mask invariance, absence of cross-component communication,
   `<=100,000` trainable parameters, and frozen Trackastra gradients/weights.
3. Prove no validation12/CV57/test path is accepted by training code. Authenticate
   fold-disjoint training and held predictions.
4. Replay APNCC and C0 graph inputs exactly; any mismatch is a contract failure, not a
   tolerance opportunity.
5. Run two deterministic fit-only dry passes that do not call the official scorer or
   reveal held graph outcomes. Record wall time, RSS/swap, and artifact hashes.

## Output

Write `outputs/c1_n2_build_validation.md` and `.json`, test logs, implementation hashes,
and proposed LEDGER row `C1-TA-HOCT-LITE-N2 | IMPLEMENT | tests/repeats | no graph score
| 0 external usage | verdict | reason`. Do not edit LEDGER; update LIVE_STATE after
the node verdict.

## Verdict

- **GO** — all tests pass, both dry predictions are byte-identical, C0/APNCC inputs
  replay exactly, and the implementation self-hash matches N1.
- **PARK** — a bounded implementation defect exists without evidence against the
  mechanism.
- **DEAD** — deterministic OOF construction or the architecture contract is
  intrinsically infeasible.
- **ESCALATE** — satisfying the implementation requires changing the preregistration,
  cache identity, scorer, or governance.

## Cascade

- On **GO** → proceed directly to `N3_44b6_graph_falsification.md` in the same run; do
  not wait for a human turn.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
