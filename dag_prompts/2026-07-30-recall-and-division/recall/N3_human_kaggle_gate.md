# Recall N3 — bounded Kaggle protocol request

> Load `DAG.md`, all `runs/N0`–`N2` outputs, LIVE_STATE, and the campaign plan.
> **HUMAN-GATE.** Halts for explicit owner approval even on a passing analysis.

## Objective / gate

Prepare — and only prepare — the single bounded, Kaggle-first, 44b6-first protocol that
would convert the N2 supply result into a frozen-scorer graph delta, so the owner can
approve or refuse it with the numbers in front of them.

## Tasks

1. State the exact preregistered statistic: held-fit **adjusted edge Jaccard delta on
   44b6**, with raw edge Jaccard and node recall as nonnegative non-regression guards, and
   the unchanged `>= +0.005` floor on each family. No proposal metric may gate.
2. State what is added and what stays frozen: the candidate-supply step is new; the
   scorer, folds, baseline, budgets, and existing action semantics are unchanged. Name
   where the new candidates enter and how FP admission is bounded per video.
3. Give a conservative GPU projection from a measured two-video calibration, not an
   estimate. Declare the ceiling in GPU-h and the abort rule if the projection exceeds it.
   The plan must fit inside the **25.47 GPU-h** launch capacity with margin and must not
   assume more than one authorized run.
4. Specify the pure-stdlib step-0 preflight that authenticates every mount, manifest,
   population hash, and wheel **before** any dependency install, model load, decode, or
   feature work, and that completes inside the five-second envelope. Enumerate the
   fail-closed cases. Reference the 17-incident / 14-family Kaggle failure audit and state
   which same-signature failures this design cannot repeat.
5. List the artifacts a completed run must emit for the result to be admissible, and the
   quarantine rule for partial shards (no held-fold partial may leak; no class without a
   complete manifest may be aggregated).
6. Request an owner decision. Do **not** create an asset, push a kernel, launch, debit
   budget, submit, or write `LEDGER.md`.

## Output

`runs/N3_HUMAN_REQUEST.md` — the protocol, the measured projection, the stop rules, the
step-0 contract, the required artifacts, and a proposed (not written) LEDGER row.

## Verdict

- **HUMAN-DECISION**: always. Compute spend and external effects are never auto-approved.

## Cascade

- On **HUMAN-DECISION** → halt and ask the owner. Do not advance, upload, or launch.
