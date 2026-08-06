# N4 — Kernel package + submission (**HUMAN-GATE**)

> Load `DAG.md`, `runs/N0_SUBMISSION_FORENSICS.md`, `scripts/cmux_ops.py`,
> `reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_AUDIT.md` and
> `kaggle_prelaunch_guard.py`.
> **This node halts for explicit owner approval even on GO.** It spends a submission slot
> and produces an outward-facing effect.

## Why this is a HUMAN-GATE

`CLAUDE.md`: *"No Kaggle GPU/submission or RunPod launch without explicit human approval."*
Auto-cascade automates analysis, never spend or external effects. Submissions are capped at
**5/day** with **2 final selections**; we have burned 6 of 13 on an unscored transport.

## Entry conditions (ALL required — do not enter on partial)

1. `N0` verdict is **GO** with the minimal fix identified and applied. Submitting before N0
   is a blind retry into a known failure mode — LIVE_STATE explicitly records *"no blind
   retry"* for this exact symptom.
2. A local CV57 score exists for the candidate, produced by the frozen official scorer.
3. `kaggle_prelaunch_guard.py` returns an eligible PASS receipt bound to the exact package
   bytes.
4. `cmux_ops.py` `competition_submit` has its required `eligible: true` predicate receipt.

## Execution constraints

- The agent shell's own token is for **read** endpoints only. All writes (kernel push,
  submission) go through `scripts/cmux_ops.py` in the owner's authenticated **cmux** session
  — never ad-hoc typed commands, per that module's doctrine ("free-form typing into an
  authenticated session is how a one-run authorization became two byte-identical kernel
  versions and a 0.74 GPU-h debit").
- `cmux.app` is a macOS GUI application with no CLI on PATH. **The agent cannot drive it
  from this session** — there is no macOS computer-use tool available here. The deliverable
  of this node is therefore an *enumerated, reviewable command plan* for the owner (or
  Codex, which historically performs cmux ops) to execute, not an agent-executed submission.
- NO retry logic, ever. One authorization = one run.

## Tasks

1. Emit the exact `cmux_ops.py` invocation plan (`command_plan()`), reviewable before it runs.
2. Bind the prelaunch PASS receipt hash to the package bytes being pushed.
3. Record the preregistered expected public score and the transfer interval from `N3`,
   **before** submitting, so the result is a test rather than a retrofit.
4. Halt. Present the plan and wait.

## Output

`runs/N4_SUBMISSION_PLAN.md` + a LEDGER row. No submission is created by this node without
a fresh owner approval recorded against this specific plan.

## Verdict & cascade

- **HUMAN-DECISION** always. Never auto-advances, regardless of analytical verdict.
