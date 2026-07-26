# N4 — One public probe

> Load `DAG.md`, `../LIVE_STATE.md`, the N3 GO result, the frozen package
> manifest, and `../p0_parity.md`. **HUMAN-GATE:** this node must not submit
> without explicit approval at this boundary.

## Objective / gate

Is one observation-distinct H1 public probe fully preregistered and authorized?
All prior gates must be GO, local CV57 must be `>=0.89`, package-path parity
must be exact, and the daily submission/final-slot budgets must remain valid.

## Tasks

1. Verify package SHA-256s, exact `submission.csv` production, package-path
   CV57 parity, and nonidentity versus all banked public outputs.
2. Record the expected coarse transfer range without claiming a precise score.
3. Present the exact human submission command and the result-banking protocol.
   Do not push, launch, or submit.

## Output

Write `../h1_public_probe_preregistration.json` and update
`../LIVE_STATE.md`. The root coordinator appends exactly one
`H1-ZWRFEAT-PUBLIC-GATE` ledger row.

## Verdict & cascade

- **HUMAN-DECISION** when all analytical checks pass: halt for explicit
  submission approval and human execution.
- PARK if the package is not observation-distinct, local score is below
  `0.89`, or a submission budget is unavailable.
- DEAD on package-path parity or identity failure.
