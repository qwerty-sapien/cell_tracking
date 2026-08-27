# C1 N3 — dependency stop

Date: 2026-07-27  
Verdict: **PARK** (`PARK_DEPENDENCY`)

N2 GO reached the first outcome-opening node, but the shared six-branch contract
requires an **owner-recorded human lock of `U0-T4CANON`** before C1 or C2 may execute
a graph gate. At this check, U0 N0/N1 are GO and U0 N2 is still executing; no owner
lock exists.

No 44b6 reconstruction, C1 fit, official scorer call, proposal-outcome inspection,
6bba run, or CV57 access occurred. This is not evidence against the C1 mechanism.

Resume exactly from N3 after the U0 owner lock. Reauthenticate N1 canonical hash
`d4a3651a…6f5e9`, N2 evidence `28b544de…f398`, and the eventual U0 locked-contract
hash before opening 44b6. The preregistered gate remains unchanged:

- adjusted delta vs APNCC `>=+0.005`;
- raw and node-recall deltas `>=0`;
- observation-distinct predictions;
- C1-minus-C0 adjusted `>=max(0.001, epsilon_null)`;
- adjusted delta `<=0` means DEAD.

## Proposed LEDGER row

`C1-TA-HOCT-LITE-N3 | DEPENDENCY | U0 owner lock absent | no graph score; N1/N2
ready | 0 external usage | PARK | resume N3 only after explicit U0-T4CANON lock`
