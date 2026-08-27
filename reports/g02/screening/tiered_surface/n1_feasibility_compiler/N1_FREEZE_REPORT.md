# N1 feasibility-exact action compiler — terminal freeze

Status: **GO**.

The compiler passed all 10 unit tests, all 6 single-action kinds,
both batch-order realizations, and all 7 negative fail-closed fixtures. The guarded
test run recorded zero swap growth. No label, held outcome, audit group, test data,
GPU, Kaggle, RunPod, external service, or submission was used.

## Frozen identities

- Interface: `2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865`
- Implementation: `457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5`
- Tests: `09107306fbdea2078d5a73006bde6215a242779ae9245015e365181dd48bee97`
- Freeze script: `b73af40931a5bb6afa0b47eb89a4e0aa1eb9ba69cc27ab737dc26e3f8e2ec254`
- Unit guard: `123560af6167b79f6265f7afbf2f51f704277b1cada430b065f1a8842ceec9cf`

## Scope confirmation

- P0: 105 train + 12 validation videos, fit-only.
- Grouped folds: 3 folds over 175 remainder groups; 20 audit groups untouched.
- Leakage intersections: zero in every fold.
- P0, plan, folds, provenance, and fit117 scope all match their locked SHA-256
  identities in `N1_FREEZE_REPORT.json`.

## Cascade

N1 is frozen and G1 advances from PARK-B to PARK-A. Phase 1 is ready, but remains
undispatched until the root reports this freeze and the updated tier census.
