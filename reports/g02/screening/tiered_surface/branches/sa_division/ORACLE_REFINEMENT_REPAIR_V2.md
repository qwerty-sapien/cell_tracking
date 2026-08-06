# SA-DIVISION oracle refinement authentication repair V2

Frozen: 2026-07-28, after failed retry 1 and before retry 2.

## Exact failure analysis

Both guarded attempts stopped in `authenticate()` before parsing P0. The
initial expected P0 digest and actual banked digest were compared
character-by-character:

| Value | SHA-256 |
|---|---|
| initial expected | `8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5f18c8889c06` |
| actual banked | `8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06` |

They differ only at zero-based index 53: initial expected `f`, actual `c`.
The first repair incorrectly removed an `a`, so retry 1 also failed
authentication. Both guards recorded zero swap growth; neither attempt wrote
`oracle_refinement_results.json`, parsed P0, fitted a model, or opened a held
scientific outcome.

The actual digest matches the P0 pin in the previously passing
`fixture_audit.py` and `EVIDENCE_INVENTORY.md`.

## Authorized V2 repair

Set the P0 expected digest exactly to the actual banked value above. Correct the
displayed preregistration and mark the first repair superseded. Change nothing
else in the script or calculation contract.

Before retry 2:

1. assert programmatically that the source constant equals a fresh SHA-256 of
   the absolute frozen P0 path;
2. commit this V2 repair;
3. run a fresh 45-second zero-swap preflight;
4. write retry-2 guard and result to new paths while retaining both failures.

Division fitting remains unauthorized and fail-closed.
