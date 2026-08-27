# U0-N1 — fail-closed T4-native interface verification

Date: 2026-07-27  
Verdict: **GO**  
Score: `NOT SCORED`

## Result

The new U0 namespace implements the 12 preregistered invariants without modifying
the old H1 contract, runner, evidence, or verdict.

- Focused verification: **12/12 PASS**.
- Authenticated positive coverage: 105 T4 base payloads/sidecars plus 105 T4
  feature payloads/sidecars; exactly 36 44b6 and 69 6bba videos.
- Negative fixtures reject a changed row, ID, fold, schema, payload hash, scorer
  hash, M5 decision argument, and forbidden-scope path.
- A video with no exact label overlap remains in its deterministic grouped graph
  partition and contributes zero rows to fit/calibration; this sparse-label behavior
  is explicitly tested.
- If a fold has no threshold satisfying the unchanged ≥100/≥0.95 calibration rule,
  the exact decision is `NO_ADMISSIBLE_THRESHOLD` and its held action is a no-op;
  the threshold is not weakened.
- The executable CLI exposes no split selector, scorer argument, metric function,
  M5 decision input, GPU action, or external action.
- U0's scorer-free reconstruction unit is an exact canonical graph-mutation payload:
  base selected-edge identity plus node-ID remove/add operations under the frozen
  APNCC thresholds and cap.
- Focused verification used 244,023,296 bytes peak RSS. Swap remained 353 MiB before
  and after, for **0 MiB growth**. The nonzero system baseline is recorded rather
  than hidden; the preregistered replay guard kills on any increase.

Contract SHA-256: `08cd90e5d67f0bf9c7822caa1a057f767b5e7e66e0889fcde04ef317cfc016c1`.
Implementation and test hashes are frozen in
`results/u0_t4canon_implementation_manifest.json`.

Old `H1-ZWRFEAT-CACHE-ATTEST` remains
`DEAD_STRUCTURAL_M5_T4_DRIFT`; no tolerance was changed.

## Proposed ledger row

`U0-T4CANON-IMPLEMENT | fail-closed T4-native interface | IMPLEMENT / CONTROL |
12/12 focused tests; 105+105 payload/sidecar coverage; eight negative mutation classes
rejected | NOT SCORED | 0 submissions | 0 GPU-h | 0 RunPod | GO | scorer/split/M5
decision surfaces absent; proceed to exact replay | U0_N1_interface_verification.md`

## Cascade

GO → `N2_exact_replay_rehearsal.md`.
