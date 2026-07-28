# SA-REPR-MULTI terminal report

Date: 2026-07-28  
Branch: `codex/sa-repr`  
Verdict: **PARK_NOT_READY — POPULATION FROZEN; NO NEW SUPPLIER SAMPLE QUEUED**

## Outcome

The supplier-agnostic 44b6 separation population is frozen and ready for a future
authenticated dense/R-family supplier. Every one of its 154 population/request
shards passed exact path, SHA-256, endpoint-array, label, fold, finite-value, and
class-total checks. The complete read-only inventory covers 44,729,626 bytes and
has aggregate SHA-256
`ebde982b68abfa1359b0323b8699c31a0e303c0982cfcb01caa2b2fa0536130c`.
The freeze artifact is `POPULATION_FREEZE.json` at
`f78c11e3956f70db1cdec749a7ce1f71e9131252e2bdaaf7a7e5f83fc7d418ae`.

The population remains exactly:

- delete: 1,345,497 actions, 1,118 beneficial, exact aggregate FP budget 261;
- occupied swap: 45,961 actions, 134 beneficial, exact aggregate FP budget 0.

These are banked population facts, not supplier performance. No new population
was materialized or copied.

## Comparison contract

`PREREGISTRATION_v1.json` was frozen before any new supplier endpoint feature,
A1/A0 score, separation-probe result, graph result, 6bba result, or CV57 result.
The sequence clarification records that a read-only filename inventory and the
already-banked population summary were inspected during orientation before v1;
they did not alter supplier selection or any rule. The preregistration inherits
the authoritative per-class amendment
`c3c86ec91307fc19a393f060912b50fe95fe42a0d68102508d154c3fe498458c`
without changing the B0/A1/A0 arms, deterministic float64 ridge probe, exact
per-video budgets, or any AUC/retention threshold. A class-local separation PASS
would only unlock a new 44b6 graph preregistration for human review; it is not a
tier promotion or graph authorization.

## Provenance verdict

The R2–R10 audit is `SUPPLIER_PROVENANCE_AUDIT.json` at
`64492bcc0278f6d609cf4b2a72a12109b380842aed963847aaa830991962b1e7`.
Only `F2-CELLECT-PUBLISHED-64D-V1` (R7) re-authenticated: its exact weight,
upstream source, license, clean pinned commit, clean-room loader, and adapter
hashes match the banked contract. Its sample is already dispatched on the
independent serial lane, so this lane neither duplicated its queue entry nor
waited for its outcome.

No other fixed R-family candidate is sample-ready:

- R2/R3: no authentic complete mask/wrfeat identity join;
- R4/R5/R6: no exact local DINOv2 ViT-S/14 weight and therefore no complete
  source/license/offline-loader authentication;
- R8: no immutable reverse-inference supplier/endpoint-return contract;
- R9: no new tracklet/event evidence supplier;
- R10: no separate `<1%` adapter contract paired with new authenticated evidence.

These are acquisition/dependency holds, not scientific failures. Every affected
node remains PARK-B and no tier move is proposed. `READY_SAMPLE_QUEUE.json` at
`ed23ed2fd11b7d7c14c18e7d4637a5503a8084975a373ac4025ad2d310ec1153`
therefore contains zero new samples and holds no GPU authorization.

## Verification and resources

The required idle preflight passed for 45.067 seconds with zero new swapouts.
Five focused fail-closed tests and 28/28 final artifact checks passed. The full audit completed in 1.774 seconds
at 0.0463 GiB peak child RSS and zero new swapouts while hashing and validating
44.7 MB. `RESOURCE_REPORT.json` records the instrumentation, four fail-closed
pre-work wrapper repairs, and one fail-closed validator-assertion repair. External use was zero: no GPU/Kaggle,
authorization request/hold, download, service, RunPod, submission, graph, 6bba,
or CV57 action.

Memo C2/K-best and Branch-B numbers were quarantined. No pseudo-mask was created.
The lane is stopped at its own verdict barrier and does not block on R7.
