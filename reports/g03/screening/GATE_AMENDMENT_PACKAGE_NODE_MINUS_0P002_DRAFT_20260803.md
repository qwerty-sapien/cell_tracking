# Draft owner-applied 44b6 node-gate amendment

Status: **DRAFT ONLY — NOT APPLIED** (2026-08-03).

## Proposed replacement

For the 44b6-first graph continuation gate, replace only the node-recall clause:

| Component | Current | Proposed |
|---|---:|---:|
| adjusted delta | `>= +0.005` | unchanged: `>= +0.005` |
| raw delta | `>= 0` | unchanged: `>= 0` |
| node-recall delta | `>= 0` | **`>= -0.002`** |

The proposed gate is therefore:

`adjusted >= +0.005 AND raw >= 0 AND node >= -0.002`.

This package records the owner's requested amendment text and evidence. It does not
modify any campaign plan, evaluator, gate artifact, historical verdict, tier, or
execution authority. Only the owner may apply it.

## Justification: APNCC counterexample

Against the exact pre-APNCC `c=0.25` held-44b6 control, APNCC-DIRECT produced:

- adjusted delta `+0.006949544395` — PASS;
- raw delta `+0.006411258660` — PASS;
- node-recall delta `-0.001307630206` — FAIL under the current `node >= 0` clause,
  but PASS under the proposed `node >= -0.002` clause.

APNCC-DIRECT nevertheless delivered the campaign's confirmed full-CV57 gain of
`+0.008372476431`. It is therefore a concrete false-negative counterexample to a
universal nonnegative-node requirement, while remaining within the proposed
`-0.002` tolerance.

## Historical-verdict audit

**No historical verdict changes.** Every reviewed 44b6 kill failed the unchanged
`adjusted >= +0.005` clause, including the five D3 recommendations accepted by the
owner. The A7 and R0 zero-edit cases also fail the adjusted clause. Relaxing only the
node floor therefore reopens none of them. The owner has separately accepted parking
the earlier R9 and AE-P3/R0 retractions and accepted all five D3 “no” recommendations.

This amendment must not be used to retrofit a PASS to any historical experiment;
future use begins only after explicit owner application and provenance resealing.

## Soft calibration anchor

Record one descriptive transfer anchor:

- held-44b6 adjusted gain: `+0.006949544395`;
- full-CV57 gain: `+0.008372476431`;
- ratio `44b6 / CV57 = 0.8300464567` (approximately `0.83`).

This is a **single anchor and soft calibration only**. It is not a regression, lower
bound, multiplicative forecast, or permission to infer CV57 performance for another
mechanism.

## Application checklist for the owner

If applied, the owner should name the effective date, update the authoritative gate
contract and plan, reseal affected evaluators/receipts, and state prospectively which
future nodes inherit the new clause. Historical artifacts and verdicts remain
immutable.

