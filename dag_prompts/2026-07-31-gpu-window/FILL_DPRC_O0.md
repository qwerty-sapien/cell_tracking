# FILL — DPRC-O0 (DINOv2 ViT-S/14 with registers): authorized, gate-negative

**Status: standing owner authorization exists; this pack's own adjudication says STOP.**
Do not start this node unless the owner overrides
[`LANE_ADJUDICATION.md`](LANE_ADJUDICATION.md) in writing, and not before `K4` completes.

## The authorization

The owner wrote **"DINOv2: go ahead"**, recorded in
`dag_prompts/2026-07-30-continuation/dinov2/runs/N3_HUMAN_REQUEST.md`. It covers exactly one
bounded route at a `5.0` GPU-h ceiling, with a mandatory `<=4.0` GPU-h calibrated projection.
It is explicitly **not** an R5 repair or retry, and no R5 object — checkpoint, source archive,
requests, rows, package, quarantined fragments, or launch lineage — may be reused.

## The gate objection, which must be written into the node report before any request

`MARGINAL_VALUE_GATE.md` clause 1 requires the realistic ceiling in writing, using any
applicable precedent collapse from this campaign. Here it is:

The non-register DINOv2 ViT-S/14 ran this exact class to completion — all 77 request shards,
`1,391,458` actions, atomic checkpoints, zero swap growth — and the sealed B0→A1→A0 gate
returned `FAIL_PARK_C` on both classes:

| Class | A1 mean AUC | Retention | Required AUC | Required retention |
|---|---:|---:|---:|---:|
| delete | `0.670412` | `0.001307` | `0.85` | `0.02039355992844365` |
| occupied swap | `0.682473` | `0` | `0.90` | `0.067910` |

Retention is short by more than `15×`. Registers remove attention artifacts from the patch
grid; they do not change the fact that a frozen generic descriptor carries no notion of
"beneficial action" at `~10^-4` prevalence. Clause 4 makes this a **class-level** STOP, on the
same logic the gate already applied to recall generators.

Three other frozen suppliers — CELLECT, reverse Trackastra, C2/K-best — failed the same way.
**Four for four.**

## If the owner overrides anyway

The authorization's own preconditions are unmet and remain mandatory:

1. A clean, **independently derived** action/materialization manifest. No R5 object.
2. An exact foreground / channel / crop contract, owner-approved before any asset exists.
3. A local preflight enforcing the 5-second envelope and the R5-exclusion rule.
4. A two-video, **non-score** calibration on the predeclared
   `test/44b6_0113de3b.zarr` and `test/44b6_0b24845f.zarr`, projecting the complete frozen
   44b6 delete + occupied-swap route at `<= 4.0` GPU-h. Outputs are timing-only and must be
   discarded without association analysis.

Authenticated provenance already on file:

| Object | Identity |
|---|---|
| Source | `github.com/facebookresearch/dinov2` at `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| License | Apache-2.0, `LICENSE` SHA-256 `600cc67c…935b7` |
| Constructor | `dinov2_vits14_reg`, exactly four register tokens |
| Checkpoint | `88,291,785` bytes, SHA-256 `f433177089a681826af819f1c4ee3bc48fd463fb38d32fc837e3dc74e5641fb` |

Everything in this pack's standing rules applies unchanged: cmux authentication, guard
receipts before every push, serial launches, quota re-query pre/post, no same-signature retry.

Any step-0 failure, unapproved or ambiguous channel transform, projection over `4.0` GPU-h,
hard-limit breach, positive swap growth, incomplete class manifest, or missing provenance
**immediately quarantines partial output and terminalizes this exact route with no retry.**

## Recommendation

Let the hours expire instead. A fresh `30` h opens at `2026-08-01T00:00:00`, and a fifth
frozen representation is worth less than one well-specified new mechanism next week.
