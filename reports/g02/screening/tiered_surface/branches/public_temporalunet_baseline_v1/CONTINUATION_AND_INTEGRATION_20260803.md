# TemporalUNet3D V32 continuation and integration contract

Date: 2026-08-03  
Status: owner-authorized continuation; no new direction

## Remote cadence and compute governance

The existing heartbeat remains hourly. It may perform at most one Kaggle status
read per hour and must not add ad-hoc polling. V32 shards remain serial and
immutable-weight. No checkpoint fine-tuning, continued training, adapter
training, or other weight refinement is permitted below exact local `0.916`.

All Kaggle GPU actions across every campaign and concurrent task use the one
canonical ledger at `../../KAGGLE_SHARED_GPU_LEDGER.json`. A launch must acquire
the canonical CMUX lock and reserve its worst-case GPU-hours before mutation.
The capacity gate is:

`sum(active reservation ceilings) + proposed ceiling <= authenticated remaining weekly GPU-h`.

The reservation stays active until terminal status and authoritative quota are
reconciled under the same lock. `scripts/cmux_ops.py` enforces this for every
bracketed `kernel_push`; a positive `gpu_reservation_h` is mandatory. The
already-running V32 shard 0 is bootstrapped into the ledger at its frozen
10,800-second (`3.0 h`) whole-kernel ceiling. This ledger covers V32 shards and
any Prompt-1 GPU retry. It does not relax the separate G-03 GPU owner lock.

## Completion and scoring

After all six V32 shard outputs pass byte, manifest, population, and graph
validation, join exactly the frozen CV57 population and score once with the
official frozen scorer. Retain the held-fit105 score where the frozen package
already supports it. Report:

- exact combined, adjusted-edge, raw-edge, and node terms;
- detection, ordinary association, matched-lineage fragmentation, and division
  errors separately;
- differences from public `0.908--0.911` and APNCC
  `0.881522069960`.

APNCC remains the legacy control and FINAL slot 1 until a candidate beats it
publicly.

## Independent global-jump measurement

Reapply the frozen G1 frame-global registration mechanism from
`../../../../../g04/lanes/data_pathology_repair/TERMINAL_REPORT.md` to the V32
pipeline before its candidate/association construction. Preserve original
output coordinates and GT rows. Score V32 and V32+G1 independently on the same
CV57 population. The APNCC G1 delta `+0.005242350300` is historical context
only and must never be imported, added, or used as the V32 estimate.

## G-04 integration

After authenticated V32 completion and exact-score clearance over APNCC,
re-point the reusable feasibility-exact compiler, exact scorer harness,
requirement surface, and tiered census to the authenticated V32 graph identity.
Every re-pointed artifact must carry both the V32 parent hash and an APNCC
legacy-control reference. Parity tests must show that only the parent graph
changed. Submission remains separately gated by local clearance and a free
terminal submission lane.
