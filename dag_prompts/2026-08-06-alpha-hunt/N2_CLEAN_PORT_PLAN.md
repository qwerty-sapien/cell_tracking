# N2 — Clean public-mechanism port plan (beta lane)

> Load `DAG.md`, `reports/g02/screening/public_solution_landscape_survey_v1.md`.
> Independent of N0 — local-CV57 work only.

## Objective / gate

Six legitimate mechanisms exist in the clean public tier and are absent or disabled in ours.
**These are BETA by construction** — publicly posted, any of 2,019 teams can copy them. They
are the *measured* route from 0.914 toward ~0.921 (best clean public author's live score),
not an edge. Fund them as cheap table-stakes, never as the campaign's differentiator.

Absent from our code entirely (all from `yusuketogashi/no-hack-...-3rd`, author live 0.916):
1. Bidirectional harmonic edge fusion (`BIDIRECTIONAL_EDGE_WEIGHT=0.20`)
2. 4-view JS-reliability log-pool edge TTA
3. 22-feature learned local association ranker (needs `pilkwang/biohub-local-association-ranker-unet300-v1`)
4. Forward-acceleration lookahead (`MAX_ACCEL_UM=4.0`, `MAX_BONUS=0.20`) — ~30 lines, pure
   geometry, no weights, no retraining. Best value per unit effort.

Present in our code but config-disabled:
5. DeepCenter center-prior veto (we set `USE_DEEPCENTER_VETO=0`)
6. Adaptive short-track rescue (we set `ADAPTIVE_SHORT_TRACK_RESCUE=0`)

**Preregistered GO threshold:** a ranked port order with exact patch anchors and a zero-GPU
CV57 validation defined per mechanism. Any anchor that will not match our V32 source must be
flagged, not silently adapted.

## Tasks

1. Diff each mechanism against our V32 source; document exact string-patch anchors.
2. Per mechanism: inference cost multiplier, extra checkpoint dataset needed, integration
   risk, local-validation feasibility, ordering dependencies.
3. Rank by expected CV57 gain per unit integration risk.
4. Flag conflicts with our constraints: outdegree cap 2, dual-seed logit blend, det
   threshold 0.96875.

## Output

`runs/N2_CLEAN_PORT_PLAN.md` + a LEDGER row.

## Verdict & cascade

- **GO** -> proceed to `N4_KERNEL_PACKAGE_SUBMIT.md` (HUMAN-GATE, halts).
- **PARK** (anchors unmatchable / integration risk too high) -> halt; report.
