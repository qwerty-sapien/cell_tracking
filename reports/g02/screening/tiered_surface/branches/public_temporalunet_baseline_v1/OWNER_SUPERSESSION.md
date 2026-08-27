# Public TemporalUNet3D baseline owner supersession

Date: 2026-08-03

This node adopts the independently trained public TemporalUNet3D pipeline as a
new candidate substrate. It is not a re-screen of the historical DEAD
from-scratch mechanism. That verdict concerned an approximately 478 T4-hour
convergence estimate and an epoch-5 checkpoint; it does not adjudicate the
public checkpoints or public end-to-end pipeline.

The immutable reference is Kaggle notebook Version 32,
`scriptVersionId=337501010`, public score `0.911`, successful runtime `18m57s`
on T4x2. Its local SHA-256 is
`3f995a97a51c9517bdf8d8da2f606ffd184a51c841de87c962d911a336b7847d`.
The primary and seed-314159 checkpoint SHA-256 values are respectively
`12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`
and `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f`.

Public provenance:

- discussion 730924:
  `https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730924`;
- exact public two-seed notebook Version 32:
  `https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-two-seeds-logit-blend?scriptVersionId=337501010`;
- primary public support pack:
  `https://www.kaggle.com/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1`;
- independent seed-314159 checkpoint:
  `https://www.kaggle.com/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1`.

Discussion 730924 reports `0.908` for the single seed, `0.910` for a 50/50
detection-logit blend, and `0.911` for the public two-seed notebook. It also
reports approximately `1.7%` fewer emitted nodes after blending, attributing
that gain to detection precision rather than association. Those are public
targets; our promotion decision remains tied to the exact frozen local scorer.

The frozen reproduction modes are:

1. primary single seed;
2. 50/50 aligned detection-logit blend with primary-only association;
3. exact public V32: the same detection blend plus secondary association weight
   `0.15`, applied only to low-margin cases where both views select the same
   parent.

Promotion requires exact official scoring on frozen held-fit105 and CV57.
Until a local score exceeds APNCC `0.881522069960`, APNCC remains the active
base, legacy control, and FINAL slot 1. Existing compiler, scorer harness,
requirement surface, and tiered census are re-pointed only after successful
reproduction. A submission is permitted only after the local threshold is
cleared and the shared serial/submission lane is free.

Owner constraint added 2026-08-03: treat this pipeline only as a substrate and
ensembling base. No checkpoint fine-tuning, continued training, adapter
training, or other weight refinement is permitted below exact local `0.916`.
The reproduction harness therefore contains no training path and records
`weight_refinement=false` in every package and result receipt.

Post-plateau work follows FOYSAL's discussion guidance: diagnose detection,
ordinary association, fragmentation, and division separately; use an extra
temporal/model view only for uncertain association cases, not as an
unconditional global edge-logit average.

The owner continuation dated 2026-08-03 is frozen in
`CONTINUATION_AND_INTEGRATION_20260803.md`: preserve hourly polling, score V32
exactly, measure V32+G1 independently, re-point reusable G-04 machinery only
after authentication and APNCC clearance, and capacity-gate every campaign's
GPU launch through the one shared reservation ledger.
