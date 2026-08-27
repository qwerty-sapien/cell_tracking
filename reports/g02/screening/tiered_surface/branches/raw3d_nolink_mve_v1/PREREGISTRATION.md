# RAW3D-NOLINK-MVE Stage 0 — frozen preregistration

Frozen at `2026-07-31T07:19:18Z`, before any RAW3D outcome was opened.

This authority is deliberately narrow: CPU-only Stage 0 on the exact frozen
1,345,497-action DELETE population. It grants no acquisition, pretrained
weight, GPU, training, graph, 6bba, CV57, or submission authority.

The extractor streams the authenticated raw 44b6 Zarrs one frame at a time and
computes 35 fixed crude endpoint summaries: intensity moments and quantiles,
3-D gradients and center Laplacians, texture entropy, core/shell contrast,
neighborhood occupancy, positive-intensity centroid offsets, and normalized
space/time context. The extractor cannot read labels. The summaries are reduced
to the same eight source/target action statistics used by the DINOv2 supplier.

The exact pre-existing B0 seal (`a9ce8c0…b4ff2`) and frozen five-fold,
class-balanced L2 ridge probe (`548b681…30ca`) are reused. A1 is authentic
raw-3D; A0 is the fixed within-frame endpoint permutation control. Temporary
8-D action shards live only in `/private/tmp`, are hashed before labels open,
and are deleted after evaluation. No voxels, crops, or endpoint/full-pool cache
is persisted.

The binding advance rule is set before outcomes:

- incumbent DINOv2-XY mean AUC: `0.6704117434526917`;
- material margin: `+0.010`;
- A1 mean AUC must be at least `0.6804117434526917`; and
- the minimum of the five held-fold AUCs must also be at least
  `0.6804117434526917`.

Both clauses must pass. Retention at the CG0 budget is mandatory reporting
against the measured `0.02039355992844365` bar, not a post-hoc extra gate. A
pass opens only Stage 1 provenance authentication. Any failure PARK-Cs the
raw-3D hypothesis and consolidates the campaign. Stage 0 has a 90-minute hard
session time-box and halts after its verdict.
