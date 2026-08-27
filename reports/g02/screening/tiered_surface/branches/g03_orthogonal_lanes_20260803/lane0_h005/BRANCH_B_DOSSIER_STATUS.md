# Branch-B dossier status under the h005 owner amendment

Date: 2026-08-03

The audited dossier remains a truthful **FAIL** as a literature-only case for
funding 30 T4-hours. The owner amendment authorizes the fixed h005 read despite
that failure; it does not rewrite the literature or fund h030. H005 is the
cheap target-population effect-size measurement that can supersede literature
uncertainty. H030 becomes eligible only if its held-fit36 slope projects to the
held-fit36 APNCC reference under the frozen formula.

The requested MIDL quantification is explicit in the full dossier. On 2-D
Fluo-N2DL-HeLa, the best learned-embedding Trackastra result is LNK `0.9944`
versus retrained Trackastra `0.9922`, an effect of `+0.0022`. On 2-D
DIC-C2DH-HeLa, the best is `0.9898` versus `0.9859`, an effect of `+0.0039`.
Both use sequence 02 for training and sequence 01 for evaluation; each trained
configuration is repeated for eight seeds on one A100. The paper discloses
neither elapsed training time nor GPU-hours, so there is no honest T4 budget
conversion or compute-to-LNK curve. These are source-metric effects, not G-03
adjusted-graph effects.

The corrected held-fit36 h000 gaps are `+0.118434428526` to comparability with
`0.881522069960`, `+0.151912358566` to `0.915`, and `+0.181912358566` to
`0.945`. No primary source calibrates the MIDL effects, Trackastra's 2-D
feature ablations, VoxelMorph brain-registration results, or Linajea's
different detection/graph interface into those target-population gains.

Full effect tables and training-budget audit:
`reports/g03/screening/phase2_zero_init_v1/BRANCH_B_EFFECT_SIZE_DOSSIER.md`.

Primary sources:

- <https://openreview.net/forum?id=4OLrHi9ROr>
- <https://github.com/MIDL26-Short-Tracking/trackastra_y>
- <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf>
