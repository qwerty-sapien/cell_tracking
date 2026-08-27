# F0 N0 — literature-to-representation contract

Date: 2026-07-27  
Verdict: **PARK** (`PARK_MISSING_LOCAL_ENCODER_WEIGHTS`)  
Outcome class: grounded research bet; not yet locally falsifiable

## Frozen hypothesis card

- **Mechanism / class:** `F0-TRIVIEW-DENSE-TA`; frozen XY dense features are the
  beta, while point-centred physical XY/XZ/YZ late fusion is the alpha.
- **Causal hypothesis:** a 44b6 identity collision that is ambiguous in the XY
  projection can remain distinguishable through axial structure. Aligning all three
  views to the same 3-D point should improve association logits; breaking XZ/YZ
  point correspondence within a frame should remove that increment.
- **Crowding defense:** using a pretrained 2-D encoder is obvious beta. Fewer teams
  will force physical-unit tri-view correspondence, share the encoder and compression,
  and preregister a within-frame permutation ablation that separates extra capacity
  from correspondence.
- **Test unit / split:** observation-distinct videos in five grouped fit folds;
  44b6 held-fit graph first, then unchanged 6bba. Edge-row splitting is forbidden.
- **Primary endpoint:** frozen official
  `weighted_adjusted_edge_jaccard + 0.1 * division_jaccard`. Proposal metrics, LNK,
  AUC, and paper benchmark scores cannot promote.
- **Expected sign:** `tri-view > XY-only >= Trackastra`; the permutation arm should
  erase the non-XY increment.
- **Null:** tri-view-minus-XY and tri-view-minus-permutation are inside the exact
  rerun envelope.
- **GO / kill:** each family needs adjusted delta `>=+0.005`, raw edge and node
  deltas `>=0`, and strict tri-view-over-XY improvement. Adjusted delta `<=0`, a
  protected regression, or a permutation gain reproducing the alpha gain kills F0.
- **Budget / boundary:** local CPU feasibility only before N2. Full extraction is
  capped at 30 T4-hours and 16 GiB, but remains an N3 GPU **HUMAN-GATE**.

The exact crop, views, encoder interface, fusion head, four arms, seed, null, and
permission limits were frozen before any F0 scorer outcome in
`N0_feature_contract.json`.

## Claim-to-source boundary

| Claim | Primary source | What the source establishes | Limit for F0 |
|---|---|---|---|
| Learned dense self-supervised/foundation features can improve Trackastra linking | Dalinghaus, Archit & Pape, *Dense Embeddings from Self-Supervision and Foundation Models Improve Cell Linking Performance*, MIDL 2026 short paper, [paper](https://openreview.net/pdf?id=4OLrHi9ROr), [record](https://openreview.net/forum?id=4OLrHi9ROr) | The authors integrate learned features into Trackastra and report improved linking on two datasets; the OpenReview record labels the accepted short paper CC BY 4.0. | It does not test G-02, 3-D tri-view crops, this scorer, or this fusion head. It is evidence for the XY beta only. |
| A learned projection/compression path can turn 3-D fluorescence volumes into tractable 2-D representations for identity embeddings | Han & Lu, *ASCENT*, ICCV 2025, [paper/record](https://openaccess.thecvf.com/content/ICCV2025/html/Han_ASCENT_Annotation-free_Self-supervised_Contrastive_Embeddings_for_3D_Neuron_Tracking_in_ICCV_2025_paper.html) | ASCENT uses self-supervised contrastive learning and a positional volume-compression module before a 2-D encoder for 3-D neuron tracking. | Its data, projection, labels, and tracking regime differ. It does not validate orthogonal MIPs or G-02 transfer. |
| Trackastra can ingest object features through a learned linear token projection and reason over a spatiotemporal window | Gallusser & Weigert, *Trackastra*, ECCV 2024, [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf), §2.2 Eq. 2 | Object features are concatenated with positional encodings and linearly projected into tokens; attention predicts association probabilities. | The paper emphasizes shallow features and does not establish that dense appearance will help G-02. |
| DINOv2 supplies a frozen ViT-S/14 with patch tokens suitable for a small point-compatible beta | Oquab et al., *DINOv2*, TMLR 2024, [paper](https://arxiv.org/abs/2304.07193), [official repository](https://github.com/facebookresearch/dinov2) | A general self-supervised visual representation and official model-loading interface exist. | Natural-image pretraining is domain-shifted; no local checkpoint/license/hash is authenticated in this run. |

The tri-view claim is intentionally **not** attributed to these papers. It is a new
synthesis whose first valid evidence must be the frozen G-02 graph test.

## Exact representation and controls

- Source scale is fixed at `(Z,Y,X)=(1.625,0.40625,0.40625)` µm/voxel.
- Each point receives one `13 × 13 × 13` µm crop, resampled to a `33³`
  isotropic grid at `0.40625` µm. Border replication and trilinear float32 sampling
  are part of the representation.
- XY, XZ, and YZ each use three fixed channels: maximum projection, mean projection,
  and the central plane. They are resized to `224²`, normalized with fit-video
  q01/q99 plus the official DINOv2 normalization, and passed through the same frozen
  `dinov2_vits14`.
- Each view contributes the 384-d CLS token plus the mean central `2×2` patch token.
  One shared `768→24` projection precedes a 32,457-trainable-parameter
  pair-residual head.
  Absolute physical coordinates stay in the frozen Trackastra path; no coordinate
  channel is smuggled into pseudo-RGB.
- The four arms are exactly: frozen Trackastra, XY-only beta, tri-view alpha, and
  tri-view with XZ/YZ independently permuted within frame by a deterministic
  SHA256-derived ordering. No view, crop, encoder, or head sweep is permitted.

## License and availability audit

The target encoder is official DINOv2 ViT-S/14, chosen only for input compatibility,
patch-token access, small size, and public upstream availability—not for any G-02
outcome. The local filesystem audit found **zero compatible frozen DINO/SAM
checkpoints** in the workspace or user model caches. Existing Trackastra payloads,
CELLECT/organizer weights, and competition-trained compact heads do not satisfy this
encoder contract.

No checkpoint was downloaded and no source repository was cloned. Consequently,
the exact weight hash, pinned source revision, and weight-license text cannot be
authenticated. That fails the N0 GO requirement that the selected weights be locally
available and legally/technically usable.

## Verdict

**PARK.** The mechanism is literature-grounded and fully preregistered, but the
compatible frozen encoder/weights are not locally present. Per the N0 contract, do
not advance to the N1 runtime smoke and do not attempt the N2 ceiling audit in this
cascade.

Resume from N0 only after the owner provides or separately approves acquisition of
`dinov2_vits14_pretrain.pth`, with its SHA256, pinned source revision/tree hash,
authenticated code and weight licenses, and an offline loader test proving no network
fallback.

Proposed root-ledger row:

`GA-F0-TRIVIEW-DENSE-TA-N0-20260727 | tri-view dense Trackastra reserve |
FOUNDATION | freeze literature/encoder/crop/four-arm/null contract before outcomes |
PARK: contract complete; 0 compatible local frozen encoder checkpoints; weight hash,
source pin, and weight license unauthenticated | fit105 contract only; no labels,
graphs, CV57, GPU, or download | not scored | 0 | 0 | 0 T4-h / $0 | PARK |
Resume only with owner-provided/approved authenticated DINOv2 ViT-S/14 weights; N1-N5
closed | TOP-50 Run 01`
