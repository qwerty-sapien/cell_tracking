# CELLECT Paper-Derived Clean-Room Specification

Primary publication: *Nature Methods*, doi
[`10.1038/s41592-025-02886-x`](https://doi.org/10.1038/s41592-025-02886-x).

This is the controlling clean-room modelling specification. It is derived from the
publication. Reference-source observations below are verification evidence only;
`external/CELLECT` must remain reference-only and must not be imported by shipped
or submission code.

## Network and outputs

- Input is two consecutive 3D frames.
- The backbone is a 3D U-Net whose downsampling uses stride-2 convolutions, not
  pooling.
- For the first input frame, branch 1 predicts centre confidence, a 64-channel
  per-voxel embedding, division probability, and coarse cell size/radius.
- Branch 2 is the Center Enhancement Network (CEN), a lightweight 3D U-Net that
  sharpens confidence toward centres. Its result is merged with branch 1.
- The paper describes a `5 × 5 × 3` centre search mask with four-level hierarchical
  centre labels.

## Embedding objective

- Use unnormalised embeddings.
- Use Euclidean distance with the square root, not squared distance.
- Use triplet margin `0.3` with hardest-positive/hardest-negative mining.
- Positives include one-to-one same-cell pairs across frames and one-to-two
  parent–daughter pairs.
- The reference verification is unambiguous: `normalize_feature=False` is the
  default at `recoloss.py:291`; the call at `s-train-rename.py:791` does not
  override it; and the distance path clamps before applying `sqrt`.

## Positive and negative sampling

- Voxels within a specified physical radius are assigned to the same cell for
  positive sampling. Radius is therefore supervision geometry as well as a model
  feature.
- With point-only labels and no masks or radii, use either a constant radius near
  `4 µm` or a radius derived from the DoG scale. Report radius sensitivity.
- Treat distinct detections in the same frame as guaranteed negatives. Mine nearby
  pairs to obtain hard negatives from unlabelled frames, then inspect whether
  hardest-negative mining over-focuses on near-duplicate detections.

## Candidate gate and MLP scorers

- Form candidate sets exclusively by physical spatial distance. Embedding distance
  must never determine the gate.
- Use the five nearest spatial candidates by default. Evaluate ten candidates for
  larger interframe displacement.
- Candidate features are embedding-vector offsets, spatial distances, estimated
  size differences, and changes in predicted division probability.
- The intraframe MLP produces six scores: five candidate scores plus `none`, for
  duplicate suppression.
- The interframe MLP produces six candidate/none scores plus a division prediction.
- The paper ablation shows that embedding-distance candidate selection becomes
  worse specifically in dense, mitosis-heavy data because sparse embedding
  supervision does not support reliable global gating.

## Frozen equivalence checks already designed

The preserved fixture generator and fixtures cover fixed-seed preprocessing,
Euclidean distance (including the clamp-before-square-root path), hard mining,
margin-`0.3` triplet loss, both MLPs, a full forward pass, a matched optimiser step,
and a short matched training trajectory. Important edge cases are zero distance,
equidistant positive/negative pairs, identical embeddings, and non-finite inputs.

Numerical fixtures are data, not imported GPL source. Any packaging guard must fail
if active shipped modules import or dynamically load `external/CELLECT`.

## Preserved extraction

The publication supplement and its extracted artefacts are under `tmp/pdfs/` in
this archive:

- `cellect-supplement.pdf` and `cellect-supplement.txt`;
- individual `page-*.pdf` splits;
- rendered pages 4 and 5 used to inspect the architecture tables.
