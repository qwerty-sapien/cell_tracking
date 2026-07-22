# CELLECT Pause Rationale

Status: paused on 2026-07-23. This is a prioritisation decision, not a retraction.

## Why paused (evidence, not preference)

- A clean public local CV scored `0.8906` with `division_jaccard` exactly `0.0`,
  while the leaderboard pack sits at `0.966`. The field reached that level on
  edges, not divisions and not embeddings. CELLECT's differentiator is not where
  the score is.
- CELLECT is estimated at 25–35 engineer-days. Cheaper directions are unexhausted:
  the metric's false-positive rule ignores most predicted edges, making recall
  nearly free, and an edge-centric attention layer has a stronger validated gain
  at an estimated 6–9 engineer-days.
- HOCT wins the Cell Tracking Challenge with 19-dimensional hand-crafted features
  and no image encoder. This argues that the marginal value of a learned
  per-voxel embedding here is lower than assumed.

## What is preserved and still correct

- **Licence boundary.** CELLECT is GPL-2.0. Competition rule 5.a.1 requires winners
  to MIT-license the source used to generate the submission, while rule 5.a.3
  exempts only input data and pretrained models, not source. Any resumption must
  therefore remain clean-room: reimplement from the publication and optionally
  load CELLECT's pretrained weights. Submission code must never import
  `external/CELLECT`.
- **Embedding and loss contract.** The contract was independently verified from
  the paper text and reference source: embeddings are unnormalised; distance is
  plain Euclidean distance, including the square root; triplet margin is `0.3`;
  and mining selects the hardest positive and hardest negative. In the reference,
  `normalize_feature` defaults to `False` at `recoloss.py:291` and is never
  overridden at `s-train-rename.py:791`. The initial draft adapter normalised
  embeddings and used squared distance; both are wrong and must remain corrected
  on resumption.
- **Architecture contract.** The paper (doi
  `10.1038/s41592-025-02886-x`) specifies a 3D U-Net with stride-2 convolutional
  downsampling, not pooling, taking two consecutive frames. For the first frame,
  branch 1 emits a centre-confidence map, a 64-channel per-voxel embedding, a
  division-probability map, and a coarse cell-size/radius estimate. Branch 2 is the
  Center Enhancement Network (CEN), a lightweight 3D U-Net that sharpens centre
  confidence; its output is merged with branch 1. Two MLPs operate over the
  spatially nearest candidates. Their inputs are embedding-vector offsets, spatial
  distances, estimated-size differences, and changes in predicted division
  probability. The intraframe MLP emits six scores (five candidates plus
  `none`) for duplicate suppression. The interframe MLP emits six scores plus a
  division prediction.
- **Candidate selection contract.** Candidate selection uses spatial distance,
  never embedding distance. The paper's ablation reports that feature-distance
  selection degrades in high-density, frequent-mitosis regimes because embedding
  supervision is sparse. The embedding is a scorer inside a spatial gate, never
  the gate. Use `k=5` by default and include `k=10` for larger displacements.
- **Radius contract.** Radius is the positive-sampling mechanism, not merely a
  feature: voxels within the specified radius are treated as belonging to the same
  cell. This dataset has point centres but neither masks nor annotated radii, so
  use a constant of approximately `4 µm` or a DoG-scale-derived radius and report
  sensitivity. The nuclei are sufficiently uniform at roughly `7–10 µm` for this
  substitution to be defensible.
- **Same-frame negatives.** Any two distinct detections in one frame are necessarily
  different cells. This supplies unlimited guaranteed hard negatives from
  unlabelled data, mineable by spatial proximity. It defuses the approximately 3%
  annotation problem and should be implemented explicitly on resumption, with the
  negative-difficulty distribution and the interaction with hardest-negative
  mining reported.

Parent–daughter pairs are positive pairs for the triplet objective, alongside
one-to-one same-cell pairs across consecutive frames.

## Resume only when

Resume only after all three conditions hold:

1. P1 (recall recalibration) and P4 (edge-centric attention) have both reported.
2. The remaining gap to the `0.966` plateau is still greater than approximately
   `0.03`.
3. No cheaper direction remains outstanding.

On resumption, treat any archived note that says embeddings should be normalised,
uses squared distance, uses pooling for downsampling, or gates candidates by
embedding distance as superseded by this document and
`PAPER_SPECIFICATION.md`.
