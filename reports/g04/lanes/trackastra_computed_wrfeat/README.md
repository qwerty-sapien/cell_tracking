# G-04 Trackastra with computed wrfeat

Status: integration-ready; scientific held-fit105 and CV57 scores are waiting
for the computed-segmentation lane's complete feature manifest at
`reports/g04/lanes/computed_segmentation/wrfeat_manifest.json`.

This lane consumes **COMPUTED SEGMENTATIONS**, never H3/authentic or
organizer-provided masks. The adapter rejects authentic-mask labels, channel
reordering, missing or unmatched detections, coordinate/time drift, nonfinite
values, and silent all-zero rows. Every requested detection must have a real
computed feature vector.

## Checkpoint-native interface

The authenticated CTC Trackastra checkpoint has `feat_dim=12` and was trained
on `wrfeat-regionprops2` in this exact order:

1. equivalent diameter;
2. mean intensity after Trackastra's per-frame float32 p1/p99.8 normalization;
3. the 3x3 inertia tensor in row-major order (nine channels); and
4. Trackastra's XY-frame-border proximity feature with a five-pixel cutoff.

The segmentation lane also computes a richer, honestly labelled morphology
sidecar (volume, three axis lengths, elongation, raw intensity distribution,
and 3-D solidity) for Lane A. That sidecar is not substituted into this frozen
checkpoint: doing so would preserve the width while changing every learned
channel meaning. The native equivalent-diameter and inertia channels already
carry checkpoint-compatible size/shape information.

## Verified behavior

- `tests/test_g04_trackastra_computed_wrfeat.py`: 9 focused tests cover the
  provenance firewall, exact channel order, no-zero-fill rule, identity join,
  CV57 manifest shape, tensor injection, and outdegree-two decoder.
- `SYNTHETIC_SMOKE_RESULT.json`: exact computed tensor reached model
  `forward`, divisions were enabled, and one parent emitted two daughters.
- `AUTHENTIC_MODEL_SMOKE.json`: the byte-authenticated 27,456,880-parameter
  checkpoint accepted a finite 3x3 association inference with nonzero wrfeat on
  CPU.

## Execution

`RUN_MANIFEST.json` contains exact commands. Run held-fit105 first as the
cheapest zero-quota complete scientific read, then CV57 without any 44b6 or
legacy `+0.005` gate. Both decoders set `divisions=true`, maximum outdegree 2,
and the Trackastra source default greedy threshold `0.5`.

The held-fit105 score is a development read on a different population. The
submission comparison is the resulting CV57 combined score versus
`0.8815220699597315`; if it is higher, the artifact is a submission candidate
under the owner's amended policy.
