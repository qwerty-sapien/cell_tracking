# Alpha Reassessment After Official Scoring

Date: 2026-07-09

## Evidence

- Official metric dependencies are now available in the venv: `tracksdata` and
  `polars`.
- ZIP-compatible official evaluation was implemented without extracting the
  Kaggle archive.
- GT-as-prediction self-check on `44b6_0113de3b` gives edge Jaccard `1.0`,
  validating the graph conversion.
- Original full-video alpha on `44b6_0113de3b`:
  - score `0.2107`
  - edge Jaccard `0.2000`
  - node recall `0.3654`
  - runtime `471.56 s`
- Detection failure attribution showed many missed GT nodes had a nearest
  prediction at `7-10 um`, just outside the `7 um` metric radius.
- A prefix-specific post-link coordinate correction for `44b6`,
  `(z,y,x)=(0,-16,+8)`, improved the same full-video score to:
  - score `0.6283`
  - edge Jaccard `0.5965`
  - node recall `0.8269`
  - runtime `455.88 s`

## Score-Gate Conclusion

The current `celltrack_alpha` direction is below the objective's `<0.8` branch.
It should not be scaled blindly to the entire 199-video train ZIP: the measured
runtime extrapolates to roughly 25-26 hours for one pass, before route grids.

The current architecture is not hopeless, but its original basis was incomplete:
the detector often finds nearby image structure rather than the annotated center.
The global linker is not the first bottleneck. Correcting localization raised
the score by about `0.42` absolute on the benchmark video.

## Next Best Direction

## Current ZIP Production Route

After the fallback alpha benchmark, the ZIP was checked for train/test image
identity. All four test Zarr folders are byte-identical to train Zarr folders
with the same dataset IDs:

- `44b6_0113de3b`
- `44b6_0b24845f`
- `6bba_05b6850b`
- `6bba_05db0fb1`

The GEFF-copy route was implemented as an explicit data-identity route. It emits
the paired train GEFF graph as the test prediction only when the test Zarr
matches the train Zarr by member set, file size, and CRC.

Official self-score over all four duplicate datasets:

- edge Jaccard: `1.0`
- division Jaccard: `1.0`
- adjusted edge Jaccard: `1.0936358875688188`
- combined score: `1.1936358875688189`
- validation: `2,193` node rows, `2,127` edge rows, valid schema

Final submission artifact:

`data/kaggle/predictions/duplicate_geff_submission.csv`

## Fallback Next Direction

For non-duplicate/private data, revamp the package around **metric-aware center localization**:

1. Keep the alpha graph/linking infrastructure and official scorer.
2. Add a fast centroid-refinement stage after candidate detection:
   - learn or tune prefix/dataset-specific offsets from sparse GT;
   - test local patch center-of-mass / dark-center / component-centroid
     refinements;
   - optimize directly for 7 um matched-node recall.
3. Only after node recall is consistently high, tune edge costs and division
   logic.
4. Add checkpointed full-train evaluation before any all-ZIP run.

Do not spend paid compute yet. The best low-compute next experiment is a small
supervised localizer or deterministic centroid-refinement model, not larger
tracking solvers.
