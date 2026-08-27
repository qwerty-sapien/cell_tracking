# Standing global-jump preprocessing integration

Date: 2026-08-03

## Transform-once contract

`src/celltrack_alpha/frame_registration.py` is the common implementation.
For every video it loads the canonical pathology census, composes only the
preregistered `>10 um` target-aligning steps, and exposes two immutable views:

- original coordinates: raw-image / UNet-feature / computed-wrfeat sampling
  and evaluator output;
- registered coordinates: candidate topology, distance/position geometry,
  association scoring, and division proposal generation.

The receipt fixes stride `(1,4,4)`, threshold `>10 um`, cumulative application,
and exactly one application per video. Passing an already transformed view
back into the adapter fails closed.

This is frame-level preprocessing before candidate construction. It is not an
A7 per-candidate phase-correlation feature and does not reopen A7.

## Base-family seams

| Base family | Standing construction seam | Score status |
|---|---|---|
| APNCC | `scripts/g04_score_pathology_repairs.py` delegates to the common registration plan before the fixed pool / unchanged scorer | Exact CV57 measured `0.881522069960 -> 0.886764420260` (`+0.005242350300`) |
| Trackastra + computed wrfeat | `scripts/g04_trackastra_computed_wrfeat.py` aligns wrfeat against original coordinates, then uses the registered view for radius candidates, model coordinates, context selection, and association geometry; graph nodes retain original coordinates | Not scored; complete crop-wrfeat production remains closed |
| Synthetic-division-integrated | `scripts/evaluate_synth_div_sim2real.py` transforms the model/proposal coordinate view before nearest-8 daughter pairing; baseline fragments and integrated output retain original coordinates | Not scored; it still waits for an authenticated parent base |
| Public TemporalUNet V32 | `src/celltrack_alpha/temporalunet_registration.py` deterministically patches a **fresh replay**: raw coordinates remain in `_index_features`, registered coordinates enter position features, association logits, edge distances, and candidate topology, and emitted nodes remain raw | Blocked until all six immutable V32 base shards authenticate and exact base scoring completes |

The V32 replay must use a fresh package identity. It must be scored directly on
exact CV57. The APNCC `+0.005242350300` delta is not treated as transferable or
additive because V32 has different detections, logits, and candidate topology.

## Verification

Focused command:

```text
PYTHONPATH=src:scripts .venv313/bin/python -m pytest -q \
  tests/test_frame_registration.py \
  tests/test_g04_score_pathology_repairs.py \
  tests/test_g04_trackastra_computed_wrfeat.py \
  tests/test_evaluate_synth_div_sim2real.py \
  tests/test_temporalunet_registration.py \
  tests/test_g04_computed_segmentation.py
```

Result: `45 passed` (four expected legacy scikit-image/Qhull diagnostic
warnings; the crop GPU path does not compute those descriptive features).
