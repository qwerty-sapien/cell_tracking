# Cap-aware temporal proposal ranker

Decision: **PARK. Do not replay on CV57.**

The preceding temporal/background gate was thresholded before applying the
deployment policy's hard per-video proposal cap. This follow-up tested the
stronger temporal scores as rankers *inside* the exact cap:

`max(16, ceil(0.005 × selected_edges))`.

For each video, labeled proposals and sampled hard+uniform background compete
in one deterministic ranking. Background wins exact score ties. Thresholds
were selected from 257 fixed positive-score quantiles on fit videos only.

## Grouped OOF result

| scorer | correct / harmful | precision | utility |
|---|---:|---:|---:|
| pair-NCC control | 2,020 / 112 | 0.9475 | 1,796 |
| linear temporal/background | 2,410 / 158 | 0.9385 | 2,094 |
| MLP-16 temporal/background | 2,872 / 251 | 0.9196 | 2,370 |

The learned rankers add substantial recall and utility, but neither meets the
predeclared 0.95 OOF precision floor. The MLP-16 diagnostic was refit on all
105 fit-train videos only and then applied once to the fixed 12-video
validation partition:

| scorer | correct / harmful | precision | utility |
|---|---:|---:|---:|
| capped pair-NCC control | 185 / 14 | 0.9296 | 157 |
| capped MLP-16 | 208 / 23 | 0.9004 | 162 |

The validation utility gain is exactly `+5`, but it comes with nine additional
harmful swaps and a 2.9-point precision loss versus the matched control. It
therefore fails both the OOF and validation precision gates.

This result is mechanistically useful: temporal/context features do recover
additional true associations under the deployment cap, but their calibration
is unstable across videos. The next representation must improve
cross-video ranking—e.g. learned image/relative-geometry features with explicit
video-domain normalization—rather than another threshold or cap change.

Artifacts:

- `scripts/fit_selected_graph_temporal_rank_gate.py`
- `reports/g02/selected_graph_temporal_rank_gate_v1/report.json`
- `tests/test_fit_selected_graph_temporal_rank_gate.py`

No CV57 replay, Kaggle submission, GPU launch, or paid compute was used.
