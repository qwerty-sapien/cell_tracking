# Association cheap experiments: E1, E2, E3, E6

Status: complete. All work was CPU-only and offline. No Kaggle action, push,
training, detector inference, or submission was performed.

## Provenance boundary

The supplied verified state combines two different frozen artifacts:

- The divisions-off CV57 baseline at adjusted edge Jaccard
  `0.8711885823490648` has edge TP/FP/FN
  `34,174 / 2,736 / 2,094` and an exact raw-coordinate long-FN cohort of 317.
- The `29,676 / 2,660 / 1,810` error budget and named 336 long FNs come from
  a 52-video prefix of the source-softmax run. That artifact scores
  `0.8550656350917426` and is division-enabled (`0 / 133 / 37`).

The joint contract “57 videos, 0.8711886, divisions off, 1,810 FNs, 336 long
FNs” therefore does not exist as a single persisted run. E1, E2, and E6 use
the named 336-event cohort as requested, while explicitly retaining its
postprocessed-coordinate limitation. E3 uses the exact divisions-off CV57
baseline because it has compatible raw caches and official score artifacts.

## Decisions

| Experiment | Decision | Evidence | Action |
|---|---|---|---|
| E1 distance rank | **PASS** rank rule; **FAIL** standalone integration | All GT edges: `30,902/31,486 = 98.1452%` rank <=10. Named long cohort: `323/336 = 96.1310%`. Independent exact-raw CV57: `35,445/36,268 = 97.7308%` and `297/317 = 93.6909%`. | Do not change production yet. The exact top-10 replay stopped on video 8 because image-support projection required `3.6833 um`, above the frozen `3.2 um` bound; no official top-k score exists. |
| E2 gate re-centering | **FAIL** | Motion-predicted distance <=10 um for `38/336 = 11.3095%`; unmatched-source-only recovery is `12/73`, only `3.5714%` of the full cohort. Independent exact-raw CV57 gives `42/317 = 13.2492%`. | Do not add the second unmatched-source predicted-position gate. |
| E3 motion coefficient | **FAIL** harmful-motion rule; backward direction is better | Forward c0 is `0.848394516` versus c1 `0.871188582`. Backward c0 is the same `0.848394516` versus c1 `0.872459558`. Best is backward c=.25 at `0.873149594`, `+0.001961011` over baseline and `+0.000653426` over forward c=.25. | Do not delete motion. No default direction/weight change was made because the predeclared implementation rule was deletion-only; backward c=.25 is the next standalone candidate. |
| E6 coherent motion | **FAIL** affine implementation gate; motion-axis-bounded rule **PASS** | Leakage-safe per-frame-pair 12-DOF affine: `99/336 = 29.4643%` below 10 um. Native 3-D TV-L1 flow at `(1,4,4)` downsampling: `144/336 = 42.8571%`. | Do not implement affine motion or pursue learned flow as the primary tail fix. |

## E3 score table

| Direction | c=0 | c=.25 | c=.5 | c=1 | c=2 |
|---|---:|---:|---:|---:|---:|
| Forward | 0.848394516 | 0.872496168 | 0.871822311 | 0.871188582 | 0.870739761 |
| Backward | 0.848394516 | **0.873149594** | 0.872908906 | 0.872459558 | 0.872149947 |

The backward run verified 57/57 coverage and all 285 fragment hashes. Backward
c=0 is byte-identical to forward c=0. The process imported checkpoint
`9012886` graph code with SHA-256
`9f1e72d8378bd73950df6ea97ea54e65beda1862fae125c26f6c9ab26ac44a75`.
Concurrent division-only work changed the on-disk source during the run; the
direction attestation records both hashes. Divisions remained disabled, so the
experiment output was unaffected.

## Reproduction and verification

```bash
.venv313/bin/python reports/g02/e1_distance_rank_prefix52_v1/run.py
PYTHONPATH=src .venv313/bin/python reports/g02/e2_gate_recentering_prefix52_v1/reproduce.py
.venv313/bin/python scripts/phase1_e6_affine_tvl1_prefix52.py --force
PYTHONPATH=/opt/homebrew/lib/python3.13/site-packages:src:scripts \
  .venv313/bin/python -m pytest -q \
  tests/test_phase1_e6_prefix52.py tests/test_phase1_e3_e6.py \
  tests/test_phase1_topk_gate.py tests/test_phase1_distance_motion_gate.py
```

Focused verification result: `18 passed`.
