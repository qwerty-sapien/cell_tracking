# Full-catalog visual render receipt — 2026-08-04

- Bundle: `reports/g04/post_scoring_visual_review/full_catalog_20260804/`
- Start/end (UTC): `2026-08-03T16:22:02Z` → `2026-08-03T16:26:54.797358Z` (manifest finalization); elapsed `292.797 s`.
- Command: `env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 .venv313/bin/python scripts/build_track_visualizer.py build --scope all --bundle reports/g04/post_scoring_visual_review/full_catalog_20260804`
- Result: complete, 57/57 catalog entries rendered and indexed; bundle size `90,888,617` bytes; content input SHA-256 `ff4e2372ea4d2ec265a451518f72fbb2e7174776fc17245d181d93a887929f08`.
- Defaults attested by manifest: `scope=all`; `clip_seconds=15.0`; FPS override `null` (derived per clip); HUD enabled, top-right, metrics `edge_adjusted_jaccard`, `division_jaccard`, `gt_division_count`.
- Validation: `build_track_visualizer.py validate --bundle ... --min-rendered 57` → PASS; probed `57` H.264 MP4s.
- Runtime note: the managed macOS sandbox emitted `nice: setpriority: Operation not permitted`, so the requested `nice -n 10` was attempted but OS enforcement could not be confirmed. BLAS thread caps were inherited. The sandbox also caused the script's documented process-pool fallback to threads; default renderer worker count was otherwise unchanged (`4`).
