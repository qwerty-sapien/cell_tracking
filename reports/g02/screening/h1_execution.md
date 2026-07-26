# H1-ZWRFEAT — ready at the human upload boundary

Verdict: **READY / STOPPED BEFORE EXTERNAL MUTATION.**

The 2026-07-26 human decision decouples the feature arms. The active H1 arm is
the established points-only Trackastra interface with twelve zero-filled
`wrfeat` channels. It needs no mask input. Authentic real masks are no longer a
pre-smoke prerequisite.

The package covers exactly 105 fit-train videos: 36 `44b6` and 69 `6bba`.
Trackastra remains the frozen 27,456,880-parameter checkpoint and Stage A
(V45/APNCC) remains process-isolated from Stage B (Trackastra mmap/meta). No
Kaggle dataset, kernel, GPU job, submission, or RunPod action was performed.

## Exact human-run create and registration sequence

Run from `/Users/agent/Desktop/cell_tracking`. The two staging directories must
not already exist; the staging code refuses overwrite.

```bash
test ! -e /private/tmp/trackastra-h1-option-c-assets-v1
test ! -e /private/tmp/trackastra-h1-fit105-cache-v1
PYTHONDONTWRITEBYTECODE=1 .venv313/bin/python scripts/prepare_h1_feature_cache.py stage-assets --stage-dir /private/tmp/trackastra-h1-option-c-assets-v1
PYTHONDONTWRITEBYTECODE=1 .venv313/bin/python scripts/prepare_h1_feature_cache.py stage-cache --stage-dir /private/tmp/trackastra-h1-fit105-cache-v1
```

The following two commands are the first external mutations and are the human
boundary at which Codex stopped:

```bash
.venv313/bin/kaggle datasets create -p /private/tmp/trackastra-h1-option-c-assets-v1 --dir-mode zip
.venv313/bin/kaggle datasets create -p /private/tmp/trackastra-h1-fit105-cache-v1 --dir-mode zip
```

Do not push the kernel until both status commands return `ready` with a
non-null positive `current_version_number`, and both file listings are
nonempty:

```bash
.venv313/bin/kaggle datasets status onebitbrain/trackastra-h1-option-c-assets-v1 --format 'json(status,current_version_number)'
.venv313/bin/kaggle datasets files onebitbrain/trackastra-h1-option-c-assets-v1 --format json --page-size 200
.venv313/bin/kaggle datasets status onebitbrain/trackastra-h1-fit105-cache-v1 --format 'json(status,current_version_number)'
.venv313/bin/kaggle datasets files onebitbrain/trackastra-h1-fit105-cache-v1 --format json --page-size 200
```

That read-only check is mandatory: a successful CLI return from `datasets
create` is not accepted as registration evidence by itself.

## Exact timing-smoke command and read-only verification

After the two dataset identities pass:

```bash
.venv313/bin/kaggle kernels push -p /Users/agent/Desktop/cell_tracking/kaggle_kernel/trackastra-h1-feature-cache-prep
.venv313/bin/kaggle kernels status onebitbrain/trackastra-h1-feature-cache
.venv313/bin/kaggle kernels files onebitbrain/trackastra-h1-feature-cache --format json --page-size 200
.venv313/bin/kaggle kernels output onebitbrain/trackastra-h1-feature-cache -p /private/tmp/trackastra-h1-smoke-output -o --file-pattern '^trackastra_h1_timing_smoke\.json$'
```

The locked contract keeps `full_run_authorized=false`, so this kernel version
can execute only the timing smoke. It cannot fall through into the 105-video
cache build.

Smoke PASS requires all of the following:

- runtime witness is exactly Tesla T4, Python 3.12.13, Torch
  2.10.0+cu128/CUDA 12.8/cuDNN 91002, seed 20260725;
- explicit `cuda:0` device attestations pass for V45, edge scorer, Trackastra,
  coordinates, and features;
- both named witnesses run:
  `train:44b6_cf8fed6b` and `train:6bba_edf14583`;
- each witness’s repeated Stage-B payload is `numpy.array_equal` for all
  integer/geometry arrays and all 49 float feature columns;
- the family-wise projection
  `(36 × 44b6 witness + 69 × 6bba witness) × 1.35` is at most 7.0 h.

If the projection is above 7.0 h, the kernel writes
`STOP_PROJECTED_RUNTIME_EXCEEDS_7_HOURS` and exits nonzero. Missing datasets,
files, checkpoint/source hashes, device witnesses, or repeatability also fail
closed. Reserve quota and do not launch the full cache after any such failure.

## Proposed M5-to-T4 tolerance

Integer and geometry/identity arrays are exact-only:
`challengers`, `conflicts`, `proposal_offsets`, `transition_offsets`,
`transition_target_t`, `geometry_confidences`, and
`geometry_probabilities`. Any mismatch is structural and stops.

Each of the 49 float/neural columns is classified independently. The
preregistered probe is `atol=1e-7`, `rtol=1e-5`; an accepted proposal is
`max(1e-7, 2×observed_max_abs_error)` and
`max(1e-5, 2×observed_max_abs_error/max_scale)` per column. A coherent
whole-column offset or any value outside the probe is structural and stops.
Every per-column proposal remains
**PROPOSED_REQUIRES_HUMAN_APPROVAL** and must be frozen before graph scoring.

## 44b6-first graph runner

The runner is
`scripts/evaluate_h1_zero_wrfeat_graph_gate.py`, governed by
`h1_zero_wrfeat_graph_gate_contract.json`. It has no split/scorer-parity
argument and obtains its 36/69 identities only from the authenticated fit105
manifest. Proposal precision/support is reported but never gates promotion.

Once the complete T4 cache is downloaded and its tolerance contract is
approved, launch only after a fresh 45-second zero-swap idle preflight:

```bash
CELLTRACK_HOLDOUT_ACCESS=0 PYTHONDONTWRITEBYTECODE=1 .venv313/bin/python scripts/evaluate_h1_zero_wrfeat_graph_gate.py --contract reports/g02/screening/h1_zero_wrfeat_graph_gate_contract.json --contract-sha256 d35e2d3dbbc93de99061d0f34c7650ac85867734f44108b28ad35af665847f48
```

It executes `44b6` first. Adjusted delta `<=0` is an immediate KILL and the
`6bba` evaluator is never called. A positive result below +0.005, any raw
regression, any node-recall regression, or observation identity PARKs H1
without `6bba`. Only a complete `44b6` pass opens `6bba`; only complete passes
on both families can authorize a later full-CV node.

## Conditional H3 fallback

`H3-MASK-SOURCE` is documented but not started. It becomes eligible only after
zero-wrfeat H1 fails the `44b6` graph gate. Its first task is provenance:
establish whether authentic per-detection masks can be obtained from
competition data or a detector rerun. Pseudo-masks are forbidden. No
zero-versus-real ablation may run before authentic masks are established.

## Frozen identities

- launch contract:
  `reports/g02/screening/h1_zero_wrfeat_launch_contract.json`
- zero-wrfeat feature contract:
  `a7e866c34c15f46671ae3c30aeb4cf1107e5dd55acd63458a22890b69ff29404`
- cache manifest:
  `ff7b205f58ad70c69954fed1c3e60a100d246c1e374a849c6ea7183934e98f28`
- raw manifest:
  `665c505b4e38a03f5a501527fde58a064d3ce8fac413898656a4f3f3e6cf781e`
- source dependency manifest:
  `999db8a2a475415abc9ff5dde1d18d366c9caf40a311ddfb09af4c008df51326`
- assets manifest:
  `154580ec7e79f61401b41e319b6a05ed6b879e2703ea1323bd07c22e03fded9c`
- timing kernel:
  `03ec4df4118f0a3c4e78261827f2aedbb45df787445fd4bc836be49da1f73d8d`
- graph contract:
  `d35e2d3dbbc93de99061d0f34c7650ac85867734f44108b28ad35af665847f48`
- graph coordinator:
  `b6427e03bbdda3002285ee3983e47e62e0535aff57b5b313bc93ca6013265060`

Graph deltas remain unmeasured. GPU-hours, uploads, and submissions remain
zero. The STOP reason is the required human-executed external-mutation boundary,
not a scientific failure.
