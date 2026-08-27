# H1-PREP — frozen Trackastra two-family feature cache

Decision: **STOP / NO DEFENSIBLE CUDA RUNTIME ESTIMATE.** No GPU job, Kaggle
submission, training run, RunPod job, validation12 access, or CV57 access
occurred.

The Option-C package is locally prepared for exactly 105 fit-train videos:
**36 44b6 + 69 6bba**. It retains Stage A once / Stage B enrichment in a
separate process, uses the frozen 27.46M checkpoint, and makes T4-produced
arrays canonical. The old M5↔T4 `numpy.array_equal` requirement is absent from
the Option-C launch path. A full 105-video cache run is now fail-closed until
a human-run two-family CUDA timing smoke supports the 7-hour budget.

## Parity contract

- Canonical environment: private, internet-disabled `Tesla T4`, `cuda:0`;
  seed `20260725`; Python `3.12.13`; NumPy `2.0.2`; PyTorch
  `2.10.0+cu128`; CUDA runtime `12.8.1`; Torch CUDA `12.8`; cuDNN `91002`;
  SciPy `1.16.3`; Zarr `3.2.1`; numcodecs `0.15.1`; donfig
  `0.8.1.post1`; PyYAML `6.0.3`. Every value is an exact fail-closed check
  grounded in banked T4 evidence.
- Option C explicitly loads Stage-A V45 and its edge scorer on `cuda:0`, and
  loads Stage-B Trackastra on `cuda:0`. Trackastra coordinates, timepoints,
  zero-wrfeat features, logits, and normalized association output are required
  to remain on that device. Both stage reports carry parameter/buffer device
  attestations. Legacy v2 continues to select `cpu`.
- Same-T4 repeatability is mandatory `numpy.array_equal` for all arrays on
  named witnesses `train:44b6_cf8fed6b` and `train:6bba_edf14583`. Stage B is
  run twice for each witness and both payloads are persisted.
- M5↔T4 remains a one-time classification only. Integer/identity and geometry
  arrays are exact: `challengers`, `conflicts`, proposal/transition offsets,
  transition target times, geometry confidences, and geometry probabilities.
- For every one of the 49 float columns (`apncc_context_00` through
  `apncc_context_47`, plus `trackastra_margin`), the initial proposed probe is
  `atol=1e-7`, `rtol=1e-5`. The comparator distinguishes bounded benign FP
  drift from shape/dtype/non-finite/coherent-offset/over-tolerance structural
  drift. These values are **not final and are not enacted**; T4 measurements
  and explicit human approval are required.

## Package and identities

- Frozen checkpoint: 27,456,880 parameters; archive
  `2ede0848…12a0`; member `24c290ce…81f`; source commit
  `fb265888c8fce2ac16a362ebf182ccd5ea48e936`. The banked private dataset
  `onebitbrain/trackastra-6bba-assets-v1`, version 4, supplies these assets.
- Kernel: `trackastra_h1_feature_cache.py`,
  SHA-256 `4106c0cb…bb71`.
- Kernel metadata: SHA-256 `c950cae9…fbf8`.
- Option-C contract: SHA-256 `9f2247cf…f6b7`.
- Executed/imported source dependency manifest: 60 pinned records across
  workspace sources, the support dataset, and the Trackastra archive,
  SHA-256 `fadc47e8…ab05`. It includes the checkpoint loader and zero-wrfeat
  inference helper and is verified before either subprocess can start.
- Assets manifest: SHA-256 `2744214e…4e03`.
- Fit105 sparse-cache manifest: 210 files / 1,447,367,388 bytes,
  SHA-256 `ff7b205f…8f28`.
- Raw fit105 manifest: 10,710 files / 45,509,091,565 bytes,
  SHA-256 `665c505b…781e`. It was built from the frozen raw archive, reusing the
  banked 6bba records and hashing the 36 44b6 stores; validation12 identities
  are excluded.

The new assets and cache datasets are fully staged by local manifests, but
their remote slugs have not been created or version-verified. Referencing those
uncreated slugs is why this report does not claim launch readiness.

## Fresh zero-swap preflight

The required `/bin/sleep 45` guard passed: 45.0652 s elapsed, 0.0 MiB initial
and peak swap, **0.0 MiB growth**, 71% initial / 69% minimum free memory, and
0.00107 GiB peak process-tree RSS. Evidence is frozen in
`kaggle_kernel/trackastra-h1-feature-cache-prep/idle_preflight.json`,
SHA-256 `15bdc13e…c34`.

## Runtime and quota

The earlier **3.103 h / 4.189 h with margin estimate is retracted**. Its Stage
A and Stage B builders executed V45 and Trackastra on CPU, so it cannot bound
the corrected `cuda:0` path. There is no banked two-family Option-C CUDA timing
and therefore no defensible full-run estimate.

The packaged kernel is timing-smoke-only under the locked contract. If a human
launches it, it runs Stage A and Stage B for exactly
`train:44b6_cf8fed6b` and `train:6bba_edf14583`, requires the CUDA device
attestations and same-T4 repeatability, projects conservatively by family
counts, retains the repeated Stage-B witness cost, adds 35%, writes
`trackastra_h1_timing_smoke.json`, and stops without producing the full cache.
A full run remains forbidden until that evidence supports `<=7 h` and a human
updates the contract. Frozen inference still does not consume the training cap.

## Exact human-run sequence — do not run yet

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/agent/Desktop/cell_tracking/.venv313/bin/python /Users/agent/Desktop/cell_tracking/scripts/prepare_h1_feature_cache.py stage-assets --stage-dir /private/tmp/trackastra-h1-option-c-assets-v1
kaggle datasets create -p /private/tmp/trackastra-h1-option-c-assets-v1 --dir-mode zip
PYTHONDONTWRITEBYTECODE=1 /Users/agent/Desktop/cell_tracking/.venv313/bin/python /Users/agent/Desktop/cell_tracking/scripts/prepare_h1_feature_cache.py manifest --output /Users/agent/Desktop/cell_tracking/kaggle_kernel/trackastra-h1-feature-cache-prep/cache_dataset_manifest.json --stage-dir /private/tmp/trackastra-h1-fit105-cache-v1
kaggle datasets create -p /private/tmp/trackastra-h1-fit105-cache-v1 --dir-mode zip
kaggle kernels push -p /Users/agent/Desktop/cell_tracking/kaggle_kernel/trackastra-h1-feature-cache-prep
```

Before the final command, a human must verify both dataset versions and decide
whether to spend quota on the **timing smoke only**. Per §6.3: **reserve quota;
do not launch if the kernel cannot be verified clean before reset.** No full
cache launch command is authorized. Any eventual caches remain quarantined
until the one-time M5↔T4 classification is reviewed and the tolerance contract
is explicitly approved.

## Missing artifact and next-node contract

The H3 real-mask arm is not fabricable from current evidence. The exact missing
artifact is an authenticated fit-train105 set of per-detection instance masks
aligned to node IDs, or equivalently aligned 12-D
`wrfeat-regionprops2` arrays, for both 44b6 and 6bba. The competition Zarr
stores expose intensity images, not authenticated instance masks. Do not
substitute pseudo-masks.

After approved cache generation, H1 runs the held-fit graph gate **44b6 first**:
kill immediately if 44b6 adjusted graph delta is `<= 0`; success requires
adjusted edge Jaccard `>= +0.005` on both families with no raw-score or
node-recall regression. H3 (zeroed versus real mask features) belongs inside
that 44b6 gate once its authenticated artifact exists. H2 shell-background
abstention stays closed until H1 shows real graph-level signal.

Verification: `PYTHONDONTWRITEBYTECODE=1 .venv313/bin/python -m unittest -v
tests.test_h1_feature_cache_prep` — **12 tests passed**, including benign-FP
classification, structural geometry drift, missing-field/dtype failure,
same-T4 nondeterminism, legacy v2 compatibility, exact two-family scope,
Stage-A/Stage-B CUDA call-path enforcement, CPU-placement rejection, exact
environment/runtime gating, and transitive source/hash-pin consistency.
