# Division parent-event Colab lane

Status: preprocessing and the sanctioned one-click-resume infrastructure are ready; scientific training is blocked pending the APNCC-vs-V32 taxonomy and a measured capacity receipt. The owner has classified the longer-context temporal input/head as a new architecture, so no V32 weight-refinement waiver is required.

## Frozen facts

- The real V32 input stride is `(Z,Y,X)=(1,4,4)`, producing `(64,64,64)` uint16 frames. Z is not strided.
- The training archive contains 199 videos / 19,900 frames, 151 parent events, and 302 daughter edges. The approximate “304 divisions” figure counts daughter edges; the parent-event detector has 151 positive events.
- The image payload without `.npy` headers is exactly 10,433,331,200 bytes (9.716796875 GiB).
- Derived competition data and all trained weights remain private. No test data enters the carrier.

## Local preprocessing

```bash
.venv313/bin/python scripts/preprocess_division_colab_data.py build
.venv313/bin/python scripts/preprocess_division_colab_data.py verify
```

The build computes the full 87 GB archive SHA-256, records the ZIP inventory, reads every raw training voxel into a per-video source digest, writes only the exact strided uint16 frames and lineage labels, and publishes each video through an immutable receipt. `--resume` verifies every completed video before skipping it.

Run the focused integrity suite with:

```bash
.venv313/bin/python -m pytest -q tests/test_colab_division_harness.py tests/test_colab_division_notebook.py tests/test_preprocess_division_colab_data.py
```

## Private transport

Move the completed carrier only through storage restricted to the account owner or competition participants who accepted the rules. A Kaggle Dataset used to bring trained Colab weights back to Kaggle must be private. Keep code, config, carrier-manifest, checkpoint, and training-receipt hashes together for reproducibility.

## Colab session contract

Use `COLAB_HARNESS.ipynb` with the private carrier, an immutable source snapshot, and a persistent checkpoint root on mounted Google Drive. It contains exactly three code cells:

1. mount Drive, verify/copy the immutable source snapshot, install its frozen requirements, immediately record the assigned GPU, verify the full input inventory, pull the exact checkpoint named by the current manifest hash, recompute its semantic digest on CPU, and classify exact-runtime versus explicitly logged cross-GPU continuity;
2. invoke the frozen `run_colab_session(context)` entrypoint, whose optimizer loop must call the supplied ten-minute checkpoint callback; and
3. publish a terminal complete-state checkpoint plus the full accumulated JSONL log and close the device-specific Colab ledger receipt.

`scripts/colab_division_harness.py` supplies:

- complete model, optimizer, scheduler, scaler, Python/NumPy/Torch CPU/CUDA RNG, sampler epoch/cursor/permutation, and provenance capture;
- semantic state digests plus immutable checkpoint directories and an atomic monotonic `LATEST.json` pointer;
- ten-minute checkpoint cadence and a graceful-yield margin for a frozen 4–6 hour session target;
- the G-03 swap guard (`256 MiB` sustained for `15 s`) and full stage timing/RSS/swap/count logging;
- a fail-closed objective validator; and
- a Colab-only hours ledger that records every assigned GPU model at session start/end, rejects any Kaggle-hour entry, and refuses a naive cross-device total. If a disconnect prevents Cell 3, the next manual Cell 1 closes the open session as a clearly labelled conservative upper bound using the last attributable checkpoint plus one cadence (or the six-hour session ceiling when no checkpoint is attributable).

An exact runtime match retains the proved bit-identical-resume claim. A different assigned GPU may continue only through the logged transition policy; semantic state/provenance remain strict, but the receipt explicitly withdraws any bit-identical numerical-trajectory claim for that boundary.

There is no browser driver, activity injection, scripted session cycling, or account multiplexing. Free Colab is manual one-click resumption only; hands-off execution means Colab Pro+ background execution.

After the objective is frozen, build the minimal immutable `source_snapshot` with every local module imported by the entrypoint explicitly included:

```bash
.venv313/bin/python scripts/build_division_colab_source_snapshot.py \
  --output /path/to/private-drive/division_temporal_colab_v1/source_snapshot \
  --file path/to/frozen_training_entrypoint.py \
  --file path/to/each_imported_local_module.py
```

An existing snapshot is accepted only if it is byte-identical. Cell 1 independently rehashes its complete declared inventory before importing any project code.

The notebook fails closed today because no manifest-bound bootstrap checkpoint exists and the objective remains pending. After the taxonomy freezes the exact temporal window, loss weight, sampler, split, capacity, stopping rule, and code hash, create the step-zero bootstrap artifact explicitly. Absence or corruption of `LATEST.json` is an error and can never trigger an implicit restart.

## Capacity status

The supplied quota snapshot leaves `24.73 - 7.67 = 17.06 T4-h` unreserved after V32. No honest total-training projection exists until the architecture and schedule are frozen and timed, so the current allocation is Kaggle `0.0 h`, Colab `{}` by GPU model, and RunPod `0.0 h`. Colab is prepared as insurance but is not yet justified by a measured capacity gap. See `CAPACITY_AND_PARKED_COMPUTE_ADJUDICATION_20260805.md`.
