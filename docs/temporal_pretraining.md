# Fold-safe alpha-pseudolink temporal adaptation

This is the G-02 replacement for the rejected same-voxel teacher-distillation experiment. It adapts the exact released `TemporalUNet3D` using persisted sparse alpha graph decisions. Only the two parameterized `temporal_blocks` are trainable (83,072 parameters); the spatial encoder/decoder, BatchNorm parameters and running state remain fixed. There is no teacher forward and no new learned head.

## Sparse objective

For a persisted adjacent-frame alpha link from source coordinate `s` to migrated target coordinate `d`, the runner gathers four feature vectors:

- positive: `cos(f_t(s), f_t+1(d))`
- forward stale negative: `cos(f_t(s), f_t+1(s))`
- reverse stale negative: `cos(f_t+1(d), f_t(d))`

The migration loss is the confidence-weighted mean of the two margin-ranking hinges. Links whose grid coordinate does not change are rejected, so the positive cannot collapse back to the old same-voxel target. A cheap L2 trust region anchors only the trainable temporal parameters to their released values.

An optional division motif is available but off by default (`--division-weight 0`). For a persisted alpha parent with exactly two candidate-backed children, it aligns the parent to the normalized child centroid and applies a sibling-cosine ceiling. Windows without a division remain valid when the objective is enabled.

The loss performs direct indexed gathers only. Its work beyond the normal dense U-Net forward is `O((K+D)*C)` for `K` declared links, `D` declared division motifs, and feature width `C`. It performs no pair discovery, `cdist`, all-cell comparison, or spatial/object `N x N` allocation. The released U-Net still has its ordinary length-two temporal attention at each voxel; the discarded dense association transformer is not loaded.

## Alpha provenance contract

Every selected source must have a complete persisted alpha bundle. The loader verifies, before image access:

- the source is one canonical `train:<stem>` member of the current fold's immutable fit set;
- its group, video, exact image identity, and image shape agree with `dataset_registry_v1.json`;
- cache metadata and intent are self-consistent and the cache SHA-256 matches;
- the inference contract is self-digested and binds the exact image identity and alpha model hashes;
- status, feature, transaction, fragment, and run-contract identities and digests agree;
- the run is graph variant A0 with its locked physical `cKDTree` candidate gate, finite candidate-only COO probabilities, and `full_frame_dense_matrix=false` attestations;
- the sparse cache covers every adjacent frame and contains `FrameDet`/`PairContext` records but no persisted dense edge tensor;
- fragment edges are unique, adjacent, merge-free, and have out-degree at most two.

Training retains only selected fragment edges whose endpoints both exist in the candidate cache. Synthesized endpoints are ignored. Migration positives must exceed `--min-migration-um`, change grid coordinate, and pass the minimum endpoint detection confidence. Per-window link and division caps are part of checkpoint identity.

The checkpoint binds the fold, immutable split/provenance/registry hashes, exact selected train sources, every alpha artifact digest, alpha and released model identities, objective, hyperparameters, and trainable scope. Restore rejects old same-voxel checkpoints, another fold or source scope, artifact identity changes, and any modification to a frozen released-backbone tensor.

## Leakage and execution boundary

- Only fold-fit competition training images and their image-derived alpha artifacts are allowed.
- Held-out folds, the untouched audit set, competition test data, GEFF labels/tracks, Kaggle, network services, and external GPUs are not used.
- Image stores are opened lazily with `load_image=False` and `require_tracks=False`; only alpha-declared adjacent windows are read.
- Execution is explicitly armed, locally time/step bounded, deterministic, and resumable by exact identity.
- Local W&B-compatible JSON sidecars are written without importing W&B or making a network call.

## Local validation and smoke

Use a group that is both in the selected fold's fit set and present in the supplied alpha artifact root. Dry-run validates the complete provenance bundle but does not open image voxels or train:

```bash
PYTHONPATH=src:scripts:external/kaggle-cell-tracking-competition/src \
  .venv313/bin/python scripts/pretrain_temporal.py \
  --fold 0 \
  --group g02-f2dd7ac03d54b37c10d2 \
  --alpha-artifact-root kaggle_kernel/bh9o7/bh9out \
  --output-dir /private/tmp/temporal-pseudolink-dry \
  --dry-run
```

A bounded one-step local mechanism smoke additionally supplies the extracted fit-only image root and explicit arming:

```bash
PYTHONPATH=src:scripts:external/kaggle-cell-tracking-competition/src \
  .venv313/bin/python scripts/pretrain_temporal.py \
  --fold 0 \
  --group g02-f2dd7ac03d54b37c10d2 \
  --alpha-artifact-root kaggle_kernel/bh9o7/bh9out \
  --data-root data/derived/t03_fold0_fit10_images_v1/train \
  --output-dir /private/tmp/temporal-pseudolink-smoke \
  --max-steps 1 \
  --checkpoint-every-steps 1 \
  --maximum-hours 0.05 \
  --armed
```

The checkpoint can then be supplied to the fold-matched supervised trainer with `scripts/train_ta.py --pretrain-checkpoint ...`.

## Evidence boundary

A finite loss, successful backward pass, temporal-only update, or short pilot is mechanism and integrity evidence only. It is not evidence of better tracking. No improvement or promotion claim is permitted without a fresh equal-budget paired supervised comparison against the released initialization under the same fold, sources, seed, optimizer, steps, labels, and evaluation protocol.
