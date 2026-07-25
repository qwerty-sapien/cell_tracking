#!/usr/bin/env python3
"""Leakage-guarded training for the organisers' UNet + node transformer.

This entrypoint deliberately imports the model, losses, data pipeline, and
augmentations from the upstream royerlab checkout.  It adds only the machinery
needed for the frozen fold-0 experiment: exact source selection, resumable
checkpoints, bounded Kaggle sessions, and fit-only convergence monitoring.

The 57-video holdout is never opened by this script.  Prediction and patched
scoring are a separate, post-training operation after the checkpoint is fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "kaggle-cell-tracking-competition"
UPSTREAM_SRC = UPSTREAM / "src"
UPSTREAM_SCRIPTS = UPSTREAM / "scripts"
for candidate in (UPSTREAM_SRC, UPSTREAM_SCRIPTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_unet_transformer as official  # noqa: E402
from tracking_cellmot.models import TemporalUNet3D  # noqa: E402


SPLIT = ROOT / "reports" / "g02" / "grouped_3fold_audit20_v1.json"
EXPECTED_SPLIT_SHA256 = "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769"
EXPECTED_OFFICIAL_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
EXPECTED_SOURCE_HASHES = {
    "scripts/train_unet_transformer.py": "83e3f30a3313ea452b5072b118bfd06539302cf6886e317c542a602deb27d281",
    "scripts/augmentations.py": "a929e73a2d6f963f2d9dc7003d96cf7c8ff4476e33601cdc1ea92cbb6604d37b",
    "src/tracking_cellmot/models/temporal_unet.py": "d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac",
    "src/tracking_cellmot/models/simple_node_transformer.py": "b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e",
}
EXPECTED_FIT_COUNT = 117
EXPECTED_PARENT_HOLDOUT_COUNT = 58
CV57_EXCLUDED_IDENTITY = "train:6bba_f4ae811c"
SCHEMA = "celltrack-alpha.organizer-unet-transformer-f0.v1"
DEFAULT_OUT = ROOT / "reports" / "g02" / "organizer_unet_transformer_f0"


class TrainingBoundaryError(RuntimeError):
    """Raised when the frozen fit-only training boundary is not satisfied."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingBoundaryError(f"cannot load JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingBoundaryError(f"JSON root is not an object: {path}")
    return value


def frozen_scope(split_path: Path = SPLIT) -> dict[str, tuple[str, ...]]:
    """Return exact fit117, parent holdout58, terminal holdout57, and audit20."""
    observed_hash = sha256_file(split_path)
    if observed_hash != EXPECTED_SPLIT_SHA256:
        raise TrainingBoundaryError(
            f"frozen split hash mismatch: expected {EXPECTED_SPLIT_SHA256}, got {observed_hash}"
        )
    split = _load_object(split_path)
    folds = split.get("folds")
    fold = folds.get("0") if isinstance(folds, Mapping) else None
    audit = split.get("audit")
    if not isinstance(fold, Mapping) or not isinstance(audit, Mapping):
        raise TrainingBoundaryError("frozen split is missing fold 0 or audit metadata")

    fit = tuple(str(value) for value in fold.get("fit_video_ids", ()))
    parent_holdout = tuple(str(value) for value in fold.get("held_out_video_ids", ()))
    audit_ids = tuple(str(value) for value in audit.get("video_ids", ()))
    if len(fit) != EXPECTED_FIT_COUNT or len(set(fit)) != EXPECTED_FIT_COUNT:
        raise TrainingBoundaryError("fold-0 fit selection is not exactly 117 unique videos")
    if (
        len(parent_holdout) != EXPECTED_PARENT_HOLDOUT_COUNT
        or len(set(parent_holdout)) != EXPECTED_PARENT_HOLDOUT_COUNT
        or CV57_EXCLUDED_IDENTITY not in parent_holdout
    ):
        raise TrainingBoundaryError("fold-0 parent holdout is not the frozen 58-video scope")
    if set(fit) & set(parent_holdout) or set(fit) & set(audit_ids):
        raise TrainingBoundaryError("fit scope overlaps held-out or audit videos")
    if any(not value.startswith("train:") for value in (*fit, *parent_holdout, *audit_ids)):
        raise TrainingBoundaryError("frozen train identities do not use the train: prefix")

    holdout57 = tuple(value for value in parent_holdout if value != CV57_EXCLUDED_IDENTITY)
    if len(holdout57) != 57:
        raise TrainingBoundaryError("terminal holdout is not exactly 57 videos")
    return {
        "fit117": fit,
        "parent_holdout58": parent_holdout,
        "holdout57": holdout57,
        "audit20": audit_ids,
    }


def fit_paths(data_dir: Path, identities: Sequence[str]) -> list[Path]:
    """Resolve fit identities only and require paired Zarr/GEFF sources."""
    paths: list[Path] = []
    resolved_root = data_dir.resolve()
    for identity in identities:
        prefix, stem = identity.split(":", 1)
        if prefix != "train" or not stem or "/" in stem or "\\" in stem:
            raise TrainingBoundaryError(f"unsafe fit identity: {identity!r}")
        path = data_dir / stem
        if path.parent.resolve() != resolved_root:
            raise TrainingBoundaryError(f"fit identity escaped data directory: {identity!r}")
        if not path.with_suffix(".zarr").is_dir() or not path.with_suffix(".geff").exists():
            raise TrainingBoundaryError(f"missing paired fit source for {identity}")
        paths.append(path)
    return paths


def monitor_identities(fit: Sequence[str], count: int = 12) -> tuple[str, ...]:
    """Choose a deterministic, spread-out fit-only convergence monitor."""
    if count < 1 or count > len(fit):
        raise TrainingBoundaryError("monitor count must be within the fit scope")
    ordered = tuple(sorted(fit))
    indices = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
    selected = tuple(ordered[int(index)] for index in indices)
    if len(set(selected)) != count:
        raise TrainingBoundaryError("monitor selection is not unique")
    return selected


def verify_official_sources() -> dict[str, str]:
    observed = {
        relative: sha256_file(UPSTREAM / relative)
        for relative in EXPECTED_SOURCE_HASHES
    }
    if observed != EXPECTED_SOURCE_HASHES:
        raise TrainingBoundaryError(
            f"official source identity mismatch: expected {EXPECTED_SOURCE_HASHES}, got {observed}"
        )
    return observed


def _normalised_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key.replace("unet.module.", "unet.", 1): value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def _load_model_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    if isinstance(getattr(model, "unet", None), nn.DataParallel):
        state = {
            (key.replace("unet.", "unet.module.", 1) if key.startswith("unet.") else key): value
            for key, value in state.items()
        }
    model.load_state_dict(state, strict=True)


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    os.replace(temporary, path)


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    best_epoch: int,
    history: Sequence[Mapping[str, Any]],
    loader_generator: torch.Generator,
    provenance: Mapping[str, Any],
    best_model_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "epoch": int(epoch),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "model_state_dict": _normalised_state_dict(model),
        "best_model_state_dict": dict(best_model_state_dict),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": list(history),
        "loader_generator_state": loader_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps_rng_state": (
            torch.mps.get_rng_state()
            if torch.backends.mps.is_available()
            else None
        ),
        "provenance": dict(provenance),
    }


def resolve_device(requested: str) -> torch.device:
    """Resolve an accelerated training device without silently falling back to CPU."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise TrainingBoundaryError("armed training requires CUDA or MPS")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise TrainingBoundaryError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise TrainingBoundaryError("MPS was requested but is unavailable")
        return torch.device("mps")
    raise TrainingBoundaryError(f"unsupported training device: {requested!r}")


def configure_accelerator_memory(
    device: torch.device,
    *,
    mps_memory_limit_gib: float | None,
) -> dict[str, Any]:
    """Apply the requested hard MPS allocator ceiling before model allocation."""
    if mps_memory_limit_gib is not None and mps_memory_limit_gib <= 0:
        raise TrainingBoundaryError("MPS memory limit must be positive")
    if device.type != "mps":
        if mps_memory_limit_gib is not None:
            raise TrainingBoundaryError("an MPS memory limit requires the MPS device")
        return {"device": device.type}

    recommended = int(torch.mps.recommended_max_memory())
    if recommended <= 0:
        raise TrainingBoundaryError("MPS did not report a recommended memory limit")
    limit_bytes = (
        int(mps_memory_limit_gib * 2**30)
        if mps_memory_limit_gib is not None
        else recommended
    )
    fraction = limit_bytes / recommended
    if fraction > 2.0:
        raise TrainingBoundaryError("requested MPS memory limit exceeds the allocator range")
    torch.mps.set_per_process_memory_fraction(fraction)
    return {
        "device": "mps",
        "mps_recommended_memory_bytes": recommended,
        "mps_memory_limit_bytes": limit_bytes,
        "mps_memory_fraction": fraction,
    }


def _accelerator_memory(device: torch.device) -> dict[str, int] | None:
    if device.type == "mps":
        return {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        }
    if device.type == "cuda":
        return {
            "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "driver_allocated_bytes": int(torch.cuda.memory_reserved(device)),
        }
    return None


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    **kwargs: Any,
) -> tuple[float, float]:
    """Run the hash-locked organiser loop with the appropriate device synchronizer."""
    if device.type == "cuda":
        return official.train_epoch(model, loader, optimizer, device, **kwargs)

    # Upstream uses ``torch.cuda.synchronize`` only for timing barriers.  Keep
    # every model/loss/update operation unchanged while substituting the MPS
    # timing barrier for this single-threaded call.
    original_synchronize = torch.cuda.synchronize
    torch.cuda.synchronize = torch.mps.synchronize  # type: ignore[method-assign]
    try:
        return official.train_epoch(model, loader, optimizer, device, **kwargs)
    finally:
        torch.cuda.synchronize = original_synchronize  # type: ignore[method-assign]


def validate_resume(checkpoint: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    if checkpoint.get("schema") != SCHEMA:
        raise TrainingBoundaryError("resume checkpoint has the wrong schema")
    recorded = checkpoint.get("provenance")
    if not isinstance(recorded, Mapping):
        raise TrainingBoundaryError("resume checkpoint lacks provenance")
    locked_keys = (
        "split_sha256",
        "fit_selection_sha256",
        "official_commit",
        "training_source_sha256",
        "augmentation_source_sha256",
        "official_source_hashes",
        "seed",
        "model_config",
    )
    for key in locked_keys:
        if recorded.get(key) != provenance.get(key):
            raise TrainingBoundaryError(f"resume checkpoint provenance mismatch for {key}")
    boundary = recorded.get("data_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("fit_only") is not True:
        raise TrainingBoundaryError("resume checkpoint does not attest fit-only training")
    if any(boundary.get(key) is not False for key in ("holdout_used", "audit_used", "test_used")):
        raise TrainingBoundaryError("resume checkpoint reports non-fit data use")


def convergence_status(
    history: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    min_epochs: int,
    patience_epochs: int,
    min_delta: float,
) -> tuple[bool, str]:
    eval_rows = [row for row in history if row.get("monitor_score") is not None]
    if epoch < min_epochs or len(eval_rows) < 2:
        return False, "minimum_epochs_not_reached"
    # ``best_epoch`` is updated only when the monitored score clears
    # ``min_delta``.  Reading it from the latest evaluation therefore measures
    # meaningful-improvement patience rather than noise around a flat maximum.
    best_epoch = int(eval_rows[-1].get("best_epoch", eval_rows[-1]["epoch"]))
    stale_epochs = epoch - best_epoch
    if stale_epochs >= patience_epochs:
        return True, f"fit_monitor_plateau_{stale_epochs}_epochs"
    return False, f"fit_monitor_patience_{stale_epochs}_of_{patience_epochs}"


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_epochs < 1 or args.eval_every < 1 or args.max_hours <= 0:
        raise TrainingBoundaryError("epoch, evaluation, and time limits must be positive")
    device = resolve_device(args.device)
    accelerator = configure_accelerator_memory(
        device,
        mps_memory_limit_gib=args.mps_memory_limit_gib,
    )

    scope = frozen_scope(args.split_manifest)
    source_hashes = verify_official_sources()
    fit_ids = scope["fit117"]
    monitor_ids = monitor_identities(fit_ids, args.monitor_videos)
    train_files = fit_paths(args.data_dir, fit_ids)
    monitor_files = fit_paths(args.data_dir, monitor_ids)
    fit_hash = hashlib.sha256(canonical_bytes(list(fit_ids))).hexdigest()

    config = {
        "unet_out_channels": 32,
        "unet_layers": [32, 64, 128],
        "downsample": [1, 4, 4],
        "window_size": 2,
        "pool_kernel_um": 5.0,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "det_loss_weight": 1.0,
        "det_neg_weight": 0.01,
    }
    provenance = {
        "schema": SCHEMA,
        "split_sha256": sha256_file(args.split_manifest),
        "fit_selection_sha256": fit_hash,
        "fit_video_count": len(fit_ids),
        "monitor_video_ids": list(monitor_ids),
        "official_repository": "https://github.com/royerlab/kaggle-cell-tracking-competition",
        "official_commit": EXPECTED_OFFICIAL_COMMIT,
        "training_source_sha256": source_hashes["scripts/train_unet_transformer.py"],
        "augmentation_source_sha256": source_hashes["scripts/augmentations.py"],
        "official_source_hashes": source_hashes,
        "augmentations": [function.__name__ for function in official.DEFAULT_AUGMENTATIONS],
        "seed": args.seed,
        "model_config": config,
        "initialization": "random_scratch_no_released_or_foreign_checkpoint",
        "runtime": accelerator,
        "data_boundary": {
            "fit_only": True,
            "holdout_used": False,
            "audit_used": False,
            "test_used": False,
        },
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    elif device.type == "mps":
        torch.mps.manual_seed(args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed)

    def load(files: Sequence[Path], label: str):
        rows = []
        for index, path in enumerate(files, start=1):
            print(f"[{label}] metadata {index}/{len(files)} {path.name}", flush=True)
            rows.append(official.load_dataset_windows(path, window_size=2, downsample=(1, 4, 4)))
        return rows

    train_video_data = load(train_files, "fit117")
    monitor_video_data = load(monitor_files, "fit-monitor")
    all_windows = [window for _, windows in train_video_data for window in windows]
    if not all_windows:
        raise TrainingBoundaryError("fit scope produced no training windows")
    max_nodes = max(max(window.node_counts) for window in all_windows)
    print(f"fit videos={len(fit_ids)} windows={len(all_windows)} max_nodes={max_nodes}", flush=True)

    train_dataset = official.FrameWindowDataset(
        train_video_data,
        max_nodes=max_nodes,
        augmentations=official.DEFAULT_AUGMENTATIONS,
    )
    monitor_dataset = official.FrameWindowDataset(monitor_video_data, max_nodes=max_nodes)

    def worker_init(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=False,
        generator=loader_generator,
        worker_init_fn=worker_init,
    )
    monitor_loader = DataLoader(
        monitor_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=False,
        worker_init_fn=worker_init,
    )

    unet = TemporalUNet3D(in_channels=1, out_channels=32, layers=(32, 64, 128))
    model = official.UNetNodeTransformer(unet=unet, unet_out_channels=32, pos_feat_dim=32).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and not args.single_gpu:
        model.unet = nn.DataParallel(model.unet)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_score = -math.inf
    best_epoch = 0
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise TrainingBoundaryError("resume checkpoint is not a mapping")
        validate_resume(checkpoint, provenance)
        _load_model_state(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        best_model_state_dict = dict(checkpoint["best_model_state_dict"])
        history = [dict(row) for row in checkpoint.get("history", ())]
        loader_generator.set_state(checkpoint["loader_generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if device.type == "mps" and checkpoint.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng_state"])
        print(f"resumed epoch={start_epoch} best={best_score:.8f} at {best_epoch}", flush=True)
    elif args.require_resume:
        raise TrainingBoundaryError("this stage requires a validated resume checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(provenance, args.output_dir / "provenance.json")
    _atomic_json(
        {
            "unet_out_channels": config["unet_out_channels"],
            "unet_layers": config["unet_layers"],
            "downsample": config["downsample"],
            "window_size": config["window_size"],
            "pool_kernel_um": config["pool_kernel_um"],
        },
        args.output_dir / "config.json",
    )
    if best_model_state_dict is not None:
        _atomic_torch_save(best_model_state_dict, args.output_dir / "edge_predictor_best.pth")
    deadline = time.monotonic() + args.max_hours * 3600.0
    stop_reason = "target_epochs_reached"
    converged = False
    completed_epoch = start_epoch

    for epoch in range(start_epoch + 1, args.target_epochs + 1):
        if time.monotonic() >= deadline - args.epoch_safety_minutes * 60.0:
            stop_reason = "session_time_cap_before_epoch"
            break
        epoch_start = time.monotonic()
        edge_loss, det_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            det_loss_weight=1.0,
            det_neg_weight=0.01,
            pool_kernel_um=5.0,
        )
        monitor_loss = monitor_acc = monitor_recall = monitor_score = None
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.target_epochs:
            monitor_loss, monitor_acc, monitor_recall = official.evaluate(
                model, monitor_loader, device, pool_kernel_um=5.0
            )
            monitor_score = monitor_acc * monitor_recall
            if monitor_score >= best_score + args.min_delta:
                best_score = monitor_score
                best_epoch = epoch
                best_model_state_dict = _normalised_state_dict(model)
                _atomic_torch_save(best_model_state_dict, args.output_dir / "edge_predictor_best.pth")

        row = {
            "epoch": epoch,
            "edge_loss": edge_loss,
            "det_loss": det_loss,
            "combined_train_loss": edge_loss + det_loss,
            "monitor_loss": monitor_loss,
            "monitor_acc": monitor_acc,
            "monitor_recall": monitor_recall,
            "monitor_score": monitor_score,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "epoch_seconds": time.monotonic() - epoch_start,
            "device": device.type,
            "accelerator_memory": _accelerator_memory(device),
        }
        history.append(row)
        completed_epoch = epoch
        converged, convergence_reason = convergence_status(
            history,
            epoch=epoch,
            min_epochs=args.min_epochs,
            patience_epochs=args.patience_epochs,
            min_delta=args.min_delta,
        )
        if best_model_state_dict is None:
            raise TrainingBoundaryError("first evaluation did not produce a best checkpoint")
        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_score=best_score,
            best_epoch=best_epoch,
            history=history,
            loader_generator=loader_generator,
            provenance=provenance,
            best_model_state_dict=best_model_state_dict,
        )
        _atomic_torch_save(checkpoint, args.output_dir / "checkpoint_last.pth")
        _atomic_json(history, args.output_dir / "history.json")
        print(json.dumps({**row, "converged": converged, "convergence_reason": convergence_reason}), flush=True)
        if converged:
            stop_reason = convergence_reason
            break

    result = {
        "schema": SCHEMA,
        "status": "converged" if converged else ("target_complete" if completed_epoch >= args.target_epochs else "resumable"),
        "stop_reason": stop_reason,
        "epoch": completed_epoch,
        "target_epochs": args.target_epochs,
        "best_epoch": best_epoch,
        "best_fit_monitor_score": best_score,
        "fit_video_count": len(fit_ids),
        "monitor_video_count": len(monitor_ids),
        "device": device.type,
        "accelerator": accelerator,
        "holdout_evaluated": False,
        "data_boundary": provenance["data_boundary"],
        "checkpoint_last_sha256": sha256_file(args.output_dir / "checkpoint_last.pth"),
        "best_checkpoint_sha256": sha256_file(args.output_dir / "edge_predictor_best.pth"),
    }
    _atomic_json(result, args.output_dir / "run.json")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--require-resume", action="store_true")
    parser.add_argument("--target-epochs", type=int, default=500)
    parser.add_argument("--min-epochs", type=int, default=300)
    parser.add_argument("--patience-epochs", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--monitor-videos", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-hours", type=float, default=11.5)
    parser.add_argument("--epoch-safety-minutes", type=float, default=20.0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps"), default="auto")
    parser.add_argument("--mps-memory-limit-gib", type=float)
    parser.add_argument("--single-gpu", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
