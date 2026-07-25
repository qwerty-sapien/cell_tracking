#!/usr/bin/env python3
"""Fit-only MPS preflight for the organisers' UNet + node transformer.

This benchmark opens one identity that is already in the frozen fold-0 fit set,
loads the validated epoch-7 checkpoint, and performs one optimiser update for
each requested physical batch size. It never resolves or opens the holdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts" / "train_organizer_f0.py"
DEFAULT_DATA = ROOT / "data" / "derived" / "t03_fold0_fit10_images_v1" / "train"
DEFAULT_CHECKPOINT = (
    ROOT
    / "reports"
    / "g02"
    / "organizer_unet_transformer_f0"
    / "stage00_v3"
    / "organizer_f0"
    / "checkpoint_last.pth"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "g02"
    / "organizer_unet_transformer_f0"
    / "mps_preflight.json"
)
DEFAULT_FIT_IDENTITY = "train:6bba_d3da753b"
FULL_FIT_WINDOWS = 11068


def load_trainer():
    spec = importlib.util.spec_from_file_location("organizer_f0_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import trainer from {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not parsed or any(size < 1 for size in parsed):
        raise argparse.ArgumentTypeError("batch sizes must all be positive")
    return parsed


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    trainer = load_trainer()
    torch = trainer.torch
    official = trainer.official

    scope = trainer.frozen_scope(args.split_manifest)
    if args.fit_identity not in scope["fit117"]:
        raise trainer.TrainingBoundaryError("benchmark identity is not in frozen fit117")
    trainer.verify_official_sources()
    fit_path = trainer.fit_paths(args.data_dir, (args.fit_identity,))[0]

    device = trainer.resolve_device("mps")
    accelerator = trainer.configure_accelerator_memory(
        device,
        mps_memory_limit_gib=args.memory_limit_gib,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise trainer.TrainingBoundaryError("benchmark checkpoint is not a mapping")
    trainer.validate_resume(checkpoint, checkpoint.get("provenance", {}))
    recorded = checkpoint["provenance"]
    if recorded.get("split_sha256") != trainer.sha256_file(args.split_manifest):
        raise trainer.TrainingBoundaryError("benchmark checkpoint split identity mismatch")

    video_data = [
        official.load_dataset_windows(
            fit_path,
            window_size=2,
            downsample=(1, 4, 4),
        )
    ]
    windows = [window for _, items in video_data for window in items]
    if not windows:
        raise trainer.TrainingBoundaryError("benchmark fit video produced no windows")
    max_nodes = max(max(window.node_counts) for window in windows)

    rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        random.seed(args.seed)
        trainer.np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.mps.manual_seed(args.seed)

        dataset = official.FrameWindowDataset(
            video_data,
            max_nodes=max_nodes,
            augmentations=official.DEFAULT_AUGMENTATIONS,
        )
        loader = trainer.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        unet = trainer.TemporalUNet3D(
            in_channels=1,
            out_channels=32,
            layers=(32, 64, 128),
        )
        model = official.UNetNodeTransformer(
            unet=unet,
            unet_out_channels=32,
            pos_feat_dim=32,
        ).to(device)
        trainer._load_model_state(model, checkpoint["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.mps.empty_cache()

        started = time.monotonic()
        row: dict[str, Any] = {"batch_size": batch_size}
        try:
            edge_loss, det_loss = trainer._train_epoch(
                model,
                loader,
                optimizer,
                device,
                det_loss_weight=1.0,
                det_neg_weight=0.01,
                pool_kernel_um=5.0,
                max_iters=args.iterations,
            )
            elapsed = time.monotonic() - started
            seconds_per_update = elapsed / args.iterations
            memory = trainer._accelerator_memory(device) or {}
            estimated_epoch_seconds = seconds_per_update * math.ceil(
                FULL_FIT_WINDOWS / batch_size
            )
            row.update(
                {
                    "status": "passed",
                    "iterations": args.iterations,
                    "seconds_per_update": seconds_per_update,
                    "estimated_full_epoch_seconds": estimated_epoch_seconds,
                    "estimated_epochs_per_7h": 7 * 3600 / estimated_epoch_seconds,
                    "edge_loss": edge_loss,
                    "det_loss": det_loss,
                    "accelerator_memory": memory,
                }
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            row.update(
                {
                    "status": "oom",
                    "seconds_before_failure": time.monotonic() - started,
                    "error": str(exc),
                }
            )
        rows.append(row)
        del optimizer, model, loader, dataset
        torch.mps.empty_cache()

    result = {
        "schema": "celltrack-alpha.organizer-f0-mps-preflight.v1",
        "fit_only": True,
        "fit_identity": args.fit_identity,
        "holdout_accessed": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": trainer.sha256_file(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "full_fit_windows_from_stage0": FULL_FIT_WINDOWS,
        "accelerator": accelerator,
        "rows": rows,
    }
    trainer._atomic_json(result, args.output)
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fit-identity", default=DEFAULT_FIT_IDENTITY)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=(1, 2, 4, 8))
    parser.add_argument("--memory-limit-gib", type=float, default=6.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.split_manifest is None:
        args.split_manifest = ROOT / "reports" / "g02" / "grouped_3fold_audit20_v1.json"
    return args


def main() -> int:
    benchmark(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
