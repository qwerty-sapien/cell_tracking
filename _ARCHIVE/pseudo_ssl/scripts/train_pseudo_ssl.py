"""Bounded fit-only Mean-Teacher experiment for the G-02 sparse tracker.

This is an explicitly armed local/private-compute entrypoint.  It has no
network, Kaggle, W&B, submission, or dense-association code path.
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
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    ROOT / "src",
    ROOT / "external" / "kaggle-cell-tracking-competition" / "src",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from celltrack_alpha.pseudo_ssl import (
    SSL_SCHEMA,
    AcceptanceLedger,
    AdaptiveThresholdCurriculum,
    BatchProvenance,
    EMATeacherState,
    FitOnlyScope,
    FlipTransform,
    PseudoSSLError,
    extract_sparse_pseudo_graph,
    permitted_trainable_names,
    select_pseudo_graph,
    sparse_pseudo_losses,
)
from celltrack_alpha.t_adapter import (
    CONFIG_SHA256,
    EXPECTED_CONFIG,
    WEIGHTS_SHA256,
    TemporalAdapterModel,
    load_released_core,
    verify_released_files,
)


SPLIT_SHA256 = "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769"
PROVENANCE_SHA256 = "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f"
REGISTRY_SHA256 = "21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158"
EXPECTED_HASHES = {
    "split_sha256": SPLIT_SHA256,
    "provenance_sha256": PROVENANCE_SHA256,
    "registry_sha256": REGISTRY_SHA256,
}
MAXIMUM_HOURS = 2.0
MAXIMUM_STEPS = 10_000
MAXIMUM_EPOCHS = 10
LOSS_KEYS = ("total", "detection", "continuation", "motion", "division")


class LaunchError(ValueError):
    """Raised before or during a bounded launch that violates the milestone."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _atomic_checkpoint(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_partition_sha256(model: torch.nn.Module, trainable_names: set[str], *, trainable: bool) -> str:
    digest = hashlib.sha256()
    found = 0
    for name, value in model.state_dict().items():
        selected = name in trainable_names
        if selected != trainable:
            continue
        tensor = value.detach().cpu().contiguous()
        metadata = _canonical_bytes({"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)})
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.numpy().tobytes(order="C"))
        found += 1
    if found == 0:
        raise LaunchError("empty parameter/state partition")
    return digest.hexdigest()


def _build_model(args: argparse.Namespace, device: torch.device) -> TemporalAdapterModel:
    from tracking_cellmot.models.temporal_unet import TemporalUNet3D

    backbone = TemporalUNet3D(
        in_channels=1,
        out_channels=EXPECTED_CONFIG["unet_out_channels"],
        layers=EXPECTED_CONFIG["unet_layers"],
    )
    detector, _ = load_released_core(backbone, args.weights_path, args.config_path)
    model = TemporalAdapterModel(backbone, detector)
    model.set_training_policy(0, detector_policy="released_teacher_distill")
    allowed = ("detect_head.", "aux_head.", "backbone.temporal_blocks.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(name.startswith(prefix) for prefix in allowed)
    permitted_trainable_names(model)
    return model.to(device)


def _hold_batchnorm_fixed(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def _quantile(quantiles: Mapping[str, Any], target: float) -> float:
    for key, value in quantiles.items():
        try:
            if abs(float(key) - target) < 1e-9 and math.isfinite(float(value)):
                return float(value)
        except (TypeError, ValueError):
            continue
    raise LaunchError("image normalization quantile is missing")


class FitImageWindows:
    """Stream whole two-frame slabs from prevalidated fit-only image stores."""

    def __init__(
        self,
        data_root: Path,
        scope: FitOnlyScope,
        records: Sequence[BatchProvenance],
        *,
        epochs: int,
        completed_windows: int,
        deadline: float,
    ) -> None:
        try:
            self.data_root = data_root.resolve(strict=True)
        except OSError as exc:
            raise LaunchError("fit image root does not exist") from exc
        if not self.data_root.is_dir():
            raise LaunchError("fit image root is not a directory")
        self.scope = scope
        self.records = tuple(records)
        self.epochs = epochs
        self.completed_windows = completed_windows
        self.deadline = deadline
        self.downsample = tuple(int(value) for value in EXPECTED_CONFIG["downsample"])
        self.stats = {"considered": 0, "read": 0, "videos": len(self.records)}

    def _metadata(self, record: BatchProvenance) -> dict[str, Any]:
        from tracking_cellmot.io import open_dataset

        record.validate(self.scope)
        stem = record.video_id.removeprefix("train:")
        try:
            zarr_path = (self.data_root / f"{stem}.zarr").resolve(strict=True)
            zarr_path.relative_to(self.data_root)
        except (OSError, ValueError) as exc:
            raise LaunchError(f"missing or escaping fit image store: {stem}") from exc
        if not zarr_path.is_dir() or not (zarr_path / "zarr.json").is_file() or not (zarr_path / "0" / "zarr.json").is_file():
            raise LaunchError(f"fit image store is malformed: {stem}")
        try:
            dataset = open_dataset(
                self.data_root / stem,
                load_image=False,
                require_tracks=False,
                normalize=False,
                device="cpu",
            )
            observed = Path(dataset.zarr_path).resolve(strict=True)
        except Exception as exc:
            raise LaunchError(f"fit image metadata is unreadable: {stem}") from exc
        if observed != zarr_path or dataset.image is not None or dataset.tracks is not None:
            raise LaunchError("image-only source unexpectedly exposed labels or eager image data")
        shape = tuple(int(value) for value in (dataset.image_shape or ()))
        if len(shape) != 4 or shape[0] < 2 or any(value < 1 for value in shape):
            raise LaunchError("fit image shape is invalid")
        low, high = _quantile(dataset.quantiles, 0.001), _quantile(dataset.quantiles, 0.999)
        if not high > low:
            raise LaunchError("fit image normalization range is invalid")
        scale = tuple(float(unit) * stride for unit, stride in zip(dataset.scale, self.downsample, strict=True))
        if len(scale) != 3 or any(not math.isfinite(value) or value <= 0 for value in scale):
            raise LaunchError("fit video has invalid physical scale")
        return {"path": zarr_path, "frames": shape[0], "low": low, "high": high, "scale": scale}

    def _read(self, metadata: Mapping[str, Any], start: int) -> torch.Tensor:
        import numpy as np
        import zarr

        dz, dy, dx = self.downsample
        try:
            store = zarr.open_group(str(metadata["path"]), mode="r")["0"]
            raw = store[start : start + 2, ::dz, ::dy, ::dx].astype(np.float32)
        except Exception as exc:
            raise LaunchError("fit image slab read failed") from exc
        if raw.ndim != 4 or raw.shape[0] != 2 or not np.isfinite(raw).all():
            raise LaunchError("fit image slab is missing or nonfinite")
        normalized = ((raw - metadata["low"]) / (metadata["high"] - metadata["low"])).clip(0.0, 1.0)
        return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(2)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        position = 0
        for epoch in range(1, self.epochs + 1):
            for record in self.records:
                if time.monotonic() >= self.deadline:
                    return
                metadata = self._metadata(record)
                for start in range(int(metadata["frames"]) - 1):
                    self.stats["considered"] += 1
                    if time.monotonic() >= self.deadline:
                        return
                    if position >= self.completed_windows:
                        frames = self._read(metadata, start)
                        self.stats["read"] += 1
                        yield {
                            "frames": frames,
                            "scale_zyx_um": metadata["scale"],
                            "provenance": record,
                            "window_id": f"{record.group_id}:{start}",
                            "epoch": epoch,
                        }
                    position += 1


def _augment(
    frames: torch.Tensor,
    *,
    generator: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, FlipTransform]:
    flips = tuple(bool(generator.getrandbits(1)) for _ in range(3))
    transform = FlipTransform(tuple(int(value) for value in frames.shape[-3:]), flips)
    weak = (frames + 0.01 * torch.randn_like(frames)).clamp(0.0, 1.0)
    gamma = generator.uniform(0.7, 1.5)
    gain = generator.uniform(0.8, 1.2)
    strong = (transform.image(frames).clamp_min(1e-6).pow(gamma) * gain)
    strong = (strong + 0.05 * torch.randn_like(strong)).clamp(0.0, 1.0)
    return weak, strong, transform


def _records(args: argparse.Namespace, scope: FitOnlyScope, teacher_sha256: str) -> tuple[BatchProvenance, ...]:
    available = {
        f"train:{path.name.removesuffix('.zarr')}"
        for path in args.data_root.glob("*.zarr")
        if path.is_dir()
    }
    if args.group:
        group_to_video = {scope.video_to_group[video]: video for video in scope.fit_video_ids}
        unknown = set(args.group) - set(group_to_video)
        if unknown:
            raise LaunchError(f"requested group is not fold-fit: {sorted(unknown)[0]}")
        selected_videos = {group_to_video[group] for group in args.group}
    else:
        selected_videos = set(scope.fit_video_ids) & available
    if not selected_videos or selected_videos - available:
        raise LaunchError("selected fit-only videos are absent from the image root")
    records = tuple(
        BatchProvenance(
            group_id=scope.video_to_group[video_id],
            video_id=video_id,
            exact_identity_sha256=scope.exact_image_identity[video_id],
            teacher_sha256=teacher_sha256,
        )
        for video_id in sorted(selected_videos)
    )
    for record in records:
        record.validate(scope)
    return records


def _identity(
    args: argparse.Namespace,
    scope: FitOnlyScope,
    records: Sequence[BatchProvenance],
    model: torch.nn.Module,
    teacher_sha256: str,
    frozen_sha256: str,
) -> dict[str, Any]:
    names = permitted_trainable_names(model)
    return {
        "schema": SSL_SCHEMA,
        "fold": scope.fold,
        "split_sha256": scope.split_sha256,
        "provenance_sha256": scope.provenance_sha256,
        "registry_sha256": scope.registry_sha256,
        "released_weights_sha256": WEIGHTS_SHA256,
        "released_config_sha256": CONFIG_SHA256,
        "initial_teacher_sha256": teacher_sha256,
        "initial_frozen_state_sha256": frozen_sha256,
        "trainable_scope": "released-detect-head+compact-aux-head+released-temporal-blocks",
        "trainable_parameter_names": list(names),
        "trainable_parameter_count": sum(dict(model.named_parameters())[name].numel() for name in names),
        "selected_sources": [record.__dict__ for record in records],
        "hyperparameters": {
            "lr": args.lr,
            "ema_decay": args.ema_decay,
            "candidate_floor": args.candidate_floor,
            "max_points_per_frame": args.max_points_per_frame,
            "edge_radius_um": args.edge_radius_um,
            "neighbor_cap": args.neighbor_cap,
            "ramp_steps": args.ramp_steps,
            "division_gate_open": args.division_gate_open,
        },
        "effects": {
            "network": False,
            "kaggle": False,
            "wandb": False,
            "external_gpu": False,
            "scored_submission": False,
            "dense_nxn_association": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    verify_released_files(args.weights_path, args.config_path)
    scope = FitOnlyScope.from_files(
        args.fold,
        args.split_manifest,
        args.provenance_manifest,
        args.registry_manifest,
        expected_hashes=EXPECTED_HASHES,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args, device)
    trainable_names = set(permitted_trainable_names(model))
    initial_trainable = _tensor_partition_sha256(model, trainable_names, trainable=True)
    initial_frozen = _tensor_partition_sha256(model, trainable_names, trainable=False)
    records = _records(args, scope, initial_trainable)
    identity = _identity(args, scope, records, model, initial_trainable, initial_frozen)
    prelaunch = {
        "artifact": "g02-pseudo-ssl-prelaunch.v1",
        "status": "not_launched" if args.dry_run else "armed",
        "identity": identity,
        "execution": {
            "device": str(device),
            "maximum_hours": args.maximum_hours,
            "max_steps": args.max_steps,
            "epochs": args.epochs,
            "batch_size": 1,
        },
        "initial_partition_sha256": {"trainable": initial_trainable, "frozen": initial_frozen},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "prelaunch.json", prelaunch)
    if args.dry_run:
        return prelaunch
    if not args.armed:
        raise LaunchError("non-dry pseudo SSL requires explicit --armed")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    ema = EMATeacherState(model, decay=args.ema_decay)
    curriculum = AdaptiveThresholdCurriculum(ramp_steps=args.ramp_steps)
    ledger = AcceptanceLedger()
    started = time.monotonic()
    deadline = started + args.maximum_hours * 3600.0
    source = FitImageWindows(
        args.data_root,
        scope,
        records,
        epochs=args.epochs,
        completed_windows=0,
        deadline=deadline,
    )
    random_generator = random.Random(args.seed)
    sums = {key: 0.0 for key in LOSS_KEYS}
    steps = 0
    windows_seen = 0
    empty_streak = 0
    status = "completed"
    last_window: str | None = None
    physical_checks: dict[str, dict[str, Any]] = {}
    iterator = iter(source)
    while steps < args.max_steps:
        if time.monotonic() >= deadline:
            status = "time_cap"
            break
        try:
            batch = next(iterator)
        except StopIteration:
            break
        provenance = batch["provenance"]
        provenance.validate(scope)
        frames = batch["frames"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        weak, strong, transform = _augment(frames, generator=random_generator)
        with ema.average_parameters(model), torch.no_grad():
            teacher_output = model(weak, scenario="universal")
        graph = extract_sparse_pseudo_graph(
            teacher_output,
            video_id=provenance.video_id,
            voxel_size_um=batch["scale_zyx_um"],
            candidate_floor=args.candidate_floor,
            max_points_per_frame=args.max_points_per_frame,
            edge_radius_um=args.edge_radius_um,
            neighbor_cap=args.neighbor_cap,
        )
        check = physical_checks.setdefault(
            graph.video_id,
            {
                "scale_zyx_um": list(graph.voxel_size_um),
                "shape_zyx": list(graph.shape_zyx),
                "windows": 0,
                "candidate_nodes": 0,
                "candidate_edges": 0,
                "candidate_divisions": 0,
                "max_edge_residual_um": 0.0,
                "neighbor_cap": graph.neighbor_cap,
                "edge_radius_um": graph.edge_radius_um,
                "dense_nxn_association": False,
            },
        )
        if check["scale_zyx_um"] != list(graph.voxel_size_um) or check["shape_zyx"] != list(graph.shape_zyx):
            raise LaunchError("one video changed physical scale or output shape between windows")
        check["windows"] += 1
        check["candidate_nodes"] += len(graph.nodes)
        check["candidate_edges"] += len(graph.edges)
        check["candidate_divisions"] += len(graph.divisions)
        check["max_edge_residual_um"] = max(
            check["max_edge_residual_um"],
            max((edge.residual_um for edge in graph.edges), default=0.0),
        )
        selected = select_pseudo_graph(
            graph,
            curriculum,
            step=windows_seen,
            division_gate_open=args.division_gate_open,
            ledger=ledger,
        )
        windows_seen += 1
        accepted = int(selected.node_mask.sum()) + int(selected.edge_mask.sum())
        empty_streak = empty_streak + 1 if accepted == 0 else 0
        if empty_streak >= args.empty_patience:
            status = "coverage_stop"
            break
        if accepted == 0:
            continue
        model.train(True)
        _hold_batchnorm_fixed(model)
        optimizer.zero_grad(set_to_none=True)
        student_output = model(strong, scenario="universal")
        losses = sparse_pseudo_losses(student_output, selected, transform)
        if not torch.isfinite(losses["total"]):
            raise LaunchError("nonfinite pseudo SSL loss")
        losses["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not any(gradient is not None for gradient in gradients) or any(
            gradient is not None and not bool(torch.isfinite(gradient).all()) for gradient in gradients
        ):
            raise LaunchError("missing or nonfinite pseudo SSL gradient")
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
        )
        if time.monotonic() >= deadline:
            status = "time_cap"
            break
        optimizer.step()
        ema.update(model)
        steps += 1
        last_window = str(batch["window_id"])
        for key in LOSS_KEYS:
            sums[key] += float(losses[key].detach().cpu())
    elapsed = time.monotonic() - started
    if _tensor_partition_sha256(model, trainable_names, trainable=False) != initial_frozen:
        raise LaunchError("frozen released state drifted during pseudo SSL")
    loss = {key: value / max(1, steps) for key, value in sums.items()}
    checkpoint = {
        "schema": SSL_SCHEMA,
        "identity": identity,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "ema": {
            **ema.state_dict(),
            "shadow": {name: value.detach().cpu() for name, value in ema.shadow.items()},
        },
        "optimizer": optimizer.state_dict(),
        "progress": {"status": status, "completed_steps": steps, "elapsed_seconds": elapsed},
        "accounting": ledger.as_dict(),
        "physical_checks": physical_checks,
    }
    checkpoint_sha256 = _atomic_checkpoint(args.output_dir / "checkpoint.pt", checkpoint)
    metrics = {
        "schema": SSL_SCHEMA,
        "milestone": "bounded-pseudo-label-ssl-prototype; not-G-02-completion",
        "identity": identity,
        "checkpoint": {"path": "checkpoint.pt", "sha256": checkpoint_sha256},
        "progress": {"status": status, "completed_steps": steps, "elapsed_seconds": elapsed, "last_window": last_window},
        "loss": loss,
        "acceptance": ledger.as_dict(),
        "physical_checks": physical_checks,
        "source_stats": source.stats,
        "frozen_state_unchanged": True,
        "improvement_claimed": False,
    }
    _atomic_json(args.output_dir / "metrics.json", metrics)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--lr", type=float, choices=(1e-5, 3e-5, 1e-4), default=3e-5)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--candidate-floor", type=float, default=0.20)
    parser.add_argument("--max-points-per-frame", type=int, default=512)
    parser.add_argument("--edge-radius-um", type=float, default=8.0)
    parser.add_argument("--neighbor-cap", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--ramp-steps", type=int, default=2_000)
    parser.add_argument("--empty-patience", type=int, default=64)
    parser.add_argument("--maximum-hours", type=float, default=1.75)
    parser.add_argument("--max-steps", type=int, default=6_000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--division-gate-open", action="store_true")
    parser.add_argument("--edge-only-grouped-cv", type=float)
    parser.add_argument("--split-manifest", type=Path, default=Path("reports/g02/grouped_3fold_audit20_v1.json"))
    parser.add_argument("--provenance-manifest", type=Path, default=Path("reports/g02/fit_provenance_v1.json"))
    parser.add_argument("--registry-manifest", type=Path, default=Path("reports/g02/dataset_registry_v1.json"))
    parser.add_argument("--weights-path", type=Path, default=Path("/private/tmp/ckpt-v1/edge_predictor_best.pth"))
    parser.add_argument("--config-path", type=Path, default=Path("/private/tmp/ckpt-v1/config.json"))
    args = parser.parse_args(argv)
    if args.dry_run == args.armed:
        raise LaunchError("choose exactly one of --dry-run or --armed")
    if not 0.0 <= args.candidate_floor <= 1.0 or not 4.0 <= args.edge_radius_um <= 10.0:
        raise LaunchError("candidate floor or physical edge radius is outside the locked range")
    if (
        not 0.0 <= args.ema_decay < 1.0
        or not 1 <= args.ramp_steps <= MAXIMUM_STEPS
        or not 1 <= args.empty_patience <= MAXIMUM_STEPS
        or not 1 <= args.max_points_per_frame <= 2_048
        or not 0.0 < args.maximum_hours <= MAXIMUM_HOURS
        or not 1 <= args.max_steps <= MAXIMUM_STEPS
        or not 1 <= args.epochs <= MAXIMUM_EPOCHS
    ):
        raise LaunchError("pseudo SSL runtime or curriculum bound is invalid")
    if args.division_gate_open or args.edge_only_grouped_cv is not None:
        raise LaunchError(
            "this prototype keeps division closed; a later G-02 integration must verify full edge-only grouped CV >= 0.90"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = run(args)
    except (LaunchError, PseudoSSLError) as exc:
        print(f"train_pseudo_ssl: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"] if "status" in result else result.get("progress", {}).get("status"), "output": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
