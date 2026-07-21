"""Armed-only, sparse two-frame trainer for the released temporal core."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.t_adapter import (
    CENTER_LOSS, CENTER_SUPERVISION, CONFIG_SHA256, DETECTOR_POLICIES, EXPECTED_CONFIG, SCENARIOS, WEIGHTS_SHA256, TargetError,
    TemporalAdapterModel, build_sparse_targets, load_released_core, multitask_loss,
    verify_released_files,
)

SPLIT_SHA256 = "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769"
PROVENANCE_SHA256 = "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f"
REGISTRY_SHA256 = "21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158"
LRS = (1e-5, 3e-5, 1e-4)
UNFREEZE_DEPTHS = (0, 1, 2)
FRACTIONS = (0.1, 0.3, 1.0)
CENTER_WEIGHTS = (0.0, 0.1)
TEACHER_WEIGHTS = (0.0, 1.0)
FIRST_CANDIDATE = {
    "scenario": "universal", "unfreeze": 0, "lr": 1e-5, "fraction": 0.1,
    "detector_policy": "frozen_aux", "center_weight": 0.0, "teacher_weight": 0.0,
}
MAX_HOURS_CAP, MAX_CHARGE_CAP, MAX_EPOCHS, MAX_STEPS = 0.75, 0.30, 10, 1_000
BATCH_SCHEMA = ("frames", "nodes", "edges", "voxel_size_um", "window_id")
NUMERICAL_POLICY = {"autocast": False, "gradient_scaler": False, "precision": "fp32"}


class LaunchError(ValueError):
    """Raised before a disallowed source, incompatible resume, or unsafe step."""


_STEM = re.compile(r"^[0-9a-f]{4}_[0-9a-f]{8}$")


@dataclass(frozen=True)
class SourceRecord:
    """Immutable fit-only registry resolution, never a filesystem-derived ID."""

    group_id: str
    video_id: str
    stem: str

    def json(self) -> dict[str, str]:
        return {"group_id": self.group_id, "video_id": self.video_id, "stem": self.stem}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
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


def _atomic_checkpoint(path: Path, state: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(state), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return _sha256_file(path)


def _load_json_exact(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file() or _sha256_file(file_path) != expected_sha256:
        raise LaunchError(f"immutable manifest mismatch: {file_path}")
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"invalid manifest: {file_path}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"manifest is not an object: {file_path}")
    return value


def _fold_data(fold: int, split: Mapping[str, Any], provenance: Mapping[str, Any]) -> tuple[tuple[str, ...], set[str]]:
    if fold not in (0, 1, 2):
        raise LaunchError("fold must be one of 0, 1, 2")
    try:
        split_fold, provenance_fold = split["folds"][str(fold)], provenance["folds"][str(fold)]
        fit, held = set(split_fold["fit_group_ids"]), set(split_fold["held_out_group_ids"])
        audit, forbidden = set(split["audit"]["group_ids"]), set(provenance_fold["forbidden_fit_group_ids"])
    except (KeyError, TypeError) as exc:
        raise LaunchError("split/provenance schema is incomplete") from exc
    if not fit or fit & held or fit & audit or fit & forbidden or not held <= forbidden or not audit <= forbidden:
        raise LaunchError("fit-source isolation invariant failed")
    return tuple(sorted(fit)), held | audit | forbidden


def select_fit_groups(fold: int, split: Mapping[str, Any], provenance: Mapping[str, Any], fraction: float, seed: int,
                      requested_groups: Iterable[str] | None = None) -> tuple[str, ...]:
    if fraction not in FRACTIONS:
        raise LaunchError("fraction is outside the fixed search grid")
    fit, forbidden = _fold_data(fold, split, provenance)
    fit_set, requested = set(fit), tuple(requested_groups or ())
    for group in requested:
        if group not in fit_set:
            raise LaunchError(f"{'forbidden' if group in forbidden else 'unknown'} group access rejected: {group}")
    candidates = tuple(sorted(set(requested))) if requested else fit
    ranked = sorted(candidates, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("ascii")).hexdigest())
    selected = tuple(ranked[:max(1, math.ceil(len(ranked) * fraction))])
    if not selected or set(selected) - fit_set or set(selected) & forbidden:
        raise LaunchError("selected sources are not fit-only")
    return selected


def resolve_fit_records(selected: Sequence[str], registry: Mapping[str, Any], forbidden: set[str]) -> tuple[SourceRecord, ...]:
    """Map attested split group IDs to one unique train stem each.

    The registry, rather than a group-ID-shaped pathname, is the only bridge
    between split identities and competition source layout.
    """
    groups = registry.get("groups")
    if not isinstance(groups, list):
        raise LaunchError("registry groups is not a list")
    selected_ids = tuple(selected)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise LaunchError("selected registry groups are missing or duplicate")
    indexed: dict[str, list[Mapping[str, Any]]] = {}
    for item in groups:
        if not isinstance(item, Mapping) or not isinstance(item.get("group_id"), str):
            raise LaunchError("registry group record is malformed")
        indexed.setdefault(item["group_id"], []).append(item)
    resolved: list[SourceRecord] = []
    stems: set[str] = set()
    for group_id in selected_ids:
        if group_id in forbidden:
            raise LaunchError(f"forbidden registry group: {group_id}")
        matches = indexed.get(group_id, [])
        if len(matches) != 1:
            raise LaunchError(f"registry mapping missing or duplicate: {group_id}")
        item = matches[0]
        video_ids, splits = item.get("video_ids"), item.get("splits")
        if (item.get("video_count") != 1 or item.get("contains_nontrain_member") is not False
                or not isinstance(video_ids, list) or len(video_ids) != 1
                or not isinstance(splits, list) or splits != ["train"]):
            raise LaunchError(f"registry mapping is not one train video: {group_id}")
        video_id = video_ids[0]
        if not isinstance(video_id, str) or not video_id.startswith("train:") or video_id.startswith("test:"):
            raise LaunchError(f"registry video is non-train or malformed: {group_id}")
        stem = video_id.removeprefix("train:")
        if not _STEM.fullmatch(stem) or stem in stems:
            raise LaunchError(f"registry stem is malformed or duplicate: {group_id}")
        stems.add(stem)
        resolved.append(SourceRecord(group_id=group_id, video_id=video_id, stem=stem))
    return tuple(resolved)


def _validate_budget(maximum_hours: float, maximum_charge: float) -> None:
    if not (0 < maximum_hours <= MAX_HOURS_CAP) or not (0 <= maximum_charge <= MAX_CHARGE_CAP):
        raise LaunchError("requested time or charge exceeds the prelaunch cap")


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    return {"fold": args.fold, "scenario": args.scenario, "unfreeze": args.unfreeze, "lr": args.lr,
            "fraction": args.fraction, "seed": args.seed, "detector_policy": args.detector_policy,
            "center_weight": args.center_weight, "teacher_weight": args.teacher_weight,
            "center_supervision": CENTER_SUPERVISION, "center_loss": CENTER_LOSS,
            "split_sha256": SPLIT_SHA256, "provenance_sha256": PROVENANCE_SHA256, "registry_sha256": REGISTRY_SHA256,
            "weights_sha256": WEIGHTS_SHA256, "config_sha256": CONFIG_SHA256}


def _training_policy(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    """Record the exact compact path that is allowed to adapt sparse labels."""
    detector = getattr(model, "detect_head", None)
    detector_trainable = detector is not None and any(parameter.requires_grad for parameter in detector.parameters())
    if args.detector_policy == "frozen_aux" and detector_trainable:
        raise LaunchError("frozen auxiliary policy unexpectedly enables released detector training")
    if args.detector_policy == "released_teacher_distill":
        if args.unfreeze != 0 or not detector_trainable:
            raise LaunchError("released teacher detector policy requires exactly the frozen-backbone detector path")
        if args.center_weight <= 0 or args.teacher_weight <= 0:
            raise LaunchError("released teacher detector policy requires positive center and teacher weights")
    accounting = getattr(model, "parameter_accounting", None)
    return {
        "center_supervision": CENTER_SUPERVISION,
        "center_loss": (
            CENTER_LOSS if args.detector_policy == "frozen_aux"
            else "annotated-positive-plus-released-logit-distillation"
        ),
        "division_enabled": False,
        "detector_policy": args.detector_policy,
        "released_detector_frozen": not detector_trainable,
        "center_weight": args.center_weight,
        "teacher_weight": args.teacher_weight,
        "numerical_policy": dict(NUMERICAL_POLICY),
        "parameter_accounting": accounting() if callable(accounting) else None,
        "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
    }


def _device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def prelaunch_manifest(args: argparse.Namespace, selected: Sequence[SourceRecord], device: str) -> dict[str, Any]:
    identity = {**_identity(args), "selected_sources": [record.json() for record in selected]}
    return {"artifact": "prelaunch-v1", "status": "not_launched" if args.dry_run else "armed",
            "identity": identity, "source_isolation": {"fit_only": True,
            "selected_group_ids": [record.group_id for record in selected],
            "selected_train_video_ids": [record.video_id for record in selected],
            "selected_sources": [record.json() for record in selected],
            "audit_used": False, "held_out_used": False, "test_used": False},
            "execution": {"device": device, "mixed_precision": False, "gradient_clip_norm": 1.0,
            "epochs": args.epochs, "max_steps": args.max_steps, "batch_size": 1, "dry_run": args.dry_run},
            "training_policy": {
                "center_supervision": CENTER_SUPERVISION,
                "center_loss": (
                    CENTER_LOSS if args.detector_policy == "frozen_aux"
                    else "annotated-positive-plus-released-logit-distillation"
                ),
                "division_enabled": False,
                "detector_policy": args.detector_policy,
                "released_detector_frozen": args.detector_policy == "frozen_aux",
                "center_weight": args.center_weight,
                "teacher_weight": args.teacher_weight,
                "numerical_policy": dict(NUMERICAL_POLICY),
            },
            "spend_gate": {"maximum_hours": args.maximum_hours, "maximum_charge": args.maximum_charge,
            "cap_hours": MAX_HOURS_CAP, "cap_charge": MAX_CHARGE_CAP, "approved_spend": False},
            "retained_path": {"scenario": "universal", "edge_only": True, "division_enabled": False,
            "specialists": "identity_until_t05"}}


def _row_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LaunchError(f"GEFF {key} must be an integer")
    return value


def _row_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LaunchError(f"GEFF {key} must be finite")
    return float(value)


def _quantile(quantiles: Mapping[str, Any], value: float) -> float:
    for key, raw in quantiles.items():
        try:
            if abs(float(key) - value) < 1e-9 and math.isfinite(float(raw)):
                return float(raw)
        except (TypeError, ValueError):
            continue
    raise LaunchError("Zarr quantile is missing or nonfinite")


class SparseWindowSource:
    """GEFF label metadata plus on-demand two-frame Zarr slabs; no video is cached."""

    def __init__(self, data_root: Path, downsample: Sequence[int]) -> None:
        try:
            self.data_root = Path(data_root).resolve(strict=True)
        except OSError as exc:
            raise LaunchError("data root does not exist") from exc
        if not self.data_root.is_dir():
            raise LaunchError("data root is not a directory")
        self.downsample = tuple(int(v) for v in downsample)
        self.window_stats: dict[str, Any] = {"considered": 0, "usable": 0, "collision_windows": 0,
                                             "skipped_window_ids": []}
        if len(self.downsample) != 3 or any(v < 1 for v in self.downsample):
            raise LaunchError("downsample must be three positive strides")

    def _contained(self, path: Path, *, directory: bool) -> Path:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise LaunchError(f"missing source path: {path.name}") from exc
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise LaunchError(f"source path escapes data root: {path.name}") from exc
        if directory:
            if not resolved.is_dir():
                raise LaunchError(f"source path is not a directory: {path.name}")
        elif not resolved.is_file():
            raise LaunchError(f"source metadata is not a regular file: {path.name}")
        return resolved

    def _source_paths(self, record: SourceRecord) -> tuple[Path, Path]:
        zarr_path = self._contained(self.data_root / f"{record.stem}.zarr", directory=True)
        geff_path = self._contained(self.data_root / f"{record.stem}.geff", directory=True)
        self._contained(zarr_path / "zarr.json", directory=False)
        self._contained(zarr_path / "0" / "zarr.json", directory=False)
        return zarr_path, geff_path

    def _metadata(self, record: SourceRecord) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], tuple[float, float, float], float, float]:
        from tracking_cellmot.io import open_dataset
        zarr_path, _ = self._source_paths(record)
        source = self.data_root / record.stem
        dataset = open_dataset(source, load_image=False, require_tracks=True, normalize=False, device="cpu")
        if dataset.image is not None or dataset.zarr_path is None or dataset.tracks is None:
            raise LaunchError(f"missing image or GEFF tracks for group {record.group_id}")
        try:
            observed = Path(dataset.zarr_path).resolve(strict=True)
        except OSError as exc:
            raise LaunchError("dataset returned an absent Zarr path") from exc
        if observed != zarr_path:
            raise LaunchError("dataset Zarr path is not the resolved train stem")
        try:
            nodes = list(dataset.tracks.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]).iter_rows(named=True))
            edges = list(dataset.tracks.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True))
        except Exception as exc:
            raise LaunchError("GEFF node_attrs/edge_attrs are unreadable") from exc
        if not nodes:
            raise LaunchError("GEFF has no annotated nodes")
        q_low, q_high = _quantile(dataset.quantiles, 0.001), _quantile(dataset.quantiles, 0.999)
        if not q_high > q_low:
            raise LaunchError("Zarr quantile range is invalid")
        voxel_size = tuple(float(scale) * factor for scale, factor in zip(dataset.scale, self.downsample))
        if len(voxel_size) != 3 or any(not math.isfinite(v) or v <= 0 for v in voxel_size):
            raise LaunchError("physical/downsampled scale is invalid")
        return dataset, nodes, edges, voxel_size, q_low, q_high

    def _windows(self, record: SourceRecord) -> Iterator[tuple[int, dict[str, Any]]]:
        dataset, raw_nodes, raw_edges, voxel_size, q_low, q_high = self._metadata(record)
        by_id: dict[int, dict[str, Any]] = {}
        by_time: dict[int, list[dict[str, Any]]] = {}
        for raw in raw_nodes:
            node_id, frame = _row_int(raw, "node_id"), _row_int(raw, "t")
            if node_id in by_id or frame < 0:
                raise LaunchError("duplicate node id or negative frame in GEFF")
            scaled = {"id": node_id, "batch": 0, "time": frame, **{axis: _row_float(raw, axis) / stride for axis, stride in zip(("z", "y", "x"), self.downsample)}}
            by_id[node_id] = scaled
            by_time.setdefault(frame, []).append(scaled)
        children: set[int] = set()
        valid_edges: list[tuple[int, int]] = []
        for raw in raw_edges:
            source_id, target_id = _row_int(raw, "source_id"), _row_int(raw, "target_id")
            if source_id not in by_id or target_id not in by_id or target_id in children:
                raise LaunchError("missing GEFF endpoint or multi-parent record")
            if by_id[target_id]["time"] != by_id[source_id]["time"] + 1:
                raise LaunchError("non-consecutive GEFF edge")
            children.add(target_id)
            valid_edges.append((source_id, target_id))
        n_frames = int(dataset.image_shape[0]) if dataset.image_shape else 0
        for start in range(max(0, n_frames - 1)):
            first, second = by_time.get(start, []), by_time.get(start + 1, [])
            if not first or not second:
                continue
            self.window_stats["considered"] += 1
            nodes = [{**row, "time": row["time"] - start} for row in first + second]
            rounded: set[tuple[int, int, int, int]] = set()
            collision = False
            for row in nodes:
                point = (row["time"], *(int(round(row[axis])) for axis in ("z", "y", "x")))
                if point in rounded:
                    collision = True
                    break
                rounded.add(point)
            if collision:
                self.window_stats["collision_windows"] += 1
                self.window_stats["skipped_window_ids"].append(f"{record.group_id}:{start}")
                continue
            edges = [{"source_id": source_id, "target_id": target_id} for source_id, target_id in valid_edges
                     if by_id[source_id]["time"] == start and by_id[target_id]["time"] == start + 1]
            self.window_stats["usable"] += 1
            yield start, {"group_id": record.group_id, "video_id": record.video_id, "stem": record.stem,
                          "start": start, "nodes": nodes, "edges": edges,
                          "voxel_size_um": voxel_size, "zarr_path": dataset.zarr_path, "q_low": q_low, "q_high": q_high}

    def _read(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np
        import zarr
        start, dz, dy, dx = int(metadata["start"]), *self.downsample
        try:
            store = zarr.open_group(str(metadata["zarr_path"]), mode="r")["0"]
            raw = store[start:start + 2, ::dz, ::dy, ::dx].astype(np.float32)
        except Exception as exc:
            raise LaunchError("two-frame Zarr slab read failed") from exc
        if raw.shape[0] != 2 or not np.isfinite(raw).all():
            raise LaunchError("missing or nonfinite Zarr slab")
        images = torch.from_numpy(((raw - float(metadata["q_low"])) / (float(metadata["q_high"]) - float(metadata["q_low"]))).clip(0.0, 1.0))
        batch = {"frames": images.unsqueeze(0).unsqueeze(2), "nodes": metadata["nodes"], "edges": metadata["edges"],
                 "voxel_size_um": metadata["voxel_size_um"], "window_id": f"{metadata['group_id']}:{metadata['start']}"}
        if tuple(batch) != BATCH_SCHEMA:
            raise LaunchError("sparse batch schema drift")
        return batch

    def batches(self, records: Sequence[SourceRecord], *, completed_windows: int, deadline: float, epochs: int = 1) -> Iterator[dict[str, Any]]:
        if completed_windows < 0 or epochs < 1:
            raise LaunchError("completed windows or epochs is invalid")
        position = 0
        saw_usable = False
        for epoch in range(1, epochs + 1):
            for record in records:
                if time.monotonic() >= deadline:
                    return
                for _, metadata in self._windows(record):
                    saw_usable = True
                    if time.monotonic() >= deadline:
                        return
                    if position >= completed_windows:
                        batch = self._read(metadata)
                        batch["epoch"] = epoch
                        yield batch
                    position += 1
        if not saw_usable:
            raise LaunchError("no usable source windows remain after collision filtering")


def _build_released_model(args: argparse.Namespace, device: torch.device) -> TemporalAdapterModel:
    from tracking_cellmot.models.temporal_unet import TemporalUNet3D
    config = EXPECTED_CONFIG
    backbone = TemporalUNet3D(in_channels=1, out_channels=config["unet_out_channels"], layers=config["unet_layers"])
    detector, _ = load_released_core(backbone, args.weights_path, args.config_path)
    model = TemporalAdapterModel(backbone, detector)
    model.set_training_policy(args.unfreeze, detector_policy=args.detector_policy)
    return model.to(device)


def _finite_tensor(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())


def _finite_state(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return _finite_tensor(value)
    if isinstance(value, Mapping):
        return all(_finite_state(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_state(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _released_teacher(model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> torch.nn.Module | None:
    if args.detector_policy == "frozen_aux":
        return None
    if args.detector_policy != "released_teacher_distill" or args.unfreeze != 0:
        raise LaunchError("unsupported compact detector training policy")
    detector = getattr(model, "detect_head", None)
    if not isinstance(detector, torch.nn.Module):
        raise LaunchError("student does not expose a released detector head")
    teacher = copy.deepcopy(detector).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise LaunchError("released detector teacher unexpectedly remains trainable")
    return teacher


def _batch_loss(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    teacher_detector: torch.nn.Module | None = None,
    center_weight: float = 0.0,
    teacher_weight: float = 0.0,
) -> Mapping[str, torch.Tensor]:
    frames = batch["frames"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    output = model(frames, scenario="universal")
    targets = build_sparse_targets(output, batch["nodes"], batch["edges"], batch["voxel_size_um"])
    teacher_logits = None
    if teacher_detector is not None:
        backbone = getattr(model, "backbone", None)
        if backbone is None or any(parameter.requires_grad for parameter in backbone.parameters()):
            raise LaunchError("released teacher logits require an entirely frozen student backbone")
        with torch.no_grad():
            features = backbone(frames)
            batch_size, steps, channels, z, y, x = features.shape
            teacher_logits = teacher_detector(features.reshape(batch_size * steps, channels, z, y, x))
            teacher_logits = teacher_logits.reshape(batch_size, steps, 1, z, y, x)
    return multitask_loss(
        output,
        targets,
        division_enabled=False,
        center_weight=center_weight,
        teacher_logits=teacher_logits,
        teacher_weight=teacher_weight,
    )


def train_steps(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batches: Iterable[Mapping[str, Any]], *,
                device: torch.device, deadline: float, max_steps: int,
                scaler: Any | None = None, teacher_detector: torch.nn.Module | None = None,
                center_weight: float = 0.0, teacher_weight: float = 0.0) -> tuple[dict[str, Any], Any | None]:
    """Execute real forward/backward/step work, checking cap before each step."""
    if max_steps < 0:
        raise LaunchError("remaining steps cannot be negative")
    if scaler is not None:
        raise LaunchError("mixed-precision scaler is prohibited for sparse adapter training")
    sums = {"total": 0.0, "center": 0.0, "teacher": 0.0, "displacement": 0.0, "log_variance": 0.0, "division": 0.0}
    by_epoch: dict[int, dict[str, Any]] = {}
    completed, window_ids, per_step, status = 0, [], [], "completed"
    model.train()
    for batch in batches:
        if completed >= max_steps:
            break
        if time.monotonic() >= deadline:
            status = "time_cap"
            break
        optimizer.zero_grad(set_to_none=True)
        losses = _batch_loss(
            model,
            batch,
            device,
            teacher_detector=teacher_detector,
            center_weight=center_weight,
            teacher_weight=teacher_weight,
        )
        loss = losses["total"]
        if not _finite_tensor(loss):
            raise LaunchError("nonfinite loss")
        loss.backward()
        for parameter in (p for p in model.parameters() if p.requires_grad and p.grad is not None):
            if not _finite_tensor(parameter.grad):
                raise LaunchError("nonfinite gradient")
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        if time.monotonic() >= deadline:
            status = "time_cap"
            break
        optimizer.step()
        if not all(_finite_tensor(parameter.detach()) for parameter in model.parameters()):
            raise LaunchError("nonfinite parameter after optimizer step")
        completed += 1
        window_ids.append(str(batch.get("window_id", completed)))
        epoch = int(batch.get("epoch", 1))
        if epoch < 1:
            raise LaunchError("batch epoch must be positive")
        aggregate = by_epoch.setdefault(epoch, {"steps": 0, "sums": {key: 0.0 for key in sums}})
        aggregate["steps"] += 1
        for key in sums:
            sums[key] += float(losses[key].detach().cpu())
            aggregate["sums"][key] += float(losses[key].detach().cpu())
        per_step.append({"step": completed, "epoch": epoch, "window_id": window_ids[-1],
                         "loss_components": {key: float(losses[key].detach().cpu()) for key in sums}})
    aggregate = {key: value / max(1, completed) for key, value in sums.items()}
    epochs = [{"epoch": epoch, "steps": item["steps"],
               "loss_components": {key: value / item["steps"] for key, value in item["sums"].items()}}
              for epoch, item in sorted(by_epoch.items())]
    return {"status": status, "steps": completed, "windows": window_ids, "per_step": per_step,
            "epoch_metrics": epochs,
            "loss_components": aggregate}, None


def _load_resume(path: Path, identity: Mapping[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scaler: Any | None) -> tuple[int, float, Any | None]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LaunchError("atomic checkpoint weights_only load failed") from exc
    if not isinstance(state, dict) or state.get("identity") != dict(identity) or not _finite_state(state):
        raise LaunchError("resume identity or finite-state check failed")
    try:
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        steps, elapsed = int(state["completed_steps"]), float(state["elapsed_seconds"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise LaunchError("resume state is incompatible") from exc
    if steps < 0 or not math.isfinite(elapsed) or elapsed < 0:
        raise LaunchError("resume progress is invalid")
    return steps, elapsed, scaler


def _write_outputs(output_dir: Path, identity: Mapping[str, Any], model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer, scaler: Any | None, result: Mapping[str, Any],
                   started: float, source_records: Sequence[SourceRecord], prior_steps: int, prior_elapsed: float,
                   epochs: int, max_steps: int) -> dict[str, Any]:
    elapsed = prior_elapsed + time.monotonic() - started
    total_steps = prior_steps + int(result["steps"])
    state = {"identity": dict(identity), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict() if scaler is not None else None, "completed_steps": total_steps,
             "elapsed_seconds": elapsed, "loss_components": dict(result["loss_components"]),
             "source_groups": [record.group_id for record in source_records],
             "source_video_ids": [record.video_id for record in source_records],
             "source_records": [record.json() for record in source_records], "completed_windows": total_steps,
             "execution": {"epochs": epochs, "max_steps": max_steps}, "status": result["status"]}
    checkpoint_sha256 = _atomic_checkpoint(output_dir / "checkpoint.pt", state)
    metrics = {"identity": dict(identity), "checkpoint_sha256": checkpoint_sha256, "completed_steps": total_steps,
               "elapsed_seconds": elapsed, "loss_components": dict(result["loss_components"]),
               "source_groups": [record.group_id for record in source_records],
               "source_video_ids": [record.video_id for record in source_records],
               "source_records": [record.json() for record in source_records], "windows": list(result["windows"]),
               "per_step": list(result.get("per_step", [])), "epoch_metrics": list(result.get("epoch_metrics", [])),
               "source_window_stats": dict(result.get("source_window_stats", {})),
               "execution": {"epochs": epochs, "max_steps": max_steps},
               "status": result["status"]}
    _atomic_json(output_dir / "metrics.json", metrics)
    _atomic_json(output_dir / "resume.json", {"identity": dict(identity), "checkpoint_sha256": checkpoint_sha256,
                 "completed_steps": total_steps, "source_groups": [record.group_id for record in source_records],
                 "source_video_ids": [record.video_id for record in source_records],
                 "execution": {"epochs": epochs, "max_steps": max_steps}, "status": result["status"]})
    return metrics


def _run_bounded_training(args: argparse.Namespace, selected: Sequence[SourceRecord], device_name: str,
                          model_factory: Callable[[argparse.Namespace, torch.device], torch.nn.Module] | None = None,
                          batch_factory: Callable[[int, float], Iterable[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    if not args.armed:
        raise LaunchError("non-dry launch requires explicit --armed before data access")
    if args.data_root is None and batch_factory is None:
        raise LaunchError("--data-root is required when dry-run is disabled")
    if args.scenario != "universal":
        raise LaunchError("specialist training is deferred until T-05")
    device = torch.device(device_name)
    identity = {**_identity(args), "selected_sources": [record.json() for record in selected]}
    _seed_all(args.seed)
    model = (model_factory or _build_released_model)(args, device)
    identity["training_policy"] = _training_policy(model, args)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise LaunchError("no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler: Any | None = None
    prior_steps, prior_elapsed = 0, 0.0
    if args.resume is not None:
        prior_steps, prior_elapsed, scaler = _load_resume(args.resume, identity, model, optimizer, scaler)
    started = time.monotonic()
    deadline = started + max(0.0, args.maximum_hours * 3600.0 - prior_elapsed)
    if prior_steps >= args.max_steps:
        empty = {key: 0.0 for key in ("total", "center", "teacher", "displacement", "log_variance", "division")}
        result = {"status": "completed", "steps": 0, "windows": [], "per_step": [],
                  "epoch_metrics": [], "loss_components": empty, "source_window_stats": {}}
    elif prior_elapsed >= args.maximum_hours * 3600.0:
        empty = {key: 0.0 for key in ("total", "center", "teacher", "displacement", "log_variance", "division")}
        result = {"status": "time_cap", "steps": 0, "windows": [], "per_step": [],
                  "epoch_metrics": [{"epoch": 1, "steps": 0, "loss_components": empty}], "loss_components": empty}
    else:
        source: SparseWindowSource | None = None
        if batch_factory is not None:
            batches = batch_factory(prior_steps, deadline)
        else:
            source = SparseWindowSource(args.data_root, EXPECTED_CONFIG["downsample"])
            batches = source.batches(selected, completed_windows=prior_steps, deadline=deadline, epochs=args.epochs)
        teacher = _released_teacher(model, args, device)
        result, scaler = train_steps(
            model,
            optimizer,
            batches,
            device=device,
            deadline=deadline,
            max_steps=args.max_steps - prior_steps,
            scaler=scaler,
            teacher_detector=teacher,
            center_weight=args.center_weight,
            teacher_weight=args.teacher_weight,
        )
        result["source_window_stats"] = dict(getattr(source, "window_stats", {})) if source is not None else {}
    return _write_outputs(args.output_dir, identity, model, optimizer, scaler, result, started, selected, prior_steps,
                          prior_elapsed, args.epochs, args.max_steps)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--lr", type=float, choices=LRS, default=FIRST_CANDIDATE["lr"])
    parser.add_argument("--unfreeze", type=int, choices=UNFREEZE_DEPTHS, default=FIRST_CANDIDATE["unfreeze"])
    parser.add_argument("--fraction", type=float, choices=FRACTIONS, default=FIRST_CANDIDATE["fraction"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scenario", choices=SCENARIOS, default="universal")
    parser.add_argument("--detector-policy", choices=DETECTOR_POLICIES, default=FIRST_CANDIDATE["detector_policy"])
    parser.add_argument("--center-weight", type=float, choices=CENTER_WEIGHTS, default=FIRST_CANDIDATE["center_weight"])
    parser.add_argument("--teacher-weight", type=float, choices=TEACHER_WEIGHTS, default=FIRST_CANDIDATE["teacher_weight"])
    parser.add_argument("--maximum-hours", type=float, default=0.125)
    parser.add_argument("--maximum-charge", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--split-manifest", type=Path, default=Path("reports/g02/grouped_3fold_audit20_v1.json"))
    parser.add_argument("--provenance-manifest", type=Path, default=Path("reports/g02/fit_provenance_v1.json"))
    parser.add_argument("--registry-manifest", type=Path, default=Path("reports/g02/dataset_registry_v1.json"))
    parser.add_argument("--weights-path", type=Path, default=Path("/private/tmp/ckpt-v1/edge_predictor_best.pth"))
    parser.add_argument("--config-path", type=Path, default=Path("/private/tmp/ckpt-v1/config.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    if not (1 <= args.epochs <= MAX_EPOCHS and 1 <= args.max_steps <= MAX_STEPS):
        raise LaunchError("epochs or steps exceed bounded launch limits")
    if args.detector_policy == "frozen_aux":
        if args.center_weight != 0.0 or args.teacher_weight != 0.0:
            raise LaunchError("frozen auxiliary policy cannot accept center or teacher loss weights")
    elif args.unfreeze != 0 or args.center_weight <= 0.0 or args.teacher_weight <= 0.0:
        raise LaunchError("released teacher detector policy requires depth 0 and positive center/teacher weights")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_budget(args.maximum_hours, args.maximum_charge)
    verify_released_files(args.weights_path, args.config_path)
    split, provenance = _load_json_exact(args.split_manifest, SPLIT_SHA256), _load_json_exact(args.provenance_manifest, PROVENANCE_SHA256)
    registry = _load_json_exact(args.registry_manifest, REGISTRY_SHA256)
    selected_groups = select_fit_groups(args.fold, split, provenance, args.fraction, args.seed, args.group)
    _, forbidden = _fold_data(args.fold, split, provenance)
    selected, device = resolve_fit_records(selected_groups, registry, forbidden), _device_name()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    digest = _atomic_json(args.output_dir / "prelaunch.json", prelaunch_manifest(args, selected, device))
    if args.dry_run:
        print(json.dumps({"manifest": str(args.output_dir / "prelaunch.json"), "sha256": digest}, sort_keys=True))
        return 0
    _run_bounded_training(args, selected, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
