"""Bounded fit-only alpha-pseudolink adaptation of released temporal blocks."""

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
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external" / "kaggle-cell-tracking-competition" / "src",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from celltrack_alpha.t_adapter import EXPECTED_CONFIG, load_released_core, verify_released_files
from celltrack_alpha.temporal_pretrain import (
    AlphaPseudoLinkArtifact,
    PRETRAIN_INPUT_POLICY,
    PRETRAIN_OBJECTIVE,
    PRETRAIN_PAIRING_POLICY,
    PRETRAIN_SCHEMA,
    ReleasedTemporalStudent,
    TemporalPretrainError,
    load_alpha_pseudo_link_artifact,
    sparse_migration_loss,
    temporal_parameter_names,
)
from train_ta import (
    CONFIG_SHA256,
    FRACTIONS,
    PROVENANCE_SHA256,
    REGISTRY_SHA256,
    SPLIT_SHA256,
    WEIGHTS_SHA256,
    LaunchError,
    SourceRecord,
    _fold_data,
    _load_json_exact,
    _quantile,
    resolve_fit_records,
    select_fit_groups,
)


MAX_HOURS_CAP = 2.0
MAX_EPOCHS = 10
MAX_STEPS = 10_000
LOSS_KEYS = (
    "total",
    "migration",
    "division",
    "release_anchor",
    "positive_similarity",
    "stale_similarity",
    "link_count",
    "division_count",
)
BATCH_SCHEMA = (
    "frames",
    "link_batch_indices",
    "source_zyx",
    "target_zyx",
    "link_weights",
    "division_batch_indices",
    "division_zyx",
    "division_weights",
    "window_id",
    "source",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


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


def _finite_tensor(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())


def _finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return _finite_tensor(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _registry_record(record: SourceRecord, registry: Mapping[str, Any]) -> Mapping[str, Any]:
    records = registry.get("records")
    if not isinstance(records, list):
        raise LaunchError("exact dataset registry records are missing")
    matches = [
        item
        for item in records
        if isinstance(item, Mapping)
        and item.get("group_id") == record.group_id
        and item.get("video_id") == record.video_id
        and item.get("dataset") == record.stem
        and item.get("split") == "train"
    ]
    if len(matches) != 1:
        raise LaunchError(f"exact train identity is missing or duplicate: {record.group_id}")
    exact = matches[0].get("exact_identity_sha256")
    metadata = matches[0].get("compact_metadata")
    if (
        not isinstance(exact, str)
        or len(exact) != 64
        or any(char not in "0123456789abcdef" for char in exact)
        or not isinstance(metadata, Mapping)
        or not isinstance(metadata.get("image_shape"), list)
        or len(metadata["image_shape"]) != 4
    ):
        raise LaunchError("exact train image identity is malformed")
    return matches[0]


def resolve_alpha_artifacts(
    args: argparse.Namespace,
    selected: Sequence[SourceRecord],
    registry: Mapping[str, Any],
) -> tuple[AlphaPseudoLinkArtifact, ...]:
    artifacts: list[AlphaPseudoLinkArtifact] = []
    for record in selected:
        exact = _registry_record(record, registry)
        try:
            artifact = load_alpha_pseudo_link_artifact(
                args.alpha_artifact_root,
                group_id=record.group_id,
                video_id=record.video_id,
                stem=record.stem,
                expected_exact_identity_sha256=str(exact["exact_identity_sha256"]),
                min_migration_um=args.min_migration_um,
                min_link_confidence=args.min_link_confidence,
                max_links_per_window=args.max_links_per_window,
                max_divisions_per_window=args.max_divisions_per_window,
            )
        except TemporalPretrainError as exc:
            raise LaunchError(f"alpha pseudo-link provenance rejected for {record.group_id}: {exc}") from exc
        if artifact.summary["frame_count"] != int(exact["compact_metadata"]["image_shape"][0]):
            raise LaunchError("alpha frame count disagrees with exact registry metadata")
        artifacts.append(artifact)
    return tuple(artifacts)


class ImageWindowSource:
    """Read only alpha-declared adjacent image windows; label APIs stay unused."""

    def __init__(self, data_root: Path, downsample: Sequence[int]) -> None:
        try:
            self.data_root = Path(data_root).resolve(strict=True)
        except OSError as exc:
            raise LaunchError("image data root does not exist") from exc
        if not self.data_root.is_dir():
            raise LaunchError("image data root is not a directory")
        self.downsample = tuple(int(value) for value in downsample)
        if len(self.downsample) != 3 or any(value < 1 for value in self.downsample):
            raise LaunchError("image downsample must have three positive strides")
        self.window_stats = {"considered": 0, "read": 0, "links": 0, "divisions": 0}

    def _metadata(self, artifact: AlphaPseudoLinkArtifact) -> dict[str, Any]:
        from tracking_cellmot.io import open_dataset

        try:
            zarr_path = (self.data_root / f"{artifact.stem}.zarr").resolve(strict=True)
            zarr_path.relative_to(self.data_root)
        except (OSError, ValueError) as exc:
            raise LaunchError(f"missing or escaping competition image store: {artifact.stem}") from exc
        if not zarr_path.is_dir() or not (zarr_path / "zarr.json").is_file() or not (zarr_path / "0" / "zarr.json").is_file():
            raise LaunchError(f"competition image store is malformed: {artifact.stem}")
        try:
            dataset = open_dataset(
                self.data_root / artifact.stem,
                load_image=False,
                require_tracks=False,
                normalize=False,
                device="cpu",
            )
        except Exception as exc:
            raise LaunchError(f"competition image metadata is unreadable: {artifact.group_id}") from exc
        try:
            observed = Path(dataset.zarr_path).resolve(strict=True)
        except (AttributeError, OSError, TypeError) as exc:
            raise LaunchError("image metadata returned an absent Zarr path") from exc
        if observed != zarr_path or dataset.image is not None or dataset.tracks is not None:
            raise LaunchError("image-only source unexpectedly resolved labels or eager image data")
        shape = tuple(int(value) for value in (dataset.image_shape or ()))
        if len(shape) != 4 or shape[0] != artifact.summary["frame_count"] or any(value < 1 for value in shape):
            raise LaunchError("competition image shape disagrees with alpha provenance")
        q_low, q_high = _quantile(dataset.quantiles, 0.001), _quantile(dataset.quantiles, 0.999)
        if not q_high > q_low:
            raise LaunchError("competition image quantile range is invalid")
        return {"zarr_path": zarr_path, "q_low": q_low, "q_high": q_high}

    def _read(
        self,
        metadata: Mapping[str, Any],
        artifact: AlphaPseudoLinkArtifact,
        window: Any,
    ) -> dict[str, Any]:
        import numpy as np
        import zarr

        dz, dy, dx = self.downsample
        start = int(window.start_frame)
        try:
            store = zarr.open_group(str(metadata["zarr_path"]), mode="r")["0"]
            raw = store[start : start + 2, ::dz, ::dy, ::dx].astype(np.float32)
        except Exception as exc:
            raise LaunchError("alpha-declared competition image slab read failed") from exc
        if raw.ndim != 4 or raw.shape[0] != 2 or not np.isfinite(raw).all():
            raise LaunchError("alpha-declared image slab is missing or nonfinite")
        scale = float(metadata["q_high"]) - float(metadata["q_low"])
        frames = torch.from_numpy(((raw - float(metadata["q_low"])) / scale).clip(0.0, 1.0))
        links = tuple(window.links)
        divisions = tuple(window.divisions)
        batch = {
            "frames": frames.unsqueeze(0).unsqueeze(2),
            "link_batch_indices": torch.zeros(len(links), dtype=torch.long),
            "source_zyx": torch.tensor([link.source_zyx for link in links], dtype=torch.long),
            "target_zyx": torch.tensor([link.target_zyx for link in links], dtype=torch.long),
            "link_weights": torch.tensor([link.confidence for link in links], dtype=torch.float32),
            "division_batch_indices": torch.zeros(len(divisions), dtype=torch.long),
            "division_zyx": torch.tensor(
                [[motif.parent_zyx, *motif.child_zyx] for motif in divisions], dtype=torch.long
            ).reshape(-1, 3, 3),
            "division_weights": torch.tensor([motif.confidence for motif in divisions], dtype=torch.float32),
            "window_id": f"{artifact.group_id}:{start}",
            "source": {
                "group_id": artifact.group_id,
                "video_id": artifact.video_id,
                "stem": artifact.stem,
                "exact_identity_sha256": artifact.exact_identity_sha256,
            },
        }
        if tuple(batch) != BATCH_SCHEMA:
            raise LaunchError("alpha sparse batch schema drift")
        self.window_stats["read"] += 1
        self.window_stats["links"] += len(links)
        self.window_stats["divisions"] += len(divisions)
        return batch

    def batches(
        self,
        artifacts: Sequence[AlphaPseudoLinkArtifact],
        *,
        completed_windows: int,
        deadline: float,
        epochs: int,
    ) -> Iterator[dict[str, Any]]:
        if completed_windows < 0 or epochs < 1:
            raise LaunchError("completed sparse windows or epochs is invalid")
        position = 0
        for epoch in range(1, epochs + 1):
            for artifact in artifacts:
                if time.monotonic() >= deadline:
                    return
                metadata = self._metadata(artifact)
                for window in artifact.windows:
                    self.window_stats["considered"] += 1
                    if time.monotonic() >= deadline:
                        return
                    if position >= completed_windows:
                        batch = self._read(metadata, artifact, window)
                        batch["epoch"] = epoch
                        yield batch
                    position += 1


def _identity(
    args: argparse.Namespace,
    selected: Sequence[SourceRecord],
    artifacts: Sequence[AlphaPseudoLinkArtifact],
) -> dict[str, Any]:
    return {
        "fold": args.fold,
        "objective": PRETRAIN_OBJECTIVE,
        "input_policy": dict(PRETRAIN_INPUT_POLICY),
        "pairing_policy": dict(PRETRAIN_PAIRING_POLICY),
        "split_sha256": SPLIT_SHA256,
        "provenance_sha256": PROVENANCE_SHA256,
        "registry_sha256": REGISTRY_SHA256,
        "released_weights_sha256": WEIGHTS_SHA256,
        "released_config_sha256": CONFIG_SHA256,
        "released_config": dict(EXPECTED_CONFIG),
        "initialization": "released-temporal-unet-exact-not-scratch",
        "trainable_scope": "temporal_blocks_only",
        "learning_rate": args.lr,
        "migration_margin": args.migration_margin,
        "release_anchor_weight": args.release_anchor_weight,
        "division_weight": args.division_weight,
        "sibling_cosine_ceiling": args.sibling_cosine_ceiling,
        "min_migration_um": args.min_migration_um,
        "min_link_confidence": args.min_link_confidence,
        "max_links_per_window": args.max_links_per_window,
        "max_divisions_per_window": args.max_divisions_per_window,
        "fraction": args.fraction,
        "seed": args.seed,
        "selected_sources": [record.json() for record in selected],
        "alpha_artifacts": [artifact.provenance() for artifact in artifacts],
    }


def prelaunch_manifest(
    args: argparse.Namespace,
    selected: Sequence[SourceRecord],
    artifacts: Sequence[AlphaPseudoLinkArtifact],
    device: str,
) -> dict[str, Any]:
    return {
        "artifact": "temporal-alpha-pseudolink-prelaunch.v2",
        "status": "not_launched" if args.dry_run else "armed",
        "identity": _identity(args, selected, artifacts),
        "source_isolation": {
            "fit_only": True,
            "selected_group_ids": [record.group_id for record in selected],
            "selected_train_video_ids": [record.video_id for record in selected],
            "alpha_artifact_root": str(Path(args.alpha_artifact_root).resolve()),
            **dict(PRETRAIN_INPUT_POLICY),
        },
        "execution": {
            "location": "local-only",
            "device": device,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "maximum_hours": args.maximum_hours,
            "batch_size": 1,
            "downsample": list(EXPECTED_CONFIG["downsample"]),
            "checkpoint_every_steps": args.checkpoint_every_steps,
            "dry_run": args.dry_run,
        },
        "memory_contract": {
            **dict(PRETRAIN_PAIRING_POLICY),
            "objective_extra_work": "O((K+D)*C)",
            "dense_nxn_allocations": 0,
            "teacher_forwards": 0,
            "new_trainable_heads": 0,
        },
        "evidence_policy": {
            "scope": "mechanism-and-integrity-only",
            "improvement_claimed": False,
            "promotion_requires": "fresh-equal-budget-paired-supervised-comparison",
        },
        "tracking": {
            "backend": "local",
            "wandb_compatible_files": ["checkpoint.pt", "wandb-summary.json", "wandb-artifact.json"],
            "wandb_imported": False,
            "network_calls": False,
            "kaggle_calls": False,
            "external_gpu_calls": False,
            "project": args.project,
            "run_id": args.run_id,
            "run_name": args.run_name,
        },
    }


def _build_model(args: argparse.Namespace, device: torch.device) -> ReleasedTemporalStudent:
    from tracking_cellmot.models.temporal_unet import TemporalUNet3D

    backbone = TemporalUNet3D(
        in_channels=1,
        out_channels=EXPECTED_CONFIG["unet_out_channels"],
        layers=EXPECTED_CONFIG["unet_layers"],
    )
    load_released_core(backbone, args.weights_path, args.config_path)
    return ReleasedTemporalStudent(backbone).to(device)


def train_steps(
    model: ReleasedTemporalStudent,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    deadline: float,
    max_steps: int,
    global_step: int,
    migration_margin: float,
    release_anchor_weight: float,
    division_weight: float,
    sibling_cosine_ceiling: float,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    checkpoint_every_steps: int = 0,
) -> dict[str, Any]:
    """Run bounded sparse steps after explicit arming by the entry point."""

    if max_steps < 0 or global_step < 0 or checkpoint_every_steps < 0:
        raise LaunchError("pretraining step counters are invalid")
    sums = {key: 0.0 for key in LOSS_KEYS}
    completed = 0
    windows: list[str] = []
    per_step: list[dict[str, Any]] = []
    status = "completed"
    model.train(True)
    iterator = iter(batches)
    while completed < max_steps:
        if time.monotonic() >= deadline:
            status = "time_cap"
            break
        try:
            batch = next(iterator)
        except StopIteration:
            break
        frames = batch["frames"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        sparse = {
            key: batch[key].to(device=device, non_blocking=device.type == "cuda")
            for key in (
                "link_batch_indices",
                "source_zyx",
                "target_zyx",
                "link_weights",
                "division_batch_indices",
                "division_zyx",
                "division_weights",
            )
        }
        optimizer.zero_grad(set_to_none=True)
        features = model(frames)
        losses = sparse_migration_loss(
            model,
            features,
            link_batch_indices=sparse["link_batch_indices"],
            source_zyx=sparse["source_zyx"],
            target_zyx=sparse["target_zyx"],
            link_weights=sparse["link_weights"],
            margin=migration_margin,
            release_anchor_weight=release_anchor_weight,
            division_batch_indices=sparse["division_batch_indices"],
            division_zyx=sparse["division_zyx"],
            division_weights=sparse["division_weights"],
            division_weight=division_weight,
            sibling_cosine_ceiling=sibling_cosine_ceiling,
        )
        losses["total"].backward()
        trainable = tuple(parameter for parameter in model.student.parameters() if parameter.requires_grad)
        if (
            not trainable
            or not any(parameter.grad is not None for parameter in trainable)
            or not all(parameter.grad is None or _finite_tensor(parameter.grad) for parameter in trainable)
        ):
            raise LaunchError("temporal pseudo-link gradient is missing or nonfinite")
        if any(parameter.grad is not None for parameter in model.student.parameters() if not parameter.requires_grad):
            raise LaunchError("frozen released parameter unexpectedly received a gradient")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if time.monotonic() >= deadline:
            status = "time_cap"
            break
        optimizer.step()
        if not all(_finite_tensor(parameter.detach()) for parameter in trainable):
            raise LaunchError("temporal pseudo-link parameter became nonfinite")
        completed += 1
        step = global_step + completed
        window_id = str(batch.get("window_id", step))
        windows.append(window_id)
        values = {key: float(losses[key].detach().cpu()) for key in LOSS_KEYS}
        for key, value in values.items():
            sums[key] += value
        per_step.append({"step": step, "epoch": int(batch.get("epoch", 1)), "window_id": window_id, "loss": values})
        if checkpoint_callback is not None and checkpoint_every_steps and completed % checkpoint_every_steps == 0:
            checkpoint_callback(
                {
                    "status": "running",
                    "steps": completed,
                    "windows": list(windows),
                    "per_step": list(per_step),
                    "loss": {key: value / completed for key, value in sums.items()},
                }
            )
    return {
        "status": status,
        "steps": completed,
        "windows": windows,
        "per_step": per_step,
        "loss": {key: value / max(1, completed) for key, value in sums.items()},
    }


def _checkpoint_payload(
    identity: Mapping[str, Any],
    model: ReleasedTemporalStudent,
    optimizer: torch.optim.Optimizer,
    *,
    completed_steps: int,
    elapsed_seconds: float,
    status: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    state = {name: value.detach().cpu() for name, value in model.student.state_dict().items()}
    payload = {
        "schema": PRETRAIN_SCHEMA,
        "identity": dict(identity),
        "backbone": state,
        "optimizer": optimizer.state_dict(),
        "progress": {
            "completed_steps": completed_steps,
            "completed_windows": completed_steps,
            "elapsed_seconds": elapsed_seconds,
            "status": status,
        },
        "metrics": dict(metrics),
    }
    if not _finite_tree(payload):
        raise LaunchError("refusing to checkpoint nonfinite temporal pseudo-link state")
    return payload


def _write_outputs(
    output_dir: Path,
    identity: Mapping[str, Any],
    model: ReleasedTemporalStudent,
    optimizer: torch.optim.Optimizer,
    result: Mapping[str, Any],
    *,
    prior_steps: int,
    prior_elapsed: float,
    started: float,
    tracker: Mapping[str, Any],
    source_stats: Mapping[str, Any],
) -> dict[str, Any]:
    total_steps = prior_steps + int(result["steps"])
    elapsed = prior_elapsed + time.monotonic() - started
    payload = _checkpoint_payload(
        identity,
        model,
        optimizer,
        completed_steps=total_steps,
        elapsed_seconds=elapsed,
        status=str(result["status"]),
        metrics={"loss": dict(result["loss"]), "source_window_stats": dict(source_stats)},
    )
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_sha256 = _atomic_checkpoint(checkpoint_path, payload)
    metrics = {
        "schema": PRETRAIN_SCHEMA,
        "identity": dict(identity),
        "checkpoint": {"path": checkpoint_path.name, "sha256": checkpoint_sha256},
        "progress": dict(payload["progress"]),
        "loss": dict(result["loss"]),
        "windows": list(result["windows"]),
        "per_step": list(result["per_step"]),
        "source_window_stats": dict(source_stats),
        "evidence_scope": "bounded-local-mechanism-smoke; no improvement claim",
        "tracking": dict(tracker),
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    _atomic_json(
        output_dir / "resume.json",
        {
            "schema": PRETRAIN_SCHEMA,
            "identity": dict(identity),
            "checkpoint_sha256": checkpoint_sha256,
            "completed_steps": total_steps,
            "status": result["status"],
        },
    )
    summary = {"global_step": total_steps, "elapsed_seconds": elapsed, "status": result["status"]}
    summary.update({f"loss/{key}": result["loss"][key] for key in LOSS_KEYS})
    _atomic_json(output_dir / "wandb-summary.json", summary)
    _atomic_json(
        output_dir / "wandb-artifact.json",
        {
            "type": "model",
            "name": f"temporal-pseudolink-fold-{identity['fold']}",
            "aliases": ["latest"],
            "files": [{"path": checkpoint_path.name, "sha256": checkpoint_sha256}],
            "metadata": {
                "schema": PRETRAIN_SCHEMA,
                "fold": identity["fold"],
                "split_sha256": SPLIT_SHA256,
                "network_calls": False,
                "labels_used": False,
                "improvement_claimed": False,
            },
        },
    )
    return metrics


def _load_resume(
    path: Path,
    identity: Mapping[str, Any],
    model: ReleasedTemporalStudent,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LaunchError("weights-only temporal pseudo-link resume load failed") from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema", "identity", "backbone", "optimizer", "progress", "metrics"}
        or payload.get("schema") != PRETRAIN_SCHEMA
        or payload.get("identity") != dict(identity)
        or not _finite_tree(payload)
    ):
        raise LaunchError("temporal pseudo-link resume identity or finite-state check failed")
    try:
        state = payload["backbone"]
        released_state = model.student.state_dict()
        if not isinstance(state, Mapping) or tuple(state) != tuple(released_state):
            raise LaunchError("temporal pseudo-link resume backbone schema mismatch")
        temporal_names = set(temporal_parameter_names(model.student))
        if any(
            name not in temporal_names
            and (
                not isinstance(value, torch.Tensor)
                or not torch.equal(value.detach().cpu(), released_state[name].detach().cpu())
            )
            for name, value in state.items()
        ):
            raise LaunchError("temporal pseudo-link resume changed the frozen released spatial core")
        model.student.load_state_dict(payload["backbone"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        progress = payload["progress"]
        steps = int(progress["completed_steps"])
        elapsed = float(progress["elapsed_seconds"])
    except LaunchError:
        raise
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise LaunchError("temporal pseudo-link resume state is incompatible") from exc
    if steps < 0 or progress.get("completed_windows") != steps or not math.isfinite(elapsed) or elapsed < 0:
        raise LaunchError("temporal pseudo-link resume progress is invalid")
    return steps, elapsed


def _run_bounded_pretraining(
    args: argparse.Namespace,
    selected: Sequence[SourceRecord],
    artifacts: Sequence[AlphaPseudoLinkArtifact],
    device_name: str,
    *,
    model_factory: Callable[[argparse.Namespace, torch.device], ReleasedTemporalStudent] | None = None,
    batch_factory: Callable[[int, float], Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    if not args.armed:
        raise LaunchError("alpha pseudo-link pretraining requires explicit --armed before image access")
    if args.data_root is None and batch_factory is None:
        raise LaunchError("--data-root is required for alpha pseudo-link pretraining")
    device = torch.device(device_name)
    identity = _identity(args, selected, artifacts)
    _seed_all(args.seed)
    model = (model_factory or _build_model)(args, device)
    trainable = tuple(parameter for parameter in model.student.parameters() if parameter.requires_grad)
    if not trainable:
        raise LaunchError("temporal pseudo-link pretraining has no trainable released parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    prior_steps = 0
    prior_elapsed = 0.0
    if args.resume is not None:
        prior_steps, prior_elapsed = _load_resume(args.resume, identity, model, optimizer)
    started = time.monotonic()
    deadline = started + max(0.0, args.maximum_hours * 3600.0 - prior_elapsed)
    tracker = {
        "backend": "local",
        "wandb_compatible": True,
        "wandb_imported": False,
        "network_calls": False,
        "kaggle_calls": False,
        "external_gpu_calls": False,
        "project": args.project,
        "run_id": args.run_id,
        "run_name": args.run_name,
    }
    source: ImageWindowSource | None = None
    if prior_steps >= args.max_steps:
        result = {
            "status": "completed",
            "steps": 0,
            "windows": [],
            "per_step": [],
            "loss": {key: 0.0 for key in LOSS_KEYS},
        }
    elif prior_elapsed >= args.maximum_hours * 3600.0:
        result = {
            "status": "time_cap",
            "steps": 0,
            "windows": [],
            "per_step": [],
            "loss": {key: 0.0 for key in LOSS_KEYS},
        }
    else:
        if batch_factory is None:
            source = ImageWindowSource(args.data_root, EXPECTED_CONFIG["downsample"])
            batches = source.batches(
                artifacts, completed_windows=prior_steps, deadline=deadline, epochs=args.epochs
            )
        else:
            batches = batch_factory(prior_steps, deadline)

        def checkpoint_callback(progress: Mapping[str, Any]) -> None:
            _write_outputs(
                args.output_dir,
                identity,
                model,
                optimizer,
                progress,
                prior_steps=prior_steps,
                prior_elapsed=prior_elapsed,
                started=started,
                tracker=tracker,
                source_stats=dict(getattr(source, "window_stats", {})),
            )

        result = train_steps(
            model,
            optimizer,
            batches,
            device=device,
            deadline=deadline,
            max_steps=args.max_steps - prior_steps,
            global_step=prior_steps,
            migration_margin=args.migration_margin,
            release_anchor_weight=args.release_anchor_weight,
            division_weight=args.division_weight,
            sibling_cosine_ceiling=args.sibling_cosine_ceiling,
            checkpoint_callback=checkpoint_callback,
            checkpoint_every_steps=args.checkpoint_every_steps,
        )
    return _write_outputs(
        args.output_dir,
        identity,
        model,
        optimizer,
        result,
        prior_steps=prior_steps,
        prior_elapsed=prior_elapsed,
        started=started,
        tracker=tracker,
        source_stats=dict(getattr(source, "window_stats", {})),
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-artifact-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights-path", type=Path, default=Path("/private/tmp/ckpt-v1/edge_predictor_best.pth"))
    parser.add_argument("--config-path", type=Path, default=Path("/private/tmp/ckpt-v1/config.json"))
    parser.add_argument("--split-manifest", type=Path, default=Path("reports/g02/grouped_3fold_audit20_v1.json"))
    parser.add_argument("--provenance-manifest", type=Path, default=Path("reports/g02/fit_provenance_v1.json"))
    parser.add_argument("--registry-manifest", type=Path, default=Path("reports/g02/dataset_registry_v1.json"))
    parser.add_argument("--fraction", type=float, choices=FRACTIONS, default=1.0)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--migration-margin", type=float, default=0.1)
    parser.add_argument("--release-anchor-weight", type=float, default=1e-3)
    parser.add_argument("--division-weight", type=float, default=0.0)
    parser.add_argument("--sibling-cosine-ceiling", type=float, default=0.9)
    parser.add_argument("--min-migration-um", type=float, default=0.5)
    parser.add_argument("--min-link-confidence", type=float, default=0.95)
    parser.add_argument("--max-links-per-window", type=int, default=256)
    parser.add_argument("--max-divisions-per-window", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50)
    parser.add_argument("--maximum-hours", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--project", default="celltrack-temporal-pseudolink")
    parser.add_argument("--run-id")
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--armed", action="store_true")
    args = parser.parse_args(argv)
    finite_nonnegative = (args.release_anchor_weight, args.division_weight)
    if args.seed < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise LaunchError("pretraining seed or learning rate is invalid")
    if not math.isfinite(args.migration_margin) or not 0 < args.migration_margin <= 2:
        raise LaunchError("migration margin must be in (0,2]")
    if any(not math.isfinite(value) or value < 0 for value in finite_nonnegative):
        raise LaunchError("release-anchor/division weights must be finite and nonnegative")
    if not math.isfinite(args.sibling_cosine_ceiling) or not -1 <= args.sibling_cosine_ceiling <= 1:
        raise LaunchError("sibling cosine ceiling must be in [-1,1]")
    if not math.isfinite(args.min_migration_um) or args.min_migration_um <= 0:
        raise LaunchError("minimum migration must be positive and finite")
    if not math.isfinite(args.min_link_confidence) or not 0 < args.min_link_confidence <= 1:
        raise LaunchError("minimum link confidence must be in (0,1]")
    if not (1 <= args.max_links_per_window <= 1024 and 1 <= args.max_divisions_per_window <= 64):
        raise LaunchError("sparse pseudo-link caps are outside bounded limits")
    if not (1 <= args.epochs <= MAX_EPOCHS and 1 <= args.max_steps <= MAX_STEPS):
        raise LaunchError("pretraining epochs or steps exceed bounded limits")
    if not (1 <= args.checkpoint_every_steps <= args.max_steps):
        raise LaunchError("checkpoint cadence must be within the requested step bound")
    if not math.isfinite(args.maximum_hours) or not (0 < args.maximum_hours <= MAX_HOURS_CAP):
        raise LaunchError("pretraining time cap is invalid")
    if args.dry_run and args.armed:
        raise LaunchError("dry-run and armed modes are mutually exclusive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    verify_released_files(args.weights_path, args.config_path)
    split = _load_json_exact(args.split_manifest, SPLIT_SHA256)
    provenance = _load_json_exact(args.provenance_manifest, PROVENANCE_SHA256)
    registry = _load_json_exact(args.registry_manifest, REGISTRY_SHA256)
    selected_groups = select_fit_groups(args.fold, split, provenance, args.fraction, args.seed, args.group)
    _, forbidden = _fold_data(args.fold, split, provenance)
    selected = resolve_fit_records(selected_groups, registry, forbidden)
    artifacts = resolve_alpha_artifacts(args, selected, registry)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    digest = _atomic_json(
        args.output_dir / "prelaunch.json", prelaunch_manifest(args, selected, artifacts, device)
    )
    if args.dry_run:
        print(json.dumps({"manifest": str(args.output_dir / "prelaunch.json"), "sha256": digest}, sort_keys=True))
        return 0
    _run_bounded_pretraining(args, selected, artifacts, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
