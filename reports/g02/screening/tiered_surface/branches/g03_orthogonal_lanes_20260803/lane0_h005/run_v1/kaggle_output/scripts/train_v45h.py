#!/usr/bin/env python3
"""Armed-only training for the 165-parameter frozen V45 auxiliary head.

This is deliberately a narrow trainer.  It validates the released V45
artifact and the G-02 split/provenance manifests before opening a single image
or GEFF file, then updates only the pointwise auxiliary head.  The released
centre detector and dense transformer never participate in optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.bh_model import DEFAULT_WEIGHT_SHA256, V45_CONFIG, ArtifactBundle, load_v45_model
from celltrack_alpha.t_adapter import TARGET_CONTRACT_SHA256, build_sparse_targets, multitask_loss
from celltrack_alpha.v45_head import FrozenV45Head, save_v45_auxiliary_checkpoint, v45_state_sha256
from train_ta import (
    FRACTIONS,
    MAX_CHARGE_CAP,
    MAX_EPOCHS,
    MAX_HOURS_CAP,
    MAX_STEPS,
    PROVENANCE_SHA256,
    REGISTRY_SHA256,
    SPLIT_SHA256,
    LaunchError,
    SparseWindowSource,
    _fold_data,
    _load_json_exact,
    resolve_fit_records,
    select_fit_groups,
)


SCHEMA = "celltrack-alpha.v45-head-train.v1"
LRS = (1e-5, 3e-5, 1e-4)
V45_DOWNSAMPLE = V45_CONFIG.spatial_stride_zyx


class V45HeadTrainError(ValueError):
    """Raised when an auxiliary-head launch violates a fixed safety contract."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    return _atomic_bytes(path, _canonical(value))


def _device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--fraction", type=float, choices=FRACTIONS, default=0.1)
    parser.add_argument("--lr", type=float, choices=LRS, default=1e-4)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--maximum-hours", type=float, default=0.125)
    parser.add_argument("--maximum-charge", type=float, default=0.05)
    parser.add_argument("--support-root", type=Path, default=ROOT / "external" / "Tracking_Support")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "reports" / "g02" / "grouped_3fold_audit20_v1.json")
    parser.add_argument("--provenance-manifest", type=Path, default=ROOT / "reports" / "g02" / "fit_provenance_v1.json")
    parser.add_argument("--registry-manifest", type=Path, default=ROOT / "reports" / "g02" / "dataset_registry_v1.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--armed", action="store_true")
    args = parser.parse_args(argv)
    if not (1 <= args.epochs <= MAX_EPOCHS and 1 <= args.max_steps <= MAX_STEPS):
        raise V45HeadTrainError("epochs or max-steps exceed the bounded training policy")
    if not (0 < args.maximum_hours <= MAX_HOURS_CAP and 0 <= args.maximum_charge <= MAX_CHARGE_CAP):
        raise V45HeadTrainError("requested time or charge exceeds the locked development cap")
    if args.armed and args.dry_run:
        raise V45HeadTrainError("--armed and --dry-run are mutually exclusive")
    if not args.dry_run and not args.armed:
        raise V45HeadTrainError("non-dry training requires explicit --armed")
    if args.armed and args.data_root is None:
        raise V45HeadTrainError("--data-root is required for armed training")
    return args


def _resolve_scope(args: argparse.Namespace) -> tuple[tuple[Any, ...], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    split = _load_json_exact(args.split_manifest, SPLIT_SHA256)
    provenance = _load_json_exact(args.provenance_manifest, PROVENANCE_SHA256)
    registry = _load_json_exact(args.registry_manifest, REGISTRY_SHA256)
    selected_groups = select_fit_groups(args.fold, split, provenance, args.fraction, args.seed, args.group)
    _, forbidden = _fold_data(args.fold, split, provenance)
    records = resolve_fit_records(selected_groups, registry, forbidden)
    if not records:
        raise V45HeadTrainError("fit selection resolved no records")
    return records, split, provenance, registry


def _identity(args: argparse.Namespace, records: Sequence[Any], bundle: ArtifactBundle) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "fold": args.fold,
        "fraction": args.fraction,
        "lr": args.lr,
        "seed": args.seed,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "split_sha256": SPLIT_SHA256,
        "provenance_sha256": PROVENANCE_SHA256,
        "registry_sha256": REGISTRY_SHA256,
        "v45_manifest_sha256": bundle.hashes.manifest_sha256,
        "v45_config_sha256": bundle.hashes.config_sha256,
        "v45_source_sha256": bundle.hashes.source_sha256,
        "v45_weight_sha256": bundle.hashes.weight_sha256,
        "v45_weight_matches_default": bundle.hashes.weight_sha256 == DEFAULT_WEIGHT_SHA256,
        "architecture": "frozen-v45-32-to-5-pointwise-auxiliary-head",
        "trainable_parameter_count": FrozenV45Head.auxiliary_parameter_count(),
        "output_schema": ["center_logit", "backward_dz_um", "backward_dy_um", "backward_dx_um", "log_variance", "division_logit"],
        "target_contract_sha256": TARGET_CONTRACT_SHA256,
        "backward_displacement": "child-at-t: parent_um-minus-child_um",
        "log_variance": "child-at-t: log-isotropic-3d-variance-um2-over-1-um2",
        "division_target_owner": "parent-at-t-minus-1",
        "division_emission": False,
        "downsample_zyx": list(V45_DOWNSAMPLE),
        "selected_sources": [record.json() for record in records],
    }


def _prelaunch(args: argparse.Namespace, records: Sequence[Any], identity: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        "artifact": "v45-head-prelaunch-v1",
        "status": "not_launched" if args.dry_run else "armed",
        "identity": dict(identity),
        "source_isolation": {
            "fit_only": True,
            "selected_group_ids": [record.group_id for record in records],
            "selected_train_video_ids": [record.video_id for record in records],
            "audit_used": False,
            "held_out_used": False,
            "test_used": False,
        },
        "execution": {
            "device": device,
            "precision": "fp32",
            "mixed_precision": False,
            "gradient_clip_norm": 1.0,
            "dense_transformer_forward": False,
            "dense_nxn_association": False,
        },
        "spend_gate": {
            "maximum_hours": args.maximum_hours,
            "maximum_charge": args.maximum_charge,
            "cap_hours": MAX_HOURS_CAP,
            "cap_charge": MAX_CHARGE_CAP,
        },
    }


def _finite(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())


def _train(args: argparse.Namespace, records: Sequence[Any], model: FrozenV45Head, identity: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    if model.parameter_accounting()["trainable"] != FrozenV45Head.auxiliary_parameter_count():
        raise V45HeadTrainError("optimizer would not be restricted to the 165-parameter auxiliary head")
    if args.data_root is None:
        raise V45HeadTrainError("armed training requires data root")
    core_sha256 = v45_state_sha256(model.core)
    source = SparseWindowSource(args.data_root, V45_DOWNSAMPLE)
    optimizer = torch.optim.AdamW(model.aux_head.parameters(), lr=args.lr)
    model.train()
    deadline = time.monotonic() + args.maximum_hours * 3600.0
    per_step: list[dict[str, Any]] = []
    sums = {name: 0.0 for name in ("total", "center", "teacher", "displacement", "log_variance", "division")}
    status = "completed"
    for raw_batch in source.batches(records, completed_windows=0, deadline=deadline, epochs=args.epochs):
        if len(per_step) >= args.max_steps:
            break
        if time.monotonic() >= deadline:
            status = "time_cap"
            break
        frames = raw_batch["frames"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        output = model(frames)
        targets = build_sparse_targets(output, raw_batch["nodes"], raw_batch["edges"], raw_batch["voxel_size_um"])
        losses = multitask_loss(output, targets, division_enabled=False)
        loss = losses["total"]
        if not _finite(loss):
            raise V45HeadTrainError("nonfinite auxiliary loss")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.core.parameters()):
            raise V45HeadTrainError("released V45 core unexpectedly received gradients")
        if any(parameter.grad is None or not _finite(parameter.grad) for parameter in model.aux_head.parameters()):
            raise V45HeadTrainError("auxiliary head received an invalid gradient")
        torch.nn.utils.clip_grad_norm_(model.aux_head.parameters(), 1.0)
        optimizer.step()
        if v45_state_sha256(model.core) != core_sha256:
            raise V45HeadTrainError("released V45 core changed during auxiliary training")
        # Division loss is inactive in this stage, so the checkpoint must
        # retain an exact zero division row rather than an arbitrary 0.5 logit.
        model.auxiliary_state_dict()
        step = {name: float(value.detach().cpu()) for name, value in losses.items()}
        for name, value in step.items():
            sums[name] += value
        per_step.append({"step": len(per_step) + 1, "window_id": str(raw_batch["window_id"]), "loss_components": step})
    if not per_step and status == "completed":
        raise V45HeadTrainError("no usable fit-only windows were available")
    return {
        "status": status,
        "completed_steps": len(per_step),
        "loss_components": {name: value / max(1, len(per_step)) for name, value in sums.items()},
        "per_step": per_step,
        "source_window_stats": dict(source.window_stats),
        "core_state_sha256": core_sha256,
        "identity": dict(identity),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    records, _split, _provenance, _registry = _resolve_scope(args)
    device_name = _device_name()
    core, bundle = load_v45_model(args.support_root, device=device_name)
    model = FrozenV45Head(core).to(device_name)
    identity = _identity(args, records, bundle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prelaunch = _prelaunch(args, records, identity, device_name)
    _atomic_json(args.output_dir / "prelaunch.json", prelaunch)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "prelaunch": str(args.output_dir / "prelaunch.json"), "identity": identity}, sort_keys=True))
        return 0
    metrics = _train(args, records, model, identity, torch.device(device_name))
    checkpoint_sha256, checkpoint_identity = save_v45_auxiliary_checkpoint(
        args.output_dir / "checkpoint.pt",
        model,
        bundle,
        target_contract_sha256=TARGET_CONTRACT_SHA256,
        completed_steps=metrics["completed_steps"],
        training_identity=identity,
    )
    report = {
        **metrics,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint": str(args.output_dir / "checkpoint.pt"),
        "checkpoint_identity": checkpoint_identity.as_dict(),
    }
    _atomic_json(args.output_dir / "metrics.json", report)
    print(json.dumps({"status": report["status"], "steps": report["completed_steps"], "checkpoint_sha256": checkpoint_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
