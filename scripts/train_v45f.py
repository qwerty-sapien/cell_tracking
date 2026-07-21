#!/usr/bin/env python3
"""Armed-only fit-only trainer for the compact one-pass V45 D4 student."""

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
from celltrack_alpha.fast_data import FastDataError, FastSource
from celltrack_alpha.t_adapter import TARGET_CONTRACT_SHA256, build_sparse_targets, multitask_loss
from celltrack_alpha.v45_fast import FrozenV45Fast, save_fast
from celltrack_alpha.v45_head import v45_state_sha256
from train_ta import (
    FRACTIONS,
    LRS,
    MAX_CHARGE_CAP,
    MAX_EPOCHS,
    MAX_HOURS_CAP,
    MAX_STEPS,
    PROVENANCE_SHA256,
    REGISTRY_SHA256,
    SPLIT_SHA256,
    SourceRecord,
    _fold_data,
    _load_json_exact,
    resolve_fit_records,
    select_fit_groups,
)


SCHEMA = "celltrack-alpha.v45-fast-train.v1"
PROJECTION_SCHEMA = "celltrack-alpha.v45-fast-projection.v1"
CENTER_WEIGHT = 0.1
TEACHER_WEIGHT = 1.0


class V45FastTrain(ValueError):
    """Raised when a fast-student launch violates the bounded G-02 contract."""


def _canon(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _atomic(path: Path, value: Mapping[str, Any]) -> str:
    payload = _canon(value)
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


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(*, armed: bool) -> str:
    if not torch.cuda.is_available():
        if armed:
            raise V45FastTrain("armed V45 fast training requires a CUDA accelerator")
        return "cpu"
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    if armed and capability < (7, 0):
        raise V45FastTrain(
            "armed V45 fast training requires CUDA capability >= 7.0; "
            f"received {capability[0]}.{capability[1]}"
        )
    return "cuda"


def _finite(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--fraction", type=float, choices=FRACTIONS, default=0.1)
    parser.add_argument("--lr", type=float, choices=LRS, default=1e-4)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--maximum-hours", type=float, default=0.75)
    parser.add_argument("--maximum-charge", type=float, default=0.30)
    parser.add_argument("--support-root", type=Path, default=ROOT / "external" / "Tracking_Support")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "reports" / "g02" / "grouped_3fold_audit20_v1.json")
    parser.add_argument("--provenance-manifest", type=Path, default=ROOT / "reports" / "g02" / "fit_provenance_v1.json")
    parser.add_argument("--registry-manifest", type=Path, default=ROOT / "reports" / "g02" / "dataset_registry_v1.json")
    parser.add_argument("--record-projection", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--armed", action="store_true")
    args = parser.parse_args(argv)
    if not (1 <= args.epochs <= MAX_EPOCHS and 1 <= args.max_steps <= MAX_STEPS):
        raise V45FastTrain("epochs or max-steps exceed the bounded training policy")
    if not (0 < args.maximum_hours <= MAX_HOURS_CAP and 0 <= args.maximum_charge <= MAX_CHARGE_CAP):
        raise V45FastTrain("requested time or charge exceeds the locked development cap")
    if args.armed == args.dry_run:
        raise V45FastTrain("select exactly one of --armed or --dry-run")
    if args.armed and args.data_root is None:
        raise V45FastTrain("armed training requires --data-root")
    if args.record_projection is not None and args.group:
        raise V45FastTrain("record projection cannot be combined with ad hoc group selection")
    return args


def _projected_records(
    path: Path,
    args: argparse.Namespace,
    split: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[SourceRecord, ...]:
    try:
        raw = path.read_bytes()
        projection = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V45FastTrain("frozen record projection is unreadable") from exc
    if not isinstance(projection, dict) or _canon(projection) != raw:
        raise V45FastTrain("frozen record projection is not canonical")
    expected_fields = {
        "schema",
        "source_registry_sha256",
        "split_sha256",
        "provenance_sha256",
        "fold",
        "fraction",
        "seed",
        "requested_groups",
        "selected_sources",
    }
    if set(projection) != expected_fields or projection.get("schema") != PROJECTION_SCHEMA:
        raise V45FastTrain("frozen record projection schema mismatch")
    if (
        projection.get("source_registry_sha256") != REGISTRY_SHA256
        or projection.get("split_sha256") != SPLIT_SHA256
        or projection.get("provenance_sha256") != PROVENANCE_SHA256
        or projection.get("fold") != args.fold
        or projection.get("fraction") != args.fraction
        or projection.get("seed") != args.seed
        or projection.get("requested_groups") != []
    ):
        raise V45FastTrain("frozen record projection identity mismatch")
    selected = select_fit_groups(args.fold, split, provenance, args.fraction, args.seed, ())
    rows = projection.get("selected_sources")
    if not isinstance(rows, list) or not rows:
        raise V45FastTrain("frozen record projection selected sources are invalid")
    records: list[SourceRecord] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"group_id", "video_id", "stem"}:
            raise V45FastTrain("frozen record projection source row is invalid")
        values = tuple(row.get(key) for key in ("group_id", "video_id", "stem"))
        if not all(isinstance(value, str) and value for value in values):
            raise V45FastTrain("frozen record projection source value is invalid")
        record = SourceRecord(*values)
        if record.json() != dict(row) or record.video_id != f"train:{record.stem}":
            raise V45FastTrain("frozen record projection source normalization changed")
        records.append(record)
    if tuple(record.group_id for record in records) != selected or len({record.stem for record in records}) != len(records):
        raise V45FastTrain("frozen record projection does not match the locked fit selection")
    _fit, forbidden = _fold_data(args.fold, split, provenance)
    if any(record.group_id in forbidden or not record.video_id.startswith("train:") for record in records):
        raise V45FastTrain("frozen record projection violates fit-only isolation")
    return tuple(records)


def _scope(args: argparse.Namespace) -> tuple[tuple[Any, ...], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    split = _load_json_exact(args.split_manifest, SPLIT_SHA256)
    provenance = _load_json_exact(args.provenance_manifest, PROVENANCE_SHA256)
    if args.record_projection is not None:
        records = _projected_records(args.record_projection, args, split, provenance)
        return records, split, provenance, {"projection_sha256": _sha_file(args.record_projection)}
    registry = _load_json_exact(args.registry_manifest, REGISTRY_SHA256)
    selected = select_fit_groups(args.fold, split, provenance, args.fraction, args.seed, args.group)
    _fit, forbidden = _fold_data(args.fold, split, provenance)
    records = resolve_fit_records(selected, registry, forbidden)
    if not records:
        raise V45FastTrain("fit selection resolved no records")
    return records, split, provenance, registry


def _ident(args: argparse.Namespace, records: Sequence[Any], bundle: ArtifactBundle) -> dict[str, Any]:
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
        "record_projection_sha256": _sha_file(args.record_projection) if args.record_projection is not None else None,
        "v45_manifest_sha256": bundle.hashes.manifest_sha256,
        "v45_config_sha256": bundle.hashes.config_sha256,
        "v45_source_sha256": bundle.hashes.source_sha256,
        "v45_weight_sha256": bundle.hashes.weight_sha256,
        "v45_weight_matches_default": bundle.hashes.weight_sha256 == DEFAULT_WEIGHT_SHA256,
        "head": "one-pass-v45-center-residual-plus-motion",
        "head_parameter_count": FrozenV45Fast.head_params(),
        "teacher": "exact-sequential-d4-8-logit-mean",
        "student_views": [0],
        "center_weight": CENTER_WEIGHT,
        "teacher_weight": TEACHER_WEIGHT,
        "target_contract_sha256": TARGET_CONTRACT_SHA256,
        "backward_displacement": "child-at-t: parent_um-minus-child_um",
        "division_emission": False,
        "downsample_zyx": list(V45_CONFIG.spatial_stride_zyx),
        "selected_sources": [record.json() for record in records],
    }


def _pre(args: argparse.Namespace, records: Sequence[Any], ident: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        "artifact": "v45-fast-prelaunch-v1",
        "status": "armed" if args.armed else "not_launched",
        "identity": dict(ident),
        "source_isolation": {
            "fit_only": True,
            "selected_group_ids": [record.group_id for record in records],
            "selected_train_video_ids": [record.video_id for record in records],
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "execution": {
            "device": device,
            "precision": "fp32",
            "mixed_precision": False,
            "student_encoder_views": 1,
            "teacher_encoder_views": 8,
            "teacher_view_batch_size": 1,
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


def _train(
    args: argparse.Namespace,
    records: Sequence[Any],
    model: FrozenV45Fast,
    ident: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if model.accounting()["trainable"] != FrozenV45Fast.head_params():
        raise V45FastTrain("optimizer would not be restricted to the compact fast head")
    if args.data_root is None:
        raise V45FastTrain("armed training requires data root")
    core_sha256 = v45_state_sha256(model.core)
    source = FastSource(args.data_root, V45_CONFIG.spatial_stride_zyx)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    deadline = time.monotonic() + args.maximum_hours * 3600.0
    sums = {name: 0.0 for name in ("total", "center", "teacher", "displacement", "log_variance", "division")}
    per_step: list[dict[str, Any]] = []
    status = "completed"
    model.train()
    try:
        batches = source.batches(records, completed_windows=0, deadline=deadline, epochs=args.epochs)
        for raw in batches:
            if len(per_step) >= args.max_steps:
                break
            if time.monotonic() >= deadline:
                status = "time_cap"
                break
            frames = raw["frames"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            teacher = model.teacher(frames)
            optimizer.zero_grad(set_to_none=True)
            output = model(frames)
            targets = build_sparse_targets(output, raw["nodes"], raw["edges"], raw["voxel_size_um"])
            losses = multitask_loss(
                output,
                targets,
                division_enabled=False,
                center_weight=CENTER_WEIGHT,
                teacher_logits=teacher,
                teacher_weight=TEACHER_WEIGHT,
            )
            loss = losses["total"]
            if not _finite(loss):
                raise V45FastTrain("nonfinite fast-student loss")
            loss.backward()
            if any(parameter.grad is not None for parameter in model.core.parameters()):
                raise V45FastTrain("frozen V45 core unexpectedly received gradients")
            if not any(parameter.grad is not None and _finite(parameter.grad) for parameter in model.parameters() if parameter.requires_grad):
                raise V45FastTrain("fast head received no finite gradient")
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
            optimizer.step()
            model.enforce()
            if v45_state_sha256(model.core) != core_sha256:
                raise V45FastTrain("frozen V45 core changed during fast-student training")
            step = {name: float(value.detach().cpu()) for name, value in losses.items()}
            for name, value in step.items():
                sums[name] += value
            per_step.append({"step": len(per_step) + 1, "window_id": str(raw["window_id"]), "loss_components": step})
    except FastDataError as exc:
        raise V45FastTrain(f"fast GEFF/Zarr source failed: {exc}") from exc
    if not per_step and status == "completed":
        raise V45FastTrain("no usable fit-only windows were available")
    return {
        "status": status,
        "completed_steps": len(per_step),
        "loss_components": {name: value / max(1, len(per_step)) for name, value in sums.items()},
        "per_step": per_step,
        "source_window_stats": dict(source.window_stats),
        "core_state_sha256": core_sha256,
        "identity": dict(ident),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    records, _split, _provenance, _registry = _scope(args)
    device = _device(armed=args.armed)
    core, bundle = load_v45_model(args.support_root, device=device)
    model = FrozenV45Fast(core).to(device)
    ident = _ident(args, records, bundle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic(args.output_dir / "prelaunch.json", _pre(args, records, ident, device))
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "identity": ident}, sort_keys=True))
        return 0
    metrics = _train(args, records, model, ident, torch.device(device))
    checkpoint_sha256, checkpoint_ident = save_fast(
        args.output_dir / "checkpoint.pt",
        model,
        bundle,
        target_sha256=TARGET_CONTRACT_SHA256,
        steps=metrics["completed_steps"],
        train_id=ident,
    )
    report = {
        **metrics,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint": str(args.output_dir / "checkpoint.pt"),
        "checkpoint_identity": checkpoint_ident.as_dict(),
    }
    _atomic(args.output_dir / "metrics.json", report)
    print(json.dumps({"status": report["status"], "steps": report["completed_steps"], "checkpoint_sha256": checkpoint_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
