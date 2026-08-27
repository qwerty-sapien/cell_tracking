#!/usr/bin/env python3
"""Build a provenance-bound, untrained one-view V45 compact-head control.

The control exists solely to distinguish one-view execution from the effect of
the compact-head training run.  It loads the released V45 artifact but never
opens image, GEFF, training, audit, or test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.bh_model import load_v45_model
from celltrack_alpha.t_adapter import TARGET_CONTRACT_SHA256
from celltrack_alpha.v45_fast import FrozenV45Fast, save_fast


SCHEMA = "celltrack-alpha.v45-fast-zero-control.v1"
DEFAULT_PARENT_METRICS = ROOT / "reports/g02/v45f4met.json"
DEFAULT_PARENT_METRICS_SHA256 = "1d67a099377dde39c04985982be1d427b3b4d3c7abcb6b4d0fa07d79c944b4b5"


class V45ZeroControlError(ValueError):
    """A zero-control input or output violates the provenance contract."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = _canonical(value)
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


def _load_parent(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    if not path.is_file() or _sha_file(path) != expected_sha256:
        raise V45ZeroControlError("parent training metrics SHA-256 does not match the pinned value")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V45ZeroControlError("parent training metrics are unreadable") from exc
    if not isinstance(value, Mapping):
        raise V45ZeroControlError("parent training metrics must be an object")
    return value


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise V45ZeroControlError(f"{label} must be a nonempty string list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise V45ZeroControlError(f"{label} contains duplicate identities")
    return result


def parent_identity(parent: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate only the immutable identity exported by the completed parent run."""

    if parent.get("status") != "completed" or type(parent.get("completed_steps")) is not int:
        raise V45ZeroControlError("parent training metrics do not attest a completed run")
    if int(parent["completed_steps"]) <= 0:
        raise V45ZeroControlError("parent training metrics must record positive completed steps")
    checkpoint_sha256 = parent.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise V45ZeroControlError("parent training metrics lack a checkpoint SHA-256")
    identity = parent.get("identity")
    if not isinstance(identity, Mapping):
        raise V45ZeroControlError("parent training metrics lack an identity")
    required = {
        "fold",
        "registry_sha256",
        "split_sha256",
        "provenance_sha256",
        "target_contract_sha256",
        "selected_sources",
        "student_views",
        "division_emission",
    }
    if not required.issubset(identity):
        raise V45ZeroControlError("parent identity lacks required fold/provenance fields")
    if identity.get("target_contract_sha256") != TARGET_CONTRACT_SHA256:
        raise V45ZeroControlError("parent identity target contract does not match the compact head")
    if identity.get("student_views") != [0] or identity.get("division_emission") is not False:
        raise V45ZeroControlError("parent identity is not the disabled-division one-view V45 run")
    sources = identity.get("selected_sources")
    if not isinstance(sources, list) or not sources:
        raise V45ZeroControlError("parent identity has no selected fit sources")
    groups: set[str] = set()
    videos: set[str] = set()
    for row in sources:
        if not isinstance(row, Mapping) or set(row) != {"group_id", "stem", "video_id"}:
            raise V45ZeroControlError("parent source identity schema is invalid")
        group = row.get("group_id")
        stem = row.get("stem")
        video = row.get("video_id")
        if not all(isinstance(value, str) and value for value in (group, stem, video)) or video != f"train:{stem}":
            raise V45ZeroControlError("parent source identity is invalid")
        if group in groups or video in videos:
            raise V45ZeroControlError("parent source identities are not group unique")
        groups.add(group)
        videos.add(video)
    for name in ("registry_sha256", "split_sha256", "provenance_sha256"):
        value = identity.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise V45ZeroControlError(f"parent identity {name} is invalid")
    if type(identity.get("fold")) is not int or int(identity["fold"]) not in {0, 1, 2}:
        raise V45ZeroControlError("parent identity fold is invalid")
    return identity


def control_identity(parent: Mapping[str, Any], parent_metrics_sha256: str) -> dict[str, Any]:
    """Derive a new identity without claiming that the zero head was trained."""

    identity = parent_identity(parent)
    if not isinstance(parent_metrics_sha256, str) or len(parent_metrics_sha256) != 64:
        raise V45ZeroControlError("parent metrics SHA-256 is invalid")
    return {
        "schema": SCHEMA,
        "control": "one-view-zero-correction-and-auxiliary",
        "fold": int(identity["fold"]),
        "registry_sha256": str(identity["registry_sha256"]),
        "split_sha256": str(identity["split_sha256"]),
        "provenance_sha256": str(identity["provenance_sha256"]),
        "target_contract_sha256": str(identity["target_contract_sha256"]),
        "parent_metrics_sha256": parent_metrics_sha256,
        "parent_checkpoint_sha256": str(parent["checkpoint_sha256"]),
        "selected_sources": [dict(row) for row in identity["selected_sources"]],
        "student_views": [0],
        "division_emission": False,
        "image_or_geff_data_opened": False,
    }


def assert_zero_state(state: Mapping[str, torch.Tensor], trainable_parameters: int) -> None:
    """Validate the exact state required for the untrained control."""

    for name in ("corr_out.weight", "corr_out.bias", "aux_head.weight", "aux_head.bias"):
        value = state.get(name)
        if value is None or not torch.equal(value, torch.zeros_like(value)):
            raise V45ZeroControlError(f"zero control state {name} is not exactly zero")
    if trainable_parameters != FrozenV45Fast.head_params():
        raise V45ZeroControlError("zero control compact-head accounting drifted")


def assert_zero_head(model: FrozenV45Fast) -> None:
    """Prove this is raw one-view V45 detection with zero auxiliary outputs."""

    if not isinstance(model, FrozenV45Fast):
        raise V45ZeroControlError("zero control requires FrozenV45Fast")
    assert_zero_state(model.fast_state(), int(model.accounting()["trainable"]))


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-metrics", type=Path, default=DEFAULT_PARENT_METRICS)
    parser.add_argument("--parent-metrics-sha256", default=DEFAULT_PARENT_METRICS_SHA256)
    parser.add_argument("--support-root", type=Path, default=ROOT / "external" / "Tracking_Support")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise V45ZeroControlError("output-dir must not already exist")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise V45ZeroControlError("CUDA zero-control creation requires an available CUDA device")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    parent = _load_parent(args.parent_metrics, args.parent_metrics_sha256)
    identity = control_identity(parent, args.parent_metrics_sha256)
    core, bundle = load_v45_model(args.support_root, device=args.device)
    model = FrozenV45Fast(core).to(args.device).eval()
    assert_zero_head(model)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "identity": identity}, ensure_ascii=True, sort_keys=True))
        return 0
    args.output_dir.mkdir(parents=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    checkpoint_sha256, checkpoint_identity = save_fast(
        checkpoint,
        model,
        bundle,
        target_sha256=TARGET_CONTRACT_SHA256,
        steps=0,
        train_id=identity,
    )
    metrics = {
        "artifact": SCHEMA,
        "status": "completed",
        "completed_steps": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_identity": checkpoint_identity.as_dict(),
        "identity": identity,
        "control_assertions": {
            "correction_head_exactly_zero": True,
            "auxiliary_head_exactly_zero": True,
            "division_disabled": True,
            "one_view": [0],
            "dense_nxn_association": False,
            "image_or_geff_data_opened": False,
        },
        "source_isolation": {
            "fit_sources_declared_from_parent": True,
            "fit_images_opened": False,
            "fit_geff_opened": False,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
    }
    metrics_sha256 = _atomic_json(args.output_dir / "metrics.json", metrics)
    print(
        json.dumps(
            {
                "status": "completed",
                "checkpoint_sha256": checkpoint_sha256,
                "metrics_sha256": metrics_sha256,
                "steps": 0,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
