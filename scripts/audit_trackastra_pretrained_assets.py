#!/usr/bin/env python3
"""Audit a pinned Trackastra source snapshot and pretrained checkpoint archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import tarfile
from typing import Any
import zipfile

import torch
import yaml


SCHEMA = "celltrack-alpha.trackastra-pretrained-asset-audit.v1"


class TrackastraAssetAuditError(RuntimeError):
    """A pretrained Trackastra asset failed its provenance contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(path: Path, expected: str, label: str) -> str:
    if len(expected) != 64:
        raise TrackastraAssetAuditError(f"{label} expected SHA-256 is malformed")
    observed = sha256(path)
    if observed != expected:
        raise TrackastraAssetAuditError(
            f"{label} SHA-256 mismatch: {observed} != {expected}"
        )
    return observed


def _yaml_member(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(archive.read(name))
    except (KeyError, yaml.YAMLError) as exc:
        raise TrackastraAssetAuditError(
            f"checkpoint archive member is invalid: {name}"
        ) from exc
    if not isinstance(value, dict):
        raise TrackastraAssetAuditError(
            f"checkpoint archive member is not a mapping: {name}"
        )
    return value


def _checkpoint_state(
    archive: zipfile.ZipFile, name: str
) -> tuple[int, int, int]:
    try:
        payload = archive.read(name)
        state = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise TrackastraAssetAuditError(
            f"cannot load checkpoint state: {name}"
        ) from exc
    if not isinstance(state, dict) or not state:
        raise TrackastraAssetAuditError("checkpoint state is empty or malformed")
    tensors = [value for value in state.values() if torch.is_tensor(value)]
    if len(tensors) != len(state):
        raise TrackastraAssetAuditError(
            "checkpoint state contains non-tensor entries"
        )
    parameter_count = sum(int(value.numel()) for value in tensors)
    tensor_bytes = sum(
        int(value.numel()) * int(value.element_size()) for value in tensors
    )
    return len(tensors), parameter_count, tensor_bytes


def _source_license(
    source_tar: Path, expected_commit: str
) -> tuple[str, str]:
    expected_root = f"trackastra-{expected_commit}"
    try:
        with tarfile.open(source_tar, "r:*") as archive:
            roots = {
                member.name.split("/", 1)[0]
                for member in archive.getmembers()
                if member.name
            }
            if roots != {expected_root}:
                raise TrackastraAssetAuditError(
                    f"source archive root mismatch: {sorted(roots)}"
                )
            member = archive.getmember(f"{expected_root}/LICENSE")
            handle = archive.extractfile(member)
            if handle is None:
                raise TrackastraAssetAuditError(
                    "source archive LICENSE is not a regular file"
                )
            license_text = handle.read().decode("utf-8")
    except (OSError, tarfile.TarError, KeyError, UnicodeError) as exc:
        raise TrackastraAssetAuditError(
            "cannot inspect source archive"
        ) from exc
    if not license_text.startswith("BSD 3-Clause License\n"):
        raise TrackastraAssetAuditError("source license is not BSD-3-Clause")
    return expected_root, hashlib.sha256(
        license_text.encode("utf-8")
    ).hexdigest()


def audit(
    *,
    checkpoint_zip: Path,
    source_tar: Path,
    expected_checkpoint_zip_sha256: str,
    expected_source_tar_sha256: str,
    expected_source_commit: str,
    release_url: str,
    source_url: str,
) -> dict[str, Any]:
    """Return a fail-closed provenance and interface report."""

    checkpoint_zip_sha256 = _require_sha(
        checkpoint_zip,
        expected_checkpoint_zip_sha256,
        "checkpoint archive",
    )
    source_tar_sha256 = _require_sha(
        source_tar, expected_source_tar_sha256, "source archive"
    )
    with zipfile.ZipFile(checkpoint_zip) as archive:
        names = set(archive.namelist())
        required = {
            "ctc/model.pt",
            "ctc/config.yaml",
            "ctc/train_config.yaml",
        }
        missing = sorted(required - names)
        if missing:
            raise TrackastraAssetAuditError(
                f"checkpoint archive members are missing: {missing}"
            )
        config = _yaml_member(archive, "ctc/config.yaml")
        train_config = _yaml_member(archive, "ctc/train_config.yaml")
        tensor_count, parameter_count, tensor_bytes = _checkpoint_state(
            archive, "ctc/model.pt"
        )
        checkpoint_sha256 = hashlib.sha256(
            archive.read("ctc/model.pt")
        ).hexdigest()

    if train_config.get("git_commit") != expected_source_commit:
        raise TrackastraAssetAuditError(
            "checkpoint training source commit does not match the pinned source"
        )
    required_config = {
        "coord_dim": 3,
        "d_model": 512,
        "feat_dim": 12,
        "feat_embed_per_dim": 8,
        "nhead": 4,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "window": 4,
        "causal_norm": "quiet_softmax",
    }
    if any(config.get(key) != value for key, value in required_config.items()):
        raise TrackastraAssetAuditError(
            "checkpoint architecture differs from the qualified configuration"
        )
    if (
        train_config.get("features") != "wrfeat"
        or int(train_config.get("max_tokens", 0)) != 1024
    ):
        raise TrackastraAssetAuditError(
            "checkpoint feature or token contract differs"
        )
    source_root, license_sha256 = _source_license(
        source_tar, expected_source_commit
    )
    train_inputs = train_config.get("input_train")
    if not isinstance(train_inputs, list) or not train_inputs:
        raise TrackastraAssetAuditError(
            "checkpoint training dataset list is missing"
        )

    return {
        "schema": SCHEMA,
        "verdict": "STATIC-PROVENANCE-PASS_FEATURE-INTERFACE-UNRESOLVED",
        "assets": {
            "checkpoint_release_url": release_url,
            "checkpoint_archive_sha256": checkpoint_zip_sha256,
            "checkpoint_archive_bytes": checkpoint_zip.stat().st_size,
            "checkpoint_member_sha256": checkpoint_sha256,
            "source_url": source_url,
            "source_archive_sha256": source_tar_sha256,
            "source_root": source_root,
            "source_commit": expected_source_commit,
            "source_license": "BSD-3-Clause",
            "source_license_sha256": license_sha256,
        },
        "checkpoint": {
            "tensor_count": tensor_count,
            "parameter_count": parameter_count,
            "tensor_bytes": tensor_bytes,
            "architecture": required_config,
            "attention_positional_bias": config.get(
                "attn_positional_bias"
            ),
            "training_features": train_config.get("features"),
            "training_max_tokens": int(train_config["max_tokens"]),
            "training_input_count": len(train_inputs),
            "training_inputs": train_inputs,
        },
        "interface": {
            "required_feature_count": 12,
            "required_feature_family": "wrfeat-regionprops2",
            "required_feature_components": [
                "equivalent_diameter_area",
                "intensity_mean",
                "inertia_tensor",
                "border_dist",
            ],
            "competition_baseline_input": "point detections without instance masks",
            "zero_feature_probe_authorized": True,
            "zero_feature_probe_interpretation": (
                "tests frozen positional association signal only; it does not "
                "establish full checkpoint compatibility"
            ),
            "training_authorized": False,
            "validation12_access": False,
            "cv57_access": False,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-zip", type=Path, required=True)
    parser.add_argument("--source-tar", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-zip-sha256", required=True)
    parser.add_argument("--expected-source-tar-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--release-url", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit(
        checkpoint_zip=args.checkpoint_zip,
        source_tar=args.source_tar,
        expected_checkpoint_zip_sha256=(
            args.expected_checkpoint_zip_sha256
        ),
        expected_source_tar_sha256=args.expected_source_tar_sha256,
        expected_source_commit=args.expected_source_commit,
        release_url=args.release_url,
        source_url=args.source_url,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
