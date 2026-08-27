#!/usr/bin/env python3
"""Fail-closed validator for the one authorized DINOv2 ViT-S/14 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
EXPECTED_SOURCE_TREE = "2a27257b79b0633b027a21014bc9360e3c1b3f43"
EXPECTED_LICENSE_SHA256 = "600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7"
EXPECTED_WEIGHT_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
    "dinov2_vits14_pretrain.pth"
)
EXPECTED_WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
EXPECTED_WEIGHT_BYTES = 88_283_115
EXPECTED_MODEL_PARAMETER_COUNT = 22_056_576
EXPECTED_STATE_DICT_KEY_COUNT = 175


class ValidationError(RuntimeError):
    """Contract failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(source: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_source(source: Path) -> dict[str, Any]:
    revision = git_value(source, "rev-parse", "HEAD")
    tree = git_value(source, "rev-parse", "HEAD^{tree}")
    status = git_value(source, "status", "--porcelain")
    license_path = source / "LICENSE"
    license_sha256 = sha256_file(license_path)
    failures = []
    if revision != EXPECTED_SOURCE_REVISION:
        failures.append(f"source revision {revision} != {EXPECTED_SOURCE_REVISION}")
    if tree != EXPECTED_SOURCE_TREE:
        failures.append(f"source tree {tree} != {EXPECTED_SOURCE_TREE}")
    if status:
        failures.append("source worktree is not clean")
    if license_sha256 != EXPECTED_LICENSE_SHA256:
        failures.append(
            f"license sha256 {license_sha256} != {EXPECTED_LICENSE_SHA256}"
        )
    if failures:
        raise ValidationError("; ".join(failures))
    return {
        "repository": "https://github.com/facebookresearch/dinov2",
        "revision": revision,
        "tree": tree,
        "clean": True,
        "license": "Apache-2.0",
        "license_path": str(license_path),
        "license_sha256": license_sha256,
    }


def validate_weight_bytes(checkpoint: Path) -> dict[str, Any]:
    byte_count = checkpoint.stat().st_size
    digest = sha256_file(checkpoint)
    if byte_count != EXPECTED_WEIGHT_BYTES:
        raise ValidationError(
            f"checkpoint bytes {byte_count} != {EXPECTED_WEIGHT_BYTES}"
        )
    if digest != EXPECTED_WEIGHT_SHA256:
        raise ValidationError(
            f"checkpoint sha256 {digest} != {EXPECTED_WEIGHT_SHA256}"
        )
    return {
        "url": EXPECTED_WEIGHT_URL,
        "path": str(checkpoint),
        "bytes": byte_count,
        "sha256": digest,
        "hash_verified_before_deserialization": True,
    }


def require_tensor_mapping(value: Any, torch: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"checkpoint root is {type(value).__name__}, not dict")
    if not value:
        raise ValidationError("checkpoint state_dict is empty")
    non_string_keys = [repr(key) for key in value if not isinstance(key, str)]
    if non_string_keys:
        raise ValidationError(f"non-string keys: {non_string_keys[:3]}")
    non_tensor_keys = [
        key for key, tensor in value.items() if not isinstance(tensor, torch.Tensor)
    ]
    if non_tensor_keys:
        raise ValidationError(f"non-tensor values: {non_tensor_keys[:3]}")
    return value


def tensor_signature(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        row = {
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
        }
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_checkpoint(source: Path, checkpoint: Path) -> dict[str, Any]:
    # The file hash is checked by validate_weight_bytes before this function is called.
    sys.path.insert(0, str(source))
    import torch

    if not str(torch.__version__).startswith("2."):
        raise ValidationError(f"unexpected torch major version: {torch.__version__}")

    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = require_tensor_mapping(state_dict, torch)

    from dinov2.hub.backbones import dinov2_vits14

    model = dinov2_vits14(pretrained=False)
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    shape_mismatches = {
        key: {
            "checkpoint": list(state_dict[key].shape),
            "model": list(model_state[key].shape),
        }
        for key in sorted(set(model_state) & set(state_dict))
        if tuple(state_dict[key].shape) != tuple(model_state[key].shape)
    }
    if missing or unexpected or shape_mismatches:
        raise ValidationError(
            "state_dict contract mismatch: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}, "
            f"shape_mismatches={list(shape_mismatches)[:3]}"
        )

    state_dict_numel = sum(tensor.numel() for tensor in state_dict.values())
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    nonfinite_keys = []
    for key, tensor in state_dict.items():
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all()):
                nonfinite_keys.append(key)
    if len(state_dict) != EXPECTED_STATE_DICT_KEY_COUNT:
        raise ValidationError(
            f"state_dict key count {len(state_dict)} != "
            f"{EXPECTED_STATE_DICT_KEY_COUNT}"
        )
    if state_dict_numel != EXPECTED_MODEL_PARAMETER_COUNT:
        raise ValidationError(
            f"state_dict numel {state_dict_numel} != "
            f"{EXPECTED_MODEL_PARAMETER_COUNT}"
        )
    if model_parameter_count != EXPECTED_MODEL_PARAMETER_COUNT:
        raise ValidationError(
            f"model parameter count {model_parameter_count} != "
            f"{EXPECTED_MODEL_PARAMETER_COUNT}"
        )
    if nonfinite_keys:
        raise ValidationError(f"nonfinite tensors: {nonfinite_keys[:3]}")

    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValidationError(f"strict load returned incompatibility: {incompatible}")
    model.eval()
    model.requires_grad_(False)
    synthetic = torch.linspace(
        -1.0,
        1.0,
        steps=3 * 224 * 224,
        dtype=torch.float32,
    ).reshape(1, 3, 224, 224)
    with torch.inference_mode():
        first = model.forward_features(synthetic)
        second = model.forward_features(synthetic)
    class_token = first["x_norm_clstoken"]
    patch_tokens = first["x_norm_patchtokens"]
    if tuple(class_token.shape) != (1, 384):
        raise ValidationError(f"CLS shape drift: {tuple(class_token.shape)}")
    if tuple(patch_tokens.shape) != (1, 256, 384):
        raise ValidationError(f"patch shape drift: {tuple(patch_tokens.shape)}")
    if not torch.equal(class_token, second["x_norm_clstoken"]) or not torch.equal(
        patch_tokens, second["x_norm_patchtokens"]
    ):
        raise ValidationError("synthetic CPU forward is not byte-exact deterministic")
    central_indices = torch.tensor([7 * 16 + 7, 7 * 16 + 8, 8 * 16 + 7, 8 * 16 + 8])
    xy_embedding = torch.cat(
        (class_token, patch_tokens[:, central_indices].mean(dim=1)),
        dim=1,
    )
    if tuple(xy_embedding.shape) != (1, 768) or not bool(
        torch.isfinite(xy_embedding).all()
    ):
        raise ValidationError("R5 XY embedding interface drift")
    xy_embedding_sha256 = hashlib.sha256(
        xy_embedding.contiguous().numpy().tobytes()
    ).hexdigest()
    return {
        "torch_version": str(torch.__version__),
        "deserialization": "torch.load(weights_only=True,mmap=True,map_location='cpu')",
        "device": "cpu",
        "checkpoint_root_type": type(state_dict).__name__,
        "tensor_only": True,
        "state_dict_key_count": len(state_dict),
        "state_dict_keyset_sha256": hashlib.sha256(
            ("\n".join(sorted(state_dict)) + "\n").encode()
        ).hexdigest(),
        "state_dict_tensor_signature_sha256": tensor_signature(state_dict),
        "state_dict_numel": state_dict_numel,
        "model_parameter_count": model_parameter_count,
        "expected_parameter_count": EXPECTED_MODEL_PARAMETER_COUNT,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": {},
        "nonfinite_tensor_count": 0,
        "strict_load": True,
        "model_eval": not model.training,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "synthetic_cpu_forward": {
            "input_shape": [1, 3, 224, 224],
            "cls_shape": [1, 384],
            "patch_shape": [1, 256, 384],
            "patch_grid": [16, 16],
            "central_patch_indices": central_indices.tolist(),
            "xy_embedding_shape": [1, 768],
            "xy_embedding_sha256": xy_embedding_sha256,
            "repeat_byte_exact": True,
            "finite": True
        },
        "network_fallback": False,
    }


def run(source: Path, checkpoint: Path) -> dict[str, Any]:
    started = time.monotonic()
    source_result = validate_source(source)
    weight_result = validate_weight_bytes(checkpoint)
    checkpoint_result = validate_checkpoint(source, checkpoint)
    return {
        "schema": "g02.sa-dinov2-wave3.safe-load-result.v1",
        "status": "PASS",
        "source": source_result,
        "weight": weight_result,
        "validation": checkpoint_result,
        "runtime": {
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "pid": os.getpid(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.source.resolve(), args.checkpoint.resolve())
    except Exception as exc:
        result = {
            "schema": "g02.sa-dinov2-wave3.safe-load-result.v1",
            "status": "FAIL_CLOSED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
