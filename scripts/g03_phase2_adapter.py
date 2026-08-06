#!/usr/bin/env python3
"""Versioned G-03 Phase-2 crop adapter and immutable h000 publisher.

The historical Wave-R R1 adapter is deliberately not imported or modified.
This module owns the Phase-2 ``48 -> 12`` interface.  Its final projection is
zero-initialized so every h000 ``wrfeat`` value is bitwise zero while the
projection remains trainable from the first optimization step.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import nn


CHECKPOINT_SCHEMA = "celltrack-alpha.g03-phase2-h000.v1"
ADAPTER_SCHEMA = "celltrack-alpha.g03-phase2-zero-init-crop-adapter.v1"
ARCHITECTURE = (
    "AvgPool3d(4,8,8)->Conv3d(3,16,k3,p1)->GELU->"
    "Conv3d(16,48,k3,p1)->GELU->AdaptiveAvgPool3d(1)->"
    "Linear(48,12;weight=0,bias=0)->tanh"
)


class Phase2AdapterError(RuntimeError):
    """The Phase-2 adapter/checkpoint contract failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to(device="cpu").contiguous()
        array = tensor.numpy()
        encoded = name.encode("utf-8")
        dtype = array.dtype.str.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(dtype).to_bytes(8, "big"))
        digest.update(dtype)
        digest.update(len(array.shape).to_bytes(8, "big"))
        for extent in array.shape:
            digest.update(int(extent).to_bytes(8, "big"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class Phase2CropAdapter(nn.Module):
    """Small 3-D crop encoder with an exact-zero 12-channel output at h000."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial_pool = nn.AvgPool3d(
            kernel_size=(4, 8, 8), stride=(4, 8, 8)
        )
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=True)
        self.activation1 = nn.GELU(approximate="none")
        self.conv2 = nn.Conv3d(16, 48, kernel_size=3, padding=1, bias=True)
        self.activation2 = nn.GELU(approximate="none")
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(48, 12, bias=True)
        self.output_activation = nn.Tanh()
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        """Make h000 exactly match Trackastra's former zero-feature input."""
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    @staticmethod
    def normalize(crops: torch.Tensor) -> torch.Tensor:
        if crops.ndim != 5 or tuple(crops.shape[1:]) != (3, 16, 32, 32):
            raise Phase2AdapterError(
                "adapter crops must have shape [N,3,16,32,32]"
            )
        value = crops.to(dtype=torch.float32)
        mean = value.mean(dim=(1, 2, 3, 4), keepdim=True)
        centered = value - mean
        rms = torch.sqrt(
            centered.square().mean(dim=(1, 2, 3, 4), keepdim=True)
        ).clamp_min(1.0e-6)
        return torch.clamp(centered / rms, -5.0, 5.0)

    def hidden(self, crops: torch.Tensor) -> torch.Tensor:
        value = self.spatial_pool(self.normalize(crops))
        value = self.activation1(self.conv1(value))
        value = self.activation2(self.conv2(value))
        value = self.global_pool(value).flatten(1)
        if value.shape[1:] != (48,) or not bool(torch.isfinite(value).all()):
            raise Phase2AdapterError("adapter hidden representation is malformed")
        return value

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        value = self.output_activation(self.projection(self.hidden(crops)))
        if value.shape[1:] != (12,) or not bool(torch.isfinite(value).all()):
            raise Phase2AdapterError("adapter emitted malformed wrfeat")
        return value


def deterministic_probe(batch_size: int = 2) -> torch.Tensor:
    values = torch.arange(
        batch_size * 3 * 16 * 32 * 32, dtype=torch.float32
    ).reshape(batch_size, 3, 16, 32, 32)
    return torch.sin(values * 0.001) + torch.cos(values * 0.00017)


def adapter_census(adapter: Phase2CropAdapter) -> dict[str, Any]:
    return {
        "schema": ADAPTER_SCHEMA,
        "architecture": ARCHITECTURE,
        "parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in adapter.parameters()
            if parameter.requires_grad
        ),
        "projection": {
            "in_features": adapter.projection.in_features,
            "out_features": adapter.projection.out_features,
            "bias": adapter.projection.bias is not None,
            "weight_nonzero": int(torch.count_nonzero(adapter.projection.weight)),
            "bias_nonzero": int(torch.count_nonzero(adapter.projection.bias)),
        },
    }


def gradient_probe(adapter: Phase2CropAdapter) -> dict[str, Any]:
    adapter.train()
    adapter.zero_grad(set_to_none=True)
    crops = deterministic_probe(batch_size=2)
    target = torch.linspace(-0.5, 0.5, 24, dtype=torch.float32).reshape(2, 12)
    hidden = adapter.hidden(crops)
    loss = (adapter(crops) - target).square().mean()
    loss.backward()
    projection_weight_grad = float(adapter.projection.weight.grad.abs().sum())
    projection_bias_grad = float(adapter.projection.bias.grad.abs().sum())
    encoder_grad = float(
        sum(
            parameter.grad.abs().sum()
            for name, parameter in adapter.named_parameters()
            if not name.startswith("projection.") and parameter.grad is not None
        )
    )
    return {
        "hidden_abs_sum": float(hidden.detach().abs().sum()),
        "projection_weight_grad_abs_sum": projection_weight_grad,
        "projection_bias_grad_abs_sum": projection_bias_grad,
        "encoder_grad_abs_sum_at_first_backward": encoder_grad,
        "projection_trainable_on_first_backward": (
            projection_weight_grad > 0.0 and projection_bias_grad > 0.0
        ),
        "encoder_first_backward_blocked_by_zero_projection": encoder_grad == 0.0,
    }


def atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Phase2AdapterError(f"refusing to overwrite immutable checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(dict(value), temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        temporary_path.unlink()
    finally:
        temporary_path.unlink(missing_ok=True)


def publish_h000(
    *,
    output: Path,
    seed: int,
    phase2_contract_path: Path,
    phase2_contract_sha256: str,
    trackastra_model_path: Path,
    trackastra_archive_path: Path,
) -> dict[str, Any]:
    if sha256(phase2_contract_path) != phase2_contract_sha256:
        raise Phase2AdapterError("Phase-2 contract SHA-256 mismatch")
    if not trackastra_model_path.is_file() or not trackastra_archive_path.is_file():
        raise Phase2AdapterError("Trackastra checkpoint carrier is incomplete")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    adapter = Phase2CropAdapter()
    census = adapter_census(adapter)
    with torch.inference_mode():
        output_probe = adapter(deterministic_probe())
    if not bool(torch.equal(output_probe, torch.zeros_like(output_probe))):
        raise Phase2AdapterError("h000 adapter output is not bitwise zero")
    gradients = gradient_probe(adapter)
    if not gradients["projection_trainable_on_first_backward"]:
        raise Phase2AdapterError("zero-init projection is not trainable")
    adapter.zero_grad(set_to_none=True)
    state = adapter.state_dict()
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_tag": "h000",
        "cumulative_t4_hours": 0.0,
        "seed": int(seed),
        "phase2_contract": {
            "path": str(phase2_contract_path),
            "sha256": phase2_contract_sha256,
        },
        "trackastra": {
            "model_path": str(trackastra_model_path),
            "model_sha256": sha256(trackastra_model_path),
            "archive_path": str(trackastra_archive_path),
            "archive_sha256": sha256(trackastra_archive_path),
        },
        "adapter": census
        | {
            "state_tensor_sha256": state_tensor_sha256(state),
            "state_dict": state,
            "output_probe_nonzero": int(torch.count_nonzero(output_probe)),
            "gradient_probe": gradients,
        },
        "training": {
            "optimizer_steps": 0,
            "scientific_gpu_hours": 0.0,
            "optimizer_state": "UNCREATED_UNTIL_BRANCH_B_FUNDING_GATE",
            "scheduler_state": "UNCREATED_UNTIL_BRANCH_B_FUNDING_GATE",
            "amp_scaler_state": "UNCREATED_UNTIL_BRANCH_B_FUNDING_GATE",
        },
        "rng": {
            "torch_cpu_state": torch.get_rng_state(),
        },
    }
    atomic_torch_save(output, payload)
    return {
        "path": str(output),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "checkpoint_tag": "h000",
        "cumulative_t4_hours": 0.0,
        "adapter_state_tensor_sha256": payload["adapter"][
            "state_tensor_sha256"
        ],
        "adapter_output_probe_nonzero": 0,
        "gradient_probe": gradients,
        "trackastra": payload["trackastra"],
        "phase2_contract": payload["phase2_contract"],
    }


def validate_h000(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(value, dict)
        or value.get("schema") != CHECKPOINT_SCHEMA
        or value.get("checkpoint_tag") != "h000"
        or float(value.get("cumulative_t4_hours", -1.0)) != 0.0
        or int(value.get("training", {}).get("optimizer_steps", -1)) != 0
    ):
        raise Phase2AdapterError("h000 checkpoint metadata drifted")
    contract = value["phase2_contract"]
    contract_path = Path(contract["path"])
    if sha256(contract_path) != contract["sha256"]:
        raise Phase2AdapterError("h000 Phase-2 contract reference drifted")
    for label, key in (("model", "model_sha256"), ("archive", "archive_sha256")):
        target = Path(value["trackastra"][f"{label}_path"])
        if sha256(target) != value["trackastra"][key]:
            raise Phase2AdapterError(f"Trackastra {label} reference drifted")
    state = value["adapter"]["state_dict"]
    if state_tensor_sha256(state) != value["adapter"]["state_tensor_sha256"]:
        raise Phase2AdapterError("adapter state tensor hash drifted")
    adapter = Phase2CropAdapter()
    adapter.load_state_dict(state, strict=True)
    census = adapter_census(adapter)
    if (
        census["projection"]["in_features"] != 48
        or census["projection"]["out_features"] != 12
        or census["projection"]["weight_nonzero"] != 0
        or census["projection"]["bias_nonzero"] != 0
    ):
        raise Phase2AdapterError("h000 projection is not exact zero Linear(48,12)")
    with torch.inference_mode():
        probe = adapter(deterministic_probe())
    if not bool(torch.equal(probe, torch.zeros_like(probe))):
        raise Phase2AdapterError("h000 output is not bitwise zero")
    gradients = gradient_probe(adapter)
    if not gradients["projection_trainable_on_first_backward"]:
        raise Phase2AdapterError("h000 projection gradient probe failed")
    return {
        "status": "PASS_H000_ZERO_INIT_AND_TRAINABILITY",
        "checkpoint": {"path": str(path), "sha256": sha256(path)},
        "adapter_state_tensor_sha256": value["adapter"]["state_tensor_sha256"],
        "projection": census["projection"],
        "output_probe_nonzero": int(torch.count_nonzero(probe)),
        "gradient_probe": gradients,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish-h000")
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("--seed", type=int, required=True)
    publish.add_argument("--phase2-contract", type=Path, required=True)
    publish.add_argument("--phase2-contract-sha256", required=True)
    publish.add_argument("--trackastra-model", type=Path, required=True)
    publish.add_argument("--trackastra-archive", type=Path, required=True)
    validate = subparsers.add_parser("validate-h000")
    validate.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "publish-h000":
        result = publish_h000(
            output=args.output,
            seed=args.seed,
            phase2_contract_path=args.phase2_contract,
            phase2_contract_sha256=args.phase2_contract_sha256,
            trackastra_model_path=args.trackastra_model,
            trackastra_archive_path=args.trackastra_archive,
        )
    else:
        result = validate_h000(args.checkpoint)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
