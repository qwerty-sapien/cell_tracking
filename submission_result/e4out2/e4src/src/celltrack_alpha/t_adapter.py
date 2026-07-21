"""Released-core temporal adapter with device-correct sparse supervision."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_SHA256 = "347915de9c33883cb2ee69832a8e4552c88b1ec692d0fbfe956422467d3d4235"
WEIGHTS_BYTES = 8_357_783
CONFIG_SHA256 = "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a"
CONFIG_BYTES = 165
STATE_SCHEMA_SHA256 = "a832aed27d60ec876a2be8417bbef67718959d42c3c2e57a59a4a67e6a150d8b"
V6_CHECKPOINT_SHA256 = "712646e9e263da018239dc46e927905708d31405c0b1774d20a260d85f480987"
V6_CHECKPOINT_BYTES = 6_375_639
EXPECTED_CONFIG = {"unet_out_channels": 32, "unet_layers": [32, 64, 128], "downsample": [1, 4, 4], "window_size": 2, "pool_kernel_um": 5.0}
SCENARIOS = ("universal", "sparse_clean", "dense_crowded", "low_snr_depth")
OUTPUT_SCHEMA = ("center_logit", "backward_dz_um", "backward_dy_um", "backward_dx_um", "log_variance", "division_logit")
CENTER_SUPERVISION = "annotated-only"
CENTER_LOSS = "report-only-excluded"
DETECTOR_POLICIES = ("frozen_aux", "released_teacher_distill")
TARGET_CONTRACT = {
    "schema": "celltrack-alpha.sparse-target-contract.v1",
    "coordinate_order": "zyx",
    "displacement": {
        "owner": "child-at-t",
        "direction": "parent_um-minus-child_um",
        "units": "um",
        "coordinates": "original-fractional-annotation-times-effective-output-voxel-size",
    },
    "log_variance": {
        "owner": "child-at-t",
        "definition": "log(isotropic-3d-variance-um2-over-1-um2)",
        "nll": "0.5*(exp(-log_variance)*squared_residual_um2+3*log_variance)",
    },
    "division": {
        "owner": "parent-at-t-minus-1",
        "positive": "more-than-one-annotated-adjacent-outgoing-child",
        "negative": "exactly-one-annotated-adjacent-outgoing-child",
        "unobserved_parent": "masked",
        "enabled": False,
    },
}
TARGET_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(TARGET_CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
).hexdigest()


class CheckpointError(ValueError):
    """Raised when the attested released checkpoint is not exact."""


class TargetError(ValueError):
    """Raised when sparse labels cannot be represented unambiguously."""


@dataclass(frozen=True)
class CoreLoad:
    tensor_count: int
    tensor_numel: int
    schema_sha256: str
    discarded_tensor_count: int
    discarded_numel: int


@dataclass(frozen=True)
class TrainedLoad:
    """Attested identity of a restored compact training checkpoint."""

    checkpoint_sha256: str
    fold: int
    scenario: str
    completed_steps: int
    source_groups: tuple[str, ...]
    source_video_ids: tuple[str, ...]
    model_tensor_count: int
    model_numel: int


@dataclass(frozen=True)
class SparseTargets:
    """Voxel targets allocated on the prediction device, never CPU by default."""

    center: torch.Tensor
    center_supervision_mask: torch.Tensor
    displacement: torch.Tensor
    displacement_mask: torch.Tensor
    division: torch.Tensor
    division_mask: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.center.device


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise CheckpointError(f"missing required file: {path}")
    if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha256:
        raise CheckpointError(f"identity mismatch for {path}")


def verify_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    _verify_file(path, CONFIG_BYTES, CONFIG_SHA256)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError("released config is not valid JSON") from exc
    if config != EXPECTED_CONFIG:
        raise CheckpointError("released config content mismatch")
    return config


def verify_released_files(weight_path: str | Path, config_path: str | Path) -> None:
    verify_config(config_path)
    _verify_file(Path(weight_path), WEIGHTS_BYTES, WEIGHTS_SHA256)


def _finite(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())


def _load_state(weight_path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(weight_path)
    _verify_file(path, WEIGHTS_BYTES, WEIGHTS_SHA256)
    try:
        first = torch.load(path, map_location="cpu", weights_only=True)
        second = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError("weights_only CPU loading failed") from exc
    if not isinstance(first, dict) or not isinstance(second, dict) or tuple(first) != tuple(second) or len(first) != 136:
        raise CheckpointError("checkpoint tensor schema drift")
    for name, tensor in first.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or not _finite(tensor):
            raise CheckpointError("checkpoint contains an invalid tensor")
        if not torch.equal(tensor, second[name]):
            raise CheckpointError("checkpoint double-load was not identical")
    return first


def _partition_state(state: Mapping[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    groups: dict[str, dict[str, torch.Tensor]] = {"unet": {}, "detect_head": {}, "transformer": {}}
    for name, value in state.items():
        prefix, sep, suffix = name.partition(".")
        if not sep or prefix not in groups or not suffix:
            raise CheckpointError(f"unknown checkpoint key: {name}")
        groups[prefix][suffix] = value
    if not all(groups.values()):
        raise CheckpointError("checkpoint lacks a released component")
    return groups


def load_released_core(backbone: nn.Module, weight_path: str | Path, config_path: str | Path) -> tuple[nn.Conv3d, CoreLoad]:
    """Strictly restore released U-Net and detector, discarding association weights."""
    verify_config(config_path)
    state = _load_state(weight_path)
    groups = _partition_state(state)
    if len(state) != 136 or sum(v.numel() for v in state.values()) != 2_077_996:
        raise CheckpointError("checkpoint tensor accounting mismatch")
    if sum(v.numel() for v in groups["transformer"].values()) != 580_353:
        raise CheckpointError("discarded component accounting mismatch")
    try:
        backbone.load_state_dict(groups["unet"], strict=True)
        detector = nn.Conv3d(32, 1, kernel_size=1)
        detector.load_state_dict(groups["detect_head"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError("strict released-core restore failed") from exc
    return detector, CoreLoad(len(state), sum(v.numel() for v in state.values()), STATE_SCHEMA_SHA256,
                              len(groups["transformer"]), sum(v.numel() for v in groups["transformer"].values()))


class _Specialist(nn.Module):
    """Identity-start residual specialist; exits are exactly zero initialized."""

    def __init__(self) -> None:
        super().__init__()
        self.reduce = nn.Conv3d(32, 8, 1)
        self.expand = nn.Conv3d(8, 32, 1)
        self.calibrate = nn.Conv3d(6, 6, 1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        nn.init.zeros_(self.calibrate.weight)
        nn.init.zeros_(self.calibrate.bias)

    def forward(self, features: torch.Tensor, universal: torch.Tensor, aux_head: nn.Module) -> torch.Tensor:
        adjusted = features + self.expand(F.gelu(self.reduce(features)))
        branch = torch.cat((universal[:, :1], aux_head(adjusted)), dim=1)
        return branch + self.calibrate(branch)


class TemporalAdapterModel(nn.Module):
    """Released temporal features plus a universal head and three specialists."""

    def __init__(self, backbone: nn.Module, detect_head: nn.Conv3d) -> None:
        super().__init__()
        self.backbone = backbone
        self.detect_head = detect_head
        self.aux_head = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.GELU(), nn.Conv3d(32, 5, 1))
        self.specialists = nn.ModuleDict({"sparse_clean": _Specialist(), "dense_crowded": _Specialist(), "low_snr_depth": _Specialist()})
        self.detector_policy = "frozen_aux"
        self.set_unfreeze_depth(0)

    @staticmethod
    def scenarios() -> tuple[str, ...]:
        return SCENARIOS

    def set_unfreeze_depth(self, depth: int) -> dict[str, int]:
        return self.set_training_policy(depth, detector_policy="frozen_aux")

    def set_training_policy(self, depth: int, *, detector_policy: str = "frozen_aux") -> dict[str, int]:
        if depth not in (0, 1, 2):
            raise ValueError("unfreeze depth must be 0, 1, or 2")
        if detector_policy not in DETECTOR_POLICIES:
            raise ValueError(f"unknown detector policy: {detector_policy}")
        if detector_policy == "released_teacher_distill" and depth != 0:
            raise ValueError("released teacher detector distillation requires frozen backbone depth 0")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.detect_head.parameters():
            parameter.requires_grad = detector_policy == "released_teacher_distill"
        if depth >= 1:
            if not hasattr(self.backbone, "head"):
                raise ValueError("released backbone must expose head")
            for parameter in self.backbone.head.parameters():
                parameter.requires_grad = True
        if depth >= 2:
            blocks = getattr(self.backbone, "decoder_blocks", None)
            if not isinstance(blocks, nn.ModuleList) or not blocks:
                raise ValueError("released backbone must expose decoder blocks")
            for parameter in blocks[-1].parameters():
                parameter.requires_grad = True
        self.detector_policy = detector_policy
        self.detect_head.train(detector_policy == "released_teacher_distill")
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self.parameter_accounting()

    def parameter_accounting(self) -> dict[str, int]:
        parameters = tuple(self.parameters())
        total = sum(p.numel() for p in parameters)
        trainable = sum(p.numel() for p in parameters if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "backbone_trainable": sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
            "detect_head_trainable": sum(p.numel() for p in self.detect_head.parameters() if p.requires_grad),
            "aux_head_trainable": sum(p.numel() for p in self.aux_head.parameters() if p.requires_grad),
        }

    def train(self, mode: bool = True) -> "TemporalAdapterModel":
        """Train compact heads without mutating the frozen released core."""
        super().train(mode)
        self.detect_head.train(mode and self.detector_policy == "released_teacher_distill")
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor, scenario: str = "universal") -> torch.Tensor:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        if not isinstance(images, torch.Tensor) or images.ndim != 6 or images.shape[2] != 1 or min(images.shape) < 1:
            raise ValueError("images must have nonempty shape (B,T,1,Z,Y,X)")
        if not _finite(images):
            raise ValueError("images must be finite")
        batch, steps, _, z, y, x = images.shape
        features = self.backbone(images)
        if tuple(features.shape) != (batch, steps, 32, z, y, x):
            raise ValueError("released backbone returned an incompatible feature shape")
        flat = features.reshape(batch * steps, 32, z, y, x)
        universal = torch.cat((self.detect_head(flat), self.aux_head(flat)), dim=1)
        output = universal if scenario == "universal" else self.specialists[scenario](flat, universal, self.aux_head)
        if tuple(output.shape) != (batch * steps, 6, z, y, x):
            raise RuntimeError("adapter output schema violation")
        return output.reshape(batch, steps, 6, z, y, x)


def load_trained_model(
    backbone: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = V6_CHECKPOINT_SHA256,
    expected_size: int = V6_CHECKPOINT_BYTES,
    expected_fold: int | None = None,
) -> tuple[TemporalAdapterModel, TrainedLoad]:
    """Strictly restore a compact trainer checkpoint without released NxN weights.

    The training artifact contains the adapter model state and optimizer resume
    state.  Inference restores only the exact compact model mapping, while the
    fit-source and fold identity remain available to the caller for split-scope
    enforcement.
    """

    path = Path(checkpoint_path)
    _verify_file(path, expected_size, expected_sha256)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError("trained checkpoint weights_only CPU loading failed") from exc
    required = {
        "completed_steps",
        "completed_windows",
        "elapsed_seconds",
        "execution",
        "identity",
        "loss_components",
        "model",
        "optimizer",
        "scaler",
        "source_groups",
        "source_records",
        "source_video_ids",
        "status",
    }
    if not isinstance(state, dict) or set(state) != required:
        raise CheckpointError("trained checkpoint top-level schema drift")
    identity = state.get("identity")
    model_state = state.get("model")
    if not isinstance(identity, Mapping) or not isinstance(model_state, Mapping):
        raise CheckpointError("trained checkpoint identity or model state is invalid")
    fold = identity.get("fold")
    scenario = identity.get("scenario")
    completed_steps = state.get("completed_steps")
    if isinstance(fold, bool) or fold not in (0, 1, 2):
        raise CheckpointError("trained checkpoint fold is invalid")
    if expected_fold is not None and fold != expected_fold:
        raise CheckpointError(f"trained checkpoint fold {fold} does not match requested fold {expected_fold}")
    if scenario != "universal":
        raise CheckpointError("trained checkpoint is not the universal edge-only candidate")
    if isinstance(completed_steps, bool) or not isinstance(completed_steps, int) or completed_steps <= 0:
        raise CheckpointError("trained checkpoint has no completed optimizer steps")
    if state.get("status") != "completed" or state.get("completed_windows") != completed_steps:
        raise CheckpointError("trained checkpoint completion state is inconsistent")
    execution = state.get("execution")
    if not isinstance(execution, Mapping) or execution.get("max_steps") != completed_steps:
        raise CheckpointError("trained checkpoint execution identity is inconsistent")

    records = identity.get("selected_sources")
    if not isinstance(records, list) or not records:
        raise CheckpointError("trained checkpoint has no selected fit sources")
    try:
        source_groups = tuple(str(record["group_id"]) for record in records)
        source_video_ids = tuple(str(record["video_id"]) for record in records)
    except (KeyError, TypeError) as exc:
        raise CheckpointError("trained checkpoint selected-source schema drift") from exc
    if (
        len(source_groups) != len(set(source_groups))
        or len(source_video_ids) != len(set(source_video_ids))
        or tuple(state.get("source_groups", ())) != source_groups
        or tuple(state.get("source_video_ids", ())) != source_video_ids
    ):
        raise CheckpointError("trained checkpoint source identity is inconsistent")

    detector = nn.Conv3d(32, 1, kernel_size=1)
    model = TemporalAdapterModel(backbone, detector)
    expected_names = tuple(model.state_dict())
    if tuple(model_state) != expected_names:
        raise CheckpointError("trained compact model tensor schema drift")
    for name, value in model_state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor) or not _finite(value):
            raise CheckpointError("trained compact model contains an invalid tensor")
    try:
        model.load_state_dict(model_state, strict=True)
        detector_policy = str(identity.get("detector_policy", "frozen_aux"))
        model.set_training_policy(int(identity.get("unfreeze")), detector_policy=detector_policy)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError("strict trained compact model restore failed") from exc
    model.eval()
    return model, TrainedLoad(
        checkpoint_sha256=expected_sha256,
        fold=int(fold),
        scenario=str(scenario),
        completed_steps=completed_steps,
        source_groups=source_groups,
        source_video_ids=source_video_ids,
        model_tensor_count=len(model_state),
        model_numel=sum(value.numel() for value in model_state.values()),
    )


def _require_int(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TargetError(f"{name} must be an integer")
    return value


def _require_float(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TargetError(f"{name} must be finite")
    return float(value)


def _voxel(row: Mapping[str, Any], shape: tuple[int, int, int]) -> tuple[int, int, int]:
    value = tuple(int(round(_require_float(row, axis))) for axis in ("z", "y", "x"))
    if any(index < 0 or index >= limit for index, limit in zip(value, shape)):
        raise TargetError("node coordinate is outside the image")
    return value


def _physical_point(row: Mapping[str, Any], scale: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return the original fractional annotation in effective output-grid µm."""

    return tuple(_require_float(row, axis) * unit for axis, unit in zip(("z", "y", "x"), scale, strict=True))


def _target_spec(output_or_shape: torch.Tensor | Sequence[int], device: torch.device | str | None) -> tuple[tuple[int, ...], torch.device]:
    if isinstance(output_or_shape, torch.Tensor):
        if device is not None and torch.device(device) != output_or_shape.device:
            raise TargetError("explicit target device disagrees with prediction device")
        return tuple(int(v) for v in output_or_shape.shape), output_or_shape.device
    if device is None:
        raise TargetError("shape-only target construction requires an explicit device")
    return tuple(int(v) for v in output_or_shape), torch.device(device)


def build_sparse_targets(output_or_shape: torch.Tensor | Sequence[int], nodes: Iterable[Mapping[str, Any]],
                         edges: Iterable[Mapping[str, Any]], voxel_size_um: Sequence[float] = (1.625, 0.40625, 0.40625),
                         *, device: torch.device | str | None = None) -> SparseTargets:
    """Build sparse labels using the locked child-backward physical target contract."""
    output_shape, target_device = _target_spec(output_or_shape, device)
    if len(output_shape) != 6 or output_shape[2] != 6 or min(output_shape) < 1:
        raise TargetError("output must have positive shape (B,T,6,Z,Y,X)")
    batch_size, steps, _, z, y, x = output_shape
    scale = tuple(float(v) for v in voxel_size_um)
    if len(scale) != 3 or any(not math.isfinite(v) or v <= 0 for v in scale):
        raise TargetError("voxel_size_um must be three positive finite values")
    target_shape = (batch_size, steps, 1, z, y, x)
    center = torch.zeros(target_shape, dtype=torch.float32, device=target_device)
    center_supervision_mask = torch.zeros(target_shape, dtype=torch.bool, device=target_device)
    displacement = torch.zeros((batch_size, steps, 3, z, y, x), dtype=torch.float32, device=target_device)
    displacement_mask = torch.zeros(target_shape, dtype=torch.bool, device=target_device)
    division = torch.zeros(target_shape, dtype=torch.float32, device=target_device)
    division_mask = torch.zeros(target_shape, dtype=torch.bool, device=target_device)
    records: dict[int, tuple[int, int, tuple[int, int, int], tuple[float, float, float]]] = {}
    occupied: set[tuple[int, int, tuple[int, int, int]]] = set()
    for row in nodes:
        if not isinstance(row, Mapping):
            raise TargetError("node rows must be mappings")
        node_id, batch, time = _require_int(row, "id"), _require_int(row, "batch"), _require_int(row, "time")
        if node_id in records or not (0 <= batch < batch_size and 0 <= time < steps):
            raise TargetError("duplicate node id or invalid node batch/time")
        point = _voxel(row, (z, y, x))
        key = (batch, time, point)
        if key in occupied:
            raise TargetError("two annotations collapse to one target voxel")
        occupied.add(key)
        records[node_id] = (batch, time, point, _physical_point(row, scale))
        center[batch, time, 0, *point] = 1.0
        center_supervision_mask[batch, time, 0, *point] = True
    children: dict[int, list[int]] = {}
    parent_for_child: dict[int, int] = {}
    for row in edges:
        if not isinstance(row, Mapping):
            raise TargetError("edge rows must be mappings")
        source, target = _require_int(row, "source_id"), _require_int(row, "target_id")
        if source not in records or target not in records:
            raise TargetError("edge endpoint is missing from nodes")
        if target in parent_for_child:
            raise TargetError("child has more than one annotated parent")
        parent_for_child[target] = source
        children.setdefault(source, []).append(target)
    for child, parent in parent_for_child.items():
        pb, pt, _pp, parent_um = records[parent]
        cb, ct, cp, child_um = records[child]
        if pb != cb or ct != pt + 1:
            raise TargetError("edges must join consecutive frames in one batch")
        vector = tuple(parent_axis - child_axis for parent_axis, child_axis in zip(parent_um, child_um, strict=True))
        displacement[cb, ct, :, *cp] = torch.tensor(vector, dtype=torch.float32, device=target_device)
        displacement_mask[cb, ct, 0, *cp] = True
    for parent, child_ids in children.items():
        batch, time, point, _parent_um = records[parent]
        division[batch, time, 0, *point] = float(len(child_ids) > 1)
        division_mask[batch, time, 0, *point] = True
    return SparseTargets(center, center_supervision_mask, displacement, displacement_mask, division, division_mask)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.to(values.dtype)).sum() / mask.sum().to(values.dtype).clamp_min(1.0)


def multitask_loss(
    output: torch.Tensor,
    targets: SparseTargets,
    division_enabled: bool = False,
    *,
    center_weight: float = 0.0,
    teacher_logits: torch.Tensor | None = None,
    teacher_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute sparse losses without reclassifying unlabeled voxels as negatives.

    The default is the historical auxiliary-only objective. The opt-in
    detector mode adds only annotated positive centers and an all-voxel anchor
    to frozen released-detector logits; unlabeled voxels are never negatives.
    """
    if output.device != targets.device:
        raise TargetError("prediction and sparse targets must share a device")
    if tuple(output.shape) != tuple(targets.center.shape[:2]) + (6,) + tuple(targets.center.shape[3:]):
        raise TargetError("output and target shapes differ")
    if targets.center_supervision_mask.dtype is not torch.bool:
        raise TargetError("center supervision mask must be boolean")
    if targets.center_supervision_mask.device != output.device:
        raise TargetError("center supervision mask must share the prediction device")
    if tuple(targets.center_supervision_mask.shape) != tuple(targets.center.shape):
        raise TargetError("center supervision mask shape differs from center target")
    if not math.isfinite(center_weight) or center_weight < 0 or not math.isfinite(teacher_weight) or teacher_weight < 0:
        raise TargetError("center and teacher weights must be finite and nonnegative")
    center_terms = F.binary_cross_entropy_with_logits(output[:, :, :1], targets.center, reduction="none")
    center = _masked_mean(center_terms, targets.center_supervision_mask)
    teacher = output.new_zeros(())
    if teacher_logits is not None:
        if tuple(teacher_logits.shape) != tuple(output[:, :, :1].shape) or teacher_logits.device != output.device:
            raise TargetError("teacher logits must match the center output on its device")
        if not _finite(teacher_logits):
            raise TargetError("teacher logits must be finite")
        teacher = F.mse_loss(output[:, :, :1], teacher_logits.detach())
    elif teacher_weight != 0:
        raise TargetError("nonzero teacher weight requires teacher logits")
    residual = output[:, :, 1:4] - targets.displacement
    absolute_error = residual.abs().sum(dim=2, keepdim=True)
    squared_error = residual.square().sum(dim=2, keepdim=True)
    displacement = _masked_mean(absolute_error, targets.displacement_mask)
    logged = output[:, :, 4:5].clamp(-10.0, 10.0)
    variance = _masked_mean(
        0.5 * (torch.exp(-logged) * squared_error + 3.0 * logged),
        targets.displacement_mask,
    )
    division = (_masked_mean(F.binary_cross_entropy_with_logits(output[:, :, 5:6], targets.division, reduction="none"), targets.division_mask)
                if division_enabled else output.new_zeros(()))
    return {
        "total": displacement + variance + division + center_weight * center + teacher_weight * teacher,
        "center": center,
        "teacher": teacher,
        "displacement": displacement,
        "log_variance": variance,
        "division": division,
    }
