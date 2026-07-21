"""Compact one-pass student for the frozen V45 D4 detector ensemble.

The released V45 core remains immutable.  During training, the exact
sequential eight-view D4 mean is used only as a teacher.  At inference the
student runs the unaugmented V45 encoder once, then applies a small residual
centre head and the sparse motion head at the resulting feature map.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bh_model import (
    ArtifactBundle,
    D4_VIEWS,
    PairContext,
    V45_CONFIG,
    V45_CONFIG_SHA256,
    VideoInference,
    _detect_batch,
    _encode_d4,
    _frame_numpy,
    _index_pair_map,
    _require_torch_and_models,
    dedup_dets,
    quantile_normalize,
)
from .v45_head import (
    AUXILIARY_CHANNELS,
    FEATURE_CHANNELS,
    V45AuxCheckpointIdentity,
    V45AuxReadout,
    V45HeadError,
    auxiliary_state_sha256,
    v45_state_sha256,
)


FAST_CHAN = 32
FAST_OUT = 6
FAST_SCHEMA = "celltrack-alpha.v45-fast.v1"
FAST_INFER_SCHEMA = "celltrack-alpha.v45-fast-infer.v1"
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class V45FastError(V45HeadError):
    """Raised when the compact one-pass V45 student contract is violated."""


def _finite(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise V45FastError(f"{label} must be a lowercase SHA-256")
    return value


def _canon(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    if not isinstance(state, Mapping) or not state:
        raise V45FastError("fast state must be a nonempty tensor mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise V45FastError("fast state contains an invalid entry")
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype != torch.float32 or not _finite(tensor):
            raise V45FastError("fast state tensors must be finite float32")
        digest.update(_canon({"name": name, "shape": list(tensor.shape), "dtype": "float32"}))
        digest.update(b"\x00")
        digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass(frozen=True)
class FastIdent:
    manifest_sha256: str
    artifact_config_sha256: str
    source_sha256: str
    weight_sha256: str
    runtime_config_sha256: str
    target_contract_sha256: str
    core_state_sha256: str
    state_sha256: str
    train_sha256: str
    steps: int
    division_enabled: bool
    schema: str = FAST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FAST_SCHEMA:
            raise V45FastError("fast checkpoint schema is unsupported")
        for name in (
            "manifest_sha256",
            "artifact_config_sha256",
            "source_sha256",
            "weight_sha256",
            "runtime_config_sha256",
            "target_contract_sha256",
            "core_state_sha256",
            "state_sha256",
            "train_sha256",
        ):
            _sha(getattr(self, name), name)
        if type(self.steps) is not int or self.steps < 0:
            raise V45FastError("fast checkpoint steps must be a nonnegative integer")
        if self.division_enabled is not False:
            raise V45FastError("fast checkpoint cannot enable division before promotion")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "manifest_sha256": self.manifest_sha256,
            "artifact_config_sha256": self.artifact_config_sha256,
            "source_sha256": self.source_sha256,
            "weight_sha256": self.weight_sha256,
            "runtime_config_sha256": self.runtime_config_sha256,
            "target_contract_sha256": self.target_contract_sha256,
            "core_state_sha256": self.core_state_sha256,
            "state_sha256": self.state_sha256,
            "train_sha256": self.train_sha256,
            "steps": self.steps,
            "division_enabled": self.division_enabled,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FastIdent":
        if not isinstance(value, Mapping):
            raise V45FastError("fast checkpoint identity must be a mapping")
        expected = {
            "schema",
            "manifest_sha256",
            "artifact_config_sha256",
            "source_sha256",
            "weight_sha256",
            "runtime_config_sha256",
            "target_contract_sha256",
            "core_state_sha256",
            "state_sha256",
            "train_sha256",
            "steps",
            "division_enabled",
        }
        if set(value) != expected:
            raise V45FastError("fast checkpoint identity has missing or unexpected fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class FastVideo:
    """One-pass student detections with a non-V45 execution identity.

    ``video`` uses the canonical node and sparse pair-context records, but its
    one raw encoder pass must never be confused with the released eight-view
    V45 detector.  The compact-head and frozen-core hashes make that boundary
    explicit for caching and later graph scoring.
    """

    video: VideoInference
    core_state_sha256: str
    fast_state_sha256: str
    schema: str = FAST_INFER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FAST_INFER_SCHEMA:
            raise V45FastError("fast inference schema is unsupported")
        if not isinstance(self.video, VideoInference):
            raise V45FastError("fast inference must contain canonical video records")
        if self.video.d4_views != (0,) or self.video.view_batch_size != 1:
            raise V45FastError("fast inference must record exactly one raw encoder view")
        _sha(self.core_state_sha256, "core_state_sha256")
        _sha(self.fast_state_sha256, "fast_state_sha256")

    @property
    def execution_sha256(self) -> str:
        return hashlib.sha256(
            _canon(
                {
                    "schema": self.schema,
                    "video_config_sha256": self.video.config_sha256,
                    "core_state_sha256": self.core_state_sha256,
                    "fast_state_sha256": self.fast_state_sha256,
                }
            )
        ).hexdigest()


class FrozenV45Fast(nn.Module):
    """Frozen V45 encoder with a 1,254-parameter centre/motion student head."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        if not isinstance(core, nn.Module) or not callable(getattr(core, "encode", None)):
            raise V45FastError("core must be an nn.Module exposing encode(images)")
        self.core = core
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0x45FA)
            self.corr_hid = nn.Conv3d(FEATURE_CHANNELS, FAST_CHAN, kernel_size=1)
            self.corr_out = nn.Conv3d(FAST_CHAN, 1, kernel_size=1)
        self.aux_head = nn.Conv3d(FEATURE_CHANNELS, AUXILIARY_CHANNELS, kernel_size=1)
        nn.init.zeros_(self.corr_out.weight)
        nn.init.zeros_(self.corr_out.bias)
        nn.init.zeros_(self.aux_head.weight)
        nn.init.zeros_(self.aux_head.bias)
        self._freeze_core()
        self._check_head()

    @staticmethod
    def head_params() -> int:
        return (
            FEATURE_CHANNELS * FAST_CHAN
            + FAST_CHAN
            + FAST_CHAN
            + 1
            + FEATURE_CHANNELS * AUXILIARY_CHANNELS
            + AUXILIARY_CHANNELS
        )

    def _freeze_core(self) -> None:
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        self.core.eval()
        if any(parameter.requires_grad for parameter in self.core.parameters()):
            raise V45FastError("released V45 core must be fully frozen")

    def _check_head(self) -> None:
        if self.head_params() != 1_254:
            raise V45FastError("fast-head parameter accounting drift")
        division_weight = self.aux_head.weight[4]
        division_bias = self.aux_head.bias[4]
        if not torch.equal(division_weight, torch.zeros_like(division_weight)) or not torch.equal(
            division_bias, torch.zeros_like(division_bias)
        ):
            raise V45FastError("division row must remain exactly zero before division promotion")

    def enforce(self) -> None:
        """Restore the disabled division row after each optimizer update."""

        with torch.no_grad():
            self.aux_head.weight[4].zero_()
            self.aux_head.bias[4].zero_()
        self._check_head()

    def train(self, mode: bool = True) -> "FrozenV45Fast":
        super().train(mode)
        self.core.eval()
        return self

    def accounting(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        core = sum(parameter.numel() for parameter in self.core.parameters())
        if trainable != self.head_params() or total != core + trainable:
            raise V45FastError("frozen-core fast-head accounting is invalid")
        return {"total": total, "frozen": core, "trainable": trainable}

    def fast_state(self) -> dict[str, torch.Tensor]:
        state = {
            "corr_hid.weight": self.corr_hid.weight.detach().cpu().contiguous().clone(),
            "corr_hid.bias": self.corr_hid.bias.detach().cpu().contiguous().clone(),
            "corr_out.weight": self.corr_out.weight.detach().cpu().contiguous().clone(),
            "corr_out.bias": self.corr_out.bias.detach().cpu().contiguous().clone(),
            "aux_head.weight": self.aux_head.weight.detach().cpu().contiguous().clone(),
            "aux_head.bias": self.aux_head.bias.detach().cpu().contiguous().clone(),
        }
        _check_state(state)
        return state

    @staticmethod
    def _in_shape(images: torch.Tensor) -> tuple[int, int, int, int]:
        if not isinstance(images, torch.Tensor) or images.ndim != 6:
            raise V45FastError("images must have shape (B, 2, 1, Z, Y, X)")
        batch, steps, channels, z, y, x = (int(value) for value in images.shape)
        if batch < 1 or steps != 2 or channels != 1 or min(z, y, x) < 1:
            raise V45FastError("images must have nonempty V45 two-frame shape")
        if not _finite(images):
            raise V45FastError("images contain nonfinite values")
        return batch, z, y, x

    @staticmethod
    def _encoded(value: Any, batch: int, shape: Sequence[int]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not isinstance(value, tuple) or len(value) != 2:
            raise V45FastError("V45 encode must return (features, logits)")
        features, logits = value
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != (
            batch,
            2,
            FEATURE_CHANNELS,
            *shape,
        ):
            raise V45FastError("V45 feature shape differs from the released contract")
        if not isinstance(logits, (tuple, list)) or len(logits) != 2:
            raise V45FastError("V45 encode must return two detector logits")
        expected = (batch, 1, *shape)
        if any(not isinstance(item, torch.Tensor) or tuple(item.shape) != expected for item in logits):
            raise V45FastError("V45 detector logit shape differs from the released contract")
        if not _finite(features) or any(not _finite(item) for item in logits):
            raise V45FastError("V45 encode produced nonfinite values")
        return features, list(logits)

    def teacher(self, images: torch.Tensor) -> torch.Tensor:
        """Return the exact sequential D4-8 teacher logits without core gradients."""

        batch, z, y, x = self._in_shape(images)
        with torch.no_grad():
            _features, logits = _encode_d4(
                torch,
                self.core,
                images[:, :, 0],
                view_batch_size=1,
                d4_views=D4_VIEWS,
            )
        expected = (batch, 2, 1, z, y, x)
        result = torch.stack(tuple(logits), dim=1)
        if tuple(result.shape) != expected or not _finite(result):
            raise V45FastError("D4 teacher logits violate the released contract")
        return result.detach()

    def encode_fast(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an unaugmented frame pair once and apply the compact head.

        The returned 32-channel maps are the frozen V45 maps used by bounded
        sparse association.  ``output`` contains centre, child-owned backward
        motion, log variance, and a permanently zero division channel.
        """

        batch, z, y, x = self._in_shape(images)
        with torch.no_grad():
            features, base_logits = self._encoded(
                self.core.encode(images[:, :, 0]),
                batch,
                (z, y, x),
            )
        flat = features.detach().reshape(batch * 2, FEATURE_CHANNELS, z, y, x)
        base = torch.stack(tuple(base_logits), dim=1).detach()
        correction = self.corr_out(F.gelu(self.corr_hid(flat))).reshape(batch, 2, 1, z, y, x)
        auxiliary = self.aux_head(flat).reshape(batch, 2, AUXILIARY_CHANNELS, z, y, x)
        output = torch.cat((base + correction, auxiliary), dim=2)
        if tuple(output.shape) != (batch, 2, FAST_OUT, z, y, x) or not _finite(output):
            raise V45FastError("fast V45 output violates its finite six-channel schema")
        self._check_head()
        return features.detach(), output

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _features, output = self.encode_fast(images)
        return output


def infer_fast(
    model: FrozenV45Fast,
    frames_value: Sequence[Any],
    *,
    q_low: float,
    q_high: float,
    start_frame_index: int = 0,
    node_id_start: int = 0,
    batch_size: int = 1,
    device: str | Any | None = None,
) -> FastVideo:
    """Infer canonical detections and bounded graph features in one core pass.

    This path deliberately does not call D4 TTA.  It is a candidate-only
    execution mode whose identity binds the compact head to the live frozen
    core.  The exact V45 inference and cache paths remain separate.
    """

    torch, _nn, _transformer, _unet = _require_torch_and_models()
    if not isinstance(model, FrozenV45Fast):
        raise V45FastError("fast inference requires FrozenV45Fast")
    if model.training:
        raise V45FastError("deterministic fast inference requires model.eval()")
    if not isinstance(frames_value, (tuple, list)) or len(frames_value) < 3:
        raise V45FastError("fast video inference requires at least three frames")
    if type(start_frame_index) is not int or start_frame_index < 0:
        raise V45FastError("start_frame_index must be a nonnegative integer")
    if type(node_id_start) is not int or node_id_start < 0:
        raise V45FastError("node_id_start must be a nonnegative integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise V45FastError("batch_size must be a positive integer")

    input_frames = [
        _frame_numpy(torch, frame, f"fast video frame {offset}")
        for offset, frame in enumerate(frames_value)
    ]
    input_shape = input_frames[0].shape
    if any(frame.shape != input_shape for frame in input_frames[1:]):
        raise V45FastError("all fast video frames must have equal spatial shapes")
    dz, dy, dx = V45_CONFIG.spatial_stride_zyx
    normalized = [
        quantile_normalize(frame[::dz, ::dy, ::dx], q_low, q_high)
        for frame in input_frames
    ]

    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise V45FastError("fast model has no parameter device/dtype") from exc
    target_device = parameter.device if device is None else torch.device(device)
    stride = np.asarray(V45_CONFIG.spatial_stride_zyx, dtype=np.int64)
    grid_shape = tuple(int(value) for value in normalized[0].shape)
    registry: dict[int, Any] = {}
    contexts: list[PairContext] = []
    next_node_id = node_id_start
    pair_count = len(normalized) - 1

    with torch.inference_mode():
        for pair_start in range(0, pair_count, batch_size):
            pair_end = min(pair_start + batch_size, pair_count)
            pair_arrays = [
                np.stack((normalized[index], normalized[index + 1]), axis=0)
                for index in range(pair_start, pair_end)
            ]
            pair_batch = torch.from_numpy(np.ascontiguousarray(np.stack(pair_arrays, axis=0))).to(
                device=target_device,
                dtype=parameter.dtype,
            )
            features, output = model.encode_fast(pair_batch[:, :, None])
            detected = [_detect_batch(torch, output[:, offset, :1]) for offset in range(2)]

            for local_index, pair_index in enumerate(range(pair_start, pair_end)):
                source_frame = start_frame_index + pair_index
                target_frame = source_frame + 1
                for offset, frame_index in enumerate((source_frame, target_frame)):
                    if frame_index in registry:
                        continue
                    grid_coords, scores = detected[offset][local_index]
                    grid_np = grid_coords.detach().cpu().numpy().astype(np.int64, copy=False)
                    coords_np = grid_np * stride[None, :]
                    canonical, _selected = dedup_dets(
                        frame_index,
                        coords_np,
                        scores.detach().cpu().numpy(),
                        grid_coords_zyx=grid_np,
                        grid_shape_zyx=grid_shape,
                        node_id_start=next_node_id,
                    )
                    registry[frame_index] = canonical
                    next_node_id += len(canonical.node_ids)

                source_det = registry[source_frame]
                target_det = registry[target_frame]
                contexts.append(
                    PairContext(
                        source_frame=source_frame,
                        target_frame=target_frame,
                        source_ids=source_det.node_ids,
                        target_ids=target_det.node_ids,
                        source_feat=_index_pair_map(torch, features[local_index, 0], source_det),
                        target_feat=_index_pair_map(torch, features[local_index, 1], target_det),
                        grid_shape_zyx=grid_shape,
                    )
                )

    video = VideoInference(
        detections=tuple(registry[start_frame_index + offset] for offset in range(len(normalized))),
        contexts=tuple(contexts),
        pair_batch_size=batch_size,
        view_batch_size=1,
        d4_views=(0,),
    )
    return FastVideo(
        video=video,
        core_state_sha256=v45_state_sha256(model.core),
        fast_state_sha256=_state_sha(model.fast_state()),
    )


def _check_state(state: Mapping[str, torch.Tensor]) -> None:
    expected = {
        "corr_hid.weight": (FAST_CHAN, FEATURE_CHANNELS, 1, 1, 1),
        "corr_hid.bias": (FAST_CHAN,),
        "corr_out.weight": (1, FAST_CHAN, 1, 1, 1),
        "corr_out.bias": (1,),
        "aux_head.weight": (AUXILIARY_CHANNELS, FEATURE_CHANNELS, 1, 1, 1),
        "aux_head.bias": (AUXILIARY_CHANNELS,),
    }
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise V45FastError("fast state tensor set is invalid")
    for name, shape in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shape or not _finite(value):
            raise V45FastError(f"fast state tensor {name!r} is invalid")
    if not torch.equal(state["aux_head.weight"][4], torch.zeros_like(state["aux_head.weight"][4])):
        raise V45FastError("fast state division weight must be exactly zero")
    if not torch.equal(state["aux_head.bias"][4], torch.zeros_like(state["aux_head.bias"][4])):
        raise V45FastError("fast state division bias must be exactly zero")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_fast(
    path: str | os.PathLike[str],
    model: FrozenV45Fast,
    bundle: ArtifactBundle,
    *,
    target_sha256: str,
    steps: int,
    train_id: Mapping[str, Any],
) -> tuple[str, FastIdent]:
    """Atomically persist only the compact one-pass head and immutable bindings."""

    if not isinstance(model, FrozenV45Fast) or not isinstance(bundle, ArtifactBundle):
        raise V45FastError("fast save requires FrozenV45Fast and a validated bundle")
    state = model.fast_state()
    ident = FastIdent(
        manifest_sha256=bundle.hashes.manifest_sha256,
        artifact_config_sha256=bundle.hashes.config_sha256,
        source_sha256=bundle.hashes.source_sha256,
        weight_sha256=bundle.hashes.weight_sha256,
        runtime_config_sha256=V45_CONFIG_SHA256,
        target_contract_sha256=_sha(target_sha256, "target_sha256"),
        core_state_sha256=v45_state_sha256(model.core),
        state_sha256=_state_sha(state),
        train_sha256=hashlib.sha256(_canon(dict(train_id))).hexdigest(),
        steps=steps,
        division_enabled=False,
    )
    payload = {"schema": FAST_SCHEMA, "identity": ident.as_dict(), "state": state}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return _file_sha(destination), ident


def load_fast(
    path: str | os.PathLike[str],
    model: FrozenV45Fast,
    bundle: ArtifactBundle,
    *,
    expected_sha256: str,
    target_sha256: str,
) -> FastIdent:
    """Restore a fast head only when every V45/target binding matches live state."""

    if not isinstance(model, FrozenV45Fast) or not isinstance(bundle, ArtifactBundle):
        raise V45FastError("fast load requires FrozenV45Fast and a validated bundle")
    source = Path(path)
    if not source.is_file() or _file_sha(source) != _sha(expected_sha256, "expected_sha256"):
        raise V45FastError("fast checkpoint file identity mismatch")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise V45FastError(f"weights-only fast checkpoint load failed: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"schema", "identity", "state"}:
        raise V45FastError("fast checkpoint top-level schema is invalid")
    if payload["schema"] != FAST_SCHEMA:
        raise V45FastError("fast checkpoint schema mismatch")
    ident = FastIdent.from_dict(payload["identity"])
    hashes = bundle.hashes
    if (
        ident.manifest_sha256,
        ident.artifact_config_sha256,
        ident.source_sha256,
        ident.weight_sha256,
    ) != (
        hashes.manifest_sha256,
        hashes.config_sha256,
        hashes.source_sha256,
        hashes.weight_sha256,
    ):
        raise V45FastError("fast checkpoint V45 artifact binding mismatch")
    if ident.runtime_config_sha256 != V45_CONFIG_SHA256:
        raise V45FastError("fast checkpoint runtime configuration mismatch")
    if ident.target_contract_sha256 != _sha(target_sha256, "target_sha256"):
        raise V45FastError("fast checkpoint target contract mismatch")
    if ident.core_state_sha256 != v45_state_sha256(model.core):
        raise V45FastError("fast checkpoint frozen-core binding mismatch")
    state = payload["state"]
    _check_state(state)
    if ident.state_sha256 != _state_sha(state):
        raise V45FastError("fast checkpoint state hash mismatch")
    try:
        model.corr_hid.load_state_dict(
            {"weight": state["corr_hid.weight"], "bias": state["corr_hid.bias"]},
            strict=True,
        )
        model.corr_out.load_state_dict(
            {"weight": state["corr_out.weight"], "bias": state["corr_out.bias"]},
            strict=True,
        )
        model.aux_head.load_state_dict(
            {"weight": state["aux_head.weight"], "bias": state["aux_head.bias"]},
            strict=True,
        )
    except RuntimeError as exc:
        raise V45FastError(f"fast state restore failed: {exc}") from exc
    model.enforce()
    return ident


def fast_readout(
    model: FrozenV45Fast,
    bundle: ArtifactBundle,
    ident: FastIdent,
    *,
    checkpoint_sha256: str,
) -> V45AuxReadout:
    """Expose the loaded compact motion head as a strictly bound sparse readout.

    The compact checkpoint stores the detector correction and auxiliary head in
    one artifact.  Association only needs the latter at already-detected V45
    feature rows, so this bridge neither re-runs the core nor materializes an
    auxiliary volume.  It fails closed if the live head differs from the exact
    checkpoint identity used for detection.
    """

    if not isinstance(model, FrozenV45Fast) or not isinstance(bundle, ArtifactBundle):
        raise V45FastError("fast readout requires a loaded fast model and validated bundle")
    if not isinstance(ident, FastIdent):
        raise V45FastError("fast readout requires a FastIdent")
    checkpoint = _sha(checkpoint_sha256, "checkpoint_sha256")
    hashes = bundle.hashes
    if (
        ident.manifest_sha256,
        ident.artifact_config_sha256,
        ident.source_sha256,
        ident.weight_sha256,
    ) != (
        hashes.manifest_sha256,
        hashes.config_sha256,
        hashes.source_sha256,
        hashes.weight_sha256,
    ):
        raise V45FastError("fast readout V45 artifact binding mismatch")
    if ident.runtime_config_sha256 != V45_CONFIG_SHA256:
        raise V45FastError("fast readout runtime configuration mismatch")
    if ident.core_state_sha256 != v45_state_sha256(model.core):
        raise V45FastError("fast readout frozen-core binding mismatch")
    if _state_sha(model.fast_state()) != ident.state_sha256:
        raise V45FastError("fast readout compact-head state binding mismatch")
    if ident.division_enabled:
        raise V45FastError("fast readout cannot enable division before promotion")
    state = {
        "weight": model.aux_head.weight.detach().cpu().contiguous().clone(),
        "bias": model.aux_head.bias.detach().cpu().contiguous().clone(),
    }
    identity = V45AuxCheckpointIdentity(
        manifest_sha256=ident.manifest_sha256,
        artifact_config_sha256=ident.artifact_config_sha256,
        source_sha256=ident.source_sha256,
        weight_sha256=ident.weight_sha256,
        runtime_config_sha256=ident.runtime_config_sha256,
        target_contract_sha256=ident.target_contract_sha256,
        core_state_sha256=ident.core_state_sha256,
        auxiliary_state_sha256=auxiliary_state_sha256(state),
        training_identity_sha256=ident.train_sha256,
        completed_steps=ident.steps,
        division_enabled=False,
    )
    return V45AuxReadout(identity, state, checkpoint_sha256=checkpoint)


__all__ = [
    "FAST_CHAN",
    "FAST_INFER_SCHEMA",
    "FAST_OUT",
    "FAST_SCHEMA",
    "FastIdent",
    "FastVideo",
    "FrozenV45Fast",
    "V45FastError",
    "fast_readout",
    "infer_fast",
    "load_fast",
    "save_fast",
]
