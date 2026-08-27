"""A compact, frozen-core auxiliary head for the released V45 detector.

The released model's ``encode`` method yields two contextual 32-channel
feature volumes and the immutable detector logits.  This module deliberately
uses that interface rather than the released dense transformer: it adds only
a pointwise 32-to-5 readout for sparse motion, uncertainty, and division
evidence.  The detector centre channel is never trainable through this path.
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

import torch
import torch.nn as nn

from .bh_model import ArtifactBundle, V45_CONFIG_SHA256


FEATURE_CHANNELS = 32
AUXILIARY_CHANNELS = 5
AUXILIARY_CHECKPOINT_SCHEMA = "celltrack-alpha.v45-auxiliary-checkpoint.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
OUTPUT_SCHEMA = (
    "center_logit",
    "backward_dz_um",
    "backward_dy_um",
    "backward_dx_um",
    "log_variance",
    "division_logit",
)


class V45HeadError(ValueError):
    """Raised when a purported V45 feature core violates the head contract."""


class V45AuxiliaryCheckpointError(V45HeadError):
    """Raised when a compact auxiliary checkpoint is not exactly compatible."""


def _finite(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise V45AuxiliaryCheckpointError(f"{label} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_state_sha256(state: Mapping[str, torch.Tensor], *, label: str) -> str:
    if not isinstance(state, Mapping) or not state:
        raise V45AuxiliaryCheckpointError(f"{label} state must be a nonempty tensor mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise V45AuxiliaryCheckpointError(f"{label} state contains an invalid entry")
        detached = value.detach().cpu().contiguous()
        if not _finite(detached):
            raise V45AuxiliaryCheckpointError(f"{label} tensor {name!r} contains nonfinite values")
        digest.update(
            _canonical_json({"name": name, "dtype": str(detached.dtype), "shape": list(detached.shape)})
        )
        digest.update(b"\x00")
        digest.update(detached.numpy().tobytes(order="C"))
        digest.update(b"\x00")
    return digest.hexdigest()


def v45_state_sha256(module: nn.Module) -> str:
    """Hash every frozen V45 tensor, including inference-affecting buffers."""

    if not isinstance(module, nn.Module):
        raise V45AuxiliaryCheckpointError("V45 state hashing requires an nn.Module")
    return _tensor_state_sha256(module.state_dict(), label="V45 core")


@dataclass(frozen=True)
class V45AuxCheckpointIdentity:
    """Strict identity for an auxiliary-only state bound to one V45 artifact."""

    manifest_sha256: str
    artifact_config_sha256: str
    source_sha256: str
    weight_sha256: str
    runtime_config_sha256: str
    target_contract_sha256: str
    core_state_sha256: str
    auxiliary_state_sha256: str
    training_identity_sha256: str
    completed_steps: int
    division_enabled: bool
    schema: str = AUXILIARY_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != AUXILIARY_CHECKPOINT_SCHEMA:
            raise V45AuxiliaryCheckpointError("auxiliary checkpoint schema is unsupported")
        for name in (
            "manifest_sha256",
            "artifact_config_sha256",
            "source_sha256",
            "weight_sha256",
            "runtime_config_sha256",
            "target_contract_sha256",
            "core_state_sha256",
            "auxiliary_state_sha256",
            "training_identity_sha256",
        ):
            _sha256(getattr(self, name), name)
        if type(self.completed_steps) is not int or self.completed_steps < 0:
            raise V45AuxiliaryCheckpointError("completed_steps must be a nonnegative integer")
        if self.division_enabled is not False:
            raise V45AuxiliaryCheckpointError("division must remain disabled for V45 auxiliary checkpoints")

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
            "auxiliary_state_sha256": self.auxiliary_state_sha256,
            "training_identity_sha256": self.training_identity_sha256,
            "completed_steps": self.completed_steps,
            "division_enabled": self.division_enabled,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "V45AuxCheckpointIdentity":
        if not isinstance(value, Mapping):
            raise V45AuxiliaryCheckpointError("checkpoint identity must be a mapping")
        expected = {
            "schema",
            "manifest_sha256",
            "artifact_config_sha256",
            "source_sha256",
            "weight_sha256",
            "runtime_config_sha256",
            "target_contract_sha256",
            "core_state_sha256",
            "auxiliary_state_sha256",
            "training_identity_sha256",
            "completed_steps",
            "division_enabled",
        }
        if set(value) != expected:
            raise V45AuxiliaryCheckpointError("checkpoint identity has missing or unexpected fields")
        return cls(**dict(value))


class V45AuxReadout(nn.Module):
    """Frozen sparse 32-to-5 readout restored from one strict head artifact.

    This deliberately contains only the small pointwise weights.  Production
    association can evaluate detections already sampled from V45 feature maps
    without registering a second V45 core or materializing an auxiliary volume.
    """

    def __init__(
        self,
        identity: V45AuxCheckpointIdentity,
        state: Mapping[str, torch.Tensor],
        *,
        checkpoint_sha256: str,
    ) -> None:
        super().__init__()
        if not isinstance(identity, V45AuxCheckpointIdentity):
            raise V45AuxiliaryCheckpointError("readout requires a V45 auxiliary identity")
        _validate_auxiliary_state(state)
        if identity.auxiliary_state_sha256 != _tensor_state_sha256(state, label="auxiliary"):
            raise V45AuxiliaryCheckpointError("readout auxiliary-state binding mismatch")
        self.identity = identity
        self.checkpoint_sha256 = _sha256(checkpoint_sha256, "checkpoint_sha256")
        self.register_buffer("weight", state["weight"].detach().cpu().contiguous().clone())
        self.register_buffer("bias", state["bias"].detach().cpu().contiguous().clone())
        self.eval()

    def train(self, mode: bool = True) -> "V45AuxReadout":
        # This object is an immutable inference readout, not another trainable head.
        super().train(False)
        return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != FEATURE_CHANNELS:
            raise V45AuxiliaryCheckpointError("point features must have shape (N, 32)")
        if not _finite(features):
            raise V45AuxiliaryCheckpointError("point features contain nonfinite values")
        weight = self.weight.reshape(AUXILIARY_CHANNELS, FEATURE_CHANNELS)
        result = features.to(device=weight.device, dtype=weight.dtype) @ weight.T + self.bias
        if tuple(result.shape) != (features.shape[0], AUXILIARY_CHANNELS) or not _finite(result):
            raise V45AuxiliaryCheckpointError("sparse auxiliary readout is invalid")
        return result


class FrozenV45Head(nn.Module):
    """Five zero-initialized pointwise outputs on frozen V45 features.

    ``core`` must be the strict, released V45 model returned by
    :func:`celltrack_alpha.bh_model.load_v45_model`.  Only its ``encode``
    method is called; its dense ``forward`` path remains fail-closed.
    """

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        if not isinstance(core, nn.Module) or not callable(getattr(core, "encode", None)):
            raise V45HeadError("core must be an nn.Module exposing encode(images)")
        self.core = core
        self.aux_head = nn.Conv3d(FEATURE_CHANNELS, AUXILIARY_CHANNELS, kernel_size=1)
        nn.init.zeros_(self.aux_head.weight)
        nn.init.zeros_(self.aux_head.bias)
        self._freeze_core()

    @staticmethod
    def auxiliary_parameter_count() -> int:
        return FEATURE_CHANNELS * AUXILIARY_CHANNELS + AUXILIARY_CHANNELS

    def _freeze_core(self) -> None:
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        self.core.eval()
        if any(parameter.requires_grad for parameter in self.core.parameters()):
            raise V45HeadError("released V45 core must be fully frozen")

    def train(self, mode: bool = True) -> "FrozenV45Head":
        """Keep V45 in inference mode even while the small head is optimized."""

        super().train(mode)
        self.core.eval()
        return self

    def parameter_accounting(self) -> dict[str, int]:
        core_total = sum(parameter.numel() for parameter in self.core.parameters())
        head_total = sum(parameter.numel() for parameter in self.aux_head.parameters())
        core_trainable = sum(parameter.numel() for parameter in self.core.parameters() if parameter.requires_grad)
        head_trainable = sum(parameter.numel() for parameter in self.aux_head.parameters() if parameter.requires_grad)
        if head_total != self.auxiliary_parameter_count() or core_trainable != 0 or head_trainable != head_total:
            raise V45HeadError("frozen-core parameter partition is invalid")
        return {
            "total": core_total + head_total,
            "trainable": head_trainable,
            "frozen": core_total,
            "core_total": core_total,
            "core_trainable": core_trainable,
            "aux_head_total": head_total,
            "aux_head_trainable": head_trainable,
        }

    def auxiliary_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the exact, CPU-only state allowed in a compact artifact."""

        state = {name: value.detach().cpu().contiguous().clone() for name, value in self.aux_head.state_dict().items()}
        _validate_auxiliary_state(state)
        return state

    def auxiliary_at_points(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the 1x1 auxiliary readout at sparse V45 feature rows only."""

        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != FEATURE_CHANNELS:
            raise V45HeadError("point features must have shape (N, 32)")
        if not _finite(features):
            raise V45HeadError("point features contain nonfinite values")
        weight = self.aux_head.weight.reshape(AUXILIARY_CHANNELS, FEATURE_CHANNELS)
        result = features.to(device=weight.device, dtype=weight.dtype) @ weight.T + self.aux_head.bias
        if tuple(result.shape) != (features.shape[0], AUXILIARY_CHANNELS) or not _finite(result):
            raise V45HeadError("pointwise auxiliary readout is invalid")
        return result

    @staticmethod
    def _validate_images(images: torch.Tensor) -> tuple[int, int, int, int, int]:
        if not isinstance(images, torch.Tensor) or images.ndim != 6:
            raise V45HeadError("images must have shape (B, 2, 1, Z, Y, X)")
        batch, frames, channels, z, y, x = (int(value) for value in images.shape)
        if frames != 2 or channels != 1 or min(batch, z, y, x) < 1:
            raise V45HeadError("images must have nonempty V45 two-frame shape (B, 2, 1, Z, Y, X)")
        if not _finite(images):
            raise V45HeadError("images contain nonfinite values")
        return batch, z, y, x, frames

    @staticmethod
    def _validate_encoded(
        value: Any,
        *,
        batch: int,
        spatial_shape: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(value, tuple) or len(value) != 2:
            raise V45HeadError("V45 encode must return (features, logits)")
        features, logits = value
        expected_features = (batch, 2, FEATURE_CHANNELS, *spatial_shape)
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected_features:
            raise V45HeadError("V45 encode feature shape differs from the released contract")
        if not _finite(features):
            raise V45HeadError("V45 encode features contain nonfinite values")
        if not isinstance(logits, (list, tuple)) or len(logits) != 2:
            raise V45HeadError("V45 encode must return two detector logits")
        expected_logit = (batch, 1, *spatial_shape)
        if any(not isinstance(logit, torch.Tensor) or tuple(logit.shape) != expected_logit for logit in logits):
            raise V45HeadError("V45 detector logit shape differs from the released contract")
        if any(not _finite(logit) for logit in logits):
            raise V45HeadError("V45 detector logits contain nonfinite values")
        centres = torch.stack(tuple(logits), dim=1).detach()
        return features.detach(), centres

    def forward(self, images: torch.Tensor, *, scenario: str = "universal") -> torch.Tensor:
        """Return immutable centres plus trainable auxiliary voxel maps.

        ``scenario`` is accepted only to keep the trainer interface explicit;
        specialist routing remains deferred until leakage-safe OOF evidence
        exists.
        """

        if scenario != "universal":
            raise V45HeadError("V45 auxiliary head permits only the universal scenario")
        batch, z, y, x, _ = self._validate_images(images)
        with torch.no_grad():
            encoded = self.core.encode(images[:, :, 0])
        features, centres = self._validate_encoded(encoded, batch=batch, spatial_shape=(z, y, x))
        flat = features.reshape(batch * 2, FEATURE_CHANNELS, z, y, x)
        auxiliary = self.aux_head(flat).reshape(batch, 2, AUXILIARY_CHANNELS, z, y, x)
        output = torch.cat((centres, auxiliary), dim=2)
        expected = (batch, 2, len(OUTPUT_SCHEMA), z, y, x)
        if tuple(output.shape) != expected or not _finite(output):
            raise V45HeadError("V45 auxiliary output violates its finite six-channel schema")
        return output


def _validate_auxiliary_state(state: Mapping[str, torch.Tensor]) -> None:
    if not isinstance(state, Mapping) or set(state) != {"weight", "bias"}:
        raise V45AuxiliaryCheckpointError("auxiliary state must contain exactly weight and bias")
    weight, bias = state["weight"], state["bias"]
    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(bias, torch.Tensor)
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or tuple(weight.shape) != (AUXILIARY_CHANNELS, FEATURE_CHANNELS, 1, 1, 1)
        or tuple(bias.shape) != (AUXILIARY_CHANNELS,)
        or not _finite(weight)
        or not _finite(bias)
    ):
        raise V45AuxiliaryCheckpointError("auxiliary state has an invalid tensor schema")
    if not torch.equal(weight[4], torch.zeros_like(weight[4])) or not torch.equal(bias[4], torch.zeros_like(bias[4])):
        raise V45AuxiliaryCheckpointError("division row must remain exactly zero while division is disabled")


def auxiliary_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Return the strict content identity of a division-disabled sparse readout."""

    _validate_auxiliary_state(state)
    return _tensor_state_sha256(state, label="auxiliary")


def save_v45_auxiliary_checkpoint(
    path: str | os.PathLike[str],
    head: FrozenV45Head,
    bundle: ArtifactBundle,
    *,
    target_contract_sha256: str,
    completed_steps: int,
    training_identity: Mapping[str, Any],
) -> tuple[str, V45AuxCheckpointIdentity]:
    """Atomically write only the compact head and its immutable V45 bindings."""

    if not isinstance(head, FrozenV45Head) or not isinstance(bundle, ArtifactBundle):
        raise V45AuxiliaryCheckpointError("save requires FrozenV45Head and validated ArtifactBundle")
    training_hash = hashlib.sha256(_canonical_json(dict(training_identity))).hexdigest()
    state = head.auxiliary_state_dict()
    identity = V45AuxCheckpointIdentity(
        manifest_sha256=bundle.hashes.manifest_sha256,
        artifact_config_sha256=bundle.hashes.config_sha256,
        source_sha256=bundle.hashes.source_sha256,
        weight_sha256=bundle.hashes.weight_sha256,
        runtime_config_sha256=V45_CONFIG_SHA256,
        target_contract_sha256=_sha256(target_contract_sha256, "target_contract_sha256"),
        core_state_sha256=v45_state_sha256(head.core),
        auxiliary_state_sha256=auxiliary_state_sha256(state),
        training_identity_sha256=training_hash,
        completed_steps=completed_steps,
        division_enabled=False,
    )
    payload = {"schema": AUXILIARY_CHECKPOINT_SCHEMA, "identity": identity.as_dict(), "auxiliary_state": state}
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
    return _file_sha256(destination), identity


def _load_auxiliary_payload(
    path: str | os.PathLike[str],
    *,
    expected_checkpoint_sha256: str,
) -> tuple[V45AuxCheckpointIdentity, Mapping[str, torch.Tensor]]:
    source = Path(path)
    checkpoint_sha256 = _sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    if not source.is_file() or _file_sha256(source) != checkpoint_sha256:
        raise V45AuxiliaryCheckpointError("auxiliary checkpoint file identity mismatch")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise V45AuxiliaryCheckpointError(f"weights-only auxiliary checkpoint load failed: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"schema", "identity", "auxiliary_state"}:
        raise V45AuxiliaryCheckpointError("auxiliary checkpoint has an invalid top-level schema")
    if payload["schema"] != AUXILIARY_CHECKPOINT_SCHEMA:
        raise V45AuxiliaryCheckpointError("auxiliary checkpoint schema mismatch")
    identity = V45AuxCheckpointIdentity.from_dict(payload["identity"])
    state = payload["auxiliary_state"]
    _validate_auxiliary_state(state)
    if identity.auxiliary_state_sha256 != auxiliary_state_sha256(state):
        raise V45AuxiliaryCheckpointError("auxiliary checkpoint state hash mismatch")
    return identity, state


def _verify_auxiliary_binding(
    identity: V45AuxCheckpointIdentity,
    bundle: ArtifactBundle,
    *,
    core_state_sha256: str,
    target_contract_sha256: str,
) -> None:
    expected_hashes = bundle.hashes
    observed_hashes = (
        identity.manifest_sha256,
        identity.artifact_config_sha256,
        identity.source_sha256,
        identity.weight_sha256,
    )
    if observed_hashes != (
        expected_hashes.manifest_sha256,
        expected_hashes.config_sha256,
        expected_hashes.source_sha256,
        expected_hashes.weight_sha256,
    ):
        raise V45AuxiliaryCheckpointError("auxiliary checkpoint V45 artifact binding mismatch")
    if identity.runtime_config_sha256 != V45_CONFIG_SHA256:
        raise V45AuxiliaryCheckpointError("auxiliary checkpoint runtime configuration mismatch")
    if identity.target_contract_sha256 != _sha256(target_contract_sha256, "target_contract_sha256"):
        raise V45AuxiliaryCheckpointError("auxiliary checkpoint target-contract mismatch")
    if identity.core_state_sha256 != _sha256(core_state_sha256, "core_state_sha256"):
        raise V45AuxiliaryCheckpointError("auxiliary checkpoint frozen-core binding mismatch")


def load_v45_auxiliary_readout(
    path: str | os.PathLike[str],
    bundle: ArtifactBundle,
    *,
    expected_checkpoint_sha256: str,
    target_contract_sha256: str,
    core_state_sha256: str,
) -> V45AuxReadout:
    """Restore only the validated sparse readout for candidate-edge scoring."""

    if not isinstance(bundle, ArtifactBundle):
        raise V45AuxiliaryCheckpointError("readout load requires a validated ArtifactBundle")
    checkpoint_sha256 = _sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    identity, state = _load_auxiliary_payload(
        path,
        expected_checkpoint_sha256=checkpoint_sha256,
    )
    _verify_auxiliary_binding(
        identity,
        bundle,
        core_state_sha256=core_state_sha256,
        target_contract_sha256=target_contract_sha256,
    )
    return V45AuxReadout(identity, state, checkpoint_sha256=checkpoint_sha256)


def load_v45_auxiliary_checkpoint(
    path: str | os.PathLike[str],
    head: FrozenV45Head,
    bundle: ArtifactBundle,
    *,
    expected_checkpoint_sha256: str,
    target_contract_sha256: str,
) -> V45AuxCheckpointIdentity:
    """Fail closed unless a compact checkpoint binds exactly to the live V45 core."""

    if not isinstance(head, FrozenV45Head) or not isinstance(bundle, ArtifactBundle):
        raise V45AuxiliaryCheckpointError("load requires FrozenV45Head and validated ArtifactBundle")
    identity, state = _load_auxiliary_payload(
        path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    _verify_auxiliary_binding(
        identity,
        bundle,
        core_state_sha256=v45_state_sha256(head.core),
        target_contract_sha256=target_contract_sha256,
    )
    try:
        head.aux_head.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise V45AuxiliaryCheckpointError(f"auxiliary state restore failed: {exc}") from exc
    return identity


__all__ = [
    "AUXILIARY_CHECKPOINT_SCHEMA",
    "AUXILIARY_CHANNELS",
    "FEATURE_CHANNELS",
    "V45AuxReadout",
    "FrozenV45Head",
    "OUTPUT_SCHEMA",
    "V45AuxCheckpointIdentity",
    "V45AuxiliaryCheckpointError",
    "V45HeadError",
    "auxiliary_state_sha256",
    "load_v45_auxiliary_checkpoint",
    "load_v45_auxiliary_readout",
    "save_v45_auxiliary_checkpoint",
    "v45_state_sha256",
]
