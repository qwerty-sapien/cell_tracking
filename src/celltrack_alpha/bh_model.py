"""Frozen V45 BioHub model inference and deterministic sparse cache support.

The numerical helpers and cache codec in this module only require NumPy.  Torch
and the public ``biohub_tracking`` model sources are imported lazily by model
construction and inference entry points.  This keeps artifact and cache tests
CPU-only while making a missing model dependency a hard, explicit failure.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import hmac
import json
import math
import mmap
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np


MODEL_SCHEMA = "celltrack-alpha.bh-v45-model.v1"
CACHE_SCHEMA = "celltrack-alpha.bh-v45-cache.v3"
CACHE_MAGIC = b"CTBHMV3\x00"

DEFAULT_WEIGHT_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
DEFAULT_CONFIG_SHA256 = "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a"
DEFAULT_MANIFEST_SHA256 = "bc20f1f04cfb682af3b27a836ce9a57f44a2fe27508dc59039200b2a23188103"
DEFAULT_SOURCE_SHA256 = "8c4621658956f724b161236b451d7d0ed30759aa3f3a5ea28021b4eeb1171e9b"

DEFAULT_SUPPORT_ROOT = Path(__file__).resolve().parents[2] / "external" / "Tracking_Support"
DEFAULT_SOURCE_PATHS = (
    "repo/src/biohub_tracking/models/temporal_unet.py",
    "repo/src/biohub_tracking/models/simple_node_transformer.py",
    "repo/scripts/train_unet_transformer.py",
)
DEFAULT_WEIGHT_RELATIVE_PATH = "weights/unet_transformer/split_0/edge_predictor_best.pth"
DEFAULT_CONFIG_RELATIVE_PATH = "weights/unet_transformer/split_0/config.json"

_ARTIFACT_MODEL_CONFIG = {
    "downsample": [1, 4, 4],
    "pool_kernel_um": 5.0,
    "unet_layers": [32, 64, 128],
    "unet_out_channels": 32,
    "window_size": 2,
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class BHModelError(RuntimeError):
    """Base exception for the frozen model/cache layer."""


class ArtifactValidationError(BHModelError):
    """An artifact bundle is not the exact expected public model artifact."""


class DependencyUnavailableError(BHModelError):
    """A required inference dependency is missing."""


class ModelValidationError(BHModelError):
    """Model input, output, or state failed a fail-closed validation."""


class CacheValidationError(BHModelError):
    """A sparse cache is malformed, tampered, or belongs to another model."""


D4_VIEWS = tuple(range(8))


def _normalize_d4_views(d4_views: Sequence[int] | None = None) -> tuple[int, ...]:
    """Return a canonical diagnostic D4 subset.

    The locked V45 path always uses all eight views.  A subset is deliberately
    restricted to include view zero because pair-context embeddings are drawn
    from the unaugmented encoder output.  Canonical ordering keeps the
    sequential accumulation order deterministic across equivalent callers.
    """

    if d4_views is None:
        return D4_VIEWS
    if isinstance(d4_views, (str, bytes)) or not isinstance(d4_views, Sequence):
        raise ModelValidationError("D4 views must be an ordered sequence of integer view IDs")
    views = tuple(d4_views)
    if not views:
        raise ModelValidationError("D4 views must contain at least the unaugmented view 0")
    if any(type(view) is not int or view not in D4_VIEWS for view in views):
        raise ModelValidationError("D4 views must be unique integers in [0, 7]")
    if len(set(views)) != len(views):
        raise ModelValidationError("D4 views must not contain duplicates")
    if 0 not in views:
        raise ModelValidationError("D4 diagnostic subsets must include unaugmented view 0")
    return tuple(sorted(views))


@dataclass(frozen=True)
class V45Config:
    """Locked V45 inference configuration.

    The artifact's ``pool_kernel_um=5`` is a training/checkpoint setting.  V45
    deliberately locks inference peak suppression to 3 um and to a 3x3x3
    kernel on the downsampled volume.
    """

    schema: str = MODEL_SCHEMA
    architecture: str = "TemporalUNet3D"
    unet_layers: tuple[int, int, int] = (32, 64, 128)
    unet_out_channels: int = 32
    window_size: int = 2
    spatial_stride_zyx: tuple[int, int, int] = (1, 4, 4)
    detection_threshold: float = 0.970
    detection_tta: str = "D4-8-view"
    edge_feature_tta: bool = False
    pool_kernel_um: float = 3.0
    pool_kernel_downsampled_zyx: tuple[int, int, int] = (3, 3, 3)
    quantile_low: float = 0.001
    quantile_high: float = 0.999
    pair_batch_size: int = 1
    view_batch_size: int = 1

    def __post_init__(self) -> None:
        locked = {
            "schema": MODEL_SCHEMA,
            "architecture": "TemporalUNet3D",
            "unet_layers": (32, 64, 128),
            "unet_out_channels": 32,
            "window_size": 2,
            "spatial_stride_zyx": (1, 4, 4),
            "detection_threshold": 0.970,
            "detection_tta": "D4-8-view",
            "edge_feature_tta": False,
            "pool_kernel_um": 3.0,
            "pool_kernel_downsampled_zyx": (3, 3, 3),
            "quantile_low": 0.001,
            "quantile_high": 0.999,
            "pair_batch_size": 1,
            "view_batch_size": 1,
        }
        for name, wanted in locked.items():
            if getattr(self, name) != wanted:
                raise ValueError(f"V45 field {name!r} is locked to {wanted!r}")

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            result[item.name] = list(value) if isinstance(value, tuple) else value
        return result


V45_CONFIG = V45Config()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of deterministic canonical JSON."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def v45_config_sha(
    *,
    pair_batch_size: int = 1,
    view_batch_size: int = 1,
    d4_views: Sequence[int] | None = None,
) -> str:
    """Hash the locked model configuration and a diagnostic execution variant.

    The all-eight-view default is intentionally byte-for-byte compatible with
    the frozen V45 cache identity.  Any reduced D4 set is diagnostic-only and
    receives an unambiguous, non-production config hash.
    """

    if type(pair_batch_size) is not int or pair_batch_size <= 0:
        raise ValueError("pair_batch_size must be a positive integer")
    if type(view_batch_size) is not int or not 1 <= view_batch_size <= 8:
        raise ValueError("view_batch_size must be an integer in [1, 8]")
    try:
        views = _normalize_d4_views(d4_views)
    except ModelValidationError as exc:
        raise ValueError(f"invalid D4 view set: {exc}") from exc
    if view_batch_size > len(views):
        raise ValueError("view_batch_size must not exceed the selected D4 view count")
    config = V45_CONFIG.as_dict()
    config["pair_batch_size"] = pair_batch_size
    config["view_batch_size"] = view_batch_size
    if views != D4_VIEWS:
        config["experimental_d4_views"] = list(views)
    return canonical_sha256(config)


V45_CONFIG_SHA256 = v45_config_sha()


def _validate_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash one regular file without loading it in memory."""

    source = Path(path)
    if not source.is_file():
        raise ArtifactValidationError(f"required artifact file is missing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(
    support_root: str | os.PathLike[str] = DEFAULT_SUPPORT_ROOT,
    relative_paths: Sequence[str] = DEFAULT_SOURCE_PATHS,
) -> str:
    """Hash the ordered authoritative architecture sources and their names."""

    root = Path(support_root)
    digest = hashlib.sha256()
    if not isinstance(relative_paths, (tuple, list)) or not relative_paths:
        raise ArtifactValidationError("source path list must be a non-empty tuple or list")
    for relative in relative_paths:
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ArtifactValidationError(f"unsafe source path: {relative!r}")
        path = root / relative
        if not path.is_file():
            raise ArtifactValidationError(f"authoritative model source is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_strict_json(path: Path, *, error_type: type[BHModelError]) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw, object_pairs_hook=_reject_duplicate_json_pairs)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise error_type(f"invalid JSON in {path}: {exc}") from exc


@dataclass(frozen=True)
class ArtifactHashes:
    manifest_sha256: str
    config_sha256: str
    source_sha256: str
    weight_sha256: str

    def __post_init__(self) -> None:
        for item in fields(self):
            _validate_sha256(getattr(self, item.name), item.name)


@dataclass(frozen=True)
class ArtifactBundle:
    support_root: Path
    manifest_path: Path
    config_path: Path
    weight_path: Path
    hashes: ArtifactHashes


def validate_artifact_bundle(
    support_root: str | os.PathLike[str] = DEFAULT_SUPPORT_ROOT,
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
    weight_path: str | os.PathLike[str] | None = None,
    expected_manifest_sha256: str = DEFAULT_MANIFEST_SHA256,
    expected_config_sha256: str = DEFAULT_CONFIG_SHA256,
    expected_source_sha256: str = DEFAULT_SOURCE_SHA256,
    expected_weight_sha256: str = DEFAULT_WEIGHT_SHA256,
) -> ArtifactBundle:
    """Validate the manifest, exact config, authoritative sources, and weights.

    Every expected hash is explicit and overrideable for a separately pinned
    mirror.  The default weight pin is the local public support artifact.
    """

    root = Path(support_root)
    manifest = Path(manifest_path) if manifest_path is not None else root / "ARTIFACT_MANIFEST.json"
    config = Path(config_path) if config_path is not None else root / DEFAULT_CONFIG_RELATIVE_PATH
    weight = Path(weight_path) if weight_path is not None else root / DEFAULT_WEIGHT_RELATIVE_PATH

    expected = ArtifactHashes(
        manifest_sha256=_validate_sha256(expected_manifest_sha256, "expected_manifest_sha256"),
        config_sha256=_validate_sha256(expected_config_sha256, "expected_config_sha256"),
        source_sha256=_validate_sha256(expected_source_sha256, "expected_source_sha256"),
        weight_sha256=_validate_sha256(expected_weight_sha256, "expected_weight_sha256"),
    )
    actual = ArtifactHashes(
        manifest_sha256=sha256_file(manifest),
        config_sha256=sha256_file(config),
        source_sha256=source_tree_sha256(root),
        weight_sha256=sha256_file(weight),
    )
    for item in fields(ArtifactHashes):
        name = item.name
        if getattr(actual, name) != getattr(expected, name):
            raise ArtifactValidationError(
                f"{name} mismatch: expected {getattr(expected, name)}, got {getattr(actual, name)}"
            )

    manifest_data = _load_strict_json(manifest, error_type=ArtifactValidationError)
    config_data = _load_strict_json(config, error_type=ArtifactValidationError)
    if not isinstance(manifest_data, dict) or not isinstance(config_data, dict):
        raise ArtifactValidationError("artifact manifest and config must be JSON objects")
    if config_data != _ARTIFACT_MODEL_CONFIG:
        raise ArtifactValidationError("artifact config is not the exact locked public checkpoint config")

    try:
        top_model = manifest_data["model"]
        named_model = manifest_data["models"]["unet_transformer"]
    except (KeyError, TypeError) as exc:
        raise ArtifactValidationError("manifest is missing the unet_transformer model entry") from exc
    if top_model != named_model or not isinstance(named_model, dict):
        raise ArtifactValidationError("manifest model entries disagree")
    required_model_keys = {"config", "method", "weight_bytes", "weight_path", "weight_sha256"}
    if set(named_model) != required_model_keys:
        raise ArtifactValidationError("manifest model entry has missing or unexpected fields")
    if named_model["config"] != config_data:
        raise ArtifactValidationError("manifest config does not match config.json")
    if named_model["method"] != "unet_transformer":
        raise ArtifactValidationError("manifest method is not unet_transformer")
    if named_model["weight_path"] != DEFAULT_WEIGHT_RELATIVE_PATH:
        raise ArtifactValidationError("manifest weight path is not the locked public path")
    if type(named_model["weight_bytes"]) is not int or named_model["weight_bytes"] != weight.stat().st_size:
        raise ArtifactValidationError("manifest weight byte count mismatch")
    if named_model["weight_sha256"] != actual.weight_sha256:
        raise ArtifactValidationError("manifest weight SHA does not match weight bytes")

    return ArtifactBundle(root, manifest, config, weight, actual)


def _require_torch_and_models() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from biohub_tracking.models import SimpleNodeTransformer, TemporalUNet3D
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyUnavailableError(
            "V45 inference requires torch and the authoritative biohub_tracking model sources"
        ) from exc
    return torch, nn, SimpleNodeTransformer, TemporalUNet3D


def build_v45_model() -> Any:
    """Construct the exact public architecture without loading or training it."""

    torch, nn, SimpleNodeTransformer, TemporalUNet3D = _require_torch_and_models()

    class DiagnosticTransformer(SimpleNodeTransformer):
        def forward(
            self,
            *args: Any,
            diagnostic_allow_dense: bool = False,
            **kwargs: Any,
        ) -> Any:
            if diagnostic_allow_dense is not True:
                raise ModelValidationError(
                    "dense transformer forward is diagnostic-only; "
                    "set diagnostic_allow_dense=True explicitly"
                )
            return super().forward(*args, **kwargs)

    class V45UNetNodeTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unet = TemporalUNet3D(
                in_channels=1,
                out_channels=V45_CONFIG.unet_out_channels,
                layers=V45_CONFIG.unet_layers,
            )
            self.unet_out_channels = V45_CONFIG.unet_out_channels
            self.detect_head = nn.Conv3d(V45_CONFIG.unet_out_channels, 1, kernel_size=1)
            self.transformer = DiagnosticTransformer(
                feat_dim=64,
                hidden_dim=128,
                n_heads=4,
                n_blocks=4,
                dropout=0.3,
            )

        def encode(self, images: Any) -> tuple[Any, list[Any]]:
            if not isinstance(images, torch.Tensor) or images.ndim != 5:
                raise ModelValidationError("encode expects a tensor shaped (B, 2, Z, Y, X)")
            if images.shape[1] != V45_CONFIG.window_size:
                raise ModelValidationError("V45 encode requires exactly two frames")
            features = self.unet(images.unsqueeze(2))
            logits = [self.detect_head(features[:, index]) for index in range(2)]
            return features, logits

        def forward(
            self,
            *args: Any,
            diagnostic_allow_dense: bool = False,
            **kwargs: Any,
        ) -> Any:
            if diagnostic_allow_dense is not True:
                raise ModelValidationError(
                    "dense transformer forward is diagnostic-only; "
                    "set diagnostic_allow_dense=True explicitly"
                )
            return self.transformer(
                *args,
                diagnostic_allow_dense=True,
                **kwargs,
            )

    return V45UNetNodeTransformer()


def _torch_tensor_is_finite(torch: Any, value: Any) -> bool:
    if value.is_floating_point() or value.is_complex():
        return bool(torch.isfinite(value).all().item())
    return True


def load_v45_model(
    support_root: str | os.PathLike[str] = DEFAULT_SUPPORT_ROOT,
    *,
    device: str | Any = "cpu",
    expected_manifest_sha256: str = DEFAULT_MANIFEST_SHA256,
    expected_config_sha256: str = DEFAULT_CONFIG_SHA256,
    expected_source_sha256: str = DEFAULT_SOURCE_SHA256,
    expected_weight_sha256: str = DEFAULT_WEIGHT_SHA256,
) -> tuple[Any, ArtifactBundle]:
    """Validate and load V45 with ``weights_only=True`` and exact state checks."""

    bundle = validate_artifact_bundle(
        support_root,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_source_sha256=expected_source_sha256,
        expected_weight_sha256=expected_weight_sha256,
    )
    torch, _nn, _transformer, _unet = _require_torch_and_models()
    model = build_v45_model()
    target_device = torch.device(device)
    try:
        state = torch.load(bundle.weight_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(f"weights_only checkpoint load failed: {exc}") from exc
    if not isinstance(state, Mapping):
        raise ArtifactValidationError("checkpoint is not a state-dict mapping")

    expected_state = model.state_dict()
    state_keys = set(state)
    expected_keys = set(expected_state)
    missing = sorted(expected_keys - state_keys)
    unexpected = sorted(state_keys - expected_keys)
    if missing or unexpected:
        raise ArtifactValidationError(
            f"checkpoint keys mismatch; missing={missing}, unexpected={unexpected}"
        )
    for name in expected_state:
        value = state[name]
        reference = expected_state[name]
        if not isinstance(value, torch.Tensor):
            raise ArtifactValidationError(f"checkpoint value {name!r} is not a tensor")
        if tuple(value.shape) != tuple(reference.shape) or value.dtype != reference.dtype:
            raise ArtifactValidationError(
                f"checkpoint tensor mismatch for {name}: expected {tuple(reference.shape)}/{reference.dtype}, "
                f"got {tuple(value.shape)}/{value.dtype}"
            )
        if not _torch_tensor_is_finite(torch, value):
            raise ArtifactValidationError(f"checkpoint tensor {name!r} contains nonfinite values")
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ArtifactValidationError(f"strict checkpoint load failed: {exc}") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ArtifactValidationError("strict checkpoint load returned incompatible keys")
    model.to(target_device)
    model.eval()
    for name, value in model.state_dict().items():
        if not _torch_tensor_is_finite(torch, value):
            raise ArtifactValidationError(f"loaded model tensor {name!r} contains nonfinite values")
    return model, bundle


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor"


def _require_finite_numpy(value: np.ndarray, label: str) -> None:
    if not np.issubdtype(value.dtype, np.number):
        raise ModelValidationError(f"{label} must contain numeric values")
    if not np.isfinite(value).all():
        raise ModelValidationError(f"{label} contains nonfinite values")


def quantile_normalize(
    values: Any,
    q_low: float | None = None,
    q_high: float | None = None,
    *,
    epsilon: float = 1e-6,
) -> Any:
    """Apply 0.001/0.999 normalization, clamping only values below zero.

    If quantile values are omitted they are computed over the supplied tensor.
    The upper tail is intentionally not clamped.
    """

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ModelValidationError("normalization epsilon must be positive and finite")
    if isinstance(values, np.ndarray):
        _require_finite_numpy(values, "normalization input")
        work = values.astype(np.float32, copy=False)
        low = float(np.quantile(work, V45_CONFIG.quantile_low)) if q_low is None else float(q_low)
        high = float(np.quantile(work, V45_CONFIG.quantile_high)) if q_high is None else float(q_high)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ModelValidationError("normalization quantiles must be finite with q_high > q_low")
        result = np.maximum((work - low) / (high - low + epsilon), np.float32(0.0))
        _require_finite_numpy(result, "normalized values")
        return result.astype(np.float32, copy=False)
    if _is_torch_tensor(values):
        torch, _nn, _transformer, _unet = _require_torch_and_models()
        if not values.is_floating_point():
            work = values.to(dtype=torch.float32)
        else:
            work = values.to(dtype=torch.float32)
        if not bool(torch.isfinite(work).all().item()):
            raise ModelValidationError("normalization input contains nonfinite values")
        low = float(torch.quantile(work, V45_CONFIG.quantile_low).item()) if q_low is None else float(q_low)
        high = float(torch.quantile(work, V45_CONFIG.quantile_high).item()) if q_high is None else float(q_high)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ModelValidationError("normalization quantiles must be finite with q_high > q_low")
        result = ((work - low) / (high - low + epsilon)).clamp_min(0.0)
        if not bool(torch.isfinite(result).all().item()):
            raise ModelValidationError("normalized values contain nonfinite values")
        return result
    raise ModelValidationError("normalization input must be a NumPy array or torch tensor")


def _validate_d4_view(view: int) -> int:
    if type(view) is not int or view not in D4_VIEWS:
        raise ModelValidationError("D4 view must be an integer in [0, 7]")
    return view


def d4_transform(values: Any, view: int) -> Any:
    """Apply one scored-notebook V45 view over the last two (Y, X) axes.

    View 7 intentionally preserves the public notebook's ``rot90`` followed by
    transpose composition.  Algebraically it repeats flip-X, but changing that
    composition would change the scored eight-view logit average.
    """

    index = _validate_d4_view(view)
    if isinstance(values, np.ndarray):
        _require_finite_numpy(values, "D4 input")
        if index == 0:
            return values
        if index == 1:
            return np.flip(values, axis=-1)
        if index == 2:
            return np.flip(values, axis=-2)
        if index == 3:
            return np.flip(values, axis=(-2, -1))
        if index in (4, 5):
            return np.rot90(values, k=1 if index == 4 else 3, axes=(-2, -1))
        if index == 6:
            return np.swapaxes(values, -1, -2)
        return np.swapaxes(np.rot90(values, k=1, axes=(-2, -1)), -1, -2)
    if _is_torch_tensor(values):
        torch, _nn, _transformer, _unet = _require_torch_and_models()
        if not bool(torch.isfinite(values).all().item()):
            raise ModelValidationError("D4 input contains nonfinite values")
        if index == 0:
            return values
        if index == 1:
            return values.flip((-1,))
        if index == 2:
            return values.flip((-2,))
        if index == 3:
            return values.flip((-2, -1))
        if index in (4, 5):
            return torch.rot90(values, k=1 if index == 4 else 3, dims=(-2, -1))
        if index == 6:
            return values.transpose(-1, -2)
        return torch.rot90(values, 1, dims=(-2, -1)).transpose(-1, -2)
    raise ModelValidationError("D4 input must be a NumPy array or torch tensor")


def d4_inverse(values: Any, view: int) -> Any:
    """Invert one :func:`d4_transform` action."""

    index = _validate_d4_view(view)
    if isinstance(values, np.ndarray):
        _require_finite_numpy(values, "D4 output")
        if index == 0:
            return values
        if index == 1:
            return np.flip(values, axis=-1)
        if index == 2:
            return np.flip(values, axis=-2)
        if index == 3:
            return np.flip(values, axis=(-2, -1))
        if index in (4, 5):
            return np.rot90(values, k=-1 if index == 4 else -3, axes=(-2, -1))
        if index == 6:
            return np.swapaxes(values, -1, -2)
        return np.rot90(np.swapaxes(values, -1, -2), k=-1, axes=(-2, -1))
    if _is_torch_tensor(values):
        torch, _nn, _transformer, _unet = _require_torch_and_models()
        if not bool(torch.isfinite(values).all().item()):
            raise ModelValidationError("D4 output contains nonfinite values")
        if index == 0:
            return values
        if index == 1:
            return values.flip((-1,))
        if index == 2:
            return values.flip((-2,))
        if index == 3:
            return values.flip((-2, -1))
        if index in (4, 5):
            return torch.rot90(values, k=-1 if index == 4 else -3, dims=(-2, -1))
        if index == 6:
            return values.transpose(-1, -2)
        return torch.rot90(values.transpose(-1, -2), -1, dims=(-2, -1))
    raise ModelValidationError("D4 output must be a NumPy array or torch tensor")


def d4_invert_and_average(outputs: Sequence[Any]) -> Any:
    """Invert and average exactly eight D4-view outputs without stacking them."""

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 8:
        raise ModelValidationError("D4 averaging requires exactly eight ordered outputs")
    restored = [d4_inverse(value, view) for view, value in enumerate(outputs)]
    shape = tuple(restored[0].shape)
    if any(tuple(value.shape) != shape for value in restored):
        raise ModelValidationError("inverted D4 outputs do not share a shape")
    if isinstance(restored[0], np.ndarray):
        if not all(isinstance(value, np.ndarray) for value in restored):
            raise ModelValidationError("D4 outputs use mixed feature containers")
        result = restored[0].astype(np.float32, copy=True)
        for value in restored[1:]:
            result += value.astype(np.float32, copy=False)
        result /= np.float32(8.0)
        _require_finite_numpy(result, "D4 average")
        return result
    if not all(_is_torch_tensor(value) for value in restored):
        raise ModelValidationError("D4 outputs use mixed feature containers")
    result = restored[0].clone()
    for value in restored[1:]:
        result.add_(value)
    result.div_(8.0)
    torch, _nn, _transformer, _unet = _require_torch_and_models()
    if not bool(torch.isfinite(result).all().item()):
        raise ModelValidationError("D4 average contains nonfinite values")
    return result


@dataclass(frozen=True)
class VoxelNode:
    node_id: int
    t: int
    z: int
    y: int
    x: int
    score: float


def _readonly_array(value: np.ndarray, dtype: str, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ModelValidationError(f"{label} must be a NumPy array; unsupported feature container")
    wanted = np.dtype(dtype)
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.complexfloating):
        raise ModelValidationError(f"{label} must contain real numeric values")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ModelValidationError(f"{label} contains nonfinite values")
    if np.issubdtype(wanted, np.integer):
        if np.issubdtype(value.dtype, np.floating) and not np.equal(value, np.trunc(value)).all():
            raise ModelValidationError(f"{label} must contain integer-valued data")
        if value.size:
            limits = np.iinfo(wanted)
            if np.min(value) < limits.min or np.max(value) > limits.max:
                raise ModelValidationError(f"{label} exceeds the supported integer range")
    result = np.ascontiguousarray(value, dtype=wanted)
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ModelValidationError(f"{label} contains nonfinite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FrameDet:
    """Canonical first-sight detections for one video frame."""

    frame_index: int
    node_ids: np.ndarray
    coords_zyx: np.ndarray
    scores: np.ndarray
    grid_coords_zyx: np.ndarray
    grid_shape_zyx: tuple[int, int, int]

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ModelValidationError("frame_index must be a nonnegative integer")
        node_ids = _readonly_array(self.node_ids, "<i8", "node_ids")
        coords = _readonly_array(self.coords_zyx, "<i8", "coords_zyx")
        scores = _readonly_array(self.scores, "<f4", "scores")
        grid_coords = _readonly_array(self.grid_coords_zyx, "<i8", "grid_coords_zyx")
        if node_ids.ndim != 1:
            raise ModelValidationError("node_ids must be one-dimensional")
        count = len(node_ids)
        if coords.shape != (count, 3) or grid_coords.shape != (count, 3):
            raise ModelValidationError("node coordinates must have shape (N, 3)")
        if scores.shape != (count,):
            raise ModelValidationError("scores must have shape (N,)")
        if count and np.any(node_ids < 0):
            raise ModelValidationError("node IDs must be nonnegative")
        if count and (np.any(scores < 0.0) or np.any(scores > 1.0)):
            raise ModelValidationError("detection scores must lie in [0, 1]")
        if len(set(int(value) for value in node_ids)) != count:
            raise ModelValidationError("duplicate node IDs within a frame")
        if len({tuple(int(v) for v in row) for row in coords}) != count:
            raise ModelValidationError("duplicate voxel coordinates must be deduplicated")
        if len(self.grid_shape_zyx) != 3 or any(
            type(value) is not int or value <= 0 for value in self.grid_shape_zyx
        ):
            raise ModelValidationError("grid_shape_zyx must contain three positive integers")
        if count:
            shape = np.asarray(self.grid_shape_zyx, dtype=np.int64)
            if np.any(grid_coords < 0) or np.any(grid_coords >= shape[None, :]):
                raise ModelValidationError("grid coordinates lie outside the feature volume")
            if np.any(coords < 0):
                raise ModelValidationError("voxel coordinates must be nonnegative")
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "coords_zyx", coords)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "grid_coords_zyx", grid_coords)

    @property
    def nodes(self) -> tuple[VoxelNode, ...]:
        return tuple(
            VoxelNode(
                node_id=int(self.node_ids[index]),
                t=self.frame_index,
                z=int(self.coords_zyx[index, 0]),
                y=int(self.coords_zyx[index, 1]),
                x=int(self.coords_zyx[index, 2]),
                score=float(self.scores[index]),
            )
            for index in range(len(self.node_ids))
        )


@dataclass(frozen=True)
class PairContext:
    """Pair-local UNet features indexed at canonical first-sight nodes."""

    source_frame: int
    target_frame: int
    source_ids: np.ndarray
    target_ids: np.ndarray
    source_feat: np.ndarray
    target_feat: np.ndarray
    grid_shape_zyx: tuple[int, int, int]

    def __post_init__(self) -> None:
        if (
            type(self.source_frame) is not int
            or self.source_frame < 0
            or type(self.target_frame) is not int
            or self.target_frame != self.source_frame + 1
        ):
            raise ModelValidationError("pair context requires adjacent forward frames")
        source_ids = _readonly_array(self.source_ids, "<i8", "pair source IDs")
        target_ids = _readonly_array(self.target_ids, "<i8", "pair target IDs")
        source_feat = _readonly_array(self.source_feat, "<f4", "pair source features")
        target_feat = _readonly_array(self.target_feat, "<f4", "pair target features")
        if source_ids.ndim != 1 or target_ids.ndim != 1:
            raise ModelValidationError("pair node IDs must be one-dimensional")
        if source_feat.shape != (len(source_ids), V45_CONFIG.unet_out_channels):
            raise ModelValidationError("pair source features must have shape (N, 32)")
        if target_feat.shape != (len(target_ids), V45_CONFIG.unet_out_channels):
            raise ModelValidationError("pair target features must have shape (N, 32)")
        if len(set(int(value) for value in source_ids)) != len(source_ids):
            raise ModelValidationError("duplicate source node ID in pair context")
        if len(set(int(value) for value in target_ids)) != len(target_ids):
            raise ModelValidationError("duplicate target node ID in pair context")
        if len(self.grid_shape_zyx) != 3 or any(
            type(value) is not int or value <= 0 for value in self.grid_shape_zyx
        ):
            raise ModelValidationError("pair grid shape must contain three positive integers")
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "source_feat", source_feat)
        object.__setattr__(self, "target_feat", target_feat)

    @property
    def frame_pair(self) -> tuple[int, int]:
        return self.source_frame, self.target_frame


@dataclass(frozen=True)
class VideoInference:
    """Canonical detections plus pair-local contexts for one video."""

    detections: tuple[FrameDet, ...]
    contexts: tuple[PairContext, ...]
    pair_batch_size: int = 1
    view_batch_size: int = 1
    d4_views: tuple[int, ...] = D4_VIEWS

    def __post_init__(self) -> None:
        if type(self.pair_batch_size) is not int or self.pair_batch_size <= 0:
            raise ModelValidationError("pair_batch_size must be a positive integer")
        if type(self.view_batch_size) is not int or not 1 <= self.view_batch_size <= len(D4_VIEWS):
            raise ModelValidationError("view_batch_size must be an integer in [1, 8]")
        views = _normalize_d4_views(self.d4_views)
        if self.view_batch_size > len(views):
            raise ModelValidationError("view_batch_size must not exceed the selected D4 view count")
        try:
            _validate_video_records(self.detections, self.contexts)
        except ModelValidationError as exc:
            raise ModelValidationError(f"invalid video inference records: {exc}") from exc
        object.__setattr__(self, "d4_views", views)

    @property
    def config_sha256(self) -> str:
        return v45_config_sha(
            pair_batch_size=self.pair_batch_size,
            view_batch_size=self.view_batch_size,
            d4_views=self.d4_views,
        )


@dataclass(frozen=True)
class FrameFeatures:
    """Legacy composite used only by the diagnostic dense edge scorer."""

    frame_index: int
    node_ids: np.ndarray
    coords_zyx: np.ndarray
    scores: np.ndarray
    embeddings: np.ndarray
    grid_coords_zyx: np.ndarray
    grid_shape_zyx: tuple[int, int, int]

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ModelValidationError("frame_index must be a nonnegative integer")
        node_ids = _readonly_array(self.node_ids, "<i8", "node_ids")
        coords = _readonly_array(self.coords_zyx, "<i8", "coords_zyx")
        scores = _readonly_array(self.scores, "<f4", "scores")
        embeddings = _readonly_array(self.embeddings, "<f4", "embeddings")
        grid_coords = _readonly_array(self.grid_coords_zyx, "<i8", "grid_coords_zyx")
        if node_ids.ndim != 1:
            raise ModelValidationError("node_ids must be one-dimensional")
        count = len(node_ids)
        if coords.shape != (count, 3) or grid_coords.shape != (count, 3):
            raise ModelValidationError("node coordinates must have shape (N, 3)")
        if scores.shape != (count,):
            raise ModelValidationError("scores must have shape (N,)")
        if embeddings.ndim != 2 or embeddings.shape[0] != count:
            raise ModelValidationError("embeddings must have shape (N, D)")
        if embeddings.shape[1] != V45_CONFIG.unet_out_channels:
            raise ModelValidationError("V45 model embeddings must have exactly 32 columns")
        if count and np.any(node_ids < 0):
            raise ModelValidationError("node IDs must be nonnegative")
        if count and (np.any(scores < 0.0) or np.any(scores > 1.0)):
            raise ModelValidationError("detection scores must lie in [0, 1]")
        if len(set(int(value) for value in node_ids)) != count:
            raise ModelValidationError("duplicate node IDs within a frame")
        if len({tuple(int(v) for v in row) for row in coords}) != count:
            raise ModelValidationError("duplicate voxel coordinates must be deduplicated")
        if any(type(value) is not int or value <= 0 for value in self.grid_shape_zyx):
            raise ModelValidationError("grid_shape_zyx must contain three positive integers")
        if len(self.grid_shape_zyx) != 3:
            raise ModelValidationError("grid_shape_zyx must contain exactly three axes")
        if count:
            shape = np.asarray(self.grid_shape_zyx, dtype=np.int64)
            if np.any(grid_coords < 0) or np.any(grid_coords >= shape[None, :]):
                raise ModelValidationError("grid coordinates lie outside the feature volume")
            if np.any(coords < 0):
                raise ModelValidationError("voxel coordinates must be nonnegative")
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "coords_zyx", coords)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "grid_coords_zyx", grid_coords)

    @property
    def nodes(self) -> tuple[VoxelNode, ...]:
        return tuple(
            VoxelNode(
                node_id=int(self.node_ids[index]),
                t=self.frame_index,
                z=int(self.coords_zyx[index, 0]),
                y=int(self.coords_zyx[index, 1]),
                x=int(self.coords_zyx[index, 2]),
                score=float(self.scores[index]),
            )
            for index in range(len(self.node_ids))
        )


def dedup_dets(
    frame_index: int,
    coords_zyx: np.ndarray,
    scores: np.ndarray,
    *,
    grid_coords_zyx: np.ndarray,
    grid_shape_zyx: tuple[int, int, int],
    node_id_start: int = 0,
) -> tuple[FrameDet, np.ndarray]:
    """Create canonical detections and return their selected input rows."""

    if type(node_id_start) is not int or node_id_start < 0:
        raise ModelValidationError("node_id_start must be a nonnegative integer")
    if not all(isinstance(value, np.ndarray) for value in (coords_zyx, scores, grid_coords_zyx)):
        raise ModelValidationError("detections must use NumPy arrays")
    coords = _readonly_array(coords_zyx, "<i8", "coords_zyx")
    score_values = _readonly_array(scores, "<f4", "scores")
    grid_coords = _readonly_array(grid_coords_zyx, "<i8", "grid_coords_zyx")
    count = len(coords)
    if coords.shape != (count, 3) or grid_coords.shape != (count, 3):
        raise ModelValidationError("detection coordinates must have shape (N, 3)")
    if score_values.shape != (count,):
        raise ModelValidationError("detection scores have an incompatible shape")
    winner: dict[tuple[int, int, int], int] = {}
    for index, row in enumerate(coords):
        key = tuple(int(value) for value in row)
        previous = winner.get(key)
        if previous is None or float(score_values[index]) > float(score_values[previous]):
            winner[key] = index
    selected = [winner[key] for key in sorted(winner)]
    selected_array = np.asarray(selected, dtype=np.int64)
    unique_count = len(selected)
    return (
        FrameDet(
            frame_index=frame_index,
            node_ids=np.arange(node_id_start, node_id_start + unique_count, dtype=np.int64),
            coords_zyx=np.ascontiguousarray(coords[selected_array], dtype=np.int64),
            scores=np.ascontiguousarray(score_values[selected_array], dtype=np.float32),
            grid_coords_zyx=np.ascontiguousarray(grid_coords[selected_array], dtype=np.int64),
            grid_shape_zyx=grid_shape_zyx,
        ),
        selected_array,
    )


def deduplicate_frame_detections(
    frame_index: int,
    coords_zyx: np.ndarray,
    scores: np.ndarray,
    embeddings: np.ndarray,
    *,
    grid_coords_zyx: np.ndarray,
    grid_shape_zyx: tuple[int, int, int],
    node_id_start: int = 0,
) -> FrameFeatures:
    """Legacy composite helper; production registries use :func:`dedup_dets`."""

    if not isinstance(embeddings, np.ndarray):
        raise ModelValidationError("detection features must use NumPy arrays")
    feature_values = _readonly_array(embeddings, "<f4", "embeddings")
    if feature_values.shape != (len(coords_zyx), V45_CONFIG.unet_out_channels):
        raise ModelValidationError("detection features have an incompatible shape")
    detections, selected = dedup_dets(
        frame_index,
        coords_zyx,
        scores,
        grid_coords_zyx=grid_coords_zyx,
        grid_shape_zyx=grid_shape_zyx,
        node_id_start=node_id_start,
    )
    return FrameFeatures(
        frame_index=detections.frame_index,
        node_ids=detections.node_ids,
        coords_zyx=detections.coords_zyx,
        scores=detections.scores,
        embeddings=np.ascontiguousarray(feature_values[selected], dtype=np.float32),
        grid_coords_zyx=detections.grid_coords_zyx,
        grid_shape_zyx=detections.grid_shape_zyx,
    )


@dataclass(frozen=True)
class TwoFrameInference:
    source: FrameFeatures
    target: FrameFeatures
    d4_views: tuple[int, ...] = D4_VIEWS
    edge_feature_tta: bool = False
    view_batch_size: int = 1

    def __post_init__(self) -> None:
        if self.target.frame_index != self.source.frame_index + 1:
            raise ModelValidationError("two-frame inference requires adjacent forward frames")
        views = _normalize_d4_views(self.d4_views)
        if self.edge_feature_tta is not False:
            raise ModelValidationError("V45 requires detection-only D4 TTA")
        if type(self.view_batch_size) is not int or not 1 <= self.view_batch_size <= len(D4_VIEWS):
            raise ModelValidationError("view_batch_size must be an integer in [1, 8]")
        if self.view_batch_size > len(views):
            raise ModelValidationError("view_batch_size must not exceed the selected D4 view count")
        object.__setattr__(self, "d4_views", views)
        validate_frame_sequence((self.source, self.target))


def _validate_encode_output(torch: Any, value: Any, input_shape: tuple[int, ...]) -> tuple[Any, list[Any]]:
    if len(input_shape) != 5 or input_shape[0] <= 0 or input_shape[1] != V45_CONFIG.window_size:
        raise ModelValidationError("encode input shape must be (B, 2, Z, Y, X) with B > 0")
    if not isinstance(value, tuple) or len(value) != 2:
        raise ModelValidationError("model.encode returned an unsupported feature container")
    features, logits = value
    if not isinstance(features, torch.Tensor):
        raise ModelValidationError("model.encode features must be one torch tensor")
    if not isinstance(logits, (list, tuple)) or len(logits) != 2:
        raise ModelValidationError("model.encode logits must be a two-element list or tuple")
    batch_size = int(input_shape[0])
    if features.ndim != 6 or tuple(features.shape[:3]) != (
        batch_size,
        V45_CONFIG.window_size,
        V45_CONFIG.unet_out_channels,
    ):
        raise ModelValidationError("model.encode returned an unexpected V45 feature shape")
    for item in logits:
        if (
            not isinstance(item, torch.Tensor)
            or item.ndim != 5
            or tuple(item.shape[:2]) != (batch_size, 1)
            or tuple(item.shape[2:]) != tuple(input_shape[-3:])
        ):
            raise ModelValidationError("model.encode returned an unsupported detection-logit container")
        if not bool(torch.isfinite(item).all().item()):
            raise ModelValidationError("model detection logits contain nonfinite values")
    if not bool(torch.isfinite(features).all().item()):
        raise ModelValidationError("model feature tensor contains nonfinite values")
    if tuple(features.shape[3:]) != tuple(input_shape[-3:]):
        raise ModelValidationError("model feature spatial shape does not match the input grid")
    return features, list(logits)


def _detect_batch(torch: Any, logits: Any) -> list[tuple[Any, Any]]:
    """Detect peaks for a batch while pooling all samples in one operation."""

    if not isinstance(logits, torch.Tensor) or logits.ndim != 5 or logits.shape[1] != 1:
        raise ModelValidationError("batched detection logits must have shape (B, 1, Z, Y, X)")
    if logits.shape[0] <= 0 or not bool(torch.isfinite(logits).all().item()):
        raise ModelValidationError("batched detection logits must be finite with B > 0")
    kernel = V45_CONFIG.pool_kernel_downsampled_zyx
    pooled = torch.nn.functional.max_pool3d(
        logits,
        kernel_size=kernel,
        stride=1,
        padding=tuple(k // 2 for k in kernel),
    )
    probabilities = torch.sigmoid(logits)
    masks = (logits == pooled) & (probabilities > V45_CONFIG.detection_threshold)
    results: list[tuple[Any, Any]] = []
    for batch_index in range(int(logits.shape[0])):
        coords = torch.nonzero(masks[batch_index, 0], as_tuple=False).to(dtype=torch.int64)
        if len(coords):
            z, y, x = coords.T
            scores = probabilities[batch_index, 0, z, y, x].to(dtype=torch.float32)
        else:
            scores = torch.empty(0, dtype=torch.float32, device=logits.device)
        results.append((coords, scores))
    return results


def _encode_d4(
    torch: Any,
    model: Any,
    model_input: Any,
    *,
    view_batch_size: int = 1,
    d4_views: Sequence[int] | None = None,
) -> tuple[Any, list[Any]]:
    """Encode public-order D4 views or a deterministic diagnostic subset.

    The full eight-view set remains the V45 default.  Reduced sets exist only
    to measure inference/quality trade-offs under frozen held-out evaluation.
    """

    if not isinstance(model_input, torch.Tensor) or model_input.ndim != 5:
        raise ModelValidationError("D4 encode input must have shape (B, 2, Z, Y, X)")
    batch_size = int(model_input.shape[0])
    if batch_size <= 0 or model_input.shape[1] != V45_CONFIG.window_size:
        raise ModelValidationError("D4 encode requires B > 0 and exactly two frames")
    selected_views = _normalize_d4_views(d4_views)
    if type(view_batch_size) is not int or not 1 <= view_batch_size <= len(selected_views):
        raise ModelValidationError("view_batch_size must be an integer in [1, selected D4 view count]")

    original_features: Any | None = None
    sums: list[Any | None] = [None, None]
    if view_batch_size == 1:
        for view in selected_views:
            transformed = d4_transform(model_input, view)
            features, logits = _validate_encode_output(
                torch,
                model.encode(transformed),
                tuple(int(value) for value in transformed.shape),
            )
            if view == 0:
                original_features = features.clone()
            for frame_offset in range(V45_CONFIG.window_size):
                restored = d4_inverse(logits[frame_offset], view)
                if tuple(restored.shape[-3:]) != tuple(model_input.shape[-3:]):
                    raise ModelValidationError(
                        "D4-inverted detection logits have an inconsistent shape"
                    )
                if sums[frame_offset] is None:
                    sums[frame_offset] = restored.clone()
                else:
                    sums[frame_offset].add_(restored)
        if original_features is None or any(value is None for value in sums):
            raise ModelValidationError("sequential D4 encoding did not produce every required view")
        averaged = [value.div_(float(len(selected_views))) for value in sums if value is not None]
        return original_features, averaged

    grouped: dict[tuple[int, int, int], list[int]] = {}
    z_size, y_size, x_size = (int(value) for value in model_input.shape[-3:])
    for view in selected_views:
        shape = (z_size, x_size, y_size) if view in (4, 5, 6) else (z_size, y_size, x_size)
        grouped.setdefault(shape, []).append(view)

    pending: dict[int, list[Any]] = {}
    next_view_index = 0
    for shape, compatible_views in grouped.items():
        for view_start in range(0, len(compatible_views), view_batch_size):
            batch_views = compatible_views[view_start : view_start + view_batch_size]
            batched = torch.empty(
                (batch_size * len(batch_views), V45_CONFIG.window_size, *shape),
                dtype=model_input.dtype,
                device=model_input.device,
            )
            for entry_index, view in enumerate(batch_views):
                start = entry_index * batch_size
                batched[start : start + batch_size].copy_(d4_transform(model_input, view))
            features, logits = _validate_encode_output(
                torch,
                model.encode(batched),
                tuple(int(value) for value in batched.shape),
            )
            for entry_index, view in enumerate(batch_views):
                start = entry_index * batch_size
                end = start + batch_size
                if view == 0:
                    original_features = features[start:end].clone()
                restored_view: list[Any] = []
                for frame_offset in range(V45_CONFIG.window_size):
                    restored = d4_inverse(logits[frame_offset][start:end], view)
                    if tuple(restored.shape[-3:]) != tuple(model_input.shape[-3:]):
                        raise ModelValidationError(
                            "D4-inverted detection logits have an inconsistent shape"
                        )
                    restored_view.append(restored.clone())
                pending[view] = restored_view
                while (
                    next_view_index < len(selected_views)
                    and selected_views[next_view_index] in pending
                ):
                    ordered = pending.pop(selected_views[next_view_index])
                    for frame_offset, restored in enumerate(ordered):
                        if sums[frame_offset] is None:
                            sums[frame_offset] = restored
                        else:
                            sums[frame_offset].add_(restored)
                    next_view_index += 1
    if (
        original_features is None
        or next_view_index != len(selected_views)
        or any(value is None for value in sums)
    ):
        raise ModelValidationError("D4 encoding did not produce every selected view")
    averaged = [value.div_(float(len(selected_views))) for value in sums if value is not None]
    return original_features, averaged


def _frame_numpy(torch: Any, frame: Any, label: str) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        result = frame
    elif isinstance(frame, torch.Tensor):
        result = frame.detach().cpu().numpy()
    else:
        raise ModelValidationError(f"{label} must be a NumPy array or torch tensor")
    if result.ndim != 3:
        raise ModelValidationError(f"{label} must have shape (Z, Y, X)")
    _require_finite_numpy(result, label)
    return result


def infer_two_frames(
    model: Any,
    frame_source: Any,
    frame_target: Any,
    *,
    source_frame_index: int,
    q_low: float,
    q_high: float,
    device: str | Any | None = None,
    node_id_start: int = 0,
    view_batch_size: int = 1,
    d4_views: Sequence[int] | None = None,
) -> TwoFrameInference:
    """Run deterministic adjacent-frame detection with detection-only D4 TTA.

    Inputs are original-resolution 3-D frames.  V45 applies the locked
    ``[1,4,4]`` stride, maps detections back to integer original voxels, and
    indexes embeddings only from the unaugmented feature map.  The default
    performs the public base-plus-seven encode calls sequentially.  A larger
    ``view_batch_size`` or a reduced ``d4_views`` set is diagnostic-only.
    """

    torch, _nn, _transformer, _unet = _require_torch_and_models()
    if type(source_frame_index) is not int or source_frame_index < 0:
        raise ModelValidationError("source_frame_index must be a nonnegative integer")
    if not hasattr(model, "encode") or not callable(model.encode):
        raise ModelValidationError("model does not expose the required encode primitive")
    if getattr(model, "training", True):
        raise ModelValidationError("deterministic inference requires model.eval()")
    selected_d4_views = _normalize_d4_views(d4_views)

    source_np = _frame_numpy(torch, frame_source, "source frame")
    target_np = _frame_numpy(torch, frame_target, "target frame")
    if source_np.shape != target_np.shape:
        raise ModelValidationError("two input frames must have equal spatial shapes")
    dz, dy, dx = V45_CONFIG.spatial_stride_zyx
    source_ds = source_np[::dz, ::dy, ::dx]
    target_ds = target_np[::dz, ::dy, ::dx]
    stacked = np.stack([source_ds, target_ds], axis=0)
    normalized = quantile_normalize(stacked, q_low, q_high)

    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration) as exc:
        raise ModelValidationError("model has no parameter device/dtype") from exc
    target_device = parameter.device if device is None else torch.device(device)
    model_input = torch.from_numpy(np.ascontiguousarray(normalized)).to(
        device=target_device, dtype=parameter.dtype
    )[None]

    with torch.inference_mode():
        original_features, averaged_logits = _encode_d4(
            torch,
            model,
            model_input,
            view_batch_size=view_batch_size,
            d4_views=selected_d4_views,
        )

        frame_results: list[FrameFeatures] = []
        next_node_id = node_id_start
        stride = np.asarray(V45_CONFIG.spatial_stride_zyx, dtype=np.int64)
        grid_shape = tuple(int(value) for value in model_input.shape[-3:])
        for offset in range(2):
            grid_coords, scores = _detect_batch(torch, averaged_logits[offset])[0]
            if len(grid_coords):
                z, y, x = grid_coords.T
                embeddings = original_features[0, offset, :, z, y, x].T
            else:
                embeddings = torch.empty(
                    (0, V45_CONFIG.unet_out_channels),
                    dtype=original_features.dtype,
                    device=original_features.device,
                )
            if not bool(torch.isfinite(embeddings).all().item()):
                raise ModelValidationError("indexed model embeddings contain nonfinite values")
            grid_np = grid_coords.detach().cpu().numpy().astype(np.int64, copy=False)
            coords_np = grid_np * stride[None, :]
            result = deduplicate_frame_detections(
                source_frame_index + offset,
                coords_np,
                scores.detach().cpu().numpy(),
                embeddings.detach().to(dtype=torch.float32).cpu().numpy(),
                grid_coords_zyx=grid_np,
                grid_shape_zyx=grid_shape,
                node_id_start=next_node_id,
            )
            frame_results.append(result)
            next_node_id += len(result.node_ids)
    return TwoFrameInference(
        frame_results[0],
        frame_results[1],
        d4_views=selected_d4_views,
        view_batch_size=view_batch_size,
    )


def _index_pair_map(torch: Any, feature_map: Any, detections: FrameDet) -> np.ndarray:
    if not isinstance(feature_map, torch.Tensor) or feature_map.ndim != 4:
        raise ModelValidationError("pair feature map must have shape (32, Z, Y, X)")
    if tuple(int(value) for value in feature_map.shape) != (
        V45_CONFIG.unet_out_channels,
        *detections.grid_shape_zyx,
    ):
        raise ModelValidationError("pair feature map does not match the canonical detection grid")
    if len(detections.node_ids) == 0:
        return np.empty((0, V45_CONFIG.unet_out_channels), dtype=np.float32)
    grid = torch.from_numpy(np.array(detections.grid_coords_zyx, dtype=np.int64, copy=True, order="C")).to(
        device=feature_map.device,
        dtype=torch.int64,
    )
    z, y, x = grid.T
    indexed = feature_map[:, z, y, x].T
    if not bool(torch.isfinite(indexed).all().item()):
        raise ModelValidationError("indexed pair-context features contain nonfinite values")
    return indexed.detach().to(dtype=torch.float32).cpu().numpy()


def infer_video(
    model: Any,
    frames_value: Sequence[Any],
    *,
    q_low: float,
    q_high: float,
    start_frame_index: int = 0,
    node_id_start: int = 0,
    batch_size: int = 1,
    view_batch_size: int = 1,
    d4_views: Sequence[int] | None = None,
    device: str | Any | None = None,
) -> VideoInference:
    """Run exact V45 two-frame sliding windows over a video of at least 3 frames.

    A frame's detections, coordinates, scores, and IDs are registered only on
    first sight.  Every adjacent pair then indexes that pair's unaugmented UNet
    maps at the registered coordinates, including an interior source frame
    whose detections came from the preceding window.  The returned contexts
    contain O(N) node features and never learned O(NxN) edge logits.

    Exact public control behavior requires the default batch sizes and all
    eight D4 views.  Reduced ``d4_views`` sets are diagnostic-only; they make
    no public-parity claim and receive a distinct execution identity.
    """

    torch, _nn, _transformer, _unet = _require_torch_and_models()
    if not isinstance(frames_value, (tuple, list)) or len(frames_value) < 3:
        raise ModelValidationError("video inference requires a tuple or list of at least 3 frames")
    if type(start_frame_index) is not int or start_frame_index < 0:
        raise ModelValidationError("start_frame_index must be a nonnegative integer")
    if type(node_id_start) is not int or node_id_start < 0:
        raise ModelValidationError("node_id_start must be a nonnegative integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ModelValidationError("batch_size must be a positive integer")
    if type(view_batch_size) is not int or not 1 <= view_batch_size <= len(D4_VIEWS):
        raise ModelValidationError("view_batch_size must be an integer in [1, 8]")
    selected_d4_views = _normalize_d4_views(d4_views)
    if view_batch_size > len(selected_d4_views):
        raise ModelValidationError("view_batch_size must not exceed the selected D4 view count")
    if not hasattr(model, "encode") or not callable(model.encode):
        raise ModelValidationError("model does not expose the required encode primitive")
    if getattr(model, "training", True):
        raise ModelValidationError("deterministic inference requires model.eval()")

    input_frames = [
        _frame_numpy(torch, frame, f"video frame {offset}")
        for offset, frame in enumerate(frames_value)
    ]
    input_shape = input_frames[0].shape
    if any(frame.shape != input_shape for frame in input_frames[1:]):
        raise ModelValidationError("all video frames must have equal spatial shapes")
    dz, dy, dx = V45_CONFIG.spatial_stride_zyx
    normalized = [
        quantile_normalize(frame[::dz, ::dy, ::dx], q_low, q_high)
        for frame in input_frames
    ]

    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration) as exc:
        raise ModelValidationError("model has no parameter device/dtype") from exc
    target_device = parameter.device if device is None else torch.device(device)
    stride = np.asarray(V45_CONFIG.spatial_stride_zyx, dtype=np.int64)
    grid_shape = tuple(int(value) for value in normalized[0].shape)
    registry: dict[int, FrameDet] = {}
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
            features, averaged_logits = _encode_d4(
                torch,
                model,
                pair_batch,
                view_batch_size=view_batch_size,
                d4_views=selected_d4_views,
            )
            detected = [_detect_batch(torch, item) for item in averaged_logits]

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
                source_feat = _index_pair_map(torch, features[local_index, 0], source_det)
                target_feat = _index_pair_map(torch, features[local_index, 1], target_det)
                contexts.append(
                    PairContext(
                        source_frame=source_frame,
                        target_frame=target_frame,
                        source_ids=source_det.node_ids,
                        target_ids=target_det.node_ids,
                        source_feat=source_feat,
                        target_feat=target_feat,
                        grid_shape_zyx=grid_shape,
                    )
                )

    detections = tuple(registry[start_frame_index + offset] for offset in range(len(normalized)))
    return VideoInference(
        detections,
        tuple(contexts),
        pair_batch_size=batch_size,
        view_batch_size=view_batch_size,
        d4_views=selected_d4_views,
    )


def positional_features(
    coords_tzyx: np.ndarray,
    image_shape_tzyx: tuple[int, int, int, int],
    *,
    per_axis_dim: int = 8,
) -> np.ndarray:
    """Public checkpoint-compatible sinusoidal positional features."""

    if not isinstance(coords_tzyx, np.ndarray) or coords_tzyx.ndim != 2 or coords_tzyx.shape[1] != 4:
        raise ModelValidationError("coords_tzyx must be a NumPy array shaped (N, 4)")
    _require_finite_numpy(coords_tzyx, "positional coordinates")
    if len(image_shape_tzyx) != 4 or any(type(value) is not int or value <= 0 for value in image_shape_tzyx):
        raise ModelValidationError("image_shape_tzyx must contain four positive integers")
    if type(per_axis_dim) is not int or per_axis_dim <= 0 or per_axis_dim % 2:
        raise ModelValidationError("per_axis_dim must be a positive even integer")
    coords = coords_tzyx.astype(np.float32, copy=False)
    parts: list[np.ndarray] = []
    frequencies = 2.0 ** np.arange(per_axis_dim // 2, dtype=np.float32)
    for axis, size in enumerate(image_shape_tzyx):
        angles = (coords[:, axis] / float(max(size, 1)))[:, None] * frequencies[None, :] * np.pi
        parts.extend((np.sin(angles), np.cos(angles)))
    result = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    _require_finite_numpy(result, "positional features")
    return result


@dataclass(frozen=True)
class SourceSoftmaxStats:
    source_count: int
    target_count: int
    configured_source_chunk: int
    peak_source_chunk: int
    candidate_count: int
    peak_chunk_candidate_count: int
    floor: float

    def __post_init__(self) -> None:
        integer_names = (
            "source_count",
            "target_count",
            "configured_source_chunk",
            "peak_source_chunk",
            "candidate_count",
            "peak_chunk_candidate_count",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in integer_names):
            raise ModelValidationError("source-softmax counts must be nonnegative integers")
        if self.configured_source_chunk <= 0:
            raise ModelValidationError("configured source chunk must be positive")
        if self.peak_source_chunk > self.configured_source_chunk:
            raise ModelValidationError("recorded peak source chunk exceeds its configured bound")
        if not math.isfinite(self.floor) or not 0.0 < self.floor <= 1.0:
            raise ModelValidationError("edge floor must be finite and in (0, 1]")


@dataclass(frozen=True)
class SparseEdgeCOO:
    """Thresholded sparse edges; no dense logits or probability matrix."""

    source: np.ndarray
    target: np.ndarray
    probabilities: np.ndarray
    stats: SourceSoftmaxStats

    def __post_init__(self) -> None:
        source = _readonly_array(self.source, "<i8", "edge source")
        target = _readonly_array(self.target, "<i8", "edge target")
        probabilities = _readonly_array(self.probabilities, "<f8", "edge probabilities")
        if source.ndim != 1 or target.ndim != 1 or probabilities.ndim != 1:
            raise ModelValidationError("COO arrays must all be one-dimensional")
        if not (len(source) == len(target) == len(probabilities) == self.stats.candidate_count):
            raise ModelValidationError("COO lengths do not match recorded candidate_count")
        if len(probabilities) and (np.any(probabilities <= self.stats.floor) or np.any(probabilities > 1.0)):
            raise ModelValidationError("COO probabilities violate the strict floor or unit bound")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True)
class PairEdges:
    """Sparse learned edges and softmax accounting for exactly one pair."""

    source_frame: int
    target_frame: int
    edges: SparseEdgeCOO

    def __post_init__(self) -> None:
        if (
            type(self.source_frame) is not int
            or self.source_frame < 0
            or type(self.target_frame) is not int
            or self.target_frame != self.source_frame + 1
        ):
            raise ModelValidationError("pair edges require adjacent forward frames")
        if not isinstance(self.edges, SparseEdgeCOO):
            raise ModelValidationError("pair edges require a SparseEdgeCOO record")

    @property
    def frame_pair(self) -> tuple[int, int]:
        return self.source_frame, self.target_frame


def _logsumexp_axis0(chunk: np.ndarray) -> np.ndarray:
    maximum = np.max(chunk, axis=0)
    return maximum + np.log(np.exp(chunk - maximum[None, :]).sum(axis=0))


def source_softmax_coo(
    source_count: int,
    target_count: int,
    logit_chunk: Callable[[int, int], np.ndarray],
    *,
    source_chunk_size: int,
    floor: float = 0.01,
) -> SparseEdgeCOO:
    """Two-pass source-axis logsumexp and strict-floor COO emission.

    ``logit_chunk(start, end)`` is invoked once in each pass and must return
    exactly ``(end-start, target_count)`` finite logits.  Only one source chunk
    and the target-sized log-normalizer are retained.
    """

    for value, label in ((source_count, "source_count"), (target_count, "target_count")):
        if type(value) is not int or value < 0:
            raise ModelValidationError(f"{label} must be a nonnegative integer")
    if type(source_chunk_size) is not int or source_chunk_size <= 0:
        raise ModelValidationError("source_chunk_size must be a positive integer")
    if not callable(logit_chunk):
        raise ModelValidationError("logit_chunk must be callable")
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ModelValidationError("edge floor must be finite and in (0, 1]")

    peak_source_chunk = min(source_count, source_chunk_size) if source_count else 0
    if source_count == 0 or target_count == 0:
        stats = SourceSoftmaxStats(
            source_count, target_count, source_chunk_size, peak_source_chunk, 0, 0, float(floor)
        )
        return SparseEdgeCOO(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            stats,
        )

    log_normalizer = np.full(target_count, -np.inf, dtype=np.float64)
    for start in range(0, source_count, source_chunk_size):
        end = min(start + source_chunk_size, source_count)
        chunk = logit_chunk(start, end)
        if not isinstance(chunk, np.ndarray):
            raise ModelValidationError("logit_chunk returned an unsupported feature container")
        if chunk.shape != (end - start, target_count):
            raise ModelValidationError("logit_chunk returned an unexpected shape")
        _require_finite_numpy(chunk, "raw edge logits")
        chunk64 = chunk.astype(np.float64, copy=False)
        log_normalizer = np.logaddexp(log_normalizer, _logsumexp_axis0(chunk64))
    if not np.isfinite(log_normalizer).all():
        raise ModelValidationError("source-softmax normalizer contains nonfinite values")

    probability_sums = np.zeros(target_count, dtype=np.float64)
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    peak_chunk_candidates = 0
    for start in range(0, source_count, source_chunk_size):
        end = min(start + source_chunk_size, source_count)
        chunk = logit_chunk(start, end)
        if not isinstance(chunk, np.ndarray) or chunk.shape != (end - start, target_count):
            raise ModelValidationError("logit_chunk changed container or shape between passes")
        _require_finite_numpy(chunk, "raw edge logits")
        probabilities = np.exp(chunk.astype(np.float64, copy=False) - log_normalizer[None, :])
        if not np.isfinite(probabilities).all():
            raise ModelValidationError("source-softmax probabilities contain nonfinite values")
        probability_sums += probabilities.sum(axis=0)
        local_source, local_target = np.nonzero(probabilities > floor)
        peak_chunk_candidates = max(peak_chunk_candidates, len(local_source))
        if len(local_source):
            source_parts.append(local_source.astype(np.int64, copy=False) + start)
            target_parts.append(local_target.astype(np.int64, copy=False))
            probability_parts.append(probabilities[local_source, local_target])

    source = np.concatenate(source_parts) if source_parts else np.empty(0, dtype=np.int64)
    target = np.concatenate(target_parts) if target_parts else np.empty(0, dtype=np.int64)
    probabilities = (
        np.concatenate(probability_parts).astype(np.float64, copy=False)
        if probability_parts
        else np.empty(0, dtype=np.float64)
    )
    if not np.allclose(probability_sums, 1.0, rtol=1e-12, atol=1e-15):
        raise ModelValidationError("source-softmax chunks changed between passes")
    cardinality_bound = target_count * int(math.ceil(1.0 / floor))
    if len(source) > cardinality_bound:
        raise ModelValidationError("edge floor cardinality bound was violated")
    stats = SourceSoftmaxStats(
        source_count=source_count,
        target_count=target_count,
        configured_source_chunk=source_chunk_size,
        peak_source_chunk=peak_source_chunk,
        candidate_count=len(source),
        peak_chunk_candidate_count=peak_chunk_candidates,
        floor=float(floor),
    )
    return SparseEdgeCOO(source, target, probabilities, stats)


def chunked_source_softmax_coo(
    raw_logits: Any,
    *,
    source_chunk_size: int,
    floor: float = 0.01,
) -> SparseEdgeCOO:
    """Fixture-friendly wrapper around :func:`source_softmax_coo`."""

    if isinstance(raw_logits, np.ndarray):
        if raw_logits.ndim != 2:
            raise ModelValidationError("raw_logits must be a two-dimensional array")
        _require_finite_numpy(raw_logits, "raw edge logits")

        def callback(start: int, end: int) -> np.ndarray:
            return np.asarray(raw_logits[start:end])

        shape = raw_logits.shape
    elif _is_torch_tensor(raw_logits):
        if raw_logits.ndim != 2:
            raise ModelValidationError("raw_logits must be a two-dimensional tensor")

        def callback(start: int, end: int) -> np.ndarray:
            return raw_logits[start:end].detach().cpu().numpy()

        shape = tuple(raw_logits.shape)
    else:
        raise ModelValidationError("raw_logits must be a NumPy array or torch tensor")
    return source_softmax_coo(
        int(shape[0]),
        int(shape[1]),
        callback,
        source_chunk_size=source_chunk_size,
        floor=floor,
    )


def _torch_sparse_source_softmax(
    torch: Any,
    source_count: int,
    target_count: int,
    callback: Callable[[int, int], Any],
    *,
    source_chunk_size: int,
    floor: float,
    device: Any,
) -> SparseEdgeCOO:
    if source_count == 0 or target_count == 0:
        stats = SourceSoftmaxStats(
            source_count,
            target_count,
            source_chunk_size,
            min(source_count, source_chunk_size) if source_count else 0,
            0,
            0,
            floor,
        )
        return SparseEdgeCOO(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            stats,
        )
    log_normalizer = torch.full((target_count,), -torch.inf, dtype=torch.float32, device=device)
    for start in range(0, source_count, source_chunk_size):
        end = min(start + source_chunk_size, source_count)
        logits = callback(start, end)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (end - start, target_count):
            raise ModelValidationError("pair scorer returned an unsupported logit chunk")
        if not bool(torch.isfinite(logits).all().item()):
            raise ModelValidationError("raw edge logits contain nonfinite values")
        log_normalizer = torch.logaddexp(log_normalizer, torch.logsumexp(logits.float(), dim=0))
    if not bool(torch.isfinite(log_normalizer).all().item()):
        raise ModelValidationError("source-softmax normalizer contains nonfinite values")

    probability_sums = torch.zeros((target_count,), dtype=torch.float64, device=device)
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    probabilities_out: list[np.ndarray] = []
    peak_chunk_candidates = 0
    for start in range(0, source_count, source_chunk_size):
        end = min(start + source_chunk_size, source_count)
        logits = callback(start, end)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (end - start, target_count):
            raise ModelValidationError("pair scorer changed its chunk container or shape")
        if not bool(torch.isfinite(logits).all().item()):
            raise ModelValidationError("raw edge logits contain nonfinite values")
        probabilities = torch.exp(logits.float() - log_normalizer[None, :])
        if not bool(torch.isfinite(probabilities).all().item()):
            raise ModelValidationError("source-softmax probabilities contain nonfinite values")
        probability_sums.add_(probabilities.double().sum(dim=0))
        indices = torch.nonzero(probabilities > floor, as_tuple=False)
        peak_chunk_candidates = max(peak_chunk_candidates, int(indices.shape[0]))
        if len(indices):
            values = probabilities[indices[:, 0], indices[:, 1]]
            sources.append(indices[:, 0].cpu().numpy().astype(np.int64, copy=False) + start)
            targets.append(indices[:, 1].cpu().numpy().astype(np.int64, copy=False))
            probabilities_out.append(values.cpu().numpy().astype(np.float64, copy=False))
    source = np.concatenate(sources) if sources else np.empty(0, dtype=np.int64)
    target = np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)
    probabilities = np.concatenate(probabilities_out) if probabilities_out else np.empty(0, dtype=np.float64)
    if not bool(torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        rtol=1e-5,
        atol=1e-7,
    )):
        raise ModelValidationError("source-softmax chunks changed between passes")
    cardinality_bound = target_count * int(math.ceil(1.0 / floor))
    if len(source) > cardinality_bound:
        raise ModelValidationError("edge floor cardinality bound was violated")
    stats = SourceSoftmaxStats(
        source_count,
        target_count,
        source_chunk_size,
        min(source_count, source_chunk_size),
        len(source),
        peak_chunk_candidates,
        floor,
    )
    return SparseEdgeCOO(source, target, probabilities, stats)


def predict_sparse_edges(
    model: Any,
    source: FrameFeatures,
    target: FrameFeatures,
    *,
    source_chunk_size: int = 32,
    floor: float = 0.01,
    diagnostic_allow_dense: bool = False,
) -> SparseEdgeCOO:
    """Diagnostic-only checkpoint scorer with chunked pair-logit emission.

    The checkpoint's cross-attention is intrinsically all-to-all even though
    pair-MLP logits are chunked.  It is therefore disabled by default and must
    never be called from production inference/cache construction.
    """

    if diagnostic_allow_dense is not True:
        raise ModelValidationError(
            "checkpoint cross-attention is diagnostic-only; "
            "set diagnostic_allow_dense=True explicitly"
        )
    if not isinstance(source, FrameFeatures) or not isinstance(target, FrameFeatures):
        raise ModelValidationError("edge features must be FrameFeatures instances")
    if target.frame_index != source.frame_index + 1:
        raise ModelValidationError("edge prediction requires an adjacent forward frame pair")
    validate_frame_sequence((source, target))
    if type(source_chunk_size) is not int or source_chunk_size <= 0:
        raise ModelValidationError("source_chunk_size must be a positive integer")
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ModelValidationError("edge floor must be finite and in (0, 1]")

    torch, _nn, _transformer_class, _unet = _require_torch_and_models()
    if getattr(model, "training", True):
        raise ModelValidationError("deterministic edge inference requires model.eval()")
    transformer = getattr(model, "transformer", None)
    required = ("proj", "norm_in", "blocks", "norm_out", "pair_mlp")
    if transformer is None or any(not hasattr(transformer, name) for name in required):
        raise ModelValidationError("model transformer does not match the public architecture")
    try:
        parameter = next(transformer.parameters())
    except (AttributeError, StopIteration) as exc:
        raise ModelValidationError("transformer has no parameters") from exc
    device = parameter.device
    dtype = parameter.dtype

    n_source = len(source.node_ids)
    n_target = len(target.node_ids)
    if n_source == 0 or n_target == 0:
        local = _torch_sparse_source_softmax(
            torch,
            n_source,
            n_target,
            lambda _start, _end: torch.empty((0, n_target), device=device),
            source_chunk_size=source_chunk_size,
            floor=float(floor),
            device=device,
        )
        return local

    if source.grid_shape_zyx != target.grid_shape_zyx:
        raise ModelValidationError("source and target feature grids must have equal shape")
    window_shape = (2,) + source.grid_shape_zyx
    source_tzyx = np.column_stack(
        (np.zeros(n_source, dtype=np.int64), source.grid_coords_zyx)
    )
    target_tzyx = np.column_stack(
        (np.ones(n_target, dtype=np.int64), target.grid_coords_zyx)
    )
    source_pos = positional_features(source_tzyx, window_shape)
    target_pos = positional_features(target_tzyx, window_shape)
    source_input = np.concatenate((source.embeddings, source_pos), axis=1)
    target_input = np.concatenate((target.embeddings, target_pos), axis=1)
    if source_input.shape[1] != 64 or target_input.shape[1] != 64:
        raise ModelValidationError("public transformer requires 64 input features per node")

    with torch.inference_mode():
        source_features = torch.from_numpy(np.ascontiguousarray(source_input)).to(device=device, dtype=dtype)[None]
        target_features = torch.from_numpy(np.ascontiguousarray(target_input)).to(device=device, dtype=dtype)[None]
        source_coords = torch.from_numpy(np.ascontiguousarray(source.coords_zyx, dtype=np.float32)).to(
            device=device, dtype=dtype
        )
        target_coords = torch.from_numpy(np.ascontiguousarray(target.coords_zyx, dtype=np.float32)).to(
            device=device, dtype=dtype
        )
        source_mask = torch.ones((1, n_source), dtype=torch.bool, device=device)
        target_mask = torch.ones((1, n_target), dtype=torch.bool, device=device)
        q = transformer.norm_in(transformer.proj(source_features))
        k = transformer.norm_in(transformer.proj(target_features))
        for block in transformer.blocks:
            q = block(q, k, kv_mask=target_mask)
            k = block(k, q, kv_mask=source_mask)
        q = transformer.norm_out(q)[0]
        k = transformer.norm_out(k)[0]
        if not bool(torch.isfinite(q).all().item()) or not bool(torch.isfinite(k).all().item()):
            raise ModelValidationError("contextualized node features contain nonfinite values")

        def pair_chunk(start: int, end: int) -> Any:
            q_chunk = q[start:end]
            coords_chunk = source_coords[start:end]
            count = end - start
            query = q_chunk[:, None, :].expand(count, n_target, -1)
            key = k[None, :, :].expand(count, n_target, -1)
            relative = (coords_chunk[:, None, :] - target_coords[None, :, :]) / 100.0
            return transformer.pair_mlp(torch.cat((query, key, relative), dim=-1)).squeeze(-1)

        local = _torch_sparse_source_softmax(
            torch,
            n_source,
            n_target,
            pair_chunk,
            source_chunk_size=source_chunk_size,
            floor=float(floor),
            device=device,
        )
    remapped_source = source.node_ids[local.source]
    remapped_target = target.node_ids[local.target]
    return SparseEdgeCOO(remapped_source, remapped_target, local.probabilities, local.stats)


def validate_frame_sequence(frames_value: Sequence[FrameFeatures | FrameDet]) -> None:
    """Reject duplicate IDs and non-increasing frame containers."""

    if not isinstance(frames_value, (tuple, list)) or not frames_value:
        raise ModelValidationError("frames must be a non-empty tuple or list")
    seen_ids: set[int] = set()
    previous_frame: int | None = None
    for frame in frames_value:
        if not isinstance(frame, (FrameFeatures, FrameDet)):
            raise ModelValidationError("frames contain an unsupported feature container")
        if previous_frame is not None and frame.frame_index <= previous_frame:
            raise ModelValidationError("frame containers must be strictly forward ordered")
        previous_frame = frame.frame_index
        for raw_id in frame.node_ids:
            node_id = int(raw_id)
            if node_id in seen_ids:
                raise ModelValidationError(f"duplicate node ID across frames: {node_id}")
            seen_ids.add(node_id)


def validate_sparse_edges(
    frames_value: Sequence[FrameFeatures | FrameDet],
    edges: SparseEdgeCOO,
) -> None:
    """Reject unknown, duplicate, backward, and non-adjacent cached edge pairs."""

    validate_frame_sequence(frames_value)
    if not isinstance(edges, SparseEdgeCOO):
        raise ModelValidationError("edges must use the SparseEdgeCOO container")
    frame_by_id = {
        int(node_id): frame.frame_index
        for frame in frames_value
        for node_id in frame.node_ids
    }
    seen_pairs: set[tuple[int, int]] = set()
    for raw_source, raw_target in zip(edges.source, edges.target):
        source_id, target_id = int(raw_source), int(raw_target)
        if source_id not in frame_by_id or target_id not in frame_by_id:
            raise ModelValidationError("edge references an unknown node ID")
        pair = (source_id, target_id)
        if pair in seen_pairs:
            raise ModelValidationError(f"duplicate edge pair: {pair}")
        seen_pairs.add(pair)
        source_frame = frame_by_id[source_id]
        target_frame = frame_by_id[target_id]
        if target_frame <= source_frame:
            raise ModelValidationError(f"backward or same-frame edge pair: {pair}")
        if target_frame != source_frame + 1:
            raise ModelValidationError(f"non-adjacent edge pair: {pair}")


def _validate_video_records(
    detections: Sequence[FrameDet],
    contexts: Sequence[PairContext],
) -> None:
    if not isinstance(detections, (tuple, list)) or len(detections) < 2:
        raise ModelValidationError("canonical detections require at least two ordered frames")
    if not all(isinstance(item, FrameDet) for item in detections):
        raise ModelValidationError("canonical detections contain an unsupported record")
    validate_frame_sequence(detections)
    for previous, current in zip(detections, detections[1:]):
        if current.frame_index != previous.frame_index + 1:
            raise ModelValidationError("canonical detection frames must be consecutive")
    if not isinstance(contexts, (tuple, list)) or len(contexts) != len(detections) - 1:
        raise ModelValidationError("pair contexts must cover every adjacent detection pair exactly once")
    for index, context in enumerate(contexts):
        if not isinstance(context, PairContext):
            raise ModelValidationError("pair contexts contain an unsupported record")
        source = detections[index]
        target = detections[index + 1]
        if context.frame_pair != (source.frame_index, target.frame_index):
            raise ModelValidationError("pair contexts are not in canonical frame order")
        if context.grid_shape_zyx != source.grid_shape_zyx or context.grid_shape_zyx != target.grid_shape_zyx:
            raise ModelValidationError("pair context grid shape disagrees with canonical detections")
        if not np.array_equal(context.source_ids, source.node_ids):
            raise ModelValidationError("pair source IDs disagree with the canonical registry")
        if not np.array_equal(context.target_ids, target.node_ids):
            raise ModelValidationError("pair target IDs disagree with the canonical registry")


def _validate_pair_edges(
    detections: Sequence[FrameDet],
    contexts: Sequence[PairContext],
    pair_edges: Sequence[PairEdges],
) -> None:
    if not isinstance(pair_edges, (tuple, list)):
        raise ModelValidationError("pair edges must be an ordered tuple or list")
    by_pair = {context.frame_pair: context for context in contexts}
    det_by_frame = {frame.frame_index: frame for frame in detections}
    seen: set[tuple[int, int]] = set()
    for item in pair_edges:
        if not isinstance(item, PairEdges):
            raise ModelValidationError("pair edges contain an unsupported record")
        pair = item.frame_pair
        if pair in seen:
            raise ModelValidationError(f"duplicate pair-scoped edge record: {pair}")
        seen.add(pair)
        if pair not in by_pair:
            raise ModelValidationError(f"pair edges have no matching context: {pair}")
        source, target = det_by_frame[pair[0]], det_by_frame[pair[1]]
        validate_sparse_edges((source, target), item.edges)
        stats = item.edges.stats
        if stats.source_count != len(source.node_ids) or stats.target_count != len(target.node_ids):
            raise ModelValidationError("pair softmax stats disagree with canonical node counts")


@dataclass(frozen=True)
class CacheIdentity:
    config_sha256: str
    source_sha256: str
    weight_sha256: str
    pair_batch_size: int = 1
    view_batch_size: int = 1

    def __post_init__(self) -> None:
        for name in ("config_sha256", "source_sha256", "weight_sha256"):
            _validate_sha256(getattr(self, name), name)
        if type(self.pair_batch_size) is not int or self.pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be a positive integer")
        if type(self.view_batch_size) is not int or not 1 <= self.view_batch_size <= len(D4_VIEWS):
            raise ValueError("view_batch_size must be an integer in [1, 8]")

    @classmethod
    def from_artifact(
        cls,
        bundle: ArtifactBundle,
        *,
        pair_batch_size: int = 1,
        view_batch_size: int = 1,
        d4_views: Sequence[int] | None = None,
    ) -> "CacheIdentity":
        if not isinstance(bundle, ArtifactBundle):
            raise CacheValidationError("cache identity requires a validated ArtifactBundle")
        try:
            config_sha256 = v45_config_sha(
                pair_batch_size=pair_batch_size,
                view_batch_size=view_batch_size,
                d4_views=d4_views,
            )
        except ValueError as exc:
            raise CacheValidationError(f"cache batching identity is invalid: {exc}") from exc
        return cls(
            config_sha256,
            bundle.hashes.source_sha256,
            bundle.hashes.weight_sha256,
            pair_batch_size,
            view_batch_size,
        )


@dataclass(frozen=True)
class SparseInferenceCache:
    identity: CacheIdentity
    detections: tuple[FrameDet, ...]
    contexts: tuple[PairContext, ...]
    edges: tuple[PairEdges, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CacheIdentity):
            raise CacheValidationError("cache identity has an unsupported container")
        if not isinstance(self.detections, tuple):
            raise CacheValidationError("cache detections must be an ordered tuple")
        if not isinstance(self.contexts, tuple):
            raise CacheValidationError("cache contexts must be an ordered tuple")
        if not isinstance(self.edges, tuple):
            raise CacheValidationError("cache edges must be pair-scoped ordered records")
        try:
            _validate_video_records(self.detections, self.contexts)
            _validate_pair_edges(self.detections, self.contexts, self.edges)
        except ModelValidationError as exc:
            raise CacheValidationError(str(exc)) from exc

    @property
    def frames(self) -> tuple[FrameDet, ...]:
        """Compatibility alias for canonical detection records."""

        return self.detections


def _pack_array(value: np.ndarray, dtype: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    return {
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        "dtype": np.dtype(dtype).str,
        "shape": list(array.shape),
    }


def _unpack_array(value: Any, dtype: str, label: str) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != {"data", "dtype", "shape"}:
        raise CacheValidationError(f"{label} has an invalid array record")
    wanted = np.dtype(dtype)
    if value["dtype"] != wanted.str:
        raise CacheValidationError(f"{label} has an unexpected dtype")
    shape = value["shape"]
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
        raise CacheValidationError(f"{label} has an invalid shape")
    try:
        raw = base64.b64decode(value["data"].encode("ascii"), validate=True)
    except (AttributeError, UnicodeError, ValueError) as exc:
        raise CacheValidationError(f"{label} has invalid base64 data") from exc
    expected_bytes = math.prod(shape) * wanted.itemsize
    if len(raw) != expected_bytes:
        raise CacheValidationError(f"{label} byte count does not match its shape")
    result = np.frombuffer(raw, dtype=wanted).reshape(tuple(shape)).copy()
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise CacheValidationError(f"{label} contains nonfinite values")
    return result


def _cache_record(cache: SparseInferenceCache) -> dict[str, Any]:
    detection_records = []
    for frame in cache.detections:
        detection_records.append(
            {
                "frame_index": frame.frame_index,
                "grid_coords_zyx": _pack_array(frame.grid_coords_zyx, "<i8"),
                "grid_shape_zyx": list(frame.grid_shape_zyx),
                "node_ids": _pack_array(frame.node_ids, "<i8"),
                "scores": _pack_array(frame.scores, "<f4"),
                "voxel_coords_zyx": _pack_array(frame.coords_zyx, "<i8"),
            }
        )
    context_records = [
        {
            "grid_shape_zyx": list(context.grid_shape_zyx),
            "source_features": _pack_array(context.source_feat, "<f4"),
            "source_frame": context.source_frame,
            "source_ids": _pack_array(context.source_ids, "<i8"),
            "target_features": _pack_array(context.target_feat, "<f4"),
            "target_frame": context.target_frame,
            "target_ids": _pack_array(context.target_ids, "<i8"),
        }
        for context in cache.contexts
    ]
    edge_records = []
    for pair in cache.edges:
        stats = pair.edges.stats
        edge_records.append(
            {
                "probabilities": _pack_array(pair.edges.probabilities, "<f8"),
                "source_frame": pair.source_frame,
                "source_ids": _pack_array(pair.edges.source, "<i8"),
                "stats": {
                    "candidate_count": stats.candidate_count,
                    "configured_source_chunk": stats.configured_source_chunk,
                    "floor": stats.floor,
                    "peak_chunk_candidate_count": stats.peak_chunk_candidate_count,
                    "peak_source_chunk": stats.peak_source_chunk,
                    "source_count": stats.source_count,
                    "target_count": stats.target_count,
                },
                "target_frame": pair.target_frame,
                "target_ids": _pack_array(pair.edges.target, "<i8"),
            }
        )
    return {
        "contexts": context_records,
        "detections": detection_records,
        "edges": edge_records,
        "hashes": {
            "config_sha256": cache.identity.config_sha256,
            "pair_batch_size": cache.identity.pair_batch_size,
            "source_sha256": cache.identity.source_sha256,
            "view_batch_size": cache.identity.view_batch_size,
            "weight_sha256": cache.identity.weight_sha256,
        },
        "schema": CACHE_SCHEMA,
    }


def _cache_bytes(cache: SparseInferenceCache) -> bytes:
    body = _canonical_json(_cache_record(cache))
    prefix = CACHE_MAGIC + len(body).to_bytes(8, "big") + body
    return prefix + hashlib.sha256(prefix).digest()


def save_sparse_cache(path: str | os.PathLike[str], cache: SparseInferenceCache) -> None:
    """Atomically write a deterministic, self-digesting sparse cache."""

    if not isinstance(cache, SparseInferenceCache):
        raise CacheValidationError("save_sparse_cache requires SparseInferenceCache")
    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir():
        raise CacheValidationError(f"cache parent directory does not exist: {parent}")
    payload = _cache_bytes(cache)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=parent)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise CacheValidationError(f"atomic cache write failed: {exc}") from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def load_sparse_cache(
    path: str | os.PathLike[str],
    *,
    expected_identity: CacheIdentity,
) -> SparseInferenceCache:
    """Load a cache only when its digest, schema, and all model hashes match."""

    if not isinstance(expected_identity, CacheIdentity):
        raise CacheValidationError("expected_identity must be a CacheIdentity")
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise CacheValidationError(f"cannot read sparse cache: {exc}") from exc
    minimum = len(CACHE_MAGIC) + 8 + hashlib.sha256().digest_size
    if len(payload) < minimum or payload[: len(CACHE_MAGIC)] != CACHE_MAGIC:
        raise CacheValidationError("cache magic or length is invalid")
    body_length = int.from_bytes(payload[len(CACHE_MAGIC) : len(CACHE_MAGIC) + 8], "big")
    body_start = len(CACHE_MAGIC) + 8
    body_end = body_start + body_length
    if body_end + 32 != len(payload):
        raise CacheValidationError("cache body length is invalid")
    prefix, stored_digest = payload[:body_end], payload[body_end:]
    if not hmac.compare_digest(hashlib.sha256(prefix).digest(), stored_digest):
        raise CacheValidationError("cache content digest mismatch")
    try:
        record = json.loads(
            payload[body_start:body_end].decode("ascii"),
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CacheValidationError(f"cache JSON is invalid: {exc}") from exc
    if not isinstance(record, dict) or set(record) != {
        "contexts",
        "detections",
        "edges",
        "hashes",
        "schema",
    }:
        raise CacheValidationError("cache top-level schema has missing or unexpected fields")
    if record["schema"] != CACHE_SCHEMA:
        raise CacheValidationError("cache schema mismatch")
    hashes = record["hashes"]
    if not isinstance(hashes, dict) or set(hashes) != {
        "config_sha256",
        "pair_batch_size",
        "source_sha256",
        "view_batch_size",
        "weight_sha256",
    }:
        raise CacheValidationError("cache hashes record is invalid")
    try:
        identity = CacheIdentity(**hashes)
    except (TypeError, ValueError) as exc:
        raise CacheValidationError(f"cache hashes are invalid: {exc}") from exc
    if identity != expected_identity:
        raise CacheValidationError("cache model/config/source/batching identity mismatch")

    detection_records = record["detections"]
    if not isinstance(detection_records, list) or len(detection_records) < 2:
        raise CacheValidationError("cache detections must contain at least two ordered frames")
    detections_out: list[FrameDet] = []
    detection_keys = {
        "frame_index",
        "grid_coords_zyx",
        "grid_shape_zyx",
        "node_ids",
        "scores",
        "voxel_coords_zyx",
    }
    for index, detection_record in enumerate(detection_records):
        if not isinstance(detection_record, dict) or set(detection_record) != detection_keys:
            raise CacheValidationError(f"cache detection {index} has missing or unexpected fields")
        grid_shape_raw = detection_record["grid_shape_zyx"]
        if (
            not isinstance(grid_shape_raw, list)
            or len(grid_shape_raw) != 3
            or any(type(value) is not int for value in grid_shape_raw)
        ):
            raise CacheValidationError(f"cache detection {index} has an invalid grid shape")
        try:
            detection = FrameDet(
                frame_index=detection_record["frame_index"],
                node_ids=_unpack_array(
                    detection_record["node_ids"], "<i8", f"detection {index} node IDs"
                ),
                coords_zyx=_unpack_array(
                    detection_record["voxel_coords_zyx"],
                    "<i8",
                    f"detection {index} voxel coordinates",
                ),
                scores=_unpack_array(
                    detection_record["scores"], "<f4", f"detection {index} scores"
                ),
                grid_coords_zyx=_unpack_array(
                    detection_record["grid_coords_zyx"],
                    "<i8",
                    f"detection {index} grid coordinates",
                ),
                grid_shape_zyx=tuple(grid_shape_raw),
            )
        except ModelValidationError as exc:
            raise CacheValidationError(f"cache detection {index} is invalid: {exc}") from exc
        detections_out.append(detection)

    context_records = record["contexts"]
    if not isinstance(context_records, list):
        raise CacheValidationError("cache contexts must be an ordered list")
    context_keys = {
        "grid_shape_zyx",
        "source_features",
        "source_frame",
        "source_ids",
        "target_features",
        "target_frame",
        "target_ids",
    }
    contexts_out: list[PairContext] = []
    for index, context_record in enumerate(context_records):
        if not isinstance(context_record, dict) or set(context_record) != context_keys:
            raise CacheValidationError(f"cache context {index} has missing or unexpected fields")
        grid_shape_raw = context_record["grid_shape_zyx"]
        if (
            not isinstance(grid_shape_raw, list)
            or len(grid_shape_raw) != 3
            or any(type(value) is not int for value in grid_shape_raw)
        ):
            raise CacheValidationError(f"cache context {index} has an invalid grid shape")
        try:
            contexts_out.append(
                PairContext(
                    source_frame=context_record["source_frame"],
                    target_frame=context_record["target_frame"],
                    source_ids=_unpack_array(
                        context_record["source_ids"], "<i8", f"context {index} source IDs"
                    ),
                    target_ids=_unpack_array(
                        context_record["target_ids"], "<i8", f"context {index} target IDs"
                    ),
                    source_feat=_unpack_array(
                        context_record["source_features"],
                        "<f4",
                        f"context {index} source features",
                    ),
                    target_feat=_unpack_array(
                        context_record["target_features"],
                        "<f4",
                        f"context {index} target features",
                    ),
                    grid_shape_zyx=tuple(grid_shape_raw),
                )
            )
        except ModelValidationError as exc:
            raise CacheValidationError(f"cache context {index} is invalid: {exc}") from exc

    edge_records = record["edges"]
    if not isinstance(edge_records, list):
        raise CacheValidationError("cache edges must be pair-scoped ordered records")
    edge_keys = {
        "probabilities",
        "source_frame",
        "source_ids",
        "stats",
        "target_frame",
        "target_ids",
    }
    stats_keys = {item.name for item in fields(SourceSoftmaxStats)}
    edges_out: list[PairEdges] = []
    for index, edge_record in enumerate(edge_records):
        if not isinstance(edge_record, dict) or set(edge_record) != edge_keys:
            raise CacheValidationError(f"cache pair edge {index} has missing or unexpected fields")
        stats_record = edge_record["stats"]
        if not isinstance(stats_record, dict) or set(stats_record) != stats_keys:
            raise CacheValidationError(f"cache pair edge {index} stats are invalid")
        try:
            edges_out.append(
                PairEdges(
                    source_frame=edge_record["source_frame"],
                    target_frame=edge_record["target_frame"],
                    edges=SparseEdgeCOO(
                        _unpack_array(
                            edge_record["source_ids"], "<i8", f"pair edge {index} source IDs"
                        ),
                        _unpack_array(
                            edge_record["target_ids"], "<i8", f"pair edge {index} target IDs"
                        ),
                        _unpack_array(
                            edge_record["probabilities"],
                            "<f8",
                            f"pair edge {index} probabilities",
                        ),
                        SourceSoftmaxStats(**stats_record),
                    ),
                )
            )
        except (ModelValidationError, TypeError) as exc:
            raise CacheValidationError(f"cache pair edge {index} is invalid: {exc}") from exc
    return SparseInferenceCache(
        identity,
        tuple(detections_out),
        tuple(contexts_out),
        tuple(edges_out),
    )


def load_sparse_cache_detections(
    path: str | os.PathLike[str],
    *,
    expected_identity: CacheIdentity,
) -> tuple[FrameDet, ...]:
    """Authenticate a cache and load only its canonical detections.

    The cache body can contain large pair-context tensors that are irrelevant
    when callers only need proposal nodes.  This reader verifies the
    complete self-digest and exact model/source/batching identity, but parses
    only the canonical detection records.  Peak memory therefore scales with
    the detection registry rather than the full cache.
    """

    if not isinstance(expected_identity, CacheIdentity):
        raise CacheValidationError("expected_identity must be a CacheIdentity")
    source = Path(path)
    minimum = len(CACHE_MAGIC) + 8 + hashlib.sha256().digest_size
    try:
        with source.open("rb") as handle:
            header = handle.read(len(CACHE_MAGIC) + 8)
            if (
                len(header) != len(CACHE_MAGIC) + 8
                or header[: len(CACHE_MAGIC)] != CACHE_MAGIC
            ):
                raise CacheValidationError("cache magic or length is invalid")
            body_length = int.from_bytes(header[len(CACHE_MAGIC) :], "big")
            body_start = len(CACHE_MAGIC) + 8
            body_end = body_start + body_length
            file_size = os.fstat(handle.fileno()).st_size
            if file_size < minimum or body_end + 32 != file_size:
                raise CacheValidationError("cache body length is invalid")

            handle.seek(0)
            digest = hashlib.sha256()
            remaining = body_end
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise CacheValidationError("cache body length is invalid")
                digest.update(chunk)
                remaining -= len(chunk)
            stored_digest = handle.read(32)
            if not hmac.compare_digest(digest.digest(), stored_digest):
                raise CacheValidationError("cache content digest mismatch")

            with mmap.mmap(
                handle.fileno(), length=0, access=mmap.ACCESS_READ
            ) as payload:
                contexts_prefix = b'{"contexts":'
                detections_tag = b',"detections":'
                edges_tag = b',"edges":'
                hashes_tag = b',"hashes":'
                schema_tag = b',"schema":'
                if (
                    payload[body_start : body_start + len(contexts_prefix)]
                    != contexts_prefix
                    or payload[body_end - 1 : body_end] != b"}"
                ):
                    raise CacheValidationError(
                        "cache top-level schema has missing or unexpected fields"
                    )
                positions = [
                    payload.find(tag, body_start, body_end)
                    for tag in (
                        detections_tag,
                        edges_tag,
                        hashes_tag,
                        schema_tag,
                    )
                ]
                if (
                    any(position < 0 for position in positions)
                    or positions != sorted(positions)
                    or any(
                        payload.find(tag, position + len(tag), body_end) >= 0
                        for tag, position in zip(
                            (
                                detections_tag,
                                edges_tag,
                                hashes_tag,
                                schema_tag,
                            ),
                            positions,
                        )
                    )
                ):
                    raise CacheValidationError(
                        "cache top-level schema has missing or unexpected fields"
                    )
                detections_at, edges_at, hashes_at, schema_at = positions
                try:
                    detection_records = json.loads(
                        payload[
                            detections_at + len(detections_tag) : edges_at
                        ].decode("ascii"),
                        object_pairs_hook=_reject_duplicate_json_pairs,
                    )
                    hashes = json.loads(
                        payload[
                            hashes_at + len(hashes_tag) : schema_at
                        ].decode("ascii"),
                        object_pairs_hook=_reject_duplicate_json_pairs,
                    )
                    schema = json.loads(
                        payload[
                            schema_at + len(schema_tag) : body_end - 1
                        ].decode("ascii"),
                    )
                except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
                    raise CacheValidationError(
                        f"cache JSON is invalid: {exc}"
                    ) from exc
    except CacheValidationError:
        raise
    except OSError as exc:
        raise CacheValidationError(f"cannot read sparse cache: {exc}") from exc

    if schema != CACHE_SCHEMA:
        raise CacheValidationError("cache schema mismatch")
    if not isinstance(hashes, dict) or set(hashes) != {
        "config_sha256",
        "pair_batch_size",
        "source_sha256",
        "view_batch_size",
        "weight_sha256",
    }:
        raise CacheValidationError("cache hashes record is invalid")
    try:
        identity = CacheIdentity(**hashes)
    except (TypeError, ValueError) as exc:
        raise CacheValidationError(f"cache hashes are invalid: {exc}") from exc
    if identity != expected_identity:
        raise CacheValidationError(
            "cache model/config/source/batching identity mismatch"
        )

    if not isinstance(detection_records, list) or len(detection_records) < 2:
        raise CacheValidationError(
            "cache detections must contain at least two ordered frames"
        )
    detection_keys = {
        "frame_index",
        "grid_coords_zyx",
        "grid_shape_zyx",
        "node_ids",
        "scores",
        "voxel_coords_zyx",
    }
    detections: list[FrameDet] = []
    for index, detection_record in enumerate(detection_records):
        if (
            not isinstance(detection_record, dict)
            or set(detection_record) != detection_keys
        ):
            raise CacheValidationError(
                f"cache detection {index} has missing or unexpected fields"
            )
        grid_shape_raw = detection_record["grid_shape_zyx"]
        if (
            not isinstance(grid_shape_raw, list)
            or len(grid_shape_raw) != 3
            or any(type(value) is not int for value in grid_shape_raw)
        ):
            raise CacheValidationError(
                f"cache detection {index} has an invalid grid shape"
            )
        try:
            detections.append(
                FrameDet(
                    frame_index=detection_record["frame_index"],
                    node_ids=_unpack_array(
                        detection_record["node_ids"],
                        "<i8",
                        f"detection {index} node IDs",
                    ),
                    coords_zyx=_unpack_array(
                        detection_record["voxel_coords_zyx"],
                        "<i8",
                        f"detection {index} voxel coordinates",
                    ),
                    scores=_unpack_array(
                        detection_record["scores"],
                        "<f4",
                        f"detection {index} scores",
                    ),
                    grid_coords_zyx=_unpack_array(
                        detection_record["grid_coords_zyx"],
                        "<i8",
                        f"detection {index} grid coordinates",
                    ),
                    grid_shape_zyx=tuple(grid_shape_raw),
                )
            )
        except ModelValidationError as exc:
            raise CacheValidationError(
                f"cache detection {index} is invalid: {exc}"
            ) from exc
    try:
        validate_frame_sequence(detections)
    except ModelValidationError as exc:
        raise CacheValidationError(str(exc)) from exc
    for previous, current in zip(detections, detections[1:]):
        if current.frame_index != previous.frame_index + 1:
            raise CacheValidationError(
                "canonical detection frames must be consecutive"
            )
    return tuple(detections)


def load_sparse_cache_node_times(
    path: str | os.PathLike[str],
    *,
    expected_identity: CacheIdentity,
) -> dict[int, int]:
    """Authenticate a cache and return its low-memory node-to-frame registry."""

    detections = load_sparse_cache_detections(
        path, expected_identity=expected_identity
    )
    return {
        int(node_id): detection.frame_index
        for detection in detections
        for node_id in detection.node_ids
    }


__all__ = [
    "ArtifactBundle",
    "ArtifactHashes",
    "ArtifactValidationError",
    "BHModelError",
    "CACHE_SCHEMA",
    "CacheIdentity",
    "CacheValidationError",
    "DEFAULT_WEIGHT_SHA256",
    "D4_VIEWS",
    "DependencyUnavailableError",
    "FrameDet",
    "FrameFeatures",
    "MODEL_SCHEMA",
    "ModelValidationError",
    "PairContext",
    "PairEdges",
    "SourceSoftmaxStats",
    "SparseEdgeCOO",
    "SparseInferenceCache",
    "TwoFrameInference",
    "V45_CONFIG",
    "V45_CONFIG_SHA256",
    "V45Config",
    "VoxelNode",
    "VideoInference",
    "build_v45_model",
    "canonical_sha256",
    "chunked_source_softmax_coo",
    "d4_inverse",
    "d4_invert_and_average",
    "d4_transform",
    "dedup_dets",
    "deduplicate_frame_detections",
    "infer_two_frames",
    "infer_video",
    "load_sparse_cache",
    "load_sparse_cache_detections",
    "load_sparse_cache_node_times",
    "load_v45_model",
    "positional_features",
    "predict_sparse_edges",
    "quantile_normalize",
    "save_sparse_cache",
    "sha256_file",
    "source_softmax_coo",
    "source_tree_sha256",
    "validate_artifact_bundle",
    "validate_frame_sequence",
    "validate_sparse_edges",
    "v45_config_sha",
]
