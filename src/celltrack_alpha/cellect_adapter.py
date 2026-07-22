"""GPL-2.0 CELLECT embedding adapter for the Biohub point tracker.

This file is a derivative integration of CELLECT's published model interface.
It is distributed under GPL-2.0-only; see
``third_party_licenses/CELLECT_GPL-2.0.txt`` and
``third_party_licenses/CELLECT_NOTICE.md``.

The production detector and graph solver are deliberately not imported here.
This module exposes an experimental, opt-in edge embedding only.  Division
logits are never emitted by this adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


UPSTREAM_COMMIT = "3586070926f7f1fd5d8df37456861d22bdc63236"
UPSTREAM_LICENSE_SHA256 = "8177f97513213526df2cf6184d8ff986c675afb514d4e68a404010521b880643"
UPSTREAM_MODEL_SOURCE_SHA256 = "3b44f7c64c4b948ab7cc14bb32d5a86f61d3f4cff0ee3425a5ec32cc6c17d3ba"
UPSTREAM_WEIGHT_SHA256 = "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5"
UPSTREAM_MODEL_FILE = "U-ext+-x3rd-149.0-4.6540.pth"
EMBEDDING_CHANNELS = 64
TRIPLET_MARGIN = 0.3


class CellectContractError(RuntimeError):
    """Raised when a frozen upstream or experiment invariant changes."""


@dataclass(frozen=True)
class CellectAdapterConfig:
    """Frozen defaults for the first fit117/CV57 CELLECT experiment."""

    crop_zyx: tuple[int, int, int] = (16, 64, 64)
    embedding_channels: int = EMBEDDING_CHANNELS
    triplet_margin: float = TRIPLET_MARGIN
    candidate_top_k: int = 10
    same_frame_weight: float = 0.1
    division_output_enabled: bool = False

    def __post_init__(self) -> None:
        if self.crop_zyx[1] % 16 or self.crop_zyx[2] % 16:
            raise CellectContractError("CELLECT lateral crop dimensions must be multiples of 16")
        if any(int(value) <= 0 for value in self.crop_zyx):
            raise CellectContractError("CELLECT crop dimensions must be positive")
        if self.embedding_channels != EMBEDDING_CHANNELS:
            raise CellectContractError("the pinned CELLECT embedding head has 64 channels")
        if self.triplet_margin <= 0.0 or not math.isfinite(self.triplet_margin):
            raise CellectContractError("triplet margin must be finite and positive")
        if self.candidate_top_k <= 0:
            raise CellectContractError("candidate_top_k must be positive")
        if self.division_output_enabled:
            raise CellectContractError("division output remains disabled until patched-metric TP proof")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_upstream_tree(upstream_root: Path) -> dict[str, str]:
    """Validate the exact cloned source, license, and published checkpoint."""

    root = upstream_root.resolve()
    model_source = root / "unetext3Dn_con7.py"
    license_path = root / "LICENSE"
    weight_path = root / "model" / UPSTREAM_MODEL_FILE
    for path in (model_source, license_path, weight_path):
        if not path.is_file():
            raise CellectContractError(f"missing pinned CELLECT file: {path}")
    observed_license = sha256_file(license_path)
    observed_source = sha256_file(model_source)
    observed_weight = sha256_file(weight_path)
    if observed_license != UPSTREAM_LICENSE_SHA256:
        raise CellectContractError(
            f"CELLECT license hash changed: {observed_license} != {UPSTREAM_LICENSE_SHA256}"
        )
    if observed_weight != UPSTREAM_WEIGHT_SHA256:
        raise CellectContractError(
            f"CELLECT checkpoint hash changed: {observed_weight} != {UPSTREAM_WEIGHT_SHA256}"
        )
    if observed_source != UPSTREAM_MODEL_SOURCE_SHA256:
        raise CellectContractError(
            f"CELLECT model source hash changed: {observed_source} != {UPSTREAM_MODEL_SOURCE_SHA256}"
        )
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "license": str(license_path),
        "license_sha256": observed_license,
        "model_source": str(model_source),
        "model_source_sha256": observed_source,
        "weight": str(weight_path),
        "weight_sha256": observed_weight,
    }


def _load_upstream_module(upstream_root: Path) -> ModuleType:
    source = upstream_root.resolve() / "unetext3Dn_con7.py"
    spec = importlib.util.spec_from_file_location("celltrack_cellect_upstream_model", source)
    if spec is None or spec.loader is None:
        raise CellectContractError(f"cannot import CELLECT model source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_published_backbone(upstream_root: Path, *, device: torch.device | str = "cpu") -> nn.Module:
    """Instantiate the exact upstream architecture and published checkpoint."""

    provenance = validate_upstream_tree(upstream_root)
    module = _load_upstream_module(upstream_root)
    model = module.UNet3D(2, 6)
    try:
        state = torch.load(provenance["weight"], map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility for the pre-staged Kaggle runtime
        state = torch.load(provenance["weight"], map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device)


def cellect_log_preprocess(volume: torch.Tensor) -> torch.Tensor:
    """Apply CELLECT's published log preprocessing without in-place mutation."""

    if volume.ndim != 5 or volume.shape[1] != 2:
        raise CellectContractError("CELLECT input must have shape (B,2,Y,X,Z)")
    if not volume.is_floating_point():
        volume = volume.float()
    positive = volume > 0
    if not bool(positive.any().item()):
        raise CellectContractError("CELLECT input contains no positive intensity")
    # Upstream uses one minimum over the input tensor, then adds 1900 and log1p.
    minimum = volume[positive].amin()
    clamped = torch.where(volume < minimum, minimum, volume)
    return torch.log1p(clamped + 1900.0)


class CellectEmbeddingAdapter(nn.Module):
    """Expose CELLECT's exact 64-channel head as normalized edge embeddings."""

    def __init__(self, backbone: nn.Module, config: CellectAdapterConfig | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or CellectAdapterConfig()

    def forward(self, paired_yxz: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(cellect_log_preprocess(paired_yxz))
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
            raise CellectContractError("upstream CELLECT forward contract changed")
        feature_map = outputs[2]
        if feature_map.ndim != 5 or feature_map.shape[1] != self.config.embedding_channels:
            raise CellectContractError("upstream CELLECT embedding shape changed")
        if not bool(torch.isfinite(feature_map).all().item()):
            raise CellectContractError("CELLECT produced non-finite embeddings")
        return F.normalize(feature_map, p=2.0, dim=1, eps=1e-8)


def sample_embeddings_zyx(feature_map: torch.Tensor, points_zyx: torch.Tensor) -> torch.Tensor:
    """Index a batch-one ``(B,C,Y,X,Z)`` map at integer ``(z,y,x)`` points."""

    if feature_map.ndim != 5 or feature_map.shape[0] != 1:
        raise CellectContractError("center sampling currently requires one feature volume")
    if points_zyx.ndim != 2 or points_zyx.shape[1] != 3:
        raise CellectContractError("points must have shape (N,3) in z,y,x order")
    points = points_zyx.to(device=feature_map.device, dtype=torch.long)
    z, y, x = points.unbind(dim=1)
    if len(points) and (
        bool((z < 0).any().item())
        or bool((y < 0).any().item())
        or bool((x < 0).any().item())
        or bool((y >= feature_map.shape[2]).any().item())
        or bool((x >= feature_map.shape[3]).any().item())
        or bool((z >= feature_map.shape[4]).any().item())
    ):
        raise CellectContractError("sample point lies outside the CELLECT feature volume")
    return feature_map[0, :, y, x, z].transpose(0, 1).contiguous()


def squared_embedding_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise CellectContractError("embedding matrices have incompatible shapes")
    return torch.sum((source[:, None, :] - target[None, :, :]) ** 2, dim=2)


def topk_candidate_mask(
    source_zyx: np.ndarray,
    target_zyx: np.ndarray,
    scale_zyx: Sequence[float],
    *,
    k: int,
) -> np.ndarray:
    """Return CELLECT-style top-k physical-neighbour candidates per source."""

    source = np.asarray(source_zyx, dtype=np.float64)
    target = np.asarray(target_zyx, dtype=np.float64)
    scale = np.asarray(tuple(scale_zyx), dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or target.ndim != 2 or target.shape[1:] != (3,):
        raise CellectContractError("candidate coordinates must have shape (N,3)")
    if scale.shape != (3,) or np.any(scale <= 0) or not np.isfinite(scale).all():
        raise CellectContractError("physical scale must contain three finite positive values")
    if type(k) is not int or k <= 0:
        raise CellectContractError("candidate k must be a positive integer")
    mask = np.zeros((len(source), len(target)), dtype=bool)
    if not len(source) or not len(target):
        return mask
    distances = np.linalg.norm((source[:, None, :] - target[None, :, :]) * scale, axis=2)
    for row in range(len(source)):
        order = np.lexsort((np.arange(len(target), dtype=np.int64), distances[row]))
        mask[row, order[: min(k, len(target))]] = True
    return mask


def batch_hard_temporal_triplet(
    source_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    margin: float = TRIPLET_MARGIN,
) -> tuple[torch.Tensor, dict[str, int]]:
    """CELLECT hard-positive/hard-negative squared-distance triplet loss.

    Multiple positives in a row naturally represent a division during training,
    while inference remains divisions-off.
    """

    distances = squared_embedding_distance(source_embedding, target_embedding)
    positive = positive_mask.to(device=distances.device, dtype=torch.bool)
    if positive.shape != distances.shape:
        raise CellectContractError("positive mask shape differs from the embedding distance matrix")
    candidates = (
        torch.ones_like(positive)
        if candidate_mask is None
        else candidate_mask.to(device=distances.device, dtype=torch.bool)
    )
    if candidates.shape != distances.shape:
        raise CellectContractError("candidate mask shape differs from the embedding distance matrix")
    candidates = candidates | positive
    negative = candidates & ~positive
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not bool(valid.any().item()):
        return distances.sum() * 0.0, {"anchors": int(len(valid)), "valid_anchors": 0}
    hardest_positive = distances.masked_fill(~positive, -torch.inf).amax(dim=1)
    hardest_negative = distances.masked_fill(~negative, torch.inf).amin(dim=1)
    losses = F.relu(hardest_positive[valid] - hardest_negative[valid] + float(margin))
    return losses.mean(), {
        "anchors": int(len(valid)),
        "valid_anchors": int(valid.sum().item()),
    }


def same_frame_separation(embedding: torch.Tensor, *, margin: float = TRIPLET_MARGIN) -> torch.Tensor:
    """Repel distinct same-frame detections, which are guaranteed negatives."""

    if embedding.ndim != 2:
        raise CellectContractError("same-frame embeddings must have shape (N,C)")
    if len(embedding) < 2:
        return embedding.sum() * 0.0
    distances = squared_embedding_distance(embedding, embedding)
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    return F.relu(float(margin) - distances[upper]).mean()


def rank_accuracy(
    squared_distances: np.ndarray,
    positive_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, float | int]:
    """Top-1 edge accuracy among candidate rows with at least one positive."""

    distances = np.asarray(squared_distances, dtype=np.float64)
    positive = np.asarray(positive_mask, dtype=bool)
    candidates = np.asarray(candidate_mask, dtype=bool) | positive
    if distances.shape != positive.shape or distances.shape != candidates.shape:
        raise CellectContractError("rank-accuracy arrays must share one shape")
    correct = 0
    rows = 0
    for row in range(distances.shape[0]):
        if not positive[row].any() or not candidates[row].any():
            continue
        candidate_indices = np.flatnonzero(candidates[row])
        best = min(candidate_indices, key=lambda index: (float(distances[row, index]), int(index)))
        rows += 1
        correct += int(positive[row, best])
    return {
        "rows": rows,
        "correct": correct,
        "top1_accuracy": float(correct / rows) if rows else float("nan"),
    }


__all__ = [
    "CellectAdapterConfig",
    "CellectContractError",
    "CellectEmbeddingAdapter",
    "EMBEDDING_CHANNELS",
    "TRIPLET_MARGIN",
    "UPSTREAM_COMMIT",
    "UPSTREAM_LICENSE_SHA256",
    "UPSTREAM_MODEL_SOURCE_SHA256",
    "UPSTREAM_MODEL_FILE",
    "UPSTREAM_WEIGHT_SHA256",
    "batch_hard_temporal_triplet",
    "cellect_log_preprocess",
    "load_published_backbone",
    "rank_accuracy",
    "same_frame_separation",
    "sample_embeddings_zyx",
    "sha256_file",
    "squared_embedding_distance",
    "topk_candidate_mask",
    "validate_upstream_tree",
]
