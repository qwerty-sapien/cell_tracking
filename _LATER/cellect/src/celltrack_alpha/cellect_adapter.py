"""Clean-room CELLECT embedding utilities for the Biohub point tracker.

The primary geometry is the geometry used to train the released embeddings:
unnormalized vectors, square-root Euclidean distance, a margin of ``0.3``, and
hardest-positive/hardest-negative mining.  Candidate admission always uses
physical distance; embedding values only score candidates after that gate.

The normalized squared-distance variant is retained solely as an explicitly
named ablation.  This module never imports or executes the reference source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cellect_model import (
    EMBEDDING_CHANNELS,
    PUBLISHED_CHECKPOINT_FILENAME,
    PUBLISHED_CHECKPOINT_SHA256,
    CellectBackbone,
    CellectModelError,
    CellectOutputs,
    load_published_backbone,
)


REFERENCE_COMMIT = "3586070926f7f1fd5d8df37456861d22bdc63236"
UPSTREAM_COMMIT = REFERENCE_COMMIT  # compatibility name used by existing reports
UPSTREAM_WEIGHT_SHA256 = PUBLISHED_CHECKPOINT_SHA256
UPSTREAM_MODEL_FILE = PUBLISHED_CHECKPOINT_FILENAME
TRIPLET_MARGIN = 0.3
DEFAULT_POSITIVE_RADIUS_UM = 4.0
DEFAULT_CANDIDATE_TOP_K = 5
LARGE_DISPLACEMENT_TOP_K = 10


class CellectContractError(RuntimeError):
    """Raised when an embedding, candidate, or checkpoint invariant changes."""


@dataclass(frozen=True)
class CellectAdapterConfig:
    """Frozen defaults for the clean-room fit117/CV57 experiment."""

    crop_zyx: tuple[int, int, int] = (16, 64, 64)
    embedding_channels: int = EMBEDDING_CHANNELS
    triplet_margin: float = TRIPLET_MARGIN
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K
    large_displacement_candidate_top_k: int = LARGE_DISPLACEMENT_TOP_K
    positive_radius_um: float = DEFAULT_POSITIVE_RADIUS_UM
    same_frame_weight: float = 0.1
    division_output_enabled: bool = False

    def __post_init__(self) -> None:
        if self.crop_zyx[1] % 16 or self.crop_zyx[2] % 16:
            raise CellectContractError("CELLECT lateral crop dimensions must be multiples of 16")
        if any(type(value) is not int or value <= 0 for value in self.crop_zyx):
            raise CellectContractError("CELLECT crop dimensions must be positive integers")
        if self.embedding_channels != EMBEDDING_CHANNELS:
            raise CellectContractError("the CELLECT embedding head has 64 channels")
        if not math.isfinite(self.triplet_margin) or self.triplet_margin <= 0.0:
            raise CellectContractError("triplet margin must be finite and positive")
        if type(self.candidate_top_k) is not int or self.candidate_top_k != 5:
            raise CellectContractError("the primary CELLECT candidate gate must use k=5")
        if (
            type(self.large_displacement_candidate_top_k) is not int
            or self.large_displacement_candidate_top_k != 10
        ):
            raise CellectContractError("the large-displacement sensitivity arm must use k=10")
        if not math.isfinite(self.positive_radius_um) or self.positive_radius_um <= 0.0:
            raise CellectContractError("positive sampling radius must be finite and positive")
        if not math.isfinite(self.same_frame_weight) or self.same_frame_weight < 0.0:
            raise CellectContractError("same-frame weight must be finite and nonnegative")
        if self.division_output_enabled:
            raise CellectContractError(
                "division graph output remains disabled until patched-metric TP proof"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_numeric_checkpoint(checkpoint: Path) -> dict[str, str | int]:
    """Validate only the exempt numeric pretrained model artifact."""

    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise CellectContractError(f"missing CELLECT numeric checkpoint: {path}")
    observed = sha256_file(path)
    if observed != PUBLISHED_CHECKPOINT_SHA256:
        raise CellectContractError(
            f"CELLECT checkpoint hash changed: {observed} != {PUBLISHED_CHECKPOINT_SHA256}"
        )
    return {
        "path": str(path),
        "sha256": observed,
        "bytes": path.stat().st_size,
        "reference_commit": REFERENCE_COMMIT,
        "artifact_kind": "pretrained-numeric-weights",
        "source_code_loaded": "false",
    }


def cellect_log_preprocess(volume: torch.Tensor) -> torch.Tensor:
    """Apply the frozen min-clamp, ``+1900``, then ``log1p`` transform."""

    if volume.ndim != 5 or volume.shape[1] != 2:
        raise CellectContractError("CELLECT input must have shape (B,2,Y,X,Z)")
    if not volume.is_floating_point():
        volume = volume.float()
    if not bool(torch.isfinite(volume).all().item()):
        raise CellectContractError("CELLECT input contains non-finite intensity")
    positive = volume > 0
    if not bool(positive.any().item()):
        raise CellectContractError("CELLECT input contains no positive intensity")
    minimum = volume[positive].amin()
    clamped = torch.where(volume < minimum, minimum, volume)
    return torch.log1p(clamped + 1900.0)


class CellectEmbeddingAdapter(nn.Module):
    """Expose the unnormalized 64-channel embedding learned by CELLECT."""

    def __init__(
        self,
        backbone: CellectBackbone | nn.Module,
        config: CellectAdapterConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or CellectAdapterConfig()

    def all_outputs(self, paired_yxz: torch.Tensor) -> CellectOutputs | tuple[torch.Tensor, ...]:
        try:
            outputs = self.backbone(cellect_log_preprocess(paired_yxz))
        except CellectModelError as exc:
            raise CellectContractError(str(exc)) from exc
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
            raise CellectContractError("CELLECT backbone must return five output maps")
        feature_map = outputs[2]
        if feature_map.ndim != 5 or feature_map.shape[1] != self.config.embedding_channels:
            raise CellectContractError("CELLECT embedding shape changed")
        if not bool(torch.isfinite(feature_map).all().item()):
            raise CellectContractError("CELLECT produced non-finite embeddings")
        return outputs

    def forward(self, paired_yxz: torch.Tensor) -> torch.Tensor:
        return self.all_outputs(paired_yxz)[2]


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


def euclidean_dist(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pairwise square-root Euclidean distance with the frozen clamp path."""

    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise CellectContractError("embedding matrices have incompatible shapes")
    if not source.is_floating_point() or not target.is_floating_point():
        raise CellectContractError("embedding matrices must be floating-point")
    if not bool(torch.isfinite(source).all().item()) or not bool(
        torch.isfinite(target).all().item()
    ):
        raise CellectContractError("embedding matrix contains non-finite values")
    rows, columns = source.shape[0], target.shape[0]
    source_norm = torch.pow(source, 2).sum(1, keepdim=True).expand(rows, columns)
    target_norm = torch.pow(target, 2).sum(1, keepdim=True).expand(columns, rows).t()
    squared = source_norm + target_norm
    squared = torch.addmm(squared, source, target.t(), beta=1.0, alpha=-2.0)
    return squared.clamp(min=1e-12).sqrt()


def hard_example_mining(
    distance_matrix: torch.Tensor,
    labels: torch.Tensor,
    *,
    return_indices: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Select the farthest same-label and nearest different-label item per row."""

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise CellectContractError("hard mining requires a square distance matrix")
    labels = labels.to(device=distance_matrix.device)
    if labels.ndim != 1 or len(labels) != len(distance_matrix):
        raise CellectContractError("hard-mining labels must have shape (N,)")
    same = labels[:, None].eq(labels[None, :])
    different = ~same
    if not bool(different.any(dim=1).all().item()):
        raise CellectContractError("every anchor requires at least one negative label")
    hardest_positive, positive_indices = distance_matrix.masked_fill(~same, -torch.inf).max(dim=1)
    hardest_negative, negative_indices = distance_matrix.masked_fill(~different, torch.inf).min(dim=1)
    if return_indices:
        return hardest_positive, hardest_negative, positive_indices, negative_indices
    return hardest_positive, hardest_negative


class TripletLoss:
    """CELLECT batch-hard triplet loss on unnormalized Euclidean embeddings."""

    def __init__(self, margin: float = TRIPLET_MARGIN) -> None:
        if not math.isfinite(margin) or margin <= 0.0:
            raise CellectContractError("triplet margin must be finite and positive")
        self.margin = float(margin)

    def __call__(
        self,
        embedding: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = euclidean_dist(embedding, embedding)
        hardest_positive, hardest_negative = hard_example_mining(distances, labels)
        loss = F.margin_ranking_loss(
            hardest_negative,
            hardest_positive,
            torch.ones_like(hardest_negative),
            margin=self.margin,
        )
        return loss, hardest_positive, hardest_negative


def normalized_squared_embedding_distance_ablation(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Non-primary ablation: unit-normalized vectors with squared distance."""

    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise CellectContractError("embedding matrices have incompatible shapes")
    source = F.normalize(source, p=2.0, dim=1, eps=1e-8)
    target = F.normalize(target, p=2.0, dim=1, eps=1e-8)
    return torch.sum((source[:, None, :] - target[None, :, :]) ** 2, dim=2)


def squared_embedding_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Deprecated compatibility alias for the explicitly labelled ablation."""

    return normalized_squared_embedding_distance_ablation(source, target)


def topk_candidate_mask(
    source_zyx: np.ndarray,
    target_zyx: np.ndarray,
    scale_zyx: Sequence[float],
    *,
    k: int = DEFAULT_CANDIDATE_TOP_K,
) -> np.ndarray:
    """Select candidates by physical distance, never by embedding distance."""

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
    stable_indices = np.arange(len(target), dtype=np.int64)
    for row in range(len(source)):
        order = np.lexsort((stable_indices, distances[row]))
        mask[row, order[: min(k, len(target))]] = True
    return mask


def same_frame_negative_pairs(
    points_zyx: np.ndarray | torch.Tensor,
    scale_zyx: Sequence[float],
    *,
    k: int = DEFAULT_CANDIDATE_TOP_K,
) -> torch.Tensor:
    """Mine unique guaranteed negatives from spatially nearest same-frame cells."""

    if isinstance(points_zyx, torch.Tensor):
        device = points_zyx.device
        points = points_zyx.detach().cpu().numpy()
    else:
        device = torch.device("cpu")
        points = np.asarray(points_zyx)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise CellectContractError("same-frame points must have shape (N,3)")
    if len(points) < 2:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    mask = topk_candidate_mask(points, points, scale_zyx, k=min(k + 1, len(points)))
    np.fill_diagonal(mask, False)
    pairs = {
        (min(row, column), max(row, column))
        for row, column in np.argwhere(mask)
        if row != column
    }
    return torch.tensor(sorted(pairs), dtype=torch.long, device=device)


def negative_difficulty_report(
    embedding: torch.Tensor,
    pairs: torch.Tensor,
    *,
    margin: float = TRIPLET_MARGIN,
    collapse_epsilon: float = 1e-3,
) -> dict[str, float | int]:
    """Summarize whether spatial hard negatives remain informative or collapse."""

    if pairs.ndim != 2 or pairs.shape[1:] != (2,):
        raise CellectContractError("negative pairs must have shape (P,2)")
    if not len(pairs):
        return {
            "pairs": 0,
            "active_pairs": 0,
            "active_fraction": 0.0,
            "collapsed_pairs": 0,
            "collapsed_fraction": 0.0,
            "distance_min": float("nan"),
            "distance_p25": float("nan"),
            "distance_median": float("nan"),
            "distance_p75": float("nan"),
            "distance_max": float("nan"),
        }
    pairs = pairs.to(device=embedding.device, dtype=torch.long)
    distances = torch.linalg.vector_norm(
        embedding[pairs[:, 0]] - embedding[pairs[:, 1]],
        dim=1,
    )
    if not bool(torch.isfinite(distances).all().item()):
        raise CellectContractError("same-frame negative distances are non-finite")
    quantiles = torch.quantile(
        distances.float(),
        torch.tensor([0.25, 0.5, 0.75], device=distances.device),
    )
    active = distances < float(margin)
    collapsed = distances <= float(collapse_epsilon)
    return {
        "pairs": int(len(distances)),
        "active_pairs": int(active.sum().item()),
        "active_fraction": float(active.float().mean().item()),
        "collapsed_pairs": int(collapsed.sum().item()),
        "collapsed_fraction": float(collapsed.float().mean().item()),
        "distance_min": float(distances.min().item()),
        "distance_p25": float(quantiles[0].item()),
        "distance_median": float(quantiles[1].item()),
        "distance_p75": float(quantiles[2].item()),
        "distance_max": float(distances.max().item()),
    }


def same_frame_separation(
    embedding: torch.Tensor,
    *,
    points_zyx: np.ndarray | torch.Tensor | None = None,
    scale_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    k: int = DEFAULT_CANDIDATE_TOP_K,
    margin: float = TRIPLET_MARGIN,
    return_report: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    """Repel spatially mined, guaranteed distinct same-frame detections."""

    if embedding.ndim != 2:
        raise CellectContractError("same-frame embeddings must have shape (N,C)")
    if points_zyx is None:
        if len(embedding) < 2:
            pairs = torch.empty((0, 2), dtype=torch.long, device=embedding.device)
        else:
            pairs = torch.triu_indices(
                len(embedding),
                len(embedding),
                offset=1,
                device=embedding.device,
            ).t()
    else:
        if len(points_zyx) != len(embedding):
            raise CellectContractError("same-frame point and embedding counts differ")
        pairs = same_frame_negative_pairs(points_zyx, scale_zyx, k=k).to(embedding.device)
    if not len(pairs):
        loss = embedding.sum() * 0.0
    else:
        distances = torch.linalg.vector_norm(
            embedding[pairs[:, 0]] - embedding[pairs[:, 1]],
            dim=1,
        ).clamp(min=1e-6)
        loss = F.relu(float(margin) - distances).mean()
    if return_report:
        return loss, negative_difficulty_report(embedding, pairs, margin=margin)
    return loss


def batch_hard_temporal_triplet(
    source_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    margin: float = TRIPLET_MARGIN,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Cross-frame hardest-positive/hardest-negative Euclidean triplet loss."""

    distances = euclidean_dist(source_embedding, target_embedding)
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
        return distances.sum() * 0.0, {
            "anchors": int(len(valid)),
            "valid_anchors": 0,
            "hardest_positive_mean": float("nan"),
            "hardest_negative_mean": float("nan"),
        }
    hardest_positive = distances.masked_fill(~positive, -torch.inf).amax(dim=1)
    hardest_negative = distances.masked_fill(~negative, torch.inf).amin(dim=1)
    losses = F.relu(hardest_positive[valid] - hardest_negative[valid] + float(margin))
    return losses.mean(), {
        "anchors": int(len(valid)),
        "valid_anchors": int(valid.sum().item()),
        "hardest_positive_mean": float(hardest_positive[valid].detach().mean().item()),
        "hardest_negative_mean": float(hardest_negative[valid].detach().mean().item()),
    }


def resolve_positive_radii_um(
    count: int,
    *,
    constant_radius_um: float = DEFAULT_POSITIVE_RADIUS_UM,
    dog_scale_um: np.ndarray | Sequence[float] | None = None,
    bounds_um: tuple[float, float] = (3.0, 5.0),
) -> np.ndarray:
    """Choose constant or bounded DoG-derived radii for positive voxel sampling."""

    if type(count) is not int or count < 0:
        raise CellectContractError("radius count must be a nonnegative integer")
    lower, upper = (float(value) for value in bounds_um)
    if not (0.0 < lower <= upper and math.isfinite(lower) and math.isfinite(upper)):
        raise CellectContractError("radius bounds must be finite, positive, and ordered")
    if not math.isfinite(constant_radius_um) or constant_radius_um <= 0.0:
        raise CellectContractError("constant radius must be finite and positive")
    if dog_scale_um is None:
        return np.full(count, float(constant_radius_um), dtype=np.float64)
    scales = np.asarray(dog_scale_um, dtype=np.float64)
    if scales.shape != (count,) or not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise CellectContractError("DoG radii must be finite positive values with shape (N,)")
    return np.clip(scales, lower, upper)


def radius_offsets_zyx(
    scale_zyx: Sequence[float],
    radius_um: float,
) -> np.ndarray:
    """Integer voxel offsets inside an anisotropic physical-radius sphere."""

    scale = np.asarray(tuple(scale_zyx), dtype=np.float64)
    if scale.shape != (3,) or np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise CellectContractError("physical scale must contain three finite positive values")
    if not math.isfinite(radius_um) or radius_um <= 0.0:
        raise CellectContractError("sampling radius must be finite and positive")
    extent = np.floor(radius_um / scale).astype(np.int64)
    grid = np.stack(
        np.meshgrid(
            np.arange(-extent[0], extent[0] + 1),
            np.arange(-extent[1], extent[1] + 1),
            np.arange(-extent[2], extent[2] + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    keep = np.linalg.norm(grid * scale[None, :], axis=1) <= radius_um + 1e-12
    return np.ascontiguousarray(grid[keep], dtype=np.int64)


def sample_radius_embeddings_zyx(
    feature_map: torch.Tensor,
    centers_zyx: torch.Tensor,
    scale_zyx: Sequence[float],
    *,
    radii_um: Sequence[float] | np.ndarray | None = None,
    constant_radius_um: float = DEFAULT_POSITIVE_RADIUS_UM,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
    """Sample per-cell positive voxels within constant or DoG-derived radii."""

    if feature_map.ndim != 5 or feature_map.shape[0] != 1:
        raise CellectContractError("radius sampling requires one feature volume")
    if centers_zyx.ndim != 2 or centers_zyx.shape[1:] != (3,):
        raise CellectContractError("centers must have shape (N,3)")
    centers = centers_zyx.detach().cpu().numpy().astype(np.int64, copy=False)
    radii = resolve_positive_radii_um(
        len(centers),
        constant_radius_um=constant_radius_um,
        dog_scale_um=radii_um,
    )
    spatial_shape = np.asarray(
        (feature_map.shape[4], feature_map.shape[2], feature_map.shape[3]),
        dtype=np.int64,
    )
    sampled_points: list[np.ndarray] = []
    sampled_labels: list[np.ndarray] = []
    clipped = 0
    for label, (center, radius) in enumerate(zip(centers, radii, strict=True)):
        points = center[None, :] + radius_offsets_zyx(scale_zyx, float(radius))
        keep = np.all((points >= 0) & (points < spatial_shape[None, :]), axis=1)
        clipped += int((~keep).sum())
        points = points[keep]
        if len(points):
            sampled_points.append(points)
            sampled_labels.append(np.full(len(points), label, dtype=np.int64))
    if not sampled_points:
        empty = feature_map.new_empty((0, feature_map.shape[1]))
        return empty, torch.empty(0, dtype=torch.long, device=feature_map.device), {
            "cells": int(len(centers)),
            "sampled_voxels": 0,
            "clipped_voxels": clipped,
            "radius_min_um": float(radii.min()) if len(radii) else float("nan"),
            "radius_max_um": float(radii.max()) if len(radii) else float("nan"),
        }
    points_array = np.concatenate(sampled_points, axis=0)
    labels_array = np.concatenate(sampled_labels, axis=0)
    points_tensor = torch.from_numpy(points_array).to(feature_map.device)
    labels_tensor = torch.from_numpy(labels_array).to(feature_map.device)
    sampled = sample_embeddings_zyx(feature_map, points_tensor)
    return sampled, labels_tensor, {
        "cells": int(len(centers)),
        "sampled_voxels": int(len(points_array)),
        "clipped_voxels": clipped,
        "radius_min_um": float(radii.min()),
        "radius_max_um": float(radii.max()),
    }


def radius_sensitivity_report(
    centers_zyx: np.ndarray,
    scale_zyx: Sequence[float],
    spatial_shape_zyx: Sequence[int],
    *,
    radii_um: Sequence[float] = (3.0, 4.0, 5.0),
) -> dict[str, dict[str, float | int]]:
    """Report positive-sampling volume sensitivity around the 4-um default."""

    centers = np.asarray(centers_zyx, dtype=np.int64)
    shape = np.asarray(tuple(spatial_shape_zyx), dtype=np.int64)
    if centers.ndim != 2 or centers.shape[1:] != (3,) or shape.shape != (3,):
        raise CellectContractError("invalid radius sensitivity geometry")
    report: dict[str, dict[str, float | int]] = {}
    for radius in radii_um:
        offsets = radius_offsets_zyx(scale_zyx, float(radius))
        counts = []
        for center in centers:
            points = center[None, :] + offsets
            counts.append(int(np.all((points >= 0) & (points < shape[None, :]), axis=1).sum()))
        report[f"{float(radius):g}"] = {
            "radius_um": float(radius),
            "cells": int(len(centers)),
            "voxels_total": int(sum(counts)),
            "voxels_per_cell_mean": float(np.mean(counts)) if counts else float("nan"),
        }
    return report


def rank_accuracy(
    distances: np.ndarray,
    positive_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, float | int]:
    """Top-1 edge accuracy within the externally supplied spatial gate."""

    values = np.asarray(distances, dtype=np.float64)
    positive = np.asarray(positive_mask, dtype=bool)
    candidates = np.asarray(candidate_mask, dtype=bool) | positive
    if values.shape != positive.shape or values.shape != candidates.shape:
        raise CellectContractError("rank-accuracy arrays must share one shape")
    if not np.isfinite(values).all():
        raise CellectContractError("rank-accuracy distances contain non-finite values")
    correct = 0
    rows = 0
    for row in range(values.shape[0]):
        if not positive[row].any() or not candidates[row].any():
            continue
        candidate_indices = np.flatnonzero(candidates[row])
        best = min(candidate_indices, key=lambda index: (float(values[row, index]), int(index)))
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
    "DEFAULT_CANDIDATE_TOP_K",
    "DEFAULT_POSITIVE_RADIUS_UM",
    "EMBEDDING_CHANNELS",
    "LARGE_DISPLACEMENT_TOP_K",
    "REFERENCE_COMMIT",
    "TRIPLET_MARGIN",
    "TripletLoss",
    "UPSTREAM_COMMIT",
    "UPSTREAM_MODEL_FILE",
    "UPSTREAM_WEIGHT_SHA256",
    "batch_hard_temporal_triplet",
    "cellect_log_preprocess",
    "euclidean_dist",
    "hard_example_mining",
    "load_published_backbone",
    "negative_difficulty_report",
    "normalized_squared_embedding_distance_ablation",
    "radius_offsets_zyx",
    "radius_sensitivity_report",
    "rank_accuracy",
    "resolve_positive_radii_um",
    "same_frame_negative_pairs",
    "same_frame_separation",
    "sample_embeddings_zyx",
    "sample_radius_embeddings_zyx",
    "sha256_file",
    "squared_embedding_distance",
    "topk_candidate_mask",
    "validate_numeric_checkpoint",
]
