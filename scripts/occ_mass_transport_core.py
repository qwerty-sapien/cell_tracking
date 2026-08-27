#!/usr/bin/env python3
"""Label-blind 3-D endpoint mass-transport features for occupied swaps."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np


SCALE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
CROP_RADIUS_ZYX = (3, 12, 12)
EPSILONS = (0.03, 0.07, 0.15)
SINKHORN_ITERATIONS = 30
PROJECTION_DIRECTIONS = np.asarray(
    (
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, -1.0),
    ),
    dtype=np.float64,
)
PROJECTION_DIRECTIONS /= np.linalg.norm(PROJECTION_DIRECTIONS, axis=1, keepdims=True)


class MassTransportError(RuntimeError):
    """The frozen supplier contract was violated."""


@dataclass(frozen=True)
class EndpointBatch:
    """A fixed-cardinality sparse probability mass representation."""

    coordinates: np.ndarray
    weights: np.ndarray
    log_total_mass: np.ndarray
    centroid: np.ndarray


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def feature_names(epsilons: Iterable[float] = EPSILONS) -> tuple[str, ...]:
    names: list[str] = []
    for epsilon in epsilons:
        token = str(epsilon).replace(".", "p")
        names.extend((f"sinkhorn_cost_eps_{token}", f"marginal_residual_eps_{token}"))
    names.extend(
        (
            "absolute_log_mass_ratio",
            "centroid_shift_normalized",
            "source_log_total_mass",
            "target_log_total_mass",
        )
    )
    return tuple(names)


def extract_reflect_patch(
    array: object,
    frame: int,
    center_zyx: np.ndarray,
    radius_zyx: tuple[int, int, int] = CROP_RADIUS_ZYX,
) -> np.ndarray:
    """Read one patch and deterministically reflect-pad every clipped boundary."""

    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 4:
        raise MassTransportError(f"raw array must be TZYX, got {shape}")
    if frame < 0 or frame >= shape[0]:
        raise MassTransportError(f"frame {frame} outside [0,{shape[0]})")
    center = np.asarray(center_zyx, dtype=np.int64)
    if center.shape != (3,):
        raise MassTransportError("endpoint coordinate is not zyx")
    starts = center - np.asarray(radius_zyx, dtype=np.int64)
    stops = center + np.asarray(radius_zyx, dtype=np.int64) + 1
    clipped_start = np.maximum(starts, 0)
    clipped_stop = np.minimum(stops, np.asarray(shape[1:], dtype=np.int64))
    if np.any(clipped_stop <= clipped_start):
        raise MassTransportError("endpoint crop has no overlap with raw volume")
    patch = np.asarray(
        array[
            frame,
            int(clipped_start[0]) : int(clipped_stop[0]),
            int(clipped_start[1]) : int(clipped_stop[1]),
            int(clipped_start[2]) : int(clipped_stop[2]),
        ],
        dtype=np.float32,
    )
    before = (starts * -1).clip(min=0)
    after = (stops - np.asarray(shape[1:], dtype=np.int64)).clip(min=0)
    padding = tuple((int(lo), int(hi)) for lo, hi in zip(before, after, strict=True))
    if any(lo or hi for lo, hi in padding):
        modes = "reflect" if all(size > 1 for size in patch.shape) else "edge"
        patch = np.pad(patch, padding, mode=modes)
    expected = tuple(2 * radius + 1 for radius in radius_zyx)
    if patch.shape != expected or not np.isfinite(patch).all():
        raise MassTransportError(f"invalid reflected crop {patch.shape}, expected {expected}")
    return patch


def _normalized_mass(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(patches, dtype=np.float32)
    if values.ndim != 4 or not np.isfinite(values).all():
        raise MassTransportError("patch batch must be finite NZYX")
    flat = values.reshape(len(values), -1)
    background = np.quantile(flat, 0.50, axis=1, keepdims=True)
    high = np.quantile(flat, 0.99, axis=1, keepdims=True)
    scale = np.maximum(high - background, 1.0e-6)
    foreground = np.clip((flat - background) / scale, 0.0, 1.0)
    total = foreground.sum(axis=1)
    # Blank crops remain well-defined and deterministic without hiding them.
    blank = total <= 1.0e-8
    if blank.any():
        foreground[blank] = 1.0
        total[blank] = foreground.shape[1]
    return foreground.astype(np.float32, copy=False), total.astype(np.float32)


def endpoint_batch(
    patches: np.ndarray,
    *,
    support: int,
    device: str = "cpu",
    radius_zyx: tuple[int, int, int] = CROP_RADIUS_ZYX,
) -> EndpointBatch:
    """Select the highest-mass support with stable index tie-breaking."""

    foreground, total = _normalized_mass(patches)
    count, voxels = foreground.shape
    if support <= 1 or support > voxels:
        raise MassTransportError(f"invalid support {support} for {voxels} voxels")
    # Stable mergesort plus a final ascending-index ordering makes ties portable.
    selected = np.argsort(-foreground, axis=1, kind="stable")[:, :support]
    selected.sort(axis=1)
    selected_mass = np.take_along_axis(foreground, selected, axis=1) + 1.0e-8
    selected_mass /= selected_mass.sum(axis=1, keepdims=True)

    shape = tuple(2 * radius + 1 for radius in radius_zyx)
    grid = np.stack(np.unravel_index(np.arange(voxels), shape), axis=1).astype(np.float32)
    radius = np.asarray(radius_zyx, dtype=np.float32)
    grid = (grid - radius) / radius
    selected_coordinates = grid[selected]
    centroid = (selected_coordinates * selected_mass[..., None]).sum(axis=1)

    if device != "cpu":
        raise MassTransportError("the canonical deterministic backend is CPU")
    return EndpointBatch(
        coordinates=np.asarray(selected_coordinates, dtype=np.float64),
        weights=np.asarray(selected_mass, dtype=np.float64),
        log_total_mass=np.asarray(np.log1p(total), dtype=np.float64),
        centroid=np.asarray(centroid, dtype=np.float64),
    )


def _logsumexp(value: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(value, axis=axis, keepdims=True)
    reduced = maximum + np.log(np.sum(np.exp(value - maximum), axis=axis, keepdims=True))
    return np.squeeze(reduced, axis=axis)


def _sinkhorn(
    source: EndpointBatch,
    target: EndpointBatch,
    epsilon: float,
    iterations: int = SINKHORN_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    if source.coordinates.shape != target.coordinates.shape:
        raise MassTransportError("source and target support shapes differ")
    if epsilon <= 0.0 or iterations <= 0:
        raise MassTransportError("invalid Sinkhorn parameters")
    cost = np.mean(
        (source.coordinates[:, :, None, :] - target.coordinates[:, None, :, :]) ** 2,
        axis=-1,
    )
    log_a = np.log(source.weights)
    log_b = np.log(target.weights)
    f = np.zeros_like(log_a)
    g = np.zeros_like(log_b)
    for _ in range(iterations):
        f = epsilon * (
            log_a - _logsumexp((g[:, None, :] - cost) / epsilon, axis=2)
        )
        g = epsilon * (
            log_b - _logsumexp((f[:, :, None] - cost) / epsilon, axis=1)
        )
    plan = np.exp((f[:, :, None] + g[:, None, :] - cost) / epsilon)
    transport_cost = np.sum(plan * cost, axis=(1, 2))
    row_residual = np.max(np.abs(plan.sum(axis=2) - source.weights), axis=1)
    column_residual = np.max(np.abs(plan.sum(axis=1) - target.weights), axis=1)
    residual = np.maximum(row_residual, column_residual)
    return transport_cost, residual


def pair_features(
    source: EndpointBatch,
    target: EndpointBatch,
    *,
    epsilons: tuple[float, ...] = EPSILONS,
    iterations: int = SINKHORN_ITERATIONS,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for epsilon in epsilons:
        cost, residual = _sinkhorn(source, target, epsilon, iterations)
        columns.extend((cost, residual))
    columns.extend(
        (
            np.abs(source.log_total_mass - target.log_total_mass),
            np.linalg.norm(source.centroid - target.centroid, axis=1),
            source.log_total_mass,
            target.log_total_mass,
        )
    )
    features = np.stack(columns, axis=1).astype(np.float32)
    if features.shape[1] != len(feature_names(epsilons)) or not np.isfinite(features).all():
        raise MassTransportError("non-finite or malformed mass-transport features")
    return features


def subset(batch: EndpointBatch, indices: np.ndarray) -> EndpointBatch:
    index = np.asarray(indices, dtype=np.int64)
    return EndpointBatch(
        coordinates=batch.coordinates[index],
        weights=batch.weights[index],
        log_total_mass=batch.log_total_mass[index],
        centroid=batch.centroid[index],
    )


def sliced_feature_names() -> tuple[str, ...]:
    return (
        "wasserstein_z",
        "wasserstein_y",
        "wasserstein_x",
        "wasserstein_radial",
        "wasserstein_proj_z_plus_y",
        "wasserstein_proj_z_minus_y",
        "wasserstein_proj_z_plus_x",
        "wasserstein_proj_z_minus_x",
        "wasserstein_proj_y_plus_x",
        "wasserstein_proj_y_minus_x",
        "absolute_log_mass_ratio",
        "centroid_shift_normalized",
        "source_log_total_mass",
        "target_log_total_mass",
    )


def _histogram_w1(first: np.ndarray, second: np.ndarray, spacing: float) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 2:
        raise MassTransportError("W1 histograms have incompatible shapes")
    return np.sum(np.abs(np.cumsum(first - second, axis=1)), axis=1) * spacing


def _binned_mass(
    mass: np.ndarray, values: np.ndarray, *, bins: int, low: float, high: float
) -> np.ndarray:
    if bins <= 1 or high <= low:
        raise MassTransportError("invalid projection histogram")
    indices = np.floor((values - low) / (high - low) * bins).astype(np.int64)
    indices = np.clip(indices, 0, bins - 1)
    output = np.zeros((len(mass), bins), dtype=np.float64)
    for column in range(mass.shape[1]):
        output[:, indices[column]] += mass[:, column]
    output /= np.maximum(output.sum(axis=1, keepdims=True), 1.0e-12)
    return output


def sliced_pair_features(
    source_patches: np.ndarray,
    target_patches: np.ndarray,
    *,
    radius_zyx: tuple[int, int, int] = CROP_RADIUS_ZYX,
) -> np.ndarray:
    """Exact deterministic sliced-W1 features on full normalized crop mass."""

    if source_patches.shape != target_patches.shape:
        raise MassTransportError("source and target crop batches differ")
    source_flat, source_total = _normalized_mass(source_patches)
    target_flat, target_total = _normalized_mass(target_patches)
    source_mass = source_flat / source_flat.sum(axis=1, keepdims=True)
    target_mass = target_flat / target_flat.sum(axis=1, keepdims=True)
    shape = tuple(2 * radius + 1 for radius in radius_zyx)
    grid = np.stack(np.unravel_index(np.arange(source_mass.shape[1]), shape), axis=1).astype(np.float64)
    radius = np.asarray(radius_zyx, dtype=np.float64)
    coordinates = (grid - radius) / radius

    columns: list[np.ndarray] = []
    reshaped_source = source_mass.reshape((len(source_mass), *shape))
    reshaped_target = target_mass.reshape((len(target_mass), *shape))
    for axis, size in enumerate(shape):
        reduce_axes = tuple(index + 1 for index in range(3) if index != axis)
        first = reshaped_source.sum(axis=reduce_axes)
        second = reshaped_target.sum(axis=reduce_axes)
        columns.append(_histogram_w1(first, second, 2.0 / max(size - 1, 1)))

    radial = np.linalg.norm(coordinates, axis=1)
    radial_high = float(np.sqrt(3.0))
    first = _binned_mass(source_mass, radial, bins=25, low=0.0, high=radial_high)
    second = _binned_mass(target_mass, radial, bins=25, low=0.0, high=radial_high)
    columns.append(_histogram_w1(first, second, radial_high / 24.0))

    projection_high = float(np.sqrt(2.0))
    for direction in PROJECTION_DIRECTIONS:
        projection = coordinates @ direction
        first = _binned_mass(
            source_mass, projection, bins=25, low=-projection_high, high=projection_high
        )
        second = _binned_mass(
            target_mass, projection, bins=25, low=-projection_high, high=projection_high
        )
        columns.append(_histogram_w1(first, second, 2.0 * projection_high / 24.0))

    source_centroid = source_mass @ coordinates
    target_centroid = target_mass @ coordinates
    columns.extend(
        (
            np.abs(np.log1p(source_total) - np.log1p(target_total)),
            np.linalg.norm(source_centroid - target_centroid, axis=1),
            np.log1p(source_total),
            np.log1p(target_total),
        )
    )
    features = np.stack(columns, axis=1).astype(np.float32)
    if features.shape[1] != len(sliced_feature_names()) or not np.isfinite(features).all():
        raise MassTransportError("non-finite or malformed sliced-W1 features")
    return features
