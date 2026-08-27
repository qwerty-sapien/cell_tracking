#!/usr/bin/env python3
"""Outcome-blind, zero-weight raw-3D endpoint features for RAW3D-NOLINK-MVE.

This module has no label or graph interface.  It consumes one raw intensity
volume and endpoint coordinates, emits fixed hand-crafted summaries, and
provides the same eight action summaries and within-frame permutation control
used by the frozen separation falsifier.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np


SUPPLIER_ID: Final = "RAW3D-NOLINK-MVE-STAGE0-V1"
BATCH_SIZE: Final = 384
Z_RADIUS: Final = 3
YX_RADIUS: Final = 5

ENDPOINT_FEATURE_NAMES: Final = (
    "center_zscore",
    "core_mean",
    "core_std",
    "core_min",
    "core_max",
    "core_q25",
    "core_median",
    "core_q75",
    "context_mean",
    "context_std",
    "context_min",
    "context_max",
    "core_minus_shell_mean",
    "context_fraction_gt_frame_mean",
    "context_fraction_gt_mean_plus_std",
    "center_context_percentile",
    "gradient_abs_z_mean",
    "gradient_abs_y_mean",
    "gradient_abs_x_mean",
    "gradient_rms_all",
    "center_laplacian_abs_z",
    "center_laplacian_abs_y",
    "center_laplacian_abs_x",
    "positive_mass_centroid_z",
    "positive_mass_centroid_y",
    "positive_mass_centroid_x",
    "texture_entropy_8bin",
    "coordinate_z_fraction",
    "coordinate_y_fraction",
    "coordinate_x_fraction",
    "time_fraction",
    "log1p_frame_mean_raw",
    "log1p_frame_std_raw",
    "core_coefficient_of_variation",
    "gradient_anisotropy_log1p",
)

ACTION_FEATURE_NAMES: Final = (
    "cosine",
    "euclidean_over_sqrt_dimension",
    "mean_absolute_difference",
    "mean_product",
    "source_norm_over_sqrt_dimension",
    "target_norm_over_sqrt_dimension",
    "norm_difference_over_sqrt_dimension",
    "centered_correlation",
)

ENDPOINT_DIMENSION: Final = len(ENDPOINT_FEATURE_NAMES)
ACTION_DIMENSION: Final = len(ACTION_FEATURE_NAMES)


class FeatureContractError(ValueError):
    """Raised when the frozen feature contract is violated."""


_Z_OFFSETS = np.arange(-Z_RADIUS, Z_RADIUS + 1, dtype=np.int64)
_Y_OFFSETS = np.arange(-YX_RADIUS, YX_RADIUS + 1, dtype=np.int64)
_X_OFFSETS = np.arange(-YX_RADIUS, YX_RADIUS + 1, dtype=np.int64)
_REL_Z, _REL_Y, _REL_X = np.meshgrid(
    _Z_OFFSETS.astype(np.float32),
    _Y_OFFSETS.astype(np.float32),
    _X_OFFSETS.astype(np.float32),
    indexing="ij",
)
_CORE_MASK = (
    (np.abs(_REL_Z) <= 1)
    & (np.abs(_REL_Y) <= 2)
    & (np.abs(_REL_X) <= 2)
).reshape(-1)
_TEXTURE_EDGES = np.asarray(
    [-np.inf, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, np.inf],
    dtype=np.float32,
)


def _validate_volume(frame: np.ndarray, coordinates_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    volume = np.asarray(frame)
    coordinates = np.asarray(coordinates_zyx, dtype=np.int64)
    if volume.ndim != 3 or volume.dtype != np.uint16:
        raise FeatureContractError(
            f"expected a uint16 3-D frame, got shape={volume.shape} dtype={volume.dtype}"
        )
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise FeatureContractError("endpoint coordinates must have shape (N,3)")
    if len(coordinates):
        lower_ok = np.all(coordinates >= 0)
        upper_ok = np.all(coordinates < np.asarray(volume.shape, dtype=np.int64))
        if not lower_ok or not upper_ok:
            raise FeatureContractError("endpoint coordinate lies outside the raw frame")
    return volume, coordinates


def _texture_entropy(flat: np.ndarray) -> np.ndarray:
    probabilities = []
    for low, high in zip(_TEXTURE_EDGES[:-1], _TEXTURE_EDGES[1:]):
        probabilities.append(np.mean((flat >= low) & (flat < high), axis=1))
    p = np.stack(probabilities, axis=1).astype(np.float32)
    entropy = -np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-12)), 0), axis=1)
    return entropy.astype(np.float32)


def endpoint_features(
    frame: np.ndarray,
    coordinates_zyx: np.ndarray,
    *,
    time_fraction: float,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Extract fixed crude intensity, texture, gradient, and context summaries."""

    volume, coordinates = _validate_volume(frame, coordinates_zyx)
    if not np.isfinite(time_fraction) or not 0.0 <= float(time_fraction) <= 1.0:
        raise FeatureContractError("time_fraction must be finite and in [0,1]")
    if batch_size <= 0:
        raise FeatureContractError("batch_size must be positive")
    if not len(coordinates):
        return np.empty((0, ENDPOINT_DIMENSION), dtype=np.float32)

    frame_mean = float(np.mean(volume, dtype=np.float64))
    frame_std = float(np.std(volume, dtype=np.float64))
    frame_std = max(frame_std, 1.0)
    output = np.empty((len(coordinates), ENDPOINT_DIMENSION), dtype=np.float32)

    for start in range(0, len(coordinates), batch_size):
        stop = min(start + batch_size, len(coordinates))
        batch = coordinates[start:stop]
        z_index = np.clip(
            batch[:, 0, None] + _Z_OFFSETS[None, :], 0, volume.shape[0] - 1
        )
        y_index = np.clip(
            batch[:, 1, None] + _Y_OFFSETS[None, :], 0, volume.shape[1] - 1
        )
        x_index = np.clip(
            batch[:, 2, None] + _X_OFFSETS[None, :], 0, volume.shape[2] - 1
        )
        raw_patch = volume[
            z_index[:, :, None, None],
            y_index[:, None, :, None],
            x_index[:, None, None, :],
        ].astype(np.float32)
        patch = (raw_patch - frame_mean) / frame_std
        flat = patch.reshape(len(batch), -1)
        core = flat[:, _CORE_MASK]

        core_mean = core.mean(axis=1)
        core_std = core.std(axis=1)
        core_quantiles = np.quantile(
            core, (0.25, 0.5, 0.75), axis=1, method="linear"
        ).T.astype(np.float32)
        context_mean = flat.mean(axis=1)
        context_std = flat.std(axis=1)
        context_sum = flat.sum(axis=1)
        shell_count = flat.shape[1] - core.shape[1]
        shell_mean = (context_sum - core.sum(axis=1)) / float(shell_count)
        center = patch[:, Z_RADIUS, YX_RADIUS, YX_RADIUS]

        grad_z = np.diff(patch, axis=1)
        grad_y = np.diff(patch, axis=2)
        grad_x = np.diff(patch, axis=3)
        grad_abs = np.stack(
            (
                np.mean(np.abs(grad_z), axis=(1, 2, 3)),
                np.mean(np.abs(grad_y), axis=(1, 2, 3)),
                np.mean(np.abs(grad_x), axis=(1, 2, 3)),
            ),
            axis=1,
        )
        grad_rms = np.sqrt(
            (
                np.mean(np.square(grad_z), axis=(1, 2, 3))
                + np.mean(np.square(grad_y), axis=(1, 2, 3))
                + np.mean(np.square(grad_x), axis=(1, 2, 3))
            )
            / 3.0
        )

        lap_z = np.abs(
            patch[:, Z_RADIUS + 1, YX_RADIUS, YX_RADIUS]
            - 2.0 * center
            + patch[:, Z_RADIUS - 1, YX_RADIUS, YX_RADIUS]
        )
        lap_y = np.abs(
            patch[:, Z_RADIUS, YX_RADIUS + 1, YX_RADIUS]
            - 2.0 * center
            + patch[:, Z_RADIUS, YX_RADIUS - 1, YX_RADIUS]
        )
        lap_x = np.abs(
            patch[:, Z_RADIUS, YX_RADIUS, YX_RADIUS + 1]
            - 2.0 * center
            + patch[:, Z_RADIUS, YX_RADIUS, YX_RADIUS - 1]
        )

        floor = np.min(patch, axis=(1, 2, 3), keepdims=True)
        weights = patch - floor + np.float32(1e-6)
        mass = np.sum(weights, axis=(1, 2, 3))
        centroid_z = np.sum(weights * _REL_Z[None, ...], axis=(1, 2, 3)) / mass
        centroid_y = np.sum(weights * _REL_Y[None, ...], axis=(1, 2, 3)) / mass
        centroid_x = np.sum(weights * _REL_X[None, ...], axis=(1, 2, 3)) / mass

        grad_min = np.maximum(np.min(grad_abs, axis=1), 1e-6)
        gradient_anisotropy = np.log1p(np.max(grad_abs, axis=1) / grad_min)
        core_cv = core_std / np.maximum(np.abs(core_mean), 0.25)

        rows = np.column_stack(
            (
                center,
                core_mean,
                core_std,
                core.min(axis=1),
                core.max(axis=1),
                core_quantiles[:, 0],
                core_quantiles[:, 1],
                core_quantiles[:, 2],
                context_mean,
                context_std,
                flat.min(axis=1),
                flat.max(axis=1),
                core_mean - shell_mean,
                np.mean(flat > 0.0, axis=1),
                np.mean(flat > 1.0, axis=1),
                np.mean(flat <= center[:, None], axis=1),
                grad_abs[:, 0],
                grad_abs[:, 1],
                grad_abs[:, 2],
                grad_rms,
                lap_z,
                lap_y,
                lap_x,
                centroid_z / float(Z_RADIUS),
                centroid_y / float(YX_RADIUS),
                centroid_x / float(YX_RADIUS),
                _texture_entropy(flat),
                batch[:, 0] / max(volume.shape[0] - 1, 1),
                batch[:, 1] / max(volume.shape[1] - 1, 1),
                batch[:, 2] / max(volume.shape[2] - 1, 1),
                np.full(len(batch), float(time_fraction), dtype=np.float32),
                np.full(len(batch), np.log1p(frame_mean), dtype=np.float32),
                np.full(len(batch), np.log1p(frame_std), dtype=np.float32),
                core_cv,
                gradient_anisotropy,
            )
        ).astype(np.float32)
        if rows.shape != (len(batch), ENDPOINT_DIMENSION) or not np.isfinite(rows).all():
            raise FeatureContractError("endpoint feature shape or finiteness drift")
        output[start:stop] = rows

    return output


def action_features(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Apply the frozen DINOv2 eight-summary action map to raw-3D endpoints."""

    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != ENDPOINT_DIMENSION
    ):
        raise FeatureContractError(
            f"expected equal (N,{ENDPOINT_DIMENSION}) endpoint arrays"
        )
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise FeatureContractError("nonfinite endpoint feature")
    difference = source - target
    source_norm = np.linalg.norm(source, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    scale = np.sqrt(float(ENDPOINT_DIMENSION))
    cosine_denominator = np.maximum(source_norm * target_norm, 1e-8)
    centered_source = source - source.mean(axis=1, keepdims=True)
    centered_target = target - target.mean(axis=1, keepdims=True)
    correlation_denominator = np.maximum(
        np.linalg.norm(centered_source, axis=1)
        * np.linalg.norm(centered_target, axis=1),
        1e-8,
    )
    result = np.stack(
        (
            np.sum(source * target, axis=1) / cosine_denominator,
            np.linalg.norm(difference, axis=1) / scale,
            np.mean(np.abs(difference), axis=1),
            np.mean(source * target, axis=1),
            source_norm / scale,
            target_norm / scale,
            (source_norm - target_norm) / scale,
            np.sum(centered_source * centered_target, axis=1)
            / correlation_denominator,
        ),
        axis=1,
    ).astype(np.float32)
    if result.shape != (len(source), ACTION_DIMENSION) or not np.isfinite(result).all():
        raise FeatureContractError("invalid action-feature result")
    return result


def within_frame_permutation(
    values: np.ndarray,
    node_ids: np.ndarray,
    *,
    identity: str,
    frame: int,
    preregistration_sha256: str,
) -> np.ndarray:
    """Deterministic one-step cyclic permutation over unique nodes in a frame."""

    values = np.asarray(values, dtype=np.float32)
    node_ids = np.asarray(node_ids)
    if values.ndim != 2 or values.shape[1] != ENDPOINT_DIMENSION:
        raise FeatureContractError("endpoint feature shape drift")
    if node_ids.ndim != 1 or len(node_ids) != len(values):
        raise FeatureContractError("node-id alignment drift")
    if len(preregistration_sha256) != 64:
        raise FeatureContractError("preregistration SHA-256 must contain 64 hex characters")
    try:
        bytes.fromhex(preregistration_sha256)
    except ValueError as exc:
        raise FeatureContractError("invalid preregistration SHA-256") from exc
    if not len(values):
        return np.empty_like(values)

    unique_ids, first = np.unique(node_ids, return_index=True)
    vectors = values[first]
    seed = (
        "R-SEPARATION-FALSIFIER-44B6"
        f"{preregistration_sha256}{identity}{int(frame)}"
    )
    order = sorted(
        range(len(unique_ids)),
        key=lambda index: (
            hashlib.sha256(
                f"{seed}{int(unique_ids[index])}".encode("ascii")
            ).digest(),
            int(unique_ids[index]),
        ),
    )
    replacement: dict[int, np.ndarray] = {}
    for position, index in enumerate(order):
        replacement[int(unique_ids[index])] = vectors[order[(position + 1) % len(order)]]
    return np.stack([replacement[int(node)] for node in node_ids]).astype(np.float32)

