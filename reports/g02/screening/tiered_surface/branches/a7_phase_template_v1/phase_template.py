#!/usr/bin/env python3
"""Frozen primitives for A7-PHASE-TRACKLET-SWAP-V1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates


GRID_SIZE = 33
PHYSICAL_EXTENT_UM = 13.0
GRID_SPACING_UM = 0.40625
QUALITY_SHIFT_SCALE_UM = 1.625


class PhaseTemplateError(RuntimeError):
    """The frozen A7 feature/selection contract could not be preserved."""


@dataclass(frozen=True)
class PhaseEvidence:
    peak: float
    second_peak: float
    peak_ratio: float
    shift_zyx_um: tuple[float, float, float]
    shift_norm_um: float
    wrapped_peak_boundary: bool

    @property
    def quality(self) -> float:
        return (
            math.log(max(self.peak, 1.0e-12))
            + math.log(max(self.peak_ratio, 1.0e-12))
            - self.shift_norm_um / QUALITY_SHIFT_SCALE_UM
        )


def periodic_hann(size: int = GRID_SIZE) -> np.ndarray:
    if size < 2:
        raise PhaseTemplateError("periodic Hann requires size >= 2")
    axis = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(size) / size)
    return (
        axis[:, None, None] * axis[None, :, None] * axis[None, None, :]
    ).astype(np.float64)


WINDOW = periodic_hann()


def _parabolic_offset(previous: float, center: float, following: float) -> float:
    denominator = previous - 2.0 * center + following
    if not math.isfinite(denominator) or abs(denominator) <= 1.0e-15:
        return 0.0
    value = 0.5 * (previous - following) / denominator
    return float(np.clip(value, -0.5, 0.5))


def phase_evidence_from_ffts(
    source_fft: np.ndarray,
    destination_fft: np.ndarray,
) -> PhaseEvidence:
    if source_fft.shape != (GRID_SIZE,) * 3 or destination_fft.shape != source_fft.shape:
        raise PhaseTemplateError("matched 33^3 FFTs required")
    product = source_fft * np.conj(destination_fft)
    normalized = product / np.maximum(np.abs(product), 1.0e-12)
    correlation = np.fft.ifftn(normalized).real
    peak_flat = int(np.argmax(correlation))
    peak_index = tuple(
        int(value) for value in np.unravel_index(peak_flat, correlation.shape)
    )
    peak = float(correlation[peak_index])

    available = np.ones(correlation.shape, dtype=bool)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                available[
                    (peak_index[0] + dz) % correlation.shape[0],
                    (peak_index[1] + dy) % correlation.shape[1],
                    (peak_index[2] + dx) % correlation.shape[2],
                ] = False
    if not np.any(available):
        raise PhaseTemplateError("phase surface has no second-peak support")
    second = float(np.max(correlation[available]))
    ratio = float(peak / max(abs(second), 1.0e-12))

    shift_voxels = []
    for axis, (index, size) in enumerate(zip(peak_index, correlation.shape, strict=True)):
        previous_index = list(peak_index)
        following_index = list(peak_index)
        previous_index[axis] = (index - 1) % size
        following_index[axis] = (index + 1) % size
        offset = _parabolic_offset(
            float(correlation[tuple(previous_index)]),
            peak,
            float(correlation[tuple(following_index)]),
        )
        integer = index if index <= size // 2 else index - size
        shift_voxels.append(float(integer + offset))
    shift_um = tuple(float(value * GRID_SPACING_UM) for value in shift_voxels)
    norm_um = float(np.linalg.norm(np.asarray(shift_um, dtype=np.float64)))
    wrapped = any(index <= 1 or index >= size - 1 for index, size in zip(peak_index, correlation.shape, strict=True))
    values = np.asarray((peak, second, ratio, *shift_um, norm_um), dtype=np.float64)
    if not np.isfinite(values).all():
        raise PhaseTemplateError("non-finite phase evidence")
    return PhaseEvidence(
        peak=peak,
        second_peak=second,
        peak_ratio=ratio,
        shift_zyx_um=shift_um,
        shift_norm_um=norm_um,
        wrapped_peak_boundary=wrapped,
    )


def template_fft(template: np.ndarray) -> np.ndarray:
    value = np.asarray(template, dtype=np.float64)
    if value.shape != (GRID_SIZE,) * 3 or not np.isfinite(value).all():
        raise PhaseTemplateError("finite 33^3 template required")
    return np.fft.fftn(value * WINDOW)


def phase_evidence(source: np.ndarray, destination: np.ndarray) -> PhaseEvidence:
    return phase_evidence_from_ffts(template_fft(source), template_fft(destination))


def normalize_frame(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    if not math.isfinite(low) or not math.isfinite(high) or not high > low:
        raise PhaseTemplateError("invalid image quantiles")
    value = np.asarray(frame, dtype=np.float32)
    value = np.clip((value - low) / (high - low), 0.0, 1.0)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise PhaseTemplateError("normalized frame is malformed")
    return value


def sample_crops(
    frame: np.ndarray,
    centers_zyx: np.ndarray,
    scale_zyx_um: Sequence[float],
    *,
    chunk_size: int = 32,
) -> np.ndarray:
    """Batch trilinear physical 13-um crops with nearest-border replication."""
    image = np.asarray(frame, dtype=np.float32)
    centers = np.asarray(centers_zyx, dtype=np.float64)
    scale = np.asarray(scale_zyx_um, dtype=np.float64)
    if (
        image.ndim != 3
        or centers.ndim != 2
        or centers.shape[1:] != (3,)
        or scale.shape != (3,)
        or np.any(scale <= 0.0)
    ):
        raise PhaseTemplateError("crop geometry is malformed")
    offsets_um = np.linspace(
        -PHYSICAL_EXTENT_UM / 2.0,
        PHYSICAL_EXTENT_UM / 2.0,
        GRID_SIZE,
        dtype=np.float64,
    )
    offsets = [offsets_um / scale[axis] for axis in range(3)]
    mesh = np.meshgrid(*offsets, indexing="ij")
    flat_offsets = np.stack([value.reshape(-1) for value in mesh], axis=0)
    result = np.empty((len(centers), GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for start in range(0, len(centers), chunk_size):
        stop = min(len(centers), start + chunk_size)
        local = centers[start:stop]
        coordinates = local[:, :, None] + flat_offsets[None, :, :]
        coordinates = np.transpose(coordinates, (1, 0, 2)).reshape(3, -1)
        sampled = map_coordinates(
            image,
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        result[start:stop] = sampled.reshape(stop - start, GRID_SIZE, GRID_SIZE, GRID_SIZE)
    if not np.isfinite(result).all():
        raise PhaseTemplateError("non-finite sampled crop")
    return result


def graph_incidence(graph: Any) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for edge in graph.edges:
        source = int(edge.source_id)
        target = int(edge.target_id)
        outgoing[source].append(target)
        incoming[target].append(source)
    return (
        {key: tuple(sorted(value)) for key, value in incoming.items()},
        {key: tuple(sorted(value)) for key, value in outgoing.items()},
    )


def source_tracklet(
    graph: Any,
    incoming: Mapping[int, Sequence[int]],
    endpoint: int,
) -> tuple[int, ...]:
    if endpoint not in graph.nodes:
        raise PhaseTemplateError("source endpoint is absent from APNCC")
    path = [int(endpoint)]
    current = int(endpoint)
    for _ in range(2):
        predecessors = tuple(int(value) for value in incoming.get(current, ()))
        if len(predecessors) != 1:
            break
        previous = predecessors[0]
        if previous not in graph.nodes or int(graph.nodes[previous].t) + 1 != int(graph.nodes[current].t):
            break
        path.append(previous)
        current = previous
    return tuple(reversed(path))


def destination_tracklet(
    graph: Any,
    outgoing: Mapping[int, Sequence[int]],
    endpoint: int,
) -> tuple[int, ...]:
    if endpoint not in graph.nodes:
        raise PhaseTemplateError("destination endpoint is absent from APNCC")
    path = [int(endpoint)]
    current = int(endpoint)
    for _ in range(2):
        successors = tuple(int(value) for value in outgoing.get(current, ()))
        if len(successors) != 1:
            break
        following = successors[0]
        if following not in graph.nodes or int(graph.nodes[current].t) + 1 != int(graph.nodes[following].t):
            break
        path.append(following)
        current = following
    return tuple(path)


def touched_nodes(
    source: int,
    target: int,
    conflicts: Sequence[tuple[int, int]],
) -> frozenset[int]:
    result = {int(source), int(target)}
    for left, right in conflicts:
        result.add(int(left))
        result.add(int(right))
    return frozenset(result)


def strict_select(
    ordered_indices: Sequence[int],
    *,
    video: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    conflicts: Sequence[Sequence[tuple[int, int]]],
    budget: int,
    per_video_cap: int,
) -> np.ndarray:
    if budget < 0 or per_video_cap < 1:
        raise PhaseTemplateError("invalid edit budget")
    used: dict[int, set[int]] = defaultdict(set)
    counts: dict[int, int] = defaultdict(int)
    selected = []
    for raw_index in ordered_indices:
        if len(selected) >= budget:
            break
        index = int(raw_index)
        video_index = int(video[index])
        if counts[video_index] >= per_video_cap:
            continue
        nodes = touched_nodes(
            int(source[index]),
            int(target[index]),
            conflicts[index],
        )
        if used[video_index].intersection(nodes):
            continue
        used[video_index].update(nodes)
        counts[video_index] += 1
        selected.append(index)
    return np.asarray(selected, dtype=np.int64)
