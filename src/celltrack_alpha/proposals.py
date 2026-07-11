"""Multi-source candidate center proposal generation."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

from celltrack_baseline.rule_based import BaselineConfig, dog_response, normalize_volume, parse_scale

from .types import CandidateNode, RouteConfig


def _validate_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume)
    if volume.ndim != 3 or any(int(value) <= 0 for value in volume.shape):
        raise ValueError(
            f"each timepoint must be a non-empty three-dimensional Z/Y/X array, got {volume.shape!r}"
        )
    if not np.issubdtype(volume.dtype, np.number) or np.issubdtype(
        volume.dtype, np.complexfloating
    ):
        raise ValueError(f"volume dtype must be real numeric, got {volume.dtype}")
    if not np.all(np.isfinite(volume)):
        raise ValueError("volume contains NaN or infinite intensity values")
    return volume


def _sigma_voxels(sigma_um: float, scale_zyx: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(max(0.35, sigma_um / scale) for scale in scale_zyx)


def _nms_size(scale_zyx: tuple[float, float, float], min_distance_um: float) -> tuple[int, int, int]:
    radii = [max(1, int(math.ceil(min_distance_um / scale))) for scale in scale_zyx]
    return tuple(2 * r + 1 for r in radii)


def _robust_unit(x: np.ndarray, low_q: float = 0.001, high_q: float = 0.999) -> np.ndarray:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, dtype=np.float32)
    lo, hi = np.quantile(finite, [low_q, high_q]).astype(np.float32)
    if hi <= lo:
        hi = lo + np.float32(1.0)
    return np.clip((x.astype(np.float32, copy=False) - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)


def _greedy_physical_nms(
    coords: np.ndarray,
    scores: np.ndarray,
    scale_zyx: tuple[float, float, float],
    min_distance_um: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(coords) == 0:
        return coords, scores
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    scale = np.asarray(scale_zyx, dtype=np.float32)
    min_sq = float(min_distance_um * min_distance_um)
    for idx in order:
        coord = coords[idx].astype(np.float32)
        keep = True
        for chosen in selected:
            delta = (coord - coords[chosen].astype(np.float32)) * scale
            if float(np.dot(delta, delta)) < min_sq:
                keep = False
                break
        if keep:
            selected.append(int(idx))
            if len(selected) >= max_points:
                break
    return coords[selected], scores[selected]


def _extract_peaks(
    score: np.ndarray,
    source: str,
    quantiles: tuple[float, ...],
    scale_zyx: tuple[float, float, float],
    min_distance_um: float,
    max_points: int,
) -> list[dict[str, Any]]:
    local = maximum_filter(score, size=_nms_size(scale_zyx, min_distance_um), mode="nearest")
    raw: list[dict[str, Any]] = []
    per_quantile_cap = max(max_points, 1)
    for quantile in quantiles:
        threshold = float(np.quantile(score, quantile))
        coords = np.argwhere((score >= threshold) & (score == local))
        if len(coords) == 0:
            continue
        values = score[coords[:, 0], coords[:, 1], coords[:, 2]]
        coords, values = _greedy_physical_nms(
            coords, values, scale_zyx, min_distance_um, per_quantile_cap
        )
        for coord, value in zip(coords, values, strict=True):
            raw.append(
                {
                    "coord": tuple(int(v) for v in coord),
                    "score": float(value),
                    "source": f"{source}@q{quantile}",
                }
            )
    return raw


def _component_centroids(
    score: np.ndarray,
    quantile: float,
    max_points: int,
) -> list[dict[str, Any]]:
    try:
        from skimage.measure import label, regionprops
    except Exception:
        return []
    threshold = float(np.quantile(score, quantile))
    labels = label(score >= threshold)
    props = regionprops(labels, intensity_image=score)
    def _max_intensity(prop: Any) -> float:
        if hasattr(prop, "intensity_max"):
            return float(prop.intensity_max)
        return float(prop.max_intensity)

    def _weighted_centroid(prop: Any) -> tuple[float, float, float]:
        if hasattr(prop, "centroid_weighted"):
            return tuple(prop.centroid_weighted)
        return tuple(prop.weighted_centroid)

    props = sorted(props, key=_max_intensity, reverse=True)[:max_points]
    raw: list[dict[str, Any]] = []
    for prop in props:
        z, y, x = [int(round(v)) for v in _weighted_centroid(prop)]
        raw.append(
            {
                "coord": (z, y, x),
                "score": _max_intensity(prop),
                "source": f"components@q{quantile}",
            }
        )
    return raw


def _hmax_peaks(
    score: np.ndarray,
    h: float,
    source: str,
    max_points: int,
) -> list[dict[str, Any]]:
    try:
        from skimage.morphology import h_maxima
    except Exception:
        return []
    mask = h_maxima(score.astype(np.float32, copy=False), h=float(h))
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return []
    values = score[coords[:, 0], coords[:, 1], coords[:, 2]]
    order = np.argsort(values)[::-1][:max_points]
    return [
        {"coord": tuple(int(v) for v in coords[idx]), "score": float(values[idx]), "source": source}
        for idx in order
    ]


def _fuse_raw_candidates(
    raw: list[dict[str, Any]],
    dataset: str,
    t: int,
    scale_zyx: tuple[float, float, float],
    config: RouteConfig,
    next_node_id: int,
    volume: np.ndarray,
) -> tuple[list[CandidateNode], int]:
    if not raw:
        return [], next_node_id
    raw = sorted(raw, key=lambda row: float(row["score"]), reverse=True)
    scale = np.asarray(scale_zyx, dtype=np.float32)
    radius_sq = float(config.fusion_radius_um * config.fusion_radius_um)
    selected: list[dict[str, Any]] = []
    for row in raw:
        coord = np.asarray(row["coord"], dtype=np.float32)
        merged = False
        for existing in selected:
            other = np.asarray(existing["coord"], dtype=np.float32)
            delta = (coord - other) * scale
            if float(np.dot(delta, delta)) <= radius_sq:
                existing["score"] = max(float(existing["score"]), float(row["score"]))
                existing["sources"].add(str(row["source"]))
                merged = True
                break
        if not merged:
            selected.append(
                {
                    "coord": tuple(int(v) for v in row["coord"]),
                    "score": float(row["score"]),
                    "sources": {str(row["source"])},
                }
            )
        if len(selected) >= config.max_points_per_t:
            break

    z_max, y_max, x_max = volume.shape
    nodes: list[CandidateNode] = []
    for row in selected[: config.max_points_per_t]:
        z, y, x = row["coord"]
        z = min(max(int(z), 0), z_max - 1)
        y = min(max(int(y), 0), y_max - 1)
        x = min(max(int(x), 0), x_max - 1)
        nodes.append(
            CandidateNode(
                dataset=dataset,
                node_id=next_node_id,
                t=int(t),
                z=z,
                y=y,
                x=x,
                score=float(np.clip(row["score"], 0.0, 1.0)),
                sources=tuple(sorted(row["sources"])),
                radius_um=float(config.fusion_radius_um),
                uncertainty=float(1.0 - np.clip(row["score"], 0.0, 1.0)),
                intensity=float(volume[z, y, x]),
            )
        )
        next_node_id += 1
    return nodes, next_node_id


def detect_candidates_one_timepoint(
    volume: np.ndarray,
    dataset: str,
    t: int,
    attrs: dict[str, Any] | None,
    scale_zyx: tuple[float, float, float],
    config: RouteConfig,
    next_node_id: int,
) -> tuple[list[CandidateNode], int]:
    """Generate fused candidates for one 3D timepoint."""

    volume = _validate_volume(volume)
    baseline = BaselineConfig(polarity="both")
    norm = normalize_volume(volume, attrs, baseline)
    raw: list[dict[str, Any]] = []

    if "dog_bright" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="bright")))
        raw.extend(_extract_peaks(score, "dog_bright", config.threshold_quantiles, scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "dog_dark" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="dark")))
        raw.extend(_extract_peaks(score, "dog_dark", config.threshold_quantiles, scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "dog_abs" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="both")))
        raw.extend(_extract_peaks(score, "dog_abs", config.threshold_quantiles, scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "local_contrast" in config.detector_sources:
        local_sigma = _sigma_voxels(2.5, scale_zyx)
        score = _robust_unit(norm - gaussian_filter(norm, local_sigma, mode="nearest"))
        raw.extend(_extract_peaks(score, "local_contrast", config.threshold_quantiles, scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "hmax" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="both")))
        raw.extend(_hmax_peaks(score, h=0.12, source="hmax", max_points=config.max_points_per_t))
    if "components" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="both")))
        raw.extend(_component_centroids(score, quantile=min(config.threshold_quantiles), max_points=config.max_points_per_t // 2))

    return _fuse_raw_candidates(raw, dataset, t, scale_zyx, config, next_node_id, volume)


def annotate_local_density(
    nodes_by_t: list[list[CandidateNode]],
    scale_zyx: tuple[float, float, float],
    radius_um: float = 7.0,
) -> None:
    """Populate local density in place for downstream edge costs."""

    scale = np.asarray(scale_zyx, dtype=np.float32)
    for nodes in nodes_by_t:
        if len(nodes) < 2:
            continue
        coords = np.asarray([node.coord_zyx for node in nodes], dtype=np.float32) * scale
        tree = cKDTree(coords)
        counts = tree.query_ball_point(coords, r=float(radius_um), return_length=True)
        for node, count in zip(nodes, counts, strict=True):
            node.local_density = float(max(0, int(count) - 1))


def generate_candidates(
    dataset: str,
    array: Any,
    attrs: dict[str, Any] | None,
    config: RouteConfig,
    start: int = 0,
    end: int | None = None,
    progress: bool = False,
) -> tuple[list[list[CandidateNode]], list[CandidateNode]]:
    """Generate candidates for a dataset/window."""

    total_t = int(array.shape[0])
    end = total_t if end is None else min(int(end), total_t)
    start = max(0, min(int(start), max(0, end - 1)))
    scale_zyx = parse_scale(attrs)
    nodes_by_t: list[list[CandidateNode]] = []
    all_nodes: list[CandidateNode] = []
    next_node_id = 1
    for t in range(start, end):
        if progress:
            print(f"{dataset}: proposals t={t + 1}/{end}", flush=True)
        volume = np.asarray(array[t])
        nodes, next_node_id = detect_candidates_one_timepoint(
            volume=volume,
            dataset=dataset,
            t=t,
            attrs=attrs,
            scale_zyx=scale_zyx,
            config=config,
            next_node_id=next_node_id,
        )
        nodes_by_t.append(nodes)
        all_nodes.extend(nodes)
    annotate_local_density(nodes_by_t, scale_zyx)
    return nodes_by_t, all_nodes
