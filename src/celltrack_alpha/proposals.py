"""Multi-source candidate center proposal generation."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.ndimage import correlate, gaussian_filter, maximum_filter
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


def _tile_shape_voxels(
    score_shape: tuple[int, int, int],
    scale_zyx: tuple[float, float, float],
) -> tuple[int, int, int]:
    # Around 26 um per side. This gives a spatially balanced fallback without
    # needing dataset-specific priors and keeps tile counts bounded for 64x256x256.
    target_um = 26.0
    return tuple(
        max(1, min(int(size), int(math.ceil(target_um / float(scale)))))
        for size, scale in zip(score_shape, scale_zyx, strict=True)
    )


def _stratified_local_peaks(
    score: np.ndarray,
    source: str,
    threshold_quantile: float,
    scale_zyx: tuple[float, float, float],
    min_distance_um: float,
    max_points: int,
    coverage_first: bool = False,
) -> list[dict[str, Any]]:
    """Select local maxima with a per-3D-tile quota.

    Standard global top-k peak selection fails when sparse labels mark moderate
    cells near a frame edge or in dim regions. This source keeps the strongest
    local maxima per spatial tile, so every region can contribute candidates.
    """

    if max_points <= 0:
        return []
    local = maximum_filter(score, size=_nms_size(scale_zyx, min_distance_um), mode="nearest")
    threshold = float(np.quantile(score, threshold_quantile))
    mask = (score >= threshold) & (score == local)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return []
    values = score[coords[:, 0], coords[:, 1], coords[:, 2]]
    tile_shape = np.asarray(_tile_shape_voxels(score.shape, scale_zyx), dtype=np.int64)
    tile_ids = coords // tile_shape
    n_tiles = max(1, len({tuple(int(v) for v in row) for row in tile_ids}))
    per_tile = max(1, int(math.ceil(max_points / n_tiles)))
    by_tile: dict[tuple[int, int, int], list[int]] = {}
    for index, tile in enumerate(tile_ids):
        by_tile.setdefault(tuple(int(v) for v in tile), []).append(index)

    for tile in sorted(by_tile):
        by_tile[tile].sort(key=lambda idx: (-float(values[idx]), int(coords[idx, 0]), int(coords[idx, 1]), int(coords[idx, 2])))

    selected: list[int] = []
    coverage_rank: dict[int, int] = {}
    if coverage_first:
        max_rank = max(len(indices) for indices in by_tile.values())
        for rank in range(max_rank):
            for tile in sorted(by_tile):
                indices = by_tile[tile]
                if rank >= len(indices):
                    continue
                idx = indices[rank]
                selected.append(idx)
                coverage_rank[idx] = rank
                if len(selected) >= max_points:
                    break
            if len(selected) >= max_points:
                break
    else:
        for tile in sorted(by_tile):
            selected.extend(by_tile[tile][:per_tile])
        selected.sort(key=lambda idx: (-float(values[idx]), int(coords[idx, 0]), int(coords[idx, 1]), int(coords[idx, 2])))
        selected = selected[:max_points]
        coverage_rank = {idx: 0 for idx in selected}

    def _priority_score(idx: int) -> float:
        if coverage_first:
            return float(1.10 - 0.001 * coverage_rank.get(idx, 0) + 0.000001 * np.clip(values[idx], 0.0, 1.0))
        return float(1.0 + 0.05 * np.clip(values[idx], 0.0, 1.0))

    return [
        {
            "coord": tuple(int(v) for v in coords[idx]),
            # Give balanced peaks source priority during fusion. They are not
            # confidence-calibrated; downstream node score clips to [0, 1].
            "score": _priority_score(idx),
            "source": f"{source}@q{threshold_quantile}",
        }
        for idx in selected
    ]


def _subtile_top_peaks(
    score: np.ndarray,
    source: str,
    threshold_quantile: float,
    scale_zyx: tuple[float, float, float],
    max_points: int,
    target_um: float = 13.0,
) -> list[dict[str, Any]]:
    if max_points <= 0:
        return []
    threshold = float(np.quantile(score, threshold_quantile))
    tile_shape = tuple(
        max(1, min(int(size), int(math.ceil(float(target_um) / float(scale)))))
        for size, scale in zip(score.shape, scale_zyx, strict=True)
    )
    raw: list[dict[str, Any]] = []
    z_size, y_size, x_size = score.shape
    for z0 in range(0, z_size, tile_shape[0]):
        for y0 in range(0, y_size, tile_shape[1]):
            for x0 in range(0, x_size, tile_shape[2]):
                patch = score[
                    z0 : min(z_size, z0 + tile_shape[0]),
                    y0 : min(y_size, y0 + tile_shape[1]),
                    x0 : min(x_size, x0 + tile_shape[2]),
                ]
                if patch.size == 0:
                    continue
                local_index = int(np.argmax(patch))
                value = float(patch.reshape(-1)[local_index])
                if value < threshold:
                    continue
                dz, dy, dx = np.unravel_index(local_index, patch.shape)
                raw.append(
                    {
                        "coord": (int(z0 + dz), int(y0 + dy), int(x0 + dx)),
                        "score": float(1.12 + 0.000001 * np.clip(value, 0.0, 1.0)),
                        "source": f"{source}@q{threshold_quantile}",
                    }
                )
    raw.sort(key=lambda row: (-float(row["score"]), int(row["coord"][0]), int(row["coord"][1]), int(row["coord"][2])))
    return raw[:max_points]


def _best_score_coord_in_radius(
    score: np.ndarray,
    center_zyx: tuple[int, int, int],
    scale_zyx: tuple[float, float, float],
    radius_um: float,
) -> tuple[tuple[int, int, int], float] | None:
    center = np.asarray(center_zyx, dtype=np.int64)
    radii = np.asarray([max(1, int(math.ceil(radius_um / float(scale)))) for scale in scale_zyx], dtype=np.int64)
    lows = np.maximum(center - radii, 0)
    highs = np.minimum(center + radii + 1, np.asarray(score.shape, dtype=np.int64))
    patch = score[lows[0] : highs[0], lows[1] : highs[1], lows[2] : highs[2]]
    if patch.size == 0:
        return None
    local_coords = np.indices(patch.shape).reshape(3, -1).T + lows[None, :]
    delta = (local_coords.astype(np.float32) - center.astype(np.float32)[None, :]) * np.asarray(scale_zyx, dtype=np.float32)
    inside = np.sqrt(np.sum(delta * delta, axis=1)) <= float(radius_um)
    if not np.any(inside):
        return None
    values = patch.reshape(-1)[inside]
    coords = local_coords[inside]
    best_idx = int(np.argmax(values))
    coord = tuple(int(v) for v in coords[best_idx])
    return coord, float(values[best_idx])


def _refined_local_score_peaks(
    seed_nodes: list[CandidateNode],
    score_maps: dict[str, np.ndarray],
    scale_zyx: tuple[float, float, float],
    radius_um: float = 13.0,
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for node in seed_nodes:
        for name, score in score_maps.items():
            best = _best_score_coord_in_radius(score, node.coord_zyx, scale_zyx, radius_um)
            if best is None:
                continue
            coord, value = best
            raw.append(
                {
                    "coord": coord,
                    "score": float(1.16 + 0.000001 * np.clip(value, 0.0, 1.0)),
                    "source": f"refine_local_score:{name}@r{radius_um:g}",
                }
            )
    return raw


def _high_quantiles(config: RouteConfig, floor: float = 0.995) -> tuple[float, ...]:
    high = tuple(float(q) for q in config.threshold_quantiles if float(q) >= floor)
    return high or (float(max(config.threshold_quantiles)),)


def _shifted_peak_rows(
    rows: list[dict[str, Any]],
    offset_zyx: tuple[int, int, int],
    source: str,
    score_base: float,
    volume_shape: tuple[int, int, int],
) -> list[dict[str, Any]]:
    dz, dy, dx = (int(v) for v in offset_zyx)
    z_max, y_max, x_max = (int(v) for v in volume_shape)
    shifted: list[dict[str, Any]] = []
    for row in rows:
        z, y, x = (int(v) for v in row["coord"])
        coord = (
            min(max(z + dz, 0), z_max - 1),
            min(max(y + dy, 0), y_max - 1),
            min(max(x + dx, 0), x_max - 1),
        )
        shifted.append(
            {
                "coord": coord,
                "score": float(score_base + 0.000001 * np.clip(float(row["score"]), 0.0, 1.0)),
                "source": f"{source}:{row['source']}",
            }
        )
    return shifted


def _template_score_peaks(
    norm: np.ndarray,
    proposal_policy: dict[str, Any] | None,
    scale_zyx: tuple[float, float, float],
    config: RouteConfig,
) -> list[dict[str, Any]]:
    if not proposal_policy:
        return []
    templates = proposal_policy.get("templates", [])
    if not isinstance(templates, list):
        return []
    raw: list[dict[str, Any]] = []
    for template in templates:
        if not isinstance(template, dict):
            continue
        try:
            kernel = np.asarray(template.get("kernel"), dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if kernel.ndim != 3 or kernel.size == 0 or not np.all(np.isfinite(kernel)):
            continue
        kernel = kernel - float(np.mean(kernel))
        norm_factor = float(np.sqrt(np.sum(kernel * kernel)))
        if norm_factor <= 1e-6:
            continue
        kernel = kernel / norm_factor
        response = correlate(norm.astype(np.float32, copy=False), kernel, mode="nearest")
        response = _robust_unit(response)
        quantiles = tuple(float(value) for value in template.get("score_quantiles", [max(config.threshold_quantiles)]))
        max_points = int(template.get("max_points", config.max_points_per_t))
        min_distance_um = float(template.get("min_distance_um", config.min_distance_um))
        source = str(template.get("source", f"template_score:{template.get('name', 'template')}"))
        peaks = _extract_peaks(
            response,
            source,
            quantiles,
            scale_zyx,
            min_distance_um,
            max_points,
        )
        for row in peaks:
            row["score"] = float(1.18 + 0.000001 * np.clip(float(row["score"]), 0.0, 1.0))
        raw.extend(peaks)
    return raw


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
    proposal_policy: dict[str, Any] | None = None,
) -> tuple[list[CandidateNode], int]:
    """Generate fused candidates for one 3D timepoint."""

    volume = _validate_volume(volume)
    baseline = BaselineConfig(polarity="both")
    norm = normalize_volume(volume, attrs, baseline)
    raw: list[dict[str, Any]] = []

    if "template_score" in config.detector_sources:
        raw.extend(_template_score_peaks(norm, proposal_policy, scale_zyx, config))
    if "dog_bright" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="bright")))
        raw.extend(_extract_peaks(score, "dog_bright", config.threshold_quantiles, scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "dog_dark" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="dark")))
        raw.extend(_extract_peaks(score, "dog_dark", config.threshold_quantiles, scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "zshift_dog_dark_m5" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="dark")))
        shifted_cap = max(8, int(config.max_points_per_t) // 4)
        rows = _extract_peaks(score, "dog_dark", _high_quantiles(config), scale_zyx, config.min_distance_um, shifted_cap)
        raw.extend(_shifted_peak_rows(rows, (-5, 0, 0), "zshift_dog_dark_m5", 1.07, volume.shape))
    if "zshift_dog_bright_m3" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="bright")))
        shifted_cap = max(8, int(config.max_points_per_t) // 4)
        rows = _extract_peaks(score, "dog_bright", _high_quantiles(config), scale_zyx, config.min_distance_um, shifted_cap)
        raw.extend(_shifted_peak_rows(rows, (-3, 0, 0), "zshift_dog_bright_m3", 1.06, volume.shape))
    if "zshift_dog_dark_m5_mid" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="dark")))
        shifted_cap = max(8, int(config.max_points_per_t) // 6)
        rows = _extract_peaks(score, "dog_dark", _high_quantiles(config), scale_zyx, config.min_distance_um, shifted_cap)
        raw.extend(_shifted_peak_rows(rows, (-5, 0, 0), "zshift_dog_dark_m5_mid", 1.035, volume.shape))
    if "zshift_dog_bright_m3_mid" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="bright")))
        shifted_cap = max(8, int(config.max_points_per_t) // 6)
        rows = _extract_peaks(score, "dog_bright", _high_quantiles(config), scale_zyx, config.min_distance_um, shifted_cap)
        raw.extend(_shifted_peak_rows(rows, (-3, 0, 0), "zshift_dog_bright_m3_mid", 1.03, volume.shape))
    if "dog_abs" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="both")))
        raw.extend(_extract_peaks(score, "dog_abs", config.threshold_quantiles, scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "local_contrast" in config.detector_sources:
        local_sigma = _sigma_voxels(2.5, scale_zyx)
        score = _robust_unit(norm - gaussian_filter(norm, local_sigma, mode="nearest"))
        raw.extend(_extract_peaks(score, "local_contrast", config.threshold_quantiles, scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "tile_raw" in config.detector_sources:
        score = _robust_unit(gaussian_filter(norm, _sigma_voxels(0.7, scale_zyx), mode="nearest"))
        raw.extend(_stratified_local_peaks(score, "tile_raw", min(config.threshold_quantiles), scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "tile_contrast" in config.detector_sources:
        local_sigma = _sigma_voxels(2.5, scale_zyx)
        score = _robust_unit(norm - gaussian_filter(norm, local_sigma, mode="nearest"))
        raw.extend(_stratified_local_peaks(score, "tile_contrast", min(config.threshold_quantiles), scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "tile_dog_bright" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="bright")))
        raw.extend(_stratified_local_peaks(score, "tile_dog_bright", min(config.threshold_quantiles), scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "tile_dog_dark" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="dark")))
        raw.extend(_stratified_local_peaks(score, "tile_dog_dark", min(config.threshold_quantiles), scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "tile_dog_abs" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="both")))
        raw.extend(_stratified_local_peaks(score, "tile_dog_abs", min(config.threshold_quantiles), scale_zyx, config.min_distance_um, config.max_points_per_t))
    if "cover_contrast" in config.detector_sources:
        local_sigma = _sigma_voxels(2.5, scale_zyx)
        score = _robust_unit(norm - gaussian_filter(norm, local_sigma, mode="nearest"))
        raw.extend(_stratified_local_peaks(score, "cover_contrast", min(config.threshold_quantiles), scale_zyx, config.min_distance_um, config.max_points_per_t, coverage_first=True))
    if "cover_raw" in config.detector_sources:
        score = _robust_unit(norm)
        raw.extend(_stratified_local_peaks(score, "cover_raw", min(config.threshold_quantiles), scale_zyx, config.min_distance_um, config.max_points_per_t, coverage_first=True))
    if "cover_smooth" in config.detector_sources:
        score = _robust_unit(gaussian_filter(norm, _sigma_voxels(0.7, scale_zyx), mode="nearest"))
        raw.extend(_stratified_local_peaks(score, "cover_smooth", min(config.threshold_quantiles), scale_zyx, config.min_distance_um, config.max_points_per_t, coverage_first=True))
    if "cover_dog_bright" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="bright")))
        raw.extend(_stratified_local_peaks(score, "cover_dog_bright", min(config.threshold_quantiles), scale_zyx, config.min_distance_um, config.max_points_per_t, coverage_first=True))
    if "cover_dog_dark" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="dark")))
        raw.extend(_stratified_local_peaks(score, "cover_dog_dark", min(config.threshold_quantiles), scale_zyx, config.min_distance_um, config.max_points_per_t, coverage_first=True))
    if "cover_dog_abs" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="both")))
        raw.extend(_stratified_local_peaks(score, "cover_dog_abs", min(config.threshold_quantiles), scale_zyx, config.min_distance_um, config.max_points_per_t, coverage_first=True))
    if "subtile_contrast" in config.detector_sources:
        local_sigma = _sigma_voxels(2.5, scale_zyx)
        score = _robust_unit(norm - gaussian_filter(norm, local_sigma, mode="nearest"))
        raw.extend(_subtile_top_peaks(score, "subtile_contrast", min(config.threshold_quantiles), scale_zyx, config.max_points_per_t))
    if "subtile_dog_bright" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="bright")))
        raw.extend(_subtile_top_peaks(score, "subtile_dog_bright", min(config.threshold_quantiles), scale_zyx, config.max_points_per_t))
    if "subtile_dog_dark" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="dark")))
        raw.extend(_subtile_top_peaks(score, "subtile_dog_dark", min(config.threshold_quantiles), scale_zyx, config.max_points_per_t))
    if "subtile_dog_abs" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="both")))
        raw.extend(_subtile_top_peaks(score, "subtile_dog_abs", min(config.threshold_quantiles), scale_zyx, config.max_points_per_t))
    if "hmax" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="both")))
        raw.extend(_hmax_peaks(score, h=0.12, source="hmax", max_points=config.max_points_per_t))
    if "components" in config.detector_sources:
        score = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="both")))
        raw.extend(_component_centroids(score, quantile=min(config.threshold_quantiles), max_points=config.max_points_per_t // 2))
    if "refine_local_score" in config.detector_sources and raw:
        seed_nodes, _ = _fuse_raw_candidates(raw, dataset, t, scale_zyx, config, 1, volume)
        local_sigma = _sigma_voxels(2.5, scale_zyx)
        refine_maps = {
            "local_contrast": _robust_unit(norm - gaussian_filter(norm, local_sigma, mode="nearest")),
            "dog_bright": _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="bright"))),
            "dog_abs": _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="both"))),
        }
        raw.extend(_refined_local_score_peaks(seed_nodes, refine_maps, scale_zyx))

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
    proposal_policy: dict[str, Any] | None = None,
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
            proposal_policy=proposal_policy,
        )
        nodes_by_t.append(nodes)
        all_nodes.extend(nodes)
    annotate_local_density(nodes_by_t, scale_zyx)
    return nodes_by_t, all_nodes
