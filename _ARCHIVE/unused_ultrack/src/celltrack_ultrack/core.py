"""Ultrack integration for ZIP-backed cell tracking experiments.

This module keeps the public surface small and lazy-loads Ultrack so that the
classical input builder can be smoke-tested before solver dependencies exist.
Images are expected to be shaped ``(T, Z, Y, X)``.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    gaussian_gradient_magnitude,
    maximum_filter,
)

from celltrack_baseline.rule_based import BaselineConfig, dog_response, normalize_volume, parse_scale


@dataclass(frozen=True)
class UltrackConfig:
    """Configuration for the first Ultrack competition prototype."""

    scale_zyx: tuple[float, float, float] | None = None
    sigmas_um: tuple[float, ...] = (0.8, 1.2, 1.8)
    dog_ratio: float = 1.7
    q_low: float = 0.01
    q_high: float = 0.999
    foreground_quantile: float = 0.9975
    foreground_threshold: float = 0.5
    local_contrast_sigma_um: float = 2.5
    seed_h: float = 0.10
    seed_dilation_um: float = 2.0
    contour_preset: str = "medium"
    min_frontier: float | None = None
    min_cell_radius_um: float = 2.0
    max_cell_radius_um: float = 4.0
    min_area: int | None = None
    max_area: int | None = None
    max_link_um: float = 8.0
    max_neighbors: int = 5
    solver_name: str = "CBC"
    window_size: int | None = 12
    overlap_size: int = 2
    solution_gap: float = 0.01
    time_limit: int = 3600
    n_workers: int = 1
    n_threads: int = -1
    working_dir: str | Path | None = None
    database_file_name: str = "ultrack.db"
    max_segments_per_time: int = 100_000

    def __post_init__(self) -> None:
        validate_ultrack_config(self)


_CONTOUR_PRESETS: dict[str, dict[str, float]] = {
    "low": {"gradient_quantile": 0.985, "frontier_weight": 0.55, "min_frontier": 0.05},
    "medium": {"gradient_quantile": 0.990, "frontier_weight": 0.75, "min_frontier": 0.10},
    "high": {"gradient_quantile": 0.995, "frontier_weight": 0.90, "min_frontier": 0.18},
}


def validate_ultrack_config(config: UltrackConfig) -> None:
    """Validate every public ``UltrackConfig`` field before backend work."""

    if not isinstance(config, UltrackConfig):
        raise TypeError(f"config must be an UltrackConfig, got {type(config).__name__}")

    checked: set[str] = set()

    def value(name: str) -> Any:
        checked.add(name)
        return getattr(config, name)

    def require_number(
        name: str,
        *,
        minimum: float | None = None,
        strict: bool = False,
        maximum: float | None = None,
    ) -> float:
        current = value(name)
        if isinstance(current, bool) or not isinstance(current, Real):
            raise ValueError(f"{name} must be a finite number, got {current!r}")
        resolved = float(current)
        if not math.isfinite(resolved):
            raise ValueError(f"{name} must be a finite number, got {current!r}")
        if minimum is not None and (resolved <= minimum if strict else resolved < minimum):
            relation = "greater than" if strict else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}, got {current!r}")
        if maximum is not None and resolved > maximum:
            raise ValueError(f"{name} must be at most {maximum}, got {current!r}")
        return resolved

    def require_optional_number(
        name: str,
        *,
        minimum: float,
        maximum: float | None = None,
    ) -> float | None:
        current = value(name)
        if current is None:
            return None
        if isinstance(current, bool) or not isinstance(current, Real):
            raise ValueError(f"{name} must be None or a finite number, got {current!r}")
        resolved = float(current)
        if (
            not math.isfinite(resolved)
            or resolved < minimum
            or (maximum is not None and resolved > maximum)
        ):
            upper = "" if maximum is None else f" and at most {maximum}"
            raise ValueError(
                f"{name} must be None or a finite number at least {minimum}{upper}, "
                f"got {current!r}"
            )
        return resolved

    def require_integer(name: str, *, minimum: int) -> int:
        current = value(name)
        if isinstance(current, bool) or not isinstance(current, Integral) or int(current) < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}, got {current!r}")
        return int(current)

    def require_optional_integer(name: str, *, minimum: int) -> int | None:
        current = value(name)
        if current is None:
            return None
        if isinstance(current, bool) or not isinstance(current, Integral) or int(current) < minimum:
            raise ValueError(
                f"{name} must be None or an integer at least {minimum}, got {current!r}"
            )
        return int(current)

    scale = value("scale_zyx")
    if scale is not None and (
        not isinstance(scale, tuple)
        or len(scale) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, Real)
            or not math.isfinite(float(item))
            or float(item) <= 0.0
            for item in scale
        )
    ):
        raise ValueError("scale_zyx must be None or a Z/Y/X tuple of three positive finite values")

    sigmas = value("sigmas_um")
    if (
        not isinstance(sigmas, tuple)
        or not sigmas
        or any(
            isinstance(item, bool)
            or not isinstance(item, Real)
            or not math.isfinite(float(item))
            or float(item) <= 0.0
            for item in sigmas
        )
    ):
        raise ValueError("sigmas_um must be a non-empty tuple of positive finite values")

    require_number("dog_ratio", minimum=1.0, strict=True)
    q_low = require_number("q_low", minimum=0.0, maximum=1.0)
    q_high = require_number("q_high", minimum=0.0, maximum=1.0)
    require_number("foreground_quantile", minimum=0.0, maximum=1.0)
    require_number("foreground_threshold", minimum=0.0, maximum=1.0)
    require_number("local_contrast_sigma_um", minimum=0.0, strict=True)
    require_number("seed_h", minimum=0.0, strict=True)
    require_number("seed_dilation_um", minimum=0.0)

    contour_preset = value("contour_preset")
    if not isinstance(contour_preset, str) or contour_preset not in _CONTOUR_PRESETS:
        raise ValueError(
            f"contour_preset must be one of {sorted(_CONTOUR_PRESETS)}, got {contour_preset!r}"
        )
    require_optional_number("min_frontier", minimum=0.0, maximum=1.0)

    min_radius = require_number("min_cell_radius_um", minimum=0.0, strict=True)
    max_radius = require_number("max_cell_radius_um", minimum=0.0, strict=True)
    require_optional_integer("min_area", minimum=1)
    require_optional_integer("max_area", minimum=1)
    require_number("max_link_um", minimum=0.0, strict=True)
    require_integer("max_neighbors", minimum=1)

    solver_name = value("solver_name")
    if not isinstance(solver_name, str) or solver_name not in {"CBC", "GUROBI"}:
        raise ValueError(f"solver_name must be 'CBC' or 'GUROBI', got {solver_name!r}")
    window_size = require_optional_integer("window_size", minimum=1)
    overlap_size = require_integer("overlap_size", minimum=0)
    require_number("solution_gap", minimum=0.0)
    require_integer("time_limit", minimum=1)
    require_integer("n_workers", minimum=1)

    n_threads = value("n_threads")
    if (
        isinstance(n_threads, bool)
        or not isinstance(n_threads, Integral)
        or int(n_threads) == 0
        or int(n_threads) < -1
    ):
        raise ValueError(f"n_threads must be -1 or a positive integer, got {n_threads!r}")

    working_dir = value("working_dir")
    if working_dir is not None and (
        not isinstance(working_dir, (str, Path))
        or (isinstance(working_dir, str) and not working_dir.strip())
    ):
        raise ValueError(f"working_dir must be None, a non-empty string, or a Path, got {working_dir!r}")

    database_file_name = value("database_file_name")
    if not isinstance(database_file_name, str) or not database_file_name.strip():
        raise ValueError(
            f"database_file_name must be a non-empty string, got {database_file_name!r}"
        )
    require_integer("max_segments_per_time", minimum=1)

    if q_low >= q_high:
        raise ValueError("q_low must be less than q_high")
    if min_radius > max_radius:
        raise ValueError("min_cell_radius_um must not exceed max_cell_radius_um")
    if window_size is not None and overlap_size >= window_size:
        raise ValueError("overlap_size must be less than window_size when window_size is set")

    expected = {item.name for item in fields(config)}
    if checked != expected:
        missing = sorted(expected - checked)
        extra = sorted(checked - expected)
        raise RuntimeError(
            f"UltrackConfig validation coverage mismatch; missing={missing}, extra={extra}"
        )


def _as_config(config: UltrackConfig | Mapping[str, Any] | None) -> UltrackConfig:
    if config is None:
        return UltrackConfig()
    if isinstance(config, UltrackConfig):
        validate_ultrack_config(config)
        return config
    return UltrackConfig(**dict(config))


def _with_scale(config: UltrackConfig, attrs: dict[str, Any] | None) -> UltrackConfig:
    if config.scale_zyx is not None:
        return config
    return replace(config, scale_zyx=parse_scale(attrs))


def _sigma_voxels(sigma_um: float, scale_zyx: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(max(0.35, sigma_um / scale) for scale in scale_zyx)


def _radius_structure(radius_um: float, scale_zyx: tuple[float, float, float]) -> np.ndarray:
    radii = [max(1, int(math.ceil(radius_um / scale))) for scale in scale_zyx]
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
    dist = np.zeros(tuple(2 * r + 1 for r in radii), dtype=np.float32)
    for grid, scale in zip(grids, scale_zyx):
        dist += (grid.astype(np.float32) * float(scale)) ** 2
    return dist <= float(radius_um * radius_um)


def _robust_unit(x: np.ndarray, low_q: float = 0.001, high_q: float = 0.999) -> np.ndarray:
    arr = x.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.quantile(finite, [low_q, high_q]).astype(np.float32)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + np.float32(1.0)
    return np.clip((arr - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0).astype(np.float32, copy=False)


def _quantile_rescale(score: np.ndarray, quantile: float) -> np.ndarray:
    threshold = float(np.quantile(score, quantile))
    high = float(np.max(score))
    if high <= threshold:
        return (score >= threshold).astype(np.float32)
    return np.clip((score - threshold) / (high - threshold), 0.0, 1.0).astype(np.float32, copy=False)


def _seed_mask(score: np.ndarray, config: UltrackConfig, scale_zyx: tuple[float, float, float]) -> np.ndarray:
    try:
        from skimage.morphology import h_maxima

        seeds = h_maxima(score.astype(np.float32, copy=False), h=float(config.seed_h))
    except Exception:
        size = tuple(2 * max(1, int(math.ceil(config.min_cell_radius_um / scale))) + 1 for scale in scale_zyx)
        seeds = score == maximum_filter(score, size=size, mode="nearest")
        seeds &= score >= float(np.quantile(score, min(0.999, config.foreground_quantile)))
    if not np.any(seeds):
        return seeds.astype(bool, copy=False)
    structure = _radius_structure(config.seed_dilation_um, scale_zyx)
    return binary_dilation(seeds.astype(bool, copy=False), structure=structure)


def _watershed_frontier(
    gradient: np.ndarray,
    seed_mask: np.ndarray,
    foreground_mask: np.ndarray,
) -> np.ndarray:
    if not np.any(seed_mask) or not np.any(foreground_mask):
        return np.zeros(gradient.shape, dtype=np.float32)
    try:
        from skimage.measure import label
        from skimage.segmentation import find_boundaries, watershed

        markers = label(seed_mask)
        if int(markers.max()) == 0:
            return np.zeros(gradient.shape, dtype=np.float32)
        labels = watershed(gradient, markers=markers, mask=foreground_mask)
        return find_boundaries(labels, mode="thick").astype(np.float32, copy=False)
    except Exception:
        return np.zeros(gradient.shape, dtype=np.float32)


def _voxel_area_from_radius(radius_um: float, scale_zyx: tuple[float, float, float]) -> int:
    voxel_volume = float(scale_zyx[0] * scale_zyx[1] * scale_zyx[2])
    sphere_volume = (4.0 / 3.0) * math.pi * float(radius_um) ** 3
    return max(1, int(round(sphere_volume / voxel_volume)))


def resolve_ultrack_areas(config: UltrackConfig) -> tuple[int, int]:
    """Return min/max candidate areas in voxels from physical-radius defaults."""

    validate_ultrack_config(config)
    scale_zyx = config.scale_zyx or (1.625, 0.40625, 0.40625)
    min_area = config.min_area
    max_area = config.max_area
    if min_area is None:
        min_area = _voxel_area_from_radius(config.min_cell_radius_um, scale_zyx)
    if max_area is None:
        max_area = _voxel_area_from_radius(config.max_cell_radius_um, scale_zyx)
    return int(min_area), max(int(max_area), int(min_area) + 1)


def build_ultrack_inputs(
    array: Any,
    attrs: dict[str, Any] | None,
    config: UltrackConfig | Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Ultrack foreground and contour arrays from a small ``T,Z,Y,X`` clip."""

    config = _with_scale(_as_config(config), attrs)
    assert config.scale_zyx is not None
    scale_zyx = config.scale_zyx
    baseline = BaselineConfig(
        sigmas_um=config.sigmas_um,
        dog_ratio=config.dog_ratio,
        q_low=config.q_low,
        q_high=config.q_high,
        polarity="both",
    )
    preset = _CONTOUR_PRESETS.get(config.contour_preset)
    if preset is None:
        raise ValueError(f"Unknown contour preset {config.contour_preset!r}; expected one of {sorted(_CONTOUR_PRESETS)}")

    n_timepoints = int(array.shape[0])
    foreground = np.empty(tuple(int(v) for v in array.shape), dtype=np.float32)
    contours = np.empty_like(foreground)
    local_sigma = _sigma_voxels(config.local_contrast_sigma_um, scale_zyx)
    gradient_sigma = _sigma_voxels(0.8, scale_zyx)

    for t in range(n_timepoints):
        volume = np.asarray(array[t])
        norm = normalize_volume(volume, attrs, baseline)
        bright = dog_response(norm, scale_zyx, replace(baseline, polarity="bright"))
        dark = dog_response(norm, scale_zyx, replace(baseline, polarity="dark"))
        local = norm - gaussian_filter(norm, local_sigma, mode="nearest")

        bright_u = _robust_unit(bright)
        dark_u = _robust_unit(dark)
        local_u = _robust_unit(local)
        score = np.maximum.reduce([bright_u, dark_u, local_u]).astype(np.float32, copy=False)
        seed_mask = _seed_mask(score, config, scale_zyx)
        if np.any(seed_mask):
            score = np.maximum(score, seed_mask.astype(np.float32) * 0.95)

        fg = _quantile_rescale(score, config.foreground_quantile)
        foreground[t] = np.maximum(fg, seed_mask.astype(np.float32) * 0.85)

        gradient = gaussian_gradient_magnitude(norm.astype(np.float32, copy=False), gradient_sigma, mode="nearest")
        gradient_u = _robust_unit(gradient, high_q=float(preset["gradient_quantile"]))
        frontier = _watershed_frontier(gradient_u, seed_mask, foreground[t] >= config.foreground_threshold)
        contours[t] = np.maximum(gradient_u, frontier * float(preset["frontier_weight"]))

    return foreground.astype(np.float32, copy=False), np.clip(contours, 0.0, 1.0).astype(np.float32, copy=False)


def _configure_ultrack(config: UltrackConfig):
    validate_ultrack_config(config)
    try:
        from ultrack import MainConfig
    except ModuleNotFoundError as exc:
        if exc.name != "ultrack":
            raise
        raise RuntimeError(
            "Ultrack is an optional backend and is not installed in this environment. "
            "Use a supported Python environment and install Ultrack explicitly."
        ) from exc
    except TypeError as exc:
        raise RuntimeError(
            "Ultrack failed during import, which usually indicates an incompatible "
            f"Ultrack/Python combination (running Python {sys.version_info.major}."
            f"{sys.version_info.minor}). Use the Python version supported by the "
            "installed Ultrack release."
        ) from exc

    ul_config = MainConfig()
    if config.working_dir is not None:
        ul_config.data_config.working_dir = Path(config.working_dir)
    ul_config.data_config.database_file_name = str(config.database_file_name)
    ul_config.data_config.n_workers = int(config.n_workers)

    min_area, max_area = resolve_ultrack_areas(config)
    ul_config.segmentation_config.threshold = float(config.foreground_threshold)
    ul_config.segmentation_config.min_frontier = (
        float(config.min_frontier)
        if config.min_frontier is not None
        else float(_CONTOUR_PRESETS[config.contour_preset]["min_frontier"])
    )
    ul_config.segmentation_config.min_area = int(min_area)
    ul_config.segmentation_config.max_area = int(max_area)
    ul_config.segmentation_config.n_workers = int(config.n_workers)

    ul_config.linking_config.max_distance = float(config.max_link_um)
    ul_config.linking_config.max_neighbors = int(config.max_neighbors)
    ul_config.linking_config.n_workers = int(config.n_workers)

    ul_config.tracking_config.solver_name = str(config.solver_name)
    ul_config.tracking_config.window_size = config.window_size
    ul_config.tracking_config.overlap_size = int(config.overlap_size)
    ul_config.tracking_config.solution_gap = float(config.solution_gap)
    ul_config.tracking_config.time_limit = int(config.time_limit)
    ul_config.tracking_config.n_threads = int(config.n_threads)
    return ul_config


def run_ultrack_clip(
    foreground: np.ndarray,
    contours: np.ndarray,
    config: UltrackConfig | Mapping[str, Any] | None,
) -> tuple[Any, Any]:
    """Run Ultrack on precomputed foreground/contours and return tracks + graph."""

    config = _as_config(config)
    ul_config = _configure_ultrack(config)

    from ultrack import Tracker

    tracker = Tracker(config=ul_config)
    segment_kwargs = {"max_segments_per_time": int(config.max_segments_per_time)}
    scale = config.scale_zyx
    try:
        tracker.track(
            foreground=foreground,
            contours=contours,
            scale=scale,
            overwrite="all",
            segment_kwargs=segment_kwargs,
        )
    except TypeError as exc:
        message = str(exc)
        if "contours" not in message and "scale" not in message and "segment_kwargs" not in message:
            raise
        try:
            tracker.track(
                foreground=foreground,
                edges=contours,
                overwrite="all",
                segment_kwargs=segment_kwargs,
            )
        except TypeError:
            tracker.segment(contours=contours, foreground=foreground, overwrite=True, **segment_kwargs)
            tracker.link(scale=scale, overwrite=True)
            tracker.solve(overwrite=True)

    try:
        return tracker.to_tracks_layer(include_parents=True, include_node_ids=True)
    except TypeError:
        return tracker.to_tracks_layer()


def _dataset_name_and_offset(dataset: str | Mapping[str, Any]) -> tuple[str, int]:
    if isinstance(dataset, Mapping):
        name = str(dataset.get("name", dataset.get("dataset", "")))
        if not name:
            raise ValueError("Dataset mapping must include a 'name' or 'dataset' field")
        return name, int(dataset.get("t_offset", 0))
    return str(dataset), 0


def _tracks_records(tracks: Any) -> list[dict[str, Any]]:
    if hasattr(tracks, "to_dict"):
        records = tracks.to_dict("records")
    else:
        arr = np.asarray(tracks)
        if arr.ndim != 2 or arr.shape[1] not in (4, 5):
            raise ValueError("Ultrack tracks must be a dataframe or a napari tracks array")
        if arr.shape[1] == 5:
            cols = ("track_id", "t", "z", "y", "x")
        else:
            cols = ("track_id", "t", "y", "x")
        records = [dict(zip(cols, row, strict=True)) for row in arr]
    out: list[dict[str, Any]] = []
    for row in records:
        if "track_id" not in row or "t" not in row:
            continue
        z_value = row.get("z", 0)
        out.append(
            {
                "track_id": int(row["track_id"]),
                "ultrack_id": int(row.get("id", row.get("node_id", -1))),
                "t": int(round(float(row["t"]))),
                "z": int(round(float(z_value))),
                "y": int(round(float(row["y"]))),
                "x": int(round(float(row["x"]))),
            }
        )
    out.sort(key=lambda r: (r["t"], r["track_id"], r["z"], r["y"], r["x"]))
    return out


def _lineage_edges(graph: Any) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    if graph is None:
        return edges
    if isinstance(graph, Mapping):
        for child, parents in graph.items():
            if parents is None:
                continue
            if isinstance(parents, (list, tuple, set)):
                parent_iter = parents
            else:
                parent_iter = [parents]
            for parent in parent_iter:
                try:
                    parent_id = int(parent)
                    child_id = int(child)
                except (TypeError, ValueError):
                    continue
                if parent_id >= 0 and child_id >= 0:
                    edges.append((parent_id, child_id))
        return edges
    if hasattr(graph, "edges"):
        for parent, child in graph.edges():
            try:
                edges.append((int(parent), int(child)))
            except (TypeError, ValueError):
                continue
    return edges


def _add_edge_once(edges: list[dict[str, int]], seen: set[tuple[int, int]], source: int, target: int) -> None:
    pair = (int(source), int(target))
    if pair[0] == pair[1] or pair in seen:
        return
    seen.add(pair)
    edges.append({"source_id": pair[0], "target_id": pair[1]})


def ultrack_tracks_to_submission(
    dataset: str | Mapping[str, Any],
    tracks: Any,
    graph: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Convert Ultrack exported tracks into Kaggle node and edge records."""

    dataset_name, t_offset = _dataset_name_and_offset(dataset)
    records = _tracks_records(tracks)
    nodes: list[dict[str, Any]] = []
    by_track: dict[int, list[dict[str, Any]]] = {}

    next_node_id = 1
    for row in records:
        node_id = next_node_id
        next_node_id += 1
        node = {
            "dataset": dataset_name,
            "node_id": node_id,
            "t": int(row["t"] + t_offset),
            "z": max(0, int(row["z"])),
            "y": max(0, int(row["y"])),
            "x": max(0, int(row["x"])),
            "track_id": int(row["track_id"]),
            "ultrack_id": int(row["ultrack_id"]),
        }
        nodes.append(node)
        row["_node_id"] = node_id
        by_track.setdefault(int(row["track_id"]), []).append(row)

    edges: list[dict[str, int]] = []
    seen_edges: set[tuple[int, int]] = set()
    for track_rows in by_track.values():
        track_rows.sort(key=lambda r: (r["t"], r["z"], r["y"], r["x"]))
        for previous, current in zip(track_rows[:-1], track_rows[1:]):
            if int(current["t"]) == int(previous["t"]) + 1:
                _add_edge_once(edges, seen_edges, previous["_node_id"], current["_node_id"])

    for parent_track, child_track in _lineage_edges(graph):
        parent_rows = by_track.get(parent_track, [])
        child_rows = by_track.get(child_track, [])
        if not parent_rows or not child_rows:
            continue
        parent = max(parent_rows, key=lambda r: r["t"])
        child = min(child_rows, key=lambda r: r["t"])
        dt = int(child["t"]) - int(parent["t"])
        if dt == 1:
            _add_edge_once(edges, seen_edges, parent["_node_id"], child["_node_id"])
        elif dt == 2:
            interp = {
                "dataset": dataset_name,
                "node_id": next_node_id,
                "t": int(parent["t"] + 1 + t_offset),
                "z": int(round((parent["z"] + child["z"]) / 2.0)),
                "y": int(round((parent["y"] + child["y"]) / 2.0)),
                "x": int(round((parent["x"] + child["x"]) / 2.0)),
                "track_id": int(child_track),
                "ultrack_id": -1,
            }
            next_node_id += 1
            nodes.append(interp)
            _add_edge_once(edges, seen_edges, parent["_node_id"], interp["node_id"])
            _add_edge_once(edges, seen_edges, interp["node_id"], child["_node_id"])

    for node in nodes:
        node.pop("_node_id", None)
    return nodes, edges
