"""Multi-scale DoG detection plus Hungarian linking baseline.

The code is intentionally dependency-light so it can run in a Kaggle notebook
without paid GPU compute. It assumes image arrays are shaped (T, Z, Y, X).
"""

from __future__ import annotations

import csv
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment


DEFAULT_SCALE_ZYX = (1.625, 0.40625, 0.40625)
SUBMISSION_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


@dataclass(frozen=True)
class BaselineConfig:
    sigmas_um: tuple[float, ...] = (0.8, 1.2, 1.8)
    dog_ratio: float = 1.7
    threshold_quantile: float = 0.99955
    min_score: float = 0.015
    min_distance_um: float = 2.8
    max_points_per_t: int = 180
    max_link_um: float = 8.0
    q_low: float = 0.01
    q_high: float = 0.999
    max_timepoints: int | None = None
    polarity: str = "bright"
    output_transform: str = "identity"


def with_timepoint_limit(config: BaselineConfig, max_timepoints: int | None) -> BaselineConfig:
    return replace(config, max_timepoints=max_timepoints)


def parse_scale(attrs: dict[str, Any] | None) -> tuple[float, float, float]:
    """Parse z/y/x voxel scale from OME-NGFF zarr attrs."""
    attrs = attrs or {}
    try:
        transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if transform.get("type") == "scale":
            scale = transform["scale"][-3:]
            return float(scale[0]), float(scale[1]), float(scale[2])
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return DEFAULT_SCALE_ZYX


def image_quantiles(attrs: dict[str, Any] | None) -> dict[float, float]:
    attrs = attrs or {}
    raw = attrs.get("image_statistics", {}).get("quantiles", {})
    out: dict[float, float] = {}
    for key, value in raw.items():
        try:
            out[float(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _nearest_quantile(quantiles: dict[float, float], q: float, tol: float = 1e-8) -> float | None:
    for key, value in quantiles.items():
        if abs(key - q) <= tol:
            return value
    return None


def normalize_volume(
    volume: np.ndarray,
    attrs: dict[str, Any] | None,
    config: BaselineConfig,
) -> np.ndarray:
    """Robustly normalize one 3D volume to [0, 1]."""
    vol = volume.astype(np.float32, copy=False)
    quantiles = image_quantiles(attrs)
    lo = _nearest_quantile(quantiles, config.q_low)
    hi = _nearest_quantile(quantiles, config.q_high)
    if lo is None or hi is None or hi <= lo:
        sample = vol[::4, ::8, ::8]
        lo, hi = np.quantile(sample, [config.q_low, config.q_high]).astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((vol - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)


def _sigma_voxels(sigma_um: float, scale_zyx: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(max(0.35, sigma_um / scale) for scale in scale_zyx)


def dog_response(
    normalized_volume: np.ndarray,
    scale_zyx: tuple[float, float, float],
    config: BaselineConfig,
) -> np.ndarray:
    """Return max-pooled multi-scale 3D Difference-of-Gaussians response."""
    response = np.full(normalized_volume.shape, -np.inf, dtype=np.float32)
    for sigma_um in config.sigmas_um:
        small = _sigma_voxels(sigma_um, scale_zyx)
        large = tuple(s * config.dog_ratio for s in small)
        fine = gaussian_filter(normalized_volume, small, mode="nearest")
        coarse = gaussian_filter(normalized_volume, large, mode="nearest")
        if config.polarity == "dark":
            band = coarse - fine
        elif config.polarity == "bright":
            band = fine - coarse
        else:
            band = np.abs(fine - coarse)
        np.maximum(response, band.astype(np.float32, copy=False), out=response)
    return response


def _nms_size(scale_zyx: tuple[float, float, float], min_distance_um: float) -> tuple[int, int, int]:
    radii = [max(1, int(math.ceil(min_distance_um / s))) for s in scale_zyx]
    return tuple(2 * r + 1 for r in radii)


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


def detect_volume(
    volume: np.ndarray,
    t: int,
    dataset: str,
    next_node_id: int,
    attrs: dict[str, Any] | None,
    scale_zyx: tuple[float, float, float],
    config: BaselineConfig,
) -> tuple[list[dict[str, Any]], int]:
    """Detect cell centers for one timepoint."""
    norm = normalize_volume(volume, attrs, config)
    response = dog_response(norm, scale_zyx, config)
    threshold = max(float(np.quantile(response, config.threshold_quantile)), config.min_score)
    local_max = maximum_filter(response, size=_nms_size(scale_zyx, config.min_distance_um), mode="nearest")
    mask = (response >= threshold) & (response == local_max)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return [], next_node_id
    scores = response[coords[:, 0], coords[:, 1], coords[:, 2]]
    coords, scores = _greedy_physical_nms(
        coords, scores, scale_zyx, config.min_distance_um, config.max_points_per_t
    )
    nodes: list[dict[str, Any]] = []
    z_max, y_max, x_max = volume.shape
    for coord, score in zip(coords, scores):
        z, y, x = [int(v) for v in coord]
        nodes.append(
            {
                "dataset": dataset,
                "node_id": int(next_node_id),
                "t": int(t),
                "z": min(max(z, 0), z_max - 1),
                "y": min(max(y, 0), y_max - 1),
                "x": min(max(x, 0), x_max - 1),
                "score": float(score),
            }
        )
        next_node_id += 1
    return nodes, next_node_id


def link_nodes(
    nodes_by_t: list[list[dict[str, Any]]],
    scale_zyx: tuple[float, float, float],
    max_link_um: float,
) -> list[dict[str, int]]:
    """Link adjacent timepoints with one-to-one Hungarian assignment."""
    edges: list[dict[str, int]] = []
    scale = np.asarray(scale_zyx, dtype=np.float32)
    big_cost = max_link_um + 1_000_000.0
    for prev_nodes, curr_nodes in zip(nodes_by_t[:-1], nodes_by_t[1:]):
        if not prev_nodes or not curr_nodes:
            continue
        prev = np.asarray([[n["z"], n["y"], n["x"]] for n in prev_nodes], dtype=np.float32)
        curr = np.asarray([[n["z"], n["y"], n["x"]] for n in curr_nodes], dtype=np.float32)
        delta = (prev[:, None, :] - curr[None, :, :]) * scale
        dist = np.sqrt(np.sum(delta * delta, axis=2))
        feasible = dist <= max_link_um
        if not feasible.any():
            continue
        cost = np.where(feasible, dist, big_cost)
        row_ind, col_ind = linear_sum_assignment(cost)
        for row, col in zip(row_ind, col_ind):
            if dist[row, col] <= max_link_um:
                edges.append(
                    {
                        "dataset": str(prev_nodes[row]["dataset"]),
                        "source_id": int(prev_nodes[row]["node_id"]),
                        "target_id": int(curr_nodes[col]["node_id"]),
                    }
                )
    return edges


def apply_output_transform(
    nodes_by_t: list[list[dict[str, Any]]],
    spatial_shape_zyx: tuple[int, int, int],
    transform: str,
) -> None:
    """Transform image coordinates into submission coordinates in place."""
    z_size, y_size, x_size = [int(v) for v in spatial_shape_zyx]
    for nodes in nodes_by_t:
        for node in nodes:
            z, y, x = int(node["z"]), int(node["y"]), int(node["x"])
            if transform == "identity":
                out = (z, y, x)
            elif transform == "xflip":
                out = (z, y, x_size - 1 - x)
            elif transform == "yflip":
                out = (z, y_size - 1 - y, x)
            elif transform == "xyflip":
                out = (z, y_size - 1 - y, x_size - 1 - x)
            elif transform == "yxswap":
                out = (z, x, y)
            else:
                raise ValueError(f"Unknown output transform: {transform}")
            node["z"], node["y"], node["x"] = out


def predict_dataset_array(
    dataset: str,
    array: Any,
    attrs: dict[str, Any] | None,
    config: BaselineConfig,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Predict nodes and edges for a zarr-like array."""
    scale_zyx = parse_scale(attrs)
    n_timepoints = int(array.shape[0])
    if config.max_timepoints is not None:
        n_timepoints = min(n_timepoints, int(config.max_timepoints))
    nodes_by_t: list[list[dict[str, Any]]] = []
    next_node_id = 1
    for t in range(n_timepoints):
        if progress:
            print(f"{dataset}: detecting t={t + 1}/{n_timepoints}", flush=True)
        detected, next_node_id = detect_volume(
            np.asarray(array[t]),
            t=t,
            dataset=dataset,
            next_node_id=next_node_id,
            attrs=attrs,
            scale_zyx=scale_zyx,
            config=config,
        )
        nodes_by_t.append(detected)
    edges = link_nodes(nodes_by_t, scale_zyx, config.max_link_um)
    if nodes_by_t:
        spatial_shape = tuple(int(v) for v in array.shape[1:4])
        apply_output_transform(nodes_by_t, spatial_shape, config.output_transform)
    nodes = [node for time_nodes in nodes_by_t for node in time_nodes]
    return nodes, edges


def submission_rows(
    predictions: Iterable[tuple[str, list[dict[str, Any]], list[dict[str, int]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_id = 0
    for dataset, nodes, edges in predictions:
        for node in nodes:
            rows.append(
                {
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": int(node["node_id"]),
                    "t": int(node["t"]),
                    "z": int(node["z"]),
                    "y": int(node["y"]),
                    "x": int(node["x"]),
                    "source_id": -1,
                    "target_id": -1,
                }
            )
            row_id += 1
        for edge in edges:
            rows.append(
                {
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": int(edge["source_id"]),
                    "target_id": int(edge["target_id"]),
                }
            )
            row_id += 1
    return rows


def write_submission(rows: list[dict[str, Any]], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def validate_submission_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    nodes_by_dataset: dict[str, set[int]] = {}
    edge_pairs: set[tuple[str, int, int]] = set()
    row_type_counts = Counter()
    for row in rows:
        dataset = str(row["dataset"])
        row_type = str(row["row_type"])
        row_type_counts[row_type] += 1
        if row_type == "node":
            node_id = int(row["node_id"])
            nodes = nodes_by_dataset.setdefault(dataset, set())
            if node_id in nodes:
                errors.append(f"duplicate node_id {dataset}:{node_id}")
            nodes.add(node_id)
        elif row_type == "edge":
            pair = (dataset, int(row["source_id"]), int(row["target_id"]))
            if pair in edge_pairs:
                errors.append(f"duplicate edge {pair}")
            edge_pairs.add(pair)
        else:
            errors.append(f"unknown row_type {row_type!r}")
    for dataset, source_id, target_id in edge_pairs:
        nodes = nodes_by_dataset.get(dataset, set())
        if source_id not in nodes or target_id not in nodes:
            errors.append(f"edge endpoint missing {dataset}:{source_id}->{target_id}")
    return {
        "valid": not errors,
        "errors": errors[:50],
        "row_type_counts": dict(row_type_counts),
        "dataset_count": len(nodes_by_dataset),
    }


def list_zip_datasets(zip_path: Path | str, split: str) -> list[str]:
    prefix = f"{split}/"
    suffix = ".zarr/0/zarr.json"
    datasets: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith(prefix) and name.endswith(suffix):
                base = name[len(prefix):-len(suffix)]
                datasets.add(base)
    return sorted(datasets)


def read_geff_arrays(store: Any, split: str, dataset: str) -> dict[str, Any]:
    """Read sparse ground-truth graph arrays from a GEFF zarr group."""
    import zarr

    base = f"{split}/{dataset}.geff"
    group = zarr.open_group(store=store, path=base, mode="r")
    attrs = dict(group.attrs)
    nodes = {
        "node_id": zarr.open_array(store=store, path=f"{base}/nodes/ids", mode="r")[:].astype(np.int64),
        "t": zarr.open_array(store=store, path=f"{base}/nodes/props/t/values", mode="r")[:].astype(np.int64),
        "z": zarr.open_array(store=store, path=f"{base}/nodes/props/z/values", mode="r")[:].astype(np.int64),
        "y": zarr.open_array(store=store, path=f"{base}/nodes/props/y/values", mode="r")[:].astype(np.int64),
        "x": zarr.open_array(store=store, path=f"{base}/nodes/props/x/values", mode="r")[:].astype(np.int64),
    }
    edges = zarr.open_array(store=store, path=f"{base}/edges/ids", mode="r")[:].astype(np.int64)
    estimated = attrs.get("geff", {}).get("extra", {}).get("estimated_number_of_nodes")
    return {"attrs": attrs, "nodes": nodes, "edges": edges, "estimated_number_of_nodes": estimated}


def approximate_sparse_metrics(
    pred_nodes: list[dict[str, Any]],
    pred_edges: list[dict[str, int]],
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
    max_distance_um: float = 7.0,
) -> dict[str, float]:
    """Approximate the official sparse edge Jaccard for quick parameter sweeps.

    This is not a replacement for the official metric implementation, but it is
    useful for cheap local/Kaggle tuning before submission.
    """
    gt_nodes = gt["nodes"]
    gt_edges = {tuple(map(int, row)) for row in gt["edges"]}
    gt_by_t: dict[int, list[int]] = {}
    for idx, t in enumerate(gt_nodes["t"]):
        gt_by_t.setdefault(int(t), []).append(idx)
    pred_by_t: dict[int, list[int]] = {}
    for idx, node in enumerate(pred_nodes):
        pred_by_t.setdefault(int(node["t"]), []).append(idx)

    pred_to_gt: dict[int, int] = {}
    matched_gt: set[int] = set()
    scale = np.asarray(scale_zyx, dtype=np.float32)
    for t, pred_idxs in pred_by_t.items():
        gt_idxs = gt_by_t.get(t, [])
        if not pred_idxs or not gt_idxs:
            continue
        pred_xyz = np.asarray(
            [[pred_nodes[i]["z"], pred_nodes[i]["y"], pred_nodes[i]["x"]] for i in pred_idxs],
            dtype=np.float32,
        )
        gt_xyz = np.asarray(
            [[gt_nodes["z"][i], gt_nodes["y"][i], gt_nodes["x"][i]] for i in gt_idxs],
            dtype=np.float32,
        )
        delta = (pred_xyz[:, None, :] - gt_xyz[None, :, :]) * scale
        dist = np.sqrt(np.sum(delta * delta, axis=2))
        feasible = dist <= max_distance_um
        if not feasible.any():
            continue
        cost = np.where(feasible, dist, max_distance_um + 1_000_000.0)
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            if dist[row, col] <= max_distance_um:
                pred_idx = pred_idxs[row]
                gt_idx = gt_idxs[col]
                pred_to_gt[int(pred_nodes[pred_idx]["node_id"])] = int(gt_nodes["node_id"][gt_idx])
                matched_gt.add(gt_idx)

    gt_out_valid = {src for src, _ in gt_edges}
    gt_in_valid = {dst for _, dst in gt_edges}
    edge_tp = 0
    edge_fp = 0
    seen_pred_edges: set[tuple[int, int]] = set()
    for edge in pred_edges:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        if (source, target) in seen_pred_edges:
            continue
        seen_pred_edges.add((source, target))
        matched_source = pred_to_gt.get(source)
        matched_target = pred_to_gt.get(target)
        is_tp = (
            matched_source is not None
            and matched_target is not None
            and (matched_source, matched_target) in gt_edges
        )
        is_valid = (
            (matched_source is not None and matched_source in gt_out_valid)
            or (matched_target is not None and matched_target in gt_in_valid)
        )
        if is_tp:
            edge_tp += 1
        elif is_valid:
            edge_fp += 1
    edge_fn = max(0, len(gt_edges) - edge_tp)
    denom = edge_tp + edge_fp + edge_fn
    edge_jaccard = edge_tp / denom if denom else float("nan")
    estimated_total = gt.get("estimated_number_of_nodes")
    try:
        estimated_total = float(estimated_total)
    except (TypeError, ValueError):
        estimated_total = float("nan")
    if estimated_total > 0 and edge_jaccard == edge_jaccard:
        total_node_ratio = (len(pred_nodes) - estimated_total) / estimated_total
        adj_edge_jaccard = max(0.0, edge_jaccard * (1.0 - 0.1 * total_node_ratio))
    else:
        total_node_ratio = float("nan")
        adj_edge_jaccard = float("nan")
    return {
        "edge_tp": float(edge_tp),
        "edge_fp": float(edge_fp),
        "edge_fn": float(edge_fn),
        "edge_jaccard": float(edge_jaccard),
        "adj_edge_jaccard": float(adj_edge_jaccard),
        "node_recall": float(len(matched_gt) / len(gt_nodes["node_id"])) if len(gt_nodes["node_id"]) else float("nan"),
        "num_pred_nodes": float(len(pred_nodes)),
        "estimated_number_of_nodes": float(estimated_total),
        "total_node_ratio": float(total_node_ratio),
    }


def config_to_json(config: BaselineConfig) -> str:
    return json.dumps(config.__dict__, indent=2, sort_keys=True)
