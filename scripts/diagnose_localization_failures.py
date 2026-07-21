#!/usr/bin/env python3
"""Inspect sparse GT localization failures against candidate routes and image score maps."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.proposals import detect_candidates_one_timepoint  # noqa: E402
from celltrack_alpha.routing import default_route_configs  # noqa: E402
from celltrack_baseline.rule_based import BaselineConfig, dog_response, normalize_volume, parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--datasets", required=True, help="Comma-separated train dataset IDs.")
    parser.add_argument("--routes", default="balanced_compact,dense_fusion")
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--radius-um", type=float, default=7.0)
    parser.add_argument("--max-gt-nodes", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("reports/general_model_localization_failure_diagnostic.json"))
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def robust_unit(x: np.ndarray, low_q: float = 0.001, high_q: float = 0.999) -> np.ndarray:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, dtype=np.float32)
    lo, hi = np.quantile(finite, [low_q, high_q]).astype(np.float32)
    if hi <= lo:
        hi = lo + np.float32(1.0)
    return np.clip((x.astype(np.float32, copy=False) - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)


def sigma_voxels(sigma_um: float, scale_zyx: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(max(0.35, float(sigma_um) / float(scale)) for scale in scale_zyx)


def clip_start_from_gt(gt: dict[str, Any], total_t: int, clip_length: int) -> int:
    times = sorted(set(int(v) for v in gt["nodes"]["t"]))
    if not times:
        return 0
    return max(0, min(times[0], max(0, int(total_t) - int(clip_length))))


def gt_rows_for_clip(gt: dict[str, Any], start: int, end: int, max_gt_nodes: int) -> list[dict[str, Any]]:
    nodes = gt["nodes"]
    mask = (nodes["t"] >= start) & (nodes["t"] < end)
    indices = np.flatnonzero(mask)
    rows: list[dict[str, Any]] = []
    for idx in indices[: max(0, int(max_gt_nodes))]:
        rows.append(
            {
                "node_id": int(nodes["node_id"][idx]),
                "t": int(nodes["t"][idx]),
                "z": float(nodes["z"][idx]),
                "y": float(nodes["y"][idx]),
                "x": float(nodes["x"][idx]),
            }
        )
    return rows


def nearest_candidate(
    gt_zyx: np.ndarray,
    candidates: list[Any],
    scale_zyx: tuple[float, float, float],
) -> dict[str, Any]:
    if not candidates:
        return {"distance_um": math.inf, "candidate": None}
    coords = np.asarray([node.coord_zyx for node in candidates], dtype=np.float32)
    scale = np.asarray(scale_zyx, dtype=np.float32)
    distances = np.sqrt(np.sum(((coords - gt_zyx[None, :]) * scale) ** 2, axis=1))
    idx = int(np.argmin(distances))
    node = candidates[idx]
    return {
        "distance_um": float(distances[idx]),
        "delta_zyx_vox": [float(v) for v in (coords[idx] - gt_zyx)],
        "candidate": {
            "node_id": int(node.node_id),
            "z": int(node.z),
            "y": int(node.y),
            "x": int(node.x),
            "score": float(node.score),
            "intensity": float(node.intensity),
            "sources": list(node.sources),
        },
    }


def local_patch_best(
    score: np.ndarray,
    gt_zyx: np.ndarray,
    scale_zyx: tuple[float, float, float],
    radius_um: float,
) -> dict[str, Any]:
    shape = score.shape
    center = np.asarray(np.round(gt_zyx), dtype=np.int64)
    radii = np.asarray([max(1, int(math.ceil(radius_um / float(scale)))) for scale in scale_zyx], dtype=np.int64)
    lows = np.maximum(center - radii, 0)
    highs = np.minimum(center + radii + 1, np.asarray(shape, dtype=np.int64))
    patch = score[lows[0] : highs[0], lows[1] : highs[1], lows[2] : highs[2]]
    if patch.size == 0:
        return {"best_score": math.nan, "best_coord_zyx": None, "distance_um": math.inf}
    local_coords = np.indices(patch.shape).reshape(3, -1).T + lows[None, :]
    delta = (local_coords.astype(np.float32) - gt_zyx[None, :]) * np.asarray(scale_zyx, dtype=np.float32)
    distances = np.sqrt(np.sum(delta * delta, axis=1))
    inside = distances <= float(radius_um)
    if not np.any(inside):
        return {"best_score": math.nan, "best_coord_zyx": None, "distance_um": math.inf}
    values = patch.reshape(-1)[inside]
    inside_coords = local_coords[inside]
    inside_distances = distances[inside]
    best_idx = int(np.argmax(values))
    best_coord = inside_coords[best_idx].astype(int)
    nearest_gt = np.clip(np.round(gt_zyx).astype(int), 0, np.asarray(shape) - 1)
    return {
        "score_at_gt_round": float(score[nearest_gt[0], nearest_gt[1], nearest_gt[2]]),
        "best_score": float(values[best_idx]),
        "best_coord_zyx": [int(v) for v in best_coord],
        "distance_um": float(inside_distances[best_idx]),
        "delta_zyx_vox": [float(v) for v in (best_coord.astype(np.float32) - gt_zyx)],
    }


def score_maps(volume: np.ndarray, attrs: dict[str, Any], scale_zyx: tuple[float, float, float]) -> dict[str, np.ndarray]:
    baseline = BaselineConfig(polarity="both")
    norm = normalize_volume(volume, attrs, baseline)
    local_sigma = sigma_voxels(2.5, scale_zyx)
    return {
        "norm": norm.astype(np.float32, copy=False),
        "smoothed_norm": robust_unit(gaussian_filter(norm, sigma_voxels(0.7, scale_zyx), mode="nearest")),
        "local_contrast": robust_unit(norm - gaussian_filter(norm, local_sigma, mode="nearest")),
        "dog_bright": robust_unit(dog_response(norm, scale_zyx, BaselineConfig(polarity="bright"))),
        "dog_dark": robust_unit(dog_response(norm, scale_zyx, BaselineConfig(polarity="dark"))),
        "dog_abs": robust_unit(dog_response(norm, scale_zyx, BaselineConfig(polarity="both"))),
    }


def summarize_distances(distances: list[float], radius_um: float) -> dict[str, Any]:
    finite = np.asarray([value for value in distances if math.isfinite(float(value))], dtype=np.float32)
    if finite.size == 0:
        return {
            "n": len(distances),
            "recall_radius": 0.0,
            "recall_10um": 0.0,
            "median_nearest_um": math.inf,
            "p90_nearest_um": math.inf,
        }
    return {
        "n": len(distances),
        "recall_radius": float(np.mean(finite <= float(radius_um))),
        "recall_10um": float(np.mean(finite <= 10.0)),
        "median_nearest_um": float(np.median(finite)),
        "p90_nearest_um": float(np.quantile(finite, 0.90)),
    }


def main() -> int:
    args = parse_args()
    if args.clip_length <= 0:
        raise ValueError("--clip-length must be positive")
    configs = default_route_configs()
    routes = parse_csv_strings(args.routes)
    unknown_routes = [route for route in routes if route not in configs]
    if unknown_routes:
        raise ValueError(f"Unknown routes: {unknown_routes}")
    datasets = parse_csv_strings(args.datasets)

    started = time.time()
    dataset_reports: list[dict[str, Any]] = []
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            scale = parse_scale(attrs)
            start = clip_start_from_gt(gt, int(array.shape[0]), args.clip_length)
            end = min(int(array.shape[0]), start + int(args.clip_length))
            gt_rows = gt_rows_for_clip(gt, start, end, args.max_gt_nodes)

            route_nodes_by_t: dict[str, dict[int, list[Any]]] = {route: {} for route in routes}
            next_ids = {route: 1 for route in routes}
            map_by_t: dict[int, dict[str, np.ndarray]] = {}
            for t in range(start, end):
                volume = np.asarray(array[t])
                map_by_t[t] = score_maps(volume, attrs, scale)
                for route in routes:
                    nodes, next_ids[route] = detect_candidates_one_timepoint(
                        volume=volume,
                        dataset=dataset,
                        t=t,
                        attrs=attrs,
                        scale_zyx=scale,
                        config=configs[route],
                        next_node_id=next_ids[route],
                    )
                    route_nodes_by_t[route][t] = nodes

            route_distances: dict[str, list[float]] = {route: [] for route in routes}
            map_distances: dict[str, list[float]] = {name: [] for name in next(iter(map_by_t.values())).keys()} if map_by_t else {}
            rows: list[dict[str, Any]] = []
            for gt_row in gt_rows:
                t = int(gt_row["t"])
                gt_zyx = np.asarray([gt_row["z"], gt_row["y"], gt_row["x"]], dtype=np.float32)
                route_results: dict[str, Any] = {}
                for route in routes:
                    nearest = nearest_candidate(gt_zyx, route_nodes_by_t[route].get(t, []), scale)
                    route_results[route] = nearest
                    route_distances[route].append(float(nearest["distance_um"]))
                map_results: dict[str, Any] = {}
                for name, score in map_by_t.get(t, {}).items():
                    best = local_patch_best(score, gt_zyx, scale, args.radius_um)
                    map_results[name] = best
                    map_distances.setdefault(name, []).append(float(best["distance_um"]))
                rows.append({**gt_row, "routes": route_results, "score_maps": map_results})

            route_summaries = {
                route: {
                    **summarize_distances(route_distances[route], args.radius_um),
                    "mean_candidates_per_t": float(np.mean([len(v) for v in route_nodes_by_t[route].values()]))
                    if route_nodes_by_t[route]
                    else 0.0,
                }
                for route in routes
            }
            map_summaries = {
                name: summarize_distances(values, args.radius_um)
                for name, values in sorted(map_distances.items())
            }
            dataset_report = {
                "dataset": dataset,
                "clip_start": start,
                "clip_end": end,
                "scale_zyx_um": list(scale),
                "radius_um": float(args.radius_um),
                "gt_nodes_evaluated": len(gt_rows),
                "route_summaries": route_summaries,
                "score_map_summaries": map_summaries,
                "gt_rows": rows,
            }
            dataset_reports.append(dataset_report)
            print(json.dumps({k: dataset_report[k] for k in ("dataset", "clip_start", "clip_end", "gt_nodes_evaluated", "route_summaries", "score_map_summaries")}, indent=2), flush=True)
    finally:
        store.close()

    payload = {
        "kind": "localization_failure_diagnostic",
        "created_at_unix": time.time(),
        "zip": str(args.zip),
        "datasets": datasets,
        "routes": routes,
        "clip_length": int(args.clip_length),
        "radius_um": float(args.radius_um),
        "dataset_reports": dataset_reports,
        "runtime_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runtime_sec": payload["runtime_sec"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
