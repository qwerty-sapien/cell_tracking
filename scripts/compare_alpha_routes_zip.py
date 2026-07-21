#!/usr/bin/env python3
"""Compare selected alpha routes on locked fold validation clips."""

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
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha import default_route_configs, predict_dataset_window  # noqa: E402
from celltrack_alpha.evaluation import optional_official_metric_status  # noqa: E402
from celltrack_alpha.official_eval import official_edge_metric_row, official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from celltrack_alpha.localization import load_offset_calibration, offset_for_dataset, sha256_file  # noqa: E402
from celltrack_alpha.scoring import load_scoring_calibration  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_SPLIT = Path("reports/general_model_split_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", default="0")
    parser.add_argument("--datasets", default=None, help="Comma-separated validation datasets. Default: prefix-balanced fold sample.")
    parser.add_argument("--prefixes", default="6bba", help="Comma-separated prefixes to sample when --datasets is absent.")
    parser.add_argument("--max-per-prefix", type=int, default=4)
    parser.add_argument("--routes", default="clean_point,clean_compact")
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--offsets-json", type=Path, default=None, help="Optional fold-safe output offset calibration JSON.")
    parser.add_argument("--scoring-json", type=Path, default=None, help="Optional fold-safe candidate scoring/correction calibration JSON.")
    parser.add_argument(
        "--edge-only-metric",
        action="store_true",
        help="Use the parity-tested official edge/node scorer and skip division scoring for faster exploratory route sweeps.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/general_model_route_compare_fold0.json"))
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_fold(path: Path, fold_id: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("split manifest root must be an object")
    fold = payload.get("folds", {}).get(str(fold_id))
    if not isinstance(fold, dict):
        raise ValueError(f"fold {fold_id!r} is not present in {path}")
    return payload, fold, sha256_file(path)


def prefix_balanced_validation(fold: dict[str, Any], prefixes: list[str], max_per_prefix: int) -> list[str]:
    validation = [str(v) for v in fold.get("validation", [])]
    selected: list[str] = []
    for prefix in prefixes:
        selected.extend([d for d in validation if d.startswith(prefix + "_")][: max(0, int(max_per_prefix))])
    return selected


def crop_geff_arrays(gt: dict[str, Any], start: int, end: int, total_timepoints: int) -> dict[str, Any]:
    nodes = gt["nodes"]
    mask = (nodes["t"] >= start) & (nodes["t"] < end)
    kept_ids = set(int(v) for v in nodes["node_id"][mask])
    edges = np.asarray(
        [edge for edge in gt["edges"] if int(edge[0]) in kept_ids and int(edge[1]) in kept_ids],
        dtype=np.int64,
    )
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    try:
        estimated = float(gt.get("estimated_number_of_nodes")) * max(0, end - start) / max(1, int(total_timepoints))
    except (TypeError, ValueError):
        estimated = float("nan")
    return {
        "attrs": gt.get("attrs", {}),
        "nodes": {key: np.asarray(value)[mask] for key, value in nodes.items()},
        "edges": edges,
        "estimated_number_of_nodes": estimated,
    }


def clip_start_from_gt(gt: dict[str, Any], total_t: int, clip_length: int) -> int:
    times = sorted(set(int(v) for v in gt["nodes"]["t"]))
    if not times:
        return 0
    max_start = max(0, int(total_t) - int(clip_length))
    return max(0, min(times[0], max_start))


def finite_score(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def main() -> int:
    args = parse_args()
    if args.clip_length <= 0:
        raise ValueError("--clip-length must be positive")
    configs = default_route_configs()
    routes = parse_csv_strings(args.routes)
    unknown_routes = [route for route in routes if route not in configs]
    if unknown_routes:
        raise ValueError(f"Unknown routes: {unknown_routes}")
    split_manifest, fold, split_hash = load_fold(args.split_manifest, str(args.fold_id))
    if args.datasets:
        datasets = parse_csv_strings(args.datasets)
    else:
        datasets = prefix_balanced_validation(fold, parse_csv_strings(args.prefixes), args.max_per_prefix)
    validation = set(str(v) for v in fold.get("validation", []))
    invalid = sorted(set(datasets) - validation)
    if invalid:
        raise ValueError(f"datasets are outside fold validation set: {invalid[:10]}")
    official_status = optional_official_metric_status()
    if not official_status["available"]:
        raise RuntimeError(f"official metric unavailable: {official_status}")
    offset_calibration = load_offset_calibration(
        args.offsets_json,
        split_manifest_path=args.split_manifest,
        expected_fold_id=str(args.fold_id) if args.offsets_json is not None else None,
    )
    scoring_calibration = load_scoring_calibration(
        args.scoring_json,
        split_manifest_path=args.split_manifest,
        expected_fold_id=str(args.fold_id) if args.scoring_json is not None else None,
    )

    metric_fn = official_edge_metric_row if args.edge_only_metric else official_metric_row
    metric_source = "official_edge_only" if args.edge_only_metric else "official"

    rows: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []
    started_all = time.time()
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            total_t = int(array.shape[0])
            gt = read_geff_arrays(store, "train", dataset)
            start = clip_start_from_gt(gt, total_t, args.clip_length)
            end = min(total_t, start + int(args.clip_length))
            gt_clip = crop_geff_arrays(gt, start, end, total_t)
            scale = parse_scale(attrs)
            if len(gt_clip["nodes"]["node_id"]) == 0:
                dataset_reports.append({"dataset": dataset, "clip_start": start, "clip_end": end, "status": "skipped_empty_gt"})
                continue
            for route in routes:
                run_started = time.time()
                result = predict_dataset_window(
                    dataset=dataset,
                    array=array,
                    attrs=attrs,
                    start=start,
                    end=end,
                    forced_route=route,
                    output_offset_override=offset_for_dataset(dataset, offset_calibration),
                    estimated_total_nodes=float(gt.get("estimated_number_of_nodes")) if gt.get("estimated_number_of_nodes") is not None else None,
                    scoring_calibration=scoring_calibration,
                    progress=False,
                )
                metrics = metric_fn(result.nodes, result.edges, gt_clip, scale)
                row = {
                    "dataset": dataset,
                    "prefix": dataset.split("_", 1)[0],
                    "fold_id": str(args.fold_id),
                    "route": route,
                    "clip_start": start,
                    "clip_end": end,
                    "gt_nodes": int(len(gt_clip["nodes"]["node_id"])),
                    "gt_edges": int(len(gt_clip["edges"])),
                    "nodes": len(result.nodes),
                    "edges": len(result.edges),
                    "edge_candidates": result.diagnostics.get("num_edge_candidates"),
                    "score": finite_score(metrics.get("adj_edge_jaccard")),
                    "edge_jaccard": finite_score(metrics.get("edge_jaccard")),
                    "adj_edge_jaccard": finite_score(metrics.get("adj_edge_jaccard")),
                    "node_recall": finite_score(metrics.get("node_recall")),
                    "division_jaccard": finite_score(metrics.get("division_jaccard")),
                    "num_dividing_nodes": result.diagnostics.get("num_dividing_nodes"),
                    "metric_source": metric_source,
                    "runtime_sec": time.time() - run_started,
                    "route_reason": result.route.reason,
                }
                for key, value in metrics.items():
                    if key not in row:
                        row[key] = finite_score(value)
                rows.append(row)
                print(json.dumps(row), flush=True)
            best = max((row for row in rows if row["dataset"] == dataset), key=lambda r: finite_score(r["score"]))
            dataset_reports.append({"dataset": dataset, "clip_start": start, "clip_end": end, "best_route": best["route"], "best_score": best["score"]})
    finally:
        store.close()

    by_route: dict[str, dict[str, Any]] = {}
    official_by_route: dict[str, dict[str, Any]] = {}
    for route in routes:
        route_rows = [row for row in rows if row["route"] == route and math.isfinite(float(row["score"]))]
        by_route[route] = {
            "n": len(route_rows),
            "mean_score": float(np.mean([row["score"] for row in route_rows])) if route_rows else float("nan"),
            "median_score": float(np.median([row["score"] for row in route_rows])) if route_rows else float("nan"),
            "mean_edge_jaccard": float(np.mean([row["edge_jaccard"] for row in route_rows])) if route_rows else float("nan"),
            "mean_node_recall": float(np.mean([row["node_recall"] for row in route_rows])) if route_rows else float("nan"),
            "mean_nodes": float(np.mean([row["nodes"] for row in route_rows])) if route_rows else float("nan"),
        }
        official_by_route[route] = official_summary(route_rows) if route_rows else {}
    wins = {route: 0 for route in routes}
    ties = {route: 0 for route in routes}
    dataset_best_routes: dict[str, list[str]] = {}
    for dataset in datasets:
        candidates = [row for row in rows if row["dataset"] == dataset and math.isfinite(float(row["score"]))]
        if candidates:
            best_score = max(row["score"] for row in candidates)
            best_routes = [row["route"] for row in candidates if abs(float(row["score"]) - best_score) <= 1e-12]
            dataset_best_routes[dataset] = best_routes
            if len(best_routes) == 1:
                wins[best_routes[0]] += 1
            else:
                for route in best_routes:
                    ties[route] += 1
    payload = {
        "kind": "route_comparison",
        "created_at_unix": time.time(),
        "zip": str(args.zip),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": split_hash,
        "fold_id": str(args.fold_id),
        "offsets_json": str(args.offsets_json) if args.offsets_json is not None else None,
        "offset_calibration_fold_id": offset_calibration.fold_id if offset_calibration is not None else None,
        "scoring_json": str(args.scoring_json) if args.scoring_json is not None else None,
        "scoring_calibration_fold_id": scoring_calibration.fold_id if scoring_calibration is not None else None,
        "datasets": datasets,
        "routes": routes,
        "clip_length": int(args.clip_length),
        "metric_mode": metric_source,
        "rows": rows,
        "dataset_reports": dataset_reports,
        "summary_by_route": by_route,
        "official_summary_by_route": official_by_route,
        "wins_by_route": wins,
        "ties_by_route": ties,
        "dataset_best_routes": dataset_best_routes,
        "runtime_sec": time.time() - started_all,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary_by_route": by_route, "official_summary_by_route": official_by_route, "wins_by_route": wins, "ties_by_route": ties}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
