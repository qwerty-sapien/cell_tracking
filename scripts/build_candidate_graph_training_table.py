#!/usr/bin/env python3
"""Build fold-safe candidate node/edge training tables from sparse GEFF labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.localization import sha256_file  # noqa: E402
from celltrack_alpha.proposals import generate_candidates  # noqa: E402
from celltrack_alpha.routing import default_route_configs  # noqa: E402
from celltrack_alpha.scoring import edge_feature_dict, node_feature_dict  # noqa: E402
from celltrack_alpha.solver import build_edge_candidates  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_SPLIT = Path("reports/general_model_split_manifest.json")
DEFAULT_OUT = Path("data/derived/edge_scorer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", default="0")
    parser.add_argument("--datasets", default=None, help="Comma-separated source datasets. Default: prefix-balanced calibration-train sample.")
    parser.add_argument("--max-per-prefix", type=int, default=1)
    parser.add_argument("--routes", default="balanced_compact")
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--clips-per-dataset", type=int, default=1)
    parser.add_argument("--match-radius-um", type=float, default=7.0)
    parser.add_argument("--hard-negative-radius-um", type=float, default=14.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=Path("reports/general_model_edge_training_table_fold0.json"))
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_fold(split_manifest_path: Path, fold_id: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("split manifest root must be an object")
    fold = payload.get("folds", {}).get(str(fold_id))
    if not isinstance(fold, dict):
        raise ValueError(f"fold {fold_id!r} is not present in {split_manifest_path}")
    return payload, fold, sha256_file(split_manifest_path)


def prefix_balanced_sample(datasets: list[str], max_per_prefix: int) -> list[str]:
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for dataset in sorted(datasets):
        by_prefix[dataset.split("_", 1)[0]].append(dataset)
    selected: list[str] = []
    for prefix in sorted(by_prefix):
        selected.extend(by_prefix[prefix][: max(0, int(max_per_prefix))])
    return selected


def clip_starts(gt_times: list[int], total_t: int, clip_length: int, clips_per_dataset: int) -> list[int]:
    if not gt_times:
        return [0]
    max_start = max(0, total_t - max(1, clip_length))
    candidates = sorted({max(0, min(int(t), max_start)) for t in gt_times})
    if len(candidates) <= clips_per_dataset:
        return candidates
    idx = np.linspace(0, len(candidates) - 1, clips_per_dataset).round().astype(int)
    return sorted({int(candidates[int(i)]) for i in idx})


def gt_coords_for_t(gt: dict[str, Any], t: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = gt["nodes"]
    mask = nodes["t"] == int(t)
    ids = np.asarray(nodes["node_id"][mask], dtype=np.int64)
    coords = np.asarray(
        list(zip(nodes["z"][mask], nodes["y"][mask], nodes["x"][mask], strict=True)),
        dtype=np.float32,
    )
    return ids, coords


def match_nodes_by_time(
    nodes_by_t: list[list[Any]],
    gt: dict[str, Any],
    *,
    start: int,
    scale_zyx: tuple[float, float, float],
    match_radius_um: float,
    hard_negative_radius_um: float,
) -> dict[int, dict[str, Any]]:
    scale = np.asarray(scale_zyx, dtype=np.float32)
    labels: dict[int, dict[str, Any]] = {}
    for rel_t, nodes in enumerate(nodes_by_t):
        t = start + rel_t
        gt_ids, gt_xyz = gt_coords_for_t(gt, t)
        for node in nodes:
            labels[node.node_id] = {
                "label": "",
                "label_status": "unknown_no_gt" if len(gt_xyz) == 0 else "unknown_far_from_sparse_gt",
                "matched_gt_id": "",
                "nearest_gt_um": "",
            }
        if len(gt_xyz) == 0 or not nodes:
            continue
        cand_xyz = np.asarray([node.coord_zyx for node in nodes], dtype=np.float32)
        dist = np.sqrt(np.sum(((gt_xyz[:, None, :] - cand_xyz[None, :, :]) * scale) ** 2, axis=2))
        nearest = np.min(dist, axis=0)
        for node, distance in zip(nodes, nearest, strict=True):
            labels[node.node_id]["nearest_gt_um"] = float(distance)
            if float(distance) <= hard_negative_radius_um:
                labels[node.node_id]["label"] = 0
                labels[node.node_id]["label_status"] = "hard_negative_near_sparse_gt"
        cost = dist.copy()
        cost[cost > match_radius_um] = 1e9
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols, strict=True):
            distance = float(dist[row, col])
            if distance > match_radius_um:
                continue
            node = nodes[int(col)]
            labels[node.node_id] = {
                "label": 1,
                "label_status": "positive_7um_assignment",
                "matched_gt_id": int(gt_ids[int(row)]),
                "nearest_gt_um": distance,
            }
    return labels


def count_gt_misses(
    nodes_by_t: list[list[Any]],
    gt: dict[str, Any],
    *,
    start: int,
    scale_zyx: tuple[float, float, float],
    match_radius_um: float,
) -> int:
    scale = np.asarray(scale_zyx, dtype=np.float32)
    misses = 0
    for rel_t, nodes in enumerate(nodes_by_t):
        gt_ids, gt_xyz = gt_coords_for_t(gt, start + rel_t)
        if len(gt_ids) == 0:
            continue
        if not nodes:
            misses += len(gt_ids)
            continue
        cand_xyz = np.asarray([node.coord_zyx for node in nodes], dtype=np.float32)
        dist = np.sqrt(np.sum(((gt_xyz[:, None, :] - cand_xyz[None, :, :]) * scale) ** 2, axis=2))
        misses += int(np.sum(np.min(dist, axis=1) > match_radius_um))
    return misses


def write_csv(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return []
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "dataset", "prefix", "fold_id", "route", "clip_start", "clip_end", "t", "node_id",
        "source_id", "target_id", "z", "y", "x", "label", "label_status", "matched_gt_id",
        "matched_source_gt_id", "matched_target_gt_id", "nearest_gt_um", "sources",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)
    return ordered


def main() -> int:
    args = parse_args()
    if args.clip_length <= 0:
        raise ValueError("--clip-length must be positive")
    if args.clips_per_dataset <= 0:
        raise ValueError("--clips-per-dataset must be positive")
    configs = default_route_configs()
    routes = parse_csv_strings(args.routes)
    unknown = [route for route in routes if route not in configs]
    if unknown:
        raise ValueError(f"Unknown routes: {unknown}")

    split_manifest, fold, split_hash = load_fold(args.split_manifest, str(args.fold_id))
    calibration_train = [str(v) for v in fold.get("calibration_train", [])]
    validation = [str(v) for v in fold.get("validation", [])]
    excluded = set(
        str(v)
        for v in split_manifest.get("excluded_from_general_score", {}).get(
            "duplicate_public_ids", []
        )
    )
    if args.datasets:
        datasets = parse_csv_strings(args.datasets)
    else:
        datasets = prefix_balanced_sample(calibration_train, args.max_per_prefix)
    invalid = sorted(set(datasets) - set(calibration_train))
    if invalid:
        raise ValueError(f"source datasets are outside fold calibration_train: {invalid[:10]}")
    dup = sorted(set(datasets) & excluded)
    if dup:
        raise ValueError(f"source datasets include duplicate public IDs excluded from general score: {dup}")

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []
    started_all = time.time()
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            scale = parse_scale(attrs)
            gt_times = sorted(set(int(t) for t in gt["nodes"]["t"]))
            starts = clip_starts(gt_times, int(array.shape[0]), args.clip_length, args.clips_per_dataset)
            for route in routes:
                config = configs[route]
                for start in starts:
                    end = min(int(array.shape[0]), int(start) + int(args.clip_length))
                    if end <= start:
                        continue
                    if args.progress:
                        print(f"{dataset}: route={route} clip={start}:{end}", flush=True)
                    run_started = time.time()
                    nodes_by_t, all_nodes = generate_candidates(
                        dataset=dataset,
                        array=array,
                        attrs=attrs,
                        config=config,
                        start=start,
                        end=end,
                        progress=False,
                    )
                    edges = build_edge_candidates(nodes_by_t, scale, config)
                    labels = match_nodes_by_time(
                        nodes_by_t,
                        gt,
                        start=start,
                        scale_zyx=scale,
                        match_radius_um=args.match_radius_um,
                        hard_negative_radius_um=args.hard_negative_radius_um,
                    )
                    gt_edge_set = {(int(s), int(t)) for s, t in gt["edges"]}
                    node_by_id = {node.node_id: node for node in all_nodes}
                    node_status_counts: Counter[str] = Counter()
                    edge_status_counts: Counter[str] = Counter()
                    for node in all_nodes:
                        label = labels[node.node_id]
                        features = node_feature_dict(node)
                        row = {
                            "dataset": dataset,
                            "prefix": dataset.split("_", 1)[0],
                            "fold_id": str(args.fold_id),
                            "route": route,
                            "clip_start": start,
                            "clip_end": end,
                            "t": int(node.t),
                            "node_id": int(node.node_id),
                            "z": int(node.z),
                            "y": int(node.y),
                            "x": int(node.x),
                            "label": label["label"],
                            "label_status": label["label_status"],
                            "matched_gt_id": label["matched_gt_id"],
                            "nearest_gt_um": label["nearest_gt_um"],
                            "sources": "|".join(node.sources),
                        }
                        row.update({f"feature__{key}": value for key, value in features.items()})
                        node_rows.append(row)
                        node_status_counts.update([str(label["label_status"])])
                    for edge in edges:
                        source = node_by_id[int(edge.source_id)]
                        target = node_by_id[int(edge.target_id)]
                        source_label = labels[int(edge.source_id)]
                        target_label = labels[int(edge.target_id)]
                        source_gt = source_label.get("matched_gt_id")
                        target_gt = target_label.get("matched_gt_id")
                        edge_label: int | str = ""
                        edge_status = "unknown_sparse_endpoint"
                        if source_label["label_status"] == "positive_7um_assignment" and target_label["label_status"] == "positive_7um_assignment":
                            pair = (int(source_gt), int(target_gt))
                            edge_label = 1 if pair in gt_edge_set else 0
                            edge_status = "positive_gt_edge" if edge_label == 1 else "hard_negative_between_matched_gt"
                        elif source_label["label_status"].startswith("hard_negative") or target_label["label_status"].startswith("hard_negative"):
                            edge_label = 0
                            edge_status = "hard_negative_endpoint"
                        features = edge_feature_dict(edge, source, target, max_link_um=config.max_link_um)
                        row = {
                            "dataset": dataset,
                            "prefix": dataset.split("_", 1)[0],
                            "fold_id": str(args.fold_id),
                            "route": route,
                            "clip_start": start,
                            "clip_end": end,
                            "source_id": int(edge.source_id),
                            "target_id": int(edge.target_id),
                            "label": edge_label,
                            "label_status": edge_status,
                            "matched_source_gt_id": source_gt,
                            "matched_target_gt_id": target_gt,
                        }
                        row.update({f"feature__{key}": value for key, value in features.items()})
                        edge_rows.append(row)
                        edge_status_counts.update([edge_status])
                    dataset_reports.append(
                        {
                            "dataset": dataset,
                            "route": route,
                            "clip_start": start,
                            "clip_end": end,
                            "nodes": len(all_nodes),
                            "edge_candidates": len(edges),
                            "gt_misses_7um": count_gt_misses(
                                nodes_by_t,
                                gt,
                                start=start,
                                scale_zyx=scale,
                                match_radius_um=args.match_radius_um,
                            ),
                            "node_label_status": dict(node_status_counts),
                            "edge_label_status": dict(edge_status_counts),
                            "runtime_sec": time.time() - run_started,
                        }
                    )
                    print(json.dumps(dataset_reports[-1], indent=2), flush=True)
    finally:
        store.close()

    route_suffix = "-".join(routes)
    dataset_suffix = f"fold{args.fold_id}_{route_suffix}_{len(datasets)}ds"
    node_csv = args.output_dir / f"candidate_nodes_{dataset_suffix}.csv"
    edge_csv = args.output_dir / f"candidate_edges_{dataset_suffix}.csv"
    node_schema = write_csv(node_csv, node_rows)
    edge_schema = write_csv(edge_csv, edge_rows)
    node_counts = Counter(str(row.get("label_status", "")) for row in node_rows)
    edge_counts = Counter(str(row.get("label_status", "")) for row in edge_rows)
    payload = {
        "kind": "candidate_graph_training_table",
        "created_at_unix": time.time(),
        "zip": str(args.zip),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": split_hash,
        "fold_id": str(args.fold_id),
        "source_datasets": datasets,
        "calibration_train_count": len(calibration_train),
        "validation_count": len(validation),
        "excluded_duplicate_public_ids": sorted(excluded),
        "routes": routes,
        "match_radius_um": float(args.match_radius_um),
        "hard_negative_radius_um": float(args.hard_negative_radius_um),
        "clip_length": int(args.clip_length),
        "clips_per_dataset": int(args.clips_per_dataset),
        "node_csv": str(node_csv),
        "edge_csv": str(edge_csv),
        "node_csv_sha256": sha256_file(node_csv),
        "edge_csv_sha256": sha256_file(edge_csv),
        "node_rows": len(node_rows),
        "edge_rows": len(edge_rows),
        "node_label_status": dict(node_counts),
        "edge_label_status": dict(edge_counts),
        "node_schema": node_schema,
        "edge_schema": edge_schema,
        "dataset_reports": dataset_reports,
        "runtime_sec": time.time() - started_all,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"node_rows": len(node_rows), "edge_rows": len(edge_rows), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
