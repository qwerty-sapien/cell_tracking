#!/usr/bin/env python3
"""Probe fine-grained post-prediction filters for offset-clone subgraphs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.evaluation import submission_graph_diagnostics  # noqa: E402
from celltrack_alpha.localization import sha256_file  # noqa: E402
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_alpha.pipeline import predict_dataset_window  # noqa: E402
from celltrack_alpha.scoring import load_scoring_calibration  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays, submission_rows  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_SPLIT = Path("reports/general_model_split_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", default="0")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--route", default="auto")
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--scoring-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_fold(path: Path, fold_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fold = payload.get("folds", {}).get(str(fold_id))
    if not isinstance(fold, dict):
        raise ValueError(f"fold {fold_id!r} is not present in {path}")
    return fold


def first_gt_clip_start(gt: dict[str, Any], total_t: int, clip_length: int) -> int:
    times = sorted(set(int(value) for value in gt["nodes"]["t"]))
    if not times:
        return 0
    max_start = max(0, int(total_t) - int(clip_length))
    return max(0, min(times[0], max_start))


def crop_geff_arrays(gt: dict[str, Any], start: int, end: int, total_timepoints: int) -> dict[str, Any]:
    nodes = gt["nodes"]
    mask = (nodes["t"] >= start) & (nodes["t"] < end)
    kept_ids = {int(value) for value in nodes["node_id"][mask]}
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


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def source_tokens(node: dict[str, Any]) -> list[str]:
    return [part for part in str(node.get("sources", "")).split("|") if part]


def is_offset_node(node: dict[str, Any]) -> bool:
    return any(token.startswith("offset_clone_") for token in source_tokens(node))


def offset_source_mode(node: dict[str, Any]) -> str:
    tokens = source_tokens(node)
    has_raw = any(token.startswith("tile_raw@") for token in tokens)
    has_contrast = any(token.startswith("tile_contrast@") for token in tokens)
    if has_raw and has_contrast:
        return "raw_contrast"
    if has_contrast:
        return "contrast_only"
    if has_raw:
        return "raw_only"
    return "other"


def offset_chain_components(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, int]],
) -> dict[int, dict[str, Any]]:
    nodes_by_id = {int(node["node_id"]): node for node in nodes}
    offset_ids = {node_id for node_id, node in nodes_by_id.items() if is_offset_node(node)}
    adjacency: dict[int, set[int]] = defaultdict(set)
    edge_count_by_node: dict[int, int] = defaultdict(int)
    for edge in edges:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        if source not in offset_ids or target not in offset_ids:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
        edge_count_by_node[source] += 1
        edge_count_by_node[target] += 1

    info_by_node: dict[int, dict[str, Any]] = {}
    unseen = set(offset_ids)
    while unseen:
        first = min(unseen)
        queue: deque[int] = deque([first])
        component: set[int] = set()
        while queue:
            node_id = queue.popleft()
            if node_id in component:
                continue
            component.add(node_id)
            unseen.discard(node_id)
            queue.extend(sorted(adjacency.get(node_id, set()) - component))
        times = {int(nodes_by_id[node_id]["t"]) for node_id in component}
        edge_total = sum(edge_count_by_node[node_id] for node_id in component) // 2
        xs = [int(nodes_by_id[node_id]["x"]) for node_id in component]
        ys = [int(nodes_by_id[node_id]["y"]) for node_id in component]
        zs = [int(nodes_by_id[node_id]["z"]) for node_id in component]
        modes = {offset_source_mode(nodes_by_id[node_id]) for node_id in component}
        info = {
            "component_size": len(component),
            "component_edges": int(edge_total),
            "component_times": sorted(times),
            "component_full_3t": times == {0, 1, 2},
            "component_min_x": min(xs) if xs else None,
            "component_max_x": max(xs) if xs else None,
            "component_min_y": min(ys) if ys else None,
            "component_max_y": max(ys) if ys else None,
            "component_min_z": min(zs) if zs else None,
            "component_max_z": max(zs) if zs else None,
            "component_modes": sorted(modes),
        }
        for node_id in component:
            info_by_node[node_id] = info
    return info_by_node


def variant_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"name": "current"},
        {"name": "no_offset", "drop_all_offsets": True},
    ]
    for mode in ("contrast_only", "raw_only", "raw_contrast", "contrast_any", "raw_any"):
        specs.append({"name": f"mode_{mode}", "source_mode": mode})
    for threshold in (16, 24, 32, 48, 64, 96, 128, 192):
        specs.append({"name": f"x_le_{threshold}", "x_le": threshold})
        specs.append({"name": f"component_x_le_{threshold}", "component_x_le": threshold})
        specs.append({"name": f"full3t_x_le_{threshold}", "full_3t": True, "x_le": threshold})
        specs.append({"name": f"full3t_component_x_le_{threshold}", "full_3t": True, "component_x_le": threshold})
    for mode in ("contrast_only", "contrast_any", "raw_contrast"):
        for threshold in (16, 24, 32, 48, 64, 96, 128):
            specs.append({"name": f"{mode}_x_le_{threshold}", "source_mode": mode, "x_le": threshold})
            specs.append(
                {
                    "name": f"{mode}_full3t_component_x_le_{threshold}",
                    "source_mode": mode,
                    "full_3t": True,
                    "component_x_le": threshold,
                }
            )
    return specs


def offset_node_matches(node: dict[str, Any], spec: dict[str, Any], component_info: dict[str, Any]) -> bool:
    if bool(spec.get("drop_all_offsets", False)):
        return False
    mode = spec.get("source_mode")
    node_mode = offset_source_mode(node)
    if mode == "contrast_any" and node_mode not in {"contrast_only", "raw_contrast"}:
        return False
    elif mode == "raw_any" and node_mode not in {"raw_only", "raw_contrast"}:
        return False
    elif mode in {"contrast_only", "raw_only", "raw_contrast"} and node_mode != mode:
        return False
    if bool(spec.get("full_3t", False)) and not bool(component_info.get("component_full_3t", False)):
        return False
    if "x_le" in spec and int(node["x"]) > int(spec["x_le"]):
        return False
    if "component_x_le" in spec:
        max_x = component_info.get("component_max_x")
        if max_x is None or int(max_x) > int(spec["component_x_le"]):
            return False
    return True


def apply_variant(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, int]],
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, int]], dict[str, Any]]:
    component_by_node = offset_chain_components(nodes, edges)
    kept_ids: set[int] = set()
    dropped_offsets = 0
    kept_offsets = 0
    for node in nodes:
        node_id = int(node["node_id"])
        if is_offset_node(node):
            if not offset_node_matches(node, spec, component_by_node.get(node_id, {})):
                dropped_offsets += 1
                continue
            kept_offsets += 1
        kept_ids.add(node_id)
    filtered_edges = [
        edge
        for edge in edges
        if int(edge["source_id"]) in kept_ids and int(edge["target_id"]) in kept_ids
    ]
    used_ids = {int(value) for edge in filtered_edges for value in (edge["source_id"], edge["target_id"])}
    filtered_nodes = [node for node in nodes if int(node["node_id"]) in kept_ids and int(node["node_id"]) in used_ids]
    stats = {
        "kept_offset_nodes_before_endpoint_filter": kept_offsets,
        "dropped_offset_nodes": dropped_offsets,
        "kept_nodes_after_endpoint_filter": len(filtered_nodes),
        "kept_edges": len(filtered_edges),
    }
    return filtered_nodes, filtered_edges, stats


def main() -> int:
    args = parse_args()
    datasets = parse_csv(args.datasets)
    fold = load_fold(args.split_manifest, str(args.fold_id))
    validation = {str(value) for value in fold.get("validation", [])}
    invalid = sorted(set(datasets) - validation)
    if invalid:
        raise ValueError(f"datasets are outside fold {args.fold_id} validation set: {invalid[:10]}")

    scoring_calibration = load_scoring_calibration(
        args.scoring_json,
        split_manifest_path=args.split_manifest,
        expected_fold_id=str(args.fold_id) if args.scoring_json is not None else None,
    )
    specs = variant_specs()
    rows: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []
    started = time.time()
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            total_t = int(array.shape[0])
            start = first_gt_clip_start(gt, total_t, args.clip_length)
            end = min(total_t, start + int(args.clip_length))
            gt_clip = crop_geff_arrays(gt, start, end, total_t)
            result = predict_dataset_window(
                dataset=dataset,
                array=array,
                attrs=attrs,
                start=start,
                end=end,
                forced_route=None if args.route == "auto" else args.route,
                allow_division_route=False,
                estimated_total_nodes=float(gt.get("estimated_number_of_nodes"))
                if gt.get("estimated_number_of_nodes") is not None
                else None,
                scoring_calibration=scoring_calibration,
                progress=False,
            )
            dataset_rows: list[dict[str, Any]] = []
            for spec in specs:
                nodes, edges, filter_stats = apply_variant(result.nodes, result.edges, spec)
                metrics = official_metric_row(nodes, edges, gt_clip, parse_scale(attrs))
                row = {
                    "dataset": dataset,
                    "route": result.route.label,
                    "clip_start": start,
                    "clip_end": end,
                    "variant": str(spec["name"]),
                    "spec": dict(spec),
                    "num_pred_nodes": int(len(nodes)),
                    "num_pred_edges": int(len(edges)),
                    "filter_stats": filter_stats,
                    "graph": submission_graph_diagnostics(submission_rows([(dataset, nodes, edges)])),
                    "route_reason": result.route.reason,
                    "offset_clone_layers": result.diagnostics.get("offset_clone_layers"),
                }
                row.update(metrics)
                rows.append(row)
                dataset_rows.append(row)
            best = max(
                dataset_rows,
                key=lambda row: (
                    finite(row.get("adj_edge_jaccard")),
                    finite(row.get("edge_jaccard")),
                    -finite(row.get("num_pred_nodes")),
                ),
            )
            current = next(row for row in dataset_rows if row["variant"] == "current")
            report = {
                "dataset": dataset,
                "route": result.route.label,
                "current_score": finite(current.get("adj_edge_jaccard")),
                "current_edge_jaccard": finite(current.get("edge_jaccard")),
                "best_variant": best["variant"],
                "best_score": finite(best.get("adj_edge_jaccard")),
                "best_edge_jaccard": finite(best.get("edge_jaccard")),
                "best_node_recall": finite(best.get("node_recall")),
                "best_nodes": int(best["num_pred_nodes"]),
                "best_edges": int(best["num_pred_edges"]),
                "delta_score": finite(best.get("adj_edge_jaccard")) - finite(current.get("adj_edge_jaccard")),
            }
            dataset_reports.append(report)
            print(json.dumps(report, default=str), flush=True)
    finally:
        store.close()

    payload = {
        "kind": "offset_clone_filter_variant_probe",
        "created_at_unix": time.time(),
        "zip": str(args.zip),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "fold_id": str(args.fold_id),
        "datasets": datasets,
        "route_arg": args.route,
        "clip_length": int(args.clip_length),
        "scoring_json": str(args.scoring_json) if args.scoring_json is not None else None,
        "summary": official_summary(rows),
        "dataset_reports": dataset_reports,
        "rows": rows,
        "runtime_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runtime_sec": payload["runtime_sec"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
