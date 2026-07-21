#!/usr/bin/env python3
"""Label final submitted edges against sparse GT matches for diagnostics."""

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
from scipy.optimize import linear_sum_assignment
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.evaluation import submission_graph_diagnostics  # noqa: E402
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
    return [part.strip() for part in str(value).split(",") if part.strip()]


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


def match_nodes(
    pred_nodes: list[dict[str, Any]],
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
    max_distance_um: float = 7.0,
) -> tuple[dict[int, int], dict[int, float]]:
    gt_nodes = gt["nodes"]
    pred_by_t: dict[int, list[int]] = {}
    gt_by_t: dict[int, list[int]] = {}
    for idx, node in enumerate(pred_nodes):
        pred_by_t.setdefault(int(node["t"]), []).append(idx)
    for idx, t in enumerate(gt_nodes["t"]):
        gt_by_t.setdefault(int(t), []).append(idx)

    scale = np.asarray(scale_zyx, dtype=np.float64)
    pred_to_gt: dict[int, int] = {}
    pred_to_gt_distance: dict[int, float] = {}
    for t, pred_idxs in sorted(pred_by_t.items()):
        gt_idxs = gt_by_t.get(t, [])
        if not pred_idxs or not gt_idxs:
            continue
        pred_xyz = np.asarray(
            [[pred_nodes[i]["z"], pred_nodes[i]["y"], pred_nodes[i]["x"]] for i in pred_idxs],
            dtype=np.float64,
        )
        gt_xyz = np.asarray(
            [[gt_nodes["z"][i], gt_nodes["y"][i], gt_nodes["x"][i]] for i in gt_idxs],
            dtype=np.float64,
        )
        dist = np.sqrt(np.sum(((pred_xyz[:, None, :] - gt_xyz[None, :, :]) * scale) ** 2, axis=2))
        feasible = dist <= float(max_distance_um)
        if not feasible.any():
            continue
        rows, cols = linear_sum_assignment(np.where(feasible, dist, max_distance_um + 1_000_000.0))
        for row, col in zip(rows, cols, strict=True):
            if dist[row, col] <= max_distance_um:
                pred_id = int(pred_nodes[pred_idxs[row]]["node_id"])
                pred_to_gt[pred_id] = int(gt_nodes["node_id"][gt_idxs[col]])
                pred_to_gt_distance[pred_id] = float(dist[row, col])
    return pred_to_gt, pred_to_gt_distance


def node_sources(node: dict[str, Any]) -> str:
    sources = node.get("sources", "")
    if isinstance(sources, (list, tuple)):
        return "|".join(str(value) for value in sources)
    return str(sources)


def source_family(sources: str) -> str:
    tokens = sources.split("|") if sources else []
    if any(token.startswith("endpoint_forward:") for token in tokens):
        return "endpoint_forward"
    if any(token.startswith("endpoint_backward:") for token in tokens):
        return "endpoint_backward"
    if any(token.startswith("offset_clone_") for token in tokens):
        return "offset_clone"
    if any(token.startswith("localmax_clone_") for token in tokens):
        return "localmax_clone"
    return "base"


def edge_component_stats(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, int]],
) -> dict[tuple[int, int], dict[str, Any]]:
    nodes_by_id = {int(node["node_id"]): node for node in nodes}
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        adjacency[source].add(target)
        adjacency[target].add(source)

    component_by_node: dict[int, dict[str, Any]] = {}
    unseen = set(nodes_by_id)
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
        component_edges = [
            edge
            for edge in edges
            if int(edge["source_id"]) in component and int(edge["target_id"]) in component
        ]
        component_nodes = [nodes_by_id[node_id] for node_id in component]
        families = sorted({source_family(node_sources(node)) for node in component_nodes})
        xs = [int(node["x"]) for node in component_nodes]
        ys = [int(node["y"]) for node in component_nodes]
        zs = [int(node["z"]) for node in component_nodes]
        ts = sorted({int(node["t"]) for node in component_nodes})
        stats = {
            "component_nodes": len(component),
            "component_edges": len(component_edges),
            "component_timepoints": ts,
            "component_families": families,
            "component_min_x": min(xs) if xs else None,
            "component_max_x": max(xs) if xs else None,
            "component_min_y": min(ys) if ys else None,
            "component_max_y": max(ys) if ys else None,
            "component_min_z": min(zs) if zs else None,
            "component_max_z": max(zs) if zs else None,
        }
        for node_id in component:
            component_by_node[node_id] = stats
    return {
        (int(edge["source_id"]), int(edge["target_id"])): component_by_node[int(edge["source_id"])]
        for edge in edges
        if int(edge["source_id"]) in component_by_node
    }


def label_edges(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, int]],
    gt_clip: dict[str, Any],
    scale_zyx: tuple[float, float, float],
) -> dict[str, Any]:
    pred_to_gt, pred_to_gt_distance = match_nodes(nodes, gt_clip, scale_zyx)
    gt_edges = {(int(a), int(b)) for a, b in gt_clip["edges"]}
    nodes_by_id = {int(node["node_id"]): node for node in nodes}
    components_by_edge = edge_component_stats(nodes, edges)
    labeled: list[dict[str, Any]] = []
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        source_gt = pred_to_gt.get(source_id)
        target_gt = pred_to_gt.get(target_id)
        if source_gt is None and target_gt is None:
            continue
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            continue
        source_sources = node_sources(source)
        target_sources = node_sources(target)
        labeled.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "source_gt": source_gt,
                "target_gt": target_gt,
                "tp": bool(source_gt is not None and target_gt is not None and (source_gt, target_gt) in gt_edges),
                "both_matched": bool(source_gt is not None and target_gt is not None),
                "source_t": int(source["t"]),
                "target_t": int(target["t"]),
                "source_zyx": [int(source["z"]), int(source["y"]), int(source["x"])],
                "target_zyx": [int(target["z"]), int(target["y"]), int(target["x"])],
                "source_match_um": pred_to_gt_distance.get(source_id),
                "target_match_um": pred_to_gt_distance.get(target_id),
                "source_sources": source_sources,
                "target_sources": target_sources,
                "source_family": source_family(source_sources),
                "target_family": source_family(target_sources),
                **components_by_edge.get((source_id, target_id), {}),
            }
        )
    tp_like = sum(1 for row in labeled if row["tp"])
    fp_like = sum(1 for row in labeled if not row["tp"])
    return {
        "matched_nodes": len(pred_to_gt),
        "valid_like_edges": len(labeled),
        "tp_like_edges": tp_like,
        "fp_like_edges": fp_like,
        "valid_edges": labeled,
        "likely_fp_edges": [row for row in labeled if not row["tp"]],
    }


def main() -> int:
    args = parse_args()
    if args.clip_length <= 0:
        raise ValueError("--clip-length must be positive")
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

    reports: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
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
            scale_zyx = parse_scale(attrs)
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
            metrics = official_metric_row(result.nodes, result.edges, gt_clip, scale_zyx)
            metric_rows.append(metrics)
            labels = label_edges(result.nodes, result.edges, gt_clip, scale_zyx)
            report = {
                "dataset": dataset,
                "route": result.route.label,
                "route_reason": result.route.reason,
                "clip_start": start,
                "clip_end": end,
                "gt_nodes": int(len(gt_clip["nodes"]["node_id"])),
                "gt_edges": int(len(gt_clip["edges"])),
                "num_pred_nodes": int(len(result.nodes)),
                "num_pred_edges": int(len(result.edges)),
                "graph": submission_graph_diagnostics(submission_rows([(dataset, result.nodes, result.edges)])),
                "metrics": metrics,
                "diagnostics": result.diagnostics,
                **labels,
            }
            reports.append(report)
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "route": result.route.label,
                        "score": finite(metrics.get("adj_edge_jaccard")),
                        "edge_jaccard": finite(metrics.get("edge_jaccard")),
                        "edge_tp": metrics.get("edge_tp"),
                        "edge_fp": metrics.get("edge_fp"),
                        "edge_fn": metrics.get("edge_fn"),
                        "tp_like": labels["tp_like_edges"],
                        "fp_like": labels["fp_like_edges"],
                    },
                    default=str,
                ),
                flush=True,
            )
    finally:
        store.close()

    payload = {
        "kind": "final_edge_label_table",
        "created_at_unix": time.time(),
        "zip": str(args.zip),
        "split_manifest": str(args.split_manifest),
        "fold_id": str(args.fold_id),
        "datasets": datasets,
        "route_arg": args.route,
        "clip_length": int(args.clip_length),
        "scoring_json": str(args.scoring_json) if args.scoring_json is not None else None,
        "summary": official_summary(metric_rows),
        "reports": reports,
        "runtime_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": payload["summary"]}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
