"""Evaluation helpers for alpha runs.

The official scorer needs optional ``tracksdata`` and ``polars`` dependencies.
This module keeps the alpha runner useful without those packages by using the
existing approximate sparse scorer and reporting explicit diagnostics.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def optional_official_metric_status() -> dict[str, Any]:
    missing: list[str] = []
    for module in ("tracksdata", "polars"):
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(module)
    return {
        "available": not missing,
        "missing": missing,
        "note": "Install optional dependencies to replace approximate CV with official metrics.",
    }


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "edge_jaccard",
        "adj_edge_jaccard",
        "node_recall",
        "num_pred_nodes",
        "estimated_number_of_nodes",
        "total_node_ratio",
        "division_tp",
        "division_fp",
        "division_fn",
        "division_jaccard",
        "gt_divisions",
        "pred_divisions",
    ]
    out: dict[str, Any] = {"n": len(rows)}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    by_route: dict[str, dict[str, Any]] = {}
    for route in sorted({str(row.get("route", "")) for row in rows}):
        route_rows = [row for row in rows if str(row.get("route", "")) == route]
        if route:
            by_route[route] = summarize_metric_rows_without_routes(route_rows)
    out["by_route"] = by_route
    return out


def summarize_metric_rows_without_routes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "edge_jaccard",
        "adj_edge_jaccard",
        "node_recall",
        "num_pred_nodes",
        "total_node_ratio",
        "division_jaccard",
        "gt_divisions",
        "pred_divisions",
    )
    return {
        "n": len(rows),
        **{
            key: float(np.mean([float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]))
            if any(key in row and math.isfinite(float(row[key])) for row in rows)
            else float("nan")
            for key in keys
        },
    }


def submission_graph_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nodes_by_dataset: dict[str, set[int]] = {}
    outdeg: Counter[tuple[str, int]] = Counter()
    indeg: Counter[tuple[str, int]] = Counter()
    for row in rows:
        dataset = str(row["dataset"])
        if row["row_type"] == "node":
            nodes_by_dataset.setdefault(dataset, set()).add(int(row["node_id"]))
        elif row["row_type"] == "edge":
            source = int(row["source_id"])
            target = int(row["target_id"])
            outdeg[(dataset, source)] += 1
            indeg[(dataset, target)] += 1
    return {
        "datasets": len(nodes_by_dataset),
        "nodes": sum(len(v) for v in nodes_by_dataset.values()),
        "edges": sum(outdeg.values()),
        "dividing_nodes": sum(1 for v in outdeg.values() if v >= 2),
        "out_gt2": sum(1 for v in outdeg.values() if v > 2),
        "merges": sum(1 for v in indeg.values() if v > 1),
    }


def approximate_division_metrics(
    pred_nodes: list[dict[str, Any]],
    pred_edges: list[dict[str, int]],
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
    max_distance_um: float = 7.0,
) -> dict[str, float]:
    """Strict, cheap division proxy for tuning.

    This is intentionally not a replacement for the official division metric:
    it lacks the official +/- timing and connected-component lineage tolerance.
    It is useful for catching obviously over-eager fork settings.
    """

    gt_nodes = gt["nodes"]
    gt_by_t: dict[int, list[int]] = {}
    for idx, t in enumerate(gt_nodes["t"]):
        gt_by_t.setdefault(int(t), []).append(idx)
    pred_by_t: dict[int, list[int]] = {}
    for idx, node in enumerate(pred_nodes):
        pred_by_t.setdefault(int(node["t"]), []).append(idx)

    pred_to_gt: dict[int, int] = {}
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
        dist = np.sqrt(np.sum(((pred_xyz[:, None, :] - gt_xyz[None, :, :]) * scale) ** 2, axis=2))
        feasible = dist <= max_distance_um
        if not feasible.any():
            continue
        rows, cols = linear_sum_assignment(np.where(feasible, dist, max_distance_um + 1_000_000.0))
        for row, col in zip(rows, cols, strict=True):
            if dist[row, col] <= max_distance_um:
                pred_to_gt[int(pred_nodes[pred_idxs[row]]["node_id"])] = int(gt_nodes["node_id"][gt_idxs[col]])

    gt_children: dict[int, set[int]] = {}
    for source, target in gt["edges"]:
        gt_children.setdefault(int(source), set()).add(int(target))
    gt_divisions = {source: children for source, children in gt_children.items() if len(children) >= 2}
    gt_annotated_parents = set(gt_children)

    pred_children: dict[int, set[int]] = {}
    for edge in pred_edges:
        pred_children.setdefault(int(edge["source_id"]), set()).add(int(edge["target_id"]))
    pred_divisions = {source: children for source, children in pred_children.items() if len(children) >= 2}

    matched_gt_divisions: set[int] = set()
    division_tp = 0
    division_fp = 0
    for pred_parent, pred_child_ids in pred_divisions.items():
        gt_parent = pred_to_gt.get(pred_parent)
        if gt_parent is None:
            continue
        gt_div_children = gt_divisions.get(gt_parent)
        if gt_div_children is None:
            if gt_parent in gt_annotated_parents:
                division_fp += 1
            continue
        matched_children = {
            pred_to_gt[child]
            for child in pred_child_ids
            if child in pred_to_gt and pred_to_gt[child] in gt_div_children
        }
        if len(matched_children) >= 2 and gt_parent not in matched_gt_divisions:
            division_tp += 1
            matched_gt_divisions.add(gt_parent)
        else:
            division_fp += 1

    division_fn = max(0, len(gt_divisions) - division_tp)
    denom = division_tp + division_fp + division_fn
    return {
        "division_tp": float(division_tp),
        "division_fp": float(division_fp),
        "division_fn": float(division_fn),
        "division_jaccard": float(division_tp / denom) if denom else float("nan"),
        "gt_divisions": float(len(gt_divisions)),
        "pred_divisions": float(len(pred_divisions)),
    }
