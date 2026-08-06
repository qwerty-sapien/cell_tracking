"""ZIP-compatible wrappers around the official tracking-cellmot metric."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import polars as pl
import tracksdata as td


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_SRC = ROOT / "external" / "kaggle-cell-tracking-competition" / "src"
if OFFICIAL_SRC.exists() and str(OFFICIAL_SRC) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_SRC))

from tracking_cellmot.metrics import (  # noqa: E402
    EvaluationResult,
    _evaluate,
    _evaluate_matched_graph,
    evaluate,
    nan_metrics_row,
    node_recall,
    per_sample_metrics,
    summarise,
)


def _empty_graph() -> td.graph.InMemoryGraph:
    graph = td.graph.InMemoryGraph()
    existing = set(graph.node_attr_keys())
    for key, dtype, default in (
        ("t", pl.Int64, 0),
        ("z", pl.Float64, 0.0),
        ("y", pl.Float64, 0.0),
        ("x", pl.Float64, 0.0),
    ):
        if key not in existing:
            graph.add_node_attr_key(key, dtype, default)
    return graph


def prediction_to_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> td.graph.InMemoryGraph:
    """Convert submission-style node/edge dicts to a tracksdata graph."""

    graph = _empty_graph()
    node_id_map: dict[int, int] = {}
    for row in sorted(nodes, key=lambda r: (int(r["t"]), int(r["node_id"]))):
        original_id = int(row["node_id"])
        graph_id = graph.add_node(
            attrs={
                "t": int(row["t"]),
                "z": float(row["z"]),
                "y": float(row["y"]),
                "x": float(row["x"]),
            }
        )
        node_id_map[original_id] = int(graph_id)
    seen: set[tuple[int, int]] = set()
    for row in edges:
        source = node_id_map.get(int(row["source_id"]))
        target = node_id_map.get(int(row["target_id"]))
        if source is None or target is None:
            continue
        pair = (source, target)
        if pair in seen or source == target:
            continue
        seen.add(pair)
        graph.add_edge(source, target, {})
    return graph


def geff_arrays_to_graph(gt: dict[str, Any]) -> td.graph.InMemoryGraph:
    """Convert ZIP-read GEFF arrays from ``read_geff_arrays`` to a graph."""

    graph = _empty_graph()
    nodes = gt["nodes"]
    node_id_map: dict[int, int] = {}
    for original_id, t, z, y, x in sorted(
        zip(nodes["node_id"], nodes["t"], nodes["z"], nodes["y"], nodes["x"], strict=True),
        key=lambda row: (int(row[1]), int(row[0])),
    ):
        graph_id = graph.add_node(
            attrs={
                "t": int(t),
                "z": float(z),
                "y": float(y),
                "x": float(x),
            }
        )
        node_id_map[int(original_id)] = int(graph_id)
    for source_id, target_id in gt["edges"]:
        source = node_id_map.get(int(source_id))
        target = node_id_map.get(int(target_id))
        if source is not None and target is not None and source != target:
            graph.add_edge(source, target, {})
    return graph


def _estimated_number_of_nodes(gt: dict[str, Any]) -> float:
    estimated = gt.get("estimated_number_of_nodes")
    try:
        return float(estimated)
    except (TypeError, ValueError):
        return float("nan")


def _nan_row(gt: dict[str, Any], num_pred_nodes: int) -> dict[str, float]:
    row = nan_metrics_row()
    row["division_jaccard"] = float("nan")
    row["estimated_number_of_nodes"] = _estimated_number_of_nodes(gt)
    row["num_pred_nodes"] = float(num_pred_nodes)
    return row


def _safe_node_recall(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
) -> float:
    if gt_graph.num_nodes() == 0:
        return float("nan")
    if pred_graph.num_nodes() == 0:
        return 0.0
    return node_recall(pred_graph, gt_graph)


def official_edge_metric_row(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
) -> dict[str, float]:
    """Evaluate only official edge/node metrics, skipping division scoring.

    This is intended for fast route and parameter probes where division metrics
    are not being optimized. Final CV audits should still use
    :func:`official_metric_row`, which runs the full Kaggle-style edge and
    division scorer.
    """

    if len(gt.get("nodes", {}).get("node_id", ())) == 0:
        return _nan_row(gt, len(nodes))

    pred_graph = prediction_to_graph(nodes, edges)
    gt_graph = geff_arrays_to_graph(gt)

    if pred_graph.num_edges() == 0 or pred_graph.num_nodes() == 0:
        edge_tp = 0
        edge_fp = 0
        edge_fn = gt_graph.num_edges()
        recall = 0.0
    else:
        _evaluate(pred_graph, gt_graph, "jaccard", scale_zyx, 7.0)
        edge_attrs = _evaluate_matched_graph(pred_graph, gt_graph)
        edge_tp = int(edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
        edge_valid_pred = int(edge_attrs["pred_valid"].sum())
        edge_fp = edge_valid_pred - edge_tp
        edge_fn = gt_graph.num_edges() - edge_tp
        recall = _safe_node_recall(pred_graph, gt_graph)

    estimated_float = _estimated_number_of_nodes(gt)
    er = EvaluationResult(
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        division_tp=0,
        division_fp=0,
        division_fn=0,
        num_pred_nodes=pred_graph.num_nodes(),
    )
    row = per_sample_metrics(er, estimated_float, recall)
    row["division_jaccard"] = float("nan")
    row["estimated_number_of_nodes"] = estimated_float
    return {key: float(value) if isinstance(value, (int, float)) else value for key, value in row.items()}


def official_metric_row(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
) -> dict[str, float]:
    """Evaluate one prediction/GT pair with the official metric implementation.

    Full competition samples have GT nodes, but clipped sparse-label diagnostics can
    legitimately contain no annotated GT nodes. The official implementation's
    Polars join currently crashes for that degenerate clip; return a NaN row so
    summaries skip it instead of converting an empty-label clip into evidence.
    """

    if len(gt.get("nodes", {}).get("node_id", ())) == 0:
        return _nan_row(gt, len(nodes))

    pred_graph = prediction_to_graph(nodes, edges)
    gt_graph = geff_arrays_to_graph(gt)
    er = evaluate(pred_graph, gt_graph, scale=scale_zyx)
    # Match the organizer's full-evaluation contract exactly. ``evaluate``
    # intentionally skips node matching when the prediction has no edges, so
    # calling ``node_recall`` in that state reads absent match attributes and
    # raises. The official CLI records zero recall for both node-only and empty
    # predictions instead (external/.../scripts/evaluate.py::evaluate_pairs).
    recall = (
        node_recall(pred_graph, gt_graph)
        if pred_graph.num_edges() > 0 and pred_graph.num_nodes() > 0
        else 0.0
    )
    estimated_float = _estimated_number_of_nodes(gt)
    row = per_sample_metrics(er, estimated_float, recall)
    denom = row["division_tp"] + row["division_fp"] + row["division_fn"]
    row["division_jaccard"] = row["division_tp"] / denom if denom else float("nan")
    row["estimated_number_of_nodes"] = estimated_float
    return {key: float(value) if isinstance(value, (int, float)) else value for key, value in row.items()}


def official_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate official per-sample rows."""

    return summarise(rows)


def score_gt_as_prediction(
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
) -> dict[str, float]:
    """Diagnostic: GEFF converted to prediction should score perfectly on edges/divisions."""

    nodes = [
        {
            "dataset": "gt",
            "node_id": int(node_id),
            "t": int(t),
            "z": int(z),
            "y": int(y),
            "x": int(x),
        }
        for node_id, t, z, y, x in zip(
            gt["nodes"]["node_id"],
            gt["nodes"]["t"],
            gt["nodes"]["z"],
            gt["nodes"]["y"],
            gt["nodes"]["x"],
            strict=True,
        )
    ]
    edges = [
        {"source_id": int(source), "target_id": int(target)}
        for source, target in gt["edges"]
    ]
    return official_metric_row(nodes, edges, gt, scale_zyx)
