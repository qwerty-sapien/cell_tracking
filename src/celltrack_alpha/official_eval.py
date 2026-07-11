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

from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise  # noqa: E402


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


def official_metric_row(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
) -> dict[str, float]:
    """Evaluate one prediction/GT pair with the official metric implementation."""

    pred_graph = prediction_to_graph(nodes, edges)
    gt_graph = geff_arrays_to_graph(gt)
    er = evaluate(pred_graph, gt_graph, scale=scale_zyx)
    recall = node_recall(pred_graph, gt_graph)
    estimated = gt.get("estimated_number_of_nodes")
    try:
        estimated_float = float(estimated)
    except (TypeError, ValueError):
        estimated_float = float("nan")
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
