#!/usr/bin/env python3
"""Measure fit-only, graph-valid action-class oracle headroom.

This diagnostic uses the same fit117 sparse caches, c=0.25 fragments, official
7 um matching, and frozen scorer implementation as CG0.  It fits no model and
never reads CV57, audit, test, or public artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np
from scipy.spatial import cKDTree
import tracksdata as td
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import analyze_candgen_ceiling as cg0  # noqa: E402
import scoreall  # noqa: E402
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.edge_data import match_detections_to_gt  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
    read_submission,
    validate_graphs,
)
from celltrack_alpha.official_eval import (  # noqa: E402
    geff_arrays_to_graph,
    official_metric_row,
    official_summary,
    prediction_to_graph,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from p4_edge_attention import _cache_identity  # noqa: E402


SCHEMA = "celltrack-alpha.action-class-oracle.v1"
FAMILIES = ("44b6", "6bba")
ACTION_CLASSES = (
    "add",
    "delete",
    "swap",
    "gap_bridge",
    "division_fork",
    "node_add",
    "node_restore",
)
TARGETS = (0.895, 0.9255, 0.9422)
FORBIDDEN_TOKENS = ("cv57", "holdout", "audit", "test")
SCALE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)

DEFAULT_REACHABILITY = ROOT / "reports/g02/extended_radius_reachability_v1.json"
DEFAULT_CG0 = ROOT / "reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.json"
DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"
DEFAULT_FRAGMENTS = ROOT / "reports/g02/fitcache_117/bh9out/fragments"
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.md"
)


class ActionOracleError(RuntimeError):
    """The diagnostic cannot preserve its frozen scope or graph contract."""


def _safe_path(path: Path) -> None:
    lowered = str(path).lower()
    token = next((value for value in FORBIDDEN_TOKENS if value in lowered), None)
    if token is not None:
        raise ActionOracleError(f"forbidden artifact token {token!r}: {path}")


def _sha256(path: Path) -> str:
    _safe_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    _safe_path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActionOracleError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ActionOracleError(f"JSON root is not an object: {path}")
    return value


def _family(identity: str) -> str:
    value = identity.split(":", 1)[1].split("_", 1)[0]
    if value not in FAMILIES:
        raise ActionOracleError(f"unexpected family: {identity}")
    return value


def _graph_edges(graph: SubmissionGraph) -> set[tuple[int, int]]:
    return {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}


def _copy_with(
    graph: SubmissionGraph,
    *,
    nodes: Mapping[int, SubmissionNode] | None = None,
    edges: set[tuple[int, int]] | None = None,
) -> SubmissionGraph:
    result = SubmissionGraph(
        graph.dataset,
        dict(graph.nodes if nodes is None else nodes),
        [
            SubmissionEdge(source, target)
            for source, target in sorted(
                _graph_edges(graph) if edges is None else edges
            )
        ],
    )
    validate_graphs({result.dataset: result})
    return result


def _degrees(
    edges: set[tuple[int, int]],
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    outgoing: dict[int, set[int]] = defaultdict(set)
    incoming: dict[int, set[int]] = defaultdict(set)
    for source, target in edges:
        outgoing[source].add(target)
        incoming[target].add(source)
    return outgoing, incoming


def _replace_incident(
    edges: set[tuple[int, int]],
    *,
    sources: set[int],
    targets: set[int],
    additions: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], int]:
    kept = {
        pair
        for pair in edges
        if pair[0] not in sources and pair[1] not in targets
    }
    removed = len(edges) - len(kept)
    kept.update(additions)
    return kept, removed


def _ordered_original_ids(graph: SubmissionGraph) -> list[int]:
    return [
        int(node.node_id)
        for node in sorted(graph.nodes.values(), key=lambda value: (value.t, value.node_id))
    ]


def official_node_matching(
    graph: SubmissionGraph,
    gt: Mapping[str, Any],
    scale_zyx: tuple[float, float, float],
) -> tuple[dict[int, int], dict[int, int]]:
    """Return original prediction↔original GT node maps under official matching."""

    nodes, _edges = scoreall._prediction_rows(graph.dataset, graph)
    pred = prediction_to_graph(nodes, [])
    truth = geff_arrays_to_graph(dict(gt))
    pred_original = _ordered_original_ids(graph)
    gt_original = [
        int(node_id)
        for node_id, t in sorted(
            zip(gt["nodes"]["node_id"], gt["nodes"]["t"], strict=True),
            key=lambda row: (int(row[1]), int(row[0])),
        )
    ]
    pred_internal = [int(value) for value in pred.node_attrs()["node_id"].to_list()]
    gt_internal = [int(value) for value in truth.node_attrs()["node_id"].to_list()]
    if pred_internal != list(range(len(pred_original))) or gt_internal != list(
        range(len(gt_original))
    ):
        raise ActionOracleError("internal graph node order is not canonical")

    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import get_options, set_options

    previous = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred.match(
                truth,
                matching=DistanceMatching(max_distance=7.0, scale=scale_zyx),
            )
    finally:
        set_options(show_progress=previous)

    attrs = pred.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    for pred_id, gt_id in attrs.iter_rows():
        if gt_id is None or int(gt_id) == -1:
            continue
        pred_original_id = pred_original[int(pred_id)]
        gt_original_id = gt_original[int(gt_id)]
        if pred_original_id in pred_to_gt or gt_original_id in gt_to_pred:
            raise ActionOracleError("official node matching is not one-to-one")
        pred_to_gt[pred_original_id] = gt_original_id
        gt_to_pred[gt_original_id] = pred_original_id
    return pred_to_gt, gt_to_pred


def _truth_topology(
    gt: Mapping[str, Any],
) -> tuple[
    set[tuple[int, int]],
    dict[int, set[int]],
    dict[int, set[int]],
    dict[int, dict[str, int]],
]:
    truth = {(int(source), int(target)) for source, target in gt["edges"]}
    successors: dict[int, set[int]] = defaultdict(set)
    predecessors: dict[int, set[int]] = defaultdict(set)
    for source, target in truth:
        successors[source].add(target)
        predecessors[target].add(source)
    nodes = {
        int(node_id): {
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
    }
    return truth, successors, predecessors, nodes


def _true_prediction_edge(
    pair: tuple[int, int],
    pred_to_gt: Mapping[int, int],
    truth: set[tuple[int, int]],
) -> bool:
    source = pred_to_gt.get(pair[0])
    target = pred_to_gt.get(pair[1])
    return source is not None and target is not None and (source, target) in truth


def _current_candidates(
    cache: Any,
    gt: Mapping[str, Any],
    *,
    successors: Mapping[int, set[int]],
    predecessors: Mapping[int, set[int]],
) -> tuple[set[tuple[int, int]], dict[str, int]]:
    """Return true non-division adjacent r10/k8 prediction edges and pool counts."""

    detection_to_gt, gt_edges_raw = match_detections_to_gt(cache.detections, gt)
    gt_to_detection = {
        int(gt_id): int(detection_id)
        for detection_id, gt_id in detection_to_gt.items()
        if gt_id is not None
    }
    gt_edges = {(int(source), int(target)) for source, target in gt_edges_raw}
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    detections: dict[int, tuple[int, np.ndarray]] = {}
    pool_size = 0
    division_pair_pool = 0
    for frame in cache.detections:
        node_ids = np.asarray(frame.node_ids, dtype=np.int64)
        points = np.asarray(frame.coords_zyx, dtype=np.float64) * SCALE_ZYX_UM
        frames[int(frame.frame_index)] = (node_ids, points)
        for node_id, point in zip(node_ids, points, strict=True):
            detections[int(node_id)] = (int(frame.frame_index), point)
    for frame_index, (_source_ids, source_points) in frames.items():
        target = frames.get(frame_index + 1)
        if target is None or not len(target[0]):
            continue
        target_ids, target_points = target
        tree = cKDTree(target_points)
        for source_point in source_points:
            indices = tree.query_ball_point(source_point, 10.0 + 1.0e-9)
            indices = sorted(
                indices,
                key=lambda index: (
                    float(np.linalg.norm(target_points[index] - source_point)),
                    int(target_ids[index]),
                    int(index),
                ),
            )[:8]
            pool_size += len(indices)
            division_pair_pool += len(indices) * (len(indices) - 1) // 2

    correct: set[tuple[int, int]] = set()
    for gt_source, gt_target in gt_edges:
        if (
            len(successors.get(gt_source, ())) != 1
            or len(predecessors.get(gt_target, ())) != 1
        ):
            continue
        source_id = gt_to_detection.get(gt_source)
        target_id = gt_to_detection.get(gt_target)
        if source_id is None or target_id is None:
            continue
        source_t, source_point = detections[source_id]
        target_t, _target_point = detections[target_id]
        if target_t != source_t + 1:
            continue
        target_ids, target_points = frames[target_t]
        distance, rank = cg0._candidate_rank(
            source_point, target_id, target_ids, target_points
        )
        if distance <= 10.0 + 1.0e-9 and rank <= 8:
            correct.add((source_id, target_id))
    return correct, {
        "adjacent_pool": pool_size,
        "adjacent_correct": len(correct),
        "adjacent_background": pool_size - len(correct),
        "division_pair_pool": division_pair_pool,
    }


def _gap_events(
    graph: SubmissionGraph,
    gt_to_pred: Mapping[int, int],
    gt_nodes: Mapping[int, Mapping[str, int]],
    successors: Mapping[int, set[int]],
    predecessors: Mapping[int, set[int]],
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    frames: dict[int, list[SubmissionNode]] = defaultdict(list)
    for node in graph.nodes.values():
        frames[int(node.t)].append(node)
    gap_pool = 0
    for frame_index, sources in frames.items():
        targets = frames.get(frame_index + 2, [])
        if not targets:
            continue
        target_ids = np.asarray([node.node_id for node in targets], dtype=np.int64)
        target_points = (
            np.asarray([node.coord_zyx for node in targets], dtype=np.float64)
            * SCALE_ZYX_UM
        )
        tree = cKDTree(target_points)
        for source in sources:
            source_point = np.asarray(source.coord_zyx, dtype=np.float64) * SCALE_ZYX_UM
            indices = tree.query_ball_point(source_point, 20.0 + 1.0e-9)
            indices = sorted(
                indices,
                key=lambda index: (
                    float(np.linalg.norm(target_points[index] - source_point)),
                    int(target_ids[index]),
                    int(index),
                ),
            )[:8]
            gap_pool += len(indices)

    events: list[tuple[int, int, int]] = []
    for gt_source, middle_values in successors.items():
        if len(middle_values) != 1:
            continue
        middle = next(iter(middle_values))
        target_values = successors.get(middle, set())
        if (
            len(predecessors.get(middle, ())) != 1
            or len(target_values) != 1
        ):
            continue
        gt_target = next(iter(target_values))
        if len(predecessors.get(gt_target, ())) != 1:
            continue
        source_id = gt_to_pred.get(gt_source)
        target_id = gt_to_pred.get(gt_target)
        if (
            source_id is None
            or target_id is None
            or middle in gt_to_pred
            or source_id not in graph.nodes
            or target_id not in graph.nodes
        ):
            continue
        source = graph.nodes[source_id]
        target = graph.nodes[target_id]
        if target.t != source.t + 2:
            continue
        source_point = np.asarray(source.coord_zyx, dtype=np.float64) * SCALE_ZYX_UM
        target_point = np.asarray(target.coord_zyx, dtype=np.float64) * SCALE_ZYX_UM
        candidates = frames[target.t]
        target_ids = np.asarray([node.node_id for node in candidates], dtype=np.int64)
        target_points = (
            np.asarray([node.coord_zyx for node in candidates], dtype=np.float64)
            * SCALE_ZYX_UM
        )
        distance, rank = cg0._candidate_rank(
            source_point, target_id, target_ids, target_points
        )
        middle_row = gt_nodes[middle]
        middle_point = (
            np.asarray(
                (middle_row["z"], middle_row["y"], middle_row["x"]),
                dtype=np.float64,
            )
            * SCALE_ZYX_UM
        )
        if (
            distance <= 20.0 + 1.0e-9
            and rank <= 8
            and np.linalg.norm((source_point + target_point) / 2.0 - middle_point)
            <= 7.0 + 1.0e-9
        ):
            events.append((gt_source, middle, gt_target))
    return events, {
        "gap_pair_pool": gap_pool,
        "gap_correct_events": len(events),
        "gap_background_pairs": gap_pool - len(events),
    }


def _fresh_ids(graph: SubmissionGraph, gt_ids: Sequence[int]) -> dict[int, int]:
    next_id = max(graph.nodes, default=-1) + 1
    result = {}
    for gt_id in sorted(gt_ids):
        while next_id in graph.nodes or next_id in result.values():
            next_id += 1
        result[int(gt_id)] = next_id
        next_id += 1
    return result


def build_action_graphs(
    graph: SubmissionGraph,
    gt: Mapping[str, Any],
    scale_zyx: tuple[float, float, float],
    cache: Any,
) -> tuple[dict[str, SubmissionGraph], dict[str, dict[str, int | str | None]]]:
    """Construct one perfect, graph-valid intervention per action class."""

    pred_to_gt, gt_to_pred = official_node_matching(graph, gt, scale_zyx)
    truth, successors, predecessors, gt_nodes = _truth_topology(gt)
    baseline_edges = _graph_edges(graph)
    true_current, pool = _current_candidates(
        cache,
        gt,
        successors=successors,
        predecessors=predecessors,
    )
    true_current = {
        pair
        for pair in true_current
        if pair[0] in graph.nodes and pair[1] in graph.nodes
        and _true_prediction_edge(pair, pred_to_gt, truth)
    }
    pool["adjacent_correct"] = len(true_current)
    pool["adjacent_background"] = int(pool["adjacent_pool"]) - len(true_current)
    gap_events, gap_pool = _gap_events(
        graph, gt_to_pred, gt_nodes, successors, predecessors
    )
    outputs: dict[str, SubmissionGraph] = {}
    counts: dict[str, dict[str, int | str | None]] = {}

    wrong_edges = {
        pair
        for pair in baseline_edges
        if not _true_prediction_edge(pair, pred_to_gt, truth)
    }
    outputs["delete"] = _copy_with(graph, edges=baseline_edges - wrong_edges)
    counts["delete"] = {
        "positive_actions": len(wrong_edges),
        "edges_added": 0,
        "edges_removed": len(wrong_edges),
        "nodes_added": 0,
        "raw_background_actions": len(baseline_edges) - len(wrong_edges),
        "background_definition": "baseline true edges that a deletion policy must retain",
    }

    outgoing, incoming = _degrees(baseline_edges)
    free_additions = {
        pair
        for pair in true_current
        if pair not in baseline_edges
        and not outgoing.get(pair[0])
        and not incoming.get(pair[1])
    }
    outputs["add"] = _copy_with(graph, edges=baseline_edges | free_additions)
    counts["add"] = {
        "positive_actions": len(free_additions),
        "edges_added": len(free_additions),
        "edges_removed": 0,
        "nodes_added": 0,
        "raw_background_actions": int(pool["adjacent_background"]),
        "background_definition": "non-GT r10/k8 adjacent candidate edges",
    }

    swap_additions = {
        pair for pair in true_current if pair not in baseline_edges
    }
    swap_sources = {source for source, _target in swap_additions}
    swap_targets = {target for _source, target in swap_additions}
    swap_edges, swap_removed = _replace_incident(
        baseline_edges,
        sources=swap_sources,
        targets=swap_targets,
        additions=swap_additions,
    )
    outputs["swap"] = _copy_with(graph, edges=swap_edges)
    counts["swap"] = {
        "positive_actions": len(swap_additions),
        "edges_added": len(swap_additions),
        "edges_removed": swap_removed,
        "nodes_added": 0,
        "raw_background_actions": int(pool["adjacent_background"]),
        "background_definition": "non-GT r10/k8 adjacent candidate edges",
    }

    gap_gt_ids = [middle for _source, middle, _target in gap_events]
    gap_ids = _fresh_ids(graph, gap_gt_ids)
    gap_nodes = dict(graph.nodes)
    gap_additions: set[tuple[int, int]] = set()
    gap_sources: set[int] = set()
    gap_targets: set[int] = set()
    for gt_source, middle, gt_target in gap_events:
        row = gt_nodes[middle]
        node_id = gap_ids[middle]
        gap_nodes[node_id] = SubmissionNode(
            node_id=node_id,
            t=int(row["t"]),
            z=int(row["z"]),
            y=int(row["y"]),
            x=int(row["x"]),
        )
        source_id = gt_to_pred[gt_source]
        target_id = gt_to_pred[gt_target]
        gap_sources.add(source_id)
        gap_targets.add(target_id)
        gap_additions.add((source_id, node_id))
        gap_additions.add((node_id, target_id))
    gap_edges, gap_removed = _replace_incident(
        baseline_edges,
        sources=gap_sources,
        targets=gap_targets,
        additions=gap_additions,
    )
    outputs["gap_bridge"] = _copy_with(
        graph, nodes=gap_nodes, edges=gap_edges
    )
    counts["gap_bridge"] = {
        "positive_actions": len(gap_events),
        "edges_added": len(gap_additions),
        "edges_removed": gap_removed,
        "nodes_added": len(gap_ids),
        "raw_background_actions": int(gap_pool["gap_background_pairs"]),
        "background_definition": "non-GT r20/k8 two-frame endpoint pairs",
    }

    division_events = []
    for parent, children in successors.items():
        if len(children) != 2 or parent not in gt_to_pred:
            continue
        if all(child in gt_to_pred for child in children):
            division_events.append((parent, tuple(sorted(children))))
    division_sources: set[int] = set()
    division_targets: set[int] = set()
    division_additions: set[tuple[int, int]] = set()
    for parent, children in division_events:
        parent_id = gt_to_pred[parent]
        division_sources.add(parent_id)
        for child in children:
            child_id = gt_to_pred[child]
            division_targets.add(child_id)
            division_additions.add((parent_id, child_id))
    division_edges, division_removed = _replace_incident(
        baseline_edges,
        sources=division_sources,
        targets=division_targets,
        additions=division_additions,
    )
    outputs["division_fork"] = _copy_with(graph, edges=division_edges)
    counts["division_fork"] = {
        "positive_actions": len(division_events),
        "edges_added": len(division_additions),
        "edges_removed": division_removed,
        "nodes_added": 0,
        "raw_background_actions": max(
            0, int(pool["division_pair_pool"]) - len(division_events)
        ),
        "background_definition": "non-GT daughter pairs among each source's r10/k8 next-frame candidates",
    }

    missing_gt = sorted(set(gt_nodes) - set(gt_to_pred))
    restored_ids = _fresh_ids(graph, missing_gt)
    added_nodes = dict(graph.nodes)
    for gt_id, node_id in restored_ids.items():
        row = gt_nodes[gt_id]
        added_nodes[node_id] = SubmissionNode(
            node_id=node_id,
            t=int(row["t"]),
            z=int(row["z"]),
            y=int(row["y"]),
            x=int(row["x"]),
        )
    outputs["node_add"] = _copy_with(
        graph, nodes=added_nodes, edges=baseline_edges
    )
    counts["node_add"] = {
        "positive_actions": len(restored_ids),
        "edges_added": 0,
        "edges_removed": 0,
        "nodes_added": len(restored_ids),
        "raw_background_actions": None,
        "background_definition": "UNDEFINED: no banked node proposal generator",
    }

    complete_gt_to_pred = {**gt_to_pred, **restored_ids}
    affected_sources = {
        source
        for source, target in truth
        if source in restored_ids or target in restored_ids
    }
    affected_targets = {
        target
        for source, target in truth
        if source in restored_ids or target in restored_ids
    }
    restore_sources = {
        complete_gt_to_pred[source]
        for source in affected_sources
        if source in complete_gt_to_pred
    }
    restore_targets = {
        complete_gt_to_pred[target]
        for target in affected_targets
        if target in complete_gt_to_pred
    }
    restore_additions = {
        (complete_gt_to_pred[source], complete_gt_to_pred[target])
        for source, target in truth
        if (source in affected_sources or target in affected_targets)
        and source in complete_gt_to_pred
        and target in complete_gt_to_pred
    }
    restore_edges, restore_removed = _replace_incident(
        baseline_edges,
        sources=restore_sources,
        targets=restore_targets,
        additions=restore_additions,
    )
    outputs["node_restore"] = _copy_with(
        graph, nodes=added_nodes, edges=restore_edges
    )
    counts["node_restore"] = {
        "positive_actions": len(restored_ids),
        "edges_added": len(restore_additions),
        "edges_removed": restore_removed,
        "nodes_added": len(restored_ids),
        "raw_background_actions": None,
        "background_definition": "UNDEFINED: no banked node proposal generator",
    }
    return outputs, counts


def _metric(
    graph: SubmissionGraph,
    gt: Mapping[str, Any],
    scale: tuple[float, float, float],
) -> dict[str, float]:
    nodes, edges = scoreall._prediction_rows(graph.dataset, graph)
    return official_metric_row(nodes, edges, dict(gt), scale)


def _delta(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    keys = (
        "score",
        "adj_edge_jaccard",
        "edge_jaccard",
        "node_recall",
        "division_jaccard",
    )
    result = {}
    for key in keys:
        left = float(candidate[key])
        right = float(baseline[key])
        result[key] = (
            left - right
            if math.isfinite(left) and math.isfinite(right)
            else float("nan")
        )
    return result


def select_oracle_row(
    action: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Return the better exact-objective row, allowing a perfect policy to abstain."""

    action_score = float(
        action["score"]
        if "score" in action
        else official_summary([dict(action)])["score"]
    )
    baseline_score = float(
        baseline["score"]
        if "score" in baseline
        else official_summary([dict(baseline)])["score"]
    )
    if not math.isfinite(action_score) or not math.isfinite(baseline_score):
        raise ActionOracleError("non-finite score cannot drive oracle abstention")
    no_op_selected = action_score < baseline_score
    return dict(baseline if no_op_selected else action), no_op_selected


def edge_fp_budget(
    summary: Mapping[str, Any], target: float
) -> dict[str, int | float | bool | None]:
    raw = float(summary["edge_jaccard"])
    adjusted = float(summary["adj_edge_jaccard"])
    division = float(summary["division_jaccard"])
    if not math.isfinite(division):
        division = 0.0
    node_adjustment = adjusted / raw if raw > 0.0 else float("nan")
    required_adjusted = target - 0.1 * division
    if required_adjusted <= 0.0:
        return {
            "target_score": target,
            "additional_edge_fp_allowed": None,
            "reason": "division contribution alone reaches target",
        }
    required_raw = required_adjusted / node_adjustment
    if not 0.0 < required_raw <= 1.0:
        return {
            "target_score": target,
            "additional_edge_fp_allowed": 0,
            "reaches_target_before_extra_fp": False,
            "reason": "oracle is below target before any extra edge FP",
        }
    tp = int(summary["edge_tp"])
    fp = int(summary["edge_fp"])
    fn = int(summary["edge_fn"])
    maximum_total_fp = math.floor(tp / required_raw - tp - fn + 1.0e-12)
    additional = max(0, maximum_total_fp - fp)
    return {
        "target_score": target,
        "maximum_total_edge_fp": maximum_total_fp,
        "observed_oracle_edge_fp": fp,
        "additional_edge_fp_allowed": additional,
        "reaches_target_before_extra_fp": float(summary["score"]) >= target,
        "reason": "holds TP/FN/node/division fixed; edge-FP-equivalent only",
    }


def _summarise(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = dict(official_summary([dict(row) for row in rows]))
    for key in (
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
    ):
        summary[key] = int(sum(int(float(row[key])) for row in rows))
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    for path in (
        args.reachability,
        args.cg0,
        args.zip,
        args.cache,
        args.fragments,
        args.output,
        args.markdown,
    ):
        _safe_path(path)
    reachability = _load_json(args.reachability)
    cg0_report = _load_json(args.cg0)
    scope = reachability.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("fit_only") is not True
        or any(scope.get(key) is not False for key in ("held_out_used", "audit_used", "test_used"))
    ):
        raise ActionOracleError("reachability scope is not fit-only")
    train_ids = tuple(str(value) for value in scope["train_video_ids"])
    validation_ids = tuple(str(value) for value in scope["validation_video_ids"])
    identities = tuple(sorted((*train_ids, *validation_ids)))
    if (
        len(train_ids) != 105
        or len(validation_ids) != 12
        or len(identities) != 117
        or len(set(identities)) != 117
    ):
        raise ActionOracleError("fit117 identity contract drifted")
    if cg0_report.get("verdict") != "GO":
        raise ActionOracleError("banked CG0 root gate is not GO")

    fragment_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.fragments.glob("train__*.csv")
    }
    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if (
        set(fragment_paths) != set(identities)
        or set(cache_paths) != set(identities)
        or set(meta_paths) != set(identities)
    ):
        raise ActionOracleError("fit117 fragment/cache coverage drifted")
    full_identities = identities
    if args.max_videos_per_family is not None:
        if args.max_videos_per_family <= 0:
            raise ActionOracleError("--max-videos-per-family must be positive")
        identities = tuple(
            identity
            for family in FAMILIES
            for identity in [
                value
                for value in full_identities
                if _family(value) == family
            ][: args.max_videos_per_family]
        )

    rows = {
        family: {"baseline": [], **{name: [] for name in ACTION_CLASSES}}
        for family in FAMILIES
    }
    counts = {
        family: {name: Counter() for name in ACTION_CLASSES}
        for family in FAMILIES
    }
    sample_records = []
    store = ZipStore(args.zip, mode="r")
    try:
        for index, identity in enumerate(identities, start=1):
            family = _family(identity)
            dataset = identity.split(":", 1)[1]
            graphs = read_submission(fragment_paths[identity])
            graph = graphs.get(dataset)
            if graph is None or len(graphs) != 1:
                raise ActionOracleError(f"fragment identity mismatch: {identity}")
            cache = bh_model.load_sparse_cache(
                cache_paths[identity],
                expected_identity=_cache_identity(meta_paths[identity]),
            )
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            baseline_row = _metric(graph, gt, scale)
            rows[family]["baseline"].append(baseline_row)
            action_graphs, action_counts = build_action_graphs(graph, gt, scale, cache)
            sample = {
                "identity": identity,
                "family": family,
                "baseline_fragment_sha256": _sha256(fragment_paths[identity]),
                "baseline": baseline_row,
                "actions": {},
            }
            for name in ACTION_CLASSES:
                raw_action_row = _metric(action_graphs[name], gt, scale)
                action_row, no_op_selected = select_oracle_row(
                    raw_action_row, baseline_row
                )
                rows[family][name].append(action_row)
                numeric = {
                    key: int(value)
                    for key, value in action_counts[name].items()
                    if isinstance(value, int)
                }
                counts[family][name].update(numeric)
                counts[family][name].update(
                    {
                        "oracle_selected_video_count": int(not no_op_selected),
                        "no_op_selected_video_count": int(no_op_selected),
                    }
                )
                sample["actions"][name] = {
                    "counts": action_counts[name],
                    "raw_action_metrics": raw_action_row,
                    "oracle_metrics": action_row,
                    "no_op_selected": no_op_selected,
                }
            sample_records.append(sample)
            print(
                f"[{index:03d}/117] {identity}",
                flush=True,
            )
            del graph, cache, gt, group, action_graphs
    finally:
        if hasattr(store, "_lock"):
            store.close()

    families: dict[str, Any] = {}
    for family in FAMILIES:
        baseline = _summarise(rows[family]["baseline"])
        action_rows = {}
        for name in ACTION_CLASSES:
            summary = _summarise(rows[family][name])
            definitions = {
                str(sample["actions"][name]["counts"]["background_definition"])
                for sample in sample_records
                if sample["family"] == family
            }
            if len(definitions) != 1:
                raise ActionOracleError(f"{family}/{name} background definition drifted")
            action_rows[name] = {
                "oracle": summary,
                "delta_vs_control": _delta(summary, baseline),
                "action_counts": {
                    **dict(counts[family][name]),
                    "background_definition": next(iter(definitions)),
                },
                "edge_fp_equivalent_budgets": {
                    f"{target:.4f}": edge_fp_budget(summary, target)
                    for target in TARGETS
                },
            }
        families[family] = {
            "video_count": len(rows[family]["baseline"]),
            "control": baseline,
            "actions": action_rows,
        }

    provenance_files = sorted(
        (
            {
                "identity": identity,
                "fragment_sha256": _sha256(fragment_paths[identity]),
                "cache_sha256": _sha256(cache_paths[identity]),
                "meta_sha256": _sha256(meta_paths[identity]),
            }
            for identity in identities
        ),
        key=lambda row: row["identity"],
    )
    report = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "purpose": "fit-only diagnostic upper bound; not a candidate or training result",
        "oracle_policy": (
            "per-video exact-objective selection between the perfect class "
            "intervention and an explicit no-op"
        ),
        "control": {
            "name": "fit117 c=0.25 fragment graph used by CG0",
            "fragment_root": str(args.fragments.relative_to(ROOT)),
            "common_across_actions": True,
            "apncc_incumbent": False,
        },
        "action_definitions": {
            "add": "add only missing official-matching true non-division r10/k8 edges with free source and target endpoints; no deletions",
            "delete": "delete every baseline edge that is not a matched GT edge",
            "swap": "add every missing official-matching true non-division r10/k8 edge and remove all conflicting source-out/target-in edges",
            "gap_bridge": "restore one missing intermediate GT node for each valid r20/k8 two-frame event and connect its two true edges",
            "division_fork": "replace parent-out and daughter-in conflicts with the exact two-child GT fork when all three detections are matched",
            "node_add": "add every unmatched GT node at its GT coordinate with no incident edges",
            "node_restore": "add every unmatched GT node and restore all GT incident edges, replacing conflicts at affected endpoints",
        },
        "scope": {
            "fit_only": True,
            "train_video_count": len(train_ids),
            "validation_video_count": len(validation_ids),
            "family_counts": {
                family: sum(_family(identity) == family for identity in identities)
                for family in FAMILIES
            },
            "cv57_used": False,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
            "model_fitted": False,
            "nonfinal_smoke": args.max_videos_per_family is not None,
        },
        "budget_semantics": {
            "targets": list(TARGETS),
            "edge_fp_equivalent": "maximum extra edge FP while holding TP/FN/node/division fixed",
            "not_false_action_budget": True,
            "node_background": "undefined because no banked node proposal generator exists",
        },
        "families": families,
        "inputs": {
            "archive": {
                "path": str(args.zip.relative_to(ROOT)),
                "bytes": args.zip.stat().st_size,
                "sha256": _sha256(args.zip),
            },
            "reachability": {
                "path": str(args.reachability.relative_to(ROOT)),
                "sha256": _sha256(args.reachability),
            },
            "cg0": {
                "path": str(args.cg0.relative_to(ROOT)),
                "sha256": _sha256(args.cg0),
            },
            "official_eval_sha256": _sha256(ROOT / "src/celltrack_alpha/official_eval.py"),
            "official_metrics_sha256": _sha256(
                ROOT
                / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py"
            ),
            "official_division_metrics_sha256": _sha256(
                ROOT
                / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py"
            ),
            "fit117_files": provenance_files,
        },
        "samples": sample_records,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "external_action": False,
            "gpu_action": False,
            "training": False,
        },
    }
    return report


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}f}" if math.isfinite(number) else "N/A"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    return value


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P0 action-class oracle headroom",
        "",
        "**COMPLETE — DIAGNOSTIC ONLY / NO TRAINING**",
        "",
        "Control: the exact fit117 c=0.25 fragment graphs used by CG0. These are",
        "not the banked APNCC incumbent, so the table ranks action surfaces but does",
        "not claim additive headroom over local 0.881522.",
        "",
    ]
    for family in FAMILIES:
        payload = report["families"][family]
        control = payload["control"]
        lines.extend(
            [
                f"## {family}",
                "",
                f"Control score: **{_fmt(control['score'])}** over "
                f"{payload['video_count']} videos.",
                "",
                "| action | oracle score | Δ score | Δ adjusted | Δ raw | Δ node | Δ division J | positive actions | raw background |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ACTION_CLASSES:
            row = payload["actions"][name]
            delta = row["delta_vs_control"]
            action_counts = row["action_counts"]
            lines.append(
                f"| {name} | {_fmt(row['oracle']['score'])} | "
                f"{_fmt(delta['score'])} | {_fmt(delta['adj_edge_jaccard'])} | "
                f"{_fmt(delta['edge_jaccard'])} | {_fmt(delta['node_recall'])} | "
                f"{_fmt(delta['division_jaccard'])} | "
                f"{int(action_counts.get('positive_actions', 0)):,} | "
                + (
                    f"{int(action_counts['raw_background_actions']):,}"
                    if "raw_background_actions" in action_counts
                    else "N/A"
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "### Edge-FP-equivalent budgets",
                "",
                "| action | target 0.895 | target 0.9255 | target 0.9422 |",
                "|---|---:|---:|---:|",
            ]
        )
        for name in ACTION_CLASSES:
            budgets = payload["actions"][name]["edge_fp_equivalent_budgets"]
            values = []
            for target in TARGETS:
                value = budgets[f"{target:.4f}"]["additional_edge_fp_allowed"]
                values.append("N/A" if value is None else f"{int(value):,}")
            lines.append(f"| {name} | {values[0]} | {values[1]} | {values[2]} |")
        lines.extend(
            [
                "",
                "Budgets are extra edge-FP equivalents with TP/FN/node/division held",
                "fixed. They are not interchangeable with false forks or false nodes.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation constraints",
            "",
            "- Every transformed graph passes the submission graph validity contract.",
            "- Labels are used deliberately for an upper-bound diagnostic; no result is",
            "  eligible for model selection, CV57, or submission.",
            "- `node_add` and `node_restore` have no honest raw-background budget because",
            "  the banked record contains no node proposal generator.",
            "- P1 feasibility compilation and P2 division fitting remain unopened.",
            "",
        ]
    )
    return "\n".join(lines)


def _immutable_write(path: Path, content: bytes) -> None:
    if path.exists():
        raise ActionOracleError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--reachability", type=Path, default=DEFAULT_REACHABILITY)
    value.add_argument("--cg0", type=Path, default=DEFAULT_CG0)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--fragments", type=Path, default=DEFAULT_FRAGMENTS)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    value.add_argument("--max-videos-per-family", type=int)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = _json_safe(run(args))
    encoded = json.dumps(
        report, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("ascii") + b"\n"
    _immutable_write(args.output, encoded)
    _immutable_write(args.markdown, markdown(report).encode("utf-8"))
    print(json.dumps(
        {
            family: {
                name: report["families"][family]["actions"][name][
                    "delta_vs_control"
                ]["score"]
                for name in ACTION_CLASSES
            }
            for family in FAMILIES
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
