#!/usr/bin/env python3
"""Run Phase-1 E1/E2 diagnostics on the exact divisions-off CV57 baseline.

This is a read-only, CPU-only diagnostic.  Official scorer matching is done on
the persisted baseline fragments.  Distance ranks and constant-velocity
predictions use the canonical raw ``FrameDet`` coordinates from the matching
inference caches, i.e. the coordinates used by the association gate before
track smoothing and image-support projection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import tracksdata as td
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import score_bh_cv57  # noqa: E402
import scoreall  # noqa: E402
from celltrack_alpha import bh_graph, bh_model  # noqa: E402
from celltrack_alpha.official_eval import (  # noqa: E402
    geff_arrays_to_graph,
    prediction_to_graph,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from tracking_cellmot.metrics import _evaluate  # noqa: E402


SCHEMA = "celltrack-alpha.phase1-distance-motion-gate.v1"
EXPECTED_BASELINE = {
    "video_count": 57,
    "adj_edge_jaccard": 0.8711885823490648,
    "edge_jaccard": 0.8761665470208184,
    "edge_tp": 34174,
    "edge_fp": 2736,
    "edge_fn": 2094,
    "node_recall": 0.9903079300966847,
}
E1_PASS_FRACTION = 0.90
E2_PRIMARY_LARGE_FRACTION = 0.50
DISTANCE_EPSILON_UM = 1.0e-12


class Phase1DiagnosticError(RuntimeError):
    """The exact CV57 diagnostic contract could not be preserved."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase1DiagnosticError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Phase1DiagnosticError(f"{label} must be a JSON object: {path}")
    return value


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def rank_summary(
    ranks: Sequence[int], *, candidate_absent: int = 0
) -> dict[str, Any]:
    if candidate_absent < 0:
        raise Phase1DiagnosticError("candidate-absent count must be nonnegative")
    counts = Counter(int(rank) for rank in ranks)
    denominator = len(ranks) + candidate_absent
    at_5 = sum(value for rank, value in counts.items() if rank <= 5)
    at_10 = sum(value for rank, value in counts.items() if rank <= 10)
    return {
        "denominator": denominator,
        "rank_observed_denominator": len(ranks),
        "candidate_absent_or_unrankable": candidate_absent,
        "rank_le_5": at_5,
        "rank_le_5_fraction": fraction(at_5, denominator),
        "rank_le_10": at_10,
        "rank_le_10_fraction": fraction(at_10, denominator),
        "rank_quantiles": quantiles([float(rank) for rank in ranks]),
        "rank_histogram_1_to_10_plus": {
            **{str(rank): counts.get(rank, 0) for rank in range(1, 11)},
            ">10_or_absent": (
                sum(value for rank, value in counts.items() if rank > 10)
                + candidate_absent
            ),
        },
    }


def distance_rank(
    source_coord_zyx: Sequence[float],
    true_target_id: int,
    target_ids: Sequence[int] | np.ndarray,
    target_coords_zyx: Sequence[Sequence[float]] | np.ndarray,
    scale_zyx: Sequence[float],
) -> tuple[int, float]:
    """Return the exact gate-order rank and physical distance of a target.

    The tie break mirrors ``bh_edge.gate_sparse_candidates``: distance, target
    node ID, then input row.  Unlike the production gate, this diagnostic does
    not truncate at radius 10 or cap 8 because E1 asks where the true target
    ranks among all next-frame candidates.
    """

    ids = np.asarray(target_ids, dtype=np.int64)
    coords = np.asarray(target_coords_zyx, dtype=np.float64)
    source = np.asarray(source_coord_zyx, dtype=np.float64)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    if ids.ndim != 1 or coords.shape != (len(ids), 3):
        raise Phase1DiagnosticError("target candidate arrays have invalid shapes")
    if source.shape != (3,) or scale.shape != (3,):
        raise Phase1DiagnosticError("source coordinate or physical scale is invalid")
    target_rows = np.flatnonzero(ids == int(true_target_id))
    if len(target_rows) != 1:
        raise Phase1DiagnosticError(
            f"true target {true_target_id} occurs {len(target_rows)} times in target frame"
        )
    delta_um = (coords - source[None, :]) * scale[None, :]
    distances = np.sqrt(np.einsum("ij,ij->i", delta_um, delta_um))
    ordered = sorted(
        range(len(ids)),
        key=lambda row: (float(distances[row]), int(ids[row]), row),
    )
    true_row = int(target_rows[0])
    return ordered.index(true_row) + 1, float(distances[true_row])


def motion_predicted_coord(
    source_coord_zyx: Sequence[float],
    parent_coord_zyx: Sequence[float] | None,
    velocity_weight: float,
) -> np.ndarray:
    """Mirror ``bh_graph``'s constant-velocity coordinate prediction."""

    source = np.asarray(source_coord_zyx, dtype=np.float64)
    if source.shape != (3,) or not np.isfinite(source).all():
        raise Phase1DiagnosticError("source coordinate is invalid")
    if parent_coord_zyx is None:
        return source.copy()
    parent = np.asarray(parent_coord_zyx, dtype=np.float64)
    if parent.shape != (3,) or not np.isfinite(parent).all():
        raise Phase1DiagnosticError("parent coordinate is invalid")
    return source + float(velocity_weight) * (source - parent)


def physical_distance(
    left_zyx: Sequence[float],
    right_zyx: Sequence[float],
    scale_zyx: Sequence[float],
) -> float:
    delta = (np.asarray(left_zyx, dtype=np.float64) - np.asarray(right_zyx, dtype=np.float64)) * np.asarray(
        scale_zyx, dtype=np.float64
    )
    return float(np.linalg.norm(delta))


def cache_identity(meta: Mapping[str, Any]) -> bh_model.CacheIdentity:
    try:
        model = meta["contract"]["model"]
        return bh_model.CacheIdentity(
            config_sha256=str(model["config_sha256"]),
            source_sha256=str(model["source_sha256"]),
            weight_sha256=str(model["weight_sha256"]),
            pair_batch_size=int(model["pair_batch_size"]),
            view_batch_size=int(model["view_batch_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Phase1DiagnosticError("cache metadata lacks a valid model identity") from exc


def load_detections(cache_path: Path, meta_path: Path) -> tuple[bh_model.FrameDet, ...]:
    meta = load_object(meta_path, "cache metadata")
    if meta.get("schema") != "celltrack-alpha.bh-cache-meta.v1":
        raise Phase1DiagnosticError(f"unexpected cache metadata schema: {meta_path}")
    expected_sha = meta.get("cache_sha256")
    observed_sha = sha256_file(cache_path)
    if expected_sha != observed_sha:
        raise Phase1DiagnosticError(
            f"cache SHA-256 mismatch for {cache_path}: expected={expected_sha} observed={observed_sha}"
        )
    cache = bh_model.load_sparse_cache(
        cache_path, expected_identity=cache_identity(meta)
    )
    return cache.detections


def prediction_rows(graph: Any) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    nodes = [
        {
            "node_id": int(node.node_id),
            "t": int(node.t),
            "z": float(node.z),
            "y": float(node.y),
            "x": float(node.x),
        }
        for node in graph.nodes.values()
    ]
    edges = [
        {"source_id": int(edge.source_id), "target_id": int(edge.target_id)}
        for edge in graph.edges
    ]
    return nodes, edges


def official_matches(
    graph: Any,
    gt: Mapping[str, Any],
    scale_zyx: tuple[float, float, float],
) -> tuple[dict[int, int], dict[int, int]]:
    nodes, edges = prediction_rows(graph)
    prediction = prediction_to_graph(nodes, edges)
    truth = geff_arrays_to_graph(dict(gt))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _evaluate(prediction, truth, "jaccard", scale_zyx, 7.0)

    ordered_pred_ids = [
        int(row["node_id"])
        for row in sorted(nodes, key=lambda row: (int(row["t"]), int(row["node_id"])))
    ]
    gt_nodes = gt["nodes"]
    ordered_gt_ids = [
        int(row[0])
        for row in sorted(
            zip(gt_nodes["node_id"], gt_nodes["t"], strict=True),
            key=lambda row: (int(row[1]), int(row[0])),
        )
    ]
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    matched = prediction.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    for pred_internal, gt_internal in matched.iter_rows():
        if gt_internal is None or int(gt_internal) == -1:
            continue
        pred_index = int(pred_internal)
        gt_index = int(gt_internal)
        if not 0 <= pred_index < len(ordered_pred_ids) or not 0 <= gt_index < len(ordered_gt_ids):
            raise Phase1DiagnosticError("official matcher returned an out-of-range internal node ID")
        pred_id = ordered_pred_ids[pred_index]
        gt_id = ordered_gt_ids[gt_index]
        if pred_id in pred_to_gt or gt_id in gt_to_pred:
            raise Phase1DiagnosticError("official node matching is not one-to-one")
        pred_to_gt[pred_id] = gt_id
        gt_to_pred[gt_id] = pred_id
    return pred_to_gt, gt_to_pred


def edge_accounting(
    graph: Any,
    gt_edges: set[tuple[int, int]],
    pred_to_gt: Mapping[int, int],
    gt_to_pred: Mapping[int, int],
) -> dict[str, int]:
    pred_edges = {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}
    gt_out: dict[int, list[int]] = defaultdict(list)
    gt_in: dict[int, list[int]] = defaultdict(list)
    for source, target in gt_edges:
        gt_out[source].append(target)
        gt_in[target].append(source)
    tp = sum(
        gt_to_pred.get(source) is not None
        and gt_to_pred.get(target) is not None
        and (gt_to_pred[source], gt_to_pred[target]) in pred_edges
        for source, target in gt_edges
    )
    fp = 0
    for source, target in pred_edges:
        gt_source = pred_to_gt.get(source)
        gt_target = pred_to_gt.get(target)
        if gt_source is not None and gt_target is not None and (gt_source, gt_target) in gt_edges:
            continue
        valid = (
            (gt_source is not None and bool(gt_out.get(gt_source)))
            or (gt_target is not None and bool(gt_in.get(gt_target)))
        )
        fp += int(valid)
    return {"edge_tp": int(tp), "edge_fp": fp, "edge_fn": len(gt_edges) - int(tp)}


def validate_baseline_contract(
    compact: Mapping[str, Any],
    scores: Mapping[str, Any],
) -> tuple[str, ...]:
    baseline = compact.get("baseline")
    scope = compact.get("scope")
    summary = scores.get("summary")
    selection = scores.get("selection")
    if not all(isinstance(value, Mapping) for value in (baseline, scope, summary, selection)):
        raise Phase1DiagnosticError("baseline artifacts have incomplete scope/summary records")
    if int(scope["videos"]) != EXPECTED_BASELINE["video_count"]:
        raise Phase1DiagnosticError("compact baseline is not the exact 57-video scope")
    if scope.get("division_enabled") is not False:
        raise Phase1DiagnosticError("compact baseline does not attest divisions OFF")
    for key in ("adj_edge_jaccard", "edge_jaccard", "node_recall"):
        if not math.isclose(float(baseline[key]), float(EXPECTED_BASELINE[key]), rel_tol=0.0, abs_tol=1e-15):
            raise Phase1DiagnosticError(f"compact baseline {key} changed")
        if not math.isclose(float(summary[key]), float(EXPECTED_BASELINE[key]), rel_tol=0.0, abs_tol=1e-15):
            raise Phase1DiagnosticError(f"score baseline {key} changed")
    for key in ("edge_tp", "edge_fp", "edge_fn"):
        if int(baseline[key]) != EXPECTED_BASELINE[key]:
            raise Phase1DiagnosticError(f"compact baseline {key} changed")
    sample_rows = scores.get("samples")
    identities = tuple(str(value) for value in selection.get("selected_video_ids", ()))
    if len(identities) != 57 or not isinstance(sample_rows, list) or len(sample_rows) != 57:
        raise Phase1DiagnosticError("score baseline does not bind exactly 57 videos")
    sample_totals = {
        key: int(sum(float(row[key]) for row in sample_rows))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    if sample_totals != {key: EXPECTED_BASELINE[key] for key in sample_totals}:
        raise Phase1DiagnosticError(f"baseline score totals changed: {sample_totals}")
    if float(summary["adj_edge_jaccard"]) != EXPECTED_BASELINE["adj_edge_jaccard"]:
        raise Phase1DiagnosticError("baseline score artifact is not the exact 0.8711885823490648 run")
    return identities


def analyse_video(
    *,
    identity: str,
    graph: Any,
    gt: Mapping[str, Any],
    scale_zyx: tuple[float, float, float],
    detections: Sequence[bh_model.FrameDet],
    expected_score: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    pred_to_gt, gt_to_pred = official_matches(graph, gt, scale_zyx)
    gt_edges = {(int(source), int(target)) for source, target in gt["edges"]}
    accounting = edge_accounting(graph, gt_edges, pred_to_gt, gt_to_pred)
    expected_accounting = {
        key: int(float(expected_score[key])) for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    if accounting != expected_accounting:
        raise Phase1DiagnosticError(
            f"official edge accounting mismatch for {identity}: observed={accounting} expected={expected_accounting}"
        )

    frame_by_t = {int(frame.frame_index): frame for frame in detections}
    if len(frame_by_t) != len(detections):
        raise Phase1DiagnosticError(f"duplicate raw detector frame in {identity}")
    raw_coord_by_id: dict[int, np.ndarray] = {}
    raw_t_by_id: dict[int, int] = {}
    for frame in detections:
        if len(frame.node_ids) > bh_graph.V45_GRAPH_CONFIG.max_nodes_per_frame:
            raise Phase1DiagnosticError(f"raw frame exceeds graph fail-closed cap in {identity}")
        for row, node_id_value in enumerate(frame.node_ids):
            node_id = int(node_id_value)
            if node_id in raw_coord_by_id:
                raise Phase1DiagnosticError(f"duplicate raw detector node ID {node_id} in {identity}")
            raw_coord_by_id[node_id] = np.asarray(frame.coords_zyx[row], dtype=np.float64)
            raw_t_by_id[node_id] = int(frame.frame_index)

    graph_t_by_id = {int(node_id): int(node.t) for node_id, node in graph.nodes.items()}
    pred_edges = {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}
    adjacent_out: dict[int, list[int]] = defaultdict(list)
    adjacent_parent: dict[int, int] = {}
    for source, target in pred_edges:
        if (
            source in raw_t_by_id
            and target in raw_t_by_id
            and raw_t_by_id[target] == raw_t_by_id[source] + 1
        ):
            adjacent_out[source].append(target)
            if target in adjacent_parent:
                raise Phase1DiagnosticError(f"multiple adjacent parents for node {target} in {identity}")
            adjacent_parent[target] = source

    gt_node_t = {
        int(node_id): int(t)
        for node_id, t in zip(gt["nodes"]["node_id"], gt["nodes"]["t"], strict=True)
    }
    events: list[dict[str, Any]] = []
    ineligible = Counter()
    for gt_source, gt_target in sorted(gt_edges):
        pred_source = gt_to_pred.get(gt_source)
        pred_target = gt_to_pred.get(gt_target)
        selected = (
            pred_source is not None
            and pred_target is not None
            and (pred_source, pred_target) in pred_edges
        )
        if pred_source is None or pred_target is None:
            ineligible["official_unmatched_endpoint"] += 1
            continue
        if pred_source not in raw_coord_by_id or pred_target not in raw_coord_by_id:
            ineligible["synthetic_or_pruned_raw_endpoint"] += 1
            continue
        source_t = raw_t_by_id[pred_source]
        target_t = raw_t_by_id[pred_target]
        if target_t != source_t + 1 or gt_node_t[gt_target] != gt_node_t[gt_source] + 1:
            ineligible["nonadjacent_endpoint"] += 1
            continue
        if graph_t_by_id.get(pred_source) != source_t or graph_t_by_id.get(pred_target) != target_t:
            raise Phase1DiagnosticError(f"raw/final node time mismatch in {identity}")
        target_frame = frame_by_t.get(target_t)
        if target_frame is None:
            raise Phase1DiagnosticError(f"missing raw target frame {target_t} in {identity}")
        rank, raw_distance_um = distance_rank(
            raw_coord_by_id[pred_source],
            pred_target,
            target_frame.node_ids,
            target_frame.coords_zyx,
            scale_zyx,
        )
        is_fn = not selected
        above_gate_fn = is_fn and raw_distance_um > bh_graph.V45_GRAPH_CONFIG.motion_relaxed_um + DISTANCE_EPSILON_UM
        parent_id = adjacent_parent.get(pred_source)
        parent_coord = raw_coord_by_id.get(parent_id) if parent_id is not None else None
        predicted = motion_predicted_coord(
            raw_coord_by_id[pred_source],
            parent_coord,
            bh_graph.V45_GRAPH_CONFIG.velocity_weight,
        )
        predicted_distance_um = physical_distance(
            predicted, raw_coord_by_id[pred_target], scale_zyx
        )
        events.append(
            {
                "identity": identity,
                "gt_source_id": gt_source,
                "gt_target_id": gt_target,
                "pred_source_id": pred_source,
                "pred_target_id": pred_target,
                "source_frame": source_t,
                "selected_edge": selected,
                "is_fn": is_fn,
                "raw_distance_um": raw_distance_um,
                "distance_rank": rank,
                "above_10um_fn": above_gate_fn,
                "motion_history_present": parent_id is not None,
                "motion_parent_id": parent_id,
                "motion_predicted_distance_um": predicted_distance_um,
                "source_unmatched_after_current_adjacent_assignment": not bool(
                    adjacent_out.get(pred_source)
                ),
            }
        )

    eligible_fn = [row for row in events if row["is_fn"]]
    above_gate_fn = [row for row in eligible_fn if row["above_10um_fn"]]
    row = {
        "identity": identity,
        **accounting,
        "eligible_gt_edges": len(events),
        "eligible_fn_edges": len(eligible_fn),
        "in_gate_fn_edges": len(eligible_fn) - len(above_gate_fn),
        "above_10um_fn_edges": len(above_gate_fn),
        "ineligible_gt_edges": int(sum(ineligible.values())),
        "ineligible_reasons": dict(sorted(ineligible.items())),
        "raw_detection_nodes": len(raw_coord_by_id),
    }
    return row, events, accounting


def e2_sensitivity(under_gate: int, denominator: int) -> list[dict[str, Any]]:
    observed = fraction(under_gate, denominator)
    thresholds = (0.25, 1.0 / 3.0, 0.50, 2.0 / 3.0, 0.75, 0.90)
    return [
        {
            "large_fraction_threshold": threshold,
            "pass": observed is not None and observed >= threshold,
        }
        for threshold in thresholds
    ]


def build_report(
    *,
    events: Sequence[Mapping[str, Any]],
    per_video: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    eligible = list(events)
    fn = [row for row in eligible if bool(row["is_fn"])]
    in_gate_fn = [row for row in fn if not bool(row["above_10um_fn"])]
    above_gate_fn = [row for row in fn if bool(row["above_10um_fn"])]
    official_gt_edges = int(
        sum(int(row["edge_tp"]) + int(row["edge_fn"]) for row in per_video)
    )
    candidate_absent = official_gt_edges - len(eligible)
    if candidate_absent < 0:
        raise Phase1DiagnosticError("eligible edge count exceeds official GT edge count")
    all_gt_rank = rank_summary(
        [int(row["distance_rank"]) for row in eligible],
        candidate_absent=candidate_absent,
    )
    eligible_rank = rank_summary([int(row["distance_rank"]) for row in eligible])
    above_rank = rank_summary([int(row["distance_rank"]) for row in above_gate_fn])
    e1_pass = (
        all_gt_rank["rank_le_10_fraction"] is not None
        and above_rank["rank_le_10_fraction"] is not None
        and all_gt_rank["rank_le_10_fraction"] >= E1_PASS_FRACTION
        and above_rank["rank_le_10_fraction"] >= E1_PASS_FRACTION
    )

    predicted_under_10 = [
        row
        for row in above_gate_fn
        if float(row["motion_predicted_distance_um"])
        <= bh_graph.V45_GRAPH_CONFIG.motion_relaxed_um + DISTANCE_EPSILON_UM
    ]
    actionable = [
        row
        for row in above_gate_fn
        if bool(row["source_unmatched_after_current_adjacent_assignment"])
    ]
    actionable_under_10 = [
        row
        for row in actionable
        if float(row["motion_predicted_distance_um"])
        <= bh_graph.V45_GRAPH_CONFIG.motion_relaxed_um + DISTANCE_EPSILON_UM
    ]
    observed_fraction = fraction(len(predicted_under_10), len(above_gate_fn))
    e2_pass = observed_fraction is not None and observed_fraction >= E2_PRIMARY_LARGE_FRACTION

    totals = {
        key: int(sum(int(row[key]) for row in per_video))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    return {
        "schema": SCHEMA,
        "status": "completed_read_only_cpu_diagnostic",
        "provenance": dict(provenance),
        "execution": {
            "cpu_only": True,
            "gpu_used": False,
            "detector_forward_calls": 0,
            "training": False,
            "kaggle_action": False,
            "production_graph_changed": False,
            "elapsed_seconds": elapsed_seconds,
        },
        "cohort": {
            "video_count": len(per_video),
            "official_edge_totals": totals,
            "official_gt_temporal_edges": official_gt_edges,
            "eligible_gt_temporal_edges": len(eligible),
            "ineligible_gt_temporal_edges": int(
                sum(int(row["ineligible_gt_edges"]) for row in per_video)
            ),
            "eligible_fn_edges": len(fn),
            "in_gate_fn_edges": len(in_gate_fn),
            "above_10um_fn_edges": len(above_gate_fn),
            "definition": (
                "Official GT edges whose two endpoints are matched by the patched scorer to "
                "original raw FrameDet nodes in adjacent frames. FN membership comes from the "
                "exact baseline topology; 10um cohorts use pre-smoothing raw FrameDet geometry."
            ),
        },
        "e1_distance_rank": {
            "decision_rule": (
                "PASS only if rank<=10 fraction is >=0.90 for both all official GT edges "
                "and the newly derived above-10um baseline FN cohort."
            ),
            "threshold": E1_PASS_FRACTION,
            "all_official_gt_edges": all_gt_rank,
            "eligible_gt_edges_secondary": eligible_rank,
            "eligible_fn_edges": rank_summary([int(row["distance_rank"]) for row in fn]),
            "in_gate_fn_edges": rank_summary([int(row["distance_rank"]) for row in in_gate_fn]),
            "above_10um_fn_edges": above_rank,
            "pass": e1_pass,
            "interpretation": (
                "PASS: replace the metric-only gate with a reviewed top-k standalone change."
                if e1_pass
                else "FAIL: the required joint >=90% rank<=10 gate is not met; do not implement top-k yet."
            ),
        },
        "e2_motion_recentring": {
            "prediction_contract": (
                "Exact bh_graph formula: predicted=source when no incoming adjacent edge, else "
                "source+0.5*(source-parent), on raw FrameDet coordinates."
            ),
            "primary_large_fraction_threshold": E2_PRIMARY_LARGE_FRACTION,
            "above_10um_fn_denominator": len(above_gate_fn),
            "motion_history_present": sum(
                bool(row["motion_history_present"]) for row in above_gate_fn
            ),
            "predicted_distance_le_10um": len(predicted_under_10),
            "predicted_distance_le_10um_fraction": observed_fraction,
            "predicted_distance_um": quantiles(
                [float(row["motion_predicted_distance_um"]) for row in above_gate_fn]
            ),
            "distance_threshold_sensitivity": {
                str(radius): {
                    "count": sum(
                        float(row["motion_predicted_distance_um"])
                        <= radius + DISTANCE_EPSILON_UM
                        for row in above_gate_fn
                    ),
                    "fraction": fraction(
                        sum(
                            float(row["motion_predicted_distance_um"])
                            <= radius + DISTANCE_EPSILON_UM
                            for row in above_gate_fn
                        ),
                        len(above_gate_fn),
                    ),
                }
                for radius in (6.0, 8.0, 10.0, 12.0, 14.0)
            },
            "large_fraction_threshold_sensitivity": e2_sensitivity(
                len(predicted_under_10), len(above_gate_fn)
            ),
            "proposed_unmatched_source_pass": {
                "denominator": len(actionable),
                "share_of_all_above_10um_fn": fraction(len(actionable), len(above_gate_fn)),
                "predicted_distance_le_10um": len(actionable_under_10),
                "within_actionable_fraction": fraction(
                    len(actionable_under_10), len(actionable)
                ),
                "recoverable_share_of_all_above_10um_fn": fraction(
                    len(actionable_under_10), len(above_gate_fn)
                ),
            },
            "pass": e2_pass,
            "interpretation": (
                "PASS at the explicit primary >=50% convention for 'large fraction'; review an unmatched-source-only second pass."
                if e2_pass
                else "FAIL at the explicit primary >=50% convention for 'large fraction'; do not implement re-gating."
            ),
        },
        "per_video": list(per_video),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    cohort = report["cohort"]
    e1 = report["e1_distance_rank"]
    e2 = report["e2_motion_recentring"]
    overall = e1["all_official_gt_edges"]
    eligible = e1["eligible_gt_edges_secondary"]
    above = e1["above_10um_fn_edges"]
    actionable = e2["proposed_unmatched_source_pass"]
    return "\n".join(
        [
            "# Phase 1 E1/E2 — exact CV57 distance and motion gate",
            "",
            "## Verified scope",
            "",
            f"- Exact divisions-OFF baseline: adjusted edge Jaccard `{report['provenance']['baseline']['adj_edge_jaccard']:.16f}`.",
            f"- Official edge totals: TP `{cohort['official_edge_totals']['edge_tp']}`, FP `{cohort['official_edge_totals']['edge_fp']}`, FN `{cohort['official_edge_totals']['edge_fn']}`.",
            f"- Newly derived raw-geometry cohort: `{cohort['eligible_fn_edges']}` eligible FNs = `{cohort['in_gate_fn_edges']}` at <=10 um + `{cohort['above_10um_fn_edges']}` at >10 um.",
            f"- All official GT temporal edges: `{cohort['official_gt_temporal_edges']}`; rank-observable eligible edges: `{cohort['eligible_gt_temporal_edges']}`; candidate-absent/unrankable: `{cohort['ineligible_gt_temporal_edges']}`.",
            "",
            "## E1 — distance rank",
            "",
            f"**{'PASS' if e1['pass'] else 'FAIL'}** against the stated joint >=90% rank<=10 rule over every official GT edge.",
            "",
            f"- All official GT edges: rank<=5 `{overall['rank_le_5']}/{overall['denominator']}` ({overall['rank_le_5_fraction']:.4%}); rank<=10 `{overall['rank_le_10']}/{overall['denominator']}` ({overall['rank_le_10_fraction']:.4%}); candidate-absent/unrankable `{overall['candidate_absent_or_unrankable']}`.",
            f"- Eligible-only secondary distribution: rank<=5 `{eligible['rank_le_5']}/{eligible['denominator']}` ({eligible['rank_le_5_fraction']:.4%}); rank<=10 `{eligible['rank_le_10']}/{eligible['denominator']}` ({eligible['rank_le_10_fraction']:.4%}).",
            f"- Above-10 um baseline FNs: rank<=5 `{above['rank_le_5']}/{above['denominator']}` ({above['rank_le_5_fraction']:.4%}); rank<=10 `{above['rank_le_10']}/{above['denominator']}` ({above['rank_le_10_fraction']:.4%}).",
            f"- Decision: {e1['interpretation']}",
            "",
            "## E2 — motion-predicted re-centring",
            "",
            f"**{'PASS' if e2['pass'] else 'FAIL'}** under the explicit primary definition of large fraction = >=50%.",
            "",
            f"- Predicted distance <=10 um: `{e2['predicted_distance_le_10um']}/{e2['above_10um_fn_denominator']}` ({e2['predicted_distance_le_10um_fraction']:.4%}).",
            f"- Motion history present: `{e2['motion_history_present']}/{e2['above_10um_fn_denominator']}`.",
            f"- Actually reachable by an unmatched-source-only pass: `{actionable['denominator']}/{e2['above_10um_fn_denominator']}` sources; <=10 um after prediction `{actionable['predicted_distance_le_10um']}/{actionable['denominator']}` ({(actionable['within_actionable_fraction'] or 0.0):.4%}), equal to `{(actionable['recoverable_share_of_all_above_10um_fn'] or 0.0):.4%}` of the full above-gate cohort.",
            f"- Decision: {e2['interpretation']}",
            "",
            "## Interpretation boundary",
            "",
            "These are diagnostic gates only. No production top-k gate or motion re-gating was implemented. FN membership uses patched official matching on the exact baseline fragments; distances/ranks use raw cached detector coordinates, not final smoothed CSV coordinates.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    compact = load_object(args.compact_baseline, "compact baseline")
    scores = load_object(args.baseline_scores, "post-patch baseline scores")
    identities = validate_baseline_contract(compact, scores)
    score_by_identity = {
        str(row["identity"]): row for row in scores["samples"]
    }
    if set(score_by_identity) != set(identities):
        raise Phase1DiagnosticError("baseline per-video scores do not match its selected identities")

    split = load_object(args.split, "frozen split")
    registry = load_object(args.registry, "dataset registry")
    archive_ids = tuple(
        str(row["video_id"])
        for row in registry.get("records", ())
        if isinstance(row, Mapping) and row.get("split") == "train"
    )
    selected = score_bh_cv57.exact_selection(
        split, split_path=args.split, archive_video_ids=archive_ids
    )
    if selected != identities:
        raise Phase1DiagnosticError("exact fold-0-minus-terminal-exclusion selection changed")
    scorer = score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)

    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(args.fragments)), selected
    )
    fragment_by_identity = dict(manifest.ordered)
    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if set(cache_paths) != set(selected) or set(meta_paths) != set(selected):
        raise Phase1DiagnosticError("cache/meta files do not exactly cover the CV57 selection")

    per_video: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    provenance_fragments: list[dict[str, Any]] = []
    provenance_caches: list[dict[str, Any]] = []
    with ZipStore(args.archive, mode="r") as store:
        for index, identity in enumerate(selected, start=1):
            _, dataset, _ = scoreall.parse_identity(identity)
            fragment = fragment_by_identity[identity]
            expected_fragment = score_by_identity[identity].get("source", {})
            fragment_sha = sha256_file(fragment)
            if fragment_sha != expected_fragment.get("sha256"):
                raise Phase1DiagnosticError(f"baseline fragment hash changed for {identity}")
            graph = scoreall._fragment_graph(fragment, identity)
            detections = load_detections(cache_paths[identity], meta_paths[identity])
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            if list(scale) != list(score_by_identity[identity]["scale_zyx"]):
                raise Phase1DiagnosticError(f"physical scale changed for {identity}")
            video_row, video_events, _ = analyse_video(
                identity=identity,
                graph=graph,
                gt=gt,
                scale_zyx=scale,
                detections=detections,
                expected_score=score_by_identity[identity],
            )
            per_video.append(video_row)
            events.extend(video_events)
            provenance_fragments.append(
                {
                    "identity": identity,
                    "path": str(fragment),
                    "sha256": fragment_sha,
                }
            )
            provenance_caches.append(
                {
                    "identity": identity,
                    "path": str(cache_paths[identity]),
                    "sha256": sha256_file(cache_paths[identity]),
                    "metadata_path": str(meta_paths[identity]),
                    "metadata_sha256": sha256_file(meta_paths[identity]),
                }
            )
            print(
                f"E1/E2 {index:02d}/{len(selected)} {identity} "
                f"eligible={video_row['eligible_gt_edges']} "
                f"fn={video_row['eligible_fn_edges']} above10={video_row['above_10um_fn_edges']}",
                flush=True,
            )

    aggregate = {
        key: int(sum(int(row[key]) for row in per_video))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    expected_totals = {
        key: int(EXPECTED_BASELINE[key]) for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    if aggregate != expected_totals:
        raise Phase1DiagnosticError(
            f"CV57 aggregate changed: observed={aggregate} expected={expected_totals}"
        )
    provenance = {
        "baseline": {
            **EXPECTED_BASELINE,
            "division_enabled": False,
            "compact_path": str(args.compact_baseline),
            "compact_sha256": sha256_file(args.compact_baseline),
            "scores_path": str(args.baseline_scores),
            "scores_sha256": sha256_file(args.baseline_scores),
        },
        "selection": {
            "split_path": str(args.split),
            "split_sha256": sha256_file(args.split),
            "video_ids": list(selected),
        },
        "competition_archive": {
            "path": str(args.archive),
            "bytes": args.archive.stat().st_size,
            "sha256_attested_by_baseline_scores": scores["frozen_scorer"]["archive"]["sha256"],
            "rehash_performed": False,
        },
        "scorer": scorer,
        "graph_constants": {
            "radius_um": bh_graph.V45_GRAPH_CONFIG.motion_relaxed_um,
            "velocity_weight": bh_graph.V45_GRAPH_CONFIG.velocity_weight,
            "scale_zyx": list(bh_graph.V45_GRAPH_CONFIG.scale_zyx),
            "graph_source_path": str(ROOT / "src/celltrack_alpha/bh_graph.py"),
            "graph_source_sha256": sha256_file(ROOT / "src/celltrack_alpha/bh_graph.py"),
            "edge_source_path": str(ROOT / "src/celltrack_alpha/bh_edge.py"),
            "edge_source_sha256": sha256_file(ROOT / "src/celltrack_alpha/bh_edge.py"),
        },
        "fragments": provenance_fragments,
        "caches": provenance_caches,
    }
    return build_report(
        events=events,
        per_video=per_video,
        provenance=provenance,
        elapsed_seconds=time.perf_counter() - started,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "reports/g02/grouped_3fold_audit20_v1.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "reports/g02/dataset_registry_v1.json",
    )
    parser.add_argument(
        "--fragments",
        type=Path,
        default=ROOT / "kaggle_kernel/bhcv57out_v3/fragments",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "kaggle_kernel/bhcv57out_v3/cache",
    )
    parser.add_argument(
        "--baseline-scores",
        type=Path,
        default=(
            ROOT
            / "_ARCHIVE/conflict_residual/reports/g02/"
            "run_bh_d4_conflict_edge_f0_holdout57_v1/scores_postpatch.json"
        ),
    )
    parser.add_argument(
        "--compact-baseline",
        type=Path,
        default=ROOT / "reports/g02/graph_cost_sweeps_holdout57_v1.json",
    )
    parser.add_argument(
        "--scorer-parity",
        type=Path,
        default=score_bh_cv57.DEFAULT_SCORER_PARITY,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports/g02/phase1_distance_motion_gate_holdout57_v1.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "reports/g02/phase1_distance_motion_gate_holdout57_v1.md",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_json.exists() or args.output_md.exists():
        raise Phase1DiagnosticError("refusing to overwrite an existing versioned report")
    report = run(args)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "cohort": report["cohort"],
                "e1": report["e1_distance_rank"],
                "e2": report["e2_motion_recentring"],
                "outputs": [str(args.output_json), str(args.output_md)],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
