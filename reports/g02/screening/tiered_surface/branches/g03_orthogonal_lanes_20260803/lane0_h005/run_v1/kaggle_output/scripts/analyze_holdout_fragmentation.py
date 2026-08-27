#!/usr/bin/env python3
"""Audit association fragmentation on the frozen 57-video fold-0 run.

The diagnostic intentionally excludes ``train:6bba_f4ae811c``.  Its fragment
finished while the run was being stopped, but it is neither assembled nor
scored here.  Candidate-score tensors were not persisted by ``v45feval``;
therefore this script separates exact topology/geometry observations from
inferences about candidate ranking.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr
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

import scoreall  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionGraph,
    read_submission,
)
from celltrack_alpha.official_eval import (  # noqa: E402
    geff_arrays_to_graph,
    official_metric_row,
    official_summary,
    prediction_to_graph,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from tracking_cellmot.metrics import _evaluate  # noqa: E402


EXCLUDED_IDENTITY = "train:6bba_f4ae811c"
KD_RADIUS_UM = 10.0
KD_NEIGHBOR_CAP = 8
PREFIX_EXPECTED = {"n": 52, "edge_tp": 29676, "edge_fp": 2660, "edge_fn": 1810}


class FragmentationAuditError(RuntimeError):
    """The frozen diagnostic contract was not satisfied."""


class DisjointSet:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {int(value): int(value) for value in values}
        self.size = {int(value): 1 for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def component_sizes(self) -> list[int]:
        counts = Counter(self.find(value) for value in self.parent)
        return sorted(counts.values())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FragmentationAuditError(f"JSON root is not an object: {path}")
    return value


def quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "p90": None, "max": None, "mean": None}
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def correlation(x: Sequence[float], y: Sequence[float]) -> dict[str, float | int]:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return {"n": int(left.size), "pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_rho": float("nan"), "spearman_p": float("nan")}
    pearson = pearsonr(left, right)
    spearman = spearmanr(left, right)
    return {
        "n": int(left.size),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def fragment_path(parts_root: Path, identity: str) -> Path:
    dataset = identity.split(":", 1)[1]
    path = parts_root / identity.replace(":", "__", 1) / "frags" / f"train__{dataset}.csv"
    if not path.is_file():
        raise FragmentationAuditError(f"missing frozen fragment: {path}")
    return path


def report_path(parts_root: Path, identity: str) -> Path:
    path = parts_root / identity.replace(":", "__", 1) / "report.json"
    if not path.is_file():
        raise FragmentationAuditError(f"missing frozen per-video report: {path}")
    return path


def load_fragment(parts_root: Path, identity: str) -> tuple[Path, SubmissionGraph]:
    path = fragment_path(parts_root, identity)
    graphs = read_submission(path)
    dataset = identity.split(":", 1)[1]
    if set(graphs) != {dataset}:
        raise FragmentationAuditError(f"fragment identity mismatch: {path}")
    return path, graphs[dataset]


def prediction_rows(graph: SubmissionGraph) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {"node_id": node.node_id, "t": node.t, "z": node.z, "y": node.y, "x": node.x}
        for node in graph.nodes.values()
    ]
    edges = [
        {"source_id": edge.source_id, "target_id": edge.target_id}
        for edge in graph.edges
    ]
    return nodes, edges


def score_57(
    store: ZipStore,
    identities: Sequence[str],
    parts_root: Path,
    prefix_scores_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    prefix = load_json(prefix_scores_path)
    prefix_ids = tuple(str(value) for value in prefix.get("selected_video_ids", ()))
    if len(prefix_ids) != PREFIX_EXPECTED["n"] or not set(prefix_ids).issubset(identities):
        raise FragmentationAuditError("prefix-52 score scope does not match the 57-video scope")
    samples = [dict(row) for row in prefix["samples"]]
    by_identity = {str(row["identity"]): row for row in samples}
    prefix_totals = {
        key: int(sum(float(row[key]) for row in samples))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    if prefix_totals != {key: PREFIX_EXPECTED[key] for key in prefix_totals}:
        raise FragmentationAuditError(f"prefix edge totals changed: {prefix_totals}")

    for identity in identities:
        if identity in by_identity:
            continue
        dataset = identity.split(":", 1)[1]
        path, graph = load_fragment(parts_root, identity)
        nodes, edges = prediction_rows(graph)
        gt = read_geff_arrays(store, "train", dataset)
        group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
        scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
        row = official_metric_row(nodes, edges, gt, scale)
        row.update(
            {
                "identity": identity,
                "dataset": dataset,
                "full_video": True,
                "cropped": False,
                "ground_truth_nodes": len(gt["nodes"]["node_id"]),
                "ground_truth_edges": len(gt["edges"]),
                "prediction_node_rows": len(nodes),
                "prediction_edge_rows": len(edges),
                "scale_zyx": list(scale),
                "source": {"mode": "fragment", "path": str(path), "sha256": sha256_file(path)},
            }
        )
        denom = row["division_tp"] + row["division_fp"] + row["division_fn"]
        row["division_jaccard"] = row["division_tp"] / denom if denom else float("nan")
        samples.append(row)
        by_identity[identity] = row

    if set(by_identity) != set(identities) or len(samples) != 57:
        raise FragmentationAuditError("complete-at-57 scoring did not cover exactly 57 identities")
    ordered = [by_identity[identity] for identity in identities]
    result = {
        "schema": "celltrack-alpha.frozen-holdout-complete-at-57.v1",
        "status": "complete_at_57",
        "selected_video_ids": list(identities),
        "excluded_identity": EXCLUDED_IDENTITY,
        "caveat": (
            "The excluded fragment finished while cancellation was being delivered, but it was not "
            "assembled or scored. This 57-video scope is terminal by explicit user direction."
        ),
        "summary": official_summary(ordered),
        "samples": ordered,
        "prefix_52": {
            "path": str(prefix_scores_path),
            "sha256": sha256_file(prefix_scores_path),
            "edge_totals": prefix_totals,
        },
    }
    return result, by_identity


def graph_component_sizes(nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> tuple[list[int], dict[int, int]]:
    values = tuple(int(value) for value in nodes)
    dsu = DisjointSet(values)
    for source, target in edges:
        dsu.union(int(source), int(target))
    component_by_node = {value: dsu.find(value) for value in values}
    return dsu.component_sizes(), component_by_node


def gate_observation(
    source_id: int,
    target_id: int,
    graph: SubmissionGraph,
    by_frame: Mapping[int, Sequence[int]],
    scale: np.ndarray,
) -> dict[str, Any]:
    source = graph.nodes[source_id]
    target = graph.nodes[target_id]
    source_xyz = np.asarray(source.coord_zyx, dtype=np.float64) * scale
    target_xyz = np.asarray(target.coord_zyx, dtype=np.float64) * scale
    distance = float(np.linalg.norm(target_xyz - source_xyz))
    candidates: list[tuple[float, int]] = []
    for candidate_id in by_frame.get(source.t + 1, ()): 
        candidate = graph.nodes[candidate_id]
        candidate_xyz = np.asarray(candidate.coord_zyx, dtype=np.float64) * scale
        candidate_distance = float(np.linalg.norm(candidate_xyz - source_xyz))
        if candidate_distance <= KD_RADIUS_UM + 1.0e-12:
            candidates.append((candidate_distance, candidate_id))
    candidates.sort(key=lambda item: (item[0], item[1]))
    rank = next((index + 1 for index, (_, node_id) in enumerate(candidates) if node_id == target_id), None)
    return {
        "distance_um": distance,
        "final_node_radius_neighbor_count": len(candidates),
        "final_node_distance_rank": rank,
    }


def spatial_thirds(
    node: Mapping[str, Any], image_shape_tzyx: Sequence[int]
) -> dict[str, int]:
    return {
        axis: min(
            2,
            int(
                3.0
                * float(node[axis])
                / max(1.0, float(image_shape_tzyx[index]))
            ),
        )
        for axis, index in (("z", 1), ("y", 2), ("x", 3))
    }


def classify_break(
    *,
    gt_source: int,
    gt_target: int,
    pred_source: int | None,
    pred_target: int | None,
    graph: SubmissionGraph,
    pred_edges: set[tuple[int, int]],
    pred_out: Mapping[int, Sequence[int]],
    pred_in: Mapping[int, Sequence[int]],
    gt_out: Mapping[int, Sequence[int]],
    gt_to_pred: Mapping[int, int],
    by_frame: Mapping[int, Sequence[int]],
    scale: np.ndarray,
    input_node_count: int,
) -> tuple[str, dict[str, Any]]:
    if pred_source is None or pred_target is None:
        return "unmatched_endpoint", {
            "matched_source": pred_source is not None,
            "matched_target": pred_target is not None,
        }
    observation = gate_observation(pred_source, pred_target, graph, by_frame, scale)
    source = graph.nodes[pred_source]
    target = graph.nodes[pred_target]
    observation.update(
        {
            "matched_source": True,
            "matched_target": True,
            "pred_source_id": pred_source,
            "pred_target_id": pred_target,
            "pred_source_synthetic": pred_source >= input_node_count,
            "pred_target_synthetic": pred_target >= input_node_count,
            "pred_source_out_targets": list(sorted(pred_out.get(pred_source, ()))),
            "pred_target_in_sources": list(sorted(pred_in.get(pred_target, ()))),
        }
    )
    if source.t + 1 != target.t:
        return "nonadjacent_matched_endpoints", observation
    if pred_source >= input_node_count or pred_target >= input_node_count:
        return "gap_synthetic_endpoint", observation
    if float(observation["distance_um"]) > KD_RADIUS_UM + 1.0e-12:
        return "final_pair_geometry_above_gate_radius", observation
    rank = observation["final_node_distance_rank"]
    if rank is None or int(rank) > KD_NEIGHBOR_CAP:
        return "final_pair_rank_above_cap8", observation

    other_gt_targets = [target_id for target_id in gt_out.get(gt_source, ()) if target_id != gt_target]
    other_daughter_selected = any(
        (pred_source, other_pred) in pred_edges
        for other_target in other_gt_targets
        if (other_pred := gt_to_pred.get(other_target)) is not None
    )
    if other_daughter_selected:
        return "solver_outdegree_division", observation

    wrong_out = [target_id for target_id in pred_out.get(pred_source, ()) if target_id != pred_target]
    wrong_in = [source_id for source_id in pred_in.get(pred_target, ()) if source_id != pred_source]
    if wrong_out or wrong_in:
        observation["wrong_outgoing_competitors"] = sorted(wrong_out)
        observation["wrong_incoming_competitors"] = sorted(wrong_in)
        return "occupied_endpoint_conflict", observation
    return "no_observed_endpoint_conflict", observation


def analyse_video(
    *,
    identity: str,
    graph: SubmissionGraph,
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
    image_shape_tzyx: Sequence[int],
    input_node_count: int,
    peak_component_matrix_dim: int,
    score_row: Mapping[str, Any],
    prefix_ids: set[str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    node_rows, edge_rows = prediction_rows(graph)
    prediction = prediction_to_graph(node_rows, edge_rows)
    truth = geff_arrays_to_graph(gt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _evaluate(prediction, truth, "jaccard", scale_zyx, 7.0)

    ordered_original_ids = [
        int(row["node_id"]) for row in sorted(node_rows, key=lambda row: (int(row["t"]), int(row["node_id"])))
    ]
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    matched = prediction.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    for pred_internal, gt_internal in matched.iter_rows():
        if gt_internal is None or int(gt_internal) == -1:
            continue
        pred_original = ordered_original_ids[int(pred_internal)]
        pred_to_gt[pred_original] = int(gt_internal)
        if int(gt_internal) in gt_to_pred:
            raise FragmentationAuditError(f"non-unique official match in {identity}")
        gt_to_pred[int(gt_internal)] = pred_original

    truth_attrs = truth.node_attrs().select("node_id", "t", "z", "y", "x")
    gt_nodes = {
        int(node_id): {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for node_id, t, z, y, x in truth_attrs.iter_rows()
    }
    gt_edges = {
        (int(source), int(target))
        for source, target in truth.edge_attrs().select("source_id", "target_id").iter_rows()
    }
    pred_edges = {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}
    gt_out: dict[int, list[int]] = defaultdict(list)
    gt_in: dict[int, list[int]] = defaultdict(list)
    pred_out: dict[int, list[int]] = defaultdict(list)
    pred_in: dict[int, list[int]] = defaultdict(list)
    for source, target in gt_edges:
        gt_out[source].append(target)
        gt_in[target].append(source)
    for source, target in pred_edges:
        pred_out[source].append(target)
        pred_in[target].append(source)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node_id, node in graph.nodes.items():
        by_frame[node.t].append(node_id)
    for values in by_frame.values():
        values.sort()

    pred_sizes, pred_component = graph_component_sizes(graph.nodes, pred_edges)
    gt_sizes, gt_component = graph_component_sizes(gt_nodes, gt_edges)
    gt_component_nodes: dict[int, list[int]] = defaultdict(list)
    for node_id, component in gt_component.items():
        gt_component_nodes[component].append(node_id)
    fragmented_gt_components = 0
    pred_segments_per_gt_component: list[int] = []
    for nodes in gt_component_nodes.values():
        components = {
            pred_component[pred_id]
            for gt_id in nodes
            if (pred_id := gt_to_pred.get(gt_id)) is not None
        }
        pred_segments_per_gt_component.append(len(components))
        if len(components) > 1:
            fragmented_gt_components += 1

    scale = np.asarray(scale_zyx, dtype=np.float64)
    breaks: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    tp = 0
    for gt_source, gt_target in sorted(gt_edges):
        pred_source = gt_to_pred.get(gt_source)
        pred_target = gt_to_pred.get(gt_target)
        selected = (
            pred_source is not None
            and pred_target is not None
            and (pred_source, pred_target) in pred_edges
        )
        if selected:
            tp += 1
            observation = gate_observation(pred_source, pred_target, graph, by_frame, scale)
            source_meta = gt_nodes[gt_source]
            event_rows.append(
                {
                    "kind": "tp",
                    "identity": identity,
                    "prefix_52": identity in prefix_ids,
                    "source_frame": int(source_meta["t"]),
                    "source_frame_decile": min(9, int(source_meta["t"]) // 10),
                    "source_spatial_thirds_zyx": spatial_thirds(
                        source_meta, image_shape_tzyx
                    ),
                    **observation,
                }
            )
            continue
        cause, observation = classify_break(
            gt_source=gt_source,
            gt_target=gt_target,
            pred_source=pred_source,
            pred_target=pred_target,
            graph=graph,
            pred_edges=pred_edges,
            pred_out=pred_out,
            pred_in=pred_in,
            gt_out=gt_out,
            gt_to_pred=gt_to_pred,
            by_frame=by_frame,
            scale=scale,
            input_node_count=input_node_count,
        )
        source_meta = gt_nodes[gt_source]
        record = {
            "identity": identity,
            "prefix_52": identity in prefix_ids,
            "gt_source_id": gt_source,
            "gt_target_id": gt_target,
            "source_frame": int(source_meta["t"]),
            "source_frame_decile": min(9, int(source_meta["t"]) // 10),
            "source_spatial_thirds_zyx": spatial_thirds(
                source_meta, image_shape_tzyx
            ),
            "cause": cause,
            **observation,
        }
        breaks.append(record)
        event_rows.append({"kind": "fn", **record})

    direct_fp = Counter()
    linked_fp = Counter()
    fn_by_gt_edge = {(int(row["gt_source_id"]), int(row["gt_target_id"])): row for row in breaks}
    fp_count = 0
    fp_rows: list[dict[str, Any]] = []
    for source, target in sorted(pred_edges):
        gt_source = pred_to_gt.get(source)
        gt_target = pred_to_gt.get(target)
        is_tp = gt_source is not None and gt_target is not None and (gt_source, gt_target) in gt_edges
        valid = (
            (gt_source is not None and bool(gt_out.get(gt_source)))
            or (gt_target is not None and bool(gt_in.get(gt_target)))
        )
        if is_tp or not valid:
            continue
        fp_count += 1
        if gt_source is not None and gt_target is not None:
            direct_class = "wrong_matched_transition"
        elif gt_source is not None:
            direct_class = "matched_source_to_unmatched_target"
        elif gt_target is not None:
            direct_class = "unmatched_source_to_matched_target"
        else:
            direct_class = "unexpected_official_validity"
        direct_fp[direct_class] += 1
        linked = []
        if gt_source is not None:
            linked.extend(
                fn_by_gt_edge[pair]["cause"]
                for pair in ((gt_source, value) for value in gt_out.get(gt_source, ()))
                if pair in fn_by_gt_edge
            )
        if gt_target is not None:
            linked.extend(
                fn_by_gt_edge[pair]["cause"]
                for pair in ((value, gt_target) for value in gt_in.get(gt_target, ()))
                if pair in fn_by_gt_edge
            )
        unique_linked = sorted(set(linked))
        if not unique_linked:
            linked_class = "unpaired_extra_edge"
        elif len(unique_linked) == 1:
            linked_class = unique_linked[0]
        else:
            linked_class = "mixed_linked_break_causes"
        linked_fp[linked_class] += 1
        fp_rows.append(
            {
                "identity": identity,
                "source_id": source,
                "target_id": target,
                "direct_class": direct_class,
                "linked_break_class": linked_class,
            }
        )

    expected = {
        "edge_tp": int(float(score_row["edge_tp"])),
        "edge_fn": int(float(score_row["edge_fn"])),
        "edge_fp": int(float(score_row["edge_fp"])),
    }
    observed = {"edge_tp": tp, "edge_fn": len(breaks), "edge_fp": fp_count}
    if observed != expected:
        raise FragmentationAuditError(
            f"edge accounting mismatch for {identity}: observed={observed} expected={expected}"
        )

    row = {
        "identity": identity,
        "prefix_52": identity in prefix_ids,
        **observed,
        "edge_jaccard": float(score_row["edge_jaccard"]),
        "adj_edge_jaccard": float(score_row["adj_edge_jaccard"]),
        "total_node_ratio": float(score_row["total_node_ratio"]),
        "peak_component_matrix_dim": peak_component_matrix_dim,
        "input_node_count": input_node_count,
        "output_node_count": len(graph.nodes),
        "gt_node_count": len(gt_nodes),
        "prediction_component_count": len(pred_sizes),
        "prediction_component_mean_nodes": float(np.mean(pred_sizes)),
        "prediction_component_median_nodes": float(np.median(pred_sizes)),
        "gt_component_count": len(gt_sizes),
        "gt_component_mean_nodes": float(np.mean(gt_sizes)),
        "gt_component_median_nodes": float(np.median(gt_sizes)),
        "gt_components_fragmented": fragmented_gt_components,
        "gt_component_fragmentation_fraction": fragmented_gt_components / len(gt_sizes),
        "prediction_segments_per_gt_component": quantiles(pred_segments_per_gt_component),
        "fn_causes": dict(sorted(Counter(row["cause"] for row in breaks).items())),
        "fp_direct_classes": dict(sorted(direct_fp.items())),
        "fp_linked_break_classes_inferred": dict(sorted(linked_fp.items())),
    }
    return row, breaks, fp_rows, event_rows


def aggregate_scope(
    per_video: Sequence[Mapping[str, Any]],
    breaks: Sequence[Mapping[str, Any]],
    fp_rows: Sequence[Mapping[str, Any]],
    *,
    prefix_only: bool,
) -> dict[str, Any]:
    videos = [row for row in per_video if not prefix_only or bool(row["prefix_52"])]
    scoped_breaks = [row for row in breaks if not prefix_only or bool(row["prefix_52"])]
    identities = {str(row["identity"]) for row in videos}
    scoped_fp = [row for row in fp_rows if str(row["identity"]) in identities]
    fn_causes = Counter(str(row["cause"]) for row in scoped_breaks)
    fp_direct = Counter(str(row["direct_class"]) for row in scoped_fp)
    fp_linked = Counter(str(row["linked_break_class"]) for row in scoped_fp)
    frame_counts = Counter(int(row["source_frame"]) for row in scoped_breaks)
    decile_counts = Counter(int(row["source_frame_decile"]) for row in scoped_breaks)
    spatial = {
        axis: dict(sorted(Counter(int(row["source_spatial_thirds_zyx"][axis]) for row in scoped_breaks).items()))
        for axis in ("z", "y", "x")
    }
    recoverable = [row for row in scoped_breaks if bool(row.get("matched_source")) and bool(row.get("matched_target"))]
    return {
        "video_count": len(videos),
        "edge_totals": {
            key: int(sum(int(row[key]) for row in videos))
            for key in ("edge_tp", "edge_fp", "edge_fn")
        },
        "fn_causes": dict(sorted(fn_causes.items())),
        "fp_direct_classes": dict(sorted(fp_direct.items())),
        "fp_linked_break_classes_inferred": dict(sorted(fp_linked.items())),
        "break_source_frame_counts": {str(key): value for key, value in sorted(frame_counts.items())},
        "break_source_frame_decile_counts": {str(key): value for key, value in sorted(decile_counts.items())},
        "break_source_spatial_thirds": spatial,
        "recoverable_break_distance_um": quantiles([float(row["distance_um"]) for row in recoverable]),
        "recoverable_break_final_node_radius_neighbors": quantiles(
            [float(row["final_node_radius_neighbor_count"]) for row in recoverable]
        ),
        "fragmented_gt_components": int(sum(int(row["gt_components_fragmented"]) for row in videos)),
        "gt_components": int(sum(int(row["gt_component_count"]) for row in videos)),
        "component_sizes": {
            "prediction_nodes": int(sum(int(row["output_node_count"]) for row in videos)),
            "prediction_components": int(
                sum(int(row["prediction_component_count"]) for row in videos)
            ),
            "pooled_prediction_component_mean_nodes": (
                sum(int(row["output_node_count"]) for row in videos)
                / sum(int(row["prediction_component_count"]) for row in videos)
            ),
            "gt_nodes": int(sum(int(row["gt_node_count"]) for row in videos)),
            "gt_weak_components": int(sum(int(row["gt_component_count"]) for row in videos)),
            "pooled_gt_weak_component_mean_nodes": (
                sum(int(row["gt_node_count"]) for row in videos)
                / sum(int(row["gt_component_count"]) for row in videos)
            ),
        },
    }


def crowded_scene_analysis(
    per_video: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    edge = [float(row["edge_jaccard"]) for row in per_video]
    peak = [float(row["peak_component_matrix_dim"]) for row in per_video]
    node_ratio = [float(row["total_node_ratio"]) for row in per_video]
    pred_component = [float(row["prediction_component_mean_nodes"]) for row in per_video]
    log_peak = [math.log1p(value) for value in peak]
    recoverable_events = [
        row for row in event_rows
        if row["kind"] == "tp"
        or (row["kind"] == "fn" and bool(row.get("matched_source")) and bool(row.get("matched_target")))
    ]
    crowded = [row for row in recoverable_events if int(row["final_node_radius_neighbor_count"]) > KD_NEIGHBOR_CAP]
    uncrowded = [row for row in recoverable_events if int(row["final_node_radius_neighbor_count"]) <= KD_NEIGHBOR_CAP]

    def event_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        fn = sum(row["kind"] == "fn" for row in rows)
        return {"events": len(rows), "fn": fn, "break_rate": fn / len(rows) if rows else float("nan")}

    def grouped_break_rates(field: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in event_rows:
            groups[str(row[field])].append(row)
        return {key: event_summary(groups[key]) for key in sorted(groups, key=int)}

    spatial_break_rates = {
        axis: {
            str(index): event_summary(
                [
                    row
                    for row in event_rows
                    if int(row["source_spatial_thirds_zyx"][axis]) == index
                ]
            )
            for index in range(3)
        }
        for axis in ("z", "y", "x")
    }

    return {
        "per_video_correlations": {
            "edge_jaccard_vs_peak_component_matrix_dim": correlation(edge, peak),
            "edge_jaccard_vs_log1p_peak_component_matrix_dim": correlation(
                edge, log_peak
            ),
            "edge_jaccard_vs_total_node_ratio": correlation(edge, node_ratio),
            "edge_jaccard_vs_prediction_component_mean_nodes": correlation(edge, pred_component),
        },
        "event_level_final_node_crowding": {
            "above_cap8": event_summary(crowded),
            "at_or_below_cap8": event_summary(uncrowded),
            "tp_neighbor_count": quantiles(
                [float(row["final_node_radius_neighbor_count"]) for row in recoverable_events if row["kind"] == "tp"]
            ),
            "fn_neighbor_count": quantiles(
                [float(row["final_node_radius_neighbor_count"]) for row in recoverable_events if row["kind"] == "fn"]
            ),
        },
        "temporal_break_rates_by_source_frame_decile": grouped_break_rates(
            "source_frame_decile"
        ),
        "spatial_break_rates_by_source_third": spatial_break_rates,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    split = load_json(args.split)
    selection = scoreall.select_video_ids(split, "fold0")
    all_identities = tuple(str(value) for value in selection.selected_video_ids)
    if len(all_identities) != 58 or EXCLUDED_IDENTITY not in all_identities:
        raise FragmentationAuditError("frozen fold-0 scope is not the expected 58 videos")
    identities = tuple(identity for identity in all_identities if identity != EXCLUDED_IDENTITY)
    prefix = load_json(args.prefix_scores)
    prefix_ids = {str(value) for value in prefix["selected_video_ids"]}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with ZipStore(args.archive, mode="r") as store:
        scores57, score_by_identity = score_57(
            store, identities, args.parts_root, args.prefix_scores
        )
        per_video: list[dict[str, Any]] = []
        breaks: list[dict[str, Any]] = []
        fp_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        pred_nodes_total = pred_components_total = 0
        gt_nodes_total = gt_components_total = 0
        for identity in identities:
            dataset = identity.split(":", 1)[1]
            _, graph = load_fragment(args.parts_root, identity)
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            image_shape = tuple(int(value) for value in group["0"].shape)
            report = load_json(report_path(args.parts_root, identity))
            result = report["results"][0]
            stats = result["graph"]["stats"]
            video_row, video_breaks, video_fp, video_events = analyse_video(
                identity=identity,
                graph=graph,
                gt=gt,
                scale_zyx=scale,
                image_shape_tzyx=image_shape,
                input_node_count=int(stats["input_node_count"]),
                peak_component_matrix_dim=int(stats["peak_component_matrix_dim"]),
                score_row=score_by_identity[identity],
                prefix_ids=prefix_ids,
            )
            per_video.append(video_row)
            breaks.extend(video_breaks)
            fp_rows.extend(video_fp)
            event_rows.extend(video_events)
            pred_nodes_total += int(video_row["output_node_count"])
            pred_components_total += int(video_row["prediction_component_count"])
            gt_nodes_total += len(gt["nodes"]["node_id"])
            gt_components_total += int(video_row["gt_component_count"])

    prefix_scope = aggregate_scope(per_video, breaks, fp_rows, prefix_only=True)
    complete_scope = aggregate_scope(per_video, breaks, fp_rows, prefix_only=False)
    if prefix_scope["edge_totals"] != {
        "edge_tp": PREFIX_EXPECTED["edge_tp"],
        "edge_fp": PREFIX_EXPECTED["edge_fp"],
        "edge_fn": PREFIX_EXPECTED["edge_fn"],
    }:
        raise FragmentationAuditError("prefix-52 attribution does not reproduce frozen edge totals")
    report = {
        "schema": "celltrack-alpha.holdout-fragmentation-audit.v1",
        "status": "completed_read_only_diagnostic",
        "scope": {
            "video_count": 57,
            "selected_video_ids": list(identities),
            "excluded_identity": EXCLUDED_IDENTITY,
            "excluded_fragment_preserved": fragment_path(args.parts_root, EXCLUDED_IDENTITY).is_file(),
            "frozen_split_sha256": sha256_file(args.split),
            "frozen_scorer_unchanged": True,
        },
        "classification_contract": {
            "verified_order": [
                "unmatched_endpoint",
                "nonadjacent_matched_endpoints",
                "gap_synthetic_endpoint",
                "final_pair_geometry_above_gate_radius",
                "final_pair_rank_above_cap8",
                "solver_outdegree_division",
                "occupied_endpoint_conflict",
                "no_observed_endpoint_conflict",
            ],
            "candidate_present_caveat": (
                "Final fragments omit detector confidences, embeddings, pre-prune detections, and candidate logits. "
                "Rank <= 8 among surviving final nodes is a lower bound, not proof of membership in the original "
                "cap-8 candidate tensor. An observed occupied endpoint is exact; 'out-ranked' is an inference."
            ),
            "gate_geometry_caveat": (
                "Final coordinates are post-line-fit and post-image-support projection, whereas candidate gating used "
                "raw detector coordinates. A final matched-pair displacement above 10 um is verified geometry but is "
                "only a proxy—not proof—that the raw pair was excluded by the radius gate."
            ),
            "fp_linking_caveat": (
                "FP direct endpoint classes are exact. FP allocation to break causes is inferred by linking each "
                "valid FP to missing GT edges incident on its matched endpoint(s); mixed and unpaired cases remain explicit."
            ),
        },
        "component_sizes": {
            "prediction_nodes": pred_nodes_total,
            "prediction_components": pred_components_total,
            "pooled_prediction_component_mean_nodes": pred_nodes_total / pred_components_total,
            "gt_nodes": gt_nodes_total,
            "gt_weak_components": gt_components_total,
            "pooled_gt_weak_component_mean_nodes": gt_nodes_total / gt_components_total,
        },
        "prefix_52": prefix_scope,
        "complete_57": complete_scope,
        "crowded_scene_hypothesis": {
            "prefix_52": crowded_scene_analysis(
                [row for row in per_video if bool(row["prefix_52"])],
                [row for row in event_rows if bool(row["prefix_52"])],
            ),
            "complete_57": crowded_scene_analysis(per_video, event_rows),
        },
        "per_video": per_video,
        "breaks": breaks,
    }
    (args.output_dir / "scores.json").write_text(
        json.dumps(scores57, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "fragmentation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return scores57, report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip")
    parser.add_argument("--split", type=Path, default=ROOT / "reports/g02/grouped_3fold_audit20_v1.json")
    parser.add_argument(
        "--parts-root",
        type=Path,
        default=ROOT / "reports/g02/source_softmax_edge_f0_fit117_holdout_parts_v1",
    )
    parser.add_argument(
        "--prefix-scores",
        type=Path,
        default=ROOT / "reports/g02/source_softmax_edge_f0_fit117_holdout_prefix52_v1/scores.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scores, report = run(args)
    print(
        json.dumps(
            {
                "scores": scores["summary"],
                "component_sizes": report["component_sizes"],
                "prefix_52_fn_causes": report["prefix_52"]["fn_causes"],
                "complete_57_fn_causes": report["complete_57"]["fn_causes"],
                "correlations": report["crowded_scene_hypothesis"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
