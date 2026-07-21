#!/usr/bin/env python3
"""Measure label-only association headroom in a preserved V45 fragment.

The tool does not run inference or alter a graph.  It matches the submitted
nodes to the sparse GEFF labels with a physical cKDTree candidate graph, then
asks whether each labelled edge whose endpoints are already matched is present
in the same 6/10 um candidate graph available to the V45 linker.  This is an
upper-bound diagnostic, not a score and not a label-aware solver.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import zarr
from scipy.spatial import cKDTree
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_bh
from celltrack_alpha import bh_graph
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays


SCHEMA = "celltrack-alpha.v45-candidate-diagnostic.v1"
FROZEN_REGISTRY_SHA256 = "21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158"
FROZEN_SPLIT_SHA256 = "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769"
MATCH_RADIUS_UM = 7.0
TIGHT_RADIUS_UM = 6.0
RELAXED_RADIUS_UM = 10.0


class V45GraphDiagnosticError(ValueError):
    """Raised when an input would make the diagnostic non-provenance-safe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise V45GraphDiagnosticError(f"{label} must be an integer") from exc
    if str(parsed) != str(value).strip():
        raise V45GraphDiagnosticError(f"{label} must be a canonical integer")
    return parsed


def _point(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(row["z"]), float(row["y"]), float(row["x"])], dtype=np.float64)


def _rows_from_fragment(path: Path) -> tuple[str, list[dict[str, int]], set[tuple[int, int]]]:
    rows = run_bh.read_rows(path)
    datasets = {str(row["dataset"]) for row in rows}
    if len(datasets) != 1:
        raise V45GraphDiagnosticError("fragment must contain exactly one dataset")
    dataset = next(iter(datasets))
    nodes: list[dict[str, int]] = []
    edges: set[tuple[int, int]] = set()
    for row in rows:
        if row["row_type"] == "node":
            nodes.append(
                {
                    "node_id": _int(row["node_id"], "node_id"),
                    "t": _int(row["t"], "t"),
                    "z": _int(row["z"], "z"),
                    "y": _int(row["y"], "y"),
                    "x": _int(row["x"], "x"),
                }
            )
        elif row["row_type"] == "edge":
            edges.add((_int(row["source_id"], "source_id"), _int(row["target_id"], "target_id")))
        else:
            raise V45GraphDiagnosticError("fragment has an unknown row type")
    if not nodes:
        raise V45GraphDiagnosticError("fragment contains no nodes")
    return dataset, nodes, edges


def _radius_pairs(
    sources: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    radius_um: float,
    scale_zyx: Sequence[float],
) -> list[tuple[int, int, float]]:
    """Discover feasible pairs without materializing a frame-by-frame matrix."""

    if not sources or not targets:
        return []
    scale = np.asarray(scale_zyx, dtype=np.float64)
    source_xyz = np.asarray([_point(node) for node in sources], dtype=np.float64) * scale
    target_xyz = np.asarray([_point(node) for node in targets], dtype=np.float64) * scale
    tree = cKDTree(target_xyz)
    result: list[tuple[int, int, float]] = []
    for source_index, neighbor_indices in enumerate(
        tree.query_ball_point(source_xyz, r=float(radius_um), return_sorted=True)
    ):
        for target_index in neighbor_indices:
            distance = float(np.linalg.norm(source_xyz[source_index] - target_xyz[int(target_index)]))
            if distance <= float(radius_um):
                result.append(
                    (
                        int(sources[source_index]["node_id"]),
                        int(targets[int(target_index)]["node_id"]),
                        distance,
                    )
                )
    return result


def sparse_node_match(
    pred_nodes: Sequence[Mapping[str, Any]],
    gt: Mapping[str, Any],
    scale_zyx: Sequence[float],
) -> tuple[dict[int, int], dict[str, int]]:
    """Reproduce centroid matching with sparse 7 um candidates per frame."""

    pred_by_t: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for node in pred_nodes:
        pred_by_t[int(node["t"])].append(node)
    gt_by_t: dict[int, list[dict[str, int]]] = defaultdict(list)
    gt_nodes = gt["nodes"]
    for node_id, t, z, y, x in zip(
        gt_nodes["node_id"], gt_nodes["t"], gt_nodes["z"], gt_nodes["y"], gt_nodes["x"], strict=True
    ):
        gt_by_t[int(t)].append(
            {"node_id": int(node_id), "t": int(t), "z": int(z), "y": int(y), "x": int(x)}
        )
    matched: dict[int, int] = {}
    candidate_count = 0
    component_count = 0
    peak_component_matrix_dim = 0
    for t in sorted(set(pred_by_t) & set(gt_by_t)):
        candidates = _radius_pairs(
            pred_by_t[t], gt_by_t[t], radius_um=MATCH_RADIUS_UM, scale_zyx=scale_zyx
        )
        candidate_count += len(candidates)
        result = bh_graph.sparse_min_cost_max_cardinality(
            (int(node["node_id"]) for node in pred_by_t[t]),
            (int(node["node_id"]) for node in gt_by_t[t]),
            candidates,
        )
        component_count += result.component_count
        peak_component_matrix_dim = max(peak_component_matrix_dim, result.peak_component_matrix_dim)
        for item in result.matches:
            if item.source_id in matched:
                raise V45GraphDiagnosticError("sparse matching returned a duplicate prediction")
            matched[item.source_id] = item.target_id
    return matched, {
        "candidate_pairs": candidate_count,
        "component_count": component_count,
        "peak_component_matrix_dim": peak_component_matrix_dim,
    }


def adjacent_candidates(
    pred_nodes: Sequence[Mapping[str, Any]], scale_zyx: Sequence[float]
) -> tuple[dict[int, tuple[tuple[int, float], ...]], int]:
    """Return the 10 um candidate targets and deterministic raw-distance ranks."""

    by_t: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for node in pred_nodes:
        by_t[int(node["t"])].append(node)
    by_source: dict[int, tuple[tuple[int, float], ...]] = {}
    total_pairs = 0
    for t in sorted(by_t):
        if t + 1 not in by_t:
            continue
        for source_id, target_id, distance in _radius_pairs(
            by_t[t], by_t[t + 1], radius_um=RELAXED_RADIUS_UM, scale_zyx=scale_zyx
        ):
            existing = list(by_source.get(source_id, ()))
            existing.append((target_id, distance))
            by_source[source_id] = tuple(existing)
            total_pairs += 1
    for source_id, candidates in tuple(by_source.items()):
        by_source[source_id] = tuple(sorted(candidates, key=lambda item: (item[1], item[0])))
    return by_source, total_pairs


def diagnose(
    pred_nodes: Sequence[Mapping[str, Any]],
    pred_edges: set[tuple[int, int]],
    gt: Mapping[str, Any],
    scale_zyx: Sequence[float],
) -> dict[str, Any]:
    """Calculate candidate coverage only; no labels influence an output graph."""

    pred_to_gt, match_stats = sparse_node_match(pred_nodes, gt, scale_zyx)
    gt_to_pred = {gt_id: pred_id for pred_id, gt_id in pred_to_gt.items()}
    if len(gt_to_pred) != len(pred_to_gt):
        raise V45GraphDiagnosticError("matching is not one-to-one")
    by_source, candidate_pair_count = adjacent_candidates(pred_nodes, scale_zyx)
    gt_edges = {(int(source), int(target)) for source, target in gt["edges"]}
    gt_nodes = gt["nodes"]
    gt_t = {int(node_id): int(t) for node_id, t in zip(gt_nodes["node_id"], gt_nodes["t"], strict=True)}
    node_by_id = {int(node["node_id"]): node for node in pred_nodes}
    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source_id, target_id in pred_edges:
        incoming[target_id].append(source_id)
        outgoing[source_id].append(target_id)

    def final_graph_cost(source_id: int, target_id: int) -> float:
        """Probe the locked V45 cost with the final graph's predecessor only."""

        source = _point(node_by_id[source_id])
        target = _point(node_by_id[target_id])
        parents = incoming.get(source_id, ())
        predicted = source
        if parents:
            predecessor = _point(node_by_id[parents[0]])
            predicted = source + bh_graph.V45_GRAPH_CONFIG.velocity_weight * (source - predecessor)
        raw_distance = bh_graph.physical_distance(source, target, scale_zyx)
        motion_error = bh_graph.physical_distance(predicted, target, scale_zyx)
        return bh_graph.motion_only_edge_cost(raw_distance, motion_error)

    endpoints_matched = 0
    nonadjacent_gt = 0
    reachable_tight = 0
    reachable_relaxed = 0
    selected_true_positive = 0
    reachable_not_selected = 0
    rank_histogram: Counter[str] = Counter()
    source_conflicts = 0
    target_conflicts = 0
    source_conflicts_where_true_cost_lower = 0
    target_conflicts_where_true_cost_lower = 0
    for source_gt, target_gt in sorted(gt_edges):
        if gt_t.get(target_gt) != gt_t.get(source_gt, -1) + 1:
            nonadjacent_gt += 1
            continue
        source_pred = gt_to_pred.get(source_gt)
        target_pred = gt_to_pred.get(target_gt)
        if source_pred is None or target_pred is None:
            continue
        endpoints_matched += 1
        candidates = by_source.get(source_pred, ())
        positions = {candidate_id: (rank, distance) for rank, (candidate_id, distance) in enumerate(candidates, start=1)}
        rank_distance = positions.get(target_pred)
        if rank_distance is None:
            continue
        rank, distance = rank_distance
        reachable_relaxed += 1
        reachable_tight += int(distance <= TIGHT_RADIUS_UM)
        rank_histogram[str(rank)] += 1
        if (source_pred, target_pred) in pred_edges:
            selected_true_positive += 1
        else:
            reachable_not_selected += 1
            source_alternatives = tuple(outgoing.get(source_pred, ()))
            target_alternatives = tuple(incoming.get(target_pred, ()))
            true_cost = final_graph_cost(source_pred, target_pred)
            if source_alternatives:
                source_conflicts += 1
                if true_cost < min(final_graph_cost(source_pred, item) for item in source_alternatives):
                    source_conflicts_where_true_cost_lower += 1
            if target_alternatives:
                target_conflicts += 1
                if true_cost < min(final_graph_cost(item, target_pred) for item in target_alternatives):
                    target_conflicts_where_true_cost_lower += 1

    predicted_true_positive = sum(
        1
        for source_pred, target_pred in pred_edges
        if (source_pred in pred_to_gt)
        and (target_pred in pred_to_gt)
        and (pred_to_gt[source_pred], pred_to_gt[target_pred]) in gt_edges
    )
    if selected_true_positive != predicted_true_positive:
        raise V45GraphDiagnosticError("selected true-positive accounting disagrees with candidate accounting")
    return {
        "matching": {
            **match_stats,
            "matched_gt_nodes": len(gt_to_pred),
            "ground_truth_nodes": len(gt_nodes["node_id"]),
            "node_recall": len(gt_to_pred) / len(gt_nodes["node_id"]) if len(gt_nodes["node_id"]) else None,
        },
        "candidate_graph": {
            "algorithm": "physical-cKDTree",
            "full_frame_dense_matrix": False,
            "tight_radius_um": TIGHT_RADIUS_UM,
            "relaxed_radius_um": RELAXED_RADIUS_UM,
            "adjacent_candidate_pairs": candidate_pair_count,
        },
        "edge_headroom": {
            "ground_truth_edges": len(gt_edges),
            "nonadjacent_ground_truth_edges": nonadjacent_gt,
            "matched_endpoint_edges": endpoints_matched,
            "reachable_at_6um": reachable_tight,
            "reachable_at_10um": reachable_relaxed,
            "coverage_of_matched_endpoints_at_10um": (
                reachable_relaxed / endpoints_matched if endpoints_matched else None
            ),
            "current_selected_true_positives": selected_true_positive,
            "reachable_but_not_selected": reachable_not_selected,
            "raw_distance_rank_histogram": dict(sorted(rank_histogram.items(), key=lambda item: int(item[0]))),
        },
        "posthoc_local_cost_probe": {
            "definition": (
                "Uses the final graph's predecessor and the locked V45 raw-distance-plus-motion "
                "cost. It is not a replay of the pre-pruning greedy solve."
            ),
            "reachable_misses": reachable_not_selected,
            "source_conflicts": source_conflicts,
            "target_conflicts": target_conflicts,
            "source_conflicts_where_true_cost_lower": source_conflicts_where_true_cost_lower,
            "target_conflicts_where_true_cost_lower": target_conflicts_where_true_cost_lower,
        },
    }


def _scope_record(dataset: str, registry_path: Path, split_path: Path, fold: int) -> Any:
    if sha256_file(registry_path) != FROZEN_REGISTRY_SHA256:
        raise V45GraphDiagnosticError("registry does not match the frozen G-02 identity")
    if sha256_file(split_path) != FROZEN_SPLIT_SHA256:
        raise V45GraphDiagnosticError("split does not match the frozen G-02 identity")
    registry = run_bh.strict_json_load(registry_path)
    split = run_bh.strict_json_load(split_path)
    if registry.get("artifact") != "dataset_registry_v1" or split.get("artifact") != "grouped_3fold_audit20_v1":
        raise V45GraphDiagnosticError("registry or split schema is not the frozen G-02 manifest")
    records = run_bh.load_registry(registry_path)
    by_dataset = {str(record.dataset): record for record in records}
    record = by_dataset.get(dataset)
    if record is None:
        raise V45GraphDiagnosticError("fragment dataset is absent from the frozen registry")
    folds = split.get("folds")
    audit = split.get("audit")
    if not isinstance(folds, Mapping) or not isinstance(audit, Mapping) or str(fold) not in folds:
        raise V45GraphDiagnosticError("frozen split does not contain the requested fold")
    fold_data = folds[str(fold)]
    held = set(fold_data.get("held_out_video_ids", ()))
    fit = set(fold_data.get("fit_video_ids", ()))
    audit_ids = set(audit.get("video_ids", ()))
    if str(record.video_id) not in held or str(record.video_id) in fit or str(record.video_id) in audit_ids:
        raise V45GraphDiagnosticError("diagnostic may read only the requested fold's held-out train video")
    if str(record.split) != "train":
        raise V45GraphDiagnosticError("diagnostic may not read an unlabeled test video")
    return record


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="competition ZIP containing the labelled train GEFF")
    parser.add_argument("--fragment", type=Path, required=True, help="one existing held-out submission fragment")
    parser.add_argument("--fragment-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--registry", type=Path, default=ROOT / "reports/g02/dataset_registry_v1.json")
    parser.add_argument("--split", type=Path, default=ROOT / "reports/g02/grouped_3fold_audit20_v1.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    if args.out.exists():
        raise V45GraphDiagnosticError("refusing to overwrite an existing diagnostic report")
    fragment_sha256 = sha256_file(args.fragment)
    if fragment_sha256 != args.fragment_sha256:
        raise V45GraphDiagnosticError("fragment SHA-256 does not match the supplied pinned value")
    dataset, nodes, edges = _rows_from_fragment(args.fragment)
    record = _scope_record(dataset, args.registry, args.split, args.fold)
    store = ZipStore(args.input, mode="r")
    try:
        group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
        gt = read_geff_arrays(store, "train", dataset)
        scale = parse_scale(dict(group.attrs))
    finally:
        store.close()
    result = diagnose(nodes, edges, gt, scale)
    report = {
        "schema": SCHEMA,
        "kind": "posthoc-label-only-candidate-coverage",
        "input": {
            "archive": str(args.input),
            "fragment": str(args.fragment),
            "fragment_sha256": fragment_sha256,
            "dataset": dataset,
            "video_id": str(record.video_id),
        },
        "scope": {
            "fold": args.fold,
            "fit_used": False,
            "audit_used": False,
            "test_used": False,
            "registry_sha256": FROZEN_REGISTRY_SHA256,
            "split_sha256": FROZEN_SPLIT_SHA256,
        },
        "interpretation": {
            "label_aware_graph_selection": False,
            "score": None,
            "meaning": (
                "reachable_but_not_selected is an upper-bound count only: it shows that a future "
                "label-free association scorer could in principle recover a currently missed edge."
            ),
            "limitations": [
                "Uses postprocessed submission nodes, not raw detector candidates.",
                "Does not reproduce predecessor-dependent motion costs, gap synthesis, pruning, or division policy.",
                "Does not establish a global solver improvement; it only rules out absent-candidate explanations.",
            ],
        },
        "result": result,
    }
    run_bh.atomic_create_json(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
