#!/usr/bin/env python3
"""Reproduce division-sandbox behavior and decompose the frozen 57-video CV.

This is a read-only analysis over repository artifacts.  It writes one JSON
report so every number in the accompanying markdown can be regenerated.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np
import polars as pl
from scipy.stats import spearmanr
import tracksdata as td

from celltrack_alpha.bh_graph import (
    GraphEdge,
    GraphNode,
    PHYSICAL_SCALE_ZYX,
    add_safe_divisions,
    physical_distance,
)
from tracking_cellmot.division_metrics import evaluate_divisions, score_divisions


ROOT = Path(__file__).resolve().parents[1]
SANDBOX_ROOT = ROOT / "external/kaggle-cell-tracking-competition/tests/assets/sandbox_examples"
SCORES_PATH = ROOT / "_ARCHIVE/conflict_residual/reports/g02/run_bh_d4_conflict_edge_f0_holdout57_v1/scores_postpatch.json"
FEATURE_ROOT = ROOT / "kaggle_kernel/bhcv57out_v3/cache"
DEFAULT_OUTPUT = ROOT / "reports/g02/division_sandbox_and_holdout_decomposition_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def td_graph(spec: dict[str, Any]) -> tuple[td.graph.InMemoryGraph, dict[int, int]]:
    graph = td.graph.InMemoryGraph()
    graph.add_node_attr_key("z", pl.Float64, 0.0)
    graph.add_node_attr_key("y", pl.Float64, 0.0)
    graph.add_node_attr_key("x", pl.Float64, 0.0)
    node_ids: dict[int, int] = {}
    for node in spec["nodes"]:
        node_ids[int(node["uid"])] = graph.add_node(
            {"t": int(node["t"]), "z": 0.0, "y": float(node["y"]), "x": 0.0}
        )
    for edge in spec["edges"]:
        graph.add_edge(node_ids[int(edge["source"])], node_ids[int(edge["target"])], {})
    return graph, node_ids


def bh_graph(spec: dict[str, Any]) -> tuple[list[GraphNode], list[GraphEdge]]:
    # Sandbox y values are physical distances.  Convert to voxels so the
    # production helper's fixed y scale reproduces those distances.
    y_scale = PHYSICAL_SCALE_ZYX[1]
    nodes = [
        GraphNode(
            node_id=int(node["uid"]),
            t=int(node["t"]),
            z=0.0,
            y=float(node["y"]) / y_scale,
            x=0.0,
        )
        for node in spec["nodes"]
    ]
    by_id = {node.node_id: node for node in nodes}
    edges: list[GraphEdge] = []
    for item in spec["edges"]:
        source = by_id[int(item["source"])]
        target = by_id[int(item["target"])]
        distance = physical_distance(source, target)
        edges.append(GraphEdge(source.node_id, target.node_id, distance, distance))
    return nodes, edges


def degree_map(edges: Iterable[dict[str, Any]], key: str) -> Counter[int]:
    return Counter(int(edge[key]) for edge in edges)


def topology_summary(spec: dict[str, Any]) -> dict[str, Any]:
    edges = spec["edges"]
    outdegree = degree_map(edges, "source")
    indegree = degree_map(edges, "target")
    forks = sorted(node for node, degree in outdegree.items() if degree >= 2)
    merges = sorted(node for node, degree in indegree.items() if degree >= 2)
    return {
        "nodes": len(spec["nodes"]),
        "edges": len(edges),
        "forks": [
            {
                "node": node,
                "time": next(int(item["t"]) for item in spec["nodes"] if int(item["uid"]) == node),
                "children": sorted(
                    int(edge["target"]) for edge in edges if int(edge["source"]) == node
                ),
                "outdegree": outdegree[node],
            }
            for node in forks
        ],
        "merges": merges,
        "roots": sorted(
            int(node["uid"]) for node in spec["nodes"] if indegree[int(node["uid"])] == 0
        ),
        "leaves": sorted(
            int(node["uid"]) for node in spec["nodes"] if outdegree[int(node["uid"])] == 0
        ),
    }


def division_counts(pred_spec: dict[str, Any], gt_spec: dict[str, Any], max_distance: float) -> dict[str, Any]:
    pred, pred_ids = td_graph(pred_spec)
    gt, _ = td_graph(gt_spec)
    result = score_divisions(pred, gt, max_distance=max_distance)
    inverse = {graph_id: uid for uid, graph_id in pred_ids.items()}
    counts = evaluate_divisions(pred, gt, max_distance=max_distance)
    return {
        "tp": int(counts.tp),
        "fp": int(counts.fp),
        "fn": int(counts.fn),
        "tp_forks": sorted(inverse[node] for node in result.tp_forks),
        "fp_forks": sorted(inverse[node] for node in result.fp_forks),
        "gt_event_scores": {str(key): int(value) for key, value in result.scores.items()},
    }


def spec_with_edges(spec: dict[str, Any], pairs: Iterable[tuple[int, int]]) -> dict[str, Any]:
    return {
        "nodes": spec["nodes"],
        "edges": [{"source": source, "target": target} for source, target in pairs],
    }


def safe_division_replay(case: dict[str, Any]) -> dict[str, Any]:
    pred_spec = case["pred"]
    gt_spec = case["gt"]
    max_distance = float(case["max_distance"])
    original_pairs = [
        (int(edge["source"]), int(edge["target"])) for edge in pred_spec["edges"]
    ]
    nodes, edges = bh_graph(pred_spec)
    direct: dict[str, Any]
    try:
        output, stats = add_safe_divisions(nodes, edges)
        output_pairs = [(edge.source_id, edge.target_id) for edge in output]
        added = sorted(set(output_pairs) - set(original_pairs))
        direct_spec = spec_with_edges(pred_spec, output_pairs)
        direct = {
            "status": "ok",
            "added_edges": [list(pair) for pair in added],
            "stats": stats,
            "official_after": division_counts(direct_spec, gt_spec, max_distance),
        }
    except Exception as exc:  # record fail-closed behavior without hiding it
        direct = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    outdegree = Counter(source for source, _ in original_pairs)
    reconstruction: list[dict[str, Any]] = []
    for source in sorted(node for node, degree in outdegree.items() if degree >= 2):
        children = sorted(target for edge_source, target in original_pairs if edge_source == source)
        for removed_child in children:
            removed_pair = (source, removed_child)
            base_pairs = [pair for pair in original_pairs if pair != removed_pair]
            base_spec = spec_with_edges(pred_spec, base_pairs)
            base_nodes, base_edges = bh_graph(base_spec)
            try:
                output, stats = add_safe_divisions(base_nodes, base_edges)
                output_pairs = [(edge.source_id, edge.target_id) for edge in output]
                added = sorted(set(output_pairs) - set(base_pairs))
                reconstructed_spec = spec_with_edges(pred_spec, output_pairs)
                reconstruction.append(
                    {
                        "removed_edge": list(removed_pair),
                        "added_edges": [list(pair) for pair in added],
                        "restored_exact_edge": removed_pair in added,
                        "stats": stats,
                        "official_after": division_counts(
                            reconstructed_spec, gt_spec, max_distance
                        ),
                    }
                )
            except Exception as exc:
                reconstruction.append(
                    {
                        "removed_edge": list(removed_pair),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return {"direct": direct, "single_edge_reconstruction": reconstruction}


def analyze_sandbox() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for path in sorted(SANDBOX_ROOT.glob("*.json")):
        case = read_json(path)
        official = division_counts(case["pred"], case["gt"], float(case["max_distance"]))
        cases.append(
            {
                "name": path.stem,
                "path": str(path.relative_to(ROOT)),
                "max_distance": float(case["max_distance"]),
                "gt": topology_summary(case["gt"]),
                "pred": topology_summary(case["pred"]),
                "official": official,
                "add_safe_divisions": safe_division_replay(case),
            }
        )
    return {
        "case_count": len(cases),
        "cases": cases,
        "tp_shapes": [row["name"] for row in cases if row["official"]["tp"] > 0],
        "non_tp_shapes": [row["name"] for row in cases if row["official"]["tp"] == 0],
    }


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [float(row["residual_error"]) for row in rows]
    weights = [float(row["edge_event_weight"]) for row in rows]
    total_weight = sum(weights)
    weighted_score = sum(
        weight * float(row["score"])
        for weight, row in zip(weights, rows, strict=True)
    ) / total_weight
    return {
        "n": len(rows),
        "unweighted_mean_score": statistics.fmean(float(row["score"]) for row in rows),
        "official_weighted_score": weighted_score,
        "official_weighted_residual_error": 1.0 - weighted_score,
        "edge_event_weight": total_weight,
        "weighted_residual_mass": sum(
            weight * error for weight, error in zip(weights, errors, strict=True)
        ),
        "median_residual_error": statistics.median(errors),
        "edge_tp": sum(int(row["edge_tp"]) for row in rows),
        "edge_fp": sum(int(row["edge_fp"]) for row in rows),
        "edge_fn": sum(int(row["edge_fn"]) for row in rows),
        "median_density_detections_per_frame": statistics.median(
            float(row["density_detections_q50"]) for row in rows
        ),
        "median_nearest_neighbor_um": statistics.median(
            float(row["nearest_neighbor_um_q50"]) for row in rows
        ),
        "median_depth_planes": statistics.median(float(row["depth_planes"]) for row in rows),
        "median_motion_unmatched_6um_fraction": statistics.median(
            float(row["motion_unmatched_6um_fraction"]) for row in rows
        ),
        "median_node_ratio": statistics.median(float(row["node_ratio"]) for row in rows),
        "median_abs_node_ratio": statistics.median(abs(float(row["node_ratio"])) for row in rows),
    }


def feature_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = [
        "density_detections_q50",
        "nearest_neighbor_um_q50",
        "depth_planes",
        "motion_unmatched_6um_fraction",
        "motion_unmatched_10um_fraction",
        "depth_edge_fraction",
        "node_ratio",
        "abs_node_ratio",
    ]
    errors = np.asarray([row["residual_error"] for row in rows], dtype=np.float64)
    result: dict[str, Any] = {}
    for feature in features:
        values = np.asarray([row[feature] for row in rows], dtype=np.float64)
        rho, pvalue = spearmanr(values, errors)
        result[feature] = {"spearman_rho": finite_float(rho), "pvalue": finite_float(pvalue)}

    # Family-demeaned Pearson is a compact check that an apparent correlation
    # is not merely the 44b6/6bba family split.
    error_centered = np.asarray(
        [
            row["residual_error"]
            - statistics.fmean(item["residual_error"] for item in rows if item["family"] == row["family"])
            for row in rows
        ],
        dtype=np.float64,
    )
    for feature in features:
        centered = np.asarray(
            [
                row[feature]
                - statistics.fmean(item[feature] for item in rows if item["family"] == row["family"])
                for row in rows
            ],
            dtype=np.float64,
        )
        if float(np.std(centered)) == 0.0 or float(np.std(error_centered)) == 0.0:
            correlation = None
        else:
            correlation = float(np.corrcoef(centered, error_centered)[0, 1])
        result[feature]["family_demeaned_pearson"] = correlation
    return result


def feature_profile(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "density_detections_q50",
        "nearest_neighbor_um_q50",
        "depth_planes",
        "depth_edge_fraction",
        "motion_unmatched_6um_fraction",
        "motion_unmatched_10um_fraction",
        "node_ratio",
        "abs_node_ratio",
    )
    return {key: statistics.median(float(row[key]) for row in rows) for key in keys}


def feature_quartiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "density_detections_q50",
        "nearest_neighbor_um_q50",
        "depth_edge_fraction",
        "motion_unmatched_6um_fraction",
        "motion_unmatched_10um_fraction",
        "node_ratio",
        "abs_node_ratio",
    )
    result: dict[str, Any] = {}
    for key in keys:
        ordered = sorted(rows, key=lambda row: (row[key], row["identity"]))
        size = math.ceil(len(rows) / 4)
        low = ordered[:size]
        high = ordered[-size:]
        result[key] = {
            "low_quartile_range": [float(low[0][key]), float(low[-1][key])],
            "low_quartile_official_weighted_score": summarize_rows(low)["official_weighted_score"],
            "high_quartile_range": [float(high[0][key]), float(high[-1][key])],
            "high_quartile_official_weighted_score": summarize_rows(high)["official_weighted_score"],
        }
    return result


def analyze_holdout() -> dict[str, Any]:
    scores = read_json(SCORES_PATH)
    rows: list[dict[str, Any]] = []
    for sample in scores["samples"]:
        identity = str(sample["identity"])
        feature_path = FEATURE_ROOT / f"{identity.replace(':', '__')}.json"
        feature_doc = read_json(feature_path)
        features = feature_doc["pregraph_features"]["features"]
        score = float(sample["adj_edge_jaccard"])
        depth_planes = round(math.exp(float(features["log_depth_planes"])) - 1.0)
        node_ratio = float(sample["total_node_ratio"])
        rows.append(
            {
                "identity": identity,
                "family": str(sample["dataset"]).split("_", maxsplit=1)[0],
                "score": score,
                "residual_error": 1.0 - score,
                "edge_tp": int(sample["edge_tp"]),
                "edge_fp": int(sample["edge_fp"]),
                "edge_fn": int(sample["edge_fn"]),
                "edge_event_weight": int(sample["edge_tp"] + sample["edge_fp"] + sample["edge_fn"]),
                "edge_fp_rate_vs_gt": float(sample["edge_fp"]) / float(sample["ground_truth_edges"]),
                "edge_fn_rate_vs_gt": float(sample["edge_fn"]) / float(sample["ground_truth_edges"]),
                "node_recall": float(sample["node_recall"]),
                "node_ratio": node_ratio,
                "abs_node_ratio": abs(node_ratio),
                "density_detections_q50": float(features["detections_q50"]),
                "nearest_neighbor_um_q50": float(features["nn_um_q50"]),
                "depth_planes": int(depth_planes),
                "depth_edge_fraction": float(features["depth_edge_fraction"]),
                "motion_unmatched_6um_fraction": float(features["unmatched_6um_fraction"]),
                "motion_unmatched_10um_fraction": float(features["unmatched_10um_fraction"]),
                "count_cv": float(features["count_cv"]),
                "count_jump_q90": float(features["count_jump_q90"]),
            }
        )
    rows.sort(key=lambda row: (-row["residual_error"], row["identity"]))
    total_residual_mass = sum(
        row["edge_event_weight"] * row["residual_error"] for row in rows
    )
    for row in rows:
        row["weighted_residual_mass"] = row["edge_event_weight"] * row["residual_error"]
        row["weighted_error_share"] = row["weighted_residual_mass"] / total_residual_mass

    # "Carries the error" means contribution to the official weighted
    # adjusted-edge aggregate, not simply the worst unweighted video score.
    contribution_order = sorted(
        rows, key=lambda row: (-row["weighted_residual_mass"], row["identity"])
    )
    cumulative = 0.0
    for rank, row in enumerate(contribution_order, start=1):
        cumulative += row["weighted_residual_mass"]
        row["weighted_error_rank"] = rank
        row["cumulative_weighted_error_share"] = cumulative / total_residual_mass
    rows.sort(key=lambda row: (row["weighted_error_rank"], row["identity"]))

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_rows[row["family"]].append(row)
    families = {name: summarize_rows(items) for name, items in sorted(family_rows.items())}
    for summary in families.values():
        summary["share_of_total_weighted_residual_error"] = (
            summary["weighted_residual_mass"] / total_residual_mass
        )

    concentration: dict[str, Any] = {}
    for count in (1, 3, 5, 10, 12):
        concentration[f"top_{count}_share"] = sum(
            row["weighted_residual_mass"] for row in rows[:count]
        ) / total_residual_mass
    concentration["videos_for_50pct_error"] = next(
        row["weighted_error_rank"]
        for row in rows
        if row["cumulative_weighted_error_share"] >= 0.5
    )
    concentration["videos_for_80pct_error"] = next(
        row["weighted_error_rank"]
        for row in rows
        if row["cumulative_weighted_error_share"] >= 0.8
    )

    return {
        "source": str(SCORES_PATH.relative_to(ROOT)),
        "source_summary": scores["summary"],
        "overall": summarize_rows(rows),
        "families": families,
        "concentration": concentration,
        "top_10_contributor_profile": feature_profile(rows[:10]),
        "remaining_47_profile": feature_profile(rows[10:]),
        "feature_quartiles": feature_quartiles(rows),
        "feature_correlations": feature_correlations(rows),
        "videos": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = {
        "schema": "celltrack-alpha.division-sandbox-holdout-decomposition.v1",
        "division_sandbox": analyze_sandbox(),
        "holdout_57": analyze_holdout(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.output.with_name(args.output.stem + "_per_video.csv")
    csv_fields = (
        "weighted_error_rank",
        "identity",
        "family",
        "score",
        "residual_error",
        "weighted_error_share",
        "cumulative_weighted_error_share",
        "edge_event_weight",
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "density_detections_q50",
        "nearest_neighbor_um_q50",
        "depth_planes",
        "depth_edge_fraction",
        "motion_unmatched_6um_fraction",
        "motion_unmatched_10um_fraction",
        "node_ratio",
        "abs_node_ratio",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["holdout_57"]["videos"])
    print(
        json.dumps(
            {
                "output": str(args.output),
                "per_video_csv": str(csv_path),
                "sandbox_cases": report["division_sandbox"]["case_count"],
                "holdout_videos": report["holdout_57"]["overall"]["n"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
