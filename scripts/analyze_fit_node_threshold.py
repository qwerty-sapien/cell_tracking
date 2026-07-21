#!/usr/bin/env python3
"""Calibrate score-threshold node pruning on available fold-fit caches only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
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

import run_bh  # noqa: E402
from celltrack_alpha import bh_graph  # noqa: E402
from celltrack_alpha.bh_model import (  # noqa: E402
    CacheIdentity,
    load_sparse_cache,
    validate_artifact_bundle,
)
from celltrack_alpha.official_eval import official_edge_metric_row  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_THRESHOLDS = (0.970, 0.975, 0.980, 0.985, 0.990, 0.9925, 0.995, 0.9975, 0.999)


class NodeThresholdError(RuntimeError):
    """Fit-only node-threshold calibration inputs are incomplete or unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise NodeThresholdError(f"JSON root is not an object: {path}")
    return value


def _scenario(identity: str) -> str:
    stem = identity.split(":", 1)[-1]
    prefix = stem.split("_", 1)[0]
    if prefix not in {"44b6", "6bba"}:
        raise NodeThresholdError(f"unknown scenario prefix: {identity}")
    return prefix


def _rows(graph: Any) -> tuple[list[dict[str, int | float]], list[dict[str, int]]]:
    nodes = [
        {
            "node_id": int(node.node_id),
            "t": int(node.t),
            "z": float(node.z),
            "y": float(node.y),
            "x": float(node.x),
        }
        for node in graph.nodes
    ]
    edges = [
        {"source_id": int(edge.source_id), "target_id": int(edge.target_id)}
        for edge in graph.edges
    ]
    return nodes, edges


def _aggregate(samples: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not samples:
        raise NodeThresholdError("cannot aggregate an empty threshold slice")
    estimated = sum(float(row["estimated_number_of_nodes"]) for row in samples)
    predicted = sum(float(row["num_pred_nodes"]) for row in samples)
    return {
        "videos": len(samples),
        "mean_adjusted_edge_jaccard": float(
            np.mean([float(row["adj_edge_jaccard"]) for row in samples])
        ),
        "mean_raw_edge_jaccard": float(
            np.mean([float(row["edge_jaccard"]) for row in samples])
        ),
        "mean_node_recall": float(np.mean([float(row["node_recall"]) for row in samples])),
        "predicted_nodes": int(predicted),
        "estimated_nodes": int(estimated),
        "aggregate_node_ratio": (predicted - estimated) / estimated,
    }


def _policy(rows: Sequence[Mapping[str, Any]], *, recall_tolerance: float) -> dict[str, Any]:
    baseline = next((row for row in rows if math.isclose(float(row["threshold"]), 0.970)), None)
    if baseline is None:
        raise NodeThresholdError("threshold grid must include the frozen 0.970 baseline")
    recall_floor = float(baseline["mean_node_recall"]) - recall_tolerance
    eligible = [
        row
        for row in rows
        if float(row["worst_node_recall_delta"]) >= -recall_tolerance - 1.0e-12
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["mean_adjusted_edge_jaccard"]),
            -float(row["threshold"]),
        ),
    )
    return {
        "selection_rule": (
            "maximize-fit-mean-adjusted-edge-subject-to-every-video-node-recall-loss<=0.002"
        ),
        "baseline_threshold": 0.970,
        "baseline_mean_node_recall": float(baseline["mean_node_recall"]),
        "recall_floor": recall_floor,
        "selected_threshold": float(selected["threshold"]),
        "selected_mean_adjusted_edge_jaccard": float(selected["mean_adjusted_edge_jaccard"]),
        "selected_mean_node_recall": float(selected["mean_node_recall"]),
        "selected_aggregate_node_ratio": float(selected["aggregate_node_ratio"]),
        "selected_worst_node_recall_delta": float(selected["worst_node_recall_delta"]),
    }


def summarize_samples(
    samples: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    *,
    recall_tolerance: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    aggregate: dict[str, list[dict[str, Any]]] = {}
    policy: dict[str, Any] = {}
    for scenario in ("44b6", "6bba", "all"):
        threshold_rows: list[dict[str, Any]] = []
        selected_samples = [
            sample for sample in samples if scenario == "all" or sample["scenario"] == scenario
        ]
        for index, threshold in enumerate(thresholds):
            current = [sample["thresholds"][index] for sample in selected_samples]
            baseline = [sample["thresholds"][0] for sample in selected_samples]
            recall_deltas = [
                float(row["node_recall"]) - float(base["node_recall"])
                for row, base in zip(current, baseline, strict=True)
            ]
            row = _aggregate(current)
            threshold_rows.append(
                {
                    "threshold": threshold,
                    **row,
                    "worst_node_recall_delta": min(recall_deltas),
                    "videos_exceeding_recall_loss": sum(
                        delta < -recall_tolerance - 1.0e-12 for delta in recall_deltas
                    ),
                }
            )
        aggregate[scenario] = threshold_rows
        policy[scenario] = _policy(threshold_rows, recall_tolerance=recall_tolerance)
    return aggregate, policy


def harden_report(
    report: Mapping[str, Any], *, recall_tolerance: float
) -> dict[str, Any]:
    scope = report.get("scope")
    method = report.get("method")
    samples = report.get("samples")
    if (
        report.get("schema") != "celltrack-alpha.fit-node-threshold.v1"
        or not isinstance(scope, Mapping)
        or scope.get("fit_only") is not True
        or any(bool(scope.get(key)) for key in ("held_out_used", "audit_used", "test_used"))
        or not isinstance(method, Mapping)
        or not isinstance(method.get("thresholds"), list)
        or not isinstance(samples, list)
        or not samples
    ):
        raise NodeThresholdError("reused threshold report is not the exact fit-only v1 artifact")
    thresholds = tuple(float(value) for value in method["thresholds"])
    aggregate, policy = summarize_samples(
        samples, thresholds, recall_tolerance=recall_tolerance
    )
    return {
        **dict(report),
        "schema": "celltrack-alpha.fit-node-threshold.v2",
        "method": {**dict(method), "recall_tolerance": recall_tolerance},
        "policy": policy,
        "aggregate": aggregate,
    }


def analyze(
    zip_path: Path,
    plan_path: Path,
    artifact_root: Path,
    support_root: Path,
    thresholds: Sequence[float],
    *,
    recall_tolerance: float,
) -> dict[str, Any]:
    if (
        not thresholds
        or tuple(sorted(set(float(value) for value in thresholds))) != tuple(thresholds)
        or not math.isclose(float(thresholds[0]), 0.970)
        or any(not 0.970 <= float(value) < 1.0 for value in thresholds)
    ):
        raise NodeThresholdError("thresholds must be unique, sorted, and start at 0.970")
    package = _load(plan_path)
    plan = package.get("plan")
    if not isinstance(plan, Mapping):
        raise NodeThresholdError("fit-cache package has no plan")
    boundary = plan.get("data_boundary")
    fit_ids = plan.get("already_complete_video_ids")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("train_fit_only") is not True
        or any(bool(boundary.get(key)) for key in ("held_out_used", "audit_used", "test_used"))
        or not isinstance(fit_ids, list)
        or len(fit_ids) != 14
        or len(set(fit_ids)) != len(fit_ids)
    ):
        raise NodeThresholdError("available cache selection is not the exact fit-only 14-video set")

    bundle = validate_artifact_bundle(support_root)
    expected_identity = CacheIdentity.from_artifact(bundle, pair_batch_size=1, view_batch_size=1)
    samples: list[dict[str, Any]] = []
    with ZipStore(zip_path, mode="r") as store:
        for identity in sorted(str(value) for value in fit_ids):
            dataset = identity.split(":", 1)[1]
            slug = identity.replace(":", "__", 1)
            cache_path = artifact_root / "cache" / f"{slug}.bhc"
            cache = load_sparse_cache(cache_path, expected_identity=expected_identity)
            graph_nodes = run_bh.detections_to_graph_nodes(cache.detections, bh_graph.GraphNode)
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            per_threshold: list[dict[str, Any]] = []
            for threshold in thresholds:
                filtered = tuple(node for node in graph_nodes if float(node.score) >= threshold)
                result = bh_graph.calibrate_v45_graph(
                    filtered, topology_mode="legacy", division_enabled=True
                )
                prediction_nodes, prediction_edges = _rows(result)
                metrics = official_edge_metric_row(
                    prediction_nodes, prediction_edges, gt, scale
                )
                per_threshold.append(
                    {
                        "threshold": threshold,
                        "input_nodes": len(filtered),
                        "output_nodes": len(result.nodes),
                        "output_edges": len(result.edges),
                        "division_added": int(result.stats["division_added"]),
                        **{
                            key: metrics[key]
                            for key in (
                                "edge_jaccard",
                                "adj_edge_jaccard",
                                "node_recall",
                                "total_node_ratio",
                                "estimated_number_of_nodes",
                                "num_pred_nodes",
                            )
                        },
                    }
                )
            samples.append(
                {
                    "identity": identity,
                    "scenario": _scenario(identity),
                    "cache": {"path": str(cache_path), "sha256": _sha256(cache_path)},
                    "thresholds": per_threshold,
                }
            )

    aggregate, policy = summarize_samples(
        samples, thresholds, recall_tolerance=recall_tolerance
    )

    return {
        "schema": "celltrack-alpha.fit-node-threshold.v2",
        "status": "completed",
        "scope": {
            "video_ids": sorted(str(value) for value in fit_ids),
            "video_count": len(fit_ids),
            "fit_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
            "provisional_until_full_117_fit": True,
        },
        "method": {
            "thresholds": list(thresholds),
            "association": "inherited-v45-motion-legacy",
            "divisions": "inherited-active",
            "image_refinement": False,
            "recall_tolerance": recall_tolerance,
        },
        "policy": policy,
        "aggregate": aggregate,
        "inputs": {
            "zip": {"path": str(zip_path), "bytes": zip_path.stat().st_size},
            "fit_plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
            "artifact_root": str(artifact_root),
            "support_weight_sha256": bundle.hashes.weight_sha256,
        },
        "samples": samples,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip", type=Path, default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
    )
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "kaggle_kernel/fitcache/plan.json"
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=ROOT / "kaggle_kernel/bh9o7/bh9out"
    )
    parser.add_argument(
        "--support-root", type=Path, default=ROOT / "external/Tracking_Support"
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--recall-tolerance", type=float, default=0.002)
    parser.add_argument(
        "--reuse-report",
        type=Path,
        help="Recompute a safe v2 policy from a completed fit-only v1 grid without replaying graphs.",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/g02/fit_node_threshold_14_v2.json"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise NodeThresholdError(f"refusing to overwrite output: {args.output}")
    if args.reuse_report is None:
        report = analyze(
            args.zip,
            args.plan,
            args.artifact_root,
            args.support_root,
            tuple(args.thresholds),
            recall_tolerance=args.recall_tolerance,
        )
    else:
        report = harden_report(
            _load(args.reuse_report), recall_tolerance=args.recall_tolerance
        )
        report["reused_grid"] = {
            "path": str(args.reuse_report),
            "sha256": _sha256(args.reuse_report),
            "graph_replay_repeated": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        json.dumps(report, allow_nan=False, sort_keys=True).encode("ascii") + b"\n"
    )
    print(json.dumps(report["policy"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
