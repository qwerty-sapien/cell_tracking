#!/usr/bin/env python3
"""Fit-only node-threshold sweep for the source-softmax edge artifact.

This analysis deliberately consumes only the frozen fold-0 fit cache corpus.
It replays the fitted sparse association and graph from cached detections, so
it does not perform detector inference and never reads held-out predictions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/kaggle-cell-tracking-competition/src",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
import v45feval  # noqa: E402
from celltrack_alpha import bh_graph  # noqa: E402
from celltrack_alpha.bh_edge import load_edge_artifact  # noqa: E402
from celltrack_alpha.bh_model import (  # noqa: E402
    CacheIdentity,
    FrameDet,
    PairContext,
    load_sparse_cache,
    load_v45_model,
    validate_artifact_bundle,
)
from celltrack_alpha.official_eval import official_edge_metric_row  # noqa: E402
from celltrack_alpha.v45_fast import (  # noqa: E402
    FrozenV45Fast,
    fast_readout,
    load_fast,
)
from celltrack_alpha.t_adapter import TARGET_CONTRACT_SHA256  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.source-softmax-fit-node-threshold.v1"
DEFAULT_THRESHOLDS = (0.970, 0.975, 0.980, 0.985, 0.990, 0.9925, 0.995, 0.9975, 0.999)
EXPECTED_FIT_MANIFEST_SHA256 = "20ef8056bea8afb60f76cf99357c7b62215b2db6c408d78a1c3ba21c41431a33"
EXPECTED_EDGE_SHA256 = "8621a5aa2f2a52894a1b9f14d0f442673eac9eeb221427cce3522f01bc4eb9db"


class ThresholdAnalysisError(RuntimeError):
    """Raised when fit-only provenance or threshold replay is unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ThresholdAnalysisError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ThresholdAnalysisError(f"JSON root is not an object: {path}")
    return value


def _scenario(identity: str) -> str:
    value = identity.split(":", 1)[-1].split("_", 1)[0]
    if value not in {"44b6", "6bba"}:
        raise ThresholdAnalysisError(f"unknown scenario prefix: {identity}")
    return value


def _filter_detection(detection: FrameDet, threshold: float) -> FrameDet:
    keep = np.asarray(detection.scores >= threshold, dtype=bool)
    return FrameDet(
        frame_index=detection.frame_index,
        node_ids=detection.node_ids[keep],
        coords_zyx=detection.coords_zyx[keep],
        scores=detection.scores[keep],
        grid_coords_zyx=detection.grid_coords_zyx[keep],
        grid_shape_zyx=detection.grid_shape_zyx,
    )


def _filter_context(context: PairContext, kept: set[int]) -> PairContext:
    source_keep = np.fromiter(
        (int(node_id) in kept for node_id in context.source_ids),
        dtype=bool,
        count=len(context.source_ids),
    )
    target_keep = np.fromiter(
        (int(node_id) in kept for node_id in context.target_ids),
        dtype=bool,
        count=len(context.target_ids),
    )
    return PairContext(
        source_frame=context.source_frame,
        target_frame=context.target_frame,
        source_ids=context.source_ids[source_keep],
        target_ids=context.target_ids[target_keep],
        source_feat=context.source_feat[source_keep],
        target_feat=context.target_feat[target_keep],
        grid_shape_zyx=context.grid_shape_zyx,
    )


def _filtered_video(cache: Any, threshold: float) -> SimpleNamespace:
    detections = tuple(_filter_detection(item, threshold) for item in cache.detections)
    kept = {
        int(node_id)
        for detection in detections
        for node_id in detection.node_ids
    }
    contexts = tuple(_filter_context(item, kept) for item in cache.contexts)
    return SimpleNamespace(detections=detections, contexts=contexts)


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


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ThresholdAnalysisError("cannot aggregate an empty threshold slice")
    weights = [
        float(row["edge_tp"]) + float(row["edge_fp"]) + float(row["edge_fn"])
        for row in rows
    ]
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ThresholdAnalysisError("fit threshold slice has zero edge-event weight")
    estimated = sum(float(row["estimated_number_of_nodes"]) for row in rows)
    predicted = sum(float(row["num_pred_nodes"]) for row in rows)
    return {
        "videos": len(rows),
        "edge_event_weight": total_weight,
        "weighted_adjusted_edge_jaccard": sum(
            weight * float(row["adj_edge_jaccard"])
            for weight, row in zip(weights, rows, strict=True)
        ) / total_weight,
        "weighted_raw_edge_jaccard": sum(
            weight * float(row["edge_jaccard"])
            for weight, row in zip(weights, rows, strict=True)
        ) / total_weight,
        "mean_node_recall": float(np.mean([float(row["node_recall"]) for row in rows])),
        "worst_node_recall": min(float(row["node_recall"]) for row in rows),
        "predicted_nodes": int(predicted),
        "estimated_nodes": int(estimated),
        "aggregate_node_ratio": (predicted - estimated) / estimated,
    }


def _summarize(
    samples: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    *,
    recall_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate: dict[str, Any] = {}
    policy: dict[str, Any] = {}
    for scenario in ("44b6", "6bba", "all"):
        chosen = [
            sample for sample in samples
            if scenario == "all" or sample["scenario"] == scenario
        ]
        if not chosen:
            aggregate[scenario] = []
            policy[scenario] = None
            continue
        rows = []
        for index, threshold in enumerate(thresholds):
            current = [sample["thresholds"][index] for sample in chosen]
            baseline = [sample["thresholds"][0] for sample in chosen]
            recall_deltas = [
                float(row["node_recall"]) - float(base["node_recall"])
                for row, base in zip(current, baseline, strict=True)
            ]
            rows.append(
                {
                    "threshold": threshold,
                    **_aggregate(current),
                    "worst_node_recall_delta": min(recall_deltas),
                    "videos_exceeding_recall_loss": sum(
                        delta < -recall_tolerance - 1.0e-12
                        for delta in recall_deltas
                    ),
                }
            )
        aggregate[scenario] = rows
        unguarded = max(
            rows,
            key=lambda row: (
                float(row["weighted_adjusted_edge_jaccard"]),
                -float(row["threshold"]),
            ),
        )
        eligible = [
            row
            for row in rows
            if float(row["worst_node_recall_delta"])
            >= -recall_tolerance - 1.0e-12
        ]
        selected = max(
            eligible,
            key=lambda row: (
                float(row["weighted_adjusted_edge_jaccard"]),
                -float(row["threshold"]),
            ),
        )
        baseline = rows[0]
        policy[scenario] = {
            "selection_rule": (
                "maximize-fit-edge-event-weighted-adjusted-edge-jaccard-subject-to-"
                f"every-video-node-recall-loss<={recall_tolerance}"
            ),
            "recall_tolerance": recall_tolerance,
            "baseline_threshold": float(baseline["threshold"]),
            "baseline_weighted_adjusted_edge_jaccard": float(
                baseline["weighted_adjusted_edge_jaccard"]
            ),
            "selected_threshold": float(selected["threshold"]),
            "selected_weighted_adjusted_edge_jaccard": float(
                selected["weighted_adjusted_edge_jaccard"]
            ),
            "fit_delta": float(selected["weighted_adjusted_edge_jaccard"])
            - float(baseline["weighted_adjusted_edge_jaccard"]),
            "selected_aggregate_node_ratio": float(selected["aggregate_node_ratio"]),
            "selected_mean_node_recall": float(selected["mean_node_recall"]),
            "selected_worst_node_recall": float(selected["worst_node_recall"]),
            "selected_worst_node_recall_delta": float(
                selected["worst_node_recall_delta"]
            ),
            "unguarded_optimum_threshold": float(unguarded["threshold"]),
            "unguarded_optimum_weighted_adjusted_edge_jaccard": float(
                unguarded["weighted_adjusted_edge_jaccard"]
            ),
            "unguarded_optimum_worst_node_recall_delta": float(
                unguarded["worst_node_recall_delta"]
            ),
        }
    return aggregate, policy


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = tuple(float(value) for value in args.thresholds)
    if (
        not thresholds
        or thresholds != tuple(sorted(set(thresholds)))
        or not math.isclose(thresholds[0], 0.970)
        or any(not 0.970 <= value < 1.0 for value in thresholds)
    ):
        raise ThresholdAnalysisError(
            "thresholds must be unique, sorted, in [0.970, 1), and start at 0.970"
        )
    if _sha256(args.fit_manifest) != args.fit_manifest_sha256:
        raise ThresholdAnalysisError("fit manifest hash does not match its frozen pin")
    if _sha256(args.edge_artifact) != args.edge_artifact_sha256:
        raise ThresholdAnalysisError("edge artifact hash does not match its frozen pin")
    manifest = _json(args.fit_manifest)
    boundary = manifest.get("data_boundary")
    videos = manifest.get("videos")
    if (
        manifest.get("schema") != "celltrack-alpha.fit-cache-corpus.v1"
        or manifest.get("status") != "validated"
        or manifest.get("fold") != 0
        or not isinstance(boundary, Mapping)
        or boundary.get("train_fit_only") is not True
        or any(bool(boundary.get(key)) for key in ("held_out_used", "audit_used", "test_used"))
        or not isinstance(videos, Mapping)
        or len(videos) != 117
        or set(videos) != set(manifest.get("video_ids", ()))
    ):
        raise ThresholdAnalysisError("fit cache manifest is not the frozen fit-only 117 corpus")
    identities = sorted(str(value) for value in videos)
    if args.limit is not None:
        identities = identities[: args.limit]

    bundle = validate_artifact_bundle(args.support_root)
    expected_cache = CacheIdentity.from_artifact(bundle, pair_batch_size=1, view_batch_size=1)
    core, bundle_from_model = load_v45_model(args.support_root, device="cpu")
    if bundle_from_model.hashes != bundle.hashes:
        raise ThresholdAnalysisError("validated support bundles disagree")
    fast_model = FrozenV45Fast(core).to("cpu").eval()
    fast_ident = load_fast(
        args.checkpoint,
        fast_model,
        bundle,
        expected_sha256=args.checkpoint_sha256,
        target_sha256=TARGET_CONTRACT_SHA256,
    )
    auxiliary = fast_readout(
        fast_model,
        bundle,
        fast_ident,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    scorer, edge_identity = load_edge_artifact(
        args.edge_artifact,
        fast_model.core,
        expected_pretrained_weight_sha256=bundle.hashes.weight_sha256,
        auxiliary=auxiliary,
    )
    fitted_parameter_count = v45feval.freeze_fitted_scorer_for_evaluation(scorer)

    samples: list[dict[str, Any]] = []
    with ZipStore(args.zip, mode="r") as store:
        for ordinal, identity in enumerate(identities, start=1):
            dataset = identity.split(":", 1)[1]
            slug = identity.replace(":", "__", 1)
            cache_path = args.cache_root / f"{slug}.bhc"
            expected_record = videos[identity]
            expected_sha = str(expected_record["files"]["cache_bhc"]["sha256"])
            if _sha256(cache_path) != expected_sha:
                raise ThresholdAnalysisError(f"cache hash mismatch: {identity}")
            cache = load_sparse_cache(cache_path, expected_identity=expected_cache)
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            per_threshold = []
            for threshold in thresholds:
                video = _filtered_video(cache, threshold)
                probabilities, no_link, association_stats = v45feval.sparse_probs(video, scorer)
                graph_nodes = run_bh.detections_to_graph_nodes(
                    video.detections, bh_graph.GraphNode
                )
                result = bh_graph.calibrate_v45_graph(
                    graph_nodes,
                    learned_edge_probabilities=probabilities,
                    source_no_link_probabilities=no_link,
                    topology_mode="legacy",
                    division_enabled=False,
                )
                prediction_nodes, prediction_edges = _rows(result)
                metrics = official_edge_metric_row(
                    prediction_nodes, prediction_edges, gt, scale
                )
                per_threshold.append(
                    {
                        "threshold": threshold,
                        "input_nodes": len(graph_nodes),
                        "output_nodes": len(result.nodes),
                        "output_edges": len(result.edges),
                        "association": association_stats,
                        **{
                            key: metrics[key]
                            for key in (
                                "edge_tp",
                                "edge_fp",
                                "edge_fn",
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
                    "cache": {"path": str(cache_path), "sha256": expected_sha},
                    "thresholds": per_threshold,
                }
            )
            print(f"[{ordinal}/{len(identities)}] {identity}", flush=True)

    aggregate, policy = _summarize(
        samples,
        thresholds,
        recall_tolerance=args.recall_tolerance,
    )
    return {
        "schema": SCHEMA,
        "status": "completed",
        "scope": {
            "fit_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
            "fold": 0,
            "video_count": len(identities),
            "full_fit_117": len(identities) == 117,
            "video_ids": identities,
        },
        "method": {
            "thresholds": list(thresholds),
            "baseline_threshold": 0.970,
            "recall_tolerance": args.recall_tolerance,
            "association": "fit-117-source-softmax-edge-artifact",
            "association_topology": "legacy",
            "division_enabled": False,
            "image_refinement": False,
            "detector_cache": "released-sequential-D4; not one-pass-fast",
            "aggregate_weight": "edge_tp+edge_fp+edge_fn per official sample",
            "transfer_limitation": (
                "threshold chosen on released D4 fit caches; one-pass fast detector "
                "holdout effect requires a post-selection replay"
            ),
        },
        "policy": policy,
        "aggregate": aggregate,
        "inputs": {
            "zip": {"path": str(args.zip), "bytes": args.zip.stat().st_size},
            "fit_manifest": {
                "path": str(args.fit_manifest),
                "sha256": args.fit_manifest_sha256,
            },
            "edge_artifact": {
                "path": str(args.edge_artifact),
                "sha256": args.edge_artifact_sha256,
                "identity": asdict(edge_identity),
                "fitted_parameter_count": fitted_parameter_count,
                "runtime_trainable_parameter_count": scorer.trainable_parameter_count,
            },
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": args.checkpoint_sha256,
                "identity": fast_ident.as_dict(),
            },
            "support_weight_sha256": bundle.hashes.weight_sha256,
        },
        "samples": samples,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    )
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        default=ROOT / "reports/g02/fitcache_117/manifest.json",
    )
    parser.add_argument(
        "--fit-manifest-sha256", default=EXPECTED_FIT_MANIFEST_SHA256
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "reports/g02/fitcache_117/bh9out/cache",
    )
    parser.add_argument(
        "--edge-artifact",
        type=Path,
        default=ROOT / "reports/g02/source_softmax_edge_f0_fit117_v1/edge.json",
    )
    parser.add_argument("--edge-artifact-sha256", default=EXPECTED_EDGE_SHA256)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "reports/g02/v45f4.pt"
    )
    parser.add_argument(
        "--checkpoint-sha256", default=v45feval.DEFAULT_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--support-root", type=Path, default=ROOT / "external/Tracking_Support"
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--recall-tolerance", type=float, default=0.002)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ThresholdAnalysisError("limit must be positive")
    if not math.isfinite(args.recall_tolerance) or args.recall_tolerance < 0:
        raise ThresholdAnalysisError("recall-tolerance must be finite and nonnegative")
    if args.output.exists():
        raise ThresholdAnalysisError(f"refusing to overwrite output: {args.output}")
    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        json.dumps(report, allow_nan=False, sort_keys=True).encode("ascii") + b"\n"
    )
    print(json.dumps(report["policy"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
