#!/usr/bin/env python3
"""Prepare fit-only rescue/control pairs from actual frozen graph selections."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from celltrack_alpha import bh_edge, bh_graph, bh_model  # noqa: E402
from celltrack_alpha.edge_data import (  # noqa: E402
    match_detections_to_gt,
    selected_graph_conflict_examples_from_video,
)
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SPLIT_REPORT,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
    _cache_identity,
    _load_base_scorer,
    _load_json,
    _sha256,
)


SCHEMA = "celltrack-alpha.selected-graph-conflict-prepare.v1"
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_conflict_prepare_v1"


class SelectedGraphPrepareError(RuntimeError):
    """The selected-graph preparation contract was violated."""


def _selected_edges(
    cache: bh_model.SparseInferenceCache, scorer: Any
) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    probabilities, no_link, association = run_bh.score_sparse_edge_cache(
        cache, scorer, edge_module=bh_edge
    )
    graph_nodes = run_bh.detections_to_graph_nodes(cache.detections, bh_graph.GraphNode)
    graph = bh_graph.calibrate_v45_graph(
        graph_nodes,
        learned_edge_probabilities=probabilities,
        source_no_link_probabilities=no_link,
        topology_mode="legacy",
        division_enabled=False,
    )
    if (
        graph.provenance.get("motion_direction") != "backward"
        or graph.provenance.get("motion_error_weight") != 0.25
        or graph.provenance.get("division_enabled") is not False
        or graph.provenance.get("topology_mode") != "legacy"
    ):
        raise SelectedGraphPrepareError("frozen baseline graph identity drifted")
    raw_ids = {
        int(node_id)
        for detection in cache.detections
        for node_id in detection.node_ids
    }
    selected = {
        (int(edge.source_id), int(edge.target_id))
        for edge in graph.edges
        if int(edge.source_id) in raw_ids and int(edge.target_id) in raw_ids
    }
    outdegree: dict[int, int] = {}
    indegree: dict[int, int] = {}
    for source, target in selected:
        outdegree[source] = outdegree.get(source, 0) + 1
        indegree[target] = indegree.get(target, 0) + 1
    if max(outdegree.values(), default=0) > 1 or max(indegree.values(), default=0) > 1:
        raise SelectedGraphPrepareError("division-disabled graph contains a branch or merge")
    return selected, {
        "raw_selected_edges": len(selected),
        "output_nodes": len(graph.nodes),
        "output_edges": len(graph.edges),
        "association": association,
        "provenance": {
            "topology_mode": graph.provenance["topology_mode"],
            "division_enabled": graph.provenance["division_enabled"],
            "motion_direction": graph.provenance["motion_direction"],
            "motion_error_weight": graph.provenance["motion_error_weight"],
            "learned_probabilities": graph.provenance["learned_probabilities"],
            "source_no_link_probabilities": graph.provenance[
                "source_no_link_probabilities"
            ],
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise SelectedGraphPrepareError(f"refusing to overwrite output: {args.output}")
    split_report = _load_json(args.split_report)
    scope = split_report.get("scope")
    if (
        not isinstance(scope, Mapping)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
        or scope.get("held_out_used") is not False
        or scope.get("audit_used") is not False
        or scope.get("test_used") is not False
    ):
        raise SelectedGraphPrepareError("parent scope is not exact fit-only 105/12")
    train_ids = tuple(str(value) for value in scope["train_video_ids"])
    validation_ids = tuple(str(value) for value in scope["validation_video_ids"])
    scorer, _model, edge_record = _load_base_scorer(
        args.edge_artifact, args.edge_auxiliary, args.support_root, "cpu"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    examples_root = args.output / "examples"
    examples_root.mkdir()
    video_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate((*train_ids, *validation_ids), start=1):
            dataset = identity.split(":", 1)[1]
            slug = identity.replace(":", "__", 1)
            cache = bh_model.load_sparse_cache(
                args.cache / f"{slug}.bhc",
                expected_identity=_cache_identity(args.cache / f"{slug}.json"),
            )
            video = bh_model.VideoInference(
                cache.detections,
                cache.contexts,
                pair_batch_size=cache.identity.pair_batch_size,
                view_batch_size=cache.identity.view_batch_size,
            )
            gt = read_geff_arrays(store, "train", dataset)
            matches, gt_edges = match_detections_to_gt(video.detections, gt)
            selected, graph_report = _selected_edges(cache, scorer)
            examples = selected_graph_conflict_examples_from_video(
                scorer, video, matches, gt_edges, selected
            )
            destination = examples_root / f"{slug}.pt"
            torch.save(
                {
                    "challenger_features": examples.challenger_features,
                    "selected_features": examples.selected_features,
                    "challenger_logits": examples.challenger_logits,
                    "selected_logits": examples.selected_logits,
                    "challenger_geometry_um": examples.challenger_geometry_um,
                    "selected_geometry_um": examples.selected_geometry_um,
                    "conflict_kind": examples.conflict_kind,
                    "labels": examples.labels,
                    "challenger_pairs": examples.challenger_pairs,
                    "selected_pairs": examples.selected_pairs,
                },
                destination,
            )
            row = {
                "identity": identity,
                "partition": "train" if identity in train_ids else "validation",
                "path": str(destination.relative_to(ROOT)),
                "sha256": _sha256(destination),
                "matched_detections": sum(value is not None for value in matches.values()),
                "candidate_count": examples.candidate_count,
                "reachable_miss_count": examples.reachable_miss_count,
                "rescue_count": examples.rescue_count,
                "control_count": examples.control_count,
                "source_conflict_count": examples.source_conflict_count,
                "target_conflict_count": examples.target_conflict_count,
                "skipped_division_truth": examples.skipped_division_truth,
                "skipped_without_candidate_conflict": examples.skipped_without_candidate_conflict,
                "candidate_topology_sha256": examples.candidate_topology_sha256,
                "graph": graph_report,
            }
            video_reports.append(row)
            print(
                f"selected graph prepare {index:03d}/117 {identity} "
                f"rescue={examples.rescue_count} control={examples.control_count}",
                flush=True,
            )
    totals = {
        name: sum(int(row[name]) for row in video_reports)
        for name in (
            "matched_detections",
            "candidate_count",
            "reachable_miss_count",
            "rescue_count",
            "control_count",
            "source_conflict_count",
            "target_conflict_count",
            "skipped_division_truth",
            "skipped_without_candidate_conflict",
        )
    }
    result = {
        "schema": SCHEMA,
        "status": "prepared",
        "scope": {
            "train_video_ids": list(train_ids),
            "validation_video_ids": list(validation_ids),
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "mechanism": {
            "winner": "actual frozen c=0.25 legacy graph selection",
            "rescues": "reachable one-to-one GT misses blocked at a selected source or target",
            "controls": "GT-correct selected edges versus fully matched non-edge challengers",
            "candidate_gate": "physical-cKDTree-10um-cap8",
            "unknown_policy": "never use unmatched candidates as controls",
            "division_policy": "exclude multi-edge GT motifs",
            "gnn": False,
        },
        "frozen_pairwise": edge_record,
        "cache_identity": asdict(
            _cache_identity(args.cache / f"{train_ids[0].replace(':', '__', 1)}.json")
        ),
        "videos": video_reports,
        "totals": totals,
        "inputs": {
            "split_report": str(args.split_report),
            "split_report_sha256": _sha256(args.split_report),
            "edge_artifact": str(args.edge_artifact),
            "edge_artifact_sha256": _sha256(args.edge_artifact),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    run_bh.atomic_create_json(args.output / "prepare.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--split-report", type=Path, default=DEFAULT_SPLIT_REPORT)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = prepare(parser().parse_args())
    print(json.dumps({"status": report["status"], "totals": report["totals"]}, sort_keys=True))
