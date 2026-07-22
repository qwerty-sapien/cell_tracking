#!/usr/bin/env python3
"""Reconstruct E1 distance ranks for the frozen prefix-52 / 336-FN cohort."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[3]
for location in (ROOT / "src", ROOT / "scripts", ROOT / "external/kaggle-cell-tracking-competition/src"):
    sys.path.insert(0, str(location))

import diagnose_phase1_distance_motion_gate as diag
import scoreall
from celltrack_baseline.rule_based import read_geff_arrays


SCORES = ROOT / "reports/g02/source_softmax_edge_f0_fit117_holdout_prefix52_v1/scores.json"
FRAGMENTATION = ROOT / "reports/g02/source_softmax_edge_f0_fit117_holdout_complete57_v2/fragmentation.json"
PARTS = ROOT / "reports/g02/source_softmax_edge_f0_fit117_holdout_parts_v1"
CACHE = ROOT / "kaggle_kernel/bhcv57out_v3/cache"
ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
OUTPUT = Path(__file__).with_name("report.json")


def fragmentation_matches(graph, gt, scale):
    """Mirror analyze_holdout_fragmentation.py's scorer-ID interpretation."""
    nodes, edges = diag.prediction_rows(graph)
    prediction = diag.prediction_to_graph(nodes, edges)
    truth = diag.geff_arrays_to_graph(dict(gt))
    with diag.warnings.catch_warnings():
        diag.warnings.simplefilter("ignore")
        diag._evaluate(prediction, truth, "jaccard", scale, 7.0)
    ordered_pred_ids = [
        int(row["node_id"])
        for row in sorted(nodes, key=lambda row: (int(row["t"]), int(row["node_id"])))
    ]
    pred_to_gt = {}
    gt_to_pred = {}
    matched = prediction.node_attrs(
        attr_keys=[diag.td.DEFAULT_ATTR_KEYS.NODE_ID, diag.td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    for pred_internal, gt_internal in matched.iter_rows():
        if gt_internal is None or int(gt_internal) == -1:
            continue
        pred_id = ordered_pred_ids[int(pred_internal)]
        gt_id = int(gt_internal)
        pred_to_gt[pred_id] = gt_id
        assert gt_id not in gt_to_pred
        gt_to_pred[gt_id] = pred_id
    return pred_to_gt, gt_to_pred


def main() -> None:
    started = time.perf_counter()
    scores = json.loads(SCORES.read_text())
    fragmentation = json.loads(FRAGMENTATION.read_text())
    sample_rows = scores["samples"]
    identities = [str(row["identity"]) for row in sample_rows]
    assert len(identities) == 52
    assert {
        key: int(sum(float(row[key]) for row in sample_rows))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    } == {"edge_tp": 29676, "edge_fp": 2660, "edge_fn": 1810}

    long_breaks = [
        row for row in fragmentation["breaks"]
        if row["prefix_52"] and row["cause"] == "final_pair_geometry_above_gate_radius"
    ]
    assert len(long_breaks) == 336
    long_keys = {
        (str(row["identity"]), int(row["gt_source_id"]), int(row["gt_target_id"]))
        for row in long_breaks
    }
    assert len(long_keys) == 336

    input_count_by_identity = {
        str(row["identity"]): int(row["input_node_count"])
        for row in fragmentation["per_video"]
        if row["prefix_52"]
    }
    all_events = []
    per_video = []
    with ZipStore(ARCHIVE, mode="r") as store:
        for index, row in enumerate(sample_rows, 1):
            identity = str(row["identity"])
            _, dataset, _ = scoreall.parse_identity(identity)
            fragment = PARTS / identity.replace(":", "__") / "frags" / f"{identity.replace(':', '__')}.csv"
            assert diag.sha256_file(fragment) == row["source"]["sha256"]
            graph = scoreall._fragment_graph(fragment, identity)
            gt = read_geff_arrays(store, "train", dataset)
            scale = tuple(float(value) for value in row["scale_zyx"])
            pred_to_gt, gt_to_pred = fragmentation_matches(graph, gt, scale)
            truth = diag.geff_arrays_to_graph(dict(gt))
            gt_edges = {
                (int(source), int(target))
                for source, target in truth.edge_attrs().select("source_id", "target_id").iter_rows()
            }
            accounting = diag.edge_accounting(graph, gt_edges, pred_to_gt, gt_to_pred)
            assert accounting == {
                key: int(float(row[key])) for key in ("edge_tp", "edge_fp", "edge_fn")
            }
            input_count = input_count_by_identity[identity]
            by_frame = {}
            for node_id, node in graph.nodes.items():
                if int(node_id) < input_count:
                    by_frame.setdefault(int(node.t), []).append(int(node_id))
            events = []
            for gt_source, gt_target in sorted(gt_edges):
                source_id = gt_to_pred.get(gt_source)
                target_id = gt_to_pred.get(gt_target)
                if source_id is None or target_id is None:
                    continue
                if source_id >= input_count or target_id >= input_count:
                    continue
                source = graph.nodes[source_id]
                target = graph.nodes[target_id]
                if int(target.t) != int(source.t) + 1:
                    continue
                target_ids = by_frame.get(int(target.t), [])
                target_coords = [graph.nodes[node_id].coord_zyx for node_id in target_ids]
                rank, distance_um = diag.distance_rank(
                    source.coord_zyx, target_id, target_ids, target_coords, scale
                )
                events.append({
                    "identity": identity,
                    "gt_source_id": int(gt_source),
                    "gt_target_id": int(gt_target),
                    "distance_rank": rank,
                    "distance_um": distance_um,
                })
            video_row = {
                "identity": identity,
                **accounting,
                "eligible_gt_edges": len(events),
                "ineligible_gt_edges": len(gt_edges) - len(events),
            }
            per_video.append(video_row)
            all_events.extend(events)
            print(f"{index:02d}/52 {identity} events={len(events)}", flush=True)

    event_by_key = {
        (str(row["identity"]), int(row["gt_source_id"]), int(row["gt_target_id"])): row
        for row in all_events
    }
    observed_long = [event_by_key[key] for key in long_keys if key in event_by_key]
    absent_long = len(long_keys) - len(observed_long)
    overall_denominator = sum(int(row["edge_tp"]) + int(row["edge_fn"]) for row in per_video)
    overall_absent = overall_denominator - len(all_events)
    overall = diag.rank_summary(
        [int(row["distance_rank"]) for row in all_events], candidate_absent=overall_absent
    )
    long_summary = diag.rank_summary(
        [int(row["distance_rank"]) for row in observed_long], candidate_absent=absent_long
    )
    passed = overall["rank_le_10_fraction"] >= 0.90 and long_summary["rank_le_10_fraction"] >= 0.90
    long_distance = diag.quantiles([float(row["distance_um"]) for row in observed_long])
    long_over_10 = sum(float(row["distance_um"]) > 10.0 + diag.DISTANCE_EPSILON_UM for row in observed_long)
    report = {
        "schema": "celltrack-alpha.e1-distance-rank-prefix52.v1",
        "status": "completed_read_only_cpu_diagnostic",
        "decision": {
            "rule": "PASS iff rank<=10 is >=90% for all official GT edges and the frozen 336 final-geometry-above-10um FNs",
            "pass": passed,
        },
        "scope": {
            "holdout_manifest_video_count": 57,
            "scored_prefix_video_count": 52,
            "adjusted_edge_jaccard": float(scores["summary"]["adj_edge_jaccard"]),
            "edge_tp": 29676,
            "edge_fp": 2660,
            "edge_fn": 1810,
            "official_gt_edges": overall_denominator,
            "division_enabled_in_cohort_artifact": True,
            "division_tp": int(sum(float(row["division_tp"]) for row in sample_rows)),
            "division_fp": int(sum(float(row["division_fp"]) for row in sample_rows)),
            "division_fn": int(sum(float(row["division_fn"]) for row in sample_rows)),
        },
        "all_official_gt_edges": overall,
        "frozen_336_final_geometry_above_10um_fns": {
            **long_summary,
            "final_distance_over_10um": long_over_10,
            "final_distance_over_10um_fraction": long_over_10 / 336,
            "final_distance_um": long_distance,
        },
        "reproducibility_caveat": (
            "The named 336 cohort and these ranks use final matched CSV node geometry. The newer run did not persist "
            "pre-prune candidate rows or a compatible raw FrameDet cache (the available bhcv57out_v3 cache has node-ID/time "
            "mismatches against these fragments), so the exact pre-gate raw candidate rank cannot be reconstructed. Ranks "
            "are among surviving non-synthetic next-frame output nodes and are therefore an optimistic upper bound on raw "
            "top-k recall. Candidate-absent, synthetic, or nonadjacent GT endpoints count as rank>10 failures."
        ),
        "provenance": {
            "scores": {"path": str(SCORES), "sha256": diag.sha256_file(SCORES)},
            "fragmentation": {"path": str(FRAGMENTATION), "sha256": diag.sha256_file(FRAGMENTATION)},
            "incompatible_older_cache_directory_checked": str(CACHE),
            "fragment_parts_directory": str(PARTS),
            "analysis_source": {"path": str(Path(__file__)), "sha256": diag.sha256_file(Path(__file__))},
        },
        "execution": {
            "cpu_only": True,
            "network": False,
            "kaggle_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
