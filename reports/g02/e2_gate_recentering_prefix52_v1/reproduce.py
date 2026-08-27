#!/usr/bin/env python3
"""Reproduce the E2 motion-recentring diagnostic on the exact 336-event cohort.

The frozen source-softmax run did not persist its pre-graph FrameDet tensors.
Consequently this script applies the exact production prediction formula and
selected adjacent-parent topology to the persisted postprocessed fragment
coordinates.  The report labels that coordinate-stage limitation explicitly.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha import bh_graph  # noqa: E402


FRAGMENTATION = (
    ROOT
    / "reports/g02/source_softmax_edge_f0_fit117_holdout_complete57_v2/fragmentation.json"
)
SCORES = ROOT / "reports/g02/source_softmax_edge_f0_fit117_holdout_prefix52_v1/scores.json"
EPSILON = 1.0e-12


def quantiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def main() -> int:
    fragmentation = json.loads(FRAGMENTATION.read_text(encoding="utf-8"))
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    events = [
        row
        for row in fragmentation["breaks"]
        if row["prefix_52"]
        and row["cause"] == "final_pair_geometry_above_gate_radius"
    ]
    edge_totals = fragmentation["prefix_52"]["edge_totals"]
    if edge_totals != {"edge_fn": 1810, "edge_fp": 2660, "edge_tp": 29676}:
        raise RuntimeError(f"frozen edge totals changed: {edge_totals}")
    if len(events) != 336:
        raise RuntimeError(f"expected 336 above-gate events, found {len(events)}")
    if bh_graph.V45_GRAPH_CONFIG.velocity_weight != 0.5:
        raise RuntimeError("production velocity weight changed")

    sample_by_identity = {row["identity"]: row for row in scores["samples"]}
    events_by_identity: dict[str, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        events_by_identity[str(event["identity"])].append(event)

    distances: list[float] = []
    actionable_distances: list[float] = []
    history_classes: Counter[str] = Counter()
    family_distances: dict[str, list[float]] = defaultdict(list)
    for identity, video_events in sorted(events_by_identity.items()):
        sample = sample_by_identity[identity]
        fragment_path = ROOT / sample["source"]["path"]
        report_path = fragment_path.parents[1] / "report.json"
        run_report = json.loads(report_path.read_text(encoding="utf-8"))
        input_node_count = int(
            run_report["results"][0]["graph"]["stats"]["input_node_count"]
        )
        wanted_sources = {int(row["pred_source_id"]) for row in video_events}
        coords: dict[int, np.ndarray] = {}
        times: dict[int, int] = {}
        incoming: dict[int, list[int]] = defaultdict(list)
        outgoing: dict[int, list[int]] = defaultdict(list)
        edge_buffer: list[tuple[int, int]] = []
        with fragment_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["row_type"] == "node":
                    node_id = int(row["node_id"])
                    coords[node_id] = np.asarray(
                        [float(row["z"]), float(row["y"]), float(row["x"])],
                        dtype=np.float64,
                    )
                    times[node_id] = int(row["t"])
                else:
                    source_id = int(row["source_id"])
                    target_id = int(row["target_id"])
                    outgoing[source_id].append(target_id)
                    if target_id in wanted_sources:
                        edge_buffer.append((source_id, target_id))
        for source_id, target_id in edge_buffer:
            if times[target_id] == times[source_id] + 1:
                incoming[target_id].append(source_id)

        scale = np.asarray(sample["scale_zyx"], dtype=np.float64)
        family = identity.split(":", 1)[1].split("_", 1)[0]
        for event in video_events:
            source_id = int(event["pred_source_id"])
            target_id = int(event["pred_target_id"])
            parents = incoming.get(source_id, [])
            source = coords[source_id]
            target = coords[target_id]
            if len(parents) == 0:
                predicted = source
                history_class = "none"
            elif len(parents) == 1:
                parent_id = parents[0]
                predicted = source + 0.5 * (source - coords[parent_id])
                if parent_id >= input_node_count:
                    history_class = "synthetic_parent_post_adjacent"
                elif len(outgoing[parent_id]) >= 2:
                    history_class = "branch_parent_ambiguous_added_edge"
                else:
                    history_class = "plain_adjacent_parent"
            else:
                raise RuntimeError(f"multiple adjacent parents for {identity}:{source_id}")
            distance = float(np.linalg.norm((predicted - target) * scale))
            distances.append(distance)
            family_distances[family].append(distance)
            history_classes[history_class] += 1
            if not event["pred_source_out_targets"]:
                actionable_distances.append(distance)

    within = sum(value <= 10.0 + EPSILON for value in distances)
    actionable_within = sum(
        value <= 10.0 + EPSILON for value in actionable_distances
    )
    report = {
        "schema": "celltrack-alpha.e2-gate-recentring-prefix52.v1",
        "status": "completed_read_only_cpu_diagnostic",
        "cohort": {
            "edge_totals": edge_totals,
            "above_10um_fn_events": len(events),
            "video_count_in_frozen_scope": fragmentation["scope"]["video_count"],
            "videos_with_above_gate_events": len(events_by_identity),
        },
        "prediction_contract": {
            "formula": "source if no adjacent parent, else source + 0.5 * (source - parent)",
            "velocity_weight": bh_graph.V45_GRAPH_CONFIG.velocity_weight,
            "gate_query_center": "source_physical (prediction is not used by _radius_pairs)",
            "history_classes_from_final_topology": dict(sorted(history_classes.items())),
        },
        "coordinate_stage": {
            "used": "persisted postprocessed source-softmax fragment coordinates",
            "production_cost_stage": "pre-gap, pre-division, pre-pruning, pre-smoothing FrameDet coordinates",
            "limitation": (
                "The source-softmax run recorded cache=disabled and did not persist FrameDet tensors. "
                "Thus the 336-event cohort and topology are exact, but these distances are a "
                "postprocessed-coordinate replay, not a byte-exact raw-coordinate replay."
            ),
        },
        "result": {
            "within_10um": within,
            "denominator": len(distances),
            "fraction": within / len(distances),
            "distance_um": quantiles(distances),
            "threshold_sensitivity": {
                str(radius): {
                    "count": sum(value <= radius + EPSILON for value in distances),
                    "fraction": sum(value <= radius + EPSILON for value in distances)
                    / len(distances),
                }
                for radius in (6.0, 8.0, 10.0, 12.0, 14.0)
            },
            "unmatched_source_only": {
                "within_10um": actionable_within,
                "denominator": len(actionable_distances),
                "within_actionable_fraction": actionable_within
                / len(actionable_distances),
                "recoverable_share_of_full_cohort": actionable_within
                / len(distances),
            },
            "families": {
                family: {
                    "within_10um": sum(value <= 10.0 + EPSILON for value in values),
                    "denominator": len(values),
                    "fraction": sum(value <= 10.0 + EPSILON for value in values)
                    / len(values),
                }
                for family, values in sorted(family_distances.items())
            },
            "pass_at_large_fraction_0p5": within / len(distances) >= 0.5,
            "decision": "FAIL; do not implement the proposed unmatched-source re-gating pass.",
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
