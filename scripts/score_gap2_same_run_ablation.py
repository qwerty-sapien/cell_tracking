#!/usr/bin/env python3
"""Score a within-run ablation of the exact public V32 GAP2 output.

The GAP2 implementation allocates two monotonically increasing node IDs for
every accepted bridge and adds the three-edge path ``end -> n1 -> n2 -> start``.
This diagnostic removes precisely those appended nodes and their incident
edges from the completed GAP2 graph, while leaving every other inference and
post-processing output unchanged.  It therefore estimates the direct graph
and node-count effect of GAP2 without confounding an independently stochastic
V32 rerun.  It is not represented as an exact rerun of the pre-GAP2 pipeline:
line-fit smoothing occurred after GAP2 and is intentionally held fixed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
    validate_graphs,
)
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", choices=("cv57", "held_fit105"), default="cv57")
    parser.add_argument(
        "--population-manifest",
        type=Path,
        default=ROOT
        / "reports/g02/screening/tiered_surface/branches/"
        "public_temporalunet_baseline_v1/POPULATIONS.json",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT
        / "reports/g02/screening/tiered_surface/branches/"
        "public_temporalunet_baseline_v1/gap2_probe/output_v1/submission.csv",
    )
    parser.add_argument(
        "--run-stats",
        type=Path,
        default=ROOT
        / "reports/g02/screening/tiered_surface/branches/"
        "public_temporalunet_baseline_v1/gap2_probe/output_v1/run_stats.csv",
    )
    parser.add_argument(
        "--full-score",
        type=Path,
        default=ROOT
        / "reports/g02/screening/tiered_surface/branches/"
        "public_temporalunet_baseline_v1/gap2_probe/CV57_SCORE.json",
        help="Existing exact full-GAP2 score; required for the selected population.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_selected_graphs(path: Path, datasets: set[str]) -> dict[str, SubmissionGraph]:
    graphs = {dataset: SubmissionGraph(dataset, {}, []) for dataset in datasets}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = str(row["dataset"])
            if dataset not in graphs:
                continue
            graph = graphs[dataset]
            if row["row_type"] == "node":
                node = SubmissionNode(
                    node_id=int(row["node_id"]),
                    t=int(row["t"]),
                    z=int(row["z"]),
                    y=int(row["y"]),
                    x=int(row["x"]),
                )
                if node.node_id in graph.nodes:
                    raise ValueError(f"duplicate node {dataset}:{node.node_id}")
                graph.nodes[node.node_id] = node
            elif row["row_type"] == "edge":
                graph.edges.append(SubmissionEdge(int(row["source_id"]), int(row["target_id"])))
            else:
                raise ValueError(f"unexpected row_type {row['row_type']!r}")
    validate_graphs(graphs, max_outdegree=None)
    return graphs


def load_stats(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[str(row["dataset"])] = {
                "pairs": int(row["gap2_pairs_selected"]),
                "added_nodes": int(row["gap2_added_nodes"]),
                "added_edges": int(row["gap2_added_edges"]),
            }
    return result


def infer_gap2_allocated_block(
    graph: SubmissionGraph,
    stats: dict[str, int],
) -> tuple[int | None, int | None, set[int]]:
    """Return the allocated GAP2 ID interval and its surviving node IDs."""

    pairs = stats["pairs"]
    count = stats["added_nodes"]
    if count != 2 * pairs or stats["added_edges"] != 3 * pairs:
        raise ValueError(f"{graph.dataset}: inconsistent GAP2 run counters")
    # GAP2 owns the final allocated ID block, but the later short-component
    # filter can remove an entire two-node bridge.  Consequently, taking the
    # last ``count`` *surviving* IDs can incorrectly pull older GAP1 nodes into
    # the ablation.  Infer the allocated block and authenticate every surviving
    # consecutive pair against its canonical internal edge.
    edge_pairs = {(edge.source_id, edge.target_id) for edge in graph.edges}
    allocated_start: int | None = None
    allocated_end: int | None = None
    if count:
        maximum_surviving_id = max(graph.nodes)
        candidates: list[tuple[int, int]] = []
        for missing_tail in range(0, count + 1, 2):
            end = maximum_surviving_id + missing_tail
            start = end - count + 1
            valid = True
            for first in range(start, end + 1, 2):
                second = first + 1
                first_present = first in graph.nodes
                second_present = second in graph.nodes
                if first_present != second_present:
                    valid = False
                    break
                if first_present:
                    if (first, second) not in edge_pairs:
                        valid = False
                        break
                    if graph.nodes[second].t != graph.nodes[first].t + 1:
                        valid = False
                        break
            if valid and any(node_id in graph.nodes for node_id in range(start, end + 1)):
                candidates.append((start, end))
        if not candidates:
            raise ValueError(
                f"{graph.dataset}: could not infer a valid GAP2 ID block"
            )
        # The final surviving pair is a canonical GAP2 pair.  Select the
        # minimum-tail candidate, i.e. the block ending at that pair.  This is
        # the only irreducible reconstruction assumption when no pre-filter ID
        # snapshot was persisted; it is recorded in the receipt below.
        allocated_start, allocated_end = min(candidates, key=lambda pair: pair[1])
        surviving = {
            node_id
            for node_id in graph.nodes
            if allocated_start <= node_id <= allocated_end
        }
    else:
        surviving = set()
    return allocated_start, allocated_end, surviving


def ablate_gap2(graph: SubmissionGraph, stats: dict[str, int]) -> tuple[SubmissionGraph, dict[str, Any]]:
    pairs = stats["pairs"]
    count = stats["added_nodes"]
    allocated_start, allocated_end, appended = infer_gap2_allocated_block(graph, stats)
    incident = [
        edge
        for edge in graph.edges
        if edge.source_id in appended or edge.target_id in appended
    ]
    internal = [
        edge
        for edge in incident
        if edge.source_id in appended and edge.target_id in appended
    ]
    incoming = [
        edge
        for edge in incident
        if edge.source_id not in appended and edge.target_id in appended
    ]
    outgoing = [
        edge
        for edge in incident
        if edge.source_id in appended and edge.target_id not in appended
    ]
    surviving_pairs = len(appended) // 2
    if not (
        len(appended) == 2 * surviving_pairs
        and len(internal) == surviving_pairs
        and len(incoming) >= surviving_pairs
        and len(outgoing) >= surviving_pairs
        and len(incident) >= 3 * surviving_pairs
    ):
        raise ValueError(
            f"{graph.dataset}: appended GAP2 path topology failed authentication "
            f"(allocated_pairs={pairs}, surviving_pairs={surviving_pairs}, "
            f"nodes={len(appended)}, incident={len(incident)}, "
            f"internal={len(internal)}, incoming={len(incoming)}, outgoing={len(outgoing)})"
        )
    ablated = SubmissionGraph(
        graph.dataset,
        {node_id: node for node_id, node in graph.nodes.items() if node_id not in appended},
        [
            edge
            for edge in graph.edges
            if edge.source_id not in appended and edge.target_id not in appended
        ],
    )
    validate_graphs({graph.dataset: ablated}, max_outdegree=None)
    return ablated, {
        "gap2_pairs_removed": pairs,
        "gap2_surviving_pairs_removed": surviving_pairs,
        "gap2_nodes_removed": len(appended),
        "gap2_edges_removed": len(incident),
        "gap2_allocated_id_start": allocated_start,
        "gap2_allocated_id_end": allocated_end,
        "allocation_tail_assumption": "final_surviving_pair_is_final_allocated_gap2_pair",
        "gap2_nodes_already_removed_downstream": count - len(appended),
        "downstream_incident_edges_removed": len(incident) - 3 * surviving_pairs,
        "minimum_removed_node_id": min(appended) if appended else None,
        "maximum_removed_node_id": max(appended) if appended else None,
    }


def graph_metric_row(graph: SubmissionGraph, gt: dict[str, Any], scale: tuple[float, float, float]) -> dict[str, Any]:
    nodes = [
        {
            "dataset": graph.dataset,
            "node_id": node.node_id,
            "t": node.t,
            "z": node.z,
            "y": node.y,
            "x": node.x,
        }
        for node in graph.nodes.values()
    ]
    edges = [
        {"source_id": edge.source_id, "target_id": edge.target_id}
        for edge in graph.edges
    ]
    return official_metric_row(nodes, edges, gt, scale)


def finite_delta(after: Any, before: Any) -> float | None:
    try:
        value = float(after) - float(before)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def summary_delta(full: dict[str, Any], control: dict[str, Any]) -> dict[str, float | None]:
    keys = (
        "score",
        "adj_edge_jaccard",
        "edge_jaccard",
        "node_recall",
        "division_jaccard",
        "division_tp",
        "division_fp",
        "division_fn",
    )
    return {key: finite_delta(full.get(key), control.get(key)) for key in keys}


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.population_manifest.read_text(encoding="utf-8"))
    identities = [str(value) for value in manifest["populations"][args.population]]
    datasets = [identity.split(":", 1)[1] for identity in identities]
    stats = load_stats(args.run_stats)
    graphs = load_selected_graphs(args.submission, set(datasets))
    full_score = json.loads(args.full_score.read_text(encoding="utf-8"))
    if full_score.get("population") != args.population:
        raise ValueError(
            f"full score population {full_score.get('population')!r} != {args.population!r}"
        )
    full_rows = {str(row["identity"]): row for row in full_score["samples"]}

    import zarr
    from zarr.storage import ZipStore

    rows: list[dict[str, Any]] = []
    authentication: list[dict[str, Any]] = []
    store = ZipStore(args.archive, mode="r")
    try:
        for index, dataset in enumerate(datasets, start=1):
            ablated, auth = ablate_gap2(graphs[dataset], stats[dataset])
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            row = {
                "identity": f"train:{dataset}",
                **graph_metric_row(ablated, gt, parse_scale(dict(group.attrs))),
            }
            rows.append(row)
            authentication.append({"identity": f"train:{dataset}", **auth})
            print(f"scored {index}/{len(datasets)} {dataset}", flush=True)
    finally:
        store.close()

    control_summary = official_summary(rows)
    family: dict[str, Any] = {}
    for prefix in ("44b6_", "6bba_"):
        control_family_rows = [row for row in rows if row["identity"].split(":", 1)[1].startswith(prefix)]
        full_family_rows = [
            full_rows[row["identity"]]
            for row in control_family_rows
        ]
        control_family = official_summary(control_family_rows)
        full_family = official_summary(full_family_rows)
        family[prefix.rstrip("_")] = {
            "n": len(control_family_rows),
            "same_run_control": control_family,
            "full_gap2": full_family,
            "gap2_minus_control": summary_delta(full_family, control_family),
        }

    payload = {
        "schema": "celltrack-alpha.gap2-same-run-ablation.v1",
        "status": "PASS",
        "population": args.population,
        "estimand": (
            "direct effect of the appended GAP2 nodes and incident edges, conditional on all "
            "other completed-run outputs including downstream line-fit-smoothed coordinates"
        ),
        "limitation": (
            "not an exact pre-GAP2 rerun because post-GAP2 line-fit smoothing is held fixed"
        ),
        "same_run_control": control_summary,
        "full_gap2": full_score["summary"],
        "gap2_minus_control": summary_delta(full_score["summary"], control_summary),
        "families": family,
        "removed_totals": {
            "pairs": sum(row["gap2_pairs_removed"] for row in authentication),
            "nodes": sum(row["gap2_nodes_removed"] for row in authentication),
            "edges": sum(row["gap2_edges_removed"] for row in authentication),
        },
        "authentication": authentication,
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload["gap2_minus_control"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
