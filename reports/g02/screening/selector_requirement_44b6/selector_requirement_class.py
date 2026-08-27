#!/usr/bin/env python3
"""Measure one frozen 44b6 selector action class in an isolated process."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np
import tracksdata as td
import zarr


ROOT = Path(
    os.environ.get("CELLTRACK_ROOT", "/Users/agent/Desktop/cell_tracking")
).resolve()
HERE = Path(__file__).resolve().parent
for location in (
    HERE,
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
import selector_requirement as base  # noqa: E402
from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.official_eval import (  # noqa: E402
    geff_arrays_to_graph,
    official_metric_row,
    official_summary,
    prediction_to_graph,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.selector-requirement-44b6-class.v1"
SWAP_PREREG_SHA256 = (
    "00e6036ef076482a39266d797c5fea3e2e4845ba4262c8532cfe70bb8ccb4628"
)
SWAP_EXECUTION_REPAIR = base.SWAP_DIR / "EXECUTION_REPAIR.json"
SWAP_EXECUTION_REPAIR_SHA256 = (
    "1996f4a8096d5cb858ec0bf42ec06786f6910c5921a4fd11d57b24a4f64f827b"
)
SWAP_LOCALITY_PROOF = base.SWAP_DIR / "N1_LOCALITY_EQUIVALENCE.json"
SWAP_LOCALITY_PROOF_SHA256 = (
    "836812c72a51e915e74b9e57b987c21e3f915290feff76b01121481d664dc55c"
)
FIT105_PREPARE = (
    ROOT / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
)
FIT105_PREPARE_SHA256 = (
    "c41bc216d4dfd642126d476a66a3fd9659f54a91dc49463f6b468a301a971d89"
)


def _verify(path: Path, expected: str, label: str) -> None:
    actual = base.sha256_file(path)
    if actual != expected:
        raise base.DiagnosticError(
            f"{label} identity drifted: expected {expected}, got {actual}"
        )


def authenticate_swap_inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    _verify(base.SWAP_PREREG, SWAP_PREREG_SHA256, "swap preregistration")
    _verify(
        SWAP_EXECUTION_REPAIR,
        SWAP_EXECUTION_REPAIR_SHA256,
        "swap execution repair",
    )
    _verify(
        SWAP_LOCALITY_PROOF,
        SWAP_LOCALITY_PROOF_SHA256,
        "swap locality proof",
    )
    prereg = base.load_json(base.SWAP_PREREG)
    if (
        prereg.get("status") != "LOCKED_BEFORE_HELD_OUTCOMES"
        or prereg.get("scope", {}).get("family_order") != ["44b6", "6bba"]
        or prereg.get("scope", {}).get("sa_repr_consumed") is not False
        or prereg.get("deployment", {}).get("proposal_metrics_gate") is not False
    ):
        raise base.DiagnosticError("swap preregistration content drifted")
    for key in (
        "n1_interface",
        "n1_implementation",
        "folds",
        "p0",
        "trackastra_run_manifest",
        "cache_attestation",
        "official_eval",
        "scoreall",
        "apncc_contract",
    ):
        pin = prereg["frozen_inputs"][key]
        _verify(ROOT / str(pin["path"]), str(pin["sha256"]), f"swap {key}")
    archive_pin = prereg["frozen_inputs"]["raw_archive"]
    base.verify_raw_transport(())
    p0 = base.load_json(ROOT / str(prereg["frozen_inputs"]["p0"]["path"]))
    if (
        p0.get("inputs", {}).get("archive", {}).get("sha256")
        != archive_pin["sha256_banked_in_p0"]
        or p0.get("inputs", {}).get("archive", {}).get("bytes")
        != archive_pin["bytes"]
    ):
        raise base.DiagnosticError("swap P0 archive identity drifted")
    _verify(FIT105_PREPARE, FIT105_PREPARE_SHA256, "fit105 preparation")
    prepare = base.load_json(FIT105_PREPARE)
    observed_provenance = (
        prepare.get("frozen_pairwise", {})
        .get("identity", {})
        .get("exposure", {})
        .get("provenance_manifest_sha256")
    )
    if observed_provenance != prereg["frozen_inputs"]["fit_provenance_sha256"]:
        raise base.DiagnosticError("swap fit provenance drifted")

    pin = prereg["frozen_inputs"]["apncc_44b6_report"]
    report_path = ROOT / str(pin["path"])
    _verify(report_path, str(pin["sha256"]), "APNCC 44b6 report")
    report = base.load_json(report_path)
    files = (
        report.get("graph", {})
        .get("control", {})
        .get("input_provenance", {})
        .get("files", [])
    )
    if not isinstance(files, list) or len(files) != 36:
        raise base.DiagnosticError("APNCC control fragment coverage drifted")
    records: dict[str, dict[str, Any]] = {}
    logical_hashes = []
    for row in files:
        identity = str(row["identity"])
        path = base.resolve_repo_path(str(row["path"])).resolve()
        _verify(path, str(row["sha256"]), f"APNCC fragment {identity}")
        if path.stat().st_size != int(row["bytes"]) or identity in records:
            raise base.DiagnosticError(f"APNCC fragment drifted: {identity}")
        records[identity] = {**dict(row), "resolved_path": str(path)}
        logical_hashes.append((identity, str(row["sha256"])))
    aggregate = scoreall.combined_fragment_sha256(logical_hashes)
    if (
        aggregate != str(pin["control_fragment_aggregate_sha256"])
        or aggregate
        != str(
            report["graph"]["control"]["input_provenance"]["aggregate_sha256"]
        )
        or set(records) != set(report.get("scope", {}).get("video_ids", []))
    ):
        raise base.DiagnosticError("APNCC control aggregate drifted")
    return prereg, records, report


def _ordered_original_ids(graph: Any) -> list[int]:
    return [
        int(node.node_id)
        for node in sorted(
            graph.nodes.values(), key=lambda value: (value.t, value.node_id)
        )
    ]


def official_node_matching(
    graph: Any,
    gt: Mapping[str, Any],
    scale_zyx: tuple[float, float, float],
) -> tuple[dict[int, int], dict[int, int]]:
    nodes, _edges = scoreall._prediction_rows(graph.dataset, graph)
    pred = prediction_to_graph(nodes, [])
    truth = geff_arrays_to_graph(dict(gt))
    pred_original = _ordered_original_ids(graph)
    gt_original = [
        int(node_id)
        for node_id, t in sorted(
            zip(gt["nodes"]["node_id"], gt["nodes"]["t"], strict=True),
            key=lambda row: (int(row[1]), int(row[0])),
        )
    ]
    pred_internal = [int(value) for value in pred.node_attrs()["node_id"].to_list()]
    gt_internal = [int(value) for value in truth.node_attrs()["node_id"].to_list()]
    if pred_internal != list(range(len(pred_original))) or gt_internal != list(
        range(len(gt_original))
    ):
        raise base.DiagnosticError("official node order drifted")
    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import get_options, set_options

    previous = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred.match(
                truth,
                matching=DistanceMatching(max_distance=7.0, scale=scale_zyx),
            )
    finally:
        set_options(show_progress=previous)
    attrs = pred.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    for pred_id, gt_id in attrs.iter_rows():
        if gt_id is None or int(gt_id) == -1:
            continue
        pred_original_id = pred_original[int(pred_id)]
        gt_original_id = gt_original[int(gt_id)]
        if pred_original_id in pred_to_gt or gt_original_id in gt_to_pred:
            raise base.DiagnosticError("official node matching is not one-to-one")
        pred_to_gt[pred_original_id] = gt_original_id
        gt_to_pred[gt_original_id] = pred_original_id
    return pred_to_gt, gt_to_pred


def truth_topology(
    gt: Mapping[str, Any],
) -> tuple[
    set[tuple[int, int]],
    dict[int, set[int]],
    dict[int, set[int]],
]:
    truth = {(int(source), int(target)) for source, target in gt["edges"]}
    successors: dict[int, set[int]] = defaultdict(set)
    predecessors: dict[int, set[int]] = defaultdict(set)
    for source, target in truth:
        successors[source].add(target)
        predecessors[target].add(source)
    return truth, successors, predecessors


def edge_sets(graph: Any) -> set[tuple[int, int]]:
    return {
        (int(edge.source_id), int(edge.target_id)) for edge in graph.edges
    }


def edge_class(
    pair: tuple[int, int],
    pred_to_gt: Mapping[int, int],
    truth: set[tuple[int, int]],
    gt_outgoing: Mapping[int, set[int]],
    gt_incoming: Mapping[int, set[int]],
) -> tuple[bool, bool]:
    source = pred_to_gt.get(pair[0])
    target = pred_to_gt.get(pair[1])
    is_true = (
        source is not None
        and target is not None
        and (source, target) in truth
    )
    is_valid = (
        (source is not None and bool(gt_outgoing.get(source)))
        or (target is not None and bool(gt_incoming.get(target)))
    )
    return is_true, is_valid


def load_delete_population() -> tuple[base.EdgePopulation, dict[str, Any]]:
    manifest = base.load_json(base.POP_MANIFEST)
    p0 = base.load_json(base.P0)
    result = base.load_json(base.DELETE_RESULT)["family_results"]["44b6"]
    by_identity = {str(row["identity"]): row for row in result["videos"]}
    arrays = base._load_npz_class("delete", manifest)
    identities = arrays[0]
    fragment_root = ROOT / str(p0["control"]["fragment_root"])
    graphs = []
    rows = []
    for identity in identities:
        dataset = identity.split(":", 1)[1]
        path = fragment_root / f"{identity.replace(':', '__', 1)}.csv"
        graph = read_submission(path)[dataset]
        record = by_identity[identity]
        if base.graph_sha256(graph) != record["before_graph_sha256"]:
            raise base.DiagnosticError(f"delete baseline graph drifted: {identity}")
        graphs.append(graph)
        rows.append(dict(record["baseline_metrics"]))
    baseline = base.edge_summary(rows)
    if base.metric_delta(baseline, result["control"]) != {
        "adjusted_edge_jaccard": 0.0,
        "raw_edge_jaccard": 0.0,
        "node_recall": 0.0,
    }:
        raise base.DiagnosticError("delete baseline summary drifted")
    d_tp = np.empty(len(arrays[1]), dtype=np.int8)
    d_fp = np.empty(len(arrays[1]), dtype=np.int8)
    d_fn = np.empty(len(arrays[1]), dtype=np.int8)
    offsets = np.cumsum(
        [0]
        + [
            int(row["count"])
            for row in manifest["classes"]["delete"]["videos"]
        ]
    )
    with base.open_raw_store() as store:
        for video_index, identity in enumerate(identities):
            dataset = identity.split(":", 1)[1]
            graph = graphs[video_index]
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            pred_to_gt, _ = official_node_matching(graph, gt, scale)
            truth, gt_out, gt_in = truth_topology(gt)
            start, end = offsets[video_index : video_index + 2]
            tp = fp = 0
            for index in range(int(start), int(end)):
                pair = (int(arrays[2][index]), int(arrays[3][index]))
                is_true, is_valid = edge_class(
                    pair, pred_to_gt, truth, gt_out, gt_in
                )
                d_tp[index] = -int(is_true)
                d_fp[index] = -int(is_valid and not is_true)
                d_fn[index] = int(is_true)
                tp += int(is_true)
                fp += int(is_valid and not is_true)
            if (
                tp != int(rows[video_index]["edge_tp"])
                or fp != int(rows[video_index]["edge_fp"])
            ):
                raise base.DiagnosticError(
                    f"delete edge truth replay drifted: {identity}"
                )
    population = base.EdgePopulation(
        "delete",
        identities,
        tuple(graphs),
        tuple(rows),
        *arrays[1:],
        d_tp,
        d_fp,
        d_fn,
        None,
    )
    return population, baseline


def load_swap_population() -> tuple[base.EdgePopulation, dict[str, Any]]:
    _prereg, records, control_report = authenticate_swap_inputs()
    manifest = base.load_json(base.POP_MANIFEST)
    arrays = base._load_npz_class("occupied_swap", manifest)
    identities = arrays[0]
    if tuple(sorted(records)) != tuple(sorted(identities)):
        raise base.DiagnosticError("swap control coverage drifted")
    graphs = []
    rows = []
    d_tp = np.empty(len(arrays[1]), dtype=np.int8)
    d_fp = np.empty(len(arrays[1]), dtype=np.int8)
    d_fn = np.empty(len(arrays[1]), dtype=np.int8)
    conflicts: list[tuple[tuple[int, int], ...]] = [
        () for _ in range(len(arrays[1]))
    ]
    offsets = np.cumsum(
        [0]
        + [
            int(row["count"])
            for row in manifest["classes"]["occupied_swap"]["videos"]
        ]
    )
    with base.open_raw_store() as store:
        for video_index, identity in enumerate(identities):
            dataset = identity.split(":", 1)[1]
            graph = read_submission(
                Path(records[identity]["resolved_path"])
            )[dataset]
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            nodes, edges = scoreall._prediction_rows(dataset, graph)
            row = official_metric_row(nodes, edges, gt, scale)
            pred_to_gt, _ = official_node_matching(graph, gt, scale)
            truth, gt_out, gt_in = truth_topology(gt)
            baseline_edges = edge_sets(graph)
            truth_by_edge = {
                pair: edge_class(pair, pred_to_gt, truth, gt_out, gt_in)
                for pair in baseline_edges
            }
            start, end = offsets[video_index : video_index + 2]
            for index in range(int(start), int(end)):
                source = int(arrays[2][index])
                target = int(arrays[3][index])
                pair = (source, target)
                local_conflicts = tuple(
                    sorted(
                        edge
                        for edge in baseline_edges
                        if edge[0] == source or edge[1] == target
                    )
                )
                if len(local_conflicts) not in (1, 2):
                    raise base.DiagnosticError(
                        "occupied-swap conflict boundary drifted"
                    )
                added_true, added_valid = edge_class(
                    pair, pred_to_gt, truth, gt_out, gt_in
                )
                removed_true = sum(
                    int(truth_by_edge[edge][0]) for edge in local_conflicts
                )
                removed_fp = sum(
                    int(truth_by_edge[edge][1] and not truth_by_edge[edge][0])
                    for edge in local_conflicts
                )
                d_tp[index] = int(added_true) - removed_true
                d_fp[index] = int(added_valid and not added_true) - removed_fp
                d_fn[index] = -d_tp[index]
                conflicts[index] = local_conflicts
            graphs.append(graph)
            rows.append(row)
    baseline = base.edge_summary(rows)
    frozen = control_report["graph"]["control"]["summary"]
    for key in ("adj_edge_jaccard", "edge_jaccard", "node_recall"):
        if not math.isclose(
            float(baseline[key]), float(frozen[key]), abs_tol=1e-12
        ):
            raise base.DiagnosticError(f"swap baseline drifted: {key}")
    population = base.EdgePopulation(
        "occupied_swap",
        identities,
        tuple(graphs),
        tuple(rows),
        *arrays[1:],
        d_tp,
        d_fp,
        d_fn,
        tuple(conflicts),
    )
    return population, baseline


def load_population(
    action_class: str,
) -> tuple[base.EdgePopulation | base.RestorePopulation, dict[str, Any]]:
    if action_class == "delete":
        return load_delete_population()
    if action_class == "occupied_swap":
        return load_swap_population()
    if action_class == "restore_topology":
        restore_module = base.load_module(
            "selector_requirement_class_restore", base.RESTORE_IMPL
        )
        population = base.load_restore_population(restore_module)
        return population, official_summary(list(population.baseline_rows))
    raise base.DiagnosticError(f"unsupported action class {action_class}")


def replay_swap_compiler(
    population: base.EdgePopulation, selected: np.ndarray
) -> dict[str, Any]:
    """Compile a conflict-disjoint swap selection as one equivalent transaction."""
    if population.action_class != "occupied_swap" or population.conflicts is None:
        raise base.DiagnosticError("bulk swap replay requires occupied swaps")
    by_video: dict[int, list[int]] = defaultdict(list)
    for index in selected:
        by_video[int(population.video[index])].append(int(index))
    receipts = []
    for video_index, indices in sorted(by_video.items()):
        add_edges = tuple(
            sorted(
                (
                    int(population.source[index]),
                    int(population.target[index]),
                )
                for index in indices
            )
        )
        conflicts = tuple(
            sorted(
                {
                    pair
                    for index in indices
                    for pair in population.conflicts[index]
                }
            )
        )
        if (
            len({source for source, _target in add_edges}) != len(add_edges)
            or len({target for _source, target in add_edges}) != len(add_edges)
            or sum(len(population.conflicts[index]) for index in indices)
            != len(conflicts)
        ):
            raise base.DiagnosticError(
                "selected swaps are not conflict-disjoint for bulk replay"
            )
        action = base.GraphAction(
            action_id=f"diagnostic.swap.bulk:{video_index}",
            kind=base.ActionKind.SWAP,
            add_edges=add_edges,
            replace_outgoing_from=tuple(source for source, _target in add_edges),
            replace_incoming_to=tuple(target for _source, target in add_edges),
            expected_conflict_edges=conflicts,
        )
        compiled = base.compile_action(population.graphs[video_index], action)
        receipts.append(
            {
                "video_index": video_index,
                "action_count": len(indices),
                "transaction_count": 1,
                "before_graph_sha256": compiled.before_graph_sha256,
                "after_graph_sha256": compiled.after_graph_sha256,
            }
        )
    return {
        "proof": (
            "Greedy ownership makes sources, targets, and removed baseline "
            "conflicts pairwise disjoint. Their primitive union is therefore one "
            "valid frozen N1 SWAP transaction with the same output graph as the "
            "atomic compile_batch union."
        ),
        "affected_videos": len(receipts),
        "compiled_actions": int(len(selected)),
        "compiler_transactions": len(receipts),
        "receipts": receipts,
    }


def measure(
    action_class: str,
    population: base.EdgePopulation | base.RestorePopulation,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    order = base.oracle_order(population.delta_s)
    curve, selections = base.curve_result(action_class, population, order)
    oracle_receipts = []
    for row, selected in zip(curve, selections, strict=True):
        if isinstance(population, base.EdgePopulation):
            if action_class == "occupied_swap":
                replay = replay_swap_compiler(population, selected)
            else:
                replay = base.replay_compiler(population, selected)
        else:
            replay = base.replay_restore_compiler(population, selected)
        oracle_receipts.append(
            {"candidate_prefix_N": row["candidate_prefix_N"], **replay}
        )
    print(
        f"ORACLE {action_class} actions={len(population.label)} "
        f"beneficial={int(population.label.sum())}",
        flush=True,
    )
    labels = population.label
    seed_runs = []
    compiler_runs = []
    if np.any(labels == 1) and np.any(labels == 0):
        for target_auc in base.TARGET_AUCS:
            for seed_index in range(base.SEED_COUNT):
                scores, observed_auc, shift = base.controlled_scores(
                    labels, action_class, seed_index, target_auc
                )
                noise_order = np.lexsort(
                    (np.arange(len(scores), dtype=np.int64), -scores)
                )
                noise_curve, noise_selections = base.curve_result(
                    action_class, population, noise_order
                )
                retention = base.retention_metrics(
                    labels,
                    scores,
                    population.video,
                    population.fold,
                    population.budgets,
                )
                seed_runs.append(
                    {
                        "target_auc": target_auc,
                        "seed_index": seed_index,
                        "empirical_auc": observed_auc,
                        "positive_shift": shift,
                        "fold_auc": base.fold_aucs(
                            labels, scores, population.fold
                        ),
                        "retention": retention,
                        "curve": noise_curve,
                    }
                )
                terminal = noise_selections[-1]
                if isinstance(population, base.EdgePopulation):
                    if action_class == "delete":
                        replay = {
                            "proof": "all delete prefixes are subsets of the frozen exact N1-valid edge population; the full-population delete replay is rank-invariant",
                            "compiled_actions": len(terminal),
                        }
                    else:
                        replay = replay_swap_compiler(population, terminal)
                else:
                    replay = base.replay_restore_compiler(population, terminal)
                compiler_runs.append(
                    {
                        "target_auc": target_auc,
                        "seed_index": seed_index,
                        **replay,
                    }
                )
                print(
                    f"NOISE {action_class} auc={target_auc:.2f} "
                    f"seed={seed_index:02d} empirical={observed_auc:.8f}",
                    flush=True,
                )
        noise_status = "COMPLETE"
        aggregate = base.aggregate_noise(action_class, seed_runs)
    else:
        noise_status = "CLASS_DEGENERATE_NO_REQUIREMENT_SURFACE"
        aggregate = None
    return {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "action_class": action_class,
        "baseline": dict(baseline),
        "population": {
            "actions": len(labels),
            "beneficial": int(labels.sum()),
            "harmful": int((labels == 0).sum()),
            "videos": len(population.identities),
        },
        "oracle": {
            "action_class": action_class,
            "population": len(labels),
            "beneficial": int(labels.sum()),
            "harmful": int((labels == 0).sum()),
            "support_grid": base.support_grid(len(labels)),
            "curve": curve,
            "compiler_replay": oracle_receipts,
        },
        "noise": {
            "status": noise_status,
            "beneficial": int(labels.sum()),
            "harmful": int((labels == 0).sum()),
            "seed_runs": seed_runs,
            "aggregate": aggregate,
            "terminal_compiler_replay": compiler_runs,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-class",
        required=True,
        choices=("delete", "occupied_swap", "restore_topology"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1":
        raise SystemExit("binding local zero-swap guard is required")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("refusing to overwrite class output")
    started = time.perf_counter()
    try:
        pins = base._verify_pins()
        population, baseline = load_population(args.action_class)
        result = measure(args.action_class, population, baseline)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["frozen_input_sha256"] = pins
        output.mkdir(parents=True)
        base.canonical_json(output / "CLASS_RESULT.json", result)
    except Exception as exc:
        print(f"class diagnostic failed closed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "action_class": args.action_class,
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
