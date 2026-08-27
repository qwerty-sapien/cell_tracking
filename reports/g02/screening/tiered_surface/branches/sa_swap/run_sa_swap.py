#!/usr/bin/env python3
"""Run the frozen SA-SWAP wave-1 44b6 exact-regret graph falsifier."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import zarr
from zarr.storage import ZipStore


ROOT = Path("/Users/agent/Desktop/cell_tracking").resolve()
HERE = Path(__file__).resolve().parent
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from analyze_action_class_oracle import (  # noqa: E402
    _truth_topology,
    official_node_matching,
)
from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionCompileError,
    ActionKind,
    GraphAction,
    compile_action,
    compile_batch,
    graph_sha256,
)
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionGraph,
    read_submission,
    write_submission,
)
from celltrack_alpha.official_eval import (  # noqa: E402
    official_metric_row,
    official_summary,
)
from celltrack_alpha.trackastra_linear import proposal_features  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


PREREG_SHA256 = "00e6036ef076482a39266d797c5fea3e2e4845ba4262c8532cfe70bb8ccb4628"
EXECUTION_REPAIR_SHA256 = (
    "1996f4a8096d5cb858ec0bf42ec06786f6910c5921a4fd11d57b24a4f64f827b"
)
N1_LOCALITY_PROOF_SHA256 = (
    "836812c72a51e915e74b9e57b987c21e3f915290feff76b01121481d664dc55c"
)
SCHEMA = "celltrack-alpha.sa-swap-exact-regret-result.v1"
FAMILY = "44b6"
FEATURE_COUNT = 49
TRACKASTRA_COLUMN = 48
TARGET_SCALE = 1.0e6


class SwapRunError(RuntimeError):
    """The frozen SA-SWAP execution contract failed closed."""


@dataclass(frozen=True)
class RidgeHead:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
            raise SwapRunError("ridge score feature shape drifted")
        return ((values - self.mean) / self.scale) @ self.weights + self.intercept


@dataclass(frozen=True)
class ActionRecord:
    action: GraphAction
    feature: np.ndarray
    target: float
    source_index: int
    removed_edges: tuple[tuple[int, int], ...]


@dataclass
class VideoData:
    identity: str
    baseline_path: Path
    baseline_sha256: str
    baseline_metrics: dict[str, Any]
    baseline_graph_sha256: str
    feature_sha256: str
    record_store: Path
    record_count: int
    witness_compile_count: int
    witness_reject_count: int
    exclusions: Counter[str]
    source_proposals: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SwapRunError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SwapRunError(f"{label} must be a JSON object")
    return value


def verify(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise SwapRunError(f"{label} identity mismatch: {path}")


def fold(identity: str, count: int = 5) -> int:
    return int.from_bytes(
        hashlib.sha256(identity.encode("ascii")).digest()[:8], "big"
    ) % count


def fit_ridge(features: np.ndarray, targets: np.ndarray) -> RidgeHead:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] != FEATURE_COUNT
        or y.shape != (len(x),)
        or len(x) < 50
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise SwapRunError("ridge fitting arrays are malformed")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    z = (x - mean) / scale
    design = np.concatenate(
        (z, np.ones((len(z), 1), dtype=np.float64)), axis=1
    )
    penalty = np.eye(FEATURE_COUNT + 1, dtype=np.float64)
    penalty[-1, -1] = 0.0
    lhs = design.T @ design / len(design) + 0.01 * penalty
    rhs = design.T @ y / len(design)
    solution = np.linalg.solve(lhs, rhs)
    if not np.isfinite(solution).all():
        raise SwapRunError("ridge solution is non-finite")
    return RidgeHead(mean, scale, solution[:-1], float(solution[-1]))


def _graph(dataset: str, path: Path) -> SubmissionGraph:
    graphs = read_submission(path)
    graph = graphs.get(dataset)
    if graph is None or len(graphs) != 1:
        raise SwapRunError(f"control fragment identity mismatch: {path}")
    return graph


def _edge_sets(
    graph: SubmissionGraph,
) -> tuple[
    set[tuple[int, int]],
    dict[int, set[int]],
    dict[int, set[int]],
]:
    edges = {
        (int(edge.source_id), int(edge.target_id)) for edge in graph.edges
    }
    outgoing: dict[int, set[int]] = defaultdict(set)
    incoming: dict[int, set[int]] = defaultdict(set)
    for source, target in edges:
        outgoing[source].add(target)
        incoming[target].add(source)
    return edges, outgoing, incoming


def _edge_class(
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


def _adjustment_factor(metrics: Mapping[str, Any]) -> float:
    raw = float(metrics["edge_jaccard"])
    adjusted = float(metrics["adj_edge_jaccard"])
    if not math.isfinite(raw) or raw <= 0.0 or not math.isfinite(adjusted):
        raise SwapRunError("baseline adjustment factor is undefined")
    return adjusted / raw


def _compile_local_witness(
    graph: SubmissionGraph,
    action: GraphAction,
    conflicts: set[tuple[int, int]],
    baseline_edges: set[tuple[int, int]],
    outgoing: Mapping[int, set[int]],
    incoming: Mapping[int, set[int]],
) -> None:
    """Prove locality obligations and call frozen N1 on the complete witness."""

    pair = action.add_edges[0] if len(action.add_edges) == 1 else (-1, -1)
    source, target = pair
    expected = {
        edge
        for edge in baseline_edges
        if edge[0] == source or edge[1] == target
    }
    node_ids = {source, target}
    for conflict_source, conflict_target in conflicts:
        node_ids.add(conflict_source)
        node_ids.add(conflict_target)
    witness_edges = [
        edge
        for edge in graph.edges
        if (int(edge.source_id), int(edge.target_id)) in conflicts
    ]
    witness = SubmissionGraph(
        dataset=graph.dataset,
        nodes={
            node_id: graph.nodes[node_id]
            for node_id in node_ids
            if node_id in graph.nodes
        },
        edges=witness_edges,
    )
    _validate_local_witness(
        graph,
        action,
        conflicts,
        baseline_edges,
        outgoing,
        incoming,
        witness,
        expected,
        node_ids,
    )


def _validate_local_witness(
    graph: SubmissionGraph,
    action: GraphAction,
    conflicts: set[tuple[int, int]],
    baseline_edges: set[tuple[int, int]],
    outgoing: Mapping[int, set[int]],
    incoming: Mapping[int, set[int]],
    witness: SubmissionGraph,
    expected: set[tuple[int, int]] | None = None,
    node_ids: set[int] | None = None,
) -> None:
    """Validate a constructed witness, including exact node payload equality."""

    if len(action.add_edges) != 1:
        raise SwapRunError("local witness requires exactly one added edge")
    pair = action.add_edges[0]
    source, target = pair
    if source not in graph.nodes or target not in graph.nodes:
        raise SwapRunError("local witness is missing a candidate endpoint")
    if graph.nodes[target].t != graph.nodes[source].t + 1:
        raise SwapRunError("local witness candidate is not time-adjacent")
    calculated_expected = {
        edge
        for edge in baseline_edges
        if edge[0] == source or edge[1] == target
    }
    if expected is not None and expected != calculated_expected:
        raise SwapRunError("local witness expected-boundary proof drifted")
    expected = calculated_expected
    if (
        conflicts != expected
        or tuple(sorted(conflicts)) != action.expected_conflict_edges
        or action.replace_outgoing_from != (source,)
        or action.replace_incoming_to != (target,)
    ):
        raise SwapRunError("local witness conflict boundary is incomplete")
    omitted = baseline_edges - conflicts
    if any(edge[0] == source or edge[1] == target for edge in omitted):
        raise SwapRunError("local witness omitted an affected baseline edge")
    if (
        len(outgoing.get(source, ())) != 1
        or len(incoming.get(target, ())) != 1
        or any(len(outgoing.get(edge[0], ())) != 1 for edge in conflicts)
    ):
        raise SwapRunError("local witness global degree proof failed")
    calculated_node_ids = {source, target}
    for conflict_source, conflict_target in conflicts:
        calculated_node_ids.add(conflict_source)
        calculated_node_ids.add(conflict_target)
    if node_ids is not None and node_ids != calculated_node_ids:
        raise SwapRunError("local witness node-set proof drifted")
    node_ids = calculated_node_ids
    if not node_ids.issubset(graph.nodes):
        raise SwapRunError("local witness conflict endpoint is missing")
    if set(witness.nodes) != node_ids or any(
        witness.nodes[node_id] != graph.nodes[node_id] for node_id in node_ids
    ):
        raise SwapRunError("local witness node payload changed")
    witness_edges = witness.edges
    if {
        (int(edge.source_id), int(edge.target_id)) for edge in witness_edges
    } != conflicts:
        raise SwapRunError("local witness edges do not equal global conflicts")
    try:
        compiled = compile_action(witness, action)
    except ActionCompileError as exc:
        raise SwapRunError("frozen N1 rejected the complete local witness") from exc
    compiled_edges = {
        (int(edge.source_id), int(edge.target_id))
        for edge in compiled.graph.edges
    }
    if (
        compiled.removed_edges != tuple(sorted(conflicts))
        or set(compiled.graph.nodes) != node_ids
        or compiled_edges != {pair}
    ):
        raise SwapRunError("local witness N1 result violated equivalence proof")


def build_population(
    graph: SubmissionGraph,
    payload: Mapping[str, Any],
    gt: Mapping[str, Any],
    scale_zyx: tuple[float, float, float],
    baseline_metrics: Mapping[str, Any],
) -> tuple[list[ActionRecord], Counter[str], dict[str, int]]:
    """Build and exact-label every preregistered N1-valid action."""

    if list(payload.get("feature_names", ())) != [
        *(f"apncc_context_{index:02d}" for index in range(48)),
        "trackastra_margin",
    ]:
        raise SwapRunError("frozen feature names drifted")
    pairs = np.asarray(payload.get("challengers"), dtype=np.int64)
    features = proposal_features(payload)
    if pairs.shape != (len(features), 2):
        raise SwapRunError("challenger/feature population drifted")

    pred_to_gt, _gt_to_pred = official_node_matching(graph, gt, scale_zyx)
    truth, gt_outgoing, gt_incoming, _gt_nodes = _truth_topology(gt)
    baseline_edges, outgoing, incoming = _edge_sets(graph)

    baseline_tp = 0
    baseline_fp = 0
    truth_by_edge: dict[tuple[int, int], tuple[bool, bool]] = {}
    for pair in baseline_edges:
        classification = _edge_class(
            pair, pred_to_gt, truth, gt_outgoing, gt_incoming
        )
        truth_by_edge[pair] = classification
        baseline_tp += int(classification[0])
        baseline_fp += int(classification[1] and not classification[0])
    if (
        baseline_tp != int(baseline_metrics["edge_tp"])
        or baseline_fp != int(baseline_metrics["edge_fp"])
        or len(truth) - baseline_tp != int(baseline_metrics["edge_fn"])
    ):
        raise SwapRunError(
            "derived edge truth does not reproduce the frozen official scorer"
        )

    factor = _adjustment_factor(baseline_metrics)
    baseline_adjusted = float(baseline_metrics["adj_edge_jaccard"])
    exclusions: Counter[str] = Counter()
    witness_compile_count = 0
    witness_reject_count = 0
    seen: set[tuple[int, int]] = set()
    records: list[ActionRecord] = []
    for index, raw_pair in enumerate(pairs):
        pair = (int(raw_pair[0]), int(raw_pair[1]))
        if pair in seen:
            exclusions["duplicate_pair"] += 1
            continue
        seen.add(pair)
        source, target = pair
        if pair in baseline_edges:
            exclusions["already_present_noop"] += 1
            continue
        if source not in graph.nodes or target not in graph.nodes:
            exclusions["missing_endpoint"] += 1
            continue
        if graph.nodes[target].t != graph.nodes[source].t + 1:
            exclusions["non_adjacent"] += 1
            continue
        if len(outgoing.get(source, ())) != 1:
            exclusions["source_not_singly_occupied"] += 1
            continue
        if len(incoming.get(target, ())) != 1:
            exclusions["target_not_singly_occupied"] += 1
            continue
        conflicts = {
            edge
            for edge in baseline_edges
            if edge[0] == source or edge[1] == target
        }
        if not conflicts:
            exclusions["unoccupied"] += 1
            continue
        if any(len(outgoing.get(edge[0], ())) != 1 for edge in conflicts):
            exclusions["division_changing_conflict"] += 1
            continue
        action = GraphAction(
            action_id=f"swap:{source}:{target}",
            kind=ActionKind.SWAP,
            add_edges=(pair,),
            replace_outgoing_from=(source,),
            replace_incoming_to=(target,),
            expected_conflict_edges=tuple(sorted(conflicts)),
        )
        try:
            _compile_local_witness(
                graph,
                action,
                conflicts,
                baseline_edges,
                outgoing,
                incoming,
            )
        except SwapRunError:
            witness_reject_count += 1
            raise
        witness_compile_count += 1
        removed_tp = sum(int(truth_by_edge[edge][0]) for edge in conflicts)
        removed_fp = sum(
            int(truth_by_edge[edge][1] and not truth_by_edge[edge][0])
            for edge in conflicts
        )
        add_true, add_valid = _edge_class(
            pair, pred_to_gt, truth, gt_outgoing, gt_incoming
        )
        after_tp = baseline_tp - removed_tp + int(add_true)
        after_fp = baseline_fp - removed_fp + int(add_valid and not add_true)
        after_fn = len(truth) - after_tp
        denominator = after_tp + after_fp + after_fn
        raw = after_tp / denominator if denominator else float("nan")
        adjusted = max(0.0, raw * factor)
        target_value = TARGET_SCALE * (adjusted - baseline_adjusted)
        if not math.isfinite(target_value):
            raise SwapRunError("compiled action regret is malformed")
        records.append(
            ActionRecord(
                action=action,
                feature=np.asarray(features[index], dtype=np.float64),
                target=float(target_value),
                source_index=index,
                removed_edges=tuple(sorted(conflicts)),
            )
        )
    if not records:
        raise SwapRunError("the exact occupied-conflict population is empty")
    if witness_compile_count != len(records) or witness_reject_count != 0:
        raise SwapRunError("N1 witness population validation did not close")
    return records, exclusions, {
        "candidate_count": len(records),
        "witness_compile_count": witness_compile_count,
        "witness_reject_count": witness_reject_count,
    }


def _record_arrays(
    records: Sequence[ActionRecord],
) -> dict[str, np.ndarray]:
    """Convert records to deterministic primitive arrays without semantic loss."""

    count = len(records)
    features = np.empty((count, FEATURE_COUNT), dtype=np.float64)
    regrets = np.empty(count, dtype=np.float64)
    source_indices = np.empty(count, dtype=np.int64)
    source_ids = np.empty(count, dtype=np.int64)
    target_ids = np.empty(count, dtype=np.int64)
    conflict_counts = np.empty(count, dtype=np.int8)
    conflicts = np.full((count, 2, 2), -1, dtype=np.int64)
    for index, record in enumerate(records):
        pair = record.action.add_edges
        if (
            record.action.kind is not ActionKind.SWAP
            or len(pair) != 1
            or len(record.removed_edges) not in (1, 2)
        ):
            raise SwapRunError("streamed record is not an exact occupied swap")
        source, target = pair[0]
        features[index] = record.feature
        regrets[index] = record.target
        source_indices[index] = record.source_index
        source_ids[index] = source
        target_ids[index] = target
        conflict_counts[index] = len(record.removed_edges)
        for conflict_index, edge in enumerate(record.removed_edges):
            conflicts[index, conflict_index] = edge
    return {
        "features": features,
        "regrets": regrets,
        "source_indices": source_indices,
        "source_ids": source_ids,
        "target_ids": target_ids,
        "conflict_counts": conflict_counts,
        "conflicts": conflicts,
    }


def _write_record_store(path: Path, records: Sequence[ActionRecord]) -> None:
    arrays = _record_arrays(records)
    path.mkdir(parents=True, exist_ok=False)
    for name, values in arrays.items():
        np.save(path / f"{name}.npy", values, allow_pickle=False)


def _load_record_arrays(path: Path) -> dict[str, np.ndarray]:
    names = (
        "features",
        "regrets",
        "source_indices",
        "source_ids",
        "target_ids",
        "conflict_counts",
        "conflicts",
    )
    arrays = {
        name: np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in names
    }
    count = len(arrays["regrets"])
    expected_shapes = {
        "features": (count, FEATURE_COUNT),
        "regrets": (count,),
        "source_indices": (count,),
        "source_ids": (count,),
        "target_ids": (count,),
        "conflict_counts": (count,),
        "conflicts": (count, 2, 2),
    }
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise SwapRunError("streamed record array shape drifted")
    return arrays


def _load_records(path: Path) -> list[ActionRecord]:
    arrays = _load_record_arrays(path)
    records = []
    for index in range(len(arrays["regrets"])):
        source = int(arrays["source_ids"][index])
        target = int(arrays["target_ids"][index])
        conflict_count = int(arrays["conflict_counts"][index])
        if conflict_count not in (1, 2):
            raise SwapRunError("streamed conflict count drifted")
        conflicts = tuple(
            (
                int(arrays["conflicts"][index, conflict_index, 0]),
                int(arrays["conflicts"][index, conflict_index, 1]),
            )
            for conflict_index in range(conflict_count)
        )
        records.append(
            ActionRecord(
                action=GraphAction(
                    action_id=f"swap:{source}:{target}",
                    kind=ActionKind.SWAP,
                    add_edges=((source, target),),
                    replace_outgoing_from=(source,),
                    replace_incoming_to=(target,),
                    expected_conflict_edges=conflicts,
                ),
                feature=np.asarray(arrays["features"][index], dtype=np.float64),
                target=float(arrays["regrets"][index]),
                source_index=int(arrays["source_indices"][index]),
                removed_edges=conflicts,
            )
        )
    return records


def _feature_records(
    prereg: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    pin = prereg["frozen_inputs"]["trackastra_feature_report"]
    path = ROOT / str(pin["path"])
    verify(path, str(pin["sha256"]), "Trackastra feature report")
    report = load_json(path, "Trackastra feature report")
    aggregate = report.get("aggregate", {})
    files = report.get("files", [])
    if (
        report.get("schema")
        != "celltrack-alpha.trackastra-full-pool-feature-cache.v2"
        or report.get("verdict") != "PASS_LABEL_FREE_FEATURE_CACHE"
        or int(aggregate.get("video_count", 0)) != 105
        or int(aggregate.get("proposal_count", 0)) != 195354
        or int(aggregate.get("row_count", 0)) != 280095
        or not isinstance(files, list)
        or len(files) != 105
    ):
        raise SwapRunError("Trackastra feature report contract drifted")
    records = {}
    for row in files:
        identity = str(row["identity"])
        feature_path = ROOT / str(row["feature_path"])
        verify(feature_path, str(row["feature_sha256"]), identity)
        if identity in records:
            raise SwapRunError("duplicate Trackastra feature identity")
        records[identity] = dict(row)
    return records


def _control_records(
    prereg: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pin = prereg["frozen_inputs"]["apncc_44b6_report"]
    report_path = ROOT / str(pin["path"])
    verify(report_path, str(pin["sha256"]), "APNCC 44b6 report")
    report = load_json(report_path, "APNCC 44b6 report")
    provenance = report.get("graph", {}).get("control", {}).get(
        "input_provenance", {}
    )
    files = provenance.get("files", [])
    if not isinstance(files, list) or len(files) != 36:
        raise SwapRunError("APNCC control fragment coverage drifted")
    records: dict[str, dict[str, Any]] = {}
    logical_hashes = []
    for row in files:
        identity = str(row["identity"])
        path = Path(str(row["path"])).resolve()
        verify(path, str(row["sha256"]), f"APNCC fragment {identity}")
        if path.stat().st_size != int(row["bytes"]):
            raise SwapRunError(f"APNCC fragment size drifted: {identity}")
        records[identity] = {**dict(row), "resolved_path": str(path)}
        logical_hashes.append((identity, str(row["sha256"])))
    aggregate = scoreall.combined_fragment_sha256(logical_hashes)
    if (
        aggregate != str(pin["control_fragment_aggregate_sha256"])
        or aggregate != str(provenance.get("aggregate_sha256"))
        or set(records) != set(report.get("scope", {}).get("video_ids", []))
    ):
        raise SwapRunError("APNCC control aggregate identity drifted")
    return records, report


def authenticate(prereg_path: Path, prereg_sha256: str) -> dict[str, Any]:
    if prereg_sha256 != PREREG_SHA256:
        raise SwapRunError("runner accepts only the frozen preregistration digest")
    verify(prereg_path, prereg_sha256, "preregistration")
    verify(
        HERE / "EXECUTION_REPAIR.json",
        EXECUTION_REPAIR_SHA256,
        "execution repair contract",
    )
    verify(
        HERE / "N1_LOCALITY_EQUIVALENCE.json",
        N1_LOCALITY_PROOF_SHA256,
        "N1 locality-equivalence proof",
    )
    prereg = load_json(prereg_path, "preregistration")
    if (
        prereg.get("status") != "LOCKED_BEFORE_HELD_OUTCOMES"
        or prereg.get("scope", {}).get("family_order") != ["44b6", "6bba"]
        or prereg.get("scope", {}).get("sa_repr_consumed") is not False
        or prereg.get("deployment", {}).get("proposal_metrics_gate") is not False
    ):
        raise SwapRunError("preregistration content is not locked")
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
        verify(ROOT / str(pin["path"]), str(pin["sha256"]), key)
    archive_pin = prereg["frozen_inputs"]["raw_archive"]
    archive = ROOT / str(archive_pin["path"])
    if not archive.is_file() or archive.stat().st_size != int(
        archive_pin["bytes"]
    ):
        raise SwapRunError("raw archive size/identity drifted")
    p0 = load_json(
        ROOT / str(prereg["frozen_inputs"]["p0"]["path"]), "P0"
    )
    if (
        p0.get("inputs", {}).get("archive", {}).get("sha256")
        != archive_pin["sha256_banked_in_p0"]
        or p0.get("inputs", {}).get("archive", {}).get("bytes")
        != archive_pin["bytes"]
    ):
        raise SwapRunError("P0 does not bank the preregistered raw archive")
    prepare_path = (
        ROOT
        / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
    )
    verify(
        prepare_path,
        "c41bc216d4dfd642126d476a66a3fd9659f54a91dc49463f6b468a301a971d89",
        "fit105 preparation provenance",
    )
    prepare = load_json(prepare_path, "fit105 preparation provenance")
    observed_provenance = (
        prepare.get("frozen_pairwise", {})
        .get("identity", {})
        .get("exposure", {})
        .get("provenance_manifest_sha256")
    )
    if observed_provenance != prereg["frozen_inputs"][
        "fit_provenance_sha256"
    ]:
        raise SwapRunError("fit provenance pin drifted")
    return prereg


def authenticate_execution_manifest(
    path: Path, expected_sha256: str
) -> dict[str, Any]:
    verify(path.resolve(), expected_sha256, "post-repair execution manifest")
    manifest = load_json(path.resolve(), "post-repair execution manifest")
    pins = manifest.get("pins", {})
    expected = {
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "test_sha256": sha256(HERE / "test_sa_swap.py"),
        "preregistration_sha256": PREREG_SHA256,
        "execution_repair_sha256": EXECUTION_REPAIR_SHA256,
        "n1_locality_proof_sha256": N1_LOCALITY_PROOF_SHA256,
    }
    if (
        manifest.get("status") != "PASS_POST_REPAIR_BARRIER"
        or any(pins.get(name) != value for name, value in expected.items())
    ):
        raise SwapRunError("post-repair execution manifest pin drifted")
    return manifest


def _load_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    path = ROOT / str(record["feature_path"])
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(value, dict)
        or value.get("schema")
        != "celltrack-alpha.trackastra-full-pool-video-features.v2"
        or value.get("identity") != record["identity"]
        or value.get("contract_sha256")
        != "4da06d487ba1787f91467a01862e4f7471daa16244674f5171960774da06334b"
        or len(value.get("challengers", ())) != int(record["proposal_count"])
    ):
        raise SwapRunError(f"feature payload drifted: {record['identity']}")
    return value


def _fit_arrays_in_memory(
    records_by_identity: Mapping[str, Sequence[ActionRecord]],
    identities: Sequence[str],
    *,
    beta: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation retained solely for focused parity tests."""

    features = np.concatenate(
        [
            np.stack(
                [record.feature for record in records_by_identity[identity]]
            )
            for identity in identities
        ]
    )
    if beta:
        features = features.copy()
        features[:, TRACKASTRA_COLUMN] = 0.0
    targets = np.concatenate(
        [
            np.asarray(
                [
                    record.target
                    for record in records_by_identity[identity]
                ],
                dtype=np.float64,
            )
            for identity in identities
        ]
    )
    return features, targets


def _streamed_fit_pair(
    data: Mapping[str, VideoData],
    identities: Sequence[str],
    work: Path,
    held_fold: int,
) -> tuple[RidgeHead, RidgeHead, dict[str, int]]:
    """Fit alpha/beta from one identically ordered memory-mapped matrix."""

    feature_path = work / f"fold_{held_fold}_features.npy"
    target_path = work / f"fold_{held_fold}_targets.npy"
    features, targets = _assemble_streamed_fold(
        data, identities, feature_path, target_path
    )
    total = len(targets)
    alpha = fit_ridge(features, targets)
    features[:, TRACKASTRA_COLUMN] = 0.0
    features.flush()
    beta = fit_ridge(features, targets)
    statistics = {
        "fit_action_count": total,
        "target_positive": int((targets > 0.0).sum()),
        "target_zero": int((targets == 0.0).sum()),
        "target_negative": int((targets < 0.0).sum()),
    }
    del features, targets
    feature_path.unlink()
    target_path.unlink()
    return alpha, beta, statistics


def _assemble_streamed_fold(
    data: Mapping[str, VideoData],
    identities: Sequence[str],
    feature_path: Path,
    target_path: Path,
) -> tuple[np.memmap, np.memmap]:
    """Assemble the exact prior concatenate order into memory-mapped arrays."""

    total = sum(data[identity].record_count for identity in identities)
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float64,
        shape=(total, FEATURE_COUNT),
    )
    targets = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float64, shape=(total,)
    )
    offset = 0
    for identity in identities:
        arrays = _load_record_arrays(data[identity].record_store)
        count = data[identity].record_count
        if len(arrays["regrets"]) != count:
            raise SwapRunError("streamed record count drifted")
        features[offset : offset + count] = arrays["features"]
        targets[offset : offset + count] = arrays["regrets"]
        offset += count
        del arrays
    if offset != total:
        raise SwapRunError("streamed fold assembly count drifted")
    features.flush()
    targets.flush()
    return features, targets


def _head_record(head: RidgeHead) -> dict[str, Any]:
    payload = np.concatenate(
        (head.mean, head.scale, head.weights, np.asarray([head.intercept]))
    )
    return {
        "mean": head.mean.tolist(),
        "scale": head.scale.tolist(),
        "weights": head.weights.tolist(),
        "intercept": head.intercept,
        "sha256": hashlib.sha256(payload.tobytes()).hexdigest(),
    }


def _select(
    graph: SubmissionGraph,
    records: Sequence[ActionRecord],
    scores: np.ndarray,
    cap: int,
) -> tuple[SubmissionGraph, dict[str, Any]]:
    if scores.shape != (len(records),) or not np.isfinite(scores).all():
        raise SwapRunError("held deployment scores are malformed")
    candidates = [
        (float(scores[index]), records[index])
        for index in range(len(records))
        if float(scores[index]) > 0.0
    ]
    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1].action.add_edges[0][0],
            item[1].action.add_edges[0][1],
            item[1].action.action_id,
        )
    )
    selected: list[GraphAction] = []
    owned_sources: set[int] = set()
    owned_targets: set[int] = set()
    owned_removed: set[tuple[int, int]] = set()
    batch_rejected = 0
    for _score, record in candidates:
        if len(selected) >= cap:
            break
        action = record.action
        sources = set(action.replace_outgoing_from)
        targets = set(action.replace_incoming_to)
        removed = set(record.removed_edges)
        if (
            sources & owned_sources
            or targets & owned_targets
            or removed & owned_removed
        ):
            batch_rejected += 1
            continue
        selected.append(action)
        owned_sources.update(sources)
        owned_targets.update(targets)
        owned_removed.update(removed)
    if selected:
        try:
            compiled = compile_batch(graph, tuple(selected))
        except ActionCompileError as exc:
            raise SwapRunError(
                "N1 rejected the structurally disjoint emitted batch"
            ) from exc
        result = compiled.graph
        batch_sha = compiled.batch_sha256
        removed = len(compiled.removed_edges)
    else:
        result = graph.copy()
        batch_sha = None
        removed = 0
    return result, {
        "population": len(records),
        "predicted_positive": len(candidates),
        "selected": len(selected),
        "selected_action_ids": [action.action_id for action in selected],
        "batch_conflict_rejected": batch_rejected,
        "cap": cap,
        "cap_active": len(selected) >= cap,
        "batch_sha256": batch_sha,
        "removed_edges": removed,
        "invalid_single_actions": 0,
        "invalid_batch_actions": 0,
    }


def _delta(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "adjusted_edge_jaccard_delta": float(
            candidate["adj_edge_jaccard"]
        )
        - float(control["adj_edge_jaccard"]),
        "raw_edge_jaccard_delta": float(candidate["edge_jaccard"])
        - float(control["edge_jaccard"]),
        "node_recall_delta": float(candidate["node_recall"])
        - float(control["node_recall"]),
    }


def _edge_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        name: sum(int(row[name]) for row in rows)
        for name in ("edge_tp", "edge_fp", "edge_fn")
    }


def _classify(delta: Mapping[str, float], distinct: bool) -> tuple[str, str]:
    adjusted = float(delta["adjusted_edge_jaccard_delta"])
    passed = (
        adjusted >= 0.005
        and float(delta["raw_edge_jaccard_delta"]) >= 0.0
        and float(delta["node_recall_delta"]) >= 0.0
        and distinct
    )
    if passed:
        return "GO_44B6", "all preregistered graph guards passed"
    if adjusted <= 0.0:
        return "KILL_44B6", "adjusted edge Jaccard delta is at or below zero"
    return "PARK_44B6", "44b6 did not clear every promotion guard"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1"
        or os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0"
    ):
        raise SwapRunError("continuous zero-swap guard is not active")
    if args.output.exists():
        raise SwapRunError(f"refusing to overwrite output: {args.output}")
    if args.family != FAMILY:
        raise SwapRunError("wave-1 runner opens 44b6 first and no other family")
    started = time.perf_counter()
    prereg = authenticate(args.prereg.resolve(), args.prereg_sha256)
    execution_manifest = authenticate_execution_manifest(
        args.execution_manifest, args.execution_manifest_sha256
    )
    feature_records = _feature_records(prereg)
    control_records, frozen_control_report = _control_records(prereg)
    identities = tuple(
        str(value)
        for value in frozen_control_report["scope"]["video_ids"]
    )
    if (
        len(identities) != 36
        or len(set(identities)) != 36
        or set(identities)
        != {
            identity
            for identity in feature_records
            if identity.startswith("train:44b6_")
        }
    ):
        raise SwapRunError("44b6 feature/control coverage drifted")

    archive = ROOT / str(prereg["frozen_inputs"]["raw_archive"]["path"])
    spool_owner = tempfile.TemporaryDirectory(
        prefix="sa_swap_stream_", dir="/private/tmp"
    )
    spool = Path(spool_owner.name)
    data: dict[str, VideoData] = {}
    baseline_rows: list[dict[str, Any]] = []
    with ZipStore(archive, mode="r") as store:
        for index, identity in enumerate(identities, start=1):
            dataset = identity.split(":", 1)[1]
            control = control_records[identity]
            baseline_path = Path(control["resolved_path"])
            graph = _graph(dataset, baseline_path)
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            nodes, edges = scoreall._prediction_rows(dataset, graph)
            baseline_metrics = official_metric_row(nodes, edges, gt, scale)
            baseline_rows.append({"identity": identity, **baseline_metrics})
            payload = _load_payload(feature_records[identity])
            records, exclusions, witness_audit = build_population(
                graph, payload, gt, scale, baseline_metrics
            )
            record_store = spool / f"{index:02d}"
            _write_record_store(record_store, records)
            data[identity] = VideoData(
                identity=identity,
                baseline_path=baseline_path,
                baseline_sha256=str(control["sha256"]),
                baseline_metrics=dict(baseline_metrics),
                baseline_graph_sha256=graph_sha256(graph),
                feature_sha256=str(feature_records[identity]["feature_sha256"]),
                record_store=record_store,
                record_count=len(records),
                witness_compile_count=witness_audit[
                    "witness_compile_count"
                ],
                witness_reject_count=witness_audit[
                    "witness_reject_count"
                ],
                exclusions=exclusions,
                source_proposals=len(payload["challengers"]),
            )
            print(
                f"population {index:02d}/36 {identity} "
                f"source={len(payload['challengers'])} exact={len(records)}",
                flush=True,
            )
            del graph, gt, group, payload, nodes, edges, records, witness_audit

    alpha_heads: dict[int, RidgeHead] = {}
    beta_heads: dict[int, RidgeHead] = {}
    head_records = []
    for held_fold in range(5):
        fit_ids = tuple(
            identity for identity in identities if fold(identity) != held_fold
        )
        held_ids = tuple(
            identity for identity in identities if fold(identity) == held_fold
        )
        if not fit_ids or not held_ids:
            raise SwapRunError(f"fold {held_fold} is empty")
        alpha, beta, fit_statistics = _streamed_fit_pair(
            data, fit_ids, spool, held_fold
        )
        alpha_heads[held_fold] = alpha
        beta_heads[held_fold] = beta
        head_records.append(
            {
                "held_fold": held_fold,
                "fit_ids": list(fit_ids),
                "held_ids": list(held_ids),
                "fit_action_count": fit_statistics["fit_action_count"],
                "held_action_count": sum(
                    data[identity].record_count for identity in held_ids
                ),
                "target_positive": fit_statistics["target_positive"],
                "target_zero": fit_statistics["target_zero"],
                "target_negative": fit_statistics["target_negative"],
                "alpha": _head_record(alpha),
                "beta": _head_record(beta),
            }
        )

    args.output.mkdir(parents=True, exist_ok=False)
    alpha_dir = args.output / "alpha_fragments"
    beta_dir = args.output / "beta_fragments"
    alpha_dir.mkdir()
    beta_dir.mkdir()
    alpha_rows: list[dict[str, Any]] = []
    beta_rows: list[dict[str, Any]] = []
    deployment_records = []
    distinct_alpha = 0
    distinct_beta = 0
    with ZipStore(archive, mode="r") as store:
        for index, identity in enumerate(identities, start=1):
            dataset = identity.split(":", 1)[1]
            video = data[identity]
            graph = _graph(dataset, video.baseline_path)
            held_fold = fold(identity)
            records = _load_records(video.record_store)
            features = np.stack([record.feature for record in records])
            alpha_scores = alpha_heads[held_fold].score(features)
            beta_features = features.copy()
            beta_features[:, TRACKASTRA_COLUMN] = 0.0
            beta_scores = beta_heads[held_fold].score(beta_features)
            cap = max(16, int(math.ceil(0.005 * len(graph.edges))))
            alpha_graph, alpha_selection = _select(
                graph, records, alpha_scores, cap
            )
            beta_graph, beta_selection = _select(
                graph, records, beta_scores, cap
            )
            if len(alpha_graph.nodes) != len(graph.nodes) or len(
                beta_graph.nodes
            ) != len(graph.nodes):
                raise SwapRunError("swap changed the frozen node set")
            alpha_distinct = graph_sha256(alpha_graph) != graph_sha256(graph)
            beta_distinct = graph_sha256(beta_graph) != graph_sha256(graph)
            distinct_alpha += int(alpha_distinct)
            distinct_beta += int(beta_distinct)
            alpha_path = alpha_dir / f"{identity.replace(':', '__', 1)}.csv"
            beta_path = beta_dir / f"{identity.replace(':', '__', 1)}.csv"
            write_submission({dataset: alpha_graph}, alpha_path)
            write_submission({dataset: beta_graph}, beta_path)
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            alpha_nodes, alpha_edges = scoreall._prediction_rows(
                dataset, alpha_graph
            )
            beta_nodes, beta_edges = scoreall._prediction_rows(dataset, beta_graph)
            alpha_metric = official_metric_row(
                alpha_nodes, alpha_edges, gt, scale
            )
            beta_metric = official_metric_row(beta_nodes, beta_edges, gt, scale)
            baseline_metric = video.baseline_metrics
            for label, metric in (("alpha", alpha_metric), ("beta", beta_metric)):
                if (
                    int(metric["division_tp"])
                    != int(baseline_metric["division_tp"])
                    or int(metric["division_fp"])
                    != int(baseline_metric["division_fp"])
                    or int(metric["division_fn"])
                    != int(baseline_metric["division_fn"])
                    or float(metric["node_recall"])
                    != float(baseline_metric["node_recall"])
                ):
                    raise SwapRunError(
                        f"{label} changed division or node metrics for {identity}"
                    )
            alpha_rows.append({"identity": identity, **alpha_metric})
            beta_rows.append({"identity": identity, **beta_metric})
            deployment_records.append(
                {
                    "identity": identity,
                    "held_fold": held_fold,
                    "fit_contains_identity": False,
                    "baseline_fragment_sha256": video.baseline_sha256,
                    "baseline_graph_sha256": video.baseline_graph_sha256,
                    "feature_sha256": video.feature_sha256,
                    "source_proposals": video.source_proposals,
                    "complete_action_population": video.record_count,
                    "n1_witness_compile_count": video.witness_compile_count,
                    "n1_witness_reject_count": video.witness_reject_count,
                    "population_exclusions": dict(video.exclusions),
                    "target_positive": sum(
                        record.target > 0.0 for record in records
                    ),
                    "target_zero": sum(
                        record.target == 0.0 for record in records
                    ),
                    "target_negative": sum(
                        record.target < 0.0 for record in records
                    ),
                    "alpha": {
                        **alpha_selection,
                        "fragment": str(alpha_path),
                        "fragment_sha256": sha256(alpha_path),
                        "observation_distinct": alpha_distinct,
                    },
                    "beta": {
                        **beta_selection,
                        "fragment": str(beta_path),
                        "fragment_sha256": sha256(beta_path),
                        "observation_distinct": beta_distinct,
                    },
                }
            )
            print(
                f"graph {index:02d}/36 {identity} fold={held_fold} "
                f"alpha={alpha_selection['selected']} "
                f"beta={beta_selection['selected']}",
                flush=True,
            )
            del (
                graph,
                alpha_graph,
                beta_graph,
                gt,
                group,
                alpha_nodes,
                alpha_edges,
                beta_nodes,
                beta_edges,
                records,
                features,
                beta_features,
                alpha_scores,
                beta_scores,
            )

    baseline_summary = official_summary(baseline_rows)
    frozen_summary = frozen_control_report["graph"]["control"]["summary"]
    for name in (
        "adj_edge_jaccard",
        "edge_jaccard",
        "node_recall",
        "division_jaccard",
    ):
        if not math.isclose(
            float(baseline_summary[name]),
            float(frozen_summary[name]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise SwapRunError(f"APNCC official summary did not reproduce: {name}")
    alpha_summary = official_summary(alpha_rows)
    beta_summary = official_summary(beta_rows)
    alpha_delta = _delta(baseline_summary, alpha_summary)
    beta_delta = _delta(baseline_summary, beta_summary)
    status, reason = _classify(alpha_delta, distinct_alpha > 0)
    six_bba_executed = False

    exclusion_totals: Counter[str] = Counter()
    for video in data.values():
        exclusion_totals.update(video.exclusions)
    alpha_files = [
        (row["identity"], row["alpha"]["fragment_sha256"])
        for row in deployment_records
    ]
    beta_files = [
        (row["identity"], row["beta"]["fragment_sha256"])
        for row in deployment_records
    ]
    result = {
        "schema": SCHEMA,
        "node_id": "SA-SWAP-W1-EXACT-REGRET",
        "status": status,
        "reason": reason,
        "tier_proposal": "PARK_C" if status != "GO_44B6" else "LIVE",
        "family_order": ["44b6", "6bba"],
        "six_bba_executed": six_bba_executed,
        "cv57_executed": False,
        "preregistration": {
            "path": str(args.prereg.resolve()),
            "sha256": args.prereg_sha256,
        },
        "post_repair_execution_manifest": {
            "path": str(args.execution_manifest.resolve()),
            "sha256": args.execution_manifest_sha256,
            "status": execution_manifest["status"],
        },
        "scope": {
            "family": "44b6",
            "video_count": 36,
            "fit_train105_only": True,
            "held_predictions_per_video": 1,
            "proposal_metrics_gate": False,
            "validation12_access": False,
            "cv57_access": False,
            "audit_access": False,
            "test_access": False,
            "sa_repr_consumed": False,
            "c1_consumed": False,
            "c2_or_kbest_consumed": False,
            "branch_b_consumed": False,
        },
        "population": {
            "source_proposals": sum(
                video.source_proposals for video in data.values()
            ),
            "complete_actions": sum(
                video.record_count for video in data.values()
            ),
            "exclusions": dict(exclusion_totals),
            "invalid_single_actions": sum(
                video.witness_reject_count for video in data.values()
            ),
            "invalid_batch_actions": 0,
            "all_incident_conflicts_complete": True,
            "all_endpoints_present": True,
            "merges": 0,
            "dangling_endpoints": 0,
            "n1_population_validation": {
                "method": "complete minimal locality-equivalent witness per candidate using frozen public compile_action",
                "candidate_count": sum(
                    video.record_count for video in data.values()
                ),
                "witness_compile_count": sum(
                    video.witness_compile_count for video in data.values()
                ),
                "reject_count": sum(
                    video.witness_reject_count for video in data.values()
                ),
                "proof_artifact": {
                    "path": "reports/g02/screening/tiered_surface/branches/sa_swap/N1_LOCALITY_EQUIVALENCE.json",
                    "sha256": N1_LOCALITY_PROOF_SHA256,
                },
                "full_baseline_final_batch_compiled": True,
            },
        },
        "model": {
            "kind": "fixed ridge exact-regret regression",
            "ridge_lambda": 0.01,
            "feature_count": 49,
            "trainable_coefficients_including_intercept": 50,
            "seed": 20260728,
            "folds": head_records,
        },
        "graph": {
            "control": {
                "summary": baseline_summary,
                "edge_counts": _edge_counts(baseline_rows),
                "fragment_aggregate_sha256": prereg["frozen_inputs"][
                    "apncc_44b6_report"
                ]["control_fragment_aggregate_sha256"],
            },
            "alpha": {
                "summary": alpha_summary,
                "edge_counts": _edge_counts(alpha_rows),
                "delta": alpha_delta,
                "observation_distinct_videos": distinct_alpha,
                "fragment_aggregate_sha256": scoreall.combined_fragment_sha256(
                    alpha_files
                ),
            },
            "beta": {
                "summary": beta_summary,
                "edge_counts": _edge_counts(beta_rows),
                "delta": beta_delta,
                "observation_distinct_videos": distinct_beta,
                "fragment_aggregate_sha256": scoreall.combined_fragment_sha256(
                    beta_files
                ),
                "diagnostic_only": True,
            },
            "gate": {
                "adjusted_edge_jaccard_delta_min": 0.005,
                "raw_edge_jaccard_delta_min": 0.0,
                "node_recall_delta_min": 0.0,
                "proposal_metrics_gate": False,
            },
            "videos": deployment_records,
        },
        "input_attestation": {
            "n1_interface_sha256": prereg["frozen_inputs"]["n1_interface"][
                "sha256"
            ],
            "n1_implementation_sha256": prereg["frozen_inputs"][
                "n1_implementation"
            ]["sha256"],
            "folds_sha256": prereg["frozen_inputs"]["folds"]["sha256"],
            "p0_sha256": prereg["frozen_inputs"]["p0"]["sha256"],
            "fit_provenance_sha256": prereg["frozen_inputs"][
                "fit_provenance_sha256"
            ],
            "trackastra_feature_report_sha256": prereg["frozen_inputs"][
                "trackastra_feature_report"
            ]["sha256"],
            "trackastra_contract_sha256": prereg["frozen_inputs"][
                "trackastra_feature_report"
            ]["contract_sha256"],
            "apncc_report_sha256": prereg["frozen_inputs"][
                "apncc_44b6_report"
            ]["sha256"],
            "raw_archive_banked_sha256": prereg["frozen_inputs"][
                "raw_archive"
            ]["sha256_banked_in_p0"],
        },
        "execution": {
            "device": "cpu",
            "guard_active": True,
            "execution_repair_sha256": EXECUTION_REPAIR_SHA256,
            "n1_locality_proof_sha256": N1_LOCALITY_PROOF_SHA256,
            "implementation_sha256": sha256(Path(__file__).resolve()),
            "streamed_record_spool": True,
            "streamed_fold_memmap": True,
            "elapsed_seconds": time.perf_counter() - started,
            "external_action": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "submission": False,
        },
    }
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    spool_owner.cleanup()
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--family", default="44b6")
    value.add_argument("--prereg", type=Path, required=True)
    value.add_argument("--prereg-sha256", required=True)
    value.add_argument("--execution-manifest", type=Path, required=True)
    value.add_argument("--execution-manifest-sha256", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
