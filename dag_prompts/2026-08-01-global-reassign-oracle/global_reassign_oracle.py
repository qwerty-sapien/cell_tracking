#!/usr/bin/env python3
"""Execute the frozen GLOBAL-REASSIGN-ORACLE-V1 diagnostic.

The implementation is an analysis harness.  It imports the frozen scorer and
compiler, but never modifies either.  Scientific outputs are written one video
at a time with atomic rename so an interrupted run can resume safely.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import zarr
from zarr.storage import ZipStore


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FRAGMENTS = RUNS / "fragments"
SELECTOR_DIR = ROOT / "reports/g02/screening/selector_requirement_44b6"
for location in (
    SELECTOR_DIR,
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
import selector_requirement as selector_base  # noqa: E402
import selector_requirement_class as selector_class  # noqa: E402
from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionCompileError,
    ActionKind,
    GraphAction,
    compile_action,
    compile_batch,
    graph_sha256,
)
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    read_submission,
    validate_graphs,
)
from celltrack_alpha.official_eval import (  # noqa: E402
    official_metric_row,
    official_summary,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.global-reassign-oracle.v1"
GT_LABEL = "[GT-DEPENDENT]"
PREREG = RUNS / "N0_PREREGISTRATION.md"
PREREG_SHA = RUNS / "N0_PREREGISTRATION.sha256"
FINAL_JSON = RUNS / "N0_GLOBAL_REASSIGN_ORACLE.json"
FINAL_MD = RUNS / "N0_GLOBAL_REASSIGN_ORACLE.md"
LEDGER_PROPOSAL = RUNS / "PROPOSED_LEDGER_ROW.md"
LIVE_LOG_PROPOSAL = RUNS / "LIVE_STATE_LOG_LINE.md"
GUARD_RECEIPT = RUNS / "N0_RUN_GUARD.json"

APNCC_REPORT = ROOT / "reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json"
REACHABILITY = ROOT / "reports/g02/extended_radius_reachability_v1.json"
P0_REPORT = ROOT / "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json"
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
ACTION_POPULATION = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_repr/"
    "phase2_separation_gate/run_v1/population_manifest.json"
)
DELETE_RESULT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_delete/graph_result.json"
)
SWAP_PREREG = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_swap/PREREGISTRATION.json"
)
N1_INTERFACE = (
    ROOT / "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json"
)
N1_IMPLEMENTATION = ROOT / "src/celltrack_alpha/action_compiler.py"
OFFICIAL_EVAL = ROOT / "src/celltrack_alpha/official_eval.py"
SCOREALL = ROOT / "scripts/scoreall.py"
OFFICIAL_METRICS = (
    ROOT
    / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py"
)
OFFICIAL_DIVISION_METRICS = (
    ROOT
    / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py"
)

EXPECTED = {
    "apncc_report": "530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d",
    "apncc_fragments": "48b99c5db7024c0998b779d2dfcf30bade9c12d74ec679325cc1974ceb9f7ccb",
    "reachability": "22a70e0b750d3d59bdd92ca5a82967d35c5d866fafe1c867dbb09b936efc2803",
    "p0": "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
    "action_population": "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    "delete_result": "28517a7c9170cedbf550d55395f38ac330888949ead7467086c3e306c7f7485a",
    "swap_prereg": "00e6036ef076482a39266d797c5fea3e2e4845ba4262c8532cfe70bb8ccb4628",
    "n1_interface": "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    "n1_implementation": "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
    "official_eval": "22f12e0eb7568c364a8503b8eeb9167147c152f182c2aa10dc10aef588bc01e6",
    "scoreall": "7b92fed9fe66e9a2faf062528a78b04a65bc1be50f806dff44f51c780da5ac6c",
    "official_metrics": "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444",
    "official_division_metrics": "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9",
    "raw_archive": "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19",
}

T3_EARLY_STOP = 0.030
T0_EXPECTED = 0.018571397
T1_EXPECTED = 0.081897667
CONTROL_TOLERANCE = 1.0e-9
CHAIN_LIMITS: tuple[int | None, ...] = (2, 3, 4, None)
COLLAPSE_RATIO = 0.2449
ACTIONABILITY = 0.005
WALL_CAP_SECONDS = 6 * 60 * 60
RADIUS_UM = 10.0
NEIGHBOR_CAP = 8
Edge = tuple[int, int]


class OracleError(RuntimeError):
    """A frozen input, graph invariant, or terminal contract failed closed."""


@dataclass(frozen=True)
class Chain:
    missing_edges: tuple[Edge, ...]
    add_edges: tuple[Edge, ...]
    replace_sources: tuple[int, ...]
    replace_targets: tuple[int, ...]
    conflicts: tuple[Edge, ...]

    @property
    def length(self) -> int:
        return len(self.missing_edges)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OracleError(f"JSON root is not an object: {path}")
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def verify_prereg_committed() -> dict[str, Any]:
    if not PREREG.is_file() or not PREREG_SHA.is_file():
        raise OracleError("preregistration or its SHA sidecar is missing")
    expected = PREREG_SHA.read_text(encoding="ascii").strip().split()[0]
    actual = sha256(PREREG)
    if expected != actual:
        raise OracleError(f"preregistration SHA mismatch: {actual} != {expected}")
    relative = str(PREREG.relative_to(ROOT))
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if not commit:
        raise OracleError("preregistration has no commit")
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != actual:
        raise OracleError("working preregistration differs from its committed bytes")
    return {"path": relative, "sha256": actual, "commit": commit}


def verify_frozen_inputs() -> dict[str, str]:
    paths = {
        "apncc_report": APNCC_REPORT,
        "reachability": REACHABILITY,
        "p0": P0_REPORT,
        "action_population": ACTION_POPULATION,
        "delete_result": DELETE_RESULT,
        "swap_prereg": SWAP_PREREG,
        "n1_interface": N1_INTERFACE,
        "n1_implementation": N1_IMPLEMENTATION,
        "official_eval": OFFICIAL_EVAL,
        "scoreall": SCOREALL,
        "official_metrics": OFFICIAL_METRICS,
        "official_division_metrics": OFFICIAL_DIVISION_METRICS,
    }
    observed: dict[str, str] = {}
    for key, path in paths.items():
        actual = sha256(path)
        expected = EXPECTED[key]
        if expected.startswith("__PREREG_"):
            raise OracleError(f"implementation placeholder was not frozen: {key}")
        if actual != expected:
            raise OracleError(f"frozen input drifted: {key}: {actual} != {expected}")
        observed[str(path.relative_to(ROOT))] = actual
    if RAW_ARCHIVE.stat().st_size != 87_393_127_165:
        raise OracleError("raw archive byte size drifted")
    p0 = load_json(P0_REPORT)
    archive_pin = p0.get("inputs", {}).get("archive", {})
    if (
        archive_pin.get("sha256") != EXPECTED["raw_archive"]
        or int(archive_pin.get("bytes", -1)) != RAW_ARCHIVE.stat().st_size
    ):
        raise OracleError("P0 raw archive pin drifted")
    observed[str(RAW_ARCHIVE.relative_to(ROOT))] = EXPECTED["raw_archive"]
    selector_base._verify_pins()
    return observed


def apncc_records() -> tuple[tuple[str, ...], dict[str, dict[str, Any]], dict[str, Any]]:
    report = load_json(APNCC_REPORT)
    identities = tuple(str(value) for value in report["scope"]["video_ids"])
    if len(identities) != 36 or len(set(identities)) != 36:
        raise OracleError("APNCC held-fit scope is not exactly 36 unique videos")
    rows = report["graph"]["control"]["input_provenance"]["files"]
    records = {str(row["identity"]): dict(row) for row in rows}
    if set(records) != set(identities):
        raise OracleError("APNCC fragment identity coverage drifted")
    logical: list[tuple[str, str]] = []
    for identity, row in records.items():
        path = Path(str(row["path"])).resolve()
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise OracleError(f"APNCC fragment drifted: {identity}")
        row["resolved_path"] = str(path)
        logical.append((identity, str(row["sha256"])))
    aggregate = scoreall.combined_fragment_sha256(logical)
    if aggregate != EXPECTED["apncc_fragments"]:
        raise OracleError("APNCC fragment aggregate drifted")
    return identities, records, report


def metric(graph: SubmissionGraph, gt: Mapping[str, Any], scale: tuple[float, float, float]) -> dict[str, Any]:
    nodes, edges = scoreall._prediction_rows(graph.dataset, graph)
    return official_metric_row(nodes, edges, gt, scale)


def metric_delta(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {
        "adjusted": float(candidate["adj_edge_jaccard"]) - float(baseline["adj_edge_jaccard"]),
        "raw": float(candidate["edge_jaccard"]) - float(baseline["edge_jaccard"]),
        "node": float(candidate["node_recall"]) - float(baseline["node_recall"]),
        "combined": float(candidate["score"]) - float(baseline["score"]),
    }


def graph_with_edges(graph: SubmissionGraph, edges: Iterable[Edge]) -> SubmissionGraph:
    result = SubmissionGraph(
        graph.dataset,
        dict(graph.nodes),
        [SubmissionEdge(int(source), int(target)) for source, target in sorted(set(edges))],
    )
    validate_graphs({result.dataset: result})
    return result


def candidate_tensor(graph: SubmissionGraph, scale: tuple[float, float, float]) -> set[Edge]:
    frames: dict[int, list[Any]] = defaultdict(list)
    for node in graph.nodes.values():
        frames[int(node.t)].append(node)
    result: set[Edge] = set()
    scale_array = np.asarray(scale, dtype=np.float64)
    for frame_index, sources in frames.items():
        targets = frames.get(frame_index + 1)
        if not targets:
            continue
        sources = sorted(sources, key=lambda node: int(node.node_id))
        targets = sorted(targets, key=lambda node: int(node.node_id))
        target_ids = np.asarray([int(node.node_id) for node in targets], dtype=np.int64)
        target_points = np.asarray(
            [[int(node.z), int(node.y), int(node.x)] for node in targets],
            dtype=np.float64,
        ) * scale_array
        tree = cKDTree(target_points)
        for source in sources:
            source_point = np.asarray(
                [int(source.z), int(source.y), int(source.x)], dtype=np.float64
            ) * scale_array
            indices = tree.query_ball_point(source_point, RADIUS_UM + 1.0e-9)
            indices = sorted(
                indices,
                key=lambda index: (
                    float(np.linalg.norm(target_points[index] - source_point)),
                    int(target_ids[index]),
                    int(index),
                ),
            )[:NEIGHBOR_CAP]
            result.update((int(source.node_id), int(target_ids[index])) for index in indices)
    return result


def edge_truth(
    graph: SubmissionGraph,
    gt: Mapping[str, Any],
    scale: tuple[float, float, float],
) -> tuple[dict[int, int], set[Edge], dict[int, set[int]], dict[int, set[int]]]:
    pred_to_gt, _ = selector_class.official_node_matching(graph, gt, scale)
    truth, outgoing, incoming = selector_class.truth_topology(gt)
    return pred_to_gt, truth, outgoing, incoming


def correct_edges(
    pairs: Iterable[Edge], pred_to_gt: Mapping[int, int], truth: set[Edge]
) -> set[Edge]:
    return {
        pair
        for pair in pairs
        if pred_to_gt.get(pair[0]) is not None
        and pred_to_gt.get(pair[1]) is not None
        and (int(pred_to_gt[pair[0]]), int(pred_to_gt[pair[1]])) in truth
    }


def edge_is_true(pair: Edge, pred_to_gt: Mapping[int, int], truth: set[Edge]) -> bool:
    source = pred_to_gt.get(pair[0])
    target = pred_to_gt.get(pair[1])
    return source is not None and target is not None and (int(source), int(target)) in truth


def build_chains(
    baseline_edges: set[Edge],
    baseline_true_edges: set[Edge],
    oracle_edges: set[Edge],
) -> list[Chain]:
    """Build closure-complete alternating reassignment chains.

    A displaced baseline edge recursively links to the correct outgoing edge of
    its source and the correct incoming edge of its target.  Correct division
    daughters sharing a source are one chain.  Only missing oracle edges count
    toward chain length; retained correct baseline edges are compiler anchors.
    """
    missing = oracle_edges - baseline_edges
    by_source: dict[int, set[Edge]] = defaultdict(set)
    by_target: dict[int, set[Edge]] = defaultdict(set)
    baseline_out: dict[int, set[Edge]] = defaultdict(set)
    baseline_in: dict[int, set[Edge]] = defaultdict(set)
    for edge in missing:
        by_source[edge[0]].add(edge)
        by_target[edge[1]].add(edge)
    for edge in baseline_edges:
        baseline_out[edge[0]].add(edge)
        baseline_in[edge[1]].add(edge)

    unseen = set(missing)
    components: list[tuple[set[Edge], set[Edge]]] = []
    while unseen:
        seed = min(unseen)
        component: set[Edge] = set()
        add_edges: set[Edge] = set()
        queue: deque[Edge] = deque([seed])
        while queue:
            edge = queue.popleft()
            if edge in add_edges:
                continue
            add_edges.add(edge)
            if edge in missing:
                component.add(edge)
            source, target = edge
            neighbors = set(by_source[source]) | set(by_target[target])
            conflicts = set(baseline_out[source]) | set(baseline_in[target])
            for conflict_source, conflict_target in conflicts:
                neighbors.update(by_source[conflict_source])
                neighbors.update(by_target[conflict_target])
            # A correct incumbent touched by the transaction is re-added as an
            # anchor.  Processing that anchor recursively closes the chain over
            # all of its own incident conflicts and missing correct edges.
            neighbors.update(conflicts & baseline_true_edges)
            queue.extend(sorted(neighbors - add_edges))
        unseen.difference_update(component)
        components.append((component, add_edges))

    chains: list[Chain] = []
    for component, closure_add_edges in sorted(components, key=lambda value: min(value[0])):
        add_edges = set(closure_add_edges)
        sources = {source for source, _ in add_edges}
        targets = {target for _, target in add_edges}
        conflicts = {
            edge
            for edge in baseline_edges
            if edge[0] in sources or edge[1] in targets
        }
        removed_true = (conflicts & baseline_true_edges) - add_edges
        if removed_true:
            raise OracleError(f"chain would remove true anchors: {sorted(removed_true)}")
        final = (baseline_edges - conflicts) | add_edges
        if len(final) != len(set(final)):
            raise OracleError("chain produced duplicate edges")
        chains.append(
            Chain(
                missing_edges=tuple(sorted(component)),
                add_edges=tuple(sorted(add_edges)),
                replace_sources=tuple(sorted(sources)),
                replace_targets=tuple(sorted(targets)),
                conflicts=tuple(sorted(conflicts)),
            )
        )
    return chains


def chain_action(identity: str, index: int, chain: Chain) -> GraphAction:
    safe_identity = identity.replace(":", "_")
    return GraphAction(
        action_id=f"global.chain.{safe_identity}.{index}",
        kind=ActionKind.SWAP,
        add_edges=chain.add_edges,
        replace_outgoing_from=chain.replace_sources,
        replace_incoming_to=chain.replace_targets,
        expected_conflict_edges=chain.conflicts,
    )


def apply_chains(
    graph: SubmissionGraph,
    identity: str,
    chains: Sequence[Chain],
) -> tuple[SubmissionGraph, dict[str, Any]]:
    if not chains:
        return graph, {
            "expressible": True,
            "reason": "vacuous_no_selected_chains",
            "compiled_chains": 0,
            "actions": [],
        }
    actions = [chain_action(identity, index, chain) for index, chain in enumerate(chains)]
    individual = []
    expressible = True
    failure: str | None = None
    for action in actions:
        try:
            compiled = compile_action(graph, action)
            individual.append(
                {
                    "action_id": action.action_id,
                    "action_sha256": compiled.action_sha256,
                    "after_graph_sha256": compiled.after_graph_sha256,
                }
            )
        except ActionCompileError as exc:
            expressible = False
            failure = f"{type(exc).__name__}: {exc}"
            break
    baseline_edges = {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}
    all_conflicts = {edge for chain in chains for edge in chain.conflicts}
    all_add = {edge for chain in chains for edge in chain.add_edges}
    analysis_graph = graph_with_edges(graph, (baseline_edges - all_conflicts) | all_add)
    batch_hash = None
    if expressible:
        try:
            batch = compile_batch(graph, actions)
            batch_hash = batch.batch_sha256
            if graph_sha256(batch.graph) != graph_sha256(analysis_graph):
                raise OracleError("frozen compiler graph differs from analysis graph")
        except ActionCompileError as exc:
            expressible = False
            failure = f"{type(exc).__name__}: {exc}"
    return analysis_graph, {
        "expressible": expressible,
        "reason": failure or "frozen_multi_edge_swap_and_batch_compile_exactly",
        "compiled_chains": len(actions) if expressible else len(individual),
        "batch_sha256": batch_hash,
        "actions": individual,
    }


def gt(value: Any) -> dict[str, Any]:
    return {"value": value, "label": GT_LABEL}


def labelled_metrics(delta: Mapping[str, Any]) -> dict[str, Any]:
    return {name: gt(float(delta[name])) for name in ("adjusted", "raw", "node", "combined")}


def stage_fragment_path(stage: str, identity: str) -> Path:
    return FRAGMENTS / stage / f"{identity.replace(':', '__', 1)}.json"


def reachability_records() -> dict[str, dict[str, Any]]:
    report = load_json(REACHABILITY)
    rows = report.get("videos")
    if not isinstance(rows, list):
        raise OracleError("reachability video table missing")
    return {str(row["identity"]): dict(row) for row in rows}


def validate_resume_fragment(path: Path, stage: str, identity: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = load_json(path)
    if (
        value.get("schema") != SCHEMA + ".fragment"
        or value.get("stage") != stage
        or value.get("identity") != identity
        or value.get("preregistration_sha256") != sha256(PREREG)
    ):
        raise OracleError(f"resume fragment identity drifted: {path}")
    return value


def run_t3() -> dict[str, Any]:
    identities, records, report = apncc_records()
    reach = reachability_records()
    baseline_rows: list[dict[str, Any]] = []
    t3_rows: list[dict[str, Any]] = []
    counts = Counter()
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        for ordinal, identity in enumerate(identities, start=1):
            path = stage_fragment_path("t3", identity)
            resumed = validate_resume_fragment(path, "T3", identity)
            if resumed is not None:
                baseline_rows.append(dict(resumed["baseline_metric"]))
                t3_rows.append(dict(resumed["candidate_metric"]))
                counts.update({key: int(value) for key, value in resumed["counts"].items()})
                continue
            dataset = identity.split(":", 1)[1]
            graph = read_submission(Path(records[identity]["resolved_path"]))[dataset]
            gt_arrays = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            baseline_metric = metric(graph, gt_arrays, scale)
            pairs = candidate_tensor(graph, scale)
            pred_to_gt, truth, gt_out, gt_in = edge_truth(graph, gt_arrays, scale)
            correct = correct_edges(pairs, pred_to_gt, truth)
            baseline_edges = {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}
            baseline_true = {
                edge for edge in baseline_edges if edge_is_true(edge, pred_to_gt, truth)
            }
            clean = {
                edge
                for edge in correct
                if len(gt_out[int(pred_to_gt[edge[0]])]) == 1
                and len(gt_in[int(pred_to_gt[edge[1]])]) == 1
            }
            expected_row = reach.get(identity, {}).get("metrics", {})
            if (
                len(pairs) != int(expected_row.get("candidate_pool", {}).get("r10_k08", -1))
                or len(clean) != int(expected_row.get("policy_reachable", {}).get("r10_k08", -1))
            ):
                raise OracleError(
                    f"{identity}: r10/k8 tensor replay drifted: "
                    f"pool={len(pairs)}, clean={len(clean)}"
                )
            candidate_graph = graph_with_edges(graph, correct)
            candidate_metric = metric(candidate_graph, gt_arrays, scale)
            local_counts = {
                "candidate_pairs": len(pairs),
                "correct_candidate_edges": len(correct),
                "clean_correct_candidate_edges": len(clean),
                "correct_edges_installed": len(correct - baseline_edges),
                "baseline_true_edges": len(baseline_true),
                "baseline_edges": len(baseline_edges),
                "candidate_edges": len(correct),
                "fps_removed": int(baseline_metric["edge_fp"]) - int(candidate_metric["edge_fp"]),
            }
            fragment = {
                "schema": SCHEMA + ".fragment",
                "stage": "T3",
                "identity": identity,
                "ordinal": ordinal,
                "preregistration_sha256": sha256(PREREG),
                "baseline_fragment_sha256": records[identity]["sha256"],
                "baseline_graph_sha256": graph_sha256(graph),
                "candidate_graph_sha256": graph_sha256(candidate_graph),
                "baseline_metric": baseline_metric,
                "candidate_metric": candidate_metric,
                "correct_candidate_edges": [list(edge) for edge in sorted(correct)],
                "baseline_true_edges": [list(edge) for edge in sorted(baseline_true)],
                "counts": local_counts,
            }
            atomic_json(path, fragment)
            baseline_rows.append(baseline_metric)
            t3_rows.append(candidate_metric)
            counts.update(local_counts)
            print(f"T3 [{ordinal:02d}/36] {identity}", flush=True)
    baseline = official_summary(baseline_rows)
    frozen = report["graph"]["control"]["summary"]
    for key in ("adj_edge_jaccard", "edge_jaccard", "node_recall", "score"):
        if not math.isclose(float(baseline[key]), float(frozen[key]), abs_tol=1e-12):
            raise OracleError(f"T3 APNCC baseline drifted: {key}")
    candidate = official_summary(t3_rows)
    delta = metric_delta(baseline, candidate)
    result = {
        "stage": "T3",
        "scope_videos": 36,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "counts": dict(counts),
        "early_stop_threshold": T3_EARLY_STOP,
        "early_stop_triggered": delta["adjusted"] < T3_EARLY_STOP,
        "fragment_count": len(list((FRAGMENTS / "t3").glob("*.json"))),
    }
    atomic_json(RUNS / "T3_RESULT.json", result)
    return result


def selected_control_graphs(
    action_class: str,
    prefix: int,
) -> tuple[Any, dict[str, Any], np.ndarray, dict[str, Any]]:
    population, baseline = selector_class.load_population(action_class)
    order = selector_base.oracle_order(population.delta_s)
    curve, selections = selector_base.score_edge_order(population, order, [prefix])
    return population, baseline, selections[0], curve[0]


def score_control_graphs(
    stage: str,
    population: Any,
    selected: np.ndarray,
    baseline: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    by_video: dict[int, list[int]] = defaultdict(list)
    for raw in selected:
        by_video[int(population.video[int(raw)])].append(int(raw))
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    fragments: list[dict[str, Any]] = []
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        for video_index, identity in enumerate(population.identities):
            graph = population.graphs[video_index]
            indices = by_video.get(video_index, [])
            selection_sha256 = canonical_hash(indices)
            fragment_path = stage_fragment_path(stage.lower(), identity)
            resumed = validate_resume_fragment(fragment_path, stage, identity)
            if resumed is not None:
                if resumed.get("selection_sha256") != selection_sha256:
                    raise OracleError(f"{stage} selection drifted on resume: {identity}")
                fragments.append(resumed)
                baseline_rows.append(dict(resumed["baseline_metric"]))
                rows.append(dict(resumed["candidate_metric"]))
                continue
            if stage == "T0":
                add_edges = tuple(
                    sorted(
                        (int(population.source[index]), int(population.target[index]))
                        for index in indices
                    )
                )
                conflicts = tuple(
                    sorted({edge for index in indices for edge in population.conflicts[index]})
                )
                if add_edges:
                    action = GraphAction(
                        action_id=f"control.t0.{video_index}",
                        kind=ActionKind.SWAP,
                        add_edges=add_edges,
                        replace_outgoing_from=tuple(source for source, _ in add_edges),
                        replace_incoming_to=tuple(target for _, target in add_edges),
                        expected_conflict_edges=conflicts,
                    )
                    candidate = compile_action(graph, action).graph
                else:
                    candidate = graph
            else:
                delete_edges = tuple(
                    sorted(
                        (int(population.source[index]), int(population.target[index]))
                        for index in indices
                    )
                )
                if delete_edges:
                    action = GraphAction(
                        action_id=f"control.t1.{video_index}",
                        kind=ActionKind.DELETE_EDGES,
                        delete_edges=delete_edges,
                    )
                    candidate = compile_action(graph, action).graph
                else:
                    candidate = graph
            dataset = identity.split(":", 1)[1]
            gt_arrays = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            baseline_metric = metric(graph, gt_arrays, scale)
            candidate_metric = metric(candidate, gt_arrays, scale)
            fragment = {
                "schema": SCHEMA + ".fragment",
                "stage": stage,
                "identity": identity,
                "preregistration_sha256": sha256(PREREG),
                "baseline_graph_sha256": graph_sha256(graph),
                "candidate_graph_sha256": graph_sha256(candidate),
                "selected_actions": len(indices),
                "selection_sha256": selection_sha256,
                "baseline_metric": baseline_metric,
                "candidate_metric": candidate_metric,
            }
            atomic_json(fragment_path, fragment)
            fragments.append(fragment)
            baseline_rows.append(baseline_metric)
            rows.append(candidate_metric)
    official_baseline = official_summary(baseline_rows)
    official_candidate = official_summary(rows)
    for key in ("adj_edge_jaccard", "edge_jaccard", "node_recall"):
        if not math.isclose(float(official_baseline[key]), float(baseline[key]), abs_tol=1e-12):
            raise OracleError(f"{stage} official baseline drifted: {key}")
    return official_baseline, official_candidate, fragments


def run_control(stage: str) -> dict[str, Any]:
    if stage == "T0":
        action_class, prefix, expected = "occupied_swap", 32768, T0_EXPECTED
    elif stage == "T1":
        action_class, prefix, expected = "delete", 2048, T1_EXPECTED
    else:
        raise OracleError(f"unknown control: {stage}")
    population, baseline, selected, analytic = selected_control_graphs(action_class, prefix)
    official_baseline, official_candidate, fragments = score_control_graphs(
        stage, population, selected, baseline
    )
    delta = metric_delta(official_baseline, official_candidate)
    analytic_adjusted = float(analytic["graph_delta"]["adjusted_edge_jaccard"])
    if not math.isclose(delta["adjusted"], analytic_adjusted, abs_tol=1e-12):
        raise OracleError(f"{stage} analytic and official replay disagree")
    error = abs(delta["adjusted"] - expected)
    result = {
        "stage": stage,
        "action_class": action_class,
        "scope_videos": len(population.identities),
        "prefix": prefix,
        "selected_actions": len(selected),
        "baseline": official_baseline,
        "candidate": official_candidate,
        "delta": delta,
        "counts": {
            "correct_edges_installed": int(official_candidate["edge_tp"])
            - int(official_baseline["edge_tp"]),
            "fps_removed": int(official_baseline["edge_fp"])
            - int(official_candidate["edge_fp"]),
            "compiled_chains": len(selected),
        },
        "expected_adjusted_delta": expected,
        "absolute_error": error,
        "tolerance": CONTROL_TOLERANCE,
        "reproduced": error <= CONTROL_TOLERANCE,
        "fragment_count": len(fragments),
    }
    atomic_json(RUNS / f"{stage}_RESULT.json", result)
    return result


def chains_for_video(graph: SubmissionGraph, t3: Mapping[str, Any]) -> list[Chain]:
    baseline_edges = {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}
    correct = {tuple(map(int, edge)) for edge in t3["correct_candidate_edges"]}
    baseline_true = {tuple(map(int, edge)) for edge in t3["baseline_true_edges"]}
    return build_chains(baseline_edges, baseline_true, correct)


def run_t2(run_started: float) -> dict[str, Any]:
    identities, records, _report = apncc_records()
    limits = {"2": 2, "3": 3, "4": 4, "unbounded": None}
    aggregate_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in limits}
    baseline_rows: list[dict[str, Any]] = []
    aggregate_counts: dict[str, Counter[str]] = {key: Counter() for key in limits}
    histograms: dict[str, Counter[int]] = {key: Counter() for key in limits}
    compiler_findings: list[dict[str, Any]] = []
    processed_candidates = 0
    total_candidates = sum(
        int(load_json(stage_fragment_path("t3", identity))["counts"]["candidate_pairs"])
        for identity in identities
    )
    t2_started = time.perf_counter()
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        for ordinal, identity in enumerate(identities, start=1):
            output_path = stage_fragment_path("t2", identity)
            resumed = validate_resume_fragment(output_path, "T2", identity)
            if resumed is None:
                dataset = identity.split(":", 1)[1]
                graph = read_submission(Path(records[identity]["resolved_path"]))[dataset]
                t3 = load_json(stage_fragment_path("t3", identity))
                chains = chains_for_video(graph, t3)
                gt_arrays = read_geff_arrays(store, "train", dataset)
                group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
                scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
                baseline_metric = metric(graph, gt_arrays, scale)
                tiers: dict[str, Any] = {}
                for name, limit in limits.items():
                    selected = [chain for chain in chains if limit is None or chain.length <= limit]
                    candidate, compiler = apply_chains(graph, identity, selected)
                    candidate_metric = metric(candidate, gt_arrays, scale)
                    histogram = Counter(chain.length for chain in selected)
                    tiers[name] = {
                        "limit": limit,
                        "candidate_graph_sha256": graph_sha256(candidate),
                        "candidate_metric": candidate_metric,
                        "correct_edges_installed": sum(chain.length for chain in selected),
                        "fps_removed": int(baseline_metric["edge_fp"]) - int(candidate_metric["edge_fp"]),
                        "chains": len(selected),
                        "chain_length_histogram": {str(key): value for key, value in sorted(histogram.items())},
                        "compiler": compiler,
                    }
                resumed = {
                    "schema": SCHEMA + ".fragment",
                    "stage": "T2",
                    "identity": identity,
                    "ordinal": ordinal,
                    "preregistration_sha256": sha256(PREREG),
                    "baseline_fragment_sha256": records[identity]["sha256"],
                    "baseline_graph_sha256": graph_sha256(graph),
                    "baseline_metric": baseline_metric,
                    "available_chain_histogram": {
                        str(key): value for key, value in sorted(Counter(chain.length for chain in chains).items())
                    },
                    "tiers": tiers,
                }
                atomic_json(output_path, resumed)
                print(f"T2 [{ordinal:02d}/36] {identity}", flush=True)
            baseline_rows.append(dict(resumed["baseline_metric"]))
            t3_fragment = load_json(stage_fragment_path("t3", identity))
            processed_candidates += int(t3_fragment["counts"]["candidate_pairs"])
            for name in limits:
                tier = resumed["tiers"][name]
                aggregate_rows[name].append(dict(tier["candidate_metric"]))
                aggregate_counts[name].update(
                    {
                        "correct_edges_installed": int(tier["correct_edges_installed"]),
                        "fps_removed": int(tier["fps_removed"]),
                        "chains": int(tier["chains"]),
                        "compiled_chains": int(tier["compiler"]["compiled_chains"]),
                    }
                )
                histograms[name].update(
                    {int(key): int(value) for key, value in tier["chain_length_histogram"].items()}
                )
                if not tier["compiler"]["expressible"]:
                    compiler_findings.append(
                        {"identity": identity, "tier": name, "reason": tier["compiler"]["reason"]}
                    )
            if ordinal >= 3 and processed_candidates:
                t2_elapsed = time.perf_counter() - t2_started
                projected_t2 = t2_elapsed * total_candidates / processed_candidates
                projected_total = time.perf_counter() - run_started + max(0.0, projected_t2 - t2_elapsed)
                if projected_total > WALL_CAP_SECONDS:
                    result = {
                        "stage": "T2",
                        "status": "ESCALATE_PROJECTED_WALL_CAP",
                        "processed_videos": ordinal,
                        "projected_total_seconds": projected_total,
                        "wall_cap_seconds": WALL_CAP_SECONDS,
                    }
                    atomic_json(RUNS / "T2_RESULT.json", result)
                    return result
    baseline = official_summary(baseline_rows)
    tiers = {}
    for name in limits:
        candidate = official_summary(aggregate_rows[name])
        tiers[name] = {
            "limit": limits[name],
            "candidate": candidate,
            "delta": metric_delta(baseline, candidate),
            "counts": dict(aggregate_counts[name]),
            "chain_length_histogram": {
                str(key): value for key, value in sorted(histograms[name].items())
            },
        }
    result = {
        "stage": "T2",
        "status": "COMPLETE",
        "scope_videos": 36,
        "baseline": baseline,
        "tiers": tiers,
        "compiler_expressible": not compiler_findings,
        "compiler_failures": compiler_findings,
        "fragment_count": len(list((FRAGMENTS / "t2").glob("*.json"))),
    }
    atomic_json(RUNS / "T2_RESULT.json", result)
    return result


def tier_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    counts = result.get("counts", {})
    return {
        "label": GT_LABEL,
        "delta": labelled_metrics(result["delta"]),
        "correct_edges_installed": gt(int(counts.get("correct_edges_installed", 0))),
        "fps_removed": gt(int(counts.get("fps_removed", 0))),
        "chains_compiled": gt(int(counts.get("compiled_chains", counts.get("chains", 0)))),
        "chain_length_histogram": {
            str(key): gt(int(value))
            for key, value in result.get("chain_length_histogram", {}).items()
        },
    }


def scientific_result(
    prereg: Mapping[str, Any],
    frozen: Mapping[str, str],
    t3: Mapping[str, Any],
    t0: Mapping[str, Any] | None,
    t1: Mapping[str, Any] | None,
    t2: Mapping[str, Any] | None,
    started: float,
) -> dict[str, Any]:
    terminal = "COMPLETE"
    if t3["early_stop_triggered"]:
        terminal = "STOP_CLASS"
    elif t0 is not None and (not t0["reproduced"] or t1 is not None and not t1["reproduced"]):
        terminal = "INVALID_HARNESS"
    elif t2 is not None and t2.get("status") == "ESCALATE_PROJECTED_WALL_CAP":
        terminal = "ESCALATE_PROJECTED_WALL_CAP"

    tiers: dict[str, Any] = {
        "T3": {
            **tier_payload(t3),
            "early_stop_threshold": gt(T3_EARLY_STOP),
            "early_stop_triggered": bool(t3["early_stop_triggered"]),
            "scope_videos": gt(int(t3["scope_videos"])),
        }
    }
    if t0 is not None:
        tiers["T0"] = {
            **tier_payload({"delta": t0["delta"], "counts": t0["counts"]}),
            "scope_videos": gt(int(t0["scope_videos"])),
            "absolute_control_error": gt(float(t0["absolute_error"])),
            "reproduced_to_1e-9": bool(t0["reproduced"]),
        }
    if t1 is not None:
        tiers["T1"] = {
            **tier_payload({"delta": t1["delta"], "counts": t1["counts"]}),
            "scope_videos": gt(int(t1["scope_videos"])),
            "integrity_only_original_run_v6_population": True,
            "absolute_control_error": gt(float(t1["absolute_error"])),
            "reproduced_to_1e-9": bool(t1["reproduced"]),
        }
    if t2 is not None and t2.get("status") == "COMPLETE":
        for name in ("2", "3", "4", "unbounded"):
            tiers[f"T2_L{name}"] = {
                **tier_payload(t2["tiers"][name]),
                "scope_videos": gt(36),
            }

    decision = None
    compiler = {
        "finding": "NOT_RUN",
        "expressible": None,
        "escalate": False,
    }
    if t2 is not None and t2.get("status") == "COMPLETE" and t0 is not None:
        raw_increment = float(t2["tiers"]["unbounded"]["delta"]["adjusted"]) - float(t0["delta"]["adjusted"])
        adjusted = COLLAPSE_RATIO * raw_increment
        stop = adjusted < ACTIONABILITY
        decision = {
            "label": GT_LABEL,
            "precedent_adjusted_gain_first": gt(adjusted),
            "collapse_ratio": gt(COLLAPSE_RATIO),
            "raw_oracle_increment_T2_unbounded_minus_T0": gt(raw_increment),
            "actionability_floor": gt(ACTIONABILITY),
            "verdict": "STOP_CLASS" if stop else "CONTINUE_PROPOSAL_ONLY",
            "stop_class_emitted": stop,
        }
        expressible = bool(t2["compiler_expressible"])
        compiler = {
            "finding": (
                "EXPRESSIBLE_AS_FROZEN_MULTI_EDGE_SWAP_TRANSACTIONS"
                if expressible
                else "INEXPRESSIBLE_IN_FROZEN_COMPILER"
            ),
            "expressible": expressible,
            "escalate": not expressible,
            "failures": t2["compiler_failures"],
            "frozen_interface_sha256": EXPECTED["n1_interface"],
            "frozen_implementation_sha256": EXPECTED["n1_implementation"],
        }
        if not expressible:
            terminal = "ESCALATE_COMPILER_INEXPRESSIBLE"

    return {
        "schema": SCHEMA,
        "status": terminal,
        "scientific_number_label": GT_LABEL,
        "preregistration": dict(prereg),
        "frozen_inputs": dict(frozen),
        "scope": {
            "local_cpu_only": True,
            "experimental_videos": 36,
            "subsampling": False,
            "selector": False,
            "auc": False,
            "training": False,
            "representation": False,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
            "submission_action": False,
        },
        "tiers": tiers,
        "section4_decision": decision,
        "compiler_expressibility": compiler,
        "execution": {
            "scientific_child_elapsed_seconds": time.perf_counter() - started,
            "wall_clock_seconds": None,
            "peak_rss_gib": None,
            "preflight_seconds": None,
            "swap_growth_mib": None,
        },
    }


def fnum(value: Any) -> str:
    return f"{float(value):+.12f} {GT_LABEL}"


def cint(value: Any) -> str:
    return f"{int(value)} {GT_LABEL}"


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# GLOBAL-REASSIGN-ORACLE-V1 terminal report",
        "",
        f"**Status:** `{result['status']}`. Every scientific number below is {GT_LABEL}.",
        "",
    ]
    decision = result.get("section4_decision")
    if decision is not None:
        adjusted = decision["precedent_adjusted_gain_first"]["value"]
        raw = decision["raw_oracle_increment_T2_unbounded_minus_T0"]["value"]
        lines.extend(
            [
                "## Section 4 decision arithmetic",
                "",
                f"Precedent-adjusted gain (reported first): {fnum(adjusted)} = "
                f"0.2449 {GT_LABEL} × (T2(unbounded) − T0).",
                f"Raw oracle increment T2(unbounded) − T0: {fnum(raw)}.",
                f"Actionability floor: {fnum(decision['actionability_floor']['value'])}.",
                f"**Verdict:** `{decision['verdict']}`; STOP_CLASS emitted: "
                f"`{str(decision['stop_class_emitted']).lower()}`.",
                "",
            ]
        )
    lines.extend(["## Tier results", ""])
    for name, tier in result["tiers"].items():
        delta = tier["delta"]
        lines.append(
            f"- **{name}** — adjusted {fnum(delta['adjusted']['value'])}; "
            f"raw {fnum(delta['raw']['value'])}; node {fnum(delta['node']['value'])}; "
            f"combined {fnum(delta['combined']['value'])}; correct edges installed "
            f"{cint(tier['correct_edges_installed']['value'])}; FPs removed "
            f"{cint(tier['fps_removed']['value'])}; chains compiled "
            f"{cint(tier['chains_compiled']['value'])}."
        )
        histogram = tier.get("chain_length_histogram", {})
        if histogram:
            rendered = ", ".join(
                f"L={length}: {cint(count['value'])}" for length, count in histogram.items()
            )
            lines.append(f"  Chain-length histogram: {rendered}.")
    compiler = result["compiler_expressibility"]
    lines.extend(
        [
            "",
            "## Compiler expressibility",
            "",
            f"Finding: `{compiler['finding']}`. ESCALATE: `{str(compiler['escalate']).lower()}`. "
            f"Frozen interface `{compiler.get('frozen_interface_sha256', EXPECTED['n1_interface'])}`; "
            f"implementation `{compiler.get('frozen_implementation_sha256', EXPECTED['n1_implementation'])}`.",
            "",
            "## Execution",
            "",
            f"Wall-clock: `{result['execution'].get('wall_clock_seconds')}` s; peak RSS: "
            f"`{result['execution'].get('peak_rss_gib')}` GiB; swap growth: "
            f"`{result['execution'].get('swap_growth_mib')}` MiB.",
            "",
            "No Kaggle, GPU, Kaggle CPU kernel, submission, RunPod, selector, AUC, training, "
            "representation, 6bba, CV57, held-fold, tier, DEAD, plan, or LEDGER mutation occurred.",
            "",
        ]
    )
    return "\n".join(lines)


def write_proposals(result: Mapping[str, Any]) -> None:
    status = result["status"]
    decision = result.get("section4_decision") or {}
    adjusted = decision.get("precedent_adjusted_gain_first", {}).get("value")
    raw = decision.get("raw_oracle_increment_T2_unbounded_minus_T0", {}).get("value")
    ledger = (
        "| 2026-08-01 | GLOBAL-REASSIGN-ORACLE-V1 | GT-dependent local CPU diagnostic | "
        f"{status}; precedent-adjusted={adjusted} {GT_LABEL}; raw T2∞−T0={raw} {GT_LABEL}; "
        f"compiler={result['compiler_expressibility']['finding']} | No tier/DEAD/plan/external action |\n"
    )
    atomic_text(LEDGER_PROPOSAL, ledger)
    log = (
        "- 2026-08-01 — **GLOBAL-REASSIGN-ORACLE-V1 "
        f"{status}.** Full 36-video local-CPU GT-dependent diagnostic completed under the "
        "hash-pinned preregistration; see `dag_prompts/2026-08-01-global-reassign-oracle/"
        "runs/N0_GLOBAL_REASSIGN_ORACLE.md`. "
        f"Section-4 precedent-adjusted gain is `{adjusted}` {GT_LABEL} before raw "
        f"`{raw}` {GT_LABEL}; compiler finding `{result['compiler_expressibility']['finding']}`. "
        "No Kaggle/GPU/CPU-kernel/submission/RunPod, selector, training, representation, tier, "
        "DEAD, plan, or LEDGER action. (Codex root)\n"
    )
    atomic_text(LIVE_LOG_PROPOSAL, log)


def finalize_guard() -> int:
    if not FINAL_JSON.is_file() or not GUARD_RECEIPT.is_file():
        raise OracleError("scientific result or guard receipt missing")
    result = load_json(FINAL_JSON)
    guard = load_json(GUARD_RECEIPT)
    result["execution"].update(
        {
            "wall_clock_seconds": guard["elapsed_seconds"],
            "peak_rss_gib": guard["peak_process_tree_rss_gib"],
            "preflight_seconds": guard["preflight_elapsed_seconds"],
            "swap_growth_mib": guard["swap_growth_mib"],
            "guard_status": guard["status"],
        }
    )
    if guard["status"] != "PASS" and not str(result["status"]).startswith("ESCALATE"):
        result["status"] = str(guard["status"])
    atomic_json(FINAL_JSON, result)
    atomic_text(FINAL_MD, render_markdown(result))
    write_proposals(result)
    return 0


def run_node() -> int:
    if os.environ.get("GLOBAL_REASSIGN_GUARD_ACTIVE") != "1":
        raise OracleError("binding local CPU guard is not active")
    started = time.perf_counter()
    prereg = verify_prereg_committed()
    frozen = verify_frozen_inputs()
    t3 = run_t3()
    t0 = t1 = t2 = None
    if not t3["early_stop_triggered"]:
        t0 = run_control("T0")
        t1 = run_control("T1") if t0["reproduced"] else None
        if t0["reproduced"] and t1 is not None and t1["reproduced"]:
            t2 = run_t2(started)
    result = scientific_result(prereg, frozen, t3, t0, t1, t2, started)
    atomic_json(FINAL_JSON, result)
    atomic_text(FINAL_MD, render_markdown(result))
    write_proposals(result)
    print(json.dumps({"status": result["status"]}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-guard", action="store_true")
    args = parser.parse_args()
    try:
        return finalize_guard() if args.finalize_guard else run_node()
    except Exception as exc:
        print(f"GLOBAL-REASSIGN-ORACLE failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
