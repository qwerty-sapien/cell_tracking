#!/usr/bin/env python3
"""Replay the frozen D1/D2 diagnostics from existing fit-only artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve()
while ROOT.parent != ROOT and not (ROOT / "src/celltrack_alpha").is_dir():
    ROOT = ROOT.parent

APNCC_REPORT = ROOT / "reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json"
PRE_APNCC_REPORT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_nodetopo/graph_gate_44b6/report.json"
)
POPULATION_MANIFEST = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_v1/population_manifest.json"
)
REVERSE_MANIFEST = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_reverse_authentic/reverse_cache/manifest.json"
)
REVERSE_SELECTION = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/reverse_endpoint_consensus_audit_v1/SELECTION.json"
)
REVERSE_SELECTOR = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/reverse_endpoint_consensus_audit_v1/select_literal_consensus.py"
)
R0_MANIFEST = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/r0_shell_nolink_own_merits_v1/truth_blind_shards_v1/MANIFEST.json"
)
R0_RESULT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/r0_shell_nolink_own_merits_v1/run_v2/RESULT.json"
)
R0_EVALUATOR = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/r0_shell_nolink_own_merits_v1/evaluate_r0_shell_nolink.py"
)
RAW_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
CACHE_ROOT = ROOT / "reports/g02/fitcache_117/bh9out/cache"


class DiagnosticError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticError(f"JSON object required: {path}")
    return value


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DiagnosticError(f"cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def vector(summary: Mapping[str, Any]) -> list[float]:
    return [
        float(summary["adj_edge_jaccard"]),
        float(summary["edge_jaccard"]),
        float(summary["node_recall"]),
    ]


def d1_gate_validity() -> dict[str, Any]:
    apncc = load_json(APNCC_REPORT)
    pre = load_json(PRE_APNCC_REPORT)
    challenger = apncc["graph"]["control"]
    incumbent = pre["graph"]["baseline"]
    challenger_rows = {
        str(row["identity"]): row for row in challenger["input_provenance"]["files"]
    }
    incumbent_rows = {
        str(row["identity"]): row for row in incumbent["input_provenance"]["files"]
    }
    if len(challenger_rows) != 36 or set(challenger_rows) != set(incumbent_rows):
        raise DiagnosticError("D1 does not have the exact common held-44b6 identity set")
    for rows in (challenger_rows, incumbent_rows):
        for identity, row in rows.items():
            path = Path(str(row["path"]))
            if not path.is_file() or sha256(path) != str(row["sha256"]):
                raise DiagnosticError(f"fragment identity mismatch: {identity} {path}")
    incumbent_vector = vector(incumbent["summary"])
    challenger_vector = vector(challenger["summary"])
    delta = [new - old for new, old in zip(challenger_vector, incumbent_vector, strict=True)]
    passed = delta[0] >= 0.005 and delta[1] >= 0.0 and delta[2] >= 0.0
    return {
        "identity_count": 36,
        "incumbent": "exact pre-APNCC c=0.25 held-44b6 control",
        "challenger": "banked APNCC-DIRECT held-44b6 graph",
        "incumbent_adjusted_raw_node": incumbent_vector,
        "challenger_adjusted_raw_node": challenger_vector,
        "delta_adjusted_raw_node": delta,
        "gate": {
            "threshold_adjusted_raw_node": [0.005, 0.0, 0.0],
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
        },
        "provenance": {
            "incumbent_report": str(PRE_APNCC_REPORT),
            "incumbent_report_sha256": sha256(PRE_APNCC_REPORT),
            "incumbent_fragment_aggregate_sha256": incumbent["input_provenance"][
                "aggregate_sha256"
            ],
            "challenger_report": str(APNCC_REPORT),
            "challenger_report_sha256": sha256(APNCC_REPORT),
            "challenger_fragment_aggregate_sha256": challenger["input_provenance"][
                "aggregate_sha256"
            ],
        },
    }


def delta_counts(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "total": int(array.size),
        "beneficial": int(np.count_nonzero(array > 0.0)),
        "neutral": int(np.count_nonzero(array == 0.0)),
        "harmful": int(np.count_nonzero(array < 0.0)),
        "beneficial_delta_sum": float(array[array > 0.0].sum()),
        "max_delta": float(array.max()) if array.size else None,
    }


def match_r0_detections_to_gt(
    detections: tuple[Any, ...], gt: Mapping[str, Any]
) -> tuple[dict[int, int | None], set[tuple[int, int]]]:
    """Exact torch-free replay of edge_data.match_detections_to_gt."""
    from celltrack_alpha.bh_graph import (
        PHYSICAL_SCALE_ZYX,
        sparse_min_cost_max_cardinality,
    )

    nodes = gt["nodes"]
    ids = np.asarray(nodes["node_id"], dtype=np.int64)
    times = np.asarray(nodes["t"], dtype=np.int64)
    coords = np.column_stack((nodes["z"], nodes["y"], nodes["x"])).astype(
        np.float64, copy=False
    )
    raw_edges = np.asarray(gt["edges"], dtype=np.int64)
    by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame in sorted(set(times.tolist())):
        rows = np.flatnonzero(times == frame)
        by_frame[int(frame)] = (ids[rows], coords[rows])
    gt_edges = {(int(source), int(target)) for source, target in raw_edges.tolist()}
    result: dict[int, int | None] = {}
    scale = np.asarray(PHYSICAL_SCALE_ZYX, dtype=np.float64)
    for detection in detections:
        detection_ids = [int(value) for value in detection.node_ids]
        result.update({node_id: None for node_id in detection_ids})
        gt_ids, gt_coords = by_frame.get(
            detection.frame_index,
            (np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float64)),
        )
        if not detection_ids or len(gt_ids) == 0:
            continue
        pred_physical = np.asarray(detection.coords_zyx, dtype=np.float64) * scale
        gt_physical = np.asarray(gt_coords, dtype=np.float64) * scale
        tree = cKDTree(gt_physical)
        feasible: list[tuple[int, int, float]] = []
        for row, candidates in enumerate(tree.query_ball_point(pred_physical, r=7.0)):
            for candidate in sorted(int(value) for value in candidates):
                distance = float(
                    np.linalg.norm(pred_physical[row] - gt_physical[candidate])
                )
                if distance <= 7.0:
                    feasible.append(
                        (detection_ids[row], int(gt_ids[candidate]), distance)
                    )
        if feasible:
            match = sparse_min_cost_max_cardinality(
                detection_ids, gt_ids.tolist(), feasible
            )
            for row in match.matches:
                result[int(row.source_id)] = int(row.target_id)
    return result, gt_edges


def r0_label(
    pair: tuple[int, int],
    detection_to_gt: Mapping[int, int | None],
    gt_edges: set[tuple[int, int]],
    outdegree: Mapping[int, int],
    indegree: Mapping[int, int],
) -> int:
    source_gt = detection_to_gt.get(pair[0])
    target_gt = detection_to_gt.get(pair[1])
    if source_gt is None or target_gt is None:
        return -1
    mapped = (int(source_gt), int(target_gt))
    if mapped not in gt_edges:
        return 0
    if outdegree[mapped[0]] != 1 or indegree[mapped[1]] != 1:
        return -2
    return 1


def occupied_swap_population() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = load_json(POPULATION_MANIFEST)
    arrays: dict[str, np.ndarray] = {}
    total = 0
    for row in manifest["classes"]["occupied_swap"]["videos"]:
        identity = str(row["identity"])
        path = ROOT / str(row["population_path"])
        if not path.is_file() or sha256(path) != str(row["population_sha256"]):
            raise DiagnosticError(f"population identity mismatch: {identity}")
        with np.load(path, allow_pickle=False) as loaded:
            values = np.asarray(loaded["oracle_delta_S"], dtype=np.float64)
            labels = np.asarray(loaded["oracle_label"], dtype=np.uint8)
        if values.shape != (int(row["count"]),) or not np.array_equal(labels, values > 0.0):
            raise DiagnosticError(f"oracle alignment mismatch: {identity}")
        arrays[identity] = values
        total += len(values)
    if total != 45_961:
        raise DiagnosticError(f"occupied-swap population drifted: {total}")
    return manifest, arrays


def d2_a7(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    values = np.concatenate(list(arrays.values()))
    counts = delta_counts(values)
    return {
        "candidate_universe": "all occupied-endpoint swap actions",
        "oracle": counts,
        "threshold_accepted": 0,
        "executed_edits": 0,
        "diagnosis": (
            "VACUOUS_ABSTENTION" if counts["beneficial"] > 0 else "NO_BENEFICIAL_SUPPORT"
        ),
    }


def d2_reverse(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    reverse = load_module("reverse_selector_for_diagnostic", REVERSE_SELECTOR)
    reverse_manifest = load_json(REVERSE_MANIFEST)
    apncc = load_json(APNCC_REPORT)
    selection = load_json(REVERSE_SELECTION)
    reverse_rows = {str(row["identity"]): row for row in reverse_manifest["files"]}
    control_rows = {
        str(row["identity"]): row
        for row in apncc["graph"]["control"]["input_provenance"]["files"]
    }
    read_submission = reverse.load_graph_reader(ROOT)
    passing: list[tuple[str, int]] = []
    population_count = 0
    for video_row in manifest["classes"]["occupied_swap"]["videos"]:
        identity = str(video_row["identity"])
        request_path = ROOT / str(video_row["request_path"])
        if sha256(request_path) != str(video_row["request_sha256"]):
            raise DiagnosticError(f"request identity mismatch: {identity}")
        reverse_row = reverse_rows[identity]
        reverse_path = REVERSE_MANIFEST.parent / str(reverse_row["path"])
        if sha256(reverse_path) != str(reverse_row["sha256"]):
            raise DiagnosticError(f"reverse identity mismatch: {identity}")
        _graph, baseline_edges = reverse.graph_contract(
            read_submission, identity, control_rows[identity]
        )
        by_source: defaultdict[int, set[tuple[int, int]]] = defaultdict(set)
        by_target: defaultdict[int, set[tuple[int, int]]] = defaultdict(set)
        for edge in baseline_edges:
            by_source[edge[0]].add(edge)
            by_target[edge[1]].add(edge)
        proposals = reverse.reverse_proposals(reverse_path)
        with np.load(request_path, allow_pickle=False) as loaded:
            sources = np.asarray(loaded["source_id"], dtype=np.int64)
            targets = np.asarray(loaded["target_id"], dtype=np.int64)
        if sources.shape != arrays[identity].shape or targets.shape != sources.shape:
            raise DiagnosticError(f"reverse request alignment mismatch: {identity}")
        population_count += len(sources)
        for index, (source, target) in enumerate(zip(sources, targets, strict=True)):
            pair = (int(source), int(target))
            expected_conflicts = by_source[pair[0]] | by_target[pair[1]]
            passed, _evidence, _reason = reverse.evidence_pass(
                pair, expected_conflicts, proposals
            )
            if passed:
                passing.append((identity, index))
    if population_count != 45_961 or len(passing) != 268:
        raise DiagnosticError(
            f"reverse population/threshold drifted: {population_count}/{len(passing)}"
        )
    passing_values = np.asarray([arrays[i][j] for i, j in passing], dtype=np.float64)
    selected_keys = {
        (str(row["identity"]), int(row["index"])) for row in selection["selected"]
    }
    selected_values = np.asarray(
        [arrays[identity][index] for identity, index in passing if (identity, index) in selected_keys],
        dtype=np.float64,
    )
    if len(selected_values) != 53:
        raise DiagnosticError(f"reverse selected set drifted: {len(selected_values)}")
    full_values = np.concatenate(list(arrays.values()))
    passing_counts = delta_counts(passing_values)
    selected_counts = delta_counts(selected_values)
    return {
        "full_occupied_swap_universe": delta_counts(full_values),
        "literal_evidence_pass_set": passing_counts,
        "node_disjoint_selected_set": selected_counts,
        "executed_edits": 0,
        "diagnosis": (
            "EVIDENCE_PASS_SET_HAS_SIGNAL_BUT_SELECTOR_LOST_IT"
            if passing_counts["beneficial"] > 0 and selected_counts["beneficial"] == 0
            else "SELECTED_EVIDENCE_SET_HAS_SIGNAL"
            if selected_counts["beneficial"] > 0
            else "LITERAL_EVIDENCE_THRESHOLD_HAS_NO_BENEFICIAL_SUPPORT"
        ),
    }


def d2_r0() -> dict[str, Any]:
    manifest = load_json(R0_MANIFEST)
    actions: list[dict[str, Any]] = []
    for row in manifest["shards"]:
        path = Path(str(row["path"]))
        if not path.is_file() or sha256(path) != str(row["sha256"]):
            raise DiagnosticError(f"R0 shard identity mismatch: {row['identity']}")
        shard = load_json(path)
        candidates = shard["candidates"]
        if len(candidates) != int(row["rows"]):
            raise DiagnosticError(f"R0 shard row mismatch: {row['identity']}")
        actions.extend(candidates)
    if len(actions) != 38_081:
        raise DiagnosticError(f"R0 population drifted: {len(actions)}")
    for location in (ROOT / "src", ROOT / "external/Tracking_Support/repo/src"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    from celltrack_alpha import bh_model
    from celltrack_baseline.rule_based import read_geff_arrays

    by_identity: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        by_identity[str(row["identity"])].append(row)
    labels: Counter[int] = Counter()
    with ZipStore(RAW_ZIP, mode="r") as store:
        for identity in sorted(by_identity):
            slug = identity.replace(":", "__", 1)
            meta = load_json(CACHE_ROOT / f"{slug}.json")
            model_identity = bh_model.CacheIdentity(**dict(meta["contract"]["model"]))
            cache = bh_model.load_sparse_cache(
                CACHE_ROOT / f"{slug}.bhc", expected_identity=model_identity
            )
            gt = read_geff_arrays(store, "train", identity.split(":", 1)[1])
            detection_to_gt, gt_edges = match_r0_detections_to_gt(cache.detections, gt)
            outdegree: defaultdict[int, int] = defaultdict(int)
            indegree: defaultdict[int, int] = defaultdict(int)
            for source, target in gt_edges:
                outdegree[source] += 1
                indegree[target] += 1
            for action in by_identity[identity]:
                labels[
                    r0_label(
                        (int(action["source_id"]), int(action["target_id"])),
                        detection_to_gt,
                        gt_edges,
                        outdegree,
                        indegree,
                    )
                ] += 1
    # For a both-ends-free add, labels 1 and -2 are existing GT edges.  The node set
    # is unchanged, so either strictly increases exact edge S(G); 0/-1 adds an FP.
    beneficial = labels[1] + labels[-2]
    selected = load_json(R0_RESULT)["selection"]["actions_with_truth"]
    selected_labels = Counter(int(row["official_fit_label"]) for row in selected)
    return {
        "candidate_universe": "all 10-12 um both-ends-free topology-0 add actions",
        "oracle": {
            "total": len(actions),
            "beneficial": beneficial,
            "beneficial_nondivision": labels[1],
            "beneficial_division_truth_edge": labels[-2],
            "known_negative": labels[0],
            "unmatched_false_positive": labels[-1],
            "label_semantics": (
                "for a free/free add, GT-edge labels 1/-2 have exact delta S(G)>0; "
                "labels 0/-1 have exact delta S(G)<0"
            ),
        },
        "threshold_selected": {
            "total": len(selected),
            "beneficial": selected_labels[1] + selected_labels[-2],
            "labels": {str(key): value for key, value in sorted(selected_labels.items())},
        },
        "executed_edits": 0,
        "diagnosis": "THRESHOLD_SELECTION_MISSED_NONEMPTY_ORACLE_SUPPORT"
        if beneficial > 0
        else "NO_BENEFICIAL_SUPPORT",
    }


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("RESULT.json")
    if output.exists():
        raise DiagnosticError(f"refusing to overwrite: {output}")
    manifest, arrays = occupied_swap_population()
    result = {
        "schema": "g02.gate-validity-abstention-oracle-diagnostic.v1",
        "status": "COMPLETE_FIT_ONLY_CPU_EXISTING_CACHES",
        "constraints": {
            "gpu_used": False,
            "kaggle_used": False,
            "cv57_opened": False,
            "six_bba_opened": False,
        },
        "d1_gate_validity": d1_gate_validity(),
        "d2_abstention_vs_evidence": {
            "a7_phase_template": d2_a7(arrays),
            "reverse_consensus": d2_reverse(manifest, arrays),
            "r0_shell_no_link": d2_r0(),
        },
        "provenance": {
            "population_manifest": str(POPULATION_MANIFEST),
            "population_manifest_sha256": sha256(POPULATION_MANIFEST),
            "reverse_manifest": str(REVERSE_MANIFEST),
            "reverse_manifest_sha256": sha256(REVERSE_MANIFEST),
            "reverse_selection": str(REVERSE_SELECTION),
            "reverse_selection_sha256": sha256(REVERSE_SELECTION),
            "r0_manifest": str(R0_MANIFEST),
            "r0_manifest_sha256": sha256(R0_MANIFEST),
            "r0_result": str(R0_RESULT),
            "r0_result_sha256": sha256(R0_RESULT),
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
