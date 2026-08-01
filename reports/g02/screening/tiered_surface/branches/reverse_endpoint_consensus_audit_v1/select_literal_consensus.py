#!/usr/bin/env python3
"""Seal a label-blind literal forward/reverse endpoint-consensus edit set."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


PREREG_SHA256 = "9d0c578b4ef09de25556096c4d31857aa14be6646183755d9954e86311d36b91"
SUPPLIER_SHA256 = "499f022666211d339bbe9a1ff551bcaea20a88a66ae18b0cc0b030b543376918"
REVERSE_MANIFEST_SHA256 = "654cb499d65db137d71e3915850d4bc94a5373660cb2026507ad070275fb6fe1"
POPULATION_MANIFEST_SHA256 = "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
APNCC_REPORT_SHA256 = "530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d"
FORWARD_MARGIN_MIN = 0.50
REVERSE_MARGIN_MIN = 0.50
RETURN_OVER_NULL_MIN = 0.25
PER_VIDEO_CAP = 2
GLOBAL_CAP = 64


class ConsensusError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_json(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected:
        raise ConsensusError(f"JSON identity mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConsensusError(f"JSON object required: {path}")
    return value


def load_graph_reader(workspace_root: Path) -> Any:
    src = workspace_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from celltrack_alpha.kaggle_calibration import read_submission

    return read_submission


def graph_contract(
    read_submission: Any, identity: str, row: Mapping[str, Any]
) -> tuple[Any, set[tuple[int, int]]]:
    path = Path(str(row["path"]))
    if not path.is_file() or sha256(path) != str(row["sha256"]):
        raise ConsensusError(f"APNCC fragment identity mismatch: {identity}")
    dataset = identity.split(":", 1)[1]
    graphs = read_submission(path)
    graph = graphs.get(dataset)
    if graph is None or len(graphs) != 1:
        raise ConsensusError(f"APNCC fragment graph mismatch: {identity}")
    edges = {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}
    return graph, edges


def reverse_proposals(path: Path) -> dict[tuple[int, int], list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as loaded:
        z = {key: loaded[key] for key in loaded.files}
    required = {
        "challengers",
        "conflicts",
        "proposal_offsets",
        "forward_margin",
        "reverse_challenger_score",
        "reverse_conflict_score",
        "reverse_margin",
        "reverse_challenger_null",
        "cycle_sign_agreement",
    }
    if not required <= set(z):
        raise ConsensusError(f"reverse payload lacks fields: {path}")
    challengers = np.asarray(z["challengers"], dtype=np.int64)
    conflicts = np.asarray(z["conflicts"], dtype=np.int64)
    offsets = np.asarray(z["proposal_offsets"], dtype=np.int64)
    forward = np.asarray(z["forward_margin"], dtype=np.float64)
    reverse = np.asarray(z["reverse_margin"], dtype=np.float64)
    challenger_score = np.asarray(z["reverse_challenger_score"], dtype=np.float64)
    challenger_null = np.asarray(z["reverse_challenger_null"], dtype=np.float64)
    conflict_score = np.asarray(z["reverse_conflict_score"], dtype=np.float64)
    agreement = np.asarray(z["cycle_sign_agreement"], dtype=np.uint8)
    rows = len(conflicts)
    if (
        challengers.shape != (len(offsets) - 1, 2)
        or int(offsets[-1]) != rows
        or any(
            value.shape != (rows,)
            for value in (
                forward,
                reverse,
                challenger_score,
                challenger_null,
                conflict_score,
                agreement,
            )
        )
        or not all(
            np.isfinite(value).all()
            for value in (
                forward,
                reverse,
                challenger_score,
                challenger_null,
                conflict_score,
            )
        )
    ):
        raise ConsensusError(f"reverse payload alignment drifted: {path}")
    result: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for proposal_index, raw_pair in enumerate(challengers):
        start = int(offsets[proposal_index])
        stop = int(offsets[proposal_index + 1])
        result[tuple(map(int, raw_pair))].append(
            {
                "conflicts": [
                    tuple(map(int, conflicts[index])) for index in range(start, stop)
                ],
                "forward_margin": forward[start:stop],
                "reverse_margin": reverse[start:stop],
                "reverse_challenger_score": challenger_score[start:stop],
                "reverse_challenger_null": challenger_null[start:stop],
                "reverse_conflict_score": conflict_score[start:stop],
                "cycle_sign_agreement": agreement[start:stop],
            }
        )
    return result


def evidence_pass(
    pair: tuple[int, int],
    expected_conflicts: set[tuple[int, int]],
    proposals: Mapping[tuple[int, int], list[dict[str, Any]]],
) -> tuple[bool, dict[str, float] | None, str]:
    matches = proposals.get(pair, [])
    if len(matches) != 1:
        return False, None, "missing_or_duplicate_challenger"
    row = matches[0]
    conflicts = list(row["conflicts"])
    if len(conflicts) != len(set(conflicts)) or set(conflicts) != expected_conflicts:
        return False, None, "removed_edge_alignment"
    if not conflicts:
        return False, None, "empty_conflict_set"
    forward = np.asarray(row["forward_margin"], dtype=np.float64)
    reverse = np.asarray(row["reverse_margin"], dtype=np.float64)
    return_gap = np.asarray(row["reverse_challenger_score"], dtype=np.float64) - np.asarray(
        row["reverse_challenger_null"], dtype=np.float64
    )
    agreement = np.asarray(row["cycle_sign_agreement"], dtype=np.uint8)
    passed = bool(
        np.all(forward >= FORWARD_MARGIN_MIN)
        and np.all(reverse >= REVERSE_MARGIN_MIN)
        and np.all(return_gap >= RETURN_OVER_NULL_MIN)
        and np.all(agreement == 1)
    )
    summary = {
        "min_forward_margin": float(forward.min()),
        "min_reverse_margin": float(reverse.min()),
        "min_reverse_return_over_null": float(return_gap.min()),
        "min_reverse_challenger_score": float(
            np.asarray(row["reverse_challenger_score"], dtype=np.float64).min()
        ),
        "max_reverse_conflict_score": float(
            np.asarray(row["reverse_conflict_score"], dtype=np.float64).max()
        ),
    }
    return passed, summary, "pass" if passed else "threshold"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ConsensusError(f"refusing to overwrite: {args.output}")
    prereg = checked_json(args.preregistration, PREREG_SHA256)
    supplier = checked_json(args.supplier_contract, SUPPLIER_SHA256)
    reverse_manifest = checked_json(args.reverse_manifest, REVERSE_MANIFEST_SHA256)
    population = checked_json(args.population_manifest, POPULATION_MANIFEST_SHA256)
    apncc = checked_json(args.apncc_report, APNCC_REPORT_SHA256)
    if (
        prereg.get("status") != "FROZEN_BEFORE_LITERAL_CONSENSUS_LABEL_OR_GRAPH_READ"
        or supplier.get("status") != "COMPLETE_AUTHENTIC_NON_GT_IDENTIFIABLE"
        or reverse_manifest.get("status") != "COMPLETE"
        or reverse_manifest.get("classification") != "NON-GT-IDENTIFIABLE"
        or reverse_manifest.get("determinism", {}).get("all_array_equal") is not True
        or population.get("status") != "COMPLETE"
    ):
        raise ConsensusError("frozen contracts are not complete")
    rows = population["classes"]["occupied_swap"]["videos"]
    reverse_rows = {str(row["identity"]): row for row in reverse_manifest["files"]}
    control_rows = {
        str(row["identity"]): row
        for row in apncc["graph"]["control"]["input_provenance"]["files"]
    }
    if len(rows) != 36 or len(reverse_rows) != 36 or set(reverse_rows) != set(control_rows):
        raise ConsensusError("expected exact 36-video common universe")
    read_submission = load_graph_reader(args.workspace_root)
    candidates: list[dict[str, Any]] = []
    rejection_counts: defaultdict[str, int] = defaultdict(int)
    population_count = 0
    for video_row in rows:
        identity = str(video_row["identity"])
        request_path = args.workspace_root / str(video_row["request_path"])
        if not request_path.is_file() or sha256(request_path) != str(video_row["request_sha256"]):
            raise ConsensusError(f"request identity mismatch: {identity}")
        reverse_row = reverse_rows.get(identity)
        if reverse_row is None:
            raise ConsensusError(f"reverse supplier missing: {identity}")
        reverse_path = args.reverse_manifest.parent / str(reverse_row["path"])
        if not reverse_path.is_file() or sha256(reverse_path) != str(reverse_row["sha256"]):
            raise ConsensusError(f"reverse payload identity mismatch: {identity}")
        graph, baseline_edges = graph_contract(read_submission, identity, control_rows[identity])
        by_source: defaultdict[int, set[tuple[int, int]]] = defaultdict(set)
        by_target: defaultdict[int, set[tuple[int, int]]] = defaultdict(set)
        for edge in baseline_edges:
            by_source[edge[0]].add(edge)
            by_target[edge[1]].add(edge)
        proposals = reverse_proposals(reverse_path)
        with np.load(request_path, allow_pickle=False) as loaded:
            source = np.asarray(loaded["source_id"], dtype=np.int64)
            target = np.asarray(loaded["target_id"], dtype=np.int64)
            source_t = np.asarray(loaded["source_t"], dtype=np.int64)
            target_t = np.asarray(loaded["target_t"], dtype=np.int64)
        count = int(video_row["count"])
        if (
            source.shape != (count,)
            or target.shape != (count,)
            or source_t.shape != (count,)
            or target_t.shape != (count,)
            or not np.array_equal(target_t, source_t + 1)
        ):
            raise ConsensusError(f"request alignment drifted: {identity}")
        population_count += count
        for index in range(count):
            pair = (int(source[index]), int(target[index]))
            if pair in baseline_edges:
                raise ConsensusError(f"occupied swap is already present: {identity} {pair}")
            expected_conflicts = by_source[pair[0]] | by_target[pair[1]]
            passed, evidence, reason = evidence_pass(pair, expected_conflicts, proposals)
            if not passed:
                rejection_counts[reason] += 1
                continue
            touched = {pair[0], pair[1]}
            for conflict in expected_conflicts:
                touched.update(conflict)
            candidates.append(
                {
                    "identity": identity,
                    "index": index,
                    "source_t": int(source_t[index]),
                    "source_id": pair[0],
                    "target_id": pair[1],
                    "removed_edges": [list(edge) for edge in sorted(expected_conflicts)],
                    "owned_nodes": sorted(touched),
                    "evidence": evidence,
                }
            )
    if population_count != 45_961:
        raise ConsensusError(f"occupied-swap population drifted: {population_count}")
    candidates.sort(
        key=lambda row: (
            row["identity"],
            row["source_t"],
            row["source_id"],
            row["target_id"],
        )
    )
    selected: list[dict[str, Any]] = []
    per_video: defaultdict[str, int] = defaultdict(int)
    owned: defaultdict[str, set[int]] = defaultdict(set)
    ownership_rejected = per_video_rejected = global_cap_rejected = 0
    for candidate in candidates:
        identity = str(candidate["identity"])
        if len(selected) >= GLOBAL_CAP:
            global_cap_rejected += 1
            continue
        if per_video[identity] >= PER_VIDEO_CAP:
            per_video_rejected += 1
            continue
        touched = set(map(int, candidate["owned_nodes"]))
        if touched & owned[identity]:
            ownership_rejected += 1
            continue
        selected.append(candidate)
        per_video[identity] += 1
        owned[identity].update(touched)
    disabled_reverse_accept_count = 0
    return {
        "schema": "g02.reverse-endpoint-consensus.selection.v1",
        "status": "SEALED_LABEL_BLIND_SELECTION",
        "preregistration_sha256": PREREG_SHA256,
        "supplier_contract_sha256": SUPPLIER_SHA256,
        "reverse_manifest_sha256": REVERSE_MANIFEST_SHA256,
        "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "scope": {
            "family": "44b6",
            "videos": 36,
            "population": population_count,
            "action_class": "occupied_swap",
            "labels_read": False,
            "graph_scored": False,
        },
        "thresholds": {
            "forward_margin_min": FORWARD_MARGIN_MIN,
            "reverse_margin_min": REVERSE_MARGIN_MIN,
            "reverse_return_over_null_min": RETURN_OVER_NULL_MIN,
            "per_video_cap": PER_VIDEO_CAP,
            "global_cap": GLOBAL_CAP,
            "ranking": False,
        },
        "selection_counts": {
            "literal_threshold_candidates": len(candidates),
            "selected": len(selected),
            "selected_videos": len({row["identity"] for row in selected}),
            "ownership_rejected": ownership_rejected,
            "per_video_cap_rejected": per_video_rejected,
            "global_cap_rejected": global_cap_rejected,
            "disabled_reverse_accept_count": disabled_reverse_accept_count,
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
        "selected": selected,
        "properties_prelabel": {
            "appearance_grounded": True,
            "narrow": len(selected) <= GLOBAL_CAP,
            "node_disjoint": True,
            "recall_preserving_action_kind": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--supplier-contract", type=Path, required=True)
    parser.add_argument("--reverse-manifest", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--apncc-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["selection_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
