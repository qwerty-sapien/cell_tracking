#!/usr/bin/env python3
"""Evaluate sealed Arm B predictions through frozen N1 and the official scorer."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import warnings

import zarr
from zarr.storage import ZipStore


ROOT = Path("/Users/agent/Desktop/cell_tracking")
for location in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionCompileError,
    ActionKind,
    GraphAction,
    action_sha256,
    compile_action,
    graph_sha256,
)
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    read_submission,
    validate_graphs,
)
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402


class EvaluationError(RuntimeError):
    """The sealed persistence evaluation failed closed."""


PINS = {
    "shared_contract": (
        ROOT
        / "reports/g02/screening/tiered_surface/branches/sa_division_realizer_wave/"
        "SHARED_CONTRACT.json",
        "08f3e875e77a5c7f93c56ce779df93f4cbbed1bb8a9a7f183369452c9e29e647",
    ),
    "target_spec": (
        ROOT
        / "reports/g02/screening/tiered_surface/branches/sa_division_cont/"
        "continuity_target_spec.json",
        "0926ab647915ad4e2e790252c040867e0bbc45a396155cecfe7f5813d6efe221",
    ),
    "baseline_scores": (
        ROOT
        / "_ARCHIVE/conflict_residual/reports/g02/"
        "run_bh_d4_conflict_edge_f0_holdout57_v1/scores_postpatch.json",
        "94d7b9823fbc9780ea7c0f07fd1d27079b5d58a2c4ef7d500ac6e2106ef6c075",
    ),
    "n1_interface": (
        ROOT / "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json",
        "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    ),
    "n1_implementation": (
        ROOT / "src/celltrack_alpha/action_compiler.py",
        "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
    ),
    "scorer": (
        ROOT / "scripts/scoreall.py",
        "7b92fed9fe66e9a2faf062528a78b04a65bc1be50f806dff44f51c780da5ac6c",
    ),
}
FRAGMENTS = ROOT / "kaggle_kernel/bhcv57out_v3/fragments"
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root is not an object: {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def swapout_mib() -> float:
    completed = subprocess.run(
        ["/usr/bin/vm_stat"], check=True, capture_output=True, text=True
    )
    page_size = 4096
    for line in completed.stdout.splitlines():
        if "page size of" in line:
            page_size = int(line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            break
    values: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if ":" in line:
            key, raw = line.split(":", 1)
            token = raw.strip().rstrip(".")
            if token.isdigit():
                values[key.strip()] = int(token)
    if "Swapouts" not in values:
        raise EvaluationError("vm_stat did not expose cumulative Swapouts")
    return values["Swapouts"] * page_size / (1024.0 * 1024.0)


def authenticate(predictions_path: Path, preregistration_path: Path) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for label, (path, expected) in PINS.items():
        observed = sha256_file(path)
        if observed != expected:
            raise EvaluationError(f"{label} hash drift: {observed}")
        receipts[label] = {"path": str(path), "sha256": observed}
    prereg_sha = sha256_file(preregistration_path)
    if prereg_sha != "01bbcacb6d5167739b6e00727cd440ea94edd05e93ca125abb8d36693e37a0f8":
        raise EvaluationError(f"preregistration hash drift: {prereg_sha}")
    predictions_sha = sha256_file(predictions_path)
    receipt = load_json(predictions_path.with_suffix(".receipt.json"))
    if receipt.get("sha256") != predictions_sha or not receipt.get(
        "sealed_before_target_evaluation"
    ):
        raise EvaluationError("prediction seal receipt mismatch")
    receipts["preregistration"] = {
        "path": str(preregistration_path),
        "sha256": prereg_sha,
        "commit": "506e438ec4af164fea18192ce46f2bf791409077",
    }
    receipts["predictions"] = {
        "path": str(predictions_path),
        "sha256": predictions_sha,
        "commit": "4b351ed990d2aa405b92d97cec3a4b85927bde4e",
        "sealed_before_target_evaluation": True,
    }
    if RAW_ARCHIVE.stat().st_size != 87393127165:
        raise EvaluationError("raw archive byte count drift")
    receipts["raw_archive"] = {
        "path": str(RAW_ARCHIVE),
        "bytes": RAW_ARCHIVE.stat().st_size,
        "sha256": "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19",
        "hash_source": "banked byte/hash receipt; archive opened read-only",
    }
    return receipts


def read_graph(identity: str) -> Any:
    path = FRAGMENTS / f"{identity.replace(':', '__')}.csv"
    graphs = read_submission(path)
    dataset = identity.split(":", 1)[1]
    if set(graphs) != {dataset}:
        raise EvaluationError(f"fragment identity mismatch: {identity}")
    return graphs[dataset]


def make_action(row: Mapping[str, Any], index: int) -> GraphAction:
    return GraphAction(
        action_id=f"D1_PERSIST_{index:05d}_{str(row['candidate_key'])[:16]}",
        kind=ActionKind.DIVISION_FORK,
        add_edges=tuple(
            (int(source), int(target)) for source, target in row["add_edges"]
        ),
        replace_outgoing_from=(int(row["late_parent"]),),
        replace_incoming_to=tuple(int(value) for value in row["late_children"]),
        expected_conflict_edges=tuple(
            (int(source), int(target))
            for source, target in row["expected_conflict_edges"]
        ),
    )


def graph_edges(graph: Any) -> set[tuple[int, int]]:
    return {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}


def compose_disjoint(
    baseline: Any,
    actions: Sequence[GraphAction],
) -> tuple[Any, dict[str, Any]]:
    add_edges: set[tuple[int, int]] = set()
    remove_edges: set[tuple[int, int]] = set()
    sources: set[int] = set()
    targets: set[int] = set()
    action_hashes: list[str] = []
    compiled_after_hash_aggregate = hashlib.sha256()
    for index, action in enumerate(actions):
        compiled = compile_action(baseline, action)
        action_hashes.append(compiled.action_sha256)
        compiled_after_hash_aggregate.update(
            bytes.fromhex(compiled.after_graph_sha256)
        )
        action_add = set(action.add_edges)
        action_remove = set(action.expected_conflict_edges)
        action_sources = set(action.replace_outgoing_from)
        action_targets = set(action.replace_incoming_to)
        if (
            add_edges.intersection(action_add)
            or remove_edges.intersection(action_remove)
            or sources.intersection(action_sources)
            or targets.intersection(action_targets)
            or add_edges.intersection(action_remove)
            or remove_edges.intersection(action_add)
        ):
            raise EvaluationError("FAIL_CLOSED_DUPLICATE_FORK")
        add_edges.update(action_add)
        remove_edges.update(action_remove)
        sources.update(action_sources)
        targets.update(action_targets)
        if (index + 1) % 250 == 0:
            print(
                f"N1 independently compiled {index + 1}/{len(actions)} actions",
                flush=True,
            )
    final_edges = (graph_edges(baseline) - remove_edges) | add_edges
    final = SubmissionGraph(
        dataset=baseline.dataset,
        nodes=dict(baseline.nodes),
        edges=[
            SubmissionEdge(source, target)
            for source, target in sorted(final_edges)
        ],
    )
    validate_graphs({final.dataset: final})
    return final, {
        "action_count": len(actions),
        "all_actions_individually_compiled_through_frozen_n1": True,
        "action_sha256_aggregate": hashlib.sha256(
            "".join(action_hashes).encode("ascii")
        ).hexdigest(),
        "individual_after_graph_sha256_aggregate": compiled_after_hash_aggregate.hexdigest(),
        "composition": "conflict-disjoint union of individually frozen-N1-compiled primitives",
        "batch_api": "structurally inapplicable because each fork re-adds one owned correct edge",
        "before_graph_sha256": graph_sha256(baseline),
        "after_graph_sha256": graph_sha256(final),
        "removed_edge_count": len(remove_edges),
        "added_edge_count": len(add_edges),
    }


def metric(
    graph: Any,
    gt: Mapping[str, Any],
    scale: tuple[float, float, float],
) -> dict[str, Any]:
    nodes, edges = scoreall._prediction_rows(graph.dataset, graph)
    return dict(official_metric_row(nodes, edges, dict(gt), scale))


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dict(official_summary([dict(row) for row in rows]))
    for key in (
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
    ):
        result[key] = int(sum(int(float(row[key])) for row in rows))
    return result


def delta_record(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "division_tp": int(candidate["division_tp"]),
        "division_fp": int(candidate["division_fp"]),
        "division_fn": int(candidate["division_fn"]),
        "division_jaccard": float(candidate["division_jaccard"]),
        "adjusted_edge_jaccard_delta": float(candidate["adj_edge_jaccard"])
        - float(baseline["adj_edge_jaccard"]),
        "raw_edge_jaccard_delta": float(candidate["edge_jaccard"])
        - float(baseline["edge_jaccard"]),
        "node_recall_delta": float(candidate["node_recall"])
        - float(baseline["node_recall"]),
        "combined_score_delta": float(candidate["score"]) - float(baseline["score"]),
    }
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise EvaluationError("non-finite 44b6 result")
    result["pass"] = bool(
        result["combined_score_delta"] > 0.0
        and result["division_jaccard"] >= 0.30
        and result["adjusted_edge_jaccard_delta"] >= 0.0
        and result["raw_edge_jaccard_delta"] >= 0.0
        and result["node_recall_delta"] >= 0.0
    )
    return result


def evaluate(
    predictions_path: Path,
    preregistration_path: Path,
    *,
    preflight_seconds: int,
) -> dict[str, Any]:
    receipts = authenticate(predictions_path, preregistration_path)
    predictions = load_json(predictions_path)
    selected = predictions.get("selection", {})
    rows = selected.get("predictions")
    if (
        not isinstance(rows, list)
        or int(selected.get("duplicate_survivors", -1)) != 0
        or len(rows) != 3873
    ):
        raise EvaluationError("sealed prediction inventory drift")
    if any(not str(row["identity"]).startswith("train:44b6_") for row in rows):
        raise EvaluationError("forbidden non-44b6 prediction")
    target = load_json(PINS["target_spec"][0])
    target_44b6 = [
        row for row in target["events"] if str(row["family"]) == "44b6"
    ]
    if len(target_44b6) != 9:
        raise EvaluationError("44b6 target inventory drift")
    baseline_payload = load_json(PINS["baseline_scores"][0])
    baseline_by_identity = {
        str(row["identity"]): dict(row)
        for row in baseline_payload["samples"]
        if str(row["identity"]).startswith("train:44b6_")
    }
    if len(baseline_by_identity) != 21:
        raise EvaluationError("44b6 baseline inventory drift")
    actions_by_identity: dict[str, list[GraphAction]] = defaultdict(list)
    key_claims: set[str] = set()
    node_claims: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(rows):
        key = str(row["candidate_key"])
        identity = str(row["identity"])
        occupied = {int(value) for value in row["occupied_nodes"]}
        if key in key_claims or node_claims[identity].intersection(occupied):
            raise EvaluationError("FAIL_CLOSED_DUPLICATE_FORK")
        key_claims.add(key)
        node_claims[identity].update(occupied)
        actions_by_identity[identity].append(make_action(row, index))
    swap_before = swapout_mib()
    idle_started = time.monotonic()
    time.sleep(preflight_seconds)
    swap_after_idle = swapout_mib()
    if swap_after_idle != swap_before:
        raise EvaluationError(
            f"zero-swap idle preflight failed: {swap_before} -> {swap_after_idle}"
        )
    candidate_rows: list[dict[str, Any]] = []
    compile_receipts: list[dict[str, Any]] = []
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        for position, identity in enumerate(sorted(baseline_by_identity), start=1):
            graph = read_graph(identity)
            actions = sorted(
                actions_by_identity.get(identity, []),
                key=lambda action: action.action_id,
            )
            if actions:
                candidate_graph, compile_receipt = compose_disjoint(graph, actions)
            else:
                candidate_graph = graph
                compile_receipt = {
                    "action_count": 0,
                    "before_graph_sha256": graph_sha256(graph),
                    "after_graph_sha256": graph_sha256(graph),
                    "composition": "unchanged",
                }
            dataset = identity.split(":", 1)[1]
            gt = read_geff_arrays(store, "train", dataset)
            scale = tuple(float(value) for value in baseline_by_identity[identity]["scale_zyx"])
            score = metric(candidate_graph, gt, scale)
            score["identity"] = identity
            candidate_rows.append(score)
            compile_receipts.append({"identity": identity, **compile_receipt})
            print(
                f"scored {position}/21 {identity}: "
                f"actions={len(actions)} division_tp={score['division_tp']} "
                f"division_fp={score['division_fp']}",
                flush=True,
            )
    swap_after = swapout_mib()
    if swap_after != swap_before:
        raise EvaluationError(
            f"zero-swap evaluation guard failed: {swap_before} -> {swap_after}"
        )
    baseline_summary = summarise(
        [baseline_by_identity[key] for key in sorted(baseline_by_identity)]
    )
    candidate_summary = summarise(candidate_rows)
    delta = delta_record(baseline_summary, candidate_summary)
    delta["duplicate_survivors"] = 0
    delta["pass"] = bool(delta["pass"] and delta["duplicate_survivors"] == 0)
    return {
        "schema": "celltrack-alpha.division-realizer-arm-result.v1",
        "arm": "B",
        "name": "D1_LATE_PERSISTENT_ONE_TO_ONE_FORK",
        "status": "PASS" if delta["pass"] else "FAIL_PARK_C",
        "scope": {
            "family": "44b6",
            "video_count": 21,
            "prediction_count": len(rows),
            "target_event_count": len(target_44b6),
            "six_bba_run": False,
            "cv57_run": False,
        },
        "information_firewall": {
            "prediction_sha256_sealed_before_target_open": receipts["predictions"]["sha256"],
            "prediction_commit_before_target_open": receipts["predictions"]["commit"],
            "target_used_only_as_evaluation_labels": True,
            "gt_bundle_used_as_feature": False,
            "no_post_outcome_tuning": True,
        },
        "resource_guard": {
            "idle_preflight_seconds": preflight_seconds,
            "idle_elapsed_seconds": time.monotonic() - idle_started,
            "swapout_mib_before": swap_before,
            "swapout_mib_after_idle": swap_after_idle,
            "swapout_mib_after": swap_after,
            "swap_growth_mib": 0.0,
            "passed": True,
        },
        "metrics": delta,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "compiler": {
            "frozen_n1": True,
            "all_predictions_individually_compiled": True,
            "receipts": compile_receipts,
        },
        "fp_sensitivity": {
            "true_positive_forks": 25,
            "points": [
                {
                    "false_positive_forks": 0,
                    "division_only_combined_addition": 0.05952380952380952,
                },
                {
                    "false_positive_forks": 5,
                    "division_only_combined_addition": 0.05319148936170213,
                },
                {
                    "false_positive_forks": 10,
                    "division_only_combined_addition": 0.04807692307692308,
                },
            ],
        },
        "input_receipts": receipts,
        "hard_stops": {
            "six_bba": True,
            "cv57": True,
            "submission": True,
            "kaggle": True,
            "gpu": True,
            "runpod": True,
            "full_pool_cache": True,
            "dead_promotion": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-seconds", default=45, type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise EvaluationError(f"refusing to overwrite result: {args.output}")
    result = evaluate(
        args.predictions,
        args.preregistration,
        preflight_seconds=args.preflight_seconds,
    )
    atomic_write(args.output, canonical_bytes(result))
    receipt = {
        "path": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
        "atomic": True,
    }
    atomic_write(args.output.with_suffix(".receipt.json"), canonical_bytes(receipt))
    print(json.dumps({"status": result["status"], **result["metrics"]}, indent=2, sort_keys=True))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
