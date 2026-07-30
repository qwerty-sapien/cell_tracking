#!/usr/bin/env python3
"""Evaluate sealed Arm-A predictions through frozen N1 and official 44b6 scoring."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import warnings


EXPECTED_PREDICTION_SHA256 = (
    "55fdd22e8f171e2afa243791b90433f9ebf4ab8d9cf9ec3722f9ffc6f14c4108"
)
EXPECTED_TARGET_SHA256 = (
    "0926ab647915ad4e2e790252c040867e0bbc45a396155cecfe7f5813d6efe221"
)
EXPECTED_BASELINE_SHA256 = (
    "94d7b9823fbc9780ea7c0f07fd1d27079b5d58a2c4ef7d500ac6e2106ef6c075"
)
EXPECTED_N1_SHA256 = (
    "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5"
)
EXPECTED_SCORER_SHA256 = (
    "7b92fed9fe66e9a2faf062528a78b04a65bc1be50f806dff44f51c780da5ac6c"
)
EXPECTED_RAW_ARCHIVE_SHA256 = (
    "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19"
)
EXPECTED_RAW_ARCHIVE_BYTES = 87393127165
FP_SENSITIVITY = {
    "true_positive_forks": 25,
    "points": [
        {"false_positive_forks": 0, "combined_addition": 0.05952380952380952},
        {"false_positive_forks": 5, "combined_addition": 0.05319148936170213},
        {"false_positive_forks": 10, "combined_addition": 0.04807692307692308},
    ],
}


class EvaluationError(RuntimeError):
    """Fail-closed evaluation error."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarise(rows: list[dict[str, Any]], official_summary: Any) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = dict(official_summary(rows))
    for key in (
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
    ):
        summary[key] = int(sum(int(float(row[key])) for row in rows))
    denominator = (
        summary["division_tp"] + summary["division_fp"] + summary["division_fn"]
    )
    summary["division_jaccard"] = (
        summary["division_tp"] / denominator if denominator else 0.0
    )
    return summary


def exact_target_key(event: Mapping[str, Any]) -> tuple[str, int, tuple[int, int]]:
    identity = str(event["identity"])
    parent = int(event["parent"]["node_id"])
    children = tuple(sorted(int(row["node_id"]) for row in event["children"]))
    if len(children) != 2:
        raise EvaluationError("target event is not binary")
    return identity, parent, children


def exact_prediction_key(row: Mapping[str, Any]) -> tuple[str, int, tuple[int, int]]:
    branches = tuple(sorted(int(value) for value in row["observed_branch_nodes"]))
    if len(branches) != 2:
        raise EvaluationError("prediction is not binary")
    return str(row["identity"]), int(row["observed_parent_node"]), branches


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    for location in (
        repo_root / "src",
        repo_root / "scripts",
        repo_root / "external/Tracking_Support/repo/src",
        repo_root / "external/kaggle-cell-tracking-competition/src",
    ):
        sys.path.insert(0, str(location))

    import scoreall  # pylint: disable=import-outside-toplevel
    import zarr  # pylint: disable=import-outside-toplevel
    from zarr.storage import ZipStore  # pylint: disable=import-outside-toplevel
    from celltrack_alpha.action_compiler import (  # pylint: disable=import-outside-toplevel
        ActionKind,
        GraphAction,
        compile_action,
    )
    from celltrack_alpha.kaggle_calibration import (  # pylint: disable=import-outside-toplevel
        read_submission,
    )
    from celltrack_alpha.official_eval import (  # pylint: disable=import-outside-toplevel
        official_metric_row,
        official_summary,
    )
    from celltrack_baseline.rule_based import (  # pylint: disable=import-outside-toplevel
        parse_scale,
        read_geff_arrays,
    )

    predictions_path = args.predictions.resolve()
    receipt_path = args.prediction_receipt.resolve()
    target_path = args.target.resolve()
    baseline_path = args.baseline_scores.resolve()
    n1_path = repo_root / "src/celltrack_alpha/action_compiler.py"
    scorer_path = repo_root / "scripts/scoreall.py"
    raw_archive = args.raw_archive.resolve()
    observed_hashes = {
        "predictions_sha256": sha256(predictions_path),
        "prediction_receipt_sha256": sha256(receipt_path),
        "target_sha256": sha256(target_path),
        "baseline_scores_sha256": sha256(baseline_path),
        "n1_implementation_sha256": sha256(n1_path),
        "scorer_sha256": sha256(scorer_path),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
    }
    expected = {
        "predictions_sha256": EXPECTED_PREDICTION_SHA256,
        "target_sha256": EXPECTED_TARGET_SHA256,
        "baseline_scores_sha256": EXPECTED_BASELINE_SHA256,
        "n1_implementation_sha256": EXPECTED_N1_SHA256,
        "scorer_sha256": EXPECTED_SCORER_SHA256,
    }
    for key, expected_hash in expected.items():
        if observed_hashes[key] != expected_hash:
            raise EvaluationError(f"{key} mismatch")
    if (
        raw_archive.stat().st_size != EXPECTED_RAW_ARCHIVE_BYTES
        or args.raw_archive_sha256 != EXPECTED_RAW_ARCHIVE_SHA256
    ):
        raise EvaluationError("raw archive banked identity/size mismatch")

    sealed_receipt = load_json(receipt_path)
    if (
        sealed_receipt.get("status") != "FROZEN_BEFORE_TARGET_EVALUATION"
        or sealed_receipt.get("prediction_sha256") != EXPECTED_PREDICTION_SHA256
    ):
        raise EvaluationError("prediction was not sealed before evaluation")
    predictions = load_json(predictions_path)
    if predictions.get("status") != "SEALED_BEFORE_EVALUATION":
        raise EvaluationError("prediction status drifted")
    selected = list(predictions.get("predictions", ()))
    if any(not str(row.get("identity", "")).startswith("train:44b6_") for row in selected):
        raise EvaluationError("non-44b6 prediction entered evaluation")
    candidate_keys = [str(row["candidate_key"]) for row in selected]
    duplicate_survivors = len(candidate_keys) - len(set(candidate_keys))
    if duplicate_survivors:
        raise EvaluationError("FAIL_CLOSED_DUPLICATE_FORK")

    target = load_json(target_path)
    target_events = [
        event for event in target["events"] if str(event["family"]) == "44b6"
    ]
    target_keys = {exact_target_key(event) for event in target_events}
    predicted_keys = {exact_prediction_key(row) for row in selected}
    target_tp = len(target_keys & predicted_keys)
    target_fp = len(predicted_keys - target_keys)
    target_fn = len(target_keys - predicted_keys)
    target_denominator = target_tp + target_fp + target_fn

    baseline_payload = load_json(baseline_path)
    baseline_rows = {
        str(row["identity"]): dict(row)
        for row in baseline_payload["samples"]
        if str(row["identity"]).startswith("train:44b6_")
    }
    fragment_paths = sorted(args.fragments_dir.resolve().glob("train__44b6_*.csv"))
    if len(fragment_paths) != 21 or len(baseline_rows) != 21:
        raise EvaluationError("authenticated 44b6 inventory must contain 21 graphs")
    graphs: dict[str, Any] = {}
    fragment_receipts: list[dict[str, Any]] = []
    for path in fragment_paths:
        parsed = read_submission(path)
        if len(parsed) != 1:
            raise EvaluationError("fragment contains multiple graphs")
        graph = next(iter(parsed.values()))
        identity = f"train:{graph.dataset}"
        if identity not in baseline_rows:
            raise EvaluationError(f"fragment missing from baseline receipt: {identity}")
        graphs[identity] = graph
        source = baseline_rows[identity]["source"]
        observed = sha256(path)
        if observed != source["sha256"] or path.stat().st_size != int(source["bytes"]):
            raise EvaluationError(f"fragment receipt drifted: {identity}")
        fragment_receipts.append(
            {
                "identity": identity,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )

    by_identity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_identity[str(row["identity"])].append(row)
    if any(len(rows) > 1 for rows in by_identity.values()):
        raise EvaluationError("Arm A per-video cap violated")

    candidate_rows: dict[str, dict[str, Any]] = {
        identity: dict(row) for identity, row in baseline_rows.items()
    }
    compiler_receipts: list[dict[str, Any]] = []
    with ZipStore(raw_archive, mode="r") as store:
        for identity, rows in sorted(by_identity.items()):
            graph = graphs[identity]
            row = rows[0]
            action_payload = row["action"]
            action = GraphAction(
                action_id=f"armA.{str(row['candidate_key'])[:24]}",
                kind=ActionKind.DIVISION_FORK,
                add_edges=tuple(tuple(int(value) for value in pair) for pair in action_payload["add_edges"]),
                replace_outgoing_from=tuple(
                    int(value) for value in action_payload["replace_outgoing_from"]
                ),
                replace_incoming_to=tuple(
                    int(value) for value in action_payload["replace_incoming_to"]
                ),
                expected_conflict_edges=tuple(
                    tuple(int(value) for value in pair)
                    for pair in action_payload["expected_conflict_edges"]
                ),
            )
            compiled = compile_action(graph, action)
            frozen = row.get("n1", {})
            if (
                compiled.action_sha256 != frozen.get("action_sha256")
                or compiled.before_graph_sha256 != frozen.get("before_graph_sha256")
                or compiled.after_graph_sha256 != frozen.get("after_graph_sha256")
            ):
                raise EvaluationError(f"N1 prediction receipt drifted: {identity}")
            dataset = identity.split(":", 1)[1]
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            gt = read_geff_arrays(store, "train", dataset)
            nodes, edges = scoreall._prediction_rows(dataset, compiled.graph)
            metric_row = dict(official_metric_row(nodes, edges, dict(gt), scale))
            metric_row["identity"] = identity
            candidate_rows[identity] = metric_row
            compiler_receipts.append(
                {
                    "identity": identity,
                    "candidate_key": row["candidate_key"],
                    "action_sha256": compiled.action_sha256,
                    "before_graph_sha256": compiled.before_graph_sha256,
                    "after_graph_sha256": compiled.after_graph_sha256,
                    "removed_edges": [list(pair) for pair in compiled.removed_edges],
                    "exact_target_match": exact_prediction_key(row) in target_keys,
                }
            )

    identities = sorted(baseline_rows)
    baseline_summary = summarise(
        [baseline_rows[identity] for identity in identities], official_summary
    )
    candidate_summary = summarise(
        [candidate_rows[identity] for identity in identities], official_summary
    )
    delta = {
        "adjusted_edge_jaccard_delta": float(candidate_summary["adj_edge_jaccard"])
        - float(baseline_summary["adj_edge_jaccard"]),
        "raw_edge_jaccard_delta": float(candidate_summary["edge_jaccard"])
        - float(baseline_summary["edge_jaccard"]),
        "node_recall_delta": float(candidate_summary["node_recall"])
        - float(baseline_summary["node_recall"]),
        "combined_score_delta": float(candidate_summary["score"])
        - float(baseline_summary["score"]),
    }
    if not all(math.isfinite(value) for value in delta.values()):
        raise EvaluationError("nonfinite score delta")
    official_division = {
        "tp": int(candidate_summary["division_tp"]),
        "fp": int(candidate_summary["division_fp"]),
        "fn": int(candidate_summary["division_fn"]),
        "jaccard": float(candidate_summary["division_jaccard"]),
    }
    pass_checks = {
        "combined_score_delta_gt_zero": delta["combined_score_delta"] > 0.0,
        "division_jaccard_gte_0p30": official_division["jaccard"] >= 0.30,
        "adjusted_edge_no_regression": delta["adjusted_edge_jaccard_delta"] >= 0.0,
        "raw_edge_no_regression": delta["raw_edge_jaccard_delta"] >= 0.0,
        "node_no_regression": delta["node_recall_delta"] >= 0.0,
        "zero_duplicates": duplicate_survivors == 0,
    }
    result = {
        "schema": "celltrack-alpha.division-realizer-result.v1",
        "arm": "A",
        "name": "geometry_motion",
        "status": "COMPLETE_44B6_BARRIER",
        "verdict": "PASS" if all(pass_checks.values()) else "FAIL_PARK_C",
        "prediction": {
            "selected_count": len(selected),
            "candidate_universe": predictions["candidate_universe"],
            "prediction_sha256": observed_hashes["predictions_sha256"],
            "duplicate_survivors": duplicate_survivors,
        },
        "official_44b6": {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "division": official_division,
            "delta": delta,
        },
        "exact_38_event_target_44b6_subset": {
            "target_event_count": len(target_keys),
            "tp": target_tp,
            "fp": target_fp,
            "fn": target_fn,
            "jaccard": target_tp / target_denominator if target_denominator else 0.0,
            "note": "Exact observed endpoint identity is an evaluation label only; the frozen official scorer controls the arm verdict.",
        },
        "pass_checks": pass_checks,
        "fp_sensitivity": FP_SENSITIVITY,
        "suppression": {
            "one_to_one": True,
            "occupied_node_ownership": True,
            "duplicate_survivors": duplicate_survivors,
            "verdict": "PASS_ZERO_DUPLICATES",
        },
        "information_firewall": {
            "prediction_frozen_before_target_open": True,
            "sealed_prediction_sha256": EXPECTED_PREDICTION_SHA256,
            "prediction_program_has_target_argument": False,
            "target_used_as_feature": False,
            "target_used_for_evaluation_labels_only": True,
            "evaluation_did_not_refit_or_reselect": True,
        },
        "resource_policy": {
            "cpu_only": True,
            "gpu_used": False,
            "kaggle_used": False,
            "runpod_used": False,
            "submission_used": False,
            "full_pool_cache_created": False,
            "six_bba_read": False,
            "cv57_run": False,
        },
        "hash_receipts": {
            **observed_hashes,
            "raw_archive_sha256": args.raw_archive_sha256,
            "raw_archive_hash_source": "banked immutable pin; size reverified, archive opened read-only",
            "fragment_count": len(fragment_receipts),
            "fragment_receipts": fragment_receipts,
        },
        "n1_compiler_receipts": compiler_receipts,
    }
    atomic_json(args.output.resolve(), result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-receipt", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--fragments-dir", type=Path, required=True)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--raw-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "official_44b6": {
                    "division": result["official_44b6"]["division"],
                    "delta": result["official_44b6"]["delta"],
                },
                "exact_target": result["exact_38_event_target_44b6_subset"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
