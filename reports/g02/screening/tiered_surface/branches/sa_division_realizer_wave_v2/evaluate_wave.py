#!/usr/bin/env python3
"""Evaluate four sealed division arms on the full 38-event / 57-graph scope."""

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


EXPECTED = {
    "target": "0926ab647915ad4e2e790252c040867e0bbc45a396155cecfe7f5813d6efe221",
    "baseline": "94d7b9823fbc9780ea7c0f07fd1d27079b5d58a2c4ef7d500ac6e2106ef6c075",
    "n1": "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
    "scorer": "7b92fed9fe66e9a2faf062528a78b04a65bc1be50f806dff44f51c780da5ac6c",
    "raw_archive": "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19",
}
RAW_ARCHIVE_BYTES = 87393127165
FAMILY_COUNTS = {"44b6": 21, "6bba": 36}
TARGET_COUNTS = {"44b6": 9, "6bba": 29}
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    with temporary.open("xb") as handle:
        handle.write(payload)
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


def event_variants(event: Mapping[str, Any]) -> list[frozenset[tuple[int, int]]]:
    variants = [
        frozenset(
            (int(source), int(child))
            for source, child in event["direct_repair"]["required_edges"]
        )
    ]
    late = event.get("late_fork_equivalence", {})
    if late.get("available"):
        records = [late["preferred"], *late.get("alternatives", [])]
        for record in records:
            variants.append(
                frozenset(
                    (int(source), int(child))
                    for source, child in record["late_fork"]["required_edges"]
                )
            )
    return variants


def prediction_edges(row: Mapping[str, Any]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (int(source), int(child)) for source, child in row["action"]["add_edges"]
    )


def exact_target_metrics(
    selected: list[dict[str, Any]], events: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    event_records = [
        {
            "event_key": event["event_key"],
            "identity": event["identity"],
            "family": event["family"],
            "variants": event_variants(event),
        }
        for event in events
    ]
    matches: list[dict[str, Any]] = []
    matched_events: set[str] = set()
    unmatched_predictions: list[dict[str, Any]] = []
    for row in selected:
        edges = prediction_edges(row)
        candidates = [
            event
            for event in event_records
            if event["identity"] == row["identity"] and edges in event["variants"]
        ]
        if len(candidates) > 1:
            raise EvaluationError("one prediction matched multiple target events")
        if not candidates:
            unmatched_predictions.append(row)
            continue
        event = candidates[0]
        if event["event_key"] in matched_events:
            raise EvaluationError("FAIL_CLOSED_DUPLICATE_FORK")
        matched_events.add(event["event_key"])
        matches.append(
            {
                "candidate_key": row["candidate_key"],
                "event_key": event["event_key"],
                "identity": row["identity"],
                "family": row["family"],
                "action_edges": [list(edge) for edge in sorted(edges)],
            }
        )
    result: dict[str, dict[str, Any]] = {}
    for family in ("44b6", "6bba"):
        family_targets = [
            event for event in event_records if event["family"] == family
        ]
        family_matches = [row for row in matches if row["family"] == family]
        family_fp = [
            row for row in unmatched_predictions if row["family"] == family
        ]
        tp = len(family_matches)
        fp = len(family_fp)
        fn = len(family_targets) - tp
        denominator = tp + fp + fn
        result[family] = {
            "target_event_count": len(family_targets),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "jaccard": tp / denominator if denominator else 0.0,
            "binding_gate": False,
        }
    tp = len(matches)
    fp = len(unmatched_predictions)
    fn = len(event_records) - tp
    denominator = tp + fp + fn
    binding = {
        "target_event_count": len(event_records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "jaccard": tp / denominator if denominator else 0.0,
        "binding_gate": True,
    }
    return binding, result, matches


def deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    values = {
        "adjusted_edge_jaccard_delta": float(candidate["adj_edge_jaccard"])
        - float(baseline["adj_edge_jaccard"]),
        "raw_edge_jaccard_delta": float(candidate["edge_jaccard"])
        - float(baseline["edge_jaccard"]),
        "node_recall_delta": float(candidate["node_recall"])
        - float(baseline["node_recall"]),
        "combined_score_delta": float(candidate["score"]) - float(baseline["score"]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise EvaluationError("nonfinite score delta")
    return values


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

    freeze = json.loads(args.implementation_freeze.read_text(encoding="utf-8"))
    if (
        freeze["programs"].get(Path(__file__).name)
        != sha256_file(Path(__file__).resolve())
    ):
        raise EvaluationError("evaluator does not match implementation freeze")
    pins = {
        "target": sha256_file(args.target),
        "baseline": sha256_file(args.baseline_scores),
        "n1": sha256_file(repo_root / "src/celltrack_alpha/action_compiler.py"),
        "scorer": sha256_file(repo_root / "scripts/scoreall.py"),
    }
    if pins != {key: EXPECTED[key] for key in pins}:
        raise EvaluationError(f"frozen input drift: {pins}")
    if (
        args.raw_archive.stat().st_size != RAW_ARCHIVE_BYTES
        or args.raw_archive_sha256 != EXPECTED["raw_archive"]
    ):
        raise EvaluationError("raw archive banked identity/size mismatch")
    events_payload = json.loads(args.target.read_text(encoding="utf-8"))
    events = list(events_payload["events"])
    if len(events) != 38:
        raise EvaluationError("target count drift")
    observed_target_counts = {
        family: sum(event["family"] == family for event in events)
        for family in TARGET_COUNTS
    }
    if observed_target_counts != TARGET_COUNTS:
        raise EvaluationError("target family inventory drift")
    baseline_payload = json.loads(args.baseline_scores.read_text(encoding="utf-8"))
    baseline_rows = {
        str(row["identity"]): dict(row) for row in baseline_payload["samples"]
    }
    paths = sorted(args.fragments_dir.glob("train__*.csv"))
    graphs: dict[str, Any] = {}
    fragment_receipts: list[dict[str, Any]] = []
    for path in paths:
        parsed = read_submission(path)
        if len(parsed) != 1:
            raise EvaluationError(f"multi-graph fragment: {path}")
        graph = next(iter(parsed.values()))
        identity = f"train:{graph.dataset}"
        if identity not in baseline_rows:
            raise EvaluationError(f"missing baseline row: {identity}")
        source = baseline_rows[identity]["source"]
        observed_sha = sha256_file(path)
        if (
            observed_sha != source["sha256"]
            or path.stat().st_size != int(source["bytes"])
        ):
            raise EvaluationError(f"fragment drift: {identity}")
        graphs[identity] = graph
        fragment_receipts.append(
            {
                "identity": identity,
                "bytes": path.stat().st_size,
                "sha256": observed_sha,
            }
        )
    observed_families = {
        family: sum(identity.startswith(f"train:{family}_") for identity in graphs)
        for family in FAMILY_COUNTS
    }
    if observed_families != FAMILY_COUNTS or len(graphs) != 57:
        raise EvaluationError("57-fragment inventory mismatch")
    baseline_all = summarise(list(baseline_rows.values()), official_summary)
    baseline_by_family = {
        family: summarise(
            [
                row
                for identity, row in baseline_rows.items()
                if identity.startswith(f"train:{family}_")
            ],
            official_summary,
        )
        for family in FAMILY_COUNTS
    }
    outputs: dict[str, Any] = {}
    with ZipStore(args.raw_archive, mode="r") as store:
        for arm, prediction_path in zip(
            ("A", "B", "C", "D"), args.predictions, strict=True
        ):
            result_path = args.output_dir / f"arm_{arm.lower()}_RESULT.json"
            if result_path.exists():
                raise EvaluationError(f"refusing to overwrite result: {result_path}")
            receipt_path = prediction_path.with_suffix(".receipt.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("status") != "FROZEN_BEFORE_TARGET_EVALUATION"
                or receipt.get("arm") != arm
                or receipt.get("prediction_sha256")
                != sha256_file(prediction_path)
            ):
                raise EvaluationError(f"invalid sealed prediction: {arm}")
            payload = json.loads(prediction_path.read_text(encoding="utf-8"))
            selected = list(payload["selection"]["predictions"])
            keys = [row["event_equivalence_key"] for row in selected]
            if (
                payload.get("status") != "SEALED_BEFORE_EVALUATION"
                or payload["selection"].get("duplicate_survivors") != 0
                or len(keys) != len(set(keys))
            ):
                raise EvaluationError(f"duplicate or unsealed prediction: {arm}")
            by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in selected:
                by_identity[row["identity"]].append(row)
            if any(len(rows) > 1 for rows in by_identity.values()):
                raise EvaluationError(f"per-video precision cap violated: {arm}")
            target_metrics, family_target_metrics, matches = exact_target_metrics(
                selected, events
            )
            candidate_rows = {
                identity: dict(row) for identity, row in baseline_rows.items()
            }
            compiler_receipts: list[dict[str, Any]] = []
            for identity, rows in sorted(by_identity.items()):
                if identity not in graphs:
                    raise EvaluationError(f"unknown prediction identity: {identity}")
                row = rows[0]
                action_payload = row["action"]
                action = GraphAction(
                    action_id=f"division_v2_{arm}.{row['candidate_key'][:20]}",
                    kind=ActionKind.DIVISION_FORK,
                    add_edges=tuple(
                        tuple(int(value) for value in pair)
                        for pair in action_payload["add_edges"]
                    ),
                    replace_outgoing_from=tuple(
                        int(value)
                        for value in action_payload["replace_outgoing_from"]
                    ),
                    replace_incoming_to=tuple(
                        int(value) for value in action_payload["replace_incoming_to"]
                    ),
                    expected_conflict_edges=tuple(
                        tuple(int(value) for value in pair)
                        for pair in action_payload["expected_conflict_edges"]
                    ),
                )
                compiled = compile_action(graphs[identity], action)
                dataset = identity.split(":", 1)[1]
                group = zarr.open_group(
                    store=store, path=f"train/{dataset}.zarr", mode="r"
                )
                scale = tuple(
                    float(value) for value in parse_scale(dict(group.attrs))
                )
                gt = read_geff_arrays(store, "train", dataset)
                nodes, edges = scoreall._prediction_rows(dataset, compiled.graph)
                metric_row = dict(
                    official_metric_row(nodes, edges, dict(gt), scale)
                )
                metric_row["identity"] = identity
                candidate_rows[identity] = metric_row
                compiler_receipts.append(
                    {
                        "identity": identity,
                        "candidate_key": row["candidate_key"],
                        "action_sha256": compiled.action_sha256,
                        "before_graph_sha256": compiled.before_graph_sha256,
                        "after_graph_sha256": compiled.after_graph_sha256,
                        "removed_edges": [
                            list(pair) for pair in compiled.removed_edges
                        ],
                    }
                )
            candidate_all = summarise(
                list(candidate_rows.values()), official_summary
            )
            candidate_by_family = {
                family: summarise(
                    [
                        row
                        for identity, row in candidate_rows.items()
                        if identity.startswith(f"train:{family}_")
                    ],
                    official_summary,
                )
                for family in FAMILY_COUNTS
            }
            delta_all = deltas(baseline_all, candidate_all)
            family_diagnostics = {
                family: {
                    "binding_gate": False,
                    "target_38_subset": family_target_metrics[family],
                    "official_baseline": baseline_by_family[family],
                    "official_candidate": candidate_by_family[family],
                    "official_delta": deltas(
                        baseline_by_family[family], candidate_by_family[family]
                    ),
                }
                for family in FAMILY_COUNTS
            }
            pass_checks = {
                "combined_score_delta_gt_zero": delta_all[
                    "combined_score_delta"
                ]
                > 0.0,
                "division_jaccard_gte_0p30": target_metrics["jaccard"] >= 0.30,
                "adjusted_edge_no_regression": delta_all[
                    "adjusted_edge_jaccard_delta"
                ]
                >= 0.0,
                "raw_edge_no_regression": delta_all["raw_edge_jaccard_delta"]
                >= 0.0,
                "node_no_regression": delta_all["node_recall_delta"] >= 0.0,
                "zero_duplicates": True,
            }
            result = {
                "schema": "celltrack-alpha.division-realizer-result.v2",
                "status": "COMPLETE_FULL38_BARRIER",
                "arm": arm,
                "verdict": "PASS" if all(pass_checks.values()) else "FAIL_PARK_C",
                "binding_full38": {
                    "division": target_metrics,
                    "official_baseline": baseline_all,
                    "official_candidate": candidate_all,
                    "official_delta": delta_all,
                },
                "per_family_generalization_diagnostics": family_diagnostics,
                "prediction": {
                    "selected_count": len(selected),
                    "sha256": sha256_file(prediction_path),
                    "receipt_sha256": sha256_file(receipt_path),
                    "duplicate_survivors": 0,
                },
                "matches": matches,
                "pass_checks": pass_checks,
                "fp_sensitivity": FP_SENSITIVITY,
                "suppression": {
                    "one_to_one": True,
                    "per_video_cap": 1,
                    "duplicate_survivors": 0,
                    "status": "PASS_ZERO_DUPLICATES",
                },
                "information_firewall": {
                    "prediction_frozen_before_target_open": True,
                    "target_used_for_evaluation_only": True,
                    "prediction_program_has_target_argument": False,
                    "evaluation_refit_or_reselection": False,
                },
                "resource_policy": {
                    "cpu_only": True,
                    "gpu_used": False,
                    "kaggle_used": False,
                    "runpod_used": False,
                    "submission_used": False,
                    "full_pool_cache_created": False,
                },
                "hash_receipts": {
                    **pins,
                    "raw_archive_sha256": args.raw_archive_sha256,
                    "raw_archive_bytes": args.raw_archive.stat().st_size,
                    "fragment_count": len(fragment_receipts),
                    "fragments": fragment_receipts,
                    "evaluator_sha256": sha256_file(Path(__file__).resolve()),
                    "implementation_freeze_sha256": sha256_file(
                        args.implementation_freeze
                    ),
                },
                "n1_compiler_receipts": compiler_receipts,
            }
            atomic_json(result_path, result)
            outputs[arm] = {
                "result_path": str(result_path),
                "result_sha256": sha256_file(result_path),
                "verdict": result["verdict"],
                "binding_division": target_metrics,
                "delta": delta_all,
            }
    wave = {
        "schema": "celltrack-alpha.division-realizer-wave-terminal.v2",
        "status": "HOLD_AT_FULL38_BARRIER",
        "arms": outputs,
        "pass_arms": [
            arm for arm, result in outputs.items() if result["verdict"] == "PASS"
        ],
        "downstream_unlock": False,
        "hard_stops": {
            "submission": True,
            "dead": True,
            "gpu_gte_5h": True,
            "runpod": True,
            "full_pool_cache": True,
        },
    }
    wave_path = args.output_dir / "WAVE_TERMINAL_RESULT.json"
    atomic_json(wave_path, wave)
    return wave


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--predictions", nargs=4, type=Path, required=True)
    parser.add_argument("--implementation-freeze", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--fragments-dir", type=Path, required=True)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--raw-archive-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
