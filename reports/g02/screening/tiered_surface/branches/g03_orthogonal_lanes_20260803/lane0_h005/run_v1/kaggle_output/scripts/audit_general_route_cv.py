#!/usr/bin/env python3
"""Audit the current general alpha route on locked validation-fold clips."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha import default_route_configs, predict_dataset_window  # noqa: E402
from celltrack_alpha.evaluation import optional_official_metric_status, submission_graph_diagnostics  # noqa: E402
from celltrack_alpha.localization import load_offset_calibration, offset_for_dataset, sha256_file  # noqa: E402
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_alpha.scoring import load_scoring_calibration  # noqa: E402
from celltrack_baseline.rule_based import (  # noqa: E402
    parse_scale,
    read_geff_arrays,
    submission_rows,
    validate_submission_rows,
    write_submission,
)


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_SPLIT = Path("reports/general_model_split_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", default="0")
    parser.add_argument("--prefixes", default="44b6,6bba")
    parser.add_argument("--max-per-prefix", type=int, default=4)
    parser.add_argument("--datasets", default=None, help="Comma-separated validation datasets. Overrides prefix sampling.")
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--route", default="auto", choices=("auto", *sorted(default_route_configs().keys())))
    parser.add_argument("--allow-division-route", action="store_true")
    parser.add_argument("--offsets-json", type=Path, default=None)
    parser.add_argument("--scoring-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/general_model_cv_audit_fold0.json"))
    parser.add_argument("--submission-output", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_fold(path: Path, fold_id: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fold = payload.get("folds", {}).get(str(fold_id))
    if not isinstance(fold, dict):
        raise ValueError(f"fold {fold_id!r} is not present in {path}")
    return payload, fold, sha256_file(path)


def selected_validation_datasets(fold: dict[str, Any], prefixes: list[str], max_per_prefix: int) -> list[str]:
    validation = [str(value) for value in fold.get("validation", [])]
    out: list[str] = []
    for prefix in prefixes:
        prefix_matches = [dataset for dataset in validation if dataset.startswith(prefix + "_")]
        out.extend(prefix_matches[: max(0, int(max_per_prefix))])
    return out


def first_gt_clip_start(gt: dict[str, Any], total_t: int, clip_length: int) -> int:
    times = sorted(set(int(value) for value in gt["nodes"]["t"]))
    if not times:
        return 0
    max_start = max(0, int(total_t) - int(clip_length))
    return max(0, min(times[0], max_start))


def crop_geff_arrays(gt: dict[str, Any], start: int, end: int, total_timepoints: int) -> dict[str, Any]:
    nodes = gt["nodes"]
    mask = (nodes["t"] >= start) & (nodes["t"] < end)
    kept_ids = {int(value) for value in nodes["node_id"][mask]}
    edges = np.asarray(
        [edge for edge in gt["edges"] if int(edge[0]) in kept_ids and int(edge[1]) in kept_ids],
        dtype=np.int64,
    )
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    try:
        estimated = float(gt.get("estimated_number_of_nodes")) * max(0, end - start) / max(1, int(total_timepoints))
    except (TypeError, ValueError):
        estimated = float("nan")
    return {
        "attrs": gt.get("attrs", {}),
        "nodes": {key: np.asarray(value)[mask] for key, value in nodes.items()},
        "edges": edges,
        "estimated_number_of_nodes": estimated,
    }


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def summarize_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for value in sorted({str(row.get(key, "")) for row in rows}):
        subset = [row for row in rows if str(row.get(key, "")) == value]
        if not subset:
            continue
        out[value] = {
            "n": len(subset),
            "official_summary": official_summary(subset),
            "mean_node_recall": float(np.mean([finite(row.get("node_recall")) for row in subset])),
            "mean_nodes": float(np.mean([finite(row.get("num_pred_nodes")) for row in subset])),
            "zero_edge_jaccard": int(sum(finite(row.get("edge_jaccard")) == 0.0 for row in subset)),
            "node_recall_lt_0_5": int(sum(finite(row.get("node_recall")) < 0.5 for row in subset)),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.clip_length <= 0:
        raise ValueError("--clip-length must be positive")
    _, fold, split_sha = load_fold(args.split_manifest, str(args.fold_id))
    validation = {str(value) for value in fold.get("validation", [])}
    if args.datasets:
        datasets = parse_csv_strings(args.datasets)
    else:
        datasets = selected_validation_datasets(
            fold,
            parse_csv_strings(args.prefixes),
            args.max_per_prefix,
        )
    invalid = sorted(set(datasets) - validation)
    if invalid:
        raise ValueError(f"datasets are outside fold {args.fold_id} validation set: {invalid[:10]}")

    official_status = optional_official_metric_status()
    if not official_status["available"]:
        raise RuntimeError(f"official metric unavailable: {official_status}")

    offset_calibration = load_offset_calibration(
        args.offsets_json,
        split_manifest_path=args.split_manifest,
        expected_fold_id=str(args.fold_id) if args.offsets_json is not None else None,
    )
    scoring_calibration = load_scoring_calibration(
        args.scoring_json,
        split_manifest_path=args.split_manifest,
        expected_fold_id=str(args.fold_id) if args.scoring_json is not None else None,
    )

    started = time.time()
    rows: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []
    predictions: list[tuple[str, list[dict[str, Any]], list[dict[str, int]]]] = []
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            total_t = int(array.shape[0])
            gt = read_geff_arrays(store, "train", dataset)
            start = first_gt_clip_start(gt, total_t, args.clip_length)
            end = min(total_t, start + int(args.clip_length))
            gt_clip = crop_geff_arrays(gt, start, end, total_t)
            if len(gt_clip["nodes"]["node_id"]) == 0:
                dataset_reports.append(
                    {"dataset": dataset, "clip_start": start, "clip_end": end, "status": "skipped_empty_gt"}
                )
                continue
            run_started = time.time()
            result = predict_dataset_window(
                dataset=dataset,
                array=array,
                attrs=attrs,
                start=start,
                end=end,
                forced_route=args.route,
                allow_division_route=args.allow_division_route,
                output_offset_override=offset_for_dataset(dataset, offset_calibration),
                estimated_total_nodes=float(gt.get("estimated_number_of_nodes"))
                if gt.get("estimated_number_of_nodes") is not None
                else None,
                scoring_calibration=scoring_calibration,
                progress=args.progress,
            )
            metrics = official_metric_row(result.nodes, result.edges, gt_clip, parse_scale(attrs))
            graph_diag = submission_graph_diagnostics(submission_rows([(dataset, result.nodes, result.edges)]))
            row = {
                "dataset": dataset,
                "prefix": dataset.split("_", 1)[0],
                "fold_id": str(args.fold_id),
                "route": result.route.label,
                "route_confidence": result.route.confidence,
                "clip_start": start,
                "clip_end": end,
                "gt_nodes": int(len(gt_clip["nodes"]["node_id"])),
                "gt_edges": int(len(gt_clip["edges"])),
                "num_pred_nodes": int(len(result.nodes)),
                "num_pred_edges": int(len(result.edges)),
                "graph": graph_diag,
                "runtime_sec": time.time() - run_started,
                "metric_source": "official",
                "route_reason": result.route.reason,
                "profile": result.diagnostics.get("profile"),
                "coordinate_correction": result.diagnostics.get("coordinate_correction"),
                "node_pruning": result.diagnostics.get("node_pruning"),
                "refined_candidate_admission": result.diagnostics.get("refined_candidate_admission"),
                "edge_scoring": result.diagnostics.get("edge_scoring"),
                "refined_chain_conflict_rescue": result.diagnostics.get("refined_chain_conflict_rescue"),
                "short_edge_pruning": result.diagnostics.get("short_edge_pruning"),
                "localmax_refinement": result.diagnostics.get("localmax_refinement"),
                "free_edge_admission": result.diagnostics.get("free_edge_admission"),
                "duplicate_endpoint_rescue": result.diagnostics.get("duplicate_endpoint_rescue"),
                "motion_endpoint_admission": result.diagnostics.get("motion_endpoint_admission"),
                "localmax_clone_layers": result.diagnostics.get("localmax_clone_layers"),
                "offset_clone_layers": result.diagnostics.get("offset_clone_layers"),
                "endpoint_backward_clones": result.diagnostics.get("endpoint_backward_clones"),
                "endpoint_forward_clones": result.diagnostics.get("endpoint_forward_clones"),
                "component_pruning": result.diagnostics.get("component_pruning"),
                "final_node_pruning": result.diagnostics.get("final_node_pruning"),
                "output_offset_zyx": result.diagnostics.get("output_offset_zyx"),
                "node_emission": result.diagnostics.get("node_emission"),
            }
            row.update(metrics)
            rows.append(row)
            predictions.append((dataset, result.nodes, result.edges))
            dataset_reports.append(
                {
                    "dataset": dataset,
                    "clip_start": start,
                    "clip_end": end,
                    "route": result.route.label,
                    "score": finite(metrics.get("adj_edge_jaccard")),
                    "edge_jaccard": finite(metrics.get("edge_jaccard")),
                    "node_recall": finite(metrics.get("node_recall")),
                    "runtime_sec": row["runtime_sec"],
                }
            )
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "route": result.route.label,
                        "score": finite(metrics.get("adj_edge_jaccard")),
                        "edge_jaccard": finite(metrics.get("edge_jaccard")),
                        "node_recall": finite(metrics.get("node_recall")),
                        "nodes": len(result.nodes),
                        "edges": len(result.edges),
                    },
                    default=str,
                ),
                flush=True,
            )
    finally:
        store.close()

    submission = submission_rows(predictions)
    validation_report = validate_submission_rows(submission)
    graph_report = submission_graph_diagnostics(submission)
    if args.submission_output is not None:
        args.submission_output.parent.mkdir(parents=True, exist_ok=True)
        write_submission(submission, args.submission_output)

    payload = {
        "kind": "general_route_cv_audit",
        "created_at_unix": time.time(),
        "zip": str(args.zip),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": split_sha,
        "fold_id": str(args.fold_id),
        "datasets": datasets,
        "route_arg": args.route,
        "clip_length": int(args.clip_length),
        "offsets_json": str(args.offsets_json) if args.offsets_json is not None else None,
        "offset_calibration_fold_id": offset_calibration.fold_id if offset_calibration is not None else None,
        "scoring_json": str(args.scoring_json) if args.scoring_json is not None else None,
        "scoring_calibration_fold_id": scoring_calibration.fold_id if scoring_calibration is not None else None,
        "official_metric": official_status,
        "summary": official_summary(rows),
        "summary_by_prefix": summarize_by(rows, "prefix"),
        "summary_by_route": summarize_by(rows, "route"),
        "rows": rows,
        "dataset_reports": dataset_reports,
        "submission_validation": validation_report,
        "submission_graph": graph_report,
        "submission_output": str(args.submission_output) if args.submission_output is not None else None,
        "runtime_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": payload["summary"], "graph": graph_report}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
