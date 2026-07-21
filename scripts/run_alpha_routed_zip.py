#!/usr/bin/env python3
"""Run the scenario-routed low-compute alpha from the Kaggle ZIP."""

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
from celltrack_alpha.localization import load_offset_calibration, offset_for_dataset  # noqa: E402
from celltrack_alpha.scoring import load_scoring_calibration  # noqa: E402
from celltrack_alpha.evaluation import (  # noqa: E402
    approximate_division_metrics,
    optional_official_metric_status,
    submission_graph_diagnostics,
    summarize_metric_rows,
)
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import (  # noqa: E402
    approximate_sparse_metrics,
    list_zip_datasets,
    parse_scale,
    read_geff_arrays,
    submission_rows,
    validate_submission_rows,
    write_submission,
)


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--datasets", default=None, help="Comma-separated datasets. Default: all test or first two train.")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--clip-start", type=int, default=0)
    parser.add_argument("--clip-length", type=int, default=None)
    parser.add_argument("--route", default="auto", choices=("auto", *sorted(default_route_configs().keys())))
    parser.add_argument(
        "--allow-division-route",
        action="store_true",
        help="Allow auto-routing to select the experimental division_likely route.",
    )
    parser.add_argument(
        "--offsets-json",
        type=Path,
        default=None,
        help="Optional fold-safe calibration JSON with dataset_offsets/prefix_offsets to override route output offsets.",
    )
    parser.add_argument(
        "--scoring-json",
        type=Path,
        default=None,
        help="Optional fold-safe candidate scoring/pruning calibration JSON.",
    )
    parser.add_argument(
        "--cv-fold",
        default=None,
        help="Expected fold ID for --offsets-json provenance validation.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("reports/general_model_split_manifest.json"),
        help="Locked split manifest used to validate offset calibration provenance.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/alpha_routed_submission.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/alpha_routed_metrics.json"))
    parser.add_argument("--evaluate", action="store_true", help="Approximate sparse-label CV metrics for train split.")
    parser.add_argument(
        "--compare-routes",
        action="store_true",
        help="For train/evaluate runs, also score every forced route without changing the emitted submission.",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def default_datasets(zip_path: Path, split: str) -> list[str]:
    datasets = list_zip_datasets(zip_path, split)
    if split == "test":
        return datasets
    selected: list[str] = []
    for prefix in ("44b6", "6bba"):
        match = next((name for name in datasets if name.startswith(prefix)), None)
        if match is not None:
            selected.append(match)
    return selected or datasets[:2]


def crop_geff_arrays(gt: dict[str, Any], start: int, end: int, total_timepoints: int) -> dict[str, Any]:
    nodes = gt["nodes"]
    mask = (nodes["t"] >= start) & (nodes["t"] < end)
    kept_ids = set(int(v) for v in nodes["node_id"][mask])
    cropped_edges = np.asarray(
        [edge for edge in gt["edges"] if int(edge[0]) in kept_ids and int(edge[1]) in kept_ids],
        dtype=np.int64,
    )
    if cropped_edges.size == 0:
        cropped_edges = np.empty((0, 2), dtype=np.int64)
    estimated = gt.get("estimated_number_of_nodes")
    try:
        estimated = float(estimated) * max(0, end - start) / max(1, int(total_timepoints))
    except (TypeError, ValueError):
        estimated = float("nan")
    return {
        "attrs": gt.get("attrs", {}),
        "nodes": {key: np.asarray(value)[mask] for key, value in nodes.items()},
        "edges": cropped_edges,
        "estimated_number_of_nodes": estimated,
    }


def estimated_total_from_geff(store: ZipStore, split: str, dataset: str) -> float | None:
    if split != "train":
        return None
    try:
        gt = read_geff_arrays(store, split, dataset)
        value = gt.get("estimated_number_of_nodes")
        return float(value) if value is not None else None
    except Exception:
        return None


def assert_every_dataset_present(rows: list[dict[str, Any]], datasets: list[str]) -> list[str]:
    present = {str(row["dataset"]) for row in rows}
    return [dataset for dataset in datasets if dataset not in present]


def main() -> int:
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    if args.datasets:
        datasets = parse_csv_strings(args.datasets)
    else:
        datasets = default_datasets(args.zip, args.split)
    if args.max_datasets is not None:
        datasets = datasets[: args.max_datasets]
    if args.evaluate and args.split != "train":
        raise ValueError("--evaluate is only valid with --split train")

    print(f"alpha routed run_id={run_id} split={args.split} route={args.route} datasets={datasets}")
    official_status = optional_official_metric_status()
    print(json.dumps({"official_metric": official_status}, indent=2))
    offset_calibration = load_offset_calibration(
        args.offsets_json,
        split_manifest_path=args.split_manifest,
        expected_fold_id=args.cv_fold,
    )
    scoring_calibration = load_scoring_calibration(
        args.scoring_json,
        split_manifest_path=args.split_manifest,
        expected_fold_id=args.cv_fold,
    )

    predictions: list[tuple[str, list[dict[str, Any]], list[dict[str, int]]]] = []
    metric_rows: list[dict[str, Any]] = []
    route_comparison_rows: list[dict[str, Any]] = []
    dataset_payloads: list[dict[str, Any]] = []
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"{args.split}/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            array = zarr.open_array(store=store, path=f"{args.split}/{dataset}.zarr/0", mode="r")
            total_t = int(array.shape[0])
            start = max(0, min(int(args.clip_start), max(0, total_t - 1)))
            end = total_t if args.clip_length is None else min(total_t, start + int(args.clip_length))
            estimated_total = estimated_total_from_geff(store, args.split, dataset)

            result = predict_dataset_window(
                dataset=dataset,
                array=array,
                attrs=attrs,
                start=start,
                end=end,
                forced_route=args.route,
                allow_division_route=args.allow_division_route,
                output_offset_override=offset_for_dataset(dataset, offset_calibration),
                estimated_total_nodes=estimated_total,
                scoring_calibration=scoring_calibration,
                progress=args.progress,
            )
            predictions.append((dataset, result.nodes, result.edges))
            payload = {
                "dataset": dataset,
                "clip_start": start,
                "clip_end": end,
                "route": result.route.label,
                "route_confidence": result.route.confidence,
                "route_reason": result.route.reason,
                "diagnostics": result.diagnostics,
            }
            dataset_payloads.append(payload)
            print(
                f"{dataset}: route={result.route.label} nodes={len(result.nodes)} "
                f"edges={len(result.edges)} divisions={result.diagnostics['num_dividing_nodes']}",
                flush=True,
            )

            if args.evaluate:
                gt = read_geff_arrays(store, args.split, dataset)
                gt_clip = crop_geff_arrays(gt, start, end, total_t)
                if official_status["available"]:
                    metrics = official_metric_row(result.nodes, result.edges, gt_clip, parse_scale(attrs))
                    approx_metrics = approximate_sparse_metrics(result.nodes, result.edges, gt_clip, parse_scale(attrs))
                    metrics["approx_edge_jaccard"] = approx_metrics["edge_jaccard"]
                    metrics["approx_adj_edge_jaccard"] = approx_metrics["adj_edge_jaccard"]
                    metrics["metric_source"] = "official"
                else:
                    metrics = approximate_sparse_metrics(result.nodes, result.edges, gt_clip, parse_scale(attrs))
                    metrics.update(approximate_division_metrics(result.nodes, result.edges, gt_clip, parse_scale(attrs)))
                    metrics["metric_source"] = "approximate"
                metrics.update(
                    {
                        "dataset": dataset,
                        "clip_start": start,
                        "clip_end": end,
                        "route": result.route.label,
                        "route_confidence": result.route.confidence,
                        "num_dividing_nodes": result.diagnostics["num_dividing_nodes"],
                        "num_edge_candidates": result.diagnostics["num_edge_candidates"],
                    }
                )
                metric_rows.append(metrics)
                metric_label = str(metrics.get("metric_source", "metric"))
                print(
                    f"{dataset}: {metric_label} adj={metrics['adj_edge_jaccard']:.4f} "
                    f"edge={metrics['edge_jaccard']:.4f} recall={metrics['node_recall']:.4f}",
                    flush=True,
                )

                if args.compare_routes:
                    for route_label in sorted(default_route_configs()):
                        if route_label == result.route.label:
                            compare_result = result
                        else:
                            compare_result = predict_dataset_window(
                                dataset=dataset,
                                array=array,
                                attrs=attrs,
                                start=start,
                                end=end,
                                forced_route=route_label,
                                allow_division_route=True,
                                output_offset_override=offset_for_dataset(dataset, offset_calibration),
                                estimated_total_nodes=estimated_total,
                                scoring_calibration=scoring_calibration,
                                progress=False,
                            )
                        compare_metrics = approximate_sparse_metrics(
                                compare_result.nodes,
                                compare_result.edges,
                                gt_clip,
                                parse_scale(attrs),
                        )
                        if official_status["available"]:
                            compare_official = official_metric_row(
                                compare_result.nodes,
                                compare_result.edges,
                                gt_clip,
                                parse_scale(attrs),
                            )
                            compare_official["approx_edge_jaccard"] = compare_metrics["edge_jaccard"]
                            compare_official["approx_adj_edge_jaccard"] = compare_metrics["adj_edge_jaccard"]
                            compare_metrics = compare_official
                            compare_metrics["metric_source"] = "official"
                        else:
                            compare_metrics.update(
                                approximate_division_metrics(
                                    compare_result.nodes,
                                    compare_result.edges,
                                    gt_clip,
                                    parse_scale(attrs),
                                )
                            )
                            compare_metrics["metric_source"] = "approximate"
                        compare_metrics.update(
                            {
                                "dataset": dataset,
                                "clip_start": start,
                                "clip_end": end,
                                "route": route_label,
                                "auto_selected": route_label == result.route.label,
                                "num_dividing_nodes": compare_result.diagnostics["num_dividing_nodes"],
                                "num_edge_candidates": compare_result.diagnostics["num_edge_candidates"],
                            }
                        )
                        route_comparison_rows.append(compare_metrics)
                    best = max(
                        route_comparison_rows,
                        key=lambda row: (
                            row["dataset"] == dataset,
                            float(row.get("adj_edge_jaccard", float("nan")))
                            if math.isfinite(float(row.get("adj_edge_jaccard", float("nan"))))
                            else -1.0,
                        ),
                    )
                    print(
                        f"{dataset}: route comparison complete; best_seen={best['route']} "
                        f"adj={best['adj_edge_jaccard']:.4f}",
                        flush=True,
                    )
    finally:
        store.close()

    rows = submission_rows(predictions)
    validation = validate_submission_rows(rows)
    missing = assert_every_dataset_present(rows, datasets)
    if missing:
        validation["valid"] = False
        validation.setdefault("errors", []).append(f"datasets missing from submission rows: {missing}")
    graph_diag = submission_graph_diagnostics(rows)
    print(json.dumps({"validation": validation, "graph": graph_diag}, indent=2))
    if not validation["valid"]:
        raise RuntimeError("Submission validation failed")

    write_submission(rows, args.output)
    payload = {
        "run_id": run_id,
        "args": vars(args),
        "datasets": dataset_payloads,
        "metrics": metric_rows,
        "summary": official_summary(metric_rows) if metric_rows and official_status["available"] else summarize_metric_rows(metric_rows) if metric_rows else {},
        "route_comparison": route_comparison_rows,
        "route_comparison_summary": official_summary(route_comparison_rows) if route_comparison_rows and official_status["available"] else summarize_metric_rows(route_comparison_rows) if route_comparison_rows else {},
        "submission_graph": graph_diag,
        "validation": validation,
        "official_metric": optional_official_metric_status(),
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
