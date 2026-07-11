#!/usr/bin/env python3
"""Run the rule-based baseline against the local Kaggle ZIP without extracting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_baseline.rule_based import (  # noqa: E402
    BaselineConfig,
    approximate_sparse_metrics,
    config_to_json,
    list_zip_datasets,
    parse_scale,
    predict_dataset_array,
    read_geff_arrays,
    submission_rows,
    validate_submission_rows,
    with_timepoint_limit,
    write_submission,
)


def parse_sigmas(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip"))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names. Default: all in split.")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-timepoints", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/kaggle/predictions/rule_based_submission.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/rule_based_cv_metrics.json"))
    parser.add_argument("--evaluate", action="store_true", help="Approximate sparse-label CV metrics; only valid for train split.")
    parser.add_argument("--sigmas-um", default="0.8,1.2,1.8")
    parser.add_argument("--dog-ratio", type=float, default=1.7)
    parser.add_argument("--threshold-quantile", type=float, default=0.99955)
    parser.add_argument("--min-score", type=float, default=0.015)
    parser.add_argument("--min-distance-um", type=float, default=2.8)
    parser.add_argument("--max-points-per-t", type=int, default=180)
    parser.add_argument("--max-link-um", type=float, default=8.0)
    parser.add_argument("--polarity", choices=("bright", "dark", "both"), default="bright")
    parser.add_argument("--output-transform", choices=("identity", "xflip", "yflip", "xyflip", "yxswap"), default="identity")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BaselineConfig(
        sigmas_um=parse_sigmas(args.sigmas_um),
        dog_ratio=args.dog_ratio,
        threshold_quantile=args.threshold_quantile,
        min_score=args.min_score,
        min_distance_um=args.min_distance_um,
        max_points_per_t=args.max_points_per_t,
        max_link_um=args.max_link_um,
        polarity=args.polarity,
        output_transform=args.output_transform,
    )
    config = with_timepoint_limit(config, args.max_timepoints)
    if args.datasets:
        datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    else:
        datasets = list_zip_datasets(args.zip, args.split)
    if args.max_datasets is not None:
        datasets = datasets[: args.max_datasets]
    print(f"Running ZIP baseline on {len(datasets)} {args.split} dataset(s)")
    print(config_to_json(config))

    predictions = []
    metrics_rows = []
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"{args.split}/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            array = zarr.open_array(store=store, path=f"{args.split}/{dataset}.zarr/0", mode="r")
            nodes, edges = predict_dataset_array(dataset, array, attrs, config, progress=True)
            predictions.append((dataset, nodes, edges))
            print(f"{dataset}: nodes={len(nodes)} edges={len(edges)}")
            if args.evaluate:
                if args.split != "train":
                    raise ValueError("--evaluate is only valid for train split")
                gt = read_geff_arrays(store, args.split, dataset)
                metrics = approximate_sparse_metrics(nodes, edges, gt, parse_scale(attrs))
                metrics["dataset"] = dataset
                metrics_rows.append(metrics)
                print(f"{dataset}: approx adj_edge_jaccard={metrics['adj_edge_jaccard']:.4f} node_recall={metrics['node_recall']:.4f}")
    finally:
        store.close()

    rows = submission_rows(predictions)
    validation = validate_submission_rows(rows)
    print(json.dumps(validation, indent=2))
    if not validation["valid"]:
        raise RuntimeError("Submission validation failed")
    write_submission(rows, args.output)
    print(f"Wrote {args.output}")

    if metrics_rows:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        numeric_keys = [
            "edge_tp", "edge_fp", "edge_fn", "edge_jaccard", "adj_edge_jaccard",
            "node_recall", "num_pred_nodes", "estimated_number_of_nodes", "total_node_ratio",
        ]
        summary = {}
        for key in numeric_keys:
            vals = [row[key] for row in metrics_rows if row[key] == row[key]]
            summary[key] = float(np.mean(vals)) if vals else float("nan")
        payload = {"config": config.__dict__, "rows": metrics_rows, "mean": summary}
        args.metrics_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
