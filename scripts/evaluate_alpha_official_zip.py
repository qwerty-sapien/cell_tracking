#!/usr/bin/env python3
"""Evaluate a submission-style CSV against train GEFFs in the Kaggle ZIP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import zarr
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.official_eval import official_metric_row, official_summary, score_gt_as_prediction  # noqa: E402
from celltrack_baseline.rule_based import list_zip_datasets, parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--datasets", default=None, help="Comma-separated train datasets. Default: datasets present in CSV.")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/alpha_official_eval.json"))
    parser.add_argument("--gt-self-check", action="store_true")
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_predictions(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    nodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            dataset = str(row["dataset"])
            if row["row_type"] == "node":
                nodes[dataset].append(row)
            elif row["row_type"] == "edge":
                edges[dataset].append(row)
    return nodes, edges


def main() -> int:
    args = parse_args()
    pred_nodes, pred_edges = read_predictions(args.prediction_csv)
    if args.datasets:
        datasets = parse_csv_strings(args.datasets)
    else:
        datasets = sorted(pred_nodes)
    if args.max_datasets is not None:
        datasets = datasets[: args.max_datasets]
    train_datasets = set(list_zip_datasets(args.zip, "train"))
    missing = [dataset for dataset in datasets if dataset not in train_datasets]
    if missing:
        raise ValueError(f"Datasets are not in train GEFFs: {missing[:10]}")

    rows: list[dict[str, Any]] = []
    gt_self_rows: list[dict[str, Any]] = []
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            gt = read_geff_arrays(store, "train", dataset)
            scale = parse_scale(attrs)
            if args.gt_self_check:
                gt_row = score_gt_as_prediction(gt, scale)
                gt_row["dataset"] = dataset
                gt_self_rows.append(gt_row)
                print(
                    f"{dataset}: gt-self edge={gt_row['edge_jaccard']:.4f} "
                    f"division={gt_row.get('division_jaccard', float('nan')):.4f}",
                    flush=True,
                )
            row = official_metric_row(pred_nodes.get(dataset, []), pred_edges.get(dataset, []), gt, scale)
            row["dataset"] = dataset
            rows.append(row)
            print(
                f"{dataset}: score row edge={row['edge_jaccard']:.4f} "
                f"adj={row['adj_edge_jaccard']:.4f} div={row.get('division_jaccard', float('nan')):.4f} "
                f"recall={row['node_recall']:.4f}",
                flush=True,
            )
    finally:
        store.close()

    summary = official_summary(rows)
    payload = {
        "prediction_csv": str(args.prediction_csv),
        "datasets": datasets,
        "rows": rows,
        "summary": summary,
        "gt_self_check": {
            "rows": gt_self_rows,
            "summary": official_summary(gt_self_rows) if gt_self_rows else {},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"summary": summary}, indent=2, default=str))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
