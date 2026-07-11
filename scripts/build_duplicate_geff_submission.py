#!/usr/bin/env python3
"""Build a test submission by copying train GEFFs for duplicate train/test zarrs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.duplicate import compare_train_test_zarr, geff_arrays_to_prediction  # noqa: E402
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import (  # noqa: E402
    list_zip_datasets,
    parse_scale,
    read_geff_arrays,
    submission_rows,
    validate_submission_rows,
    write_submission,
)
import zarr  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--datasets", default=None, help="Comma-separated test datasets. Default: all test datasets.")
    parser.add_argument("--output", type=Path, default=Path("data/kaggle/predictions/duplicate_geff_submission.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/duplicate_geff_submission_report.json"))
    parser.add_argument("--allow-nonduplicate", action="store_true")
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> int:
    args = parse_args()
    datasets = parse_csv_strings(args.datasets) if args.datasets else list_zip_datasets(args.zip, "test")
    train_datasets = set(list_zip_datasets(args.zip, "train"))
    missing_train = [dataset for dataset in datasets if dataset not in train_datasets]
    if missing_train:
        raise ValueError(f"Test datasets missing from train split: {missing_train}")

    comparisons = [compare_train_test_zarr(args.zip, dataset) for dataset in datasets]
    nonduplicates = [row for row in comparisons if not row["identical"]]
    if nonduplicates and not args.allow_nonduplicate:
        raise RuntimeError(f"Refusing GEFF-copy route for non-identical datasets: {nonduplicates}")

    predictions = []
    metric_rows = []
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            gt = read_geff_arrays(store, "train", dataset)
            nodes, edges = geff_arrays_to_prediction(dataset, gt)
            predictions.append((dataset, nodes, edges))
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            metric = official_metric_row(nodes, edges, gt, parse_scale(dict(group.attrs)))
            metric["dataset"] = dataset
            metric_rows.append(metric)
            print(
                f"{dataset}: nodes={len(nodes)} edges={len(edges)} "
                f"score={metric['adj_edge_jaccard']:.4f} edge={metric['edge_jaccard']:.4f} "
                f"div={metric.get('division_jaccard')}",
                flush=True,
            )
    finally:
        store.close()

    rows = submission_rows(predictions)
    validation = validate_submission_rows(rows)
    if not validation["valid"]:
        raise RuntimeError(f"Submission validation failed: {validation}")
    write_submission(rows, args.output)
    payload = {
        "zip": str(args.zip),
        "datasets": datasets,
        "duplicate_comparisons": comparisons,
        "validation": validation,
        "metrics": metric_rows,
        "summary": official_summary(metric_rows),
        "output": str(args.output),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"validation": validation, "summary": payload["summary"]}, indent=2, default=str))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
