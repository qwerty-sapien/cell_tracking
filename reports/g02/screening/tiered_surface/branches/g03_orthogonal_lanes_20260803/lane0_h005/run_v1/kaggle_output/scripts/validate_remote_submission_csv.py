#!/usr/bin/env python3
"""Validate a downloaded Kaggle submission CSV before scored upload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUBMISSION_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]

DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_CSV = Path("reports/kaggle_g01/kernel_output/submission.csv")
DEFAULT_OUTPUT = Path("reports/kaggle_g01/remote_submission_validation_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--zip",
        type=Path,
        default=DEFAULT_ZIP,
        help="Local competition ZIP used to derive expected test datasets and bounds.",
    )
    parser.add_argument(
        "--expected-datasets",
        default=None,
        help="Comma-separated dataset names. Overrides --expected-datasets-file.",
    )
    parser.add_argument(
        "--expected-datasets-file",
        type=Path,
        default=None,
        help="Text file with one expected dataset name per line.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int(value: str, field: str, row_number: int, errors: list[str]) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"row {row_number}: {field} is not an integer: {value!r}")
        return None


def expected_from_zip(zip_path: Path) -> tuple[list[str], dict[str, tuple[int, int, int, int]]]:
    datasets: set[str] = set()
    shapes: dict[str, tuple[int, int, int, int]] = {}
    if not zip_path.exists():
        return [], shapes
    prefix = "test/"
    suffix = ".zarr/0/zarr.json"
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith(prefix) and name.endswith(suffix):
                dataset = name[len(prefix) : -len(suffix)]
                datasets.add(dataset)
                try:
                    metadata = json.loads(zf.read(name).decode("utf-8"))
                    shape = tuple(int(v) for v in metadata.get("shape", []))
                    if len(shape) == 4:
                        shapes[dataset] = shape  # type: ignore[assignment]
                except Exception:
                    continue
    return sorted(datasets), shapes


def expected_from_args(args: argparse.Namespace) -> tuple[list[str], dict[str, tuple[int, int, int, int]], str]:
    if args.expected_datasets:
        datasets = sorted({v.strip() for v in args.expected_datasets.split(",") if v.strip()})
        return datasets, {}, "argument"
    if args.expected_datasets_file is not None:
        values = [
            line.strip()
            for line in args.expected_datasets_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return sorted(set(values)), {}, "file"
    datasets, shapes = expected_from_zip(args.zip)
    return datasets, shapes, "zip" if datasets else "none"


def read_csv_rows(path: Path, errors: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        errors.append(f"CSV does not exist: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != SUBMISSION_COLUMNS:
            errors.append(f"header mismatch: expected {SUBMISSION_COLUMNS}, got {reader.fieldnames}")
        return list(reader)


def validate_rows(
    rows: list[dict[str, str]],
    *,
    expected_datasets: list[str],
    shapes: dict[str, tuple[int, int, int, int]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    row_type_counts: Counter[str] = Counter()
    dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    nodes_by_dataset: dict[str, dict[int, dict[str, int]]] = defaultdict(dict)
    edges_by_dataset: dict[str, set[tuple[int, int]]] = defaultdict(set)
    outdeg: Counter[tuple[str, int]] = Counter()
    indeg: Counter[tuple[str, int]] = Counter()

    expected = set(expected_datasets)
    seen_row_ids: set[int] = set()
    expected_next_id = 0

    for index, row in enumerate(rows, start=2):
        row_id = parse_int(row.get("id", ""), "id", index, errors)
        if row_id is not None:
            if row_id in seen_row_ids:
                errors.append(f"row {index}: duplicate id {row_id}")
            seen_row_ids.add(row_id)
            if row_id != expected_next_id:
                errors.append(f"row {index}: expected id {expected_next_id}, got {row_id}")
            expected_next_id += 1

        dataset = row.get("dataset", "")
        if not dataset:
            errors.append(f"row {index}: dataset is empty")
        elif expected and dataset not in expected:
            errors.append(f"row {index}: unexpected dataset {dataset!r}")

        row_type = row.get("row_type", "")
        row_type_counts[row_type] += 1
        dataset_counts[dataset][row_type] += 1

        numeric = {
            field: parse_int(row.get(field, ""), field, index, errors)
            for field in ("node_id", "t", "z", "y", "x", "source_id", "target_id")
        }
        if any(value is None for value in numeric.values()):
            continue

        node_id = int(numeric["node_id"])
        t = int(numeric["t"])
        z = int(numeric["z"])
        y = int(numeric["y"])
        x = int(numeric["x"])
        source_id = int(numeric["source_id"])
        target_id = int(numeric["target_id"])

        if row_type == "node":
            if source_id != -1 or target_id != -1:
                errors.append(f"row {index}: node row must use source_id=target_id=-1")
            if node_id < 0:
                errors.append(f"row {index}: node_id must be non-negative for node row")
            if min(t, z, y, x) < 0:
                errors.append(f"row {index}: node coordinates and t must be non-negative")
            if node_id in nodes_by_dataset[dataset]:
                errors.append(f"row {index}: duplicate node_id {dataset}:{node_id}")
            nodes_by_dataset[dataset][node_id] = {"t": t, "z": z, "y": y, "x": x}
            shape = shapes.get(dataset)
            if shape is not None:
                max_t, max_z, max_y, max_x = shape
                if not (0 <= t < max_t and 0 <= z < max_z and 0 <= y < max_y and 0 <= x < max_x):
                    errors.append(
                        f"row {index}: node {dataset}:{node_id} outside shape "
                        f"{shape}: t/z/y/x={(t, z, y, x)}"
                    )
        elif row_type == "edge":
            if (node_id, t, z, y, x) != (-1, -1, -1, -1, -1):
                errors.append(f"row {index}: edge row must use node_id=t=z=y=x=-1")
            if source_id < 0 or target_id < 0:
                errors.append(f"row {index}: edge endpoints must be non-negative")
            if source_id == target_id:
                errors.append(f"row {index}: self-loop edge {dataset}:{source_id}->{target_id}")
            pair = (source_id, target_id)
            if pair in edges_by_dataset[dataset]:
                errors.append(f"row {index}: duplicate edge {dataset}:{source_id}->{target_id}")
            edges_by_dataset[dataset].add(pair)
            outdeg[(dataset, source_id)] += 1
            indeg[(dataset, target_id)] += 1
        else:
            errors.append(f"row {index}: unknown row_type {row_type!r}")

    for dataset, pairs in edges_by_dataset.items():
        nodes = nodes_by_dataset.get(dataset, {})
        for source_id, target_id in pairs:
            source = nodes.get(source_id)
            target = nodes.get(target_id)
            if source is None or target is None:
                errors.append(f"edge endpoint missing {dataset}:{source_id}->{target_id}")
                continue
            if target["t"] <= source["t"]:
                errors.append(
                    f"edge is not forward in time {dataset}:{source_id}(t={source['t']})->"
                    f"{target_id}(t={target['t']})"
                )
            elif target["t"] - source["t"] > 3:
                warnings.append(
                    f"edge spans more than 3 frames {dataset}:{source_id}(t={source['t']})->"
                    f"{target_id}(t={target['t']})"
                )

    missing = [dataset for dataset in expected_datasets if dataset not in nodes_by_dataset and dataset not in edges_by_dataset]
    if missing:
        errors.append(f"datasets missing from submission rows: {missing}")
    datasets_without_nodes = [dataset for dataset in expected_datasets if not nodes_by_dataset.get(dataset)]
    if datasets_without_nodes:
        errors.append(f"datasets without node rows: {datasets_without_nodes}")

    merges = [(dataset, node_id, count) for (dataset, node_id), count in indeg.items() if count > 1]
    out_gt2 = [(dataset, node_id, count) for (dataset, node_id), count in outdeg.items() if count > 2]
    if merges:
        errors.append(f"merge nodes found (first 20): {merges[:20]}")
    if out_gt2:
        errors.append(f"nodes with >2 outgoing edges found (first 20): {out_gt2[:20]}")

    return {
        "valid": not errors,
        "errors": errors[:100],
        "warnings": warnings[:100],
        "row_count": len(rows),
        "row_type_counts": dict(row_type_counts),
        "dataset_count": len(set(dataset_counts)),
        "expected_dataset_count": len(expected_datasets),
        "missing_dataset_count": len(missing),
        "datasets_without_nodes_count": len(datasets_without_nodes),
        "node_count": sum(len(nodes) for nodes in nodes_by_dataset.values()),
        "edge_count": sum(len(edges) for edges in edges_by_dataset.values()),
        "dividing_nodes": sum(1 for value in outdeg.values() if value == 2),
        "out_gt2": len(out_gt2),
        "merges": len(merges),
        "per_dataset": {
            dataset: {
                "nodes": len(nodes_by_dataset.get(dataset, {})),
                "edges": len(edges_by_dataset.get(dataset, set())),
                "rows": sum(dataset_counts.get(dataset, Counter()).values()),
            }
            for dataset in sorted(set(expected_datasets) | set(dataset_counts))
        },
    }


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    rows = read_csv_rows(args.csv, errors)
    expected_datasets, shapes, expected_source = expected_from_args(args)
    if not expected_datasets:
        errors.append("no expected test datasets were available from --zip, --expected-datasets, or --expected-datasets-file")
    validation = validate_rows(rows, expected_datasets=expected_datasets, shapes=shapes)
    if errors:
        validation["valid"] = False
        validation["errors"] = [*errors, *validation["errors"]][:100]
    report = {
        "schema_version": 1,
        "csv": str(args.csv),
        "csv_exists": args.csv.exists(),
        "csv_sha256": sha256_file(args.csv) if args.csv.exists() else None,
        "csv_bytes": args.csv.stat().st_size if args.csv.exists() else None,
        "expected_dataset_source": expected_source,
        "expected_zip": str(args.zip),
        "validation": validation,
        "ready_for_scored_submission": bool(validation["valid"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
