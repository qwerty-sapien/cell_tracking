#!/usr/bin/env python3
"""Train fold-safe source-family coordinate corrections for candidate nodes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.localization import sha256_file  # noqa: E402
from celltrack_alpha.proposals import generate_candidates  # noqa: E402
from celltrack_alpha.routing import default_route_configs  # noqa: E402
from celltrack_alpha.scoring import load_scoring_calibration  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_SPLIT = Path("reports/general_model_split_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", default="0")
    parser.add_argument("--datasets", default=None, help="Comma-separated calibration-train datasets. Default: prefix-balanced sample.")
    parser.add_argument("--max-per-prefix", type=int, default=4)
    parser.add_argument("--routes", default="balanced_compact")
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--clips-per-dataset", type=int, default=1)
    parser.add_argument("--match-radius-um", type=float, default=14.0)
    parser.add_argument("--min-source-matches", type=int, default=3)
    parser.add_argument("--max-abs-offset-zyx", default="3,16,16")
    parser.add_argument("--include-default-offset", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/general_model_coordinate_corrector_fold0.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/general_model_coordinate_corrector_fold0_report.json"))
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_triplet(value: str) -> list[int]:
    parts = parse_csv_strings(value)
    if len(parts) != 3:
        raise ValueError("triplet must contain exactly three comma-separated integers")
    return [int(part) for part in parts]


def load_fold(path: Path, fold_id: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fold = payload.get("folds", {}).get(str(fold_id))
    if not isinstance(fold, dict):
        raise ValueError(f"fold {fold_id!r} is not present in {path}")
    return payload, fold, sha256_file(path)


def prefix_balanced_sample(datasets: list[str], max_per_prefix: int) -> list[str]:
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for dataset in sorted(datasets):
        by_prefix[dataset.split("_", 1)[0]].append(dataset)
    selected: list[str] = []
    for prefix in sorted(by_prefix):
        selected.extend(by_prefix[prefix][: max(0, int(max_per_prefix))])
    return selected


def clip_starts(gt_times: list[int], total_t: int, clip_length: int, clips_per_dataset: int) -> list[int]:
    if not gt_times:
        return [0]
    max_start = max(0, int(total_t) - int(clip_length))
    candidates = sorted({max(0, min(int(t), max_start)) for t in gt_times})
    if len(candidates) <= clips_per_dataset:
        return candidates
    idx = np.linspace(0, len(candidates) - 1, clips_per_dataset).round().astype(int)
    return sorted({int(candidates[int(i)]) for i in idx})


def gt_coords_for_t(gt: dict[str, Any], t: int) -> np.ndarray:
    nodes = gt["nodes"]
    mask = nodes["t"] == int(t)
    return np.asarray(
        list(zip(nodes["z"][mask], nodes["y"][mask], nodes["x"][mask], strict=True)),
        dtype=np.float32,
    )


def source_families(sources: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for source in sources:
        family = str(source).split("@", 1)[0]
        if family not in out:
            out.append(family)
    return tuple(out)


def matched_deltas(
    nodes: list[Any],
    gt_xyz: np.ndarray,
    scale_zyx: tuple[float, float, float],
    match_radius_um: float,
) -> list[tuple[Any, np.ndarray, float]]:
    if not nodes or len(gt_xyz) == 0:
        return []
    cand_xyz = np.asarray([node.coord_zyx for node in nodes], dtype=np.float32)
    scale = np.asarray(scale_zyx, dtype=np.float32)
    dist = np.sqrt(np.sum(((gt_xyz[:, None, :] - cand_xyz[None, :, :]) * scale) ** 2, axis=2))
    cost = dist.copy()
    cost[cost > float(match_radius_um)] = 1e9
    rows, cols = linear_sum_assignment(cost)
    out: list[tuple[Any, np.ndarray, float]] = []
    for row, col in zip(rows, cols, strict=True):
        distance = float(dist[int(row), int(col)])
        if distance > float(match_radius_um):
            continue
        node = nodes[int(col)]
        out.append((node, gt_xyz[int(row)] - cand_xyz[int(col)], distance))
    return out


def main() -> int:
    args = parse_args()
    if args.clip_length <= 0 or args.clips_per_dataset <= 0:
        raise ValueError("clip length and clips-per-dataset must be positive")
    configs = default_route_configs()
    routes = parse_csv_strings(args.routes)
    unknown = [route for route in routes if route not in configs]
    if unknown:
        raise ValueError(f"unknown routes: {unknown}")
    split_manifest, fold, split_hash = load_fold(args.split_manifest, str(args.fold_id))
    calibration_train = [str(v) for v in fold.get("calibration_train", [])]
    validation = [str(v) for v in fold.get("validation", [])]
    excluded = set(str(v) for v in split_manifest.get("excluded_from_general_score", {}).get("duplicate_public_ids", []))
    datasets = parse_csv_strings(args.datasets) if args.datasets else prefix_balanced_sample(calibration_train, args.max_per_prefix)
    invalid = sorted(set(datasets) - set(calibration_train))
    if invalid:
        raise ValueError(f"datasets outside calibration_train: {invalid[:10]}")
    dup = sorted(set(datasets) & excluded)
    if dup:
        raise ValueError(f"datasets include duplicate public IDs excluded from G-01: {dup}")

    max_abs = parse_triplet(args.max_abs_offset_zyx)
    by_route_prefix: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    all_by_route_prefix: dict[str, list[list[float]]] = defaultdict(list)
    dataset_reports: list[dict[str, Any]] = []
    started = time.time()
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            scale = parse_scale(attrs)
            starts = clip_starts(sorted(set(int(t) for t in gt["nodes"]["t"])), int(array.shape[0]), args.clip_length, args.clips_per_dataset)
            for route in routes:
                config = configs[route]
                match_count = 0
                source_counts: dict[str, int] = defaultdict(int)
                for start in starts:
                    end = min(int(array.shape[0]), int(start) + int(args.clip_length))
                    nodes_by_t, _ = generate_candidates(dataset, array, attrs, config, start=start, end=end)
                    for rel_t, nodes in enumerate(nodes_by_t):
                        gt_xyz = gt_coords_for_t(gt, int(start) + rel_t)
                        for node, delta, _distance in matched_deltas(nodes, gt_xyz, scale, args.match_radius_um):
                            key = f"{route}:{dataset.split('_', 1)[0]}"
                            delta_list = [float(v) for v in delta.tolist()]
                            all_by_route_prefix[key].append(delta_list)
                            for family in source_families(tuple(str(s) for s in node.sources)):
                                by_route_prefix[key][family].append(delta_list)
                                source_counts[family] += 1
                            match_count += 1
                dataset_reports.append(
                    {
                        "dataset": dataset,
                        "route": route,
                        "clip_starts": starts,
                        "matches": match_count,
                        "source_counts": dict(sorted(source_counts.items())),
                    }
                )
                print(json.dumps(dataset_reports[-1], indent=2), flush=True)
    finally:
        store.close()

    coordinate_by_route_prefix: dict[str, Any] = {}
    model_reports: dict[str, Any] = {}
    for key in sorted(by_route_prefix):
        source_offsets: dict[str, list[int]] = {}
        source_reports: dict[str, Any] = {}
        for family, deltas in sorted(by_route_prefix[key].items()):
            arr = np.asarray(deltas, dtype=np.float32)
            median = np.median(arr, axis=0)
            rounded = np.rint(np.clip(median, -np.asarray(max_abs), np.asarray(max_abs))).astype(int)
            source_reports[family] = {
                "n": int(len(deltas)),
                "median_delta_zyx": [float(v) for v in median.tolist()],
                "rounded_offset_zyx": [int(v) for v in rounded.tolist()],
            }
            if len(deltas) >= int(args.min_source_matches):
                source_offsets[family] = [int(v) for v in rounded.tolist()]
        row: dict[str, Any] = {
            "source_offsets_zyx": source_offsets,
            "max_abs_offset_zyx": max_abs,
        }
        if args.include_default_offset and all_by_route_prefix[key]:
            all_arr = np.asarray(all_by_route_prefix[key], dtype=np.float32)
            default = np.rint(np.clip(np.median(all_arr, axis=0), -np.asarray(max_abs), np.asarray(max_abs))).astype(int)
            row["offset_zyx"] = [int(v) for v in default.tolist()]
        if source_offsets or "offset_zyx" in row:
            coordinate_by_route_prefix[key] = row
        model_reports[key] = {
            "n_all_matches": int(len(all_by_route_prefix[key])),
            "source_reports": source_reports,
            "emitted_policy": row if key in coordinate_by_route_prefix else None,
        }

    payload = {
        "kind": "candidate_graph_scoring",
        "coordinate_policy": {"by_route_prefix": coordinate_by_route_prefix},
        "provenance": {
            "status": "fold_safe",
            "fold_id": str(args.fold_id),
            "split_manifest": str(args.split_manifest),
            "split_manifest_sha256": split_hash,
            "source_datasets": datasets,
            "calibration_train": calibration_train,
            "validation_datasets": validation,
            "excluded_duplicate_public_ids": sorted(excluded),
            "calibration_script": "scripts/train_coordinate_corrector.py",
            "match_radius_um": float(args.match_radius_um),
            "min_source_matches": int(args.min_source_matches),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report = {
        "kind": "coordinate_corrector_training_report",
        "created_at_unix": time.time(),
        "output": str(args.output),
        "routes": routes,
        "datasets": datasets,
        "dataset_reports": dataset_reports,
        "model_reports": model_reports,
        "runtime_sec": time.time() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    load_scoring_calibration(args.output, split_manifest_path=args.split_manifest, expected_fold_id=str(args.fold_id))
    print(json.dumps({"output": str(args.output), "report": str(args.report), "policies": sorted(coordinate_by_route_prefix)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
