#!/usr/bin/env python3
"""Calibrate low-compute output offsets from sparse GEFF annotations."""

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
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.proposals import detect_candidates_one_timepoint  # noqa: E402
from celltrack_alpha.routing import default_route_configs  # noqa: E402
from celltrack_baseline.rule_based import list_zip_datasets, parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--datasets", default=None, help="Comma-separated train datasets. Default: prefix-balanced sample.")
    parser.add_argument("--max-per-prefix", type=int, default=2)
    parser.add_argument("--route", default="clean_point", choices=tuple(sorted(default_route_configs())))
    parser.add_argument("--max-timepoints", type=int, default=12)
    parser.add_argument("--match-radius-um", type=float, default=14.0)
    parser.add_argument("--output", type=Path, default=Path("reports/alpha_offset_calibration.json"))
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def default_datasets(zip_path: Path, max_per_prefix: int) -> list[str]:
    all_datasets = list_zip_datasets(zip_path, "train")
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for dataset in all_datasets:
        by_prefix[dataset.split("_")[0]].append(dataset)
    selected: list[str] = []
    for prefix in sorted(by_prefix):
        # Prefer datasets with more annotations by using names later in the ZIP list
        # only if users did not provide an explicit set. This stays deterministic.
        selected.extend(by_prefix[prefix][:max_per_prefix])
    return selected


def sample_times(times: list[int], max_timepoints: int) -> list[int]:
    times = sorted(set(int(t) for t in times))
    if len(times) <= max_timepoints:
        return times
    idx = np.linspace(0, len(times) - 1, max_timepoints).round().astype(int)
    return [times[int(i)] for i in idx]


def nearest_matches(
    candidates: list[Any],
    gt_xyz: np.ndarray,
    scale_zyx: tuple[float, float, float],
    match_radius_um: float,
) -> list[dict[str, Any]]:
    if len(gt_xyz) == 0 or not candidates:
        return []
    cand_xyz = np.asarray([node.coord_zyx for node in candidates], dtype=np.float32)
    scale = np.asarray(scale_zyx, dtype=np.float32)
    dist = np.sqrt(np.sum(((gt_xyz[:, None, :] - cand_xyz[None, :, :]) * scale) ** 2, axis=2))
    nearest = np.argmin(dist, axis=1)
    out: list[dict[str, Any]] = []
    for gt_i, cand_i in enumerate(nearest):
        distance = float(dist[gt_i, cand_i])
        if distance <= match_radius_um:
            out.append(
                {
                    "distance_um": distance,
                    "delta_zyx": (gt_xyz[gt_i] - cand_xyz[cand_i]).astype(float).tolist(),
                    "gt_zyx": gt_xyz[gt_i].astype(float).tolist(),
                    "candidate_zyx": cand_xyz[cand_i].astype(float).tolist(),
                }
            )
    return out


def summarize_offsets(matches: list[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {
            "offset_zyx": [0, 0, 0],
            "n_matches": 0,
            "median_delta_zyx": [0.0, 0.0, 0.0],
            "mean_distance_um": float("nan"),
        }
    deltas = np.asarray([row["delta_zyx"] for row in matches], dtype=np.float32)
    median = np.median(deltas, axis=0)
    offset = np.rint(median).astype(int)
    return {
        "offset_zyx": [int(v) for v in offset.tolist()],
        "n_matches": len(matches),
        "median_delta_zyx": [float(v) for v in median.tolist()],
        "mean_distance_um": float(np.mean([row["distance_um"] for row in matches])),
    }


def recall_after_offset(
    candidates_by_t: dict[int, list[Any]],
    gt_by_t: dict[int, np.ndarray],
    scale_zyx: tuple[float, float, float],
    offset_zyx: tuple[int, int, int],
) -> float:
    scale = np.asarray(scale_zyx, dtype=np.float32)
    offset = np.asarray(offset_zyx, dtype=np.float32)
    hits = 0
    total = 0
    for t, gt_xyz in gt_by_t.items():
        total += len(gt_xyz)
        candidates = candidates_by_t.get(t, [])
        if not candidates or len(gt_xyz) == 0:
            continue
        cand_xyz = np.asarray([node.coord_zyx for node in candidates], dtype=np.float32) + offset
        dist = np.sqrt(np.sum(((gt_xyz[:, None, :] - cand_xyz[None, :, :]) * scale) ** 2, axis=2))
        hits += int(np.sum(np.min(dist, axis=1) <= 7.0))
    return hits / total if total else float("nan")


def main() -> int:
    args = parse_args()
    configs = default_route_configs()
    config = configs[args.route]
    datasets = parse_csv_strings(args.datasets) if args.datasets else default_datasets(args.zip, args.max_per_prefix)

    dataset_matches: dict[str, list[dict[str, Any]]] = {}
    dataset_rows: dict[str, dict[str, Any]] = {}
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            started = time.time()
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            scale = parse_scale(attrs)
            chosen_times = sample_times(list(gt["nodes"]["t"]), args.max_timepoints)
            gt_by_t: dict[int, np.ndarray] = {}
            candidates_by_t: dict[int, list[Any]] = {}
            matches: list[dict[str, Any]] = []
            next_node_id = 1
            for t in chosen_times:
                mask = gt["nodes"]["t"] == int(t)
                gt_xyz = np.asarray(
                    list(zip(gt["nodes"]["z"][mask], gt["nodes"]["y"][mask], gt["nodes"]["x"][mask], strict=True)),
                    dtype=np.float32,
                )
                gt_by_t[int(t)] = gt_xyz
                candidates, next_node_id = detect_candidates_one_timepoint(
                    volume=np.asarray(array[int(t)]),
                    dataset=dataset,
                    t=int(t),
                    attrs=attrs,
                    scale_zyx=scale,
                    config=config,
                    next_node_id=next_node_id,
                )
                candidates_by_t[int(t)] = candidates
                for row in nearest_matches(candidates, gt_xyz, scale, args.match_radius_um):
                    row["dataset"] = dataset
                    row["t"] = int(t)
                    matches.append(row)
            summary = summarize_offsets(matches)
            before = recall_after_offset(candidates_by_t, gt_by_t, scale, (0, 0, 0))
            after = recall_after_offset(candidates_by_t, gt_by_t, scale, tuple(summary["offset_zyx"]))
            summary.update(
                {
                    "dataset": dataset,
                    "prefix": dataset.split("_")[0],
                    "route": args.route,
                    "sampled_timepoints": chosen_times,
                    "gt_nodes": int(sum(len(v) for v in gt_by_t.values())),
                    "recall_7um_before": before,
                    "recall_7um_after": after,
                    "runtime_sec": time.time() - started,
                }
            )
            dataset_matches[dataset] = matches
            dataset_rows[dataset] = summary
            print(json.dumps(summary, indent=2), flush=True)
    finally:
        store.close()

    prefix_matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dataset, matches in dataset_matches.items():
        prefix_matches[dataset.split("_")[0]].extend(matches)
    prefix_offsets = {
        prefix: summarize_offsets(matches)
        for prefix, matches in sorted(prefix_matches.items())
    }
    for prefix, row in prefix_offsets.items():
        row["prefix"] = prefix

    payload = {
        "route": args.route,
        "match_radius_um": args.match_radius_um,
        "datasets": datasets,
        "dataset_offsets": dataset_rows,
        "prefix_offsets": prefix_offsets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("prefix_offsets")
    print(json.dumps(prefix_offsets, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
