#!/usr/bin/env python3
"""Train fold-safe 3D template proposal kernels from sparse GT point patches."""

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

from celltrack_alpha.localization import sha256_file  # noqa: E402
from celltrack_alpha.routing import default_route_configs  # noqa: E402
from celltrack_alpha.scoring import load_scoring_calibration  # noqa: E402
from celltrack_baseline.rule_based import BaselineConfig, normalize_volume, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_SPLIT = Path("reports/general_model_split_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", default="0")
    parser.add_argument("--datasets", default=None, help="Comma-separated calibration-train datasets. Default: prefix-balanced sample.")
    parser.add_argument("--max-per-prefix", type=int, default=4)
    parser.add_argument("--route", default="template_point_compact")
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--clips-per-dataset", type=int, default=1)
    parser.add_argument("--radius-zyx", default="3,16,16")
    parser.add_argument("--score-quantiles", default="0.995,0.9985")
    parser.add_argument("--max-points", type=int, default=180)
    parser.add_argument("--min-distance-um", type=float, default=1.8)
    parser.add_argument("--min-patches", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/general_model_template_proposal_fold0.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/general_model_template_proposal_fold0_report.json"))
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_triplet(value: str) -> tuple[int, int, int]:
    parts = parse_csv_strings(value)
    if len(parts) != 3:
        raise ValueError("triplet must contain exactly three comma-separated integers")
    out = tuple(int(part) for part in parts)
    if any(v < 0 for v in out):
        raise ValueError("triplet values must be non-negative")
    return out


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


def extract_patch(volume: np.ndarray, center_zyx: tuple[int, int, int], radius_zyx: tuple[int, int, int]) -> np.ndarray:
    rz, ry, rx = radius_zyx
    padded = np.pad(volume, ((rz, rz), (ry, ry), (rx, rx)), mode="edge")
    z, y, x = (int(center_zyx[0]) + rz, int(center_zyx[1]) + ry, int(center_zyx[2]) + rx)
    patch = padded[z - rz : z + rz + 1, y - ry : y + ry + 1, x - rx : x + rx + 1]
    return patch.astype(np.float32, copy=False)


def standardize_patch(patch: np.ndarray) -> np.ndarray | None:
    patch = np.asarray(patch, dtype=np.float32)
    if patch.size == 0 or not np.all(np.isfinite(patch)):
        return None
    centered = patch - float(np.mean(patch))
    scale = float(np.std(centered))
    if scale <= 1e-6:
        return None
    return centered / scale


def build_kernel(patches: list[np.ndarray]) -> list[Any]:
    stack = np.stack(patches).astype(np.float32, copy=False)
    kernel = np.mean(stack, axis=0)
    kernel = kernel - float(np.mean(kernel))
    norm = float(np.sqrt(np.sum(kernel * kernel)))
    if norm <= 1e-6:
        raise ValueError("template kernel is degenerate")
    kernel = kernel / norm
    return np.round(kernel.astype(np.float32), 6).tolist()


def main() -> int:
    args = parse_args()
    if args.clip_length <= 0 or args.clips_per_dataset <= 0:
        raise ValueError("clip length and clips-per-dataset must be positive")
    if args.route not in default_route_configs():
        raise ValueError(f"unknown route: {args.route}")
    radius = parse_triplet(args.radius_zyx)
    score_quantiles = [float(value) for value in parse_csv_strings(args.score_quantiles)]
    if not score_quantiles or any(value < 0.0 or value > 1.0 for value in score_quantiles):
        raise ValueError("score quantiles must be in [0, 1]")
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

    patches_by_prefix: dict[str, list[np.ndarray]] = defaultdict(list)
    dataset_reports: list[dict[str, Any]] = []
    started = time.time()
    baseline = BaselineConfig(polarity="both")
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            gt_times = sorted(set(int(t) for t in gt["nodes"]["t"]))
            starts = clip_starts(gt_times, int(array.shape[0]), args.clip_length, args.clips_per_dataset)
            prefix = dataset.split("_", 1)[0]
            dataset_patch_count = 0
            for start in starts:
                end = min(int(array.shape[0]), int(start) + int(args.clip_length))
                for t in range(start, end):
                    mask = gt["nodes"]["t"] == int(t)
                    if not np.any(mask):
                        continue
                    norm = normalize_volume(np.asarray(array[t]), attrs, baseline)
                    for z, y, x in zip(gt["nodes"]["z"][mask], gt["nodes"]["y"][mask], gt["nodes"]["x"][mask], strict=True):
                        center = (int(round(float(z))), int(round(float(y))), int(round(float(x))))
                        patch = standardize_patch(extract_patch(norm, center, radius))
                        if patch is None:
                            continue
                        patches_by_prefix[prefix].append(patch)
                        dataset_patch_count += 1
            row = {"dataset": dataset, "clip_starts": starts, "patches": dataset_patch_count}
            dataset_reports.append(row)
            print(json.dumps(row, indent=2), flush=True)
    finally:
        store.close()

    by_route_prefix: dict[str, Any] = {}
    model_reports: dict[str, Any] = {}
    for prefix in sorted(patches_by_prefix):
        patches = patches_by_prefix[prefix]
        key = f"{args.route}:{prefix}"
        model_reports[key] = {"patches": len(patches), "radius_zyx": list(radius)}
        if len(patches) < int(args.min_patches):
            model_reports[key]["status"] = "skipped_too_few_patches"
            continue
        template = {
            "name": f"{prefix}_norm_template",
            "source": f"template_score:{prefix}_norm",
            "kernel": build_kernel(patches),
            "score_quantiles": score_quantiles,
            "max_points": int(args.max_points),
            "min_distance_um": float(args.min_distance_um),
        }
        by_route_prefix[key] = {"templates": [template]}
        model_reports[key]["status"] = "emitted"

    payload = {
        "kind": "candidate_graph_scoring",
        "proposal_policy": {"by_route_prefix": by_route_prefix},
        "provenance": {
            "status": "fold_safe",
            "fold_id": str(args.fold_id),
            "split_manifest": str(args.split_manifest),
            "split_manifest_sha256": split_hash,
            "source_datasets": datasets,
            "calibration_train": calibration_train,
            "validation_datasets": validation,
            "excluded_duplicate_public_ids": sorted(excluded),
            "calibration_script": "scripts/train_template_proposal.py",
            "radius_zyx": list(radius),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report = {
        "kind": "template_proposal_training_report",
        "created_at_unix": time.time(),
        "output": str(args.output),
        "route": args.route,
        "datasets": datasets,
        "dataset_reports": dataset_reports,
        "model_reports": model_reports,
        "runtime_sec": time.time() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    load_scoring_calibration(args.output, split_manifest_path=args.split_manifest, expected_fold_id=str(args.fold_id))
    print(json.dumps({"output": str(args.output), "report": str(args.report), "policies": sorted(by_route_prefix)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
