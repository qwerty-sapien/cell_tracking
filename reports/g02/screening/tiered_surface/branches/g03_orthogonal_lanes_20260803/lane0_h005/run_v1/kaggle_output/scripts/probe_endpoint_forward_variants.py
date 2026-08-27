#!/usr/bin/env python3
"""Probe endpoint-forward clone variants with the official local metric."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha import pipeline  # noqa: E402
from celltrack_alpha.localization import load_offset_calibration, offset_for_dataset, sha256_file  # noqa: E402
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_alpha.scoring import load_scoring_calibration  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_SPLIT = Path("reports/general_model_split_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", default="0")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--route", default="auto")
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--offsets-json", type=Path, default=None)
    parser.add_argument("--scoring-json", type=Path, default=None)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Variant as name=radius_um,sources,max_per_t,requires_history. "
            "Use + between multiple sources, for example smooth+dog_abs. "
            "Use name=current for the unmodified production route config."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def parse_variants(raw_variants: list[str]) -> list[tuple[str, dict[str, Any] | None]]:
    variants: list[tuple[str, dict[str, Any] | None]] = []
    for raw in raw_variants:
        if "=" not in raw:
            raise ValueError(f"variant must be name=radius,sources,max_per,requires_history, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"variant name is empty in {raw!r}")
        if name == "current" or value.strip().lower() in {"current", "base", "default"}:
            variants.append((name, None))
            continue
        parts = parse_csv(value)
        if len(parts) != 4:
            raise ValueError(f"variant {name!r} must have radius_um,sources,max_per_t,requires_history")
        sources = tuple(part.strip() for part in parts[1].split("+") if part.strip())
        if not sources:
            raise ValueError(f"variant {name!r} must include at least one source")
        variants.append(
            (
                name,
                {
                    "endpoint_forward_radius_um": float(parts[0]),
                    "endpoint_forward_sources": sources,
                    "endpoint_forward_max_per_t": int(parts[2]),
                    "endpoint_forward_requires_history": parse_bool(parts[3]),
                },
            )
        )
    if not variants:
        variants.append(("current", None))
    return variants


def load_fold(path: Path, fold_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fold = payload.get("folds", {}).get(str(fold_id))
    if not isinstance(fold, dict):
        raise ValueError(f"fold {fold_id!r} is not present in {path}")
    return fold


def load_split(path: Path, fold_id: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fold = payload.get("folds", {}).get(str(fold_id))
    if not isinstance(fold, dict):
        raise ValueError(f"fold {fold_id!r} is not present in {path}")
    return fold, sha256_file(path)


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


def main() -> int:
    args = parse_args()
    if args.clip_length <= 0:
        raise ValueError("--clip-length must be positive")
    datasets = parse_csv(args.datasets)
    variants = parse_variants(args.variant)
    fold, split_sha = load_split(args.split_manifest, str(args.fold_id))
    validation = {str(value) for value in fold.get("validation", [])}
    invalid = sorted(set(datasets) - validation)
    if invalid:
        raise ValueError(f"datasets are outside fold {args.fold_id} validation set: {invalid[:10]}")

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
    original_route_for_profile = pipeline.route_for_profile
    started = time.time()
    rows: list[dict[str, Any]] = []
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
            scale_zyx = parse_scale(attrs)
            estimated_total_nodes = (
                float(gt.get("estimated_number_of_nodes"))
                if gt.get("estimated_number_of_nodes") is not None
                else None
            )
            for variant_name, kwargs in variants:
                def patched_route_for_profile(profile, configs, forced_route=None, allow_division_route=False):
                    decision = original_route_for_profile(
                        profile,
                        configs,
                        forced_route,
                        allow_division_route=allow_division_route,
                    )
                    if kwargs is None:
                        return decision
                    return replace(
                        decision,
                        config=replace(
                            decision.config,
                            enable_endpoint_forward_clones=True,
                            **kwargs,
                        ),
                    )

                pipeline.route_for_profile = patched_route_for_profile
                try:
                    result = pipeline.predict_dataset_window(
                        dataset=dataset,
                        array=array,
                        attrs=attrs,
                        start=start,
                        end=end,
                        forced_route=None if args.route == "auto" else args.route,
                        allow_division_route=False,
                        output_offset_override=offset_for_dataset(dataset, offset_calibration),
                        estimated_total_nodes=estimated_total_nodes,
                        scoring_calibration=scoring_calibration,
                        progress=False,
                    )
                finally:
                    pipeline.route_for_profile = original_route_for_profile
                metrics = official_metric_row(result.nodes, result.edges, gt_clip, scale_zyx)
                row = {
                    "dataset": dataset,
                    "variant": variant_name,
                    "route": result.route.label,
                    "clip_start": start,
                    "clip_end": end,
                    "endpoint_forward_variant": kwargs,
                    "num_pred_nodes": len(result.nodes),
                    "num_pred_edges": len(result.edges),
                    "endpoint_forward_clones": result.diagnostics.get("endpoint_forward_clones"),
                }
                row.update(metrics)
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "variant": variant_name,
                            "score": row.get("adj_edge_jaccard"),
                            "edge_jaccard": row.get("edge_jaccard"),
                            "node_recall": row.get("node_recall"),
                            "edge_tp": row.get("edge_tp"),
                            "edge_fp": row.get("edge_fp"),
                            "edge_fn": row.get("edge_fn"),
                            "nodes": len(result.nodes),
                            "edges": len(result.edges),
                            "endpoint_forward_clones": result.diagnostics.get("endpoint_forward_clones"),
                        },
                        default=str,
                    ),
                    flush=True,
                )
    finally:
        pipeline.route_for_profile = original_route_for_profile
        store.close()

    best = (
        max(
            rows,
            key=lambda row: (
                finite(row.get("adj_edge_jaccard")),
                finite(row.get("edge_jaccard")),
                -finite(row.get("num_pred_nodes")),
            ),
        )
        if rows
        else None
    )
    payload = {
        "kind": "endpoint_forward_variant_probe",
        "created_at_unix": time.time(),
        "zip": str(args.zip),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": split_sha,
        "fold_id": str(args.fold_id),
        "datasets": datasets,
        "route": args.route,
        "clip_length": int(args.clip_length),
        "offsets_json": str(args.offsets_json) if args.offsets_json is not None else None,
        "scoring_json": str(args.scoring_json) if args.scoring_json is not None else None,
        "variants": [{"name": name, "kwargs": kwargs} for name, kwargs in variants],
        "rows": rows,
        "summary": official_summary(rows),
        "best": best,
        "runtime_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "best": best, "summary": payload["summary"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
