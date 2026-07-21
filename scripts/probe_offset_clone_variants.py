#!/usr/bin/env python3
"""Probe route offset-clone variants with the official local metric."""

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
    parser.add_argument("--scoring-json", type=Path, default=None)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Variant as name=dz,dy,dx|dz,dy,dx. Use name=current for the unmodified "
            "production route config."
        ),
    )
    parser.add_argument("--skip-clipped", action="store_true")
    parser.add_argument(
        "--filter-source-mode",
        choices=("keep", "any", "contrast_only", "raw_only", "raw_contrast", "contrast_any", "raw_any"),
        default="keep",
        help="Override offset_clone_filter_source_mode for non-current variants; default keeps route config.",
    )
    parser.add_argument(
        "--filter-min-component-timepoints",
        type=int,
        default=None,
        help="Override offset_clone_filter_min_component_timepoints for non-current variants.",
    )
    parser.add_argument(
        "--filter-component-max-x",
        default=None,
        help="Override offset_clone_filter_component_max_x for non-current variants; use none to disable.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_variants(raw_variants: list[str]) -> list[tuple[str, tuple[tuple[int, int, int], ...] | None]]:
    variants: list[tuple[str, tuple[tuple[int, int, int], ...] | None]] = []
    for raw in raw_variants:
        if "=" not in raw:
            raise ValueError(f"variant must be name=offsets, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"variant name is empty in {raw!r}")
        if name == "current" or value.strip().lower() in {"current", "base", "default"}:
            variants.append((name, None))
            continue
        offsets: list[tuple[int, int, int]] = []
        for part in value.split("|"):
            values = [int(piece.strip()) for piece in part.split(",") if piece.strip()]
            if len(values) != 3:
                raise ValueError(f"offset {part!r} in variant {name!r} is not dz,dy,dx")
            offsets.append((values[0], values[1], values[2]))
        if not offsets:
            raise ValueError(f"variant {name!r} has no offsets")
        variants.append((name, tuple(offsets)))
    if not variants:
        variants.append(("current", None))
    return variants


def load_fold(path: Path, fold_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fold = payload.get("folds", {}).get(str(fold_id))
    if not isinstance(fold, dict):
        raise ValueError(f"fold {fold_id!r} is not present in {path}")
    return fold


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


def parse_optional_int(value: str | None, *, name: str) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"none", "null", "off", "disable", "disabled"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer or 'none', got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative when set, got {parsed!r}")
    return parsed


def main() -> int:
    args = parse_args()
    if args.clip_length <= 0:
        raise ValueError("--clip-length must be positive")
    if args.filter_min_component_timepoints is not None and args.filter_min_component_timepoints < 0:
        raise ValueError("--filter-min-component-timepoints must be non-negative")
    filter_component_max_x = parse_optional_int(
        args.filter_component_max_x, name="--filter-component-max-x"
    )
    datasets = parse_csv(args.datasets)
    variants = parse_variants(args.variant)
    fold = load_fold(args.split_manifest, str(args.fold_id))
    validation = {str(value) for value in fold.get("validation", [])}
    invalid = sorted(set(datasets) - validation)
    if invalid:
        raise ValueError(f"datasets are outside fold {args.fold_id} validation set: {invalid[:10]}")

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
            for variant_name, offsets in variants:
                def patched_route_for_profile(profile, configs, forced_route=None, allow_division_route=False):
                    decision = original_route_for_profile(
                        profile,
                        configs,
                        forced_route,
                        allow_division_route=allow_division_route,
                    )
                    if offsets is None:
                        return decision
                    override_kwargs: dict[str, Any] = {}
                    if args.filter_source_mode != "keep":
                        override_kwargs["offset_clone_filter_source_mode"] = str(args.filter_source_mode)
                    if args.filter_min_component_timepoints is not None:
                        override_kwargs["offset_clone_filter_min_component_timepoints"] = int(
                            args.filter_min_component_timepoints
                        )
                    if args.filter_component_max_x is not None:
                        override_kwargs["offset_clone_filter_component_max_x"] = filter_component_max_x
                    return replace(
                        decision,
                        config=replace(
                            decision.config,
                            enable_offset_clone_layers=True,
                            offset_clone_offsets_zyx=offsets,
                            offset_clone_skip_clipped=bool(args.skip_clipped),
                            **override_kwargs,
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
                    "offsets_zyx": None if offsets is None else [list(offset) for offset in offsets],
                    "skip_clipped": bool(args.skip_clipped) if offsets is not None else None,
                    "filter_source_mode": None if offsets is None else (
                        args.filter_source_mode if args.filter_source_mode != "keep" else "route"
                    ),
                    "filter_min_component_timepoints": None if offsets is None else args.filter_min_component_timepoints,
                    "filter_component_max_x": None if offsets is None else (
                        filter_component_max_x if args.filter_component_max_x is not None else "route"
                    ),
                    "num_pred_nodes": len(result.nodes),
                    "num_pred_edges": len(result.edges),
                    "graph": {
                        key: value
                        for key, value in result.diagnostics.items()
                        if key in {"datasets", "nodes", "edges", "dividing_nodes", "out_gt2", "merges"}
                    },
                    "offset_clone_layers": result.diagnostics.get("offset_clone_layers"),
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
                        },
                        default=str,
                    ),
                    flush=True,
                )
    finally:
        pipeline.route_for_profile = original_route_for_profile
        store.close()

    best = max(
        rows,
        key=lambda row: (
            finite(row.get("adj_edge_jaccard")),
            finite(row.get("edge_jaccard")),
            -finite(row.get("num_pred_nodes")),
        ),
    ) if rows else None
    payload = {
        "kind": "offset_clone_variant_probe",
        "created_at_unix": time.time(),
        "zip": str(args.zip),
        "split_manifest": str(args.split_manifest),
        "fold_id": str(args.fold_id),
        "datasets": datasets,
        "route": args.route,
        "clip_length": int(args.clip_length),
        "scoring_json": str(args.scoring_json) if args.scoring_json is not None else None,
        "variants": [
            {"name": name, "offsets_zyx": None if offsets is None else [list(offset) for offset in offsets]}
            for name, offsets in variants
        ],
        "skip_clipped": bool(args.skip_clipped),
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
