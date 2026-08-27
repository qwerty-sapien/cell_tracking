#!/usr/bin/env python3
"""Probe post-clone node/edge filtering variants on routed predictions."""

from __future__ import annotations

import argparse
import itertools
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

from celltrack_alpha.evaluation import submission_graph_diagnostics  # noqa: E402
from celltrack_alpha.localization import sha256_file  # noqa: E402
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_alpha.pipeline import predict_dataset_window  # noqa: E402
from celltrack_alpha.scoring import load_scoring_calibration  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays, submission_rows  # noqa: E402


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
    parser.add_argument("--max-layer-combinations", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


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


def offset_layers_in_nodes(nodes: list[dict[str, Any]]) -> tuple[int, ...]:
    layers: set[int] = set()
    for node in nodes:
        for source in str(node.get("sources", "")).split("|"):
            if source.startswith("offset_clone_"):
                try:
                    layers.add(int(source.rsplit("_", 1)[-1]))
                except ValueError:
                    continue
    return tuple(sorted(layers))


def node_offset_layers(node: dict[str, Any]) -> set[int]:
    layers: set[int] = set()
    for source in str(node.get("sources", "")).split("|"):
        if source.startswith("offset_clone_"):
            try:
                layers.add(int(source.rsplit("_", 1)[-1]))
            except ValueError:
                pass
    return layers


def has_source_token(node: dict[str, Any], prefix: str) -> bool:
    return any(source.startswith(prefix) for source in str(node.get("sources", "")).split("|"))


def variant_specs(layers: tuple[int, ...], max_layer_combinations: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"name": "current", "keep_layers": None, "keep_endpoint_forward": True},
        {"name": "no_endpoint_forward", "keep_layers": None, "keep_endpoint_forward": False},
        {"name": "no_offset_layers", "keep_layers": (), "keep_endpoint_forward": True},
    ]
    if not layers:
        return specs

    combinations: list[tuple[int, ...]] = []
    for size in range(1, len(layers) + 1):
        combinations.extend(tuple(combo) for combo in itertools.combinations(layers, size))
    combinations = combinations[: max(0, int(max_layer_combinations))]
    for combo in combinations:
        suffix = "_".join(str(value) for value in combo)
        specs.append({"name": f"keep_offset_{suffix}", "keep_layers": combo, "keep_endpoint_forward": True})
        specs.append({"name": f"keep_offset_{suffix}_noef", "keep_layers": combo, "keep_endpoint_forward": False})
    for layer in layers:
        kept = tuple(value for value in layers if value != layer)
        specs.append({"name": f"drop_offset_{layer}", "keep_layers": kept, "keep_endpoint_forward": True})
    return specs


def apply_variant(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, int]],
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    keep_layers = spec.get("keep_layers")
    keep_endpoint_forward = bool(spec.get("keep_endpoint_forward", True))
    keep_node_ids: set[int] = set()
    filtered_nodes: list[dict[str, Any]] = []
    for node in nodes:
        if not keep_endpoint_forward and has_source_token(node, "endpoint_forward:"):
            continue
        node_layers = node_offset_layers(node)
        if keep_layers is not None and node_layers and not node_layers.intersection(set(keep_layers)):
            continue
        filtered_nodes.append(node)
        keep_node_ids.add(int(node["node_id"]))
    filtered_edges = [
        edge
        for edge in edges
        if int(edge["source_id"]) in keep_node_ids and int(edge["target_id"]) in keep_node_ids
    ]
    used_node_ids = {int(value) for edge in filtered_edges for value in (edge["source_id"], edge["target_id"])}
    endpoint_nodes = [node for node in filtered_nodes if int(node["node_id"]) in used_node_ids]
    return endpoint_nodes, filtered_edges


def main() -> int:
    args = parse_args()
    datasets = parse_csv(args.datasets)
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
    rows: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []
    started = time.time()
    store = ZipStore(args.zip, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            total_t = int(array.shape[0])
            start = first_gt_clip_start(gt, total_t, args.clip_length)
            end = min(total_t, start + int(args.clip_length))
            gt_clip = crop_geff_arrays(gt, start, end, total_t)
            result = predict_dataset_window(
                dataset=dataset,
                array=array,
                attrs=attrs,
                start=start,
                end=end,
                forced_route=None if args.route == "auto" else args.route,
                allow_division_route=False,
                estimated_total_nodes=float(gt.get("estimated_number_of_nodes"))
                if gt.get("estimated_number_of_nodes") is not None
                else None,
                scoring_calibration=scoring_calibration,
                progress=False,
            )
            layers = offset_layers_in_nodes(result.nodes)
            specs = variant_specs(layers, args.max_layer_combinations)
            dataset_rows: list[dict[str, Any]] = []
            for spec in specs:
                nodes, edges = apply_variant(result.nodes, result.edges, spec)
                metrics = official_metric_row(nodes, edges, gt_clip, parse_scale(attrs))
                graph = submission_graph_diagnostics(submission_rows([(dataset, nodes, edges)]))
                row = {
                    "dataset": dataset,
                    "route": result.route.label,
                    "clip_start": start,
                    "clip_end": end,
                    "variant": spec["name"],
                    "keep_layers": None if spec.get("keep_layers") is None else list(spec["keep_layers"]),
                    "keep_endpoint_forward": bool(spec.get("keep_endpoint_forward", True)),
                    "num_pred_nodes": int(len(nodes)),
                    "num_pred_edges": int(len(edges)),
                    "graph": graph,
                    "available_offset_layers": list(layers),
                    "route_reason": result.route.reason,
                    "offset_clone_layers": result.diagnostics.get("offset_clone_layers"),
                    "endpoint_forward_clones": result.diagnostics.get("endpoint_forward_clones"),
                }
                row.update(metrics)
                rows.append(row)
                dataset_rows.append(row)
            best = max(
                dataset_rows,
                key=lambda row: (
                    finite(row.get("adj_edge_jaccard")),
                    finite(row.get("edge_jaccard")),
                    -finite(row.get("num_pred_nodes")),
                ),
            )
            current = next(row for row in dataset_rows if row["variant"] == "current")
            report = {
                "dataset": dataset,
                "route": result.route.label,
                "current_score": finite(current.get("adj_edge_jaccard")),
                "current_edge_jaccard": finite(current.get("edge_jaccard")),
                "best_variant": best["variant"],
                "best_score": finite(best.get("adj_edge_jaccard")),
                "best_edge_jaccard": finite(best.get("edge_jaccard")),
                "best_node_recall": finite(best.get("node_recall")),
                "best_nodes": int(best["num_pred_nodes"]),
                "best_edges": int(best["num_pred_edges"]),
                "best_keep_layers": best.get("keep_layers"),
                "best_keep_endpoint_forward": bool(best.get("keep_endpoint_forward")),
                "delta_score": finite(best.get("adj_edge_jaccard")) - finite(current.get("adj_edge_jaccard")),
            }
            dataset_reports.append(report)
            print(json.dumps(report, default=str), flush=True)
    finally:
        store.close()

    payload = {
        "kind": "postclone_filter_variant_probe",
        "created_at_unix": time.time(),
        "zip": str(args.zip),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "fold_id": str(args.fold_id),
        "datasets": datasets,
        "route_arg": args.route,
        "clip_length": int(args.clip_length),
        "scoring_json": str(args.scoring_json) if args.scoring_json is not None else None,
        "summary": official_summary(rows),
        "dataset_reports": dataset_reports,
        "rows": rows,
        "runtime_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runtime_sec": payload["runtime_sec"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
