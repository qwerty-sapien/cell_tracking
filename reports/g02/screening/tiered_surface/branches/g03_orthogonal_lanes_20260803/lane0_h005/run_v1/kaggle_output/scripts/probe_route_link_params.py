#!/usr/bin/env python3
"""Probe route link parameters with cached candidate proposals."""

from __future__ import annotations

import argparse
import copy
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

from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_alpha.pipeline import (  # noqa: E402
    _admit_free_candidate_edges,
    _admit_motion_endpoint_candidates,
    _clone_selected_nodes_by_localmax_layers,
    _localmax_refinement_maps,
    _nodes_used_by_edges,
    _refine_selected_nodes_by_localmax,
    _rescue_refined_chain_conflicts,
    _refresh_node_intensities,
    _shift_node_dicts,
)
from celltrack_alpha.proposals import annotate_local_density, generate_candidates  # noqa: E402
from celltrack_alpha.routing import default_route_configs, profile_scenario, route_for_profile  # noqa: E402
from celltrack_alpha.scoring import (  # noqa: E402
    correct_nodes_by_t,
    load_scoring_calibration,
    proposal_policy_for_dataset,
    prune_nodes_by_t,
    score_edge_candidates,
    scoring_for_dataset,
)
from celltrack_alpha.solver import build_edge_candidates, graph_diagnostics, select_edges  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_SPLIT = Path("reports/general_model_split_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", default="0")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument(
        "--use-auto-route-config",
        action="store_true",
        help="Resolve the production auto/profile config and require that it selects --route.",
    )
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--scoring-json", type=Path, default=None)
    parser.add_argument("--max-link-um", default="6.0")
    parser.add_argument("--max-neighbors", default="7")
    parser.add_argument("--edge-cost-threshold", default="0.85")
    parser.add_argument(
        "--birth-cost",
        default="auto",
        help="Comma-separated birth costs, or auto to use the route default.",
    )
    parser.add_argument(
        "--death-cost",
        default="auto",
        help="Comma-separated death costs, or auto to use the route default.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(part) for part in parse_csv(value)]


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in parse_csv(value)]


def parse_optional_floats(value: str) -> list[float | None]:
    parsed: list[float | None] = []
    for part in parse_csv(value):
        if part.lower() in {"auto", "none", "default", "null"}:
            parsed.append(None)
        else:
            parsed.append(float(part))
    return parsed or [None]


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


def main() -> int:
    args = parse_args()
    datasets = parse_csv(args.datasets)
    fold = load_fold(args.split_manifest, str(args.fold_id))
    validation = {str(value) for value in fold.get("validation", [])}
    invalid = sorted(set(datasets) - validation)
    if invalid:
        raise ValueError(f"datasets are outside fold {args.fold_id} validation set: {invalid[:10]}")

    configs = default_route_configs()
    if args.route not in configs:
        raise ValueError(f"unknown route {args.route!r}")
    scoring_calibration = load_scoring_calibration(
        args.scoring_json,
        split_manifest_path=args.split_manifest,
        expected_fold_id=str(args.fold_id) if args.scoring_json is not None else None,
    )

    grid = [
        (
            float(max_link_um),
            int(max_neighbors),
            float(edge_cost_threshold),
            birth_cost,
            death_cost,
        )
        for max_link_um in parse_floats(args.max_link_um)
        for max_neighbors in parse_ints(args.max_neighbors)
        for edge_cost_threshold in parse_floats(args.edge_cost_threshold)
        for birth_cost in parse_optional_floats(args.birth_cost)
        for death_cost in parse_optional_floats(args.death_cost)
    ]
    started = time.time()
    rows: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []
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
            if args.use_auto_route_config:
                profile = profile_scenario(
                    dataset=dataset,
                    array=array,
                    attrs=attrs,
                    start=start,
                    end=end,
                    estimated_total_nodes=float(gt.get("estimated_number_of_nodes"))
                    if gt.get("estimated_number_of_nodes") is not None
                    else None,
                )
                decision = route_for_profile(profile, configs)
                if decision.label != args.route:
                    raise ValueError(
                        f"auto route for {dataset} selected {decision.label!r}, not requested {args.route!r}"
                    )
                base_config = decision.config
                route_reason = decision.reason
            else:
                base_config = configs[args.route]
                route_reason = "forced base route config"
            scoring_payload = scoring_for_dataset(dataset, scoring_calibration)
            proposal_policy = proposal_policy_for_dataset(
                scoring_payload, dataset=dataset, route_label=args.route
            )
            nodes_by_t, candidate_nodes = generate_candidates(
                dataset=dataset,
                array=array,
                attrs=attrs,
                config=base_config,
                start=start,
                end=end,
                progress=False,
                proposal_policy=proposal_policy,
            )
            nodes_by_t, correction_stats = correct_nodes_by_t(
                nodes_by_t,
                dataset=dataset,
                route_label=args.route,
                calibration_payload=scoring_payload,
                spatial_shape_zyx=tuple(int(v) for v in array.shape[1:4]),
            )
            if correction_stats is not None:
                _refresh_node_intensities(nodes_by_t, array, start)
                annotate_local_density(nodes_by_t, scale_zyx)
            nodes_by_t, pruning_stats = prune_nodes_by_t(
                nodes_by_t,
                dataset=dataset,
                route_label=args.route,
                calibration_payload=scoring_payload,
            )
            if pruning_stats is not None:
                annotate_local_density(nodes_by_t, scale_zyx)
            annotate_local_density(nodes_by_t, scale_zyx)
            candidate_nodes = [node for nodes in nodes_by_t for node in nodes]
            localmax_score_maps = None
            if getattr(base_config, "enable_localmax_refinement", False):
                localmax_score_maps = {
                    int(rel_t): _localmax_refinement_maps(
                        np.asarray(array[start + int(rel_t)]),
                        attrs,
                        scale_zyx,
                        tuple(str(source) for source in base_config.localmax_refine_sources),
                    )
                    for rel_t in range(end - start)
                }
            for max_link_um, max_neighbors, edge_cost_threshold, birth_cost, death_cost in grid:
                config = replace(
                    base_config,
                    max_link_um=float(max_link_um),
                    max_neighbors=int(max_neighbors),
                    edge_cost_threshold=float(edge_cost_threshold),
                    birth_cost=birth_cost,
                    death_cost=death_cost,
                )
                loop_nodes_by_t = copy.deepcopy(nodes_by_t)
                loop_candidate_nodes = [node for nodes in loop_nodes_by_t for node in nodes]
                edge_candidates = build_edge_candidates(loop_nodes_by_t, scale_zyx, config)
                edge_candidates, edge_scoring_stats = score_edge_candidates(
                    edge_candidates,
                    loop_nodes_by_t,
                    dataset=dataset,
                    route_label=args.route,
                    route_config=config,
                    calibration_payload=scoring_payload,
                )
                selected_edges, forks = select_edges(loop_nodes_by_t, edge_candidates, scale_zyx, config)
                selected_edges, refined_chain_conflict_stats = _rescue_refined_chain_conflicts(
                    selected_edges, edge_candidates, loop_candidate_nodes, config
                )
                localmax_refinement_stats = _refine_selected_nodes_by_localmax(
                    loop_nodes_by_t,
                    selected_edges,
                    array,
                    attrs,
                    start,
                    scale_zyx,
                    config,
                    score_maps_by_rel_t=localmax_score_maps,
                )
                selected_edges, free_edge_admission_stats = _admit_free_candidate_edges(
                    selected_edges, edge_candidates, loop_candidate_nodes, config
                )
                selected_edges, motion_endpoint_admission_stats = _admit_motion_endpoint_candidates(
                    selected_edges, loop_candidate_nodes, scale_zyx, config
                )
                emitted_candidate_nodes, node_emission = _nodes_used_by_edges(loop_candidate_nodes, selected_edges)
                emitted_candidate_nodes, selected_edges, localmax_clone_stats = (
                    _clone_selected_nodes_by_localmax_layers(
                        emitted_candidate_nodes, selected_edges, array, attrs, start, scale_zyx, config
                    )
                )
                nodes = [node.to_submission_node() for node in emitted_candidate_nodes]
                nodes = _shift_node_dicts(nodes, (0, 0, 0), tuple(int(v) for v in array.shape[1:4]))
                edges = [edge.to_submission_edge() for edge in selected_edges]
                metrics = official_metric_row(nodes, edges, gt_clip, scale_zyx)
                row = {
                    "dataset": dataset,
                    "route": args.route,
                    "use_auto_route_config": bool(args.use_auto_route_config),
                    "route_reason": route_reason,
                    "clip_start": start,
                    "clip_end": end,
                    "max_link_um": float(max_link_um),
                    "max_neighbors": int(max_neighbors),
                    "edge_cost_threshold": float(edge_cost_threshold),
                    "birth_cost": None if birth_cost is None else float(birth_cost),
                    "death_cost": None if death_cost is None else float(death_cost),
                    "num_pred_nodes": int(len(nodes)),
                    "num_pred_edges": int(len(edges)),
                    "num_edge_candidates": int(len(edge_candidates)),
                    "num_selected_forks": int(len(forks)),
                    "node_emission": node_emission,
                    "coordinate_correction": correction_stats.as_dict() if correction_stats is not None else None,
                    "node_pruning": pruning_stats.as_dict() if pruning_stats is not None else None,
                    "edge_scoring": edge_scoring_stats.as_dict() if edge_scoring_stats is not None else None,
                    "refined_chain_conflict_rescue": refined_chain_conflict_stats,
                    "localmax_refinement": localmax_refinement_stats,
                    "free_edge_admission": free_edge_admission_stats,
                    "motion_endpoint_admission": motion_endpoint_admission_stats,
                    "localmax_clone_layers": localmax_clone_stats,
                    "graph": graph_diagnostics(emitted_candidate_nodes, selected_edges),
                }
                row.update(metrics)
                rows.append(row)
            best = max(
                (row for row in rows if row["dataset"] == dataset),
                key=lambda row: (finite(row.get("adj_edge_jaccard")), finite(row.get("edge_jaccard")), -finite(row.get("num_pred_nodes"))),
            )
            dataset_reports.append(
                {
                    "dataset": dataset,
                    "best_score": finite(best.get("adj_edge_jaccard")),
                    "best_edge_jaccard": finite(best.get("edge_jaccard")),
                    "best_node_recall": finite(best.get("node_recall")),
                    "best_max_link_um": best["max_link_um"],
                    "best_max_neighbors": best["max_neighbors"],
                    "best_edge_cost_threshold": best["edge_cost_threshold"],
                    "best_birth_cost": best["birth_cost"],
                    "best_death_cost": best["death_cost"],
                    "best_nodes": best["num_pred_nodes"],
                    "best_edges": best["num_pred_edges"],
                }
            )
            print(json.dumps(dataset_reports[-1], default=str), flush=True)
    finally:
        store.close()

    payload = {
        "kind": "route_link_param_probe",
        "created_at_unix": time.time(),
        "zip": str(args.zip),
        "split_manifest": str(args.split_manifest),
        "fold_id": str(args.fold_id),
        "datasets": datasets,
        "route": args.route,
        "use_auto_route_config": bool(args.use_auto_route_config),
        "clip_length": int(args.clip_length),
        "scoring_json": str(args.scoring_json) if args.scoring_json is not None else None,
        "grid": {
            "max_link_um": parse_floats(args.max_link_um),
            "max_neighbors": parse_ints(args.max_neighbors),
            "edge_cost_threshold": parse_floats(args.edge_cost_threshold),
            "birth_cost": parse_optional_floats(args.birth_cost),
            "death_cost": parse_optional_floats(args.death_cost),
        },
        "summary": official_summary(rows),
        "dataset_reports": dataset_reports,
        "rows": rows,
        "runtime_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": payload["summary"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
