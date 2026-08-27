#!/usr/bin/env python3
"""Diagnose whether missed GT edges are present in the candidate graph."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.pipeline import (  # noqa: E402
    _add_endpoint_backward_clones,
    _add_endpoint_forward_clones,
    _admit_free_candidate_edges,
    _admit_motion_endpoint_candidates,
    _admit_refined_candidate_nodes,
    _clone_selected_nodes_by_localmax_layers,
    _clone_selected_nodes_by_offset_layers,
    _nodes_used_by_edges,
    _prune_submission_components,
    _prune_unlinked_submission_nodes,
    _prune_short_selected_edges,
    _refine_selected_nodes_by_localmax,
    _rewire_localmax_clone_targets,
    _rescue_duplicate_endpoint_conflicts,
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
from celltrack_alpha.solver import build_edge_candidates, select_edges  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_SPLIT = Path("reports/general_model_split_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", default="0")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--route", default="auto")
    parser.add_argument("--scoring-json", type=Path, default=None)
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


def match_nodes(
    pred_nodes: list[dict[str, Any]],
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
    max_distance_um: float = 7.0,
) -> dict[int, int]:
    gt_nodes = gt["nodes"]
    pred_by_t: dict[int, list[int]] = {}
    gt_by_t: dict[int, list[int]] = {}
    for idx, node in enumerate(pred_nodes):
        pred_by_t.setdefault(int(node["t"]), []).append(idx)
    for idx, t in enumerate(gt_nodes["t"]):
        gt_by_t.setdefault(int(t), []).append(idx)
    scale = np.asarray(scale_zyx, dtype=np.float32)
    pred_to_gt: dict[int, int] = {}
    for t, pred_idxs in sorted(pred_by_t.items()):
        gt_idxs = gt_by_t.get(t, [])
        if not pred_idxs or not gt_idxs:
            continue
        pred_xyz = np.asarray(
            [[pred_nodes[i]["z"], pred_nodes[i]["y"], pred_nodes[i]["x"]] for i in pred_idxs],
            dtype=np.float32,
        )
        gt_xyz = np.asarray(
            [[gt_nodes["z"][i], gt_nodes["y"][i], gt_nodes["x"][i]] for i in gt_idxs],
            dtype=np.float32,
        )
        dist = np.sqrt(np.sum(((pred_xyz[:, None, :] - gt_xyz[None, :, :]) * scale) ** 2, axis=2))
        feasible = dist <= float(max_distance_um)
        if not feasible.any():
            continue
        rows, cols = linear_sum_assignment(np.where(feasible, dist, max_distance_um + 1_000_000.0))
        for row, col in zip(rows, cols, strict=True):
            if dist[row, col] <= max_distance_um:
                pred_to_gt[int(pred_nodes[pred_idxs[row]]["node_id"])] = int(gt_nodes["node_id"][gt_idxs[col]])
    return pred_to_gt


def source_family(sources: tuple[str, ...] | list[str]) -> str:
    tokens = tuple(str(source) for source in sources)
    if any(source.startswith("endpoint_forward:") for source in tokens):
        return "endpoint_forward"
    if any(source.startswith("endpoint_backward:") for source in tokens):
        return "endpoint_backward"
    if any(source.startswith("offset_clone_") for source in tokens):
        return "offset_clone"
    if any(source.startswith("localmax_clone_") for source in tokens):
        return "localmax_clone"
    return "base"


def nearest_emitted_for_gt(
    gt_row: dict[str, Any],
    emitted_nodes: list[Any],
    scale_zyx: tuple[float, float, float],
) -> dict[str, Any] | None:
    same_t = [node for node in emitted_nodes if int(node.t) == int(gt_row["t"])]
    if not same_t:
        return None
    gt_coord = np.asarray([gt_row["z"], gt_row["y"], gt_row["x"]], dtype=np.float64)
    coords = np.asarray([node.coord_zyx for node in same_t], dtype=np.float64)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    distances = np.sqrt(np.sum(((coords - gt_coord[None, :]) * scale) ** 2, axis=1))
    idx = int(np.argmin(distances))
    node = same_t[idx]
    return {
        "node_id": int(node.node_id),
        "z": int(node.z),
        "y": int(node.y),
        "x": int(node.x),
        "distance_um": float(distances[idx]),
        "delta_zyx_vox": [
            float(value)
            for value in (
                coords[idx] - gt_coord
            )
        ],
        "source_family": source_family(node.sources),
        "sources": list(node.sources),
    }


def gt_node_rows(gt: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = gt["nodes"]
    rows: list[dict[str, Any]] = []
    for idx, node_id in enumerate(nodes["node_id"]):
        rows.append(
            {
                "node_id": int(node_id),
                "t": int(nodes["t"][idx]),
                "z": float(nodes["z"][idx]),
                "y": float(nodes["y"][idx]),
                "x": float(nodes["x"][idx]),
            }
        )
    return rows


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
    configs = default_route_configs()
    started = time.time()
    reports: list[dict[str, Any]] = []
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
            estimated_total_nodes = float(gt.get("estimated_number_of_nodes")) if gt.get("estimated_number_of_nodes") is not None else None
            profile = profile_scenario(dataset, array, attrs, start=start, end=end, estimated_total_nodes=estimated_total_nodes)
            decision = route_for_profile(profile, configs, args.route, allow_division_route=False)
            scoring_payload = scoring_for_dataset(dataset, scoring_calibration)
            proposal_policy = proposal_policy_for_dataset(scoring_payload, dataset=dataset, route_label=decision.label)
            nodes_by_t, _candidate_nodes = generate_candidates(
                dataset=dataset,
                array=array,
                attrs=attrs,
                config=decision.config,
                start=start,
                end=end,
                progress=False,
                proposal_policy=proposal_policy,
            )
            nodes_by_t, correction_stats = correct_nodes_by_t(
                nodes_by_t,
                dataset=dataset,
                route_label=decision.label,
                calibration_payload=scoring_payload,
                spatial_shape_zyx=tuple(int(v) for v in array.shape[1:4]),
            )
            if correction_stats is not None:
                _refresh_node_intensities(nodes_by_t, array, start)
                annotate_local_density(nodes_by_t, scale_zyx)
            nodes_by_t, pruning_stats = prune_nodes_by_t(
                nodes_by_t,
                dataset=dataset,
                route_label=decision.label,
                calibration_payload=scoring_payload,
            )
            if pruning_stats is not None:
                annotate_local_density(nodes_by_t, scale_zyx)
            annotate_local_density(nodes_by_t, scale_zyx)
            nodes_by_t, refined_admission_stats = _admit_refined_candidate_nodes(
                dataset, array, attrs, start, end, nodes_by_t, scale_zyx, decision.config
            )
            candidate_nodes = [node for nodes in nodes_by_t for node in nodes]
            raw_candidate_submission_nodes = [node.to_submission_node() for node in candidate_nodes]
            raw_candidate_to_gt = match_nodes(raw_candidate_submission_nodes, gt_clip, scale_zyx)
            edge_candidates = build_edge_candidates(nodes_by_t, scale_zyx, decision.config)
            edge_candidates, edge_scoring_stats = score_edge_candidates(
                edge_candidates,
                nodes_by_t,
                dataset=dataset,
                route_label=decision.label,
                route_config=decision.config,
                calibration_payload=scoring_payload,
            )
            selected_edges, forks = select_edges(nodes_by_t, edge_candidates, scale_zyx, decision.config)
            selected_edges, refined_chain_conflict_stats = _rescue_refined_chain_conflicts(
                selected_edges, edge_candidates, candidate_nodes, decision.config
            )
            selected_edges, short_edge_pruning_stats = _prune_short_selected_edges(
                selected_edges, decision.config
            )
            localmax_stats = _refine_selected_nodes_by_localmax(
                nodes_by_t, selected_edges, array, attrs, start, scale_zyx, decision.config
            )
            selected_edges, free_edge_admission_stats = _admit_free_candidate_edges(
                selected_edges, edge_candidates, candidate_nodes, decision.config
            )
            selected_edges, duplicate_endpoint_rescue_stats = _rescue_duplicate_endpoint_conflicts(
                selected_edges, edge_candidates, candidate_nodes, scale_zyx, decision.config
            )
            selected_edges, motion_endpoint_admission_stats = _admit_motion_endpoint_candidates(
                selected_edges, candidate_nodes, scale_zyx, decision.config
            )
            emitted_nodes, node_emission = _nodes_used_by_edges(candidate_nodes, selected_edges)
            emitted_nodes, selected_edges, clone_stats = _clone_selected_nodes_by_localmax_layers(
                emitted_nodes, selected_edges, array, attrs, start, scale_zyx, decision.config
            )
            emitted_nodes, selected_edges, offset_clone_stats = _clone_selected_nodes_by_offset_layers(
                emitted_nodes, selected_edges, array, start, decision.config
            )
            emitted_nodes, selected_edges, endpoint_backward_stats = _add_endpoint_backward_clones(
                emitted_nodes, selected_edges, array, start, scale_zyx, decision.config
            )
            emitted_nodes, selected_edges, endpoint_forward_stats = _add_endpoint_forward_clones(
                emitted_nodes, selected_edges, array, attrs, start, scale_zyx, decision.config
            )
            emitted_nodes, selected_edges, localmax_clone_rewire_stats = _rewire_localmax_clone_targets(
                emitted_nodes, selected_edges, scale_zyx, decision.config
            )
            emitted_nodes, selected_edges, component_pruning_stats = _prune_submission_components(
                emitted_nodes, selected_edges, decision.config
            )
            emitted_nodes, final_node_pruning_stats = _prune_unlinked_submission_nodes(
                emitted_nodes, selected_edges
            )
            submission_nodes = [node.to_submission_node() for node in emitted_nodes]
            submission_nodes = _shift_node_dicts(submission_nodes, (0, 0, 0), tuple(int(v) for v in array.shape[1:4]))
            pred_to_gt = match_nodes(submission_nodes, gt_clip, scale_zyx)
            emitted_by_id = {int(node.node_id): node for node in emitted_nodes}
            gt_by_id = {int(row["node_id"]): row for row in gt_node_rows(gt_clip)}
            gt_to_pred = {int(gt_id): int(pred_id) for pred_id, gt_id in pred_to_gt.items()}
            matched_node_source_counts = Counter(
                source_family(emitted_by_id[pred_id].sources)
                for pred_id in pred_to_gt
                if pred_id in emitted_by_id
            )
            matched_gt_nodes = []
            unmatched_gt_nodes = []
            for gt_id, gt_row in sorted(gt_by_id.items()):
                pred_id = gt_to_pred.get(gt_id)
                nearest = nearest_emitted_for_gt(gt_row, emitted_nodes, scale_zyx)
                if pred_id is None:
                    unmatched_gt_nodes.append({**gt_row, "nearest_emitted": nearest})
                    continue
                pred_node = emitted_by_id.get(pred_id)
                matched_gt_nodes.append(
                    {
                        **gt_row,
                        "pred_node_id": int(pred_id),
                        "pred_zyx": [int(pred_node.z), int(pred_node.y), int(pred_node.x)] if pred_node is not None else None,
                        "source_family": source_family(pred_node.sources) if pred_node is not None else None,
                        "sources": list(pred_node.sources) if pred_node is not None else [],
                        "nearest_emitted": nearest,
                    }
                )
            gt_edges = {(int(a), int(b)) for a, b in gt_clip["edges"]}
            selected_pairs = {(int(edge.source_id), int(edge.target_id)) for edge in selected_edges}
            selected_gt_pairs: set[tuple[int, int]] = set()
            selected_tp_pairs: set[tuple[int, int]] = set()
            selected_valid = 0
            for edge in selected_edges:
                pair = (int(edge.source_id), int(edge.target_id))
                src_gt = pred_to_gt.get(pair[0])
                tgt_gt = pred_to_gt.get(pair[1])
                if src_gt is None and tgt_gt is None:
                    continue
                # Official pred_valid marks an edge valid if either endpoint is matched
                # to a GT node with corresponding in/out annotation. This is a compact
                # approximation for oracle triage, not a replacement for final scoring.
                selected_valid += 1
                if src_gt is not None and tgt_gt is not None:
                    gt_pair = (src_gt, tgt_gt)
                    selected_gt_pairs.add(gt_pair)
                    if gt_pair in gt_edges:
                        selected_tp_pairs.add(gt_pair)
            emitted_ids = {int(node.node_id) for node in emitted_nodes}
            unselected_emitted_hits: list[dict[str, Any]] = []
            unselected_emitted_conflicts = Counter()
            raw_candidate_gt_edge_hits: list[dict[str, Any]] = []
            raw_candidate_gt_edge_conflicts = Counter()
            selected_out = Counter(int(edge.source_id) for edge in selected_edges)
            selected_in = Counter(int(edge.target_id) for edge in selected_edges)
            for edge in edge_candidates:
                pair = (int(edge.source_id), int(edge.target_id))
                raw_src_gt = raw_candidate_to_gt.get(pair[0])
                raw_tgt_gt = raw_candidate_to_gt.get(pair[1])
                if raw_src_gt is not None and raw_tgt_gt is not None and (raw_src_gt, raw_tgt_gt) in gt_edges:
                    conflicts: list[str] = []
                    if selected_out[pair[0]]:
                        conflicts.append("source_out_used")
                    if selected_in[pair[1]]:
                        conflicts.append("target_in_used")
                    if not conflicts:
                        conflicts.append("free")
                    for conflict in conflicts:
                        raw_candidate_gt_edge_conflicts[conflict] += 1
                    raw_candidate_gt_edge_hits.append(
                        {
                            "selected": pair in selected_pairs,
                            "emitted_source": pair[0] in emitted_ids,
                            "emitted_target": pair[1] in emitted_ids,
                            "source_id": pair[0],
                            "target_id": pair[1],
                            "gt_source_id": raw_src_gt,
                            "gt_target_id": raw_tgt_gt,
                            "distance_um": float(edge.distance_um),
                            "cost": float(edge.cost),
                            "score": float(edge.score),
                            "features": dict(edge.features),
                            "conflicts": conflicts,
                        }
                    )
                if pair in selected_pairs or pair[0] not in emitted_ids or pair[1] not in emitted_ids:
                    continue
                src_gt = pred_to_gt.get(pair[0])
                tgt_gt = pred_to_gt.get(pair[1])
                if src_gt is None or tgt_gt is None or (src_gt, tgt_gt) not in gt_edges:
                    continue
                conflicts = []
                if selected_out[pair[0]]:
                    conflicts.append("source_out_used")
                if selected_in[pair[1]]:
                    conflicts.append("target_in_used")
                if not conflicts:
                    conflicts.append("free")
                for conflict in conflicts:
                    unselected_emitted_conflicts[conflict] += 1
                unselected_emitted_hits.append(
                    {
                        "source_id": pair[0],
                        "target_id": pair[1],
                        "gt_source_id": src_gt,
                        "gt_target_id": tgt_gt,
                        "distance_um": float(edge.distance_um),
                        "cost": float(edge.cost),
                        "score": float(edge.score),
                        "features": dict(edge.features),
                        "conflicts": conflicts,
                    }
                )
            raw_candidate_gt_edges = {(row["gt_source_id"], row["gt_target_id"]) for row in raw_candidate_gt_edge_hits}
            missed_gt_edges = sorted(gt_edges - selected_tp_pairs)
            report = {
                "dataset": dataset,
                "route": decision.label,
                "route_reason": decision.reason,
                "clip_start": start,
                "clip_end": end,
                "gt_nodes": int(len(gt_clip["nodes"]["node_id"])),
                "gt_edges": int(len(gt_edges)),
                "candidate_nodes": int(len(candidate_nodes)),
                "edge_candidates": int(len(edge_candidates)),
                "selected_edges": int(len(selected_edges)),
                "emitted_nodes": int(len(emitted_nodes)),
                "matched_emitted_nodes": int(len(pred_to_gt)),
                "matched_raw_candidate_nodes": int(len(raw_candidate_to_gt)),
                "matched_emitted_source_counts": dict(matched_node_source_counts),
                "matched_gt_nodes": matched_gt_nodes,
                "unmatched_gt_nodes": unmatched_gt_nodes,
                "selected_gt_edge_tp": int(len(selected_tp_pairs)),
                "selected_gt_edge_pairs": [list(pair) for pair in sorted(selected_tp_pairs)],
                "missed_gt_edges": [list(pair) for pair in missed_gt_edges],
                "raw_candidate_gt_edge_pairs": [list(pair) for pair in sorted(raw_candidate_gt_edges)],
                "raw_candidate_gt_edge_hit_count": len(raw_candidate_gt_edge_hits),
                "raw_candidate_gt_edge_hits": raw_candidate_gt_edge_hits,
                "raw_candidate_gt_edge_conflicts": dict(raw_candidate_gt_edge_conflicts),
                "unselected_emitted_gt_edge_hits": unselected_emitted_hits,
                "unselected_emitted_gt_edge_hit_count": len(unselected_emitted_hits),
                "unselected_emitted_conflicts": dict(unselected_emitted_conflicts),
                "selected_valid_edge_approx": int(selected_valid),
                "node_emission": node_emission,
                "coordinate_correction": correction_stats.as_dict() if correction_stats is not None else None,
                "node_pruning": pruning_stats.as_dict() if pruning_stats is not None else None,
                "refined_candidate_admission": refined_admission_stats,
                "edge_scoring": edge_scoring_stats.as_dict() if edge_scoring_stats is not None else None,
                "refined_chain_conflict_rescue": refined_chain_conflict_stats,
                "short_edge_pruning": short_edge_pruning_stats,
                "localmax_refinement": localmax_stats,
                "free_edge_admission": free_edge_admission_stats,
                "duplicate_endpoint_rescue": duplicate_endpoint_rescue_stats,
                "motion_endpoint_admission": motion_endpoint_admission_stats,
                "localmax_clone_layers": clone_stats,
                "offset_clone_layers": offset_clone_stats,
                "endpoint_backward_clones": endpoint_backward_stats,
                "endpoint_forward_clones": endpoint_forward_stats,
                "localmax_clone_target_rewire": localmax_clone_rewire_stats,
                "component_pruning": component_pruning_stats,
                "final_node_pruning": final_node_pruning_stats,
                "forks": len(forks),
            }
            reports.append(report)
            print(json.dumps({k: report[k] for k in ("dataset", "route", "gt_edges", "selected_gt_edge_tp", "matched_raw_candidate_nodes", "raw_candidate_gt_edge_hit_count", "raw_candidate_gt_edge_conflicts", "unselected_emitted_gt_edge_hit_count", "unselected_emitted_conflicts")}, default=str), flush=True)
    finally:
        store.close()

    payload = {
        "kind": "candidate_edge_oracle_diagnostic",
        "created_at_unix": time.time(),
        "zip": str(args.zip),
        "split_manifest": str(args.split_manifest),
        "fold_id": str(args.fold_id),
        "datasets": datasets,
        "route_arg": args.route,
        "clip_length": int(args.clip_length),
        "scoring_json": str(args.scoring_json) if args.scoring_json is not None else None,
        "reports": reports,
        "runtime_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runtime_sec": payload["runtime_sec"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
