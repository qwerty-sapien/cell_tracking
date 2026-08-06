#!/usr/bin/env python3
"""Score the two frozen P4 arms on the exact patched CV57 holdout."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
import scoreall  # noqa: E402
import score_bh_cv57  # noqa: E402
import sweep_bh_graph_weights as sweep_support  # noqa: E402
from celltrack_alpha import bh_edge, bh_graph, bh_model  # noqa: E402
from celltrack_alpha.edge_attention import (  # noqa: E402
    EdgeResidualModel,
    attention_neighbours,
)


SCHEMA = "celltrack-alpha.p4-edge-attention-cv57.v1"
BANKED_BASELINE = 0.8731495935289003
BANKED_COUNTS = {"edge_tp": 34205, "edge_fp": 2685, "edge_fn": 2063}
DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_CACHE = ROOT / "kaggle_kernel/bhcv57out_v3/cache"
DEFAULT_EDGE = ROOT / "_ARCHIVE/conflict_residual/reports/g02/conflict_edge_f0_fit117_v1/edge.json"
DEFAULT_AUXILIARY = ROOT / "_ARCHIVE/compact_head/reports/g02/v45f4.pt"
DEFAULT_SUPPORT = ROOT / "external/Tracking_Support"
DEFAULT_PARITY = score_bh_cv57.DEFAULT_SCORER_PARITY
DEFAULT_MODELS = ROOT / "reports/g02/p4_edge_attention_prototype_v1/models.pt"
DEFAULT_PREPARE = ROOT / "reports/g02/p4_edge_attention_prototype_v1/prepare.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/p4_edge_attention_cv57_v1"
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


class P4CVError(RuntimeError):
    """The exact P4 CV57 evaluation contract was violated."""


def _load_models(path: Path, device: torch.device) -> tuple[dict[str, EdgeResidualModel], float]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != "celltrack-alpha.p4-edge-attention-prototype.v1":
        raise P4CVError("P4 model artifact schema is invalid")
    hidden, heads = int(payload["hidden"]), int(payload["heads"])
    models = {
        "no_attention": EdgeResidualModel(attention=False, hidden=hidden, heads=heads).to(device),
        "attention": EdgeResidualModel(attention=True, hidden=hidden, heads=heads).to(device),
    }
    for name, model in models.items():
        model.load_state_dict(payload[name], strict=True)
        model.eval()
    return models, float(payload["dummy_logit"])


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _normalize(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    source_index: np.ndarray,
    source_node_ids: np.ndarray,
    logits: np.ndarray,
    dummy_logit: float,
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    probabilities: dict[tuple[int, int], float] = {}
    no_link: dict[int, float] = {}
    for source_row, source_id in enumerate(source_node_ids):
        rows = np.flatnonzero(source_index == source_row)
        if len(rows) == 0:
            no_link[int(source_id)] = 1.0
            continue
        local = logits[rows]
        maximum = max(dummy_logit, float(np.max(local)))
        candidate_exp = np.exp(local - maximum)
        dummy_exp = np.exp(dummy_logit - maximum)
        denominator = float(candidate_exp.sum() + dummy_exp)
        no_link[int(source_id)] = float(dummy_exp / denominator)
        for row, value in zip(rows, candidate_exp / denominator, strict=True):
            probabilities[(int(source_ids[row]), int(target_ids[row]))] = float(value)
    return probabilities, no_link


def _score_cache(
    cache: Any,
    scorer: Any,
    models: Mapping[str, EdgeResidualModel],
    dummy_logit: float,
    device: torch.device,
) -> tuple[dict[str, tuple[dict[Any, float], dict[int, float]]], dict[str, Any]]:
    by_frame = {int(frame.frame_index): frame for frame in cache.detections}
    combined = {name: ({}, {}) for name in models}
    timings = {"base_pairwise_seconds": 0.0, "no_attention_seconds": 0.0, "attention_seconds": 0.0}
    candidate_count = attention_cells = 0
    for context in cache.contexts:
        source = by_frame[int(context.source_frame)]
        target = by_frame[int(context.target_frame)]
        _synchronize(device)
        started = time.perf_counter()
        candidates, checkpoint, features = scorer.candidate_features(context, source, target)
        with torch.no_grad():
            base = checkpoint + scorer.calibrator(features)
        _synchronize(device)
        timings["base_pairwise_seconds"] += time.perf_counter() - started
        source_geometry = np.asarray(source.coords_zyx, dtype=np.float64) * SCALE_ZYX
        target_geometry = np.asarray(target.coords_zyx, dtype=np.float64) * SCALE_ZYX
        geometry_np = np.column_stack(
            (source_geometry[candidates.source_index], target_geometry[candidates.target_index])
        ).astype(np.float32, copy=False)
        geometry = torch.as_tensor(geometry_np, device=device)
        empty_index = torch.empty((len(candidates), 0), dtype=torch.long, device=device)
        empty_mask = torch.empty((len(candidates), 0), dtype=torch.bool, device=device)
        _synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            no_attention_logits = base + models["no_attention"](
                features, geometry, empty_index, empty_mask
            )
        _synchronize(device)
        timings["no_attention_seconds"] += time.perf_counter() - started
        _synchronize(device)
        started = time.perf_counter()
        neighbours_np, mask_np = attention_neighbours(
            candidates.source_index, candidates.target_index, geometry_np
        )
        neighbours = torch.as_tensor(neighbours_np, device=device)
        mask = torch.as_tensor(mask_np, device=device)
        with torch.no_grad():
            attention_logits = base + models["attention"](
                features, geometry, neighbours, mask
            )
        _synchronize(device)
        timings["attention_seconds"] += time.perf_counter() - started
        candidate_count += len(candidates)
        attention_cells += int(mask_np.sum())
        for name, logits in (
            ("no_attention", no_attention_logits), ("attention", attention_logits)
        ):
            local_probabilities, local_no_link = _normalize(
                candidates.source_ids,
                candidates.target_ids,
                candidates.source_index,
                source.node_ids,
                logits.detach().cpu().numpy().astype(np.float64, copy=False),
                dummy_logit,
            )
            probabilities, no_link = combined[name]
            if set(probabilities).intersection(local_probabilities) or set(no_link).intersection(local_no_link):
                raise P4CVError("candidate or source IDs repeat across frame pairs")
            probabilities.update(local_probabilities)
            no_link.update(local_no_link)
    pair_count = len(cache.contexts)
    return combined, {
        "pair_count": pair_count,
        "candidate_count": candidate_count,
        "attention_cells": attention_cells,
        **timings,
        "base_pairwise_seconds_per_pair": timings["base_pairwise_seconds"] / max(pair_count, 1),
        "no_attention_incremental_seconds_per_pair": timings["no_attention_seconds"] / max(pair_count, 1),
        "attention_incremental_seconds_per_pair": timings["attention_seconds"] / max(pair_count, 1),
    }


def _solve(
    *,
    identity: str,
    dataset: str,
    shape: tuple[int, ...],
    graph_nodes: Sequence[Any],
    probabilities: Mapping[Any, float],
    no_link: Mapping[int, float],
    refiner: Any,
    fragment_dir: Path,
) -> dict[str, Any]:
    graph = bh_graph.calibrate_v45_graph(
        graph_nodes,
        learned_edge_probabilities=probabilities,
        source_no_link_probabilities=no_link,
        image_refine_callback=refiner,
        topology_mode="legacy",
        division_enabled=False,
    )
    if (
        graph.provenance.get("motion_direction") != "backward"
        or graph.provenance.get("motion_error_weight") != 0.25
        or graph.provenance.get("division_enabled") is not False
    ):
        raise P4CVError("P4 graph did not use the banked backward c=0.25 baseline")
    graph = run_bh.project_graph_result_to_image_support(
        dataset,
        graph,
        shape,
        scale_zyx=bh_graph.V45_GRAPH_CONFIG.scale_zyx,
        max_correction_um=bh_graph.V45_GRAPH_CONFIG.refine_max_shift_um,
    )
    rows = run_bh.graph_rows(dataset, graph, shape)
    destination = fragment_dir / f"{identity.replace(':', '__')}.csv"
    run_bh.write_rows_atomic(destination, rows, deadline_epoch=None)
    return {"fragment": str(destination), "nodes": len(graph.nodes), "edges": len(graph.edges)}


def _edge_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {name: int(sum(float(row[name]) for row in rows)) for name in ("edge_tp", "edge_fp", "edge_fn")}


def _decision(delta: float) -> tuple[str, str]:
    if delta >= 0.005:
        return "PROCEED", "proceed to the full 6-9 day build"
    if delta >= 0.002:
        return "HUMAN_DECISION", "report and stop for a human decision"
    return "DEAD", "P4 is in the closed-direction effect band"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and not args.resume_fragments:
        raise P4CVError(f"refusing to overwrite output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise P4CVError("CUDA was requested but is unavailable")
    if not args.output.exists():
        args.output.mkdir(parents=True)
    device = torch.device(args.device)
    prepare = json.loads(args.prepare_report.read_text(encoding="ascii"))
    if prepare.get("scope", {}).get("held_out_used") is not False:
        raise P4CVError("P4 fit preparation touched held-out data")
    split = sweep_support._load(args.split, "frozen split")
    score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    selected = score_bh_cv57.exact_selection(
        split,
        split_path=args.split,
        archive_video_ids=scoreall.archive_train_video_ids(args.zip),
    )
    if len(selected) != 57:
        raise P4CVError("P4 selection is not exact CV57")
    cache_paths = {path.stem.replace("__", ":", 1): path for path in args.cache.glob("*.bhc")}
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if set(cache_paths) != set(selected) or set(meta_paths) != set(selected):
        raise P4CVError("CV57 cache coverage is not exact")
    base_model, bundle = bh_model.load_v45_model(args.support_root, device=str(device))
    scorer, _record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sweep_support._sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=bh_edge,
    )
    models, dummy_logit = _load_models(args.models, device)
    settings: dict[str, dict[str, Any]] = {}
    for name in models:
        fragment_dir = args.output / name / "fragments"
        fragment_dir.mkdir(parents=True, exist_ok=args.resume_fragments)
        settings[name] = {"fragment_dir": fragment_dir, "videos": []}
    timing_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(selected, start=1) if not args.resume_fragments else ():
            dataset = identity.split(":", 1)[1]
            meta = sweep_support._load(meta_paths[identity], f"cache metadata for {identity}")
            cache = bh_model.load_sparse_cache(
                cache_paths[identity], expected_identity=sweep_support._cache_identity(meta)
            )
            scored, timing = _score_cache(cache, scorer, models, dummy_logit, device)
            timing_rows.append({"identity": identity, **timing})
            graph_nodes = run_bh.detections_to_graph_nodes(cache.detections, bh_graph.GraphNode)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            image = group["0"]
            shape = tuple(int(value) for value in image.shape)
            refiner = sweep_support.MemoizedImageRefiner(image)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    name: executor.submit(
                        _solve,
                        identity=identity,
                        dataset=dataset,
                        shape=shape,
                        graph_nodes=graph_nodes,
                        probabilities=scored[name][0],
                        no_link=scored[name][1],
                        refiner=refiner,
                        fragment_dir=settings[name]["fragment_dir"],
                    )
                    for name in models
                }
                for name in models:
                    settings[name]["videos"].append({"identity": identity, **futures[name].result()})
            print(f"P4 replay {index:02d}/57 {identity}", flush=True)
        scored_settings: dict[str, Any] = {}
        for name in models:
            manifest = scoreall.validate_fragment_set(
                scoreall.resolve_fragment_paths(str(settings[name]["fragment_dir"])), selected
            )
            rows, provenance, coverage = scoreall.score_fragment_input(store, manifest)
            by_identity = {row["identity"]: row for row in rows}
            ordered = [by_identity[identity] for identity in selected]
            summary = scoreall.official_summary(ordered)
            scored_settings[name] = {
                "summary": summary,
                "edge_counts": _edge_counts(ordered),
                "coverage": coverage,
                "input_provenance": provenance,
                "fragments": {
                    "fragment_dir": str(settings[name]["fragment_dir"]),
                    "videos": settings[name]["videos"],
                },
            }
            print(f"P4 score {name} adj={summary['adj_edge_jaccard']:.12f}", flush=True)
    attention_score = float(scored_settings["attention"]["summary"]["adj_edge_jaccard"])
    no_attention_score = float(scored_settings["no_attention"]["summary"]["adj_edge_jaccard"])
    baseline_delta = attention_score - BANKED_BASELINE
    causal_delta = attention_score - no_attention_score
    status, action = _decision(baseline_delta)
    total_pairs = sum(int(row["pair_count"]) for row in timing_rows)
    timing: dict[str, Any] = {
        "device": str(device),
        "actual_gpu_measurement": device.type == "cuda",
        "pair_count": total_pairs,
        "budget_gpu_seconds_per_pair": 3.6,
        "records": timing_rows,
        "resumed_from_complete_fragments": bool(args.resume_fragments),
    }
    if total_pairs:
        timing.update(
            {
                "base_pairwise_seconds_per_pair": sum(row["base_pairwise_seconds"] for row in timing_rows) / total_pairs,
                "no_attention_incremental_seconds_per_pair": sum(row["no_attention_seconds"] for row in timing_rows) / total_pairs,
                "attention_incremental_seconds_per_pair": sum(row["attention_seconds"] for row in timing_rows) / total_pairs,
            }
        )
        timing["no_attention_total_seconds_per_pair"] = timing["base_pairwise_seconds_per_pair"] + timing["no_attention_incremental_seconds_per_pair"]
        timing["attention_total_seconds_per_pair"] = timing["base_pairwise_seconds_per_pair"] + timing["attention_incremental_seconds_per_pair"]
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "selection": {"video_count": 57, "video_ids": list(selected)},
        "banked_baseline": {
            "adj_edge_jaccard": BANKED_BASELINE,
            **BANKED_COUNTS,
            "motion_direction": "backward",
            "motion_error_weight": 0.25,
        },
        "settings": scored_settings,
        "decision": {
            "status": status,
            "action": action,
            "primary_rule_delta_attention_minus_banked_baseline": baseline_delta,
            "causal_delta_attention_minus_identical_no_attention": causal_delta,
            "thresholds": {"proceed": 0.005, "human_decision": 0.002},
        },
        "runtime": timing,
        "execution": {
            "division_enabled": False,
            "motion_direction": "backward",
            "motion_error_weight": 0.25,
            "gnn": False,
            "kaggle_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--scorer-parity", type=Path, default=DEFAULT_PARITY)
    value.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    value.add_argument("--prepare-report", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--resume-fragments", action="store_true")
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "decision": result["decision"]}))
