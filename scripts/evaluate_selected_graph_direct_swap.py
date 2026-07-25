#!/usr/bin/env python3
"""Replay frozen appearance proposals as exact local adjacent-graph swaps."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
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
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import (  # noqa: E402
    appearance_swap,
    bh_edge,
    bh_graph,
    bh_model,
    temporal_rank_swap,
)
from evaluate_selected_graph_conflict_gate import (  # noqa: E402
    _edge_counts,
    _realization,
)


SCHEMA = "celltrack-alpha.selected-graph-direct-swap.v1"
DEFAULT_PARENT = ROOT / "reports/g02/selected_graph_appearance_screen10_v1/report.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_appearance_direct_screen10_v1"


class DirectSwapError(RuntimeError):
    """The frozen direct-swap replay contract was violated."""


def decision_reference(
    parent: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    temporal_family_gate: bool,
) -> tuple[str, float, float]:
    """Return the preregistered comparison role, score, and promotion margin."""
    if temporal_family_gate:
        confirmation = contract.get("local_confirmation")
        if not isinstance(confirmation, Mapping):
            raise DirectSwapError("runtime contract lacks banked confirmation")
        try:
            score = float(confirmation["candidate_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DirectSwapError(
                "runtime contract lacks a valid banked candidate score"
            ) from exc
        return "banked-appearance-direct-swap", score, 0.0

    baseline = parent.get("baseline")
    if not isinstance(baseline, Mapping):
        raise DirectSwapError("parent report lacks baseline")
    summary = baseline.get("summary")
    if not isinstance(summary, Mapping):
        raise DirectSwapError("parent report lacks baseline summary")
    try:
        score = float(summary["adj_edge_jaccard"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DirectSwapError("parent baseline score is invalid") from exc
    return "parent-c0p25-baseline", score, 0.005


def swap_adjacent_edges(
    nodes: Sequence[bh_graph.GraphNode],
    adjacent_edges: Sequence[bh_graph.GraphEdge],
    proposals: Sequence[Mapping[str, Any]],
    probabilities: Mapping[tuple[int, int], float],
) -> tuple[tuple[bh_graph.GraphEdge, ...], dict[str, int]]:
    """Apply node-disjoint proposals exactly once to adjacent edges."""
    try:
        return appearance_swap.swap_post_gap_edges(
            nodes, adjacent_edges, proposals, probabilities
        )
    except appearance_swap.AppearanceSwapError as exc:
        raise DirectSwapError(str(exc)) from exc


def _solve_direct(
    *,
    identity: str,
    dataset: str,
    shape: tuple[int, ...],
    graph_nodes: Sequence[bh_graph.GraphNode],
    gap_nodes: Sequence[bh_graph.GraphNode],
    gap_edges: Sequence[bh_graph.GraphEdge],
    proposals: Sequence[Mapping[str, Any]],
    probabilities: Mapping[tuple[int, int], float],
    fragment_dir: Path,
) -> dict[str, Any]:
    swapped, swap_counts = swap_adjacent_edges(
        gap_nodes, gap_edges, proposals, probabilities
    )
    final_nodes, final_edges, post_stats = bh_graph.postprocess_graph(
        gap_nodes, swapped
    )
    graph = bh_graph.GraphResult(
        nodes=final_nodes,
        edges=final_edges,
        stats={
            "input_node_count": len(graph_nodes),
            "output_node_count": len(final_nodes),
            "output_edge_count": len(final_edges),
            **swap_counts,
            **post_stats,
        },
        provenance={
            "schema": SCHEMA,
            "calibration": "frozen-selected-graph-appearance-direct-swap",
            "motion_direction": "backward",
            "motion_error_weight": 0.25,
            "division_enabled": False,
            "topology_mode": "legacy-direct-swap",
            "image_refinement": "callback",
            "deterministic": True,
            "external_access": False,
        },
    )
    bh_graph.validate_graph(graph.nodes, graph.edges)
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
    return {
        "fragment": str(destination),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "swap_counts": swap_counts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise DirectSwapError(f"refusing to overwrite output: {args.output}")
    parent = json.loads(args.parent_report.read_text(encoding="ascii"))
    selection = parent.get("selection")
    configuration = parent.get("configuration")
    if (
        parent.get("schema") != "celltrack-alpha.selected-graph-conflict-screen.v1"
        or not isinstance(selection, Mapping)
        or not isinstance(configuration, Mapping)
        or configuration.get("appearance_mode") is not True
        or configuration.get("division_enabled") is not False
        or parent.get("decision", {}).get("delta_adjusted_edge_jaccard", 0.0)
        < args.parent_minimum_delta
    ):
        raise DirectSwapError("parent appearance screen is not admissible")
    selected_ids = tuple(str(value) for value in selection["video_ids"])
    expected_count = 10 if args.selection == "top10" else 57
    if len(selected_ids) != expected_count:
        raise DirectSwapError("parent selection does not match requested replay")
    parent_by_id = {
        str(row["identity"]): row for row in parent["candidate"]["videos"]
    }
    if set(parent_by_id) != set(selected_ids):
        raise DirectSwapError("parent video/proposal coverage is incomplete")

    split = sweep._load(args.split, "frozen split")
    score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    cv57 = score_bh_cv57.exact_selection(
        split,
        split_path=args.split,
        archive_video_ids=scoreall.archive_train_video_ids(args.zip),
    )
    if not set(selected_ids).issubset(cv57):
        raise DirectSwapError("parent selection escaped exact CV57")
    baseline_names = {
        identity.replace(":", "__", 1) + ".csv" for identity in selected_ids
    }
    baseline_manifest = scoreall.validate_fragment_set(
        [
            path
            for path in scoreall.resolve_fragment_paths(
                str(args.baseline_fragments)
            )
            if path.name in baseline_names
        ],
        selected_ids,
    )
    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if not set(selected_ids).issubset(cache_paths) or not set(
        selected_ids
    ).issubset(meta_paths):
        raise DirectSwapError("cache coverage is incomplete")
    base_model, bundle = bh_model.load_v45_model(
        args.support_root, device="cpu"
    )
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sweep._sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None:
        raise DirectSwapError("frozen edge scorer did not restore")
    contract, runtime_gate = appearance_swap.load_contract(
        args.runtime_contract,
        expected_sha256=args.runtime_contract_sha256,
        root=ROOT,
    )
    temporal_runtime = None
    if args.temporal_family_gate is not None:
        if (
            args.temporal_family_gate_sha256 is None
            or args.selection != "full57"
        ):
            raise DirectSwapError(
                "temporal family gate requires SHA-256 and full57 selection"
            )
        temporal_runtime = temporal_rank_swap.load_gate(
            args.temporal_family_gate,
            expected_sha256=args.temporal_family_gate_sha256,
            expected_family="6bba",
        )

    args.output.mkdir(parents=True)
    fragments = args.output / "fragments"
    fragments.mkdir()
    videos: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(selected_ids, start=1):
            dataset = identity.split(":", 1)[1]
            meta = sweep._load(
                meta_paths[identity], f"cache metadata for {identity}"
            )
            cache = bh_model.load_sparse_cache(
                cache_paths[identity],
                expected_identity=sweep._cache_identity(meta),
            )
            probabilities, no_link, association = run_bh.score_sparse_edge_cache(
                cache, scorer, edge_module=bh_edge
            )
            graph_nodes = run_bh.detections_to_graph_nodes(
                cache.detections, bh_graph.GraphNode
            )
            adjacent_edges, adjacent_stats = bh_graph.build_adjacent_edges(
                graph_nodes,
                learned_edge_probabilities=probabilities,
                source_no_link_probabilities=no_link,
                topology_mode="legacy",
            )
            parent_video = parent_by_id[identity]
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            image = group["0"]
            refiner = sweep.MemoizedImageRefiner(image)
            gap_nodes, gap_edges, gap_stats = bh_graph.close_one_frame_gaps(
                graph_nodes,
                adjacent_edges,
                image_refine_callback=refiner,
            )
            _baseline_nodes, baseline_edges, _baseline_post = (
                bh_graph.postprocess_graph(gap_nodes, gap_edges)
            )
            raw_ids = {int(node.node_id) for node in graph_nodes}
            baseline_selected = {
                (int(edge.source_id), int(edge.target_id))
                for edge in baseline_edges
                if int(edge.source_id) in raw_ids
                and int(edge.target_id) in raw_ids
            }
            expected_selected = int(
                parent_video["baseline_graph"]["raw_selected_edges"]
            )
            parent_conflicts = {
                tuple(int(value) for value in pair)
                for proposal in parent_video["proposals"]
                for pair in proposal["conflicts"]
            }
            if (
                len(baseline_selected) != expected_selected
                or not parent_conflicts.issubset(baseline_selected)
            ):
                raise DirectSwapError(
                    f"post-gap selected identity drifted for {identity}"
                )
            cap_spec = contract["proposal_cap"]
            proposal_cap = max(
                int(cap_spec["minimum_per_video"]),
                int(
                    math.ceil(
                        float(cap_spec["selected_edge_fraction"])
                        * len(baseline_selected)
                    )
                ),
            )
            use_temporal = (
                temporal_runtime is not None and dataset.startswith("6bba_")
            )
            if use_temporal:
                temporal_gate, temporal_mean, temporal_scale, temporal_threshold = (
                    temporal_runtime
                )
                runtime_proposals, runtime_counts = (
                    temporal_rank_swap.propose_swaps(
                        cache,
                        scorer,
                        baseline_selected,
                        runtime_gate,
                        temporal_gate,
                        temporal_mean,
                        temporal_scale,
                        image,
                        geometry_threshold=float(
                            contract["geometry_gate"]["threshold"]
                        ),
                        temporal_threshold=temporal_threshold,
                        maximum=proposal_cap,
                    )
                )
            else:
                runtime_proposals, runtime_counts = appearance_swap.propose_swaps(
                    cache,
                    scorer,
                    baseline_selected,
                    runtime_gate,
                    image,
                    geometry_threshold=float(
                        contract["geometry_gate"]["threshold"]
                    ),
                    appearance_threshold=float(
                        contract["appearance"]["threshold"]
                    ),
                    maximum=proposal_cap,
                )
            parent_proposals = parent_video["proposals"]
            proposal_parity = not use_temporal and json.dumps(
                    runtime_proposals,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ) == json.dumps(
                    parent_proposals,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            if not use_temporal and not proposal_parity:
                raise DirectSwapError(
                    f"packaged runtime proposal identity drifted for {identity}"
                )
            solved = _solve_direct(
                identity=identity,
                dataset=dataset,
                shape=tuple(int(value) for value in image.shape),
                graph_nodes=graph_nodes,
                gap_nodes=gap_nodes,
                gap_edges=gap_edges,
                proposals=runtime_proposals,
                probabilities=probabilities,
                fragment_dir=fragments,
            )
            realization = _realization(
                Path(solved["fragment"]), runtime_proposals
            )
            if realization["retained_conflicts"] != 0:
                raise DirectSwapError(
                    f"direct swap realization failed for {identity}"
                )
            videos.append(
                {
                    "identity": identity,
                    "association": association,
                    "adjacent_stats": adjacent_stats,
                    "gap_stats": gap_stats,
                    "proposal_count": len(runtime_proposals),
                    "runtime_proposal_counts": runtime_counts,
                    "runtime_proposal_parity": proposal_parity,
                    "proposal_model": (
                        "temporal-family-rank-6bba"
                        if use_temporal
                        else "banked-appearance-ncc"
                    ),
                    "realization": realization,
                    **solved,
                }
            )
            for name, value in {
                **solved["swap_counts"],
                **realization,
            }.items():
                if isinstance(value, (int, np.integer)) and not isinstance(
                    value, bool
                ):
                    totals[name] = totals.get(name, 0) + int(value)
            print(
                f"direct swap {index:02d}/{len(selected_ids)} {identity} "
                f"proposals={len(runtime_proposals)} "
                f"parity={'exact' if proposal_parity else 'intentional-replacement'}",
                flush=True,
            )

        candidate_manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(fragments)), selected_ids
        )
        baseline_rows, baseline_provenance, baseline_coverage = (
            scoreall.score_fragment_input(store, baseline_manifest)
        )
        candidate_rows, candidate_provenance, candidate_coverage = (
            scoreall.score_fragment_input(store, candidate_manifest)
        )
    baseline_by_id = {row["identity"]: row for row in baseline_rows}
    candidate_by_id = {row["identity"]: row for row in candidate_rows}
    ordered_baseline = [baseline_by_id[value] for value in selected_ids]
    ordered_candidate = [candidate_by_id[value] for value in selected_ids]
    baseline_summary = scoreall.official_summary(ordered_baseline)
    candidate_summary = scoreall.official_summary(ordered_candidate)
    reference_role, reference_score, promotion_margin = decision_reference(
        parent,
        contract,
        temporal_family_gate=temporal_runtime is not None,
    )
    delta = float(candidate_summary["adj_edge_jaccard"]) - reference_score
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "selection": {
            "role": (
                "preregistered-worst-residual-top10"
                if args.selection == "top10"
                else "authorized-full-cv57-confirm"
            ),
            "video_count": len(selected_ids),
            "video_ids": list(selected_ids),
        },
        "mechanism": {
            "proposal_source": (
                "frozen APNCC for 44b6 plus temporal family ranker for 6bba"
                if temporal_runtime is not None
                else "frozen geometry-prefiltered full-3D NCC parent"
            ),
            "integration": (
                "exact node-disjoint swaps after unchanged gap closure and "
                "before unchanged pruning/smoothing"
            ),
            "global_probability_resolve": False,
            "division_enabled": False,
            "gnn": False,
        },
        "baseline": {
            "summary": baseline_summary,
            "edge_counts": _edge_counts(ordered_baseline),
            "coverage": baseline_coverage,
            "input_provenance": baseline_provenance,
        },
        "candidate": {
            "summary": candidate_summary,
            "edge_counts": _edge_counts(ordered_candidate),
            "coverage": candidate_coverage,
            "input_provenance": candidate_provenance,
            "totals": totals,
            "videos": videos,
        },
        "decision": {
            "delta_adjusted_edge_jaccard": delta,
            "reference_role": reference_role,
            "reference_score": reference_score,
            "promotion_margin": promotion_margin,
            "status": (
                "BANK"
                if temporal_runtime is not None and delta > promotion_margin
                else "CONFIRM"
                if temporal_runtime is None and delta >= promotion_margin
                else "PARK"
            ),
            "full_cv57_authorized": (
                temporal_runtime is None
                and args.selection == "top10"
                and delta >= promotion_margin
            ),
        },
        "provenance": {
            "parent_report": str(args.parent_report),
            "parent_report_sha256": sweep._sha256(args.parent_report),
            "runtime_contract": str(args.runtime_contract),
            "runtime_contract_sha256": contract["contract_sha256"],
            "runtime_proposal_parity": (
                "exact-44b6; intentional-temporal-replacement-6bba"
                if temporal_runtime is not None
                else "exact-all-videos"
            ),
            "temporal_family_gate": (
                str(args.temporal_family_gate)
                if args.temporal_family_gate is not None
                else None
            ),
            "temporal_family_gate_sha256": (
                args.temporal_family_gate_sha256
                if args.temporal_family_gate is not None
                else None
            ),
            "edge_artifact": edge_record,
            "scorer_parity": str(args.scorer_parity),
        },
        "execution": {
            "device": "cpu",
            "kaggle_action": False,
            "gpu_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--parent-report", type=Path, default=DEFAULT_PARENT)
    value.add_argument(
        "--runtime-contract",
        type=Path,
        default=ROOT / "reports/g02/appearance_direct_swap_contract_v1.json",
    )
    value.add_argument(
        "--runtime-contract-sha256",
        default=(
            "cee547ed7a82d1bf7f6afe8cbbc3cc9d554181f9e19e87d319a5350863718cf7"
        ),
    )
    value.add_argument(
        "--parent-minimum-delta", type=float, default=0.005
    )
    value.add_argument("--temporal-family-gate", type=Path)
    value.add_argument("--temporal-family-gate-sha256")
    value.add_argument(
        "--selection", choices=("top10", "full57"), default="top10"
    )
    value.add_argument("--zip", type=Path, default=sweep.DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=sweep.DEFAULT_SPLIT)
    value.add_argument("--cache", type=Path, default=sweep.DEFAULT_CACHE)
    value.add_argument(
        "--baseline-fragments", type=Path, default=sweep.DEFAULT_BASELINE
    )
    value.add_argument("--edge-artifact", type=Path, default=sweep.DEFAULT_EDGE)
    value.add_argument(
        "--edge-auxiliary", type=Path, default=sweep.DEFAULT_AUXILIARY
    )
    value.add_argument("--support-root", type=Path, default=sweep.DEFAULT_SUPPORT)
    value.add_argument("--scorer-parity", type=Path, default=sweep.DEFAULT_PARITY)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(
        json.dumps(
            {"status": report["status"], "decision": report["decision"]},
            sort_keys=True,
        )
    )
