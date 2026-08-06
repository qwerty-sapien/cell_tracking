#!/usr/bin/env python3
"""Build and exactly score the banked APNCC control on native-wave fit105."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_appearance_direct_policy_fit as fit  # noqa: E402


SCHEMA = "celltrack-alpha.apncc-fit105-control.v1"


class ControlError(RuntimeError):
    """The frozen APNCC fit105 control contract was violated."""


def sha256(path: Path) -> str:
    return fit.sweep._sha256(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ControlError(f"refusing to overwrite output: {args.output}")
    selection_source = json.loads(args.selection_report.read_text(encoding="ascii"))
    scope = selection_source.get("scope", {})
    selected_ids = tuple(str(value) for value in scope.get("train_video_ids", []))
    if (
        selection_source.get("schema")
        != "celltrack-alpha.selected-graph-appearance-audit.v1"
        or scope.get("fit_only") is not True
        or scope.get("audit_used") is not False
        or scope.get("held_out_used") is not False
        or len(selected_ids) != 105
        or len(set(selected_ids)) != 105
    ):
        raise ControlError("fit105 selection source is inadmissible")
    cache_manifest = json.loads(args.fit_manifest.read_text(encoding="ascii"))
    manifest_ids = tuple(str(value) for value in cache_manifest.get("identities", []))
    if set(manifest_ids) != set(selected_ids) or len(manifest_ids) != 105:
        raise ControlError("fit105 selection and cache manifest disagree")

    split = fit.sweep._load(args.split, "frozen split")
    fit.score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    cv57 = set(
        fit.score_bh_cv57.exact_selection(
            split,
            split_path=args.split,
            archive_video_ids=fit.scoreall.archive_train_video_ids(args.zip),
        )
    )
    if set(selected_ids).intersection(cv57):
        raise ControlError("fit105 control overlaps CV57")

    cache_paths = {
        path.stem.replace("__", ":", 1): path for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if not set(selected_ids).issubset(cache_paths) or not set(selected_ids).issubset(meta_paths):
        raise ControlError("local sparse-cache coverage is incomplete")

    contract, runtime_gate = fit.appearance_swap.load_contract(
        args.runtime_contract,
        expected_sha256=args.runtime_contract_sha256,
        root=ROOT,
    )
    threshold = float(contract["appearance"]["threshold"])
    minimum = int(contract["proposal_cap"]["minimum_per_video"])
    fraction = float(contract["proposal_cap"]["selected_edge_fraction"])

    base_model, bundle = fit.bh_model.load_v45_model(args.support_root, device="cpu")
    scorer, edge_record = fit.run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=fit.bh_edge,
    )
    if scorer is None or edge_record is None:
        raise ControlError("frozen edge scorer did not restore")

    args.output.mkdir(parents=True)
    fragment_dir = args.output / "fragments"
    fragment_dir.mkdir()
    videos: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(selected_ids, start=1):
            dataset = identity.split(":", 1)[1]
            meta = fit.sweep._load(meta_paths[identity], f"cache metadata for {identity}")
            cache = fit.bh_model.load_sparse_cache(
                cache_paths[identity], expected_identity=fit.sweep._cache_identity(meta)
            )
            probabilities, no_link, association = fit.run_bh.score_sparse_edge_cache(
                cache, scorer, edge_module=fit.bh_edge
            )
            graph_nodes = fit.run_bh.detections_to_graph_nodes(
                cache.detections, fit.bh_graph.GraphNode
            )
            adjacent_edges, adjacent_stats = fit.bh_graph.build_adjacent_edges(
                graph_nodes,
                learned_edge_probabilities=probabilities,
                source_no_link_probabilities=no_link,
                topology_mode="legacy",
            )
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            image = group["0"]
            refiner = fit.sweep.MemoizedImageRefiner(image)
            gap_nodes, gap_edges, gap_stats = fit.bh_graph.close_one_frame_gaps(
                graph_nodes, adjacent_edges, image_refine_callback=refiner
            )
            _nodes, baseline_edges, _post = fit.bh_graph.postprocess_graph(
                gap_nodes, gap_edges
            )
            raw_ids = {int(node.node_id) for node in graph_nodes}
            baseline_selected = {
                (int(edge.source_id), int(edge.target_id))
                for edge in baseline_edges
                if int(edge.source_id) in raw_ids and int(edge.target_id) in raw_ids
            }
            cap = max(minimum, int(math.ceil(fraction * len(baseline_selected))))
            proposals, proposal_counts = fit.appearance_swap.propose_swaps(
                cache,
                scorer,
                baseline_selected,
                runtime_gate,
                image,
                geometry_threshold=float(contract["geometry_gate"]["threshold"]),
                appearance_threshold=threshold,
                maximum=max(1, len(baseline_selected)),
            )
            proposals = [row for row in proposals if float(row["confidence"]) >= threshold][:cap]
            solved = fit._solve_direct(
                identity=identity,
                dataset=dataset,
                shape=tuple(int(value) for value in image.shape),
                graph_nodes=graph_nodes,
                gap_nodes=gap_nodes,
                gap_edges=gap_edges,
                proposals=proposals,
                probabilities=probabilities,
                fragment_dir=fragment_dir,
            )
            realized = fit._realization(Path(solved["fragment"]), proposals)
            if realized["retained_conflicts"] != 0:
                raise ControlError(f"APNCC realization retained a conflict for {identity}")
            videos.append(
                {
                    "identity": identity,
                    "association": association,
                    "adjacent_stats": adjacent_stats,
                    "gap_stats": gap_stats,
                    "proposal_generation": proposal_counts,
                    "accepted_proposals": len(proposals),
                    "realization": realized,
                }
            )
            print(
                f"APNCC fit105 {index:03d}/105 {identity} proposals={len(proposals)}",
                flush=True,
            )

        manifest = fit.scoreall.validate_fragment_set(
            fit.scoreall.resolve_fragment_paths(str(fragment_dir)), selected_ids
        )
        rows, provenance, coverage = fit.scoreall.score_fragment_input(store, manifest)

    summary = fit.scoreall.official_summary(rows)
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED_EXACT_FIT105_CONTROL",
        "selection": {
            "role": "fit-train105-control",
            "video_ids": list(selected_ids),
            "cv57_overlap": [],
        },
        "control": {
            "name": "APNCC_DIRECT_C0P25_DIVISIONS_OFF",
            "summary": summary,
            "edge_counts": fit._edge_counts(rows),
            "coverage": coverage,
            "input_provenance": provenance,
        },
        "videos": videos,
        "provenance": {
            "selection_report": str(args.selection_report),
            "selection_report_sha256": sha256(args.selection_report),
            "fit_manifest": str(args.fit_manifest),
            "fit_manifest_sha256": sha256(args.fit_manifest),
            "runtime_contract": str(args.runtime_contract),
            "runtime_contract_sha256": contract["contract_sha256"],
            "edge_artifact": edge_record,
            "scorer_parity": str(args.scorer_parity),
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
        },
    }
    fit.run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--selection-report",
        type=Path,
        default=ROOT / "reports/g02/selected_graph_appearance_audit_v1.json",
    )
    value.add_argument(
        "--fit-manifest",
        type=Path,
        default=ROOT / "kaggle_kernel/trackastra-h1-feature-cache-prep/cache_dataset_manifest.json",
    )
    value.add_argument(
        "--runtime-contract",
        type=Path,
        default=ROOT / "reports/g02/appearance_direct_swap_contract_v1.json",
    )
    value.add_argument(
        "--runtime-contract-sha256",
        default="cee547ed7a82d1bf7f6afe8cbbc3cc9d554181f9e19e87d319a5350863718cf7",
    )
    value.add_argument("--zip", type=Path, default=fit.sweep.DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=fit.sweep.DEFAULT_SPLIT)
    value.add_argument("--cache", type=Path, default=fit.DEFAULT_FIT_CACHE)
    value.add_argument("--edge-artifact", type=Path, default=fit.sweep.DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=fit.sweep.DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=fit.sweep.DEFAULT_SUPPORT)
    value.add_argument("--scorer-parity", type=Path, default=fit.sweep.DEFAULT_PARITY)
    value.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1/apncc_fit105_control",
    )
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps(report["control"]["summary"], sort_keys=True))
