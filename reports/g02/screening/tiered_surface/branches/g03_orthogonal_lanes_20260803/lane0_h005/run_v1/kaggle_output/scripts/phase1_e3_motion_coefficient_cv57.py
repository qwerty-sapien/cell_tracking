#!/usr/bin/env python3
"""Run the CPU-only E3 motion-error coefficient sweep on exact CV57 caches."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

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


SCHEMA = "celltrack-alpha.phase1-e3-motion-coefficient-cv57.v1"
EXPECTED_BASELINE = {
    "adj_edge_jaccard": 0.8711885823490648,
    "edge_jaccard": 0.8761665470208184,
    "edge_tp": 34174,
    "edge_fp": 2736,
    "edge_fn": 2094,
    "division_tp": 0.0,
    "division_fp": 0.0,
    "division_fn": 42.0,
}
SHARP_ADJ_MARGIN = 0.001

DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_CACHE = ROOT / "kaggle_kernel/bhcv57out_v3/cache"
DEFAULT_BASELINE = ROOT / "kaggle_kernel/bhcv57out_v3/fragments"
DEFAULT_EDGE = (
    ROOT / "_ARCHIVE/conflict_residual/reports/g02/conflict_edge_f0_fit117_v1/edge.json"
)
DEFAULT_AUXILIARY = ROOT / "_ARCHIVE/compact_head/reports/g02/v45f4.pt"
DEFAULT_SUPPORT = ROOT / "external/Tracking_Support"
DEFAULT_PARITY = ROOT / "reports/g02/scorer_parity_v1.json"
DEFAULT_BASELINE_SCORES = (
    ROOT
    / "_ARCHIVE/conflict_residual/reports/g02/"
    "run_bh_d4_conflict_edge_f0_holdout57_v1/scores_postpatch.json"
)
DEFAULT_OUTPUT = ROOT / "reports/g02/phase1_e3_motion_coefficient_cv57_v1"


class E3Error(RuntimeError):
    """The exact offline E3 contract was not satisfied."""


def _setting_name(weight: float) -> str:
    return f"motion_error_weight_{format(weight, 'g').replace('.', 'p')}"


def _validate_weights(values: Sequence[float]) -> tuple[float, ...]:
    weights = tuple(float(value) for value in values)
    if (
        not weights
        or len(set(weights)) != len(weights)
        or any(not math.isfinite(value) or value < 0.0 for value in weights)
        or set(weights) != {0.0, 0.25, 0.5, 1.0, 2.0}
    ):
        raise E3Error("E3 weights must be exactly {0, 0.25, 0.5, 1, 2}")
    return weights


def _edge_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        name: int(sum(float(row[name]) for row in rows))
        for name in ("edge_tp", "edge_fp", "edge_fn")
    }


def _validate_baseline_scores(payload: Mapping[str, Any]) -> None:
    summary = payload.get("summary")
    samples = payload.get("samples")
    execution = payload.get("execution")
    if not isinstance(summary, Mapping) or not isinstance(samples, list):
        raise E3Error("baseline score report has the wrong schema")
    if not isinstance(execution, Mapping) or execution.get("division_enabled") is not False:
        raise E3Error("baseline score report is not divisions-off")
    observed = {
        "adj_edge_jaccard": float(summary["adj_edge_jaccard"]),
        "edge_jaccard": float(summary["edge_jaccard"]),
        **_edge_counts(samples),
        "division_tp": float(summary["division_tp"]),
        "division_fp": float(summary["division_fp"]),
        "division_fn": float(summary["division_fn"]),
    }
    if observed != EXPECTED_BASELINE:
        raise E3Error(f"exact CV57 baseline changed: {observed}")


def _solve_setting(
    *,
    weight: float,
    experiment_id: str,
    identity: str,
    dataset: str,
    shape: tuple[int, ...],
    graph_nodes: Sequence[Any],
    probabilities: Mapping[Any, Any],
    no_link: Mapping[Any, Any],
    refiner: sweep_support.MemoizedImageRefiner,
    fragment_dir: Path,
    baseline_fragment: Path,
) -> dict[str, Any]:
    config = sweep_support.SweepGraphConfig(
        learned_bonus=bh_graph.V45_GRAPH_CONFIG.learned_bonus,
        raw_distance_weight=bh_graph.V45_GRAPH_CONFIG.raw_distance_weight,
    )
    sweep_provenance = {
        "schema": bh_graph.EXPERIMENTAL_SWEEP_SCHEMA,
        "experiment_id": experiment_id,
        "purpose": "offline-cached-cv57-sweep",
        "varied_fields": ["motion_error_weight"],
        "production_eligible": False,
        "submit_eligible": False,
    }
    graph_result = bh_graph.calibrate_v45_graph(
        graph_nodes,
        learned_edge_probabilities=probabilities,
        source_no_link_probabilities=no_link,
        image_refine_callback=refiner,
        topology_mode="legacy",
        division_enabled=False,
        config=config,  # type: ignore[arg-type]
        _experimental_sweep_provenance=sweep_provenance,
        _experimental_motion_error_weight=weight,
    )
    if (
        graph_result.provenance.get("production_eligible") is not False
        or graph_result.provenance.get("experimental_sweep") != sweep_provenance
        or graph_result.provenance.get("division_enabled") is not False
        or graph_result.provenance.get("motion_error_weight") != weight
        or graph_result.stats.get("motion_error_weight") != weight
    ):
        raise E3Error("graph result lost offline-only/motion-weight provenance")
    graph_result = run_bh.project_graph_result_to_image_support(
        dataset,
        graph_result,
        shape,
        scale_zyx=bh_graph.V45_GRAPH_CONFIG.scale_zyx,
        max_correction_um=bh_graph.V45_GRAPH_CONFIG.refine_max_shift_um,
    )
    rows = run_bh.graph_rows(dataset, graph_result, shape)
    fragment = fragment_dir / (identity.replace(":", "__") + ".csv")
    run_bh.write_rows_atomic(fragment, rows, deadline_epoch=None)
    parity = None
    if weight == 1.0:
        parity = sweep_support._fragment_parity(fragment, baseline_fragment)
        if (
            parity["node_identity_time_exact"] is not True
            or parity["edge_topology_exact"] is not True
        ):
            raise E3Error(f"motion weight 1 topology differs for {identity}")
    return {
        "fragment": fragment,
        "fragment_sha256": sweep_support._sha256(fragment),
        "nodes": len(graph_result.nodes),
        "edges": len(graph_result.edges),
        "prediction_component_count": sweep_support._prediction_component_count(
            graph_result
        ),
        "baseline_parity": parity,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    weights = _validate_weights(args.motion_error_weight)
    if args.output.exists():
        raise E3Error(f"refusing to overwrite E3 output: {args.output}")
    args.output.mkdir(parents=True)

    split = sweep_support._load(args.split, "frozen split")
    baseline_scores = sweep_support._load(args.baseline_scores, "baseline scores")
    _validate_baseline_scores(baseline_scores)
    scorer_parity = score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    archive_ids = scoreall.archive_train_video_ids(args.zip)
    selected = score_bh_cv57.exact_selection(
        split, split_path=args.split, archive_video_ids=archive_ids
    )
    baseline_manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(args.baseline_fragments)), selected
    )
    baseline_paths = dict(baseline_manifest.ordered)
    cache_paths = {
        path.stem.replace("__", ":", 1): path for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if set(cache_paths) != set(selected) or set(meta_paths) != set(selected):
        raise E3Error("persisted cache/meta files do not exactly cover CV57")

    model, bundle = bh_model.load_v45_model(args.support_root, device="cpu")
    edge_sha256 = sweep_support._sha256(args.edge_artifact)
    edge_scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        edge_sha256,
        args.edge_auxiliary,
        model,
        bundle,
        edge_module=bh_edge,
    )
    if (
        edge_scorer is None
        or edge_record is None
        or edge_record.get("division_enabled") is not False
    ):
        raise E3Error("released edge scorer was not restored divisions-off")

    settings: dict[float, dict[str, Any]] = {}
    for weight in weights:
        name = _setting_name(weight)
        fragment_dir = args.output / name / "fragments"
        fragment_dir.mkdir(parents=True)
        settings[weight] = {
            "name": name,
            "fragment_dir": fragment_dir,
            "videos": [],
            "prediction_component_count": 0,
            "baseline_parity": [],
        }

    shared_refinement_frame_reads = 0
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(selected, start=1):
            _, dataset, _ = scoreall.parse_identity(identity)
            meta = sweep_support._load(
                meta_paths[identity], f"cache metadata for {identity}"
            )
            identity_record = sweep_support._cache_identity(meta)
            cache = bh_model.load_sparse_cache(
                cache_paths[identity], expected_identity=identity_record
            )
            probabilities, no_link, association = run_bh.score_sparse_edge_cache(
                cache, edge_scorer, edge_module=bh_edge
            )
            graph_nodes = run_bh.detections_to_graph_nodes(
                cache.detections, bh_graph.GraphNode
            )
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            image = group["0"]
            shape = tuple(int(value) for value in image.shape)
            if shape != (100, 64, 256, 256):
                raise E3Error(f"unexpected image shape for {identity}: {shape}")
            shared_refiner = sweep_support.MemoizedImageRefiner(image)
            with ThreadPoolExecutor(max_workers=len(weights)) as executor:
                futures = {
                    weight: executor.submit(
                        _solve_setting,
                        weight=weight,
                        experiment_id=args.output.name,
                        identity=identity,
                        dataset=dataset,
                        shape=shape,
                        graph_nodes=graph_nodes,
                        probabilities=probabilities,
                        no_link=no_link,
                        refiner=shared_refiner,
                        fragment_dir=settings[weight]["fragment_dir"],
                        baseline_fragment=baseline_paths[identity],
                    )
                    for weight in weights
                }
                solved = {weight: futures[weight].result() for weight in weights}
            refinement = shared_refiner.report()
            for weight in weights:
                current = solved[weight]
                setting = settings[weight]
                if current["baseline_parity"] is not None:
                    setting["baseline_parity"].append(
                        {"identity": identity, **current["baseline_parity"]}
                    )
                components = int(current["prediction_component_count"])
                setting["prediction_component_count"] += components
                setting["videos"].append(
                    {
                        "identity": identity,
                        "fragment": str(current["fragment"]),
                        "fragment_sha256": current["fragment_sha256"],
                        "nodes": current["nodes"],
                        "edges": current["edges"],
                        "prediction_component_count": components,
                        "association_candidate_count": association["candidate_count"],
                        "image_refinement": {
                            **refinement,
                            "shared_across_settings": True,
                        },
                    }
                )
            shared_refinement_frame_reads += int(refinement["frame_reads"])
            print(f"E3 replay {index:02d}/{len(selected)} {identity}", flush=True)

        scored_settings: list[dict[str, Any]] = []
        for weight in weights:
            setting = settings[weight]
            manifest = scoreall.validate_fragment_set(
                scoreall.resolve_fragment_paths(str(setting["fragment_dir"])),
                selected,
            )
            rows, input_provenance, coverage = scoreall.score_fragment_input(
                store, manifest
            )
            by_identity = {row["identity"]: row for row in rows}
            if set(by_identity) != set(selected):
                raise E3Error("patched official scoring escaped exact CV57")
            rows = [by_identity[identity] for identity in selected]
            summary = scoreall.official_summary(rows)
            scored_settings.append(
                {
                    "motion_error_weight": weight,
                    "name": setting["name"],
                    "summary": summary,
                    "edge_counts": _edge_counts(rows),
                    "prediction_component_count": setting[
                        "prediction_component_count"
                    ],
                    "fragments": {
                        "path": str(setting["fragment_dir"]),
                        "count": len(setting["videos"]),
                        "records": setting["videos"],
                    },
                    "coverage": coverage,
                    "input_provenance": input_provenance,
                    "baseline_fragment_parity": (
                        {
                            "topology_exact_count": sum(
                                item["edge_topology_exact"]
                                and item["node_identity_time_exact"]
                                for item in setting["baseline_parity"]
                            ),
                            "records": setting["baseline_parity"],
                        }
                        if setting["baseline_parity"]
                        else None
                    ),
                }
            )
            print(
                f"E3 score c={weight:g} adj={summary['adj_edge_jaccard']:.12f}",
                flush=True,
            )

    control = next(
        setting for setting in scored_settings if setting["motion_error_weight"] == 1.0
    )
    zero = next(
        setting for setting in scored_settings if setting["motion_error_weight"] == 0.0
    )
    if {
        "adj_edge_jaccard": control["summary"]["adj_edge_jaccard"],
        "edge_jaccard": control["summary"]["edge_jaccard"],
        **control["edge_counts"],
        "division_tp": control["summary"]["division_tp"],
        "division_fp": control["summary"]["division_fp"],
        "division_fn": control["summary"]["division_fn"],
    } != EXPECTED_BASELINE:
        raise E3Error("motion weight 1 control did not reproduce the exact baseline")
    ranked = sorted(
        scored_settings,
        key=lambda setting: (
            -float(setting["summary"]["adj_edge_jaccard"]),
            float(setting["motion_error_weight"]),
        ),
    )
    best = ranked[0]
    best_margin = float(best["summary"]["adj_edge_jaccard"]) - float(
        ranked[1]["summary"]["adj_edge_jaccard"]
    )
    harmful = float(zero["summary"]["adj_edge_jaccard"]) >= float(
        control["summary"]["adj_edge_jaccard"]
    )
    sharp_positive = (
        float(best["motion_error_weight"]) > 0.0
        and best_margin >= SHARP_ADJ_MARGIN
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "selection": {"video_count": len(selected), "video_ids": list(selected)},
        "execution": {
            "device": "cpu",
            "detector_inference": False,
            "detector_forward_calls": 0,
            "training": False,
            "graph_solve_only": True,
            "cache_inputs": ["FrameDet", "PairContext", "PairEdges"],
            "division_enabled": False,
            "geometry_changed": False,
            "production_config_changed": False,
            "production_eligible": False,
            "submit_eligible": False,
            "internet_access": False,
            "shared_image_refinement_frame_reads": shared_refinement_frame_reads,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "decision": {
            "harmful_rule": {
                "rule": "PASS iff adjusted(c=0.0) >= adjusted(c=1.0)",
                "status": "PASS" if harmful else "FAIL",
                "delta_c0_minus_c1": float(
                    zero["summary"]["adj_edge_jaccard"]
                )
                - float(control["summary"]["adj_edge_jaccard"]),
                "interpretation": (
                    "constant-velocity prior is actively harmful; deletion is supported"
                    if harmful
                    else "deleting constant-velocity motion is not supported"
                ),
            },
            "load_bearing_rule": {
                "rule": (
                    "PASS iff the unique best tested c*>0 exceeds the runner-up by "
                    f">={SHARP_ADJ_MARGIN:g} adjusted Jaccard"
                ),
                "status": "PASS" if sharp_positive else "FAIL",
                "best_weight": best["motion_error_weight"],
                "best_runner_up_margin": best_margin,
                "sharpness_operationalization": SHARP_ADJ_MARGIN,
                "interpretation": (
                    "motion is load-bearing and a learned motion model has headroom"
                    if sharp_positive
                    else "the sampled curve does not show a sharp positive motion peak"
                ),
            },
        },
        "provenance": {
            "cache": str(args.cache),
            "cache_file_count": len(cache_paths),
            "baseline_fragments": str(args.baseline_fragments),
            "baseline_scores": {
                "path": str(args.baseline_scores),
                "sha256": sweep_support._sha256(args.baseline_scores),
            },
            "edge_artifact": {"path": str(args.edge_artifact), "sha256": edge_sha256},
            "edge_auxiliary": {
                "path": str(args.edge_auxiliary),
                "sha256": sweep_support._sha256(args.edge_auxiliary),
            },
            "scorer_parity": scorer_parity,
            "graph_source": {
                "path": str(ROOT / "src/celltrack_alpha/bh_graph.py"),
                "sha256": sweep_support._sha256(
                    ROOT / "src/celltrack_alpha/bh_graph.py"
                ),
            },
            "script_source": {
                "path": str(Path(__file__)),
                "sha256": sweep_support._sha256(Path(__file__)),
            },
        },
        "settings": scored_settings,
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--baseline-fragments", type=Path, default=DEFAULT_BASELINE)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--scorer-parity", type=Path, default=DEFAULT_PARITY)
    value.add_argument("--baseline-scores", type=Path, default=DEFAULT_BASELINE_SCORES)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument(
        "--motion-error-weight",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 2.0],
    )
    return value


def main() -> None:
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "decision": result["decision"]}))


if __name__ == "__main__":
    main()
