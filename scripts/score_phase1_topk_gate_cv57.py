#!/usr/bin/env python3
"""Replay and officially score the E1-mandated top-k gate on exact CV57.

The detector, learned edge scorer, graph costs, postprocessing, and divisions-off
setting are frozen to the released baseline.  Only the adjacent candidate gate
changes, from a 10 um radius query to the ten nearest next-frame detections.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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


SCHEMA = "celltrack-alpha.phase1-e1-topk-gate-cv57.v1"
TOP_K = 10
EXPECTED_BASELINE = {
    "adj_edge_jaccard": 0.8711885823490648,
    "edge_jaccard": 0.8761665470208184,
    "edge_tp": 34_174,
    "edge_fp": 2_736,
    "edge_fn": 2_094,
    "division_tp": 0.0,
    "division_fp": 0.0,
    "division_fn": 42.0,
}

DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_CACHE = ROOT / "kaggle_kernel/bhcv57out_v3/cache"
DEFAULT_BASELINE = ROOT / "kaggle_kernel/bhcv57out_v3/fragments"
DEFAULT_EDGE = (
    ROOT / "_ARCHIVE/conflict_residual/reports/g02/conflict_edge_f0_fit117_v1/edge.json"
)
DEFAULT_AUXILIARY = ROOT / "_ARCHIVE/compact_head/reports/g02/v45f4.pt"
DEFAULT_SUPPORT = ROOT / "external/Tracking_Support"
DEFAULT_PARITY = score_bh_cv57.DEFAULT_SCORER_PARITY
DEFAULT_BASELINE_SCORES = (
    ROOT
    / "_ARCHIVE/conflict_residual/reports/g02/"
    "run_bh_d4_conflict_edge_f0_holdout57_v1/scores_postpatch.json"
)
DEFAULT_OUTPUT = ROOT / "reports/g02/phase1_e1_topk_gate_cv57_v1"


class TopKGateError(RuntimeError):
    """The exact offline top-k replay contract was not satisfied."""


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
        raise TopKGateError("baseline score report has the wrong schema")
    if not isinstance(execution, Mapping) or execution.get("division_enabled") is not False:
        raise TopKGateError("baseline score report is not divisions-off")
    observed = {
        "adj_edge_jaccard": float(summary["adj_edge_jaccard"]),
        "edge_jaccard": float(summary["edge_jaccard"]),
        **_edge_counts(samples),
        "division_tp": float(summary["division_tp"]),
        "division_fp": float(summary["division_fp"]),
        "division_fn": float(summary["division_fn"]),
    }
    if observed != EXPECTED_BASELINE:
        raise TopKGateError(f"exact CV57 baseline changed: {observed}")


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise TopKGateError(f"refusing to overwrite report: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_markdown(payload: Mapping[str, Any]) -> str:
    baseline = payload["baseline"]
    topk = payload["topk"]
    delta = payload["delta"]
    return "\n".join(
        (
            "# Phase 1 E1 standalone top-k gate, exact CV57",
            "",
            "## Frozen baseline",
            "",
            f"- Adjusted edge Jaccard: `{baseline['adj_edge_jaccard']:.12f}`",
            f"- Edge TP / FP / FN: `{baseline['edge_tp']} / {baseline['edge_fp']} / {baseline['edge_fn']}`",
            "- Divisions: OFF",
            "",
            "## Top-10 gate",
            "",
            f"- Adjusted edge Jaccard: `{topk['adj_edge_jaccard']:.12f}`",
            f"- Edge TP / FP / FN: `{topk['edge_tp']} / {topk['edge_fp']} / {topk['edge_fn']}`",
            f"- Delta adjusted edge Jaccard: `{delta['adj_edge_jaccard']:+.12f}`",
            f"- Delta edge TP / FP / FN: `{delta['edge_tp']:+d} / {delta['edge_fp']:+d} / {delta['edge_fn']:+d}`",
            "",
            "## Boundary",
            "",
            "This is a CPU-only replay from persisted CV57 detector/edge caches. The only graph change is the adjacent candidate gate: ten nearest next-frame detections replace the 10 um metric-radius query. Divisions remain off; no training, Kaggle action, or submission occurred.",
            "",
        )
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.top_k != TOP_K:
        raise TopKGateError(f"E1 decision rule requires top_k={TOP_K}")
    if args.output.exists():
        raise TopKGateError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    fragment_dir = args.output / "fragments"
    fragment_dir.mkdir()

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
        raise TopKGateError("persisted cache/meta files do not exactly cover CV57")

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
        raise TopKGateError("released edge scorer was not restored divisions-off")

    experiment = {
        "schema": bh_graph.EXPERIMENTAL_SWEEP_SCHEMA,
        "experiment_id": args.output.name,
        "purpose": "offline-cached-cv57-sweep",
        "varied_fields": ["candidate_top_k"],
        "production_eligible": False,
        "submit_eligible": False,
    }
    videos: list[dict[str, Any]] = []
    candidate_pairs = 0
    radius_candidate_pairs = 0
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(selected, start=1):
            _, dataset, _ = scoreall.parse_identity(identity)
            meta = sweep_support._load(
                meta_paths[identity], f"cache metadata for {identity}"
            )
            cache = bh_model.load_sparse_cache(
                cache_paths[identity],
                expected_identity=sweep_support._cache_identity(meta),
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
                raise TopKGateError(f"unexpected image shape for {identity}: {shape}")
            refiner = sweep_support.MemoizedImageRefiner(image)
            graph_result = bh_graph.calibrate_v45_graph(
                graph_nodes,
                learned_edge_probabilities=probabilities,
                source_no_link_probabilities=no_link,
                image_refine_callback=refiner,
                topology_mode="legacy",
                division_enabled=False,
                _experimental_sweep_provenance=experiment,
                _experimental_candidate_top_k=args.top_k,
            )
            if (
                graph_result.provenance.get("production_eligible") is not False
                or graph_result.provenance.get("experimental_sweep") != experiment
                or graph_result.provenance.get("division_enabled") is not False
                or graph_result.provenance.get("candidate_top_k") != args.top_k
                or graph_result.stats.get("candidate_top_k") != args.top_k
            ):
                raise TopKGateError("graph result lost top-k/offline-only provenance")
            graph_result = run_bh.project_graph_result_to_image_support(
                dataset,
                graph_result,
                shape,
                scale_zyx=bh_graph.V45_GRAPH_CONFIG.scale_zyx,
                max_correction_um=bh_graph.V45_GRAPH_CONFIG.refine_max_shift_um,
            )
            fragment = fragment_dir / (identity.replace(":", "__") + ".csv")
            run_bh.write_rows_atomic(
                fragment,
                run_bh.graph_rows(dataset, graph_result, shape),
                deadline_epoch=None,
            )
            baseline_parity = sweep_support._fragment_parity(
                fragment, baseline_paths[identity]
            )
            current_candidate_pairs = int(
                graph_result.stats["adjacent_candidate_pairs"]
            )
            current_radius_pairs = int(
                graph_result.stats["radius_candidate_pairs_diagnostic"]
            )
            candidate_pairs += current_candidate_pairs
            radius_candidate_pairs += current_radius_pairs
            videos.append(
                {
                    "identity": identity,
                    "fragment": str(fragment),
                    "fragment_sha256": sweep_support._sha256(fragment),
                    "nodes": len(graph_result.nodes),
                    "edges": len(graph_result.edges),
                    "adjacent_candidate_pairs": current_candidate_pairs,
                    "radius_candidate_pairs_diagnostic": current_radius_pairs,
                    "association_candidate_count": association["candidate_count"],
                    "baseline_parity": baseline_parity,
                    "image_refinement": refiner.report(),
                }
            )
            print(f"E1 top-k replay {index:02d}/{len(selected)} {identity}", flush=True)

        manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(fragment_dir)), selected
        )
        rows, input_provenance, coverage = scoreall.score_fragment_input(store, manifest)
    by_identity = {row["identity"]: row for row in rows}
    if set(by_identity) != set(selected):
        raise TopKGateError("patched official scoring escaped exact CV57")
    rows = [by_identity[identity] for identity in selected]
    summary = scoreall.official_summary(rows)
    counts = _edge_counts(rows)
    topk = {
        "adj_edge_jaccard": float(summary["adj_edge_jaccard"]),
        "edge_jaccard": float(summary["edge_jaccard"]),
        **counts,
        "division_tp": float(summary["division_tp"]),
        "division_fp": float(summary["division_fp"]),
        "division_fn": float(summary["division_fn"]),
    }
    if (topk["division_tp"], topk["division_fp"], topk["division_fn"]) != (
        0.0,
        0.0,
        42.0,
    ):
        raise TopKGateError("top-k replay did not keep divisions off")
    delta = {
        key: topk[key] - EXPECTED_BASELINE[key]
        for key in ("adj_edge_jaccard", "edge_jaccard", "edge_tp", "edge_fp", "edge_fn")
    }
    payload = {
        "schema": SCHEMA,
        "status": "completed",
        "baseline": EXPECTED_BASELINE,
        "topk": topk,
        "delta": delta,
        "selection": {"video_count": len(selected), "video_ids": list(selected)},
        "execution": {
            "device": "cpu",
            "detector_inference": False,
            "training": False,
            "division_enabled": False,
            "candidate_gate": {
                "baseline": "physical radius <= 10 um",
                "experiment": "nearest next-frame candidates",
                "top_k": args.top_k,
                "otherwise_baseline_unchanged": True,
            },
            "production_config_changed": False,
            "production_eligible": False,
            "submit_eligible": False,
            "internet_access": False,
        },
        "candidate_pairs": {
            "topk": candidate_pairs,
            "radius_10um_diagnostic": radius_candidate_pairs,
            "delta": candidate_pairs - radius_candidate_pairs,
        },
        "fragments": {"path": str(fragment_dir), "count": len(videos), "records": videos},
        "official_scoring": {
            "summary": summary,
            "coverage": coverage,
            "input_provenance": input_provenance,
            "scorer_parity": scorer_parity,
        },
        "provenance": {
            "baseline_scores": {
                "path": str(args.baseline_scores),
                "sha256": sweep_support._sha256(args.baseline_scores),
            },
            "split": {"path": str(args.split), "sha256": sweep_support._sha256(args.split)},
            "edge_artifact": {"path": str(args.edge_artifact), "sha256": edge_sha256},
            "graph_source": {
                "path": str(ROOT / "src/celltrack_alpha/bh_graph.py"),
                "sha256": sweep_support._sha256(ROOT / "src/celltrack_alpha/bh_graph.py"),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sweep_support._sha256(Path(__file__).resolve()),
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_report(args.output / "report.json", payload)
    (args.output / "report.md").write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--baseline-fragments", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-scores", type=Path, default=DEFAULT_BASELINE_SCORES)
    parser.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    parser.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--scorer-parity", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main() -> int:
    try:
        payload = run(parse_args())
    except (TopKGateError, OSError, RuntimeError, ValueError) as exc:
        print(f"E1 top-k gate: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"topk": payload["topk"], "delta": payload["delta"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
