#!/usr/bin/env python3
"""Offline-only CV57 graph-weight sweep over persisted D4 inference caches."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
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
from celltrack_alpha import bh_edge, bh_graph, bh_model  # noqa: E402


SCHEMA = "celltrack-alpha.bh-cv57-graph-weight-sweep.v1"
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
DEFAULT_OUTPUT = ROOT / "reports/g02/learned_bonus_sweep_holdout57_v1"


class SweepError(RuntimeError):
    """The cached graph sweep cannot preserve its offline-only contract."""


class SweepGraphConfig:
    """Read-only graph config assembled only by this offline sweep script."""

    __slots__ = ("_values",)

    def __init__(self, *, learned_bonus: float, raw_distance_weight: float) -> None:
        values = bh_graph.V45_GRAPH_CONFIG.as_dict()
        values["learned_bonus"] = float(learned_bonus)
        values["raw_distance_weight"] = float(raw_distance_weight)
        values["scale_zyx"] = tuple(values["scale_zyx"])
        object.__setattr__(self, "_values", values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("sweep graph config is read-only")

    def as_dict(self) -> dict[str, Any]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in self._values.items()
        }


class MemoizedImageRefiner:
    """Share exact midpoint refinements across settings for one video."""

    def __init__(self, image: Any) -> None:
        self.base = run_bh.LazyImageRefiner(
            image,
            bh_graph.intensity_centroid_refine,
            deadline_epoch=None,
        )
        self.cache: dict[tuple[int, tuple[float, float, float]], Any] = {}
        self.queries = 0
        self.hits = 0
        self.lock = threading.Lock()

    def __call__(self, frame: int, coord_zyx: tuple[float, float, float]) -> Any:
        key = (int(frame), tuple(float(value) for value in coord_zyx))
        with self.lock:
            self.queries += 1
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            value = self.base(*key)
            self.cache[key] = value
            return value

    def report(self) -> dict[str, int | bool]:
        return {
            **self.base.report(),
            "refinement_queries": self.queries,
            "refinement_result_cache_hits": self.hits,
            "unique_refinements": len(self.cache),
        }


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SweepError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SweepError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_identity(meta: Mapping[str, Any]) -> bh_model.CacheIdentity:
    try:
        model = meta["contract"]["model"]
        return bh_model.CacheIdentity(
            str(model["config_sha256"]),
            str(model["source_sha256"]),
            str(model["weight_sha256"]),
            int(model["pair_batch_size"]),
            int(model["view_batch_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SweepError("cache metadata lacks a valid model identity") from exc


def _prediction_component_count(graph_result: Any) -> int:
    parent = {int(node.node_id): int(node.node_id) for node in graph_result.nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for edge in graph_result.edges:
        left = find(int(edge.source_id))
        right = find(int(edge.target_id))
        if left != right:
            parent[right] = left
    return len({find(node) for node in parent})


def _setting_name(learned_bonus: float, raw_distance_weight: float) -> str:
    bonus = format(learned_bonus, "g").replace(".", "p")
    distance = format(raw_distance_weight, "g").replace(".", "p")
    return f"learned_bonus_{bonus}__raw_distance_weight_{distance}"


def _validate_values(values: Sequence[float], raw_distance_weight: float) -> tuple[float, ...]:
    bonuses = tuple(float(value) for value in values)
    if (
        not bonuses
        or len(set(bonuses)) != len(bonuses)
        or any(not math.isfinite(value) or value <= 0.0 for value in bonuses)
    ):
        raise SweepError("learned bonuses must be unique positive finite values")
    if not math.isfinite(raw_distance_weight) or raw_distance_weight <= 0.0:
        raise SweepError("raw-distance weight must be positive and finite")
    return bonuses


def _fragment_parity(left: Path, right: Path) -> dict[str, Any]:
    replay = run_bh.read_rows(left)
    baseline = run_bh.read_rows(right)
    replay_nodes = {
        int(row["node_id"]): (
            int(row["t"]),
            int(row["z"]),
            int(row["y"]),
            int(row["x"]),
        )
        for row in replay
        if row["row_type"] == "node"
    }
    baseline_nodes = {
        int(row["node_id"]): (
            int(row["t"]),
            int(row["z"]),
            int(row["y"]),
            int(row["x"]),
        )
        for row in baseline
        if row["row_type"] == "node"
    }
    replay_edges = {
        (int(row["source_id"]), int(row["target_id"]))
        for row in replay
        if row["row_type"] == "edge"
    }
    baseline_edges = {
        (int(row["source_id"]), int(row["target_id"]))
        for row in baseline
        if row["row_type"] == "edge"
    }
    node_identity_time_exact = (
        set(replay_nodes) == set(baseline_nodes)
        and all(replay_nodes[node][0] == baseline_nodes[node][0] for node in replay_nodes)
    )
    coordinate_mismatches = [
        node
        for node in replay_nodes.keys() & baseline_nodes.keys()
        if replay_nodes[node][1:] != baseline_nodes[node][1:]
    ]
    max_coordinate_delta_voxels = max(
        (
            max(
                abs(left_value - right_value)
                for left_value, right_value in zip(
                    replay_nodes[node][1:], baseline_nodes[node][1:], strict=True
                )
            )
            for node in coordinate_mismatches
        ),
        default=0,
    )
    return {
        "row_exact": replay == baseline,
        "row_count_replay": len(replay),
        "row_count_baseline": len(baseline),
        "node_identity_time_exact": node_identity_time_exact,
        "edge_topology_exact": replay_edges == baseline_edges,
        "coordinate_mismatch_nodes": len(coordinate_mismatches),
        "max_coordinate_delta_voxels": max_coordinate_delta_voxels,
    }


def _solve_setting(
    *,
    bonus: float,
    raw_distance_weight: float,
    experiment_id: str,
    identity: str,
    dataset: str,
    shape: tuple[int, ...],
    graph_nodes: Sequence[Any],
    probabilities: Mapping[Any, Any],
    no_link: Mapping[Any, Any],
    refiner: MemoizedImageRefiner,
    fragment_dir: Path,
    baseline_fragment: Path,
) -> dict[str, Any]:
    config = SweepGraphConfig(
        learned_bonus=bonus,
        raw_distance_weight=raw_distance_weight,
    )
    varied_fields = ["learned_bonus"]
    if raw_distance_weight != bh_graph.V45_GRAPH_CONFIG.raw_distance_weight:
        varied_fields.append("raw_distance_weight")
    sweep_provenance = {
        "schema": bh_graph.EXPERIMENTAL_SWEEP_SCHEMA,
        "experiment_id": experiment_id,
        "purpose": "offline-cached-cv57-sweep",
        "varied_fields": varied_fields,
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
    )
    if (
        graph_result.provenance.get("production_eligible") is not False
        or graph_result.provenance.get("experimental_sweep") != sweep_provenance
        or graph_result.provenance.get("division_enabled") is not False
    ):
        raise SweepError("graph result lost offline-only/division-off provenance")
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
    if bonus == 1.0 and raw_distance_weight == 0.05:
        parity = _fragment_parity(fragment, baseline_fragment)
        if (
            parity["node_identity_time_exact"] is not True
            or parity["edge_topology_exact"] is not True
        ):
            raise SweepError(
                f"learned_bonus=1 replay topology differs from released fragment for {identity}"
            )
    return {
        "fragment": fragment,
        "fragment_sha256": _sha256(fragment),
        "nodes": len(graph_result.nodes),
        "edges": len(graph_result.edges),
        "prediction_component_count": _prediction_component_count(graph_result),
        "baseline_parity": parity,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    bonuses = _validate_values(args.learned_bonus, args.raw_distance_weight)
    if args.output.exists():
        raise SweepError(f"refusing to overwrite sweep output: {args.output}")
    args.output.mkdir(parents=True)
    split = _load(args.split, "frozen split")
    baseline_scores = _load(args.baseline_scores, "post-patch baseline scores")
    scorer_parity = score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    archive_ids = scoreall.archive_train_video_ids(args.zip)
    selected = score_bh_cv57.exact_selection(
        split,
        split_path=args.split,
        archive_video_ids=archive_ids,
    )
    baseline_manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(args.baseline_fragments)), selected
    )
    baseline_paths = dict(baseline_manifest.ordered)
    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if set(cache_paths) != set(selected) or set(meta_paths) != set(selected):
        raise SweepError("persisted cache/meta files do not exactly cover CV57")

    model, bundle = bh_model.load_v45_model(args.support_root, device="cpu")
    edge_sha256 = _sha256(args.edge_artifact)
    edge_scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        edge_sha256,
        args.edge_auxiliary,
        model,
        bundle,
        edge_module=bh_edge,
    )
    if edge_scorer is None or edge_record is None or edge_record.get("division_enabled") is not False:
        raise SweepError("the fitted edge scorer was not restored with divisions disabled")

    settings: dict[float, dict[str, Any]] = {}
    for bonus in bonuses:
        name = _setting_name(bonus, args.raw_distance_weight)
        fragment_dir = args.output / name / "fragments"
        fragment_dir.mkdir(parents=True)
        settings[bonus] = {
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
            meta = _load(meta_paths[identity], f"cache metadata for {identity}")
            identity_record = _cache_identity(meta)
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
                raise SweepError(f"unexpected image shape for {identity}: {shape}")
            shared_refiner = MemoizedImageRefiner(image)

            with ThreadPoolExecutor(max_workers=min(len(bonuses), 5)) as executor:
                futures = {
                    bonus: executor.submit(
                        _solve_setting,
                        bonus=bonus,
                        raw_distance_weight=args.raw_distance_weight,
                        experiment_id=args.output.name,
                        identity=identity,
                        dataset=dataset,
                        shape=shape,
                        graph_nodes=graph_nodes,
                        probabilities=probabilities,
                        no_link=no_link,
                        refiner=shared_refiner,
                        fragment_dir=settings[bonus]["fragment_dir"],
                        baseline_fragment=baseline_paths[identity],
                    )
                    for bonus in bonuses
                }
                solved = {bonus: futures[bonus].result() for bonus in bonuses}
            refinement = shared_refiner.report()
            for bonus in bonuses:
                setting = settings[bonus]
                current = solved[bonus]
                parity = current["baseline_parity"]
                if parity is not None:
                    setting["baseline_parity"].append({"identity": identity, **parity})
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
                        "image_refinement": {
                            **refinement,
                            "shared_across_settings": True,
                        },
                        "association_candidate_count": association["candidate_count"],
                    }
                )
            shared_refinement_frame_reads += int(refinement["frame_reads"])
            print(f"graph replay {index:02d}/{len(selected)} {identity}", flush=True)

        scored_settings = []
        for bonus in bonuses:
            setting = settings[bonus]
            manifest = scoreall.validate_fragment_set(
                scoreall.resolve_fragment_paths(str(setting["fragment_dir"])), selected
            )
            rows, input_provenance, coverage = scoreall.score_fragment_input(store, manifest)
            by_identity = {row["identity"]: row for row in rows}
            if set(by_identity) != set(selected):
                raise SweepError("official scoring escaped the exact CV57 selection")
            rows = [by_identity[identity] for identity in selected]
            summary = scoreall.official_summary(rows)
            edge_counts = {
                name: int(sum(float(row[name]) for row in rows))
                for name in ("edge_tp", "edge_fp", "edge_fn")
            }
            scored_settings.append(
                {
                    "learned_bonus": bonus,
                    "raw_distance_weight": args.raw_distance_weight,
                    "name": setting["name"],
                    "summary": summary,
                    "edge_counts": edge_counts,
                    "prediction_component_count": setting["prediction_component_count"],
                    "prediction_component_count_mean_per_video": (
                        setting["prediction_component_count"] / len(selected)
                    ),
                    "fragments": {
                        "path": str(setting["fragment_dir"]),
                        "count": len(setting["videos"]),
                        "records": setting["videos"],
                    },
                    "coverage": coverage,
                    "input_provenance": input_provenance,
                    "baseline_fragment_parity": (
                        {
                            "exact_fragment_count": sum(
                                item["row_exact"] for item in setting["baseline_parity"]
                            ),
                            "topology_exact_count": sum(
                                item["edge_topology_exact"]
                                and item["node_identity_time_exact"]
                                for item in setting["baseline_parity"]
                            ),
                            "coordinate_mismatch_nodes": sum(
                                item["coordinate_mismatch_nodes"]
                                for item in setting["baseline_parity"]
                            ),
                            "max_coordinate_delta_voxels": max(
                                (
                                    item["max_coordinate_delta_voxels"]
                                    for item in setting["baseline_parity"]
                                ),
                                default=0,
                            ),
                            "records": setting["baseline_parity"],
                        }
                        if setting["baseline_parity"]
                        else None
                    ),
                }
            )
            print(
                f"official score learned_bonus={bonus:g} "
                f"adj_edge={summary['adj_edge_jaccard']:.12f}",
                flush=True,
            )

    result = {
        "schema": SCHEMA,
        "status": "completed",
        "selection": {
            "video_count": len(selected),
            "video_ids": list(selected),
        },
        "execution": {
            "detector_inference": False,
            "detector_forward_calls": 0,
            "graph_solve_only": True,
            "cache_inputs": ["FrameDet", "PairContext", "PairEdges"],
            "edge_probability_source": "fitted-scorer replay over persisted PairContext",
            "image_access": "original voxels for locked lazy p20 centroid refinement only",
            "division_enabled": False,
            "topology_mode": "legacy",
            "production_eligible": False,
            "submit_eligible": False,
            "shared_image_refinement_frame_reads": shared_refinement_frame_reads,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "provenance": {
            "cache": str(args.cache),
            "cache_file_count": len(cache_paths),
            "baseline_fragments": str(args.baseline_fragments),
            "edge_artifact": {"path": str(args.edge_artifact), "sha256": edge_sha256},
            "edge_auxiliary": {
                "path": str(args.edge_auxiliary),
                "sha256": _sha256(args.edge_auxiliary),
            },
            "competition_archive": {
                "path": str(args.zip),
                "sha256": baseline_scores.get("frozen_scorer", {})
                .get("archive", {})
                .get("sha256"),
                "identity_source": str(args.baseline_scores),
                "bytes_rehashed_in_sweep": False,
            },
            "scorer_parity": scorer_parity,
            "graph_source": {
                "path": str(ROOT / "src/celltrack_alpha/bh_graph.py"),
                "sha256": _sha256(ROOT / "src/celltrack_alpha/bh_graph.py"),
            },
            "sweep_source": {"path": str(Path(__file__)), "sha256": _sha256(Path(__file__))},
        },
        "settings": scored_settings,
    }
    default_setting = next(
        (
            setting
            for setting in scored_settings
            if setting["learned_bonus"] == 1.0
            and setting["raw_distance_weight"] == 0.05
        ),
        None,
    )
    baseline_summary = baseline_scores.get("summary")
    if default_setting is not None and isinstance(baseline_summary, Mapping):
        metric_names = (
            "adj_edge_jaccard",
            "edge_jaccard",
            "node_recall",
            "division_jaccard",
            "division_tp",
            "division_fp",
            "division_fn",
        )
        metric_exact = all(
            default_setting["summary"].get(name) == baseline_summary.get(name)
            for name in metric_names
        )
        result["default_control"] = {
            "reference": {
                "path": str(args.baseline_scores),
                "sha256": _sha256(args.baseline_scores),
                "summary": baseline_summary,
            },
            "replay_metric_exact": metric_exact,
            "metric_names": list(metric_names),
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
        "--learned-bonus",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0, 16.0],
    )
    value.add_argument("--raw-distance-weight", type=float, default=0.05)
    return value


def main() -> None:
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "settings": len(result["settings"])}))


if __name__ == "__main__":
    main()
