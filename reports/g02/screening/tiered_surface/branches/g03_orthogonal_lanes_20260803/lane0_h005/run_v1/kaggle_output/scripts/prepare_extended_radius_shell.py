#!/usr/bin/env python3
"""Prepare fit-only 10--12 um shell candidates with exact graph conflicts.

The preparation expands the deterministic source candidate radius from 10 to
12 um while retaining cap 8, then records only newly admitted shell pairs.
Each pair is classified against the exact frozen c=0.25 selected graph and the
official fit-only detection matching.  All positives and known negatives are
kept; unknown deployment background is represented by deterministic hard and
hash-uniform samples.  No prediction is emitted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import heapq
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import torch
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
from celltrack_alpha import bh_edge, bh_graph, bh_model  # noqa: E402
from celltrack_alpha.edge_data import match_detections_to_gt  # noqa: E402
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
    _cache_identity,
    _load_base_scorer,
)


SCHEMA = "celltrack-alpha.extended-radius-shell-prepare.v1"
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
FEATURE_NAMES = (
    "distance_um",
    "retained_rank",
    "displacement_z_um",
    "displacement_y_um",
    "displacement_x_um",
    "source_density_10um",
    "target_density_10um",
    "topology_code",
    "conflict_count",
    "source_history_available",
    "source_acceleration_residual_um",
    "target_future_available",
    "target_continuation_residual_um",
    "source_no_link_probability",
    "incumbent_probability_available",
    "maximum_incumbent_probability",
    "minimum_incumbent_distance_um",
    "incumbent_minus_challenger_distance_um",
)
DEFAULT_PREPARE = (
    ROOT / "reports/g02/selected_graph_conflict_prepare_v1/prepare.json"
)
DEFAULT_REACHABILITY = (
    ROOT / "reports/g02/extended_radius_reachability_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/extended_radius_shell_prepare_v1"
)


class ShellPrepareError(RuntimeError):
    """A shell preparation identity, topology, or sampling invariant failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShellPrepareError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ShellPrepareError(f"expected JSON object: {path}")
    return value


def topology_code(
    source: int,
    target: int,
    selected_out: Mapping[int, int],
    selected_in: Mapping[int, int],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Return free/source/target/two-endpoint class and exact conflicts."""

    conflicts: list[tuple[int, int]] = []
    source_target = selected_out.get(source)
    if source_target is not None and source_target != target:
        conflicts.append((source, int(source_target)))
    target_source = selected_in.get(target)
    if target_source is not None and target_source != source:
        pair = (int(target_source), target)
        if pair not in conflicts:
            conflicts.append(pair)
    source_conflict = source_target is not None and source_target != target
    target_conflict = target_source is not None and target_source != source
    code = int(source_conflict) + 2 * int(target_conflict)
    return code, tuple(conflicts)


def label_pair(
    pair: tuple[int, int],
    detection_to_gt: Mapping[int, int | None],
    gt_edges: set[tuple[int, int]],
    gt_outdegree: Mapping[int, int],
    gt_indegree: Mapping[int, int],
) -> int:
    """Return 1 positive, 0 known negative, -1 unknown, or -2 division truth."""

    source_gt = detection_to_gt.get(pair[0])
    target_gt = detection_to_gt.get(pair[1])
    if source_gt is None or target_gt is None:
        return -1
    mapped = (int(source_gt), int(target_gt))
    if mapped not in gt_edges:
        return 0
    if gt_outdegree[mapped[0]] != 1 or gt_indegree[mapped[1]] != 1:
        return -2
    return 1


def _baseline(
    cache: bh_model.SparseInferenceCache,
    scorer: Any,
) -> tuple[
    set[tuple[int, int]],
    Mapping[tuple[int, int], float],
    Mapping[int, float],
    dict[str, Any],
]:
    probabilities, no_link, association = run_bh.score_sparse_edge_cache(
        cache, scorer, edge_module=bh_edge
    )
    graph_nodes = run_bh.detections_to_graph_nodes(
        cache.detections, bh_graph.GraphNode
    )
    graph = bh_graph.calibrate_v45_graph(
        graph_nodes,
        learned_edge_probabilities=probabilities,
        source_no_link_probabilities=no_link,
        topology_mode="legacy",
        division_enabled=False,
    )
    if (
        graph.provenance.get("motion_direction") != "backward"
        or graph.provenance.get("motion_error_weight") != 0.25
        or graph.provenance.get("division_enabled") is not False
        or graph.provenance.get("topology_mode") != "legacy"
    ):
        raise ShellPrepareError("frozen baseline graph identity drifted")
    raw_ids = {
        int(node_id)
        for frame in cache.detections
        for node_id in frame.node_ids
    }
    selected = {
        (int(edge.source_id), int(edge.target_id))
        for edge in graph.edges
        if int(edge.source_id) in raw_ids and int(edge.target_id) in raw_ids
    }
    selected_out = {source: target for source, target in selected}
    selected_in = {target: source for source, target in selected}
    if len(selected_out) != len(selected) or len(selected_in) != len(selected):
        raise ShellPrepareError("division-off selected graph is not one-to-one")
    return selected, probabilities, no_link, {
        "raw_selected_edges": len(selected),
        "output_nodes": len(graph.nodes),
        "output_edges": len(graph.edges),
        "association": association,
    }


def _feature(
    *,
    pair: tuple[int, int],
    rank: int,
    topology: int,
    conflicts: Sequence[tuple[int, int]],
    detections: Mapping[int, tuple[int, np.ndarray]],
    source_density: int,
    target_density: int,
    selected_in: Mapping[int, int],
    selected_out: Mapping[int, int],
    probabilities: Mapping[tuple[int, int], float],
    no_link: Mapping[int, float],
) -> np.ndarray:
    source, target = pair
    source_t, source_point = detections[source]
    target_t, target_point = detections[target]
    if target_t != source_t + 1:
        raise ShellPrepareError("shell pair is not adjacent")
    displacement = target_point - source_point
    distance = float(np.linalg.norm(displacement))
    predecessor = selected_in.get(source)
    history_available = False
    acceleration = 20.0
    if predecessor is not None:
        predecessor_t, predecessor_point = detections[predecessor]
        if predecessor_t == source_t - 1:
            history_available = True
            acceleration = float(
                np.linalg.norm(displacement - (source_point - predecessor_point))
            )
    successor = selected_out.get(target)
    future_available = False
    continuation = 20.0
    if successor is not None:
        successor_t, successor_point = detections[successor]
        if successor_t == target_t + 1:
            future_available = True
            continuation = float(
                np.linalg.norm((successor_point - target_point) - displacement)
            )
    conflict_distances = [
        float(np.linalg.norm(detections[b][1] - detections[a][1]))
        for a, b in conflicts
    ]
    conflict_probabilities = [
        float(probabilities[conflict])
        for conflict in conflicts
        if conflict in probabilities
    ]
    incumbent_distance = min(conflict_distances, default=20.0)
    incumbent_probability = max(conflict_probabilities, default=-1.0)
    value = np.asarray(
        (
            distance,
            rank,
            *displacement.tolist(),
            source_density,
            target_density,
            topology,
            len(conflicts),
            int(history_available),
            acceleration,
            int(future_available),
            continuation,
            float(no_link.get(source, 1.0)),
            int(bool(conflict_probabilities)),
            incumbent_probability,
            incumbent_distance,
            incumbent_distance - distance,
        ),
        dtype=np.float32,
    )
    if value.shape != (len(FEATURE_NAMES),) or not np.isfinite(value).all():
        raise ShellPrepareError("shell feature vector is malformed")
    return value


def _hash_rank(identity: str, pair: tuple[int, int]) -> str:
    return hashlib.sha256(
        f"{identity}:{pair[0]}:{pair[1]}".encode("ascii")
    ).hexdigest()


def _sample_unknown(
    rows: Sequence[dict[str, Any]],
    *,
    identity: str,
    hard_limit: int,
    uniform_limit: int,
) -> list[dict[str, Any]]:
    hard = sorted(
        rows,
        key=lambda row: (
            min(float(row["features"][10]), float(row["features"][12])),
            float(row["features"][0]),
            tuple(row["pair"]),
        ),
    )[:hard_limit]
    uniform = sorted(
        rows,
        key=lambda row: _hash_rank(identity, tuple(row["pair"])),
    )[:uniform_limit]
    by_pair = {tuple(row["pair"]): row for row in (*hard, *uniform)}
    return [by_pair[pair] for pair in sorted(by_pair)]


def _bounded_push(
    heap: list[tuple[Any, ...]],
    entry: tuple[Any, ...],
    *,
    limit: int,
) -> None:
    """Keep entries whose inverted key ranks largest (original key smallest)."""

    if limit < 1:
        return
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def shell_candidate_indices(
    source_ids: np.ndarray,
    source_points_um: np.ndarray,
    target_ids: np.ndarray,
    target_points_um: np.ndarray,
    *,
    target_tree: cKDTree | None = None,
) -> list[tuple[int, int, int, float, np.ndarray]]:
    """Return the exact deterministic 10--12 um/cap-8 shell rows.

    This is the shared pool-construction primitive used by the original shell
    preparation and downstream deterministic diagnostics.  Each tuple contains
    ``(source_index, target_index, retained_rank, distance_um, displacement_um)``.
    """

    source_ids = np.asarray(source_ids, dtype=np.int64)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    source_points = np.asarray(source_points_um, dtype=np.float64)
    target_points = np.asarray(target_points_um, dtype=np.float64)
    if (
        source_ids.shape != (len(source_points),)
        or target_ids.shape != (len(target_points),)
        or source_points.shape[1:] != (3,)
        or target_points.shape[1:] != (3,)
        or not np.isfinite(source_points).all()
        or not np.isfinite(target_points).all()
    ):
        raise ShellPrepareError("shell candidate inputs are malformed")
    tree = target_tree if target_tree is not None else cKDTree(target_points)
    rows: list[tuple[int, int, int, float, np.ndarray]] = []
    for source_index, source_point in enumerate(source_points):
        neighbors = [
            int(value) for value in tree.query_ball_point(source_point, 12.0)
        ]
        ordered = sorted(
            neighbors,
            key=lambda index: (
                float(np.linalg.norm(target_points[index] - source_point)),
                int(target_ids[index]),
                index,
            ),
        )[:8]
        for retained_rank, target_index in enumerate(ordered, start=1):
            displacement = target_points[target_index] - source_point
            distance = float(np.linalg.norm(displacement))
            if distance <= 10.0 + 1.0e-9:
                continue
            rows.append(
                (
                    source_index,
                    target_index,
                    retained_rank,
                    distance,
                    displacement,
                )
            )
    return rows


def _video(
    *,
    identity: str,
    cache: bh_model.SparseInferenceCache,
    scorer: Any,
    gt: Mapping[str, Any],
    expected_selected_edges: int,
    hard_limit: int,
    uniform_limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected, probabilities, no_link, graph = _baseline(cache, scorer)
    if graph["raw_selected_edges"] != expected_selected_edges:
        raise ShellPrepareError(
            f"selected graph count drifted for {identity}: "
            f"{graph['raw_selected_edges']} != {expected_selected_edges}"
        )
    selected_out = {source: target for source, target in selected}
    selected_in = {target: source for source, target in selected}
    detection_to_gt, gt_edges = match_detections_to_gt(cache.detections, gt)
    gt_outdegree: dict[int, int] = defaultdict(int)
    gt_indegree: dict[int, int] = defaultdict(int)
    for source, target in gt_edges:
        gt_outdegree[source] += 1
        gt_indegree[target] += 1

    detections: dict[int, tuple[int, np.ndarray]] = {}
    for frame in cache.detections:
        for node_id, coordinate in zip(
            frame.node_ids, frame.coords_zyx, strict=True
        ):
            detections[int(node_id)] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float64) * SCALE_ZYX,
            )
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    unknown_hard: list[tuple[Any, ...]] = []
    unknown_uniform: list[tuple[Any, ...]] = []
    counts: dict[str, int] = defaultdict(int)
    for source_frame, target_frame in zip(
        cache.detections[:-1], cache.detections[1:], strict=True
    ):
        source_ids = np.asarray(source_frame.node_ids, dtype=np.int64)
        target_ids = np.asarray(target_frame.node_ids, dtype=np.int64)
        source_points = (
            np.asarray(source_frame.coords_zyx, dtype=np.float64) * SCALE_ZYX
        )
        target_points = (
            np.asarray(target_frame.coords_zyx, dtype=np.float64) * SCALE_ZYX
        )
        if not len(source_points) or not len(target_points):
            continue
        target_tree = cKDTree(target_points)
        source_tree = cKDTree(source_points)
        source_density = np.asarray(
            source_tree.query_ball_point(
                source_points, 10.0, return_length=True
            ),
            dtype=np.int64,
        )
        target_density = np.asarray(
            target_tree.query_ball_point(
                target_points, 10.0, return_length=True
            ),
            dtype=np.int64,
        )
        shell_rows = shell_candidate_indices(
            source_ids,
            source_points,
            target_ids,
            target_points,
            target_tree=target_tree,
        )
        for (
            source_index,
            target_index,
            local_rank,
            distance,
            _displacement,
        ) in shell_rows:
            counts["shell_pairs"] += 1
            pair = (
                int(source_ids[source_index]),
                int(target_ids[target_index]),
            )
            topology, conflicts = topology_code(
                pair[0], pair[1], selected_out, selected_in
            )
            label = label_pair(
                pair,
                detection_to_gt,
                gt_edges,
                gt_outdegree,
                gt_indegree,
            )
            selected_identity = pair in selected
            counts[f"topology_{topology}"] += 1
            counts[
                {
                    1: "positive_pairs",
                    0: "known_negative_pairs",
                    -1: "unknown_pairs",
                    -2: "division_truth_pairs",
                }[label]
            ] += 1
            counts["already_selected_pairs"] += int(selected_identity)
            counts["already_selected_positive_pairs"] += int(
                selected_identity and label == 1
            )
            if selected_identity or label == -2:
                continue
            row = {
                "pair": list(pair),
                "conflicts": [list(conflict) for conflict in conflicts],
                "topology": topology,
                "label": label,
                "features": _feature(
                    pair=pair,
                    rank=local_rank,
                    topology=topology,
                    conflicts=conflicts,
                    detections=detections,
                    source_density=int(source_density[source_index]),
                    target_density=int(target_density[target_index]),
                    selected_in=selected_in,
                    selected_out=selected_out,
                    probabilities=probabilities,
                    no_link=no_link,
                ),
            }
            if label == 1:
                positives.append(row)
            elif label == 0:
                negatives.append(row)
            else:
                hard_residual = min(
                    float(row["features"][10]),
                    float(row["features"][12]),
                )
                _bounded_push(
                    unknown_hard,
                    (
                        -hard_residual,
                        -float(row["features"][0]),
                        -pair[0],
                        -pair[1],
                        row,
                    ),
                    limit=hard_limit,
                )
                hash_value = int(_hash_rank(identity, pair)[:16], 16)
                _bounded_push(
                    unknown_uniform,
                    (-hash_value, -pair[0], -pair[1], row),
                    limit=uniform_limit,
                )
    sampled_by_pair = {
        tuple(entry[-1]["pair"]): entry[-1]
        for entry in (*unknown_hard, *unknown_uniform)
    }
    sampled_unknown = [
        sampled_by_pair[pair] for pair in sorted(sampled_by_pair)
    ]
    sampled = (*positives, *negatives, *sampled_unknown)
    feature_array = (
        np.stack([row["features"] for row in sampled]).astype(
            np.float32, copy=False
        )
        if sampled
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    )
    labels = np.asarray([int(row["label"]) for row in sampled], dtype=np.int8)
    roles = np.asarray(
        [
            1 if row["label"] == 1 else 0 if row["label"] == 0 else -1
            for row in sampled
        ],
        dtype=np.int8,
    )
    pairs = np.asarray(
        [row["pair"] for row in sampled], dtype=np.int64
    ).reshape(-1, 2)
    topology = np.asarray(
        [row["topology"] for row in sampled], dtype=np.int8
    )
    if (
        feature_array.shape != (len(sampled), len(FEATURE_NAMES))
        or labels.shape != roles.shape
        or pairs.shape != (len(sampled), 2)
        or topology.shape != labels.shape
    ):
        raise ShellPrepareError("sampled shell arrays disagree")
    example = {
        "feature_names": FEATURE_NAMES,
        "features": feature_array,
        "labels": labels,
        "sample_roles": roles,
        "pairs": pairs,
        "topology": topology,
    }
    report = {
        "identity": identity,
        "graph": graph,
        "counts": dict(counts),
        "sample": {
            "positives": len(positives),
            "known_negatives": len(negatives),
            "unknown_total": int(counts.get("unknown_pairs", 0)),
            "unknown_sampled": len(sampled_unknown),
            "rows": len(sampled),
        },
    }
    return report, example


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ShellPrepareError(f"refusing to overwrite output: {args.output}")
    parent = _load_json(args.prepare)
    reachability = _load_json(args.reachability)
    scope = parent.get("scope")
    videos = parent.get("videos")
    if (
        parent.get("schema")
        != "celltrack-alpha.selected-graph-conflict-prepare.v1"
        or reachability.get("schema")
        != "celltrack-alpha.extended-radius-reachability-audit.v1"
        or not isinstance(scope, Mapping)
        or not isinstance(videos, list)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
        or reachability.get("scope", {}).get("train_video_ids")
        != scope.get("train_video_ids")
        or reachability.get("scope", {}).get("validation_video_ids")
        != scope.get("validation_video_ids")
    ):
        raise ShellPrepareError("parent artifacts do not share exact fit-only scope")
    train_ids = tuple(str(v) for v in scope["train_video_ids"])
    validation_ids = tuple(str(v) for v in scope["validation_video_ids"])
    parent_by_id = {str(row["identity"]): row for row in videos}
    if set(parent_by_id) != set((*train_ids, *validation_ids)):
        raise ShellPrepareError("parent video coverage is incomplete")

    scorer, _model, edge_record = _load_base_scorer(
        args.edge_artifact,
        args.edge_auxiliary,
        args.support_root,
        "cpu",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    examples = args.output / "examples"
    examples.mkdir()
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate((*train_ids, *validation_ids), start=1):
            slug = identity.replace(":", "__", 1)
            cache_path = args.cache / f"{slug}.bhc"
            metadata_path = args.cache / f"{slug}.json"
            cache = bh_model.load_sparse_cache(
                cache_path, expected_identity=_cache_identity(metadata_path)
            )
            dataset = identity.split(":", 1)[1]
            report, example = _video(
                identity=identity,
                cache=cache,
                scorer=scorer,
                gt=read_geff_arrays(store, "train", dataset),
                expected_selected_edges=int(
                    parent_by_id[identity]["graph"]["raw_selected_edges"]
                ),
                hard_limit=args.background_hard_per_video,
                uniform_limit=args.background_uniform_per_video,
            )
            destination = examples / f"{slug}.pt"
            torch.save(example, destination)
            report.update(
                {
                    "partition": (
                        "train" if identity in train_ids else "validation"
                    ),
                    "path": str(destination.relative_to(ROOT)),
                    "sha256": _sha256(destination),
                }
            )
            reports.append(report)
            print(
                f"shell prepare {index:03d}/117 {identity} "
                f"shell={report['counts'].get('shell_pairs', 0)} "
                f"positive={report['counts'].get('positive_pairs', 0)} "
                f"sample={report['sample']['rows']}",
                flush=True,
            )
    totals = {
        name: sum(int(row["counts"].get(name, 0)) for row in reports)
        for name in sorted(
            {name for row in reports for name in row["counts"]}
        )
    }
    train_positive = sum(
        int(row["counts"].get("positive_pairs", 0))
        for row in reports
        if row["partition"] == "train"
    )
    validation_positive = sum(
        int(row["counts"].get("positive_pairs", 0))
        for row in reports
        if row["partition"] == "validation"
    )
    if (
        totals.get("shell_pairs")
        != (
            int(
                next(
                    row
                    for row in reachability["all_fit"]["policies"]
                    if row["policy"] == "r12_k08"
                )["candidate_pairs"]
            )
            - int(
                next(
                    row
                    for row in reachability["all_fit"]["policies"]
                    if row["policy"] == "r10_k08"
                )["candidate_pairs"]
            )
        )
        or train_positive != 405
        or validation_positive != 51
    ):
        raise ShellPrepareError(
            "shell/positive counts do not reconcile reachability evidence"
        )
    result = {
        "schema": SCHEMA,
        "status": "prepared",
        "scope": {
            "train_video_ids": list(train_ids),
            "validation_video_ids": list(validation_ids),
            "fit_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "mechanism": {
            "shell": "10 < physical distance <= 12 um",
            "neighbor_cap": 8,
            "ordering": "distance, target node ID, target input index",
            "graph": "exact frozen c=0.25 legacy selected graph",
            "labels": (
                "officially matched one-to-one GT edge / known matched non-edge "
                "/ unmatched-background unknown"
            ),
            "predictions_emitted": False,
            "gnn": False,
        },
        "feature_names": list(FEATURE_NAMES),
        "background_sampling": {
            "hard_per_video": args.background_hard_per_video,
            "uniform_per_video": args.background_uniform_per_video,
            "hard_order": (
                "minimum available trajectory residual, distance, pair identity"
            ),
            "uniform_order": "sha256(video:source:target)",
            "all_positives_and_known_negatives_retained": True,
        },
        "totals": totals,
        "videos": reports,
        "frozen_pairwise": edge_record,
        "cache_identity": asdict(
            _cache_identity(
                args.cache / f"{train_ids[0].replace(':', '__', 1)}.json"
            )
        ),
        "provenance": {
            "prepare": str(args.prepare),
            "prepare_sha256": _sha256(args.prepare),
            "reachability": str(args.reachability),
            "reachability_sha256": _sha256(args.reachability),
            "edge_artifact": str(args.edge_artifact),
            "edge_artifact_sha256": _sha256(args.edge_artifact),
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_action": False,
            "kaggle_action": False,
        },
    }
    run_bh.atomic_create_json(args.output / "prepare.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--reachability", type=Path, default=DEFAULT_REACHABILITY)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--background-hard-per-video", type=int, default=512)
    value.add_argument("--background-uniform-per-video", type=int, default=512)
    return value


if __name__ == "__main__":
    report = prepare(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "totals": report["totals"],
            },
            sort_keys=True,
        )
    )
