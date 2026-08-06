#!/usr/bin/env python3
"""Audit and exact-score preregistered frozen/global-registration repairs."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
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
import scoreall  # noqa: E402
import score_bh_cv57  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import bh_edge, bh_graph, bh_model  # noqa: E402
from celltrack_alpha.edge_data import match_detections_to_gt  # noqa: E402
from celltrack_alpha.frame_registration import (  # noqa: E402
    RegistrationPlan,
    build_registration_plan,
    coordinate_views,
    load_registration_plan,
)
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.g04-data-pathology-repair-score.v1"
DEFAULT_CENSUS = ROOT / "reports/g04/lanes/data_pathology_repair/CENSUS.json"
DEFAULT_BASE = ROOT / "reports/g02/selected_graph_appearance_direct_cv57_v1/fragments"
DEFAULT_BASE_REPORT = ROOT / "reports/g02/selected_graph_appearance_direct_cv57_v1/report.json"
DEFAULT_OUTPUT = ROOT / "reports/g04/lanes/data_pathology_repair/score"
FIT_CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"
SHELL_ROOT = ROOT / "reports/g02/extended_radius_shell_prepare_v1/examples"
CSV_FIELDS = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


class RepairScoreError(RuntimeError):
    """A frozen input or exact-score contract drifted."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepairScoreError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RepairScoreError(f"JSON is not an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    run_bh.atomic_create_json(path, value)


def cache_identity(path: Path) -> bh_model.CacheIdentity:
    value = load_json(path).get("contract", {}).get("model")
    if not isinstance(value, Mapping):
        raise RepairScoreError(f"cache metadata has no model identity: {path}")
    return bh_model.CacheIdentity(**dict(value))


def load_cache(root: Path, identity: str) -> bh_model.SparseInferenceCache:
    slug = identity.replace(":", "__", 1)
    return bh_model.load_sparse_cache(
        root / f"{slug}.bhc",
        expected_identity=cache_identity(root / f"{slug}.json"),
    )


def census_by_identity(census_path: Path) -> dict[str, Mapping[str, Any]]:
    census = load_json(census_path)
    if census.get("verdict") != "PASS_COMPLETE_CENSUS":
        raise RepairScoreError("pathology census is incomplete")
    result: dict[str, Mapping[str, Any]] = {}
    for row in census.get("videos", []):
        path = census_path.parent / str(row["path"])
        video = load_json(path)
        identity = f"{video['split']}:{video['dataset']}"
        result[identity] = video
    if len(result) != int(census.get("counts", {}).get("videos", -1)):
        raise RepairScoreError("pathology census video coverage drifted")
    return result


def selected_cv57(args: argparse.Namespace) -> tuple[str, ...]:
    split = load_json(args.split)
    score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    result = score_bh_cv57.exact_selection(
        split,
        split_path=args.split,
        archive_video_ids=scoreall.archive_train_video_ids(args.zip),
    )
    report_ids = tuple(str(value) for value in load_json(args.base_report)["selection"]["video_ids"])
    if result != report_ids:
        raise RepairScoreError("base report is not exact CV57 in canonical order")
    return result


def read_fragment(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise RepairScoreError(f"fragment schema drifted: {path}")
        rows = [dict(row) for row in reader]
    return rows


def write_fragment(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RepairScoreError(f"stale fragment temporary exists: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            value = dict(row)
            value["id"] = index
            writer.writerow({field: value[field] for field in CSV_FIELDS})
    temporary.replace(path)


def fragment_path(root: Path, identity: str) -> Path:
    return root / f"{identity.replace(':', '__', 1)}.csv"


def node_times(rows: Sequence[Mapping[str, str]]) -> dict[int, int]:
    result = {
        int(row["node_id"]): int(row["t"])
        for row in rows
        if row["row_type"] == "node"
    }
    if not result:
        raise RepairScoreError("fragment has no nodes")
    return result


def edge_pairs(rows: Sequence[Mapping[str, str]]) -> set[tuple[int, int]]:
    return {
        (int(row["source_id"]), int(row["target_id"]))
        for row in rows
        if row["row_type"] == "edge"
    }


def rows_with_edges(
    rows: Sequence[Mapping[str, str]], edges: set[tuple[int, int]]
) -> list[dict[str, Any]]:
    nodes = [dict(row) for row in rows if row["row_type"] == "node"]
    dataset = str(nodes[0]["dataset"])
    node_set = {int(row["node_id"]) for row in nodes}
    if any(source not in node_set or target not in node_set for source, target in edges):
        raise RepairScoreError("repair edge references a missing fragment node")
    edge_rows = [
        {
            "id": -1,
            "dataset": dataset,
            "row_type": "edge",
            "node_id": -1,
            "t": -1,
            "z": -1,
            "y": -1,
            "x": -1,
            "source_id": source,
            "target_id": target,
        }
        for source, target in sorted(edges)
    ]
    return nodes + edge_rows


def exact_coordinate_pairs(
    cache: bh_model.SparseInferenceCache, frozen: set[int]
) -> dict[int, set[tuple[int, int]]]:
    by_frame = {int(frame.frame_index): frame for frame in cache.detections}
    result: dict[int, set[tuple[int, int]]] = {}
    for frame in sorted(frozen):
        source = by_frame[frame]
        target = by_frame[frame + 1]
        target_by_coord = {
            tuple(int(value) for value in coord): int(node_id)
            for coord, node_id in zip(target.coords_zyx, target.node_ids, strict=True)
        }
        pairs = {
            (int(source_id), target_by_coord[coord])
            for source_id, raw_coord in zip(source.node_ids, source.coords_zyx, strict=True)
            if (coord := tuple(int(value) for value in raw_coord)) in target_by_coord
        }
        result[frame] = pairs
    return result


def apply_frozen(
    rows: Sequence[Mapping[str, str]], pairs_by_frame: Mapping[int, set[tuple[int, int]]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    edges = edge_pairs(rows)
    times = node_times(rows)
    removed = novel = retained = installed = skipped_pruned = 0
    for frame, pairs in sorted(pairs_by_frame.items()):
        for source, target in sorted(pairs):
            if source not in times or target not in times:
                skipped_pruned += 1
                continue
            conflicts = {
                edge
                for edge in edges
                if (edge[0] == source or edge[1] == target)
                and times[edge[0]] == frame
                and times[edge[1]] == frame + 1
                and edge != (source, target)
            }
            removed += len(conflicts)
            edges.difference_update(conflicts)
            installed += 1
            if (source, target) in edges:
                retained += 1
            else:
                edges.add((source, target))
                novel += 1
    return rows_with_edges(rows, edges), {
        "installed_pairs": installed,
        "retained_existing_pairs": retained,
        "novel_pairs": novel,
        "removed_conflicting_edges": removed,
        "skipped_pruned_pairs": skipped_pruned,
    }


def frame_shift_raw(transition: Mapping[str, Any]) -> np.ndarray:
    registered = np.asarray(transition["shift_registered_zyx"], dtype=np.int64)
    if registered.shape != (3,):
        raise RepairScoreError("registered shift is malformed")
    return registered * np.asarray((1, 4, 4), dtype=np.int64)


def cumulative_frame_transforms(
    transitions: Sequence[Mapping[str, Any]], detections: Sequence[bh_model.FrameDet]
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    compatibility_rows = [
        {
            **row,
            "shift_magnitude_um": row.get(
                "shift_magnitude_um",
                10.000000000000002 if bool(row.get("pool_breaking")) else 0.0,
            ),
        }
        for row in transitions
    ]
    plan = build_registration_plan(
        identity="train:compatibility-wrapper",
        transitions=compatibility_rows,
        census_sha256="0" * 64,
        frame_count=len(transitions) + 1,
    )
    transforms = {
        frame: np.asarray(plan.cumulative_shift_raw_zyx[frame], dtype=np.int64)
        for frame in range(plan.frame_count)
    }
    if set(transforms) != {int(frame.frame_index) for frame in detections}:
        raise RepairScoreError("registration transform coverage is incomplete")
    timepoints = np.concatenate(
        [
            np.full(len(frame.node_ids), int(frame.frame_index), dtype=np.int64)
            for frame in detections
        ]
    )
    coordinates = np.concatenate(
        [np.asarray(frame.coords_zyx, dtype=np.float32) for frame in detections],
        axis=0,
    )
    views = coordinate_views(
        plan,
        timepoints,
        coordinates,
        nonnegative_candidate_storage=True,
    )
    return (
        transforms,
        np.asarray(views.common_offset_raw_zyx, dtype=np.int64),
    )


def translated_frames(
    cache: bh_model.SparseInferenceCache,
    transitions: Sequence[Mapping[str, Any]],
    *,
    plan: RegistrationPlan | None = None,
) -> tuple[bh_model.FrameDet, ...]:
    if plan is None:
        transforms, common_offset = cumulative_frame_transforms(
            transitions, cache.detections
        )
    else:
        transforms = {
            frame: np.asarray(plan.cumulative_shift_raw_zyx[frame], dtype=np.int64)
            for frame in range(plan.frame_count)
        }
        if set(transforms) != {
            int(frame.frame_index) for frame in cache.detections
        }:
            raise RepairScoreError("registration plan/cache coverage disagrees")
        timepoints = np.concatenate(
            [
                np.full(len(frame.node_ids), int(frame.frame_index), dtype=np.int64)
                for frame in cache.detections
            ]
        )
        coordinates = np.concatenate(
            [
                np.asarray(frame.coords_zyx, dtype=np.float32)
                for frame in cache.detections
            ],
            axis=0,
        )
        views = coordinate_views(
            plan,
            timepoints,
            coordinates,
            nonnegative_candidate_storage=True,
        )
        common_offset = np.asarray(
            views.common_offset_raw_zyx, dtype=np.int64
        )
    return tuple(
        replace(
            frame,
            coords_zyx=(
                np.asarray(frame.coords_zyx, dtype=np.int64)
                + transforms[int(frame.frame_index)][None, :]
                + common_offset[None, :]
            ),
        )
        for frame in cache.detections
    )


def score_pairs(
    cache: bh_model.SparseInferenceCache,
    frames: Sequence[bh_model.FrameDet],
    scorer: bh_edge.SparseEdgeScorer,
) -> tuple[dict[tuple[int, int], float], dict[int, float], dict[int, Any]]:
    by_frame = {int(frame.frame_index): frame for frame in frames}
    probabilities: dict[tuple[int, int], float] = {}
    no_link: dict[int, float] = {}
    scored_by_frame: dict[int, Any] = {}
    for context in cache.contexts:
        source = by_frame[int(context.source_frame)]
        target = by_frame[int(context.target_frame)]
        scored = scorer.score_pair(context, source, target)
        scored_by_frame[int(context.source_frame)] = scored
        for source_id, target_id, probability in zip(
            scored.source_ids, scored.target_ids, scored.probabilities, strict=True
        ):
            probabilities[(int(source_id), int(target_id))] = float(probability)
        for source_id, probability in zip(
            scored.no_link_source_ids, scored.no_link_probabilities, strict=True
        ):
            no_link[int(source_id)] = float(probability)
    return probabilities, no_link, scored_by_frame


def registered_local_edges(
    frames: Sequence[bh_model.FrameDet],
    probabilities: Mapping[tuple[int, int], float],
    no_link: Mapping[int, float],
    flagged: set[int],
) -> dict[int, set[tuple[int, int]]]:
    by_frame = {int(frame.frame_index): frame for frame in frames}
    result: dict[int, set[tuple[int, int]]] = {}
    for frame in sorted(flagged):
        nodes = run_bh.detections_to_graph_nodes(
            (by_frame[frame], by_frame[frame + 1]), bh_graph.GraphNode
        )
        ids = {int(node.node_id) for node in nodes}
        local_probabilities = {
            pair: value
            for pair, value in probabilities.items()
            if pair[0] in ids and pair[1] in ids
        }
        local_no_link = {node_id: no_link[node_id] for node_id in by_frame[frame].node_ids}
        edges, _stats = bh_graph.build_adjacent_edges(
            nodes,
            learned_edge_probabilities=local_probabilities,
            source_no_link_probabilities=local_no_link,
            topology_mode="legacy",
        )
        result[frame] = {
            (int(edge.source_id), int(edge.target_id)) for edge in edges
        }
    return result


def apply_registered(
    rows: Sequence[Mapping[str, str]],
    registered_by_frame: Mapping[int, set[tuple[int, int]]],
    raw_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    edges = edge_pairs(rows)
    times = node_times(rows)
    retained_raw_ids = raw_ids & set(times)
    removed = added = retained_synthetic = skipped_pruned = 0
    for frame, replacements in sorted(registered_by_frame.items()):
        raw_existing = {
            edge
            for edge in edges
            if edge[0] in retained_raw_ids
            and edge[1] in retained_raw_ids
            and times[edge[0]] == frame
            and times[edge[1]] == frame + 1
        }
        edges.difference_update(raw_existing)
        removed += len(raw_existing)
        for source, target in sorted(replacements):
            if source not in times or target not in times:
                skipped_pruned += 1
                continue
            conflicts = {
                edge
                for edge in edges
                if (edge[0] == source or edge[1] == target)
                and times[edge[0]] == frame
                and times[edge[1]] == frame + 1
            }
            retained_synthetic += sum(
                edge[0] not in retained_raw_ids or edge[1] not in retained_raw_ids
                for edge in conflicts
            )
            edges.difference_update(conflicts)
            edges.add((source, target))
            added += 1
    return rows_with_edges(rows, edges), {
        "removed_raw_edges": removed,
        "added_registered_edges": added,
        "removed_conflicting_synthetic_edges": retained_synthetic,
        "skipped_pruned_pairs": skipped_pruned,
    }


def candidate_reachability(
    *,
    cache: bh_model.SparseInferenceCache,
    registered: Sequence[bh_model.FrameDet],
    transitions: Sequence[Mapping[str, Any]],
    gt: Mapping[str, Any],
) -> dict[str, int]:
    matches, gt_edges = match_detections_to_gt(cache.detections, gt)
    original = {int(frame.frame_index): frame for frame in cache.detections}
    corrected = {int(frame.frame_index): frame for frame in registered}
    totals = Counter()
    for row in transitions:
        if not bool(row["pool_breaking"]):
            continue
        frame = int(row["frame"])
        before = bh_edge.gate_sparse_candidates(original[frame], original[frame + 1])
        after = bh_edge.gate_sparse_candidates(corrected[frame], corrected[frame + 1])
        before_pairs = set(zip(before.source_ids.tolist(), before.target_ids.tolist(), strict=True))
        after_pairs = set(zip(after.source_ids.tolist(), after.target_ids.tolist(), strict=True))
        totals["transitions"] += 1
        totals["candidates_before"] += len(before_pairs)
        totals["candidates_after"] += len(after_pairs)
        node_pairs: set[tuple[int, int]] = set()
        for source_id in original[frame].node_ids:
            source_gt = matches[int(source_id)]
            if source_gt is None:
                continue
            for target_id in original[frame + 1].node_ids:
                target_gt = matches[int(target_id)]
                if target_gt is not None and (source_gt, target_gt) in gt_edges:
                    node_pairs.add((int(source_id), int(target_id)))
        totals["matched_gt_continuations"] += len(node_pairs)
        totals["reachable_before"] += len(node_pairs & before_pairs)
        totals["reachable_after"] += len(node_pairs & after_pairs)
    return dict(totals)


def gt_transition_audit(
    *,
    gt: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
    frozen: set[int],
    flagged: set[int],
) -> dict[str, int]:
    """Count official-GT edges incompatible with observed acquisition states."""

    nodes = gt.get("nodes")
    if not isinstance(nodes, Mapping):
        raise RepairScoreError("GT node table is missing")
    ids = np.asarray(nodes["node_id"], dtype=np.int64)
    times = np.asarray(nodes["t"], dtype=np.int64)
    coords = np.column_stack((nodes["z"], nodes["y"], nodes["x"])).astype(
        np.float64, copy=False
    )
    node = {
        int(node_id): (int(frame), coord)
        for node_id, frame, coord in zip(ids, times, coords, strict=True)
    }
    transition_by_frame = {int(row["frame"]): row for row in transitions}
    counts = Counter()
    for source_raw, target_raw in np.asarray(gt["edges"], dtype=np.int64):
        source = int(source_raw)
        target = int(target_raw)
        if source not in node or target not in node:
            raise RepairScoreError("GT edge references missing node")
        source_t, source_coord = node[source]
        target_t, target_coord = node[target]
        if target_t != source_t + 1:
            continue
        displacement = (target_coord - source_coord) * SCALE_ZYX
        distance = float(np.linalg.norm(displacement))
        if source_t in frozen:
            counts["frozen_gt_edges"] += 1
            counts["frozen_gt_edges_exact_zero"] += int(distance == 0.0)
            counts["frozen_gt_edges_nonzero"] += int(distance != 0.0)
            counts["frozen_gt_edges_over_10um"] += int(distance > 10.0)
        if source_t in flagged:
            shift = np.asarray(
                transition_by_frame[source_t]["shift_um_zyx"], dtype=np.float64
            )
            corrected_distance = float(np.linalg.norm(displacement + shift))
            counts["jump_gt_edges"] += 1
            counts["jump_gt_edges_raw_over_10um"] += int(distance > 10.0)
            counts["jump_gt_edges_corrected_inside_10um"] += int(
                corrected_distance <= 10.0
            )
            counts["jump_gt_edges_rescued_inside_10um"] += int(
                distance > 10.0 and corrected_distance <= 10.0
            )
    return dict(counts)


def shell_crosscheck(census: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    distance_before: list[float] = []
    distance_after: list[float] = []
    for path in sorted(SHELL_ROOT.glob("*.pt")):
        identity = path.stem.replace("__", ":", 1)
        if identity not in census:
            raise RepairScoreError(f"shell identity absent from census: {identity}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        labels = np.asarray(payload["labels"], dtype=np.int8)
        pairs = np.asarray(payload["pairs"], dtype=np.int64)
        features = np.asarray(payload["features"], dtype=np.float64)
        positive = np.flatnonzero(labels == 1)
        if len(positive) == 0:
            continue
        cache = load_cache(FIT_CACHE, identity)
        node_frame = {
            int(node_id): int(frame.frame_index)
            for frame in cache.detections
            for node_id in frame.node_ids
        }
        transitions = {
            int(row["frame"]): row for row in census[identity]["transitions"]
        }
        for index in positive:
            source = int(pairs[index, 0])
            frame = node_frame[source]
            row = transitions[frame]
            displacement = features[index, 2:5]
            applied_shift = (
                np.asarray(row["shift_um_zyx"], dtype=np.float64)
                if bool(row["pool_breaking"])
                else np.zeros(3, dtype=np.float64)
            )
            corrected = displacement + applied_shift
            before = float(np.linalg.norm(displacement))
            after = float(np.linalg.norm(corrected))
            distance_before.append(before)
            distance_after.append(after)
            counts["positive_edges"] += 1
            counts["pool_breaking_transition"] += int(bool(row["pool_breaking"]))
            counts["non_pool_breaking_transition"] += int(not bool(row["pool_breaking"]))
            counts["inside_10um_after_preregistered_correction"] += int(after <= 10.0)
            counts["rescued_inside_10um_on_pool_breaking"] += int(
                after <= 10.0 and bool(row["pool_breaking"])
            )
    if counts["positive_edges"] != 456:
        raise RepairScoreError(f"GA-R12REACH positive count drifted: {counts}")
    return {
        "counts": dict(counts),
        "distance_um": {
            "before": {
                "minimum": min(distance_before),
                "median": float(np.median(distance_before)),
                "maximum": max(distance_before),
            },
            "after": {
                "minimum": min(distance_after),
                "median": float(np.median(distance_after)),
                "maximum": max(distance_after),
            },
        },
        "population_note": "fit-only117 bank; not exact CV57",
    }


def score_fragment_dir(
    zip_path: Path, fragment_dir: Path, selected: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(fragment_dir)), selected
    )
    with ZipStore(zip_path, mode="r") as store:
        rows, provenance, coverage = scoreall.score_fragment_input(store, manifest)
    by_identity = {str(row["identity"]): row for row in rows}
    ordered = [by_identity[value] for value in selected]
    summary = scoreall.official_summary(ordered)
    counts = {
        key: int(sum(int(row[key]) for row in ordered))
        for key in (
            "edge_tp",
            "edge_fp",
            "edge_fn",
            "division_tp",
            "division_fp",
            "division_fn",
        )
    }
    return summary, {
        "aggregate_sha256": provenance["aggregate_sha256"],
        "coverage": coverage,
        "counts": counts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise RepairScoreError(f"refusing to overwrite output: {args.output}")
    started = time.perf_counter()
    selected = selected_cv57(args)
    census = census_by_identity(args.census)
    if not set(selected).issubset(census):
        raise RepairScoreError("CV57 is not fully covered by pathology census")
    args.output.mkdir(parents=True)
    fragment_roots = {
        "frozen_f1": args.output / "frozen_f1" / "fragments",
        "registered_g1": args.output / "registered_g1" / "fragments",
        "combined_f1_g1": args.output / "combined_f1_g1" / "fragments",
    }
    for root in fragment_roots.values():
        root.mkdir(parents=True)

    base_summary, base_provenance = score_fragment_dir(args.zip, args.base_fragments, selected)
    expected_base = float(load_json(args.base_report)["candidate"]["summary"]["adj_edge_jaccard"])
    if float(base_summary["adj_edge_jaccard"]) != expected_base:
        raise RepairScoreError("Prompt-1 base failed exact reproduction")

    model, bundle = bh_model.load_v45_model(args.support_root, device="cpu")
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sha256(args.edge_artifact),
        args.edge_auxiliary,
        model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None:
        raise RepairScoreError("frozen edge scorer did not restore")

    totals: dict[str, Counter[str]] = {
        "frozen_f1": Counter(),
        "registered_g1": Counter(),
        "combined_f1_g1": Counter(),
    }
    reachability = Counter()
    gt_audit = Counter()
    video_rows: list[dict[str, Any]] = []
    store: ZipStore | None = None
    try:
        for ordinal, identity in enumerate(selected, start=1):
            source_path = fragment_path(args.base_fragments, identity)
            base_rows = read_fragment(source_path)
            video = census[identity]
            frozen = {int(value) for value in video["frozen_transitions"]}
            flagged = {
                int(row["frame"])
                for row in video["transitions"]
                if bool(row["pool_breaking"])
            }
            if not frozen and not flagged:
                for root in fragment_roots.values():
                    shutil.copy2(source_path, fragment_path(root, identity))
                video_rows.append(
                    {"identity": identity, "frozen_transitions": 0, "pool_breaking_transitions": 0}
                )
                print(f"repair {ordinal:02d}/57 {identity} identity", flush=True)
                continue
            cache = load_cache(args.cache, identity)
            frozen_pairs = exact_coordinate_pairs(cache, frozen)
            f1_rows, f1_counts = apply_frozen(base_rows, frozen_pairs)
            for key, value in f1_counts.items():
                totals["frozen_f1"][key] += value

            if flagged:
                registration_plan = load_registration_plan(args.census, identity)
                registered = translated_frames(
                    cache,
                    video["transitions"],
                    plan=registration_plan,
                )
                probabilities, no_link, scored = score_pairs(cache, registered, scorer)
                replacements = registered_local_edges(
                    registered, probabilities, no_link, flagged
                )
                raw_ids = {
                    int(node_id) for frame in cache.detections for node_id in frame.node_ids
                }
                g1_rows, g1_counts = apply_registered(
                    base_rows, replacements, raw_ids
                )
                combined_before_f1, combined_g1_counts = apply_registered(
                    base_rows, replacements, raw_ids
                )
                combined_rows, combined_f1_counts = apply_frozen(
                    combined_before_f1, frozen_pairs
                )
                for key, value in g1_counts.items():
                    totals["registered_g1"][key] += value
                    totals["combined_f1_g1"][key] += combined_g1_counts[key]
                for key, value in combined_f1_counts.items():
                    totals["combined_f1_g1"][key] += value
                candidate_before = sum(
                    len(bh_edge.gate_sparse_candidates(cache.detections[frame], cache.detections[frame + 1]))
                    for frame in flagged
                )
                registered_by_frame = {int(frame.frame_index): frame for frame in registered}
                candidate_after = sum(
                    len(bh_edge.gate_sparse_candidates(registered_by_frame[frame], registered_by_frame[frame + 1]))
                    for frame in flagged
                )
                dataset = identity.split(":", 1)[1]
                if store is None:
                    store = ZipStore(args.zip, mode="r")
                gt = read_geff_arrays(store, "train", dataset)
                reached = candidate_reachability(
                    cache=cache,
                    registered=registered,
                    transitions=video["transitions"],
                    gt=gt,
                )
                reachability.update(reached)
                gt_audit.update(
                    gt_transition_audit(
                        gt=gt,
                        transitions=video["transitions"],
                        frozen=frozen,
                        flagged=flagged,
                    )
                )
                g1_detail = {
                    **g1_counts,
                    "candidate_count_before": candidate_before,
                    "candidate_count_after": candidate_after,
                    "scored_candidate_count_after": sum(
                        int(scored[frame].stats.candidate_count) for frame in flagged
                    ),
                    "reachability": reached,
                }
            else:
                g1_rows = [dict(row) for row in base_rows]
                combined_rows = f1_rows
                g1_detail = {
                    "candidate_count_before": 0,
                    "candidate_count_after": 0,
                    "scored_candidate_count_after": 0,
                    "reachability": {},
                }
                if frozen:
                    dataset = identity.split(":", 1)[1]
                    if store is None:
                        store = ZipStore(args.zip, mode="r")
                    gt = read_geff_arrays(store, "train", dataset)
                    gt_audit.update(
                        gt_transition_audit(
                            gt=gt,
                            transitions=video["transitions"],
                            frozen=frozen,
                            flagged=flagged,
                        )
                    )
            write_fragment(fragment_path(fragment_roots["frozen_f1"], identity), f1_rows)
            write_fragment(fragment_path(fragment_roots["registered_g1"], identity), g1_rows)
            write_fragment(fragment_path(fragment_roots["combined_f1_g1"], identity), combined_rows)
            video_rows.append(
                {
                    "identity": identity,
                    "frozen_transitions": len(frozen),
                    "frozen_exact_pairs": sum(len(value) for value in frozen_pairs.values()),
                    "pool_breaking_transitions": len(flagged),
                    "g1": g1_detail,
                }
            )
            print(
                f"repair {ordinal:02d}/57 {identity} frozen={len(frozen)} jump={len(flagged)}",
                flush=True,
            )
    finally:
        if store is not None and hasattr(store, "_lock"):
            store.close()

    scores: dict[str, Any] = {
        "base": {"summary": base_summary, **base_provenance}
    }
    for name, root in fragment_roots.items():
        summary, provenance = score_fragment_dir(args.zip, root, selected)
        scores[name] = {
            "summary": summary,
            **provenance,
            "delta_adjusted_edge_jaccard": (
                float(summary["adj_edge_jaccard"])
                - float(base_summary["adj_edge_jaccard"])
            ),
        }
    shell = shell_crosscheck(census)
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "mechanism": {
            "frozen": "exact-coordinate-one-to-one-on-byte-confirmed-frozen-transitions",
            "registration": (
                "whole-frame-phase-transform-before-10um-cap8-candidates; "
                "pair associations locally re-solved by unchanged scorer/legacy graph matcher"
            ),
            "a7_reopened": False,
            "final_node_coordinates": "original APNCC fragment bytes",
        },
        "selection": {"count": len(selected), "identities": list(selected)},
        "scores": scores,
        "candidate_reachability_pool_breaking_cv57": dict(reachability),
        "gt_annotation_audit_cv57": dict(gt_audit),
        "ga_r12reach_crosscheck": shell,
        "totals": {name: dict(value) for name, value in totals.items()},
        "videos": video_rows,
        "provenance": {
            "census": str(args.census),
            "census_sha256": sha256(args.census),
            "base_report": str(args.base_report),
            "base_report_sha256": sha256(args.base_report),
            "edge_artifact": edge_record,
        },
        "execution": {
            "device": "cpu",
            "gpu_action": False,
            "kaggle_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    atomic_json(args.output / "REPORT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    value.add_argument("--base-fragments", type=Path, default=DEFAULT_BASE)
    value.add_argument("--base-report", type=Path, default=DEFAULT_BASE_REPORT)
    value.add_argument("--zip", type=Path, default=sweep.DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=sweep.DEFAULT_SPLIT)
    value.add_argument("--cache", type=Path, default=sweep.DEFAULT_CACHE)
    value.add_argument("--edge-artifact", type=Path, default=sweep.DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=sweep.DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=sweep.DEFAULT_SUPPORT)
    value.add_argument("--scorer-parity", type=Path, default=sweep.DEFAULT_PARITY)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps({"status": report["status"], "scores": report["scores"]}, sort_keys=True))
