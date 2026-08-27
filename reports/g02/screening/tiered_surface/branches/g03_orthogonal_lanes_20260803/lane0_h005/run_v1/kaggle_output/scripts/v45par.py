#!/usr/bin/env python3
"""Fail-closed replay parity harness for the pinned public V45 run."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import resource
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zarr
from zarr.storage import ZipStore

from celltrack_alpha.bh_graph import (
    PHYSICAL_SCALE_ZYX,
    V45_GRAPH_CONFIG,
    GraphNode,
    GraphResult,
    calibrate_v45_graph,
)


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_ZIP = ROOT / "reports/g02/v45out.zip"
EXPECTED_SUBMISSION = ROOT / "reports/g02/v45sub.csv"
EXPECTED_RUN = ROOT / "reports/g02/v45run.csv"
IMAGE_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
REPORT_PATHS = {
    "sparse": ROOT / "reports/g02/v45par.json",
    "dense_diagnostic": ROOT / "reports/g02/v45diag.json",
}
REPORT_PATH = REPORT_PATHS["sparse"]

PINNED_SHA256 = {
    PREDICTION_ZIP: "75304b22389a81f28e9616ff7c63d6d200ef50f7d4348ee68818d4fdcbc97994",
    EXPECTED_SUBMISSION: "6f8d81f6e5d9eaa066e47f0cdd2d22df48eaa74853d2940d0bc957f3ef3a19e1",
    EXPECTED_RUN: "2c2c5e6b345a24426ba28c4ab68c544048ab2a07aa503b0319ac6481d267f090",
}

DATASETS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)
PREDICTION_ROOT = "tracking_repo/predictions/unknown/unet_transformer/split_0"
SUBMISSION_COLUMNS = (
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
RUN_COLUMNS = (
    "dataset",
    "raw_nodes",
    "nodes",
    "raw_edges",
    "edges",
    "division_like_sources",
    "edge_to_node_ratio",
    "gap_added_nodes_frac",
    "dropped_nonconsecutive_edges",
    "dropped_long_edges",
    "dropped_multi_parent_edges",
    "dropped_multi_child_edges",
    "dropped_division_edges",
    "gap_candidates",
    "gap_pairs_selected",
    "gap_reused_existing",
    "gap_inserted_synthetic",
    "gap_added_nodes",
    "gap_added_edges",
    "gap_skipped_node_cap",
    "gap_density_nodes_scored",
    "gap_density_candidates_expanded",
    "gap_density_candidates_restricted",
    "gap_density_selected_outside_base",
    "gap_density_step_delta_milli_sum",
    "variant_candidates_added_vs_090",
    "variant_candidates_removed_vs_090",
    "variant_selected_outside_090",
    "variant_selected_inside_090",
    "variant_threshold_delta_milli_sum",
    "variant_threshold_delta_abs_milli_sum",
    "gap_refined_synthetic",
    "gap_refine_failed",
    "gap_refine_rejected_shift",
    "pruned_isolated_nodes",
    "motion_relink_edges",
    "motion_relink_tight_edges",
    "motion_relink_relaxed_edges",
    "motion_relink_frames",
    "motion_relink_replaced_raw_edges",
    "motion_relink_fallback_raw",
    "motion_relink_skipped_large_frame",
    "gap2_candidates",
    "gap2_pairs_selected",
    "gap2_added_nodes",
    "gap2_added_edges",
    "gap2_skipped_cap",
    "safe_division_candidates",
    "safe_divisions_added",
    "safe_division_skipped_cap",
    "deepcenter_gap_checked",
    "deepcenter_gap_bypassed_strong_motion",
    "deepcenter_gap_bypassed_synthetic_node",
    "deepcenter_gap_accepted",
    "deepcenter_gap_rejected",
    "deepcenter_gap_missing",
    "deepcenter_safe_div_checked",
    "deepcenter_safe_div_accepted",
    "deepcenter_safe_div_rejected",
    "deepcenter_safe_div_missing",
    "short_track_components_removed",
    "short_track_nodes_removed",
    "short_track_edges_removed",
    "short_track_filter_skipped_all",
    "short_track_rescue_triggered",
    "short_track_rescue_components",
    "short_track_rescue_nodes",
    "short_track_rescue_budget",
    "linefit_smoothed_nodes",
    "linefit_skipped_nodes",
    "gap_close_effective_max_gap",
    "predict_minutes_total",
    "experiment_tag",
)

# These notebook counters have no like-for-like value in GraphResult.stats.
# They remain visible in each report as pinned reference values, but are not
# silently equated to a differently defined sparse-rewrite statistic.
REFERENCE_ONLY_COUNTERS = (
    "gap_density_nodes_scored",
    "gap_density_candidates_expanded",
    "gap_density_candidates_restricted",
    "gap_density_selected_outside_base",
    "gap_density_step_delta_milli_sum",
    "variant_candidates_added_vs_090",
    "variant_candidates_removed_vs_090",
    "variant_selected_outside_090",
    "variant_selected_inside_090",
    "variant_threshold_delta_milli_sum",
    "variant_threshold_delta_abs_milli_sum",
    "safe_division_skipped_cap",
    "predict_minutes_total",
    "experiment_tag",
)


class V45ParityError(RuntimeError):
    """The pinned replay input or comparison is unsafe."""


@dataclass(frozen=True)
class RawGraph:
    nodes: tuple[GraphNode, ...]
    learned_probabilities: Mapping[tuple[int, int], float]
    raw_edge_count: int
    dropped_long_edges: int
    adjacent_frame_pairs: int
    array_bytes: int


@dataclass(frozen=True)
class ExpectedGraph:
    nodes: frozenset[tuple[int, int, int, int, int]]
    edges: frozenset[tuple[int, int]]


def sha256_file(path: Path, *, block_bytes: int = 1024 * 1024) -> str:
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(block_bytes), b""):
                digest.update(block)
    except OSError as exc:
        raise V45ParityError(f"cannot read pinned input {path}: {exc}") from exc
    return digest.hexdigest()


def validate_hashes(expected: Mapping[Path, str]) -> dict[str, str]:
    """Hash every supplied path and reject any missing or changed byte."""

    observed: dict[str, str] = {}
    for path, wanted in expected.items():
        if not path.is_file():
            raise V45ParityError(f"missing pinned input: {path}")
        actual = sha256_file(path)
        observed[str(path)] = actual
        if actual != wanted:
            raise V45ParityError(
                f"pinned SHA256 mismatch for {path}: expected {wanted}, observed {actual}"
            )
    return observed


def _datasets_in_zip(path: Path, prefix: str, suffix: str) -> tuple[str, ...]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise V45ParityError(f"cannot inspect ZIP {path}: {exc}") from exc
    datasets = {
        name[len(prefix) : -len(suffix)]
        for name in names
        if name.startswith(prefix) and name.endswith(suffix)
        and "/" not in name[len(prefix) : -len(suffix)]
    }
    return tuple(sorted(datasets))


def validate_archive_roots(prediction_zip: Path, image_zip: Path) -> None:
    prediction_datasets = _datasets_in_zip(
        prediction_zip, f"{PREDICTION_ROOT}/", ".geff/zarr.json"
    )
    image_datasets = _datasets_in_zip(image_zip, "test/", ".zarr/zarr.json")
    if prediction_datasets != DATASETS:
        raise V45ParityError(
            f"prediction GEFF roots differ: expected {DATASETS}, observed {prediction_datasets}"
        )
    if image_datasets != DATASETS:
        raise V45ParityError(
            f"test image roots differ: expected {DATASETS}, observed {image_datasets}"
        )


def _array(group: zarr.Group, path: str) -> np.ndarray:
    try:
        return np.asarray(group[path][:])
    except Exception as exc:
        raise V45ParityError(f"cannot read GEFF array {path}: {exc}") from exc


def _require_integer_array(array: np.ndarray, name: str) -> None:
    if array.dtype.kind not in "iu":
        raise V45ParityError(f"{name} must have an integer dtype, observed {array.dtype}")


def load_raw_geff(
    store: ZipStore,
    dataset: str,
    *,
    prediction_root: str = PREDICTION_ROOT,
) -> RawGraph:
    """Load one raw V45 GEFF root and reject malformed learned-edge input."""

    path = f"{prediction_root}/{dataset}.geff"
    try:
        group = zarr.open_group(store=store, path=path, mode="r")
    except Exception as exc:
        raise V45ParityError(f"cannot open GEFF root {path}: {exc}") from exc

    ids = _array(group, "nodes/ids")
    times = _array(group, "nodes/props/t/values")
    z_pos = _array(group, "nodes/props/z/values")
    y_pos = _array(group, "nodes/props/y/values")
    x_pos = _array(group, "nodes/props/x/values")
    edge_ids = _array(group, "edges/ids")
    probabilities = _array(group, "edges/props/edge_prob/values")
    arrays = (ids, times, z_pos, y_pos, x_pos, edge_ids, probabilities)

    if ids.ndim != 1 or not ids.size:
        raise V45ParityError(f"{dataset}: nodes/ids must be a nonempty vector")
    node_count = int(ids.size)
    for name, array in (
        ("t", times),
        ("z", z_pos),
        ("y", y_pos),
        ("x", x_pos),
    ):
        if array.shape != (node_count,):
            raise V45ParityError(
                f"{dataset}: node {name} shape {array.shape} != {(node_count,)}"
            )
    _require_integer_array(ids, "nodes/ids")
    _require_integer_array(times, "nodes/props/t/values")
    if np.any(ids < 0) or np.any(times < 0):
        raise V45ParityError(f"{dataset}: negative node ID or time")
    if np.unique(ids).size != ids.size:
        raise V45ParityError(f"{dataset}: duplicate raw node ID")
    for name, array in (("z", z_pos), ("y", y_pos), ("x", x_pos)):
        if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise V45ParityError(f"{dataset}: nonfinite or nonnumeric node {name}")

    if edge_ids.ndim != 2 or edge_ids.shape[1:] != (2,):
        raise V45ParityError(f"{dataset}: edges/ids must have shape (N, 2)")
    _require_integer_array(edge_ids, "edges/ids")
    edge_count = int(edge_ids.shape[0])
    if probabilities.shape != (edge_count,):
        raise V45ParityError(
            f"{dataset}: edge probability shape {probabilities.shape} != {(edge_count,)}"
        )
    if probabilities.dtype.kind not in "fiu" or not np.all(np.isfinite(probabilities)):
        raise V45ParityError(f"{dataset}: nonfinite or nonnumeric edge probability")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise V45ParityError(f"{dataset}: edge probability outside [0, 1]")

    node_ids = [int(value) for value in ids]
    index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    learned: dict[tuple[int, int], float] = {}
    long_edges = 0
    scale = np.asarray(PHYSICAL_SCALE_ZYX, dtype=np.float64)
    coordinates = np.stack((z_pos, y_pos, x_pos), axis=1).astype(np.float64, copy=False)
    for row, probability in zip(edge_ids, probabilities, strict=True):
        source_id, target_id = int(row[0]), int(row[1])
        pair = (source_id, target_id)
        if pair in learned:
            raise V45ParityError(f"{dataset}: duplicate raw edge {pair}")
        if source_id not in index_by_id or target_id not in index_by_id:
            raise V45ParityError(f"{dataset}: raw edge {pair} has an unknown endpoint")
        source_index = index_by_id[source_id]
        target_index = index_by_id[target_id]
        if int(times[target_index]) != int(times[source_index]) + 1:
            raise V45ParityError(f"{dataset}: raw edge {pair} is not adjacent and forward")
        learned[pair] = float(probability)
        distance = float(
            np.linalg.norm((coordinates[source_index] - coordinates[target_index]) * scale)
        )
        if distance > V45_GRAPH_CONFIG.output_edge_max_um:
            long_edges += 1

    nodes = tuple(
        GraphNode(int(node_id), int(t), float(z), float(y), float(x))
        for node_id, t, z, y, x in zip(
            ids, times, z_pos, y_pos, x_pos, strict=True
        )
    )
    present_frames = {int(value) for value in times}
    frame_pairs = sum(1 for frame in present_frames if frame + 1 in present_frames)
    return RawGraph(
        nodes=nodes,
        learned_probabilities=learned,
        raw_edge_count=edge_count,
        dropped_long_edges=long_edges,
        adjacent_frame_pairs=frame_pairs,
        array_bytes=sum(int(array.nbytes) for array in arrays),
    )


def load_expected_submission(path: Path, dataset: str) -> ExpectedGraph:
    nodes: set[tuple[int, int, int, int, int]] = set()
    edges: set[tuple[int, int]] = set()
    seen_datasets: set[str] = set()
    expected_row_id = 0
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise V45ParityError(f"cannot read expected submission {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
            raise V45ParityError(f"unexpected submission header: {reader.fieldnames}")
        for row in reader:
            try:
                row_id = int(row["id"])
            except (TypeError, ValueError) as exc:
                raise V45ParityError("submission row has a noninteger id") from exc
            if row_id != expected_row_id:
                raise V45ParityError(
                    f"submission row id {row_id} is not contiguous at {expected_row_id}"
                )
            expected_row_id += 1
            row_dataset = str(row["dataset"])
            if row_dataset not in DATASETS:
                raise V45ParityError(f"unexpected submission dataset {row_dataset!r}")
            seen_datasets.add(row_dataset)
            row_type = row["row_type"]
            try:
                if row_type == "node":
                    values = tuple(
                        int(row[name]) for name in ("node_id", "t", "z", "y", "x")
                    )
                    if any(value < 0 for value in values):
                        raise V45ParityError("submission node values must be nonnegative")
                    if int(row["source_id"]) != -1 or int(row["target_id"]) != -1:
                        raise V45ParityError("submission node placeholders must be -1")
                    if row_dataset == dataset:
                        if values in nodes:
                            raise V45ParityError(f"duplicate expected node row {values}")
                        nodes.add(values)
                elif row_type == "edge":
                    if any(int(row[name]) != -1 for name in ("node_id", "t", "z", "y", "x")):
                        raise V45ParityError("submission edge placeholders must be -1")
                    pair = (int(row["source_id"]), int(row["target_id"]))
                    if pair[0] < 0 or pair[1] < 0:
                        raise V45ParityError("submission edge IDs must be nonnegative")
                    if row_dataset == dataset:
                        if pair in edges:
                            raise V45ParityError(f"duplicate expected edge row {pair}")
                        edges.add(pair)
                else:
                    raise V45ParityError(f"unexpected row_type {row_type!r}")
            except (TypeError, ValueError) as exc:
                raise V45ParityError(f"malformed submission row {row_id}: {exc}") from exc
    if tuple(sorted(seen_datasets)) != DATASETS:
        raise V45ParityError(
            f"submission datasets differ: expected {DATASETS}, observed {tuple(sorted(seen_datasets))}"
        )
    if not nodes:
        raise V45ParityError(f"expected submission has no nodes for {dataset}")
    by_id = {node[0]: node for node in nodes}
    if len(by_id) != len(nodes):
        raise V45ParityError(f"expected submission repeats a node ID for {dataset}")
    for source_id, target_id in edges:
        if source_id not in by_id or target_id not in by_id:
            raise V45ParityError(f"expected edge {(source_id, target_id)} is dangling")
        if by_id[target_id][1] != by_id[source_id][1] + 1:
            raise V45ParityError(f"expected edge {(source_id, target_id)} is not adjacent")
    return ExpectedGraph(frozenset(nodes), frozenset(edges))


def load_expected_run(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise V45ParityError(f"cannot read expected run CSV {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RUN_COLUMNS:
            raise V45ParityError(f"unexpected run CSV header: {reader.fieldnames}")
        for row in reader:
            dataset = str(row["dataset"])
            if dataset not in DATASETS or dataset in rows:
                raise V45ParityError(f"unexpected or duplicate run row {dataset!r}")
            rows[dataset] = dict(row)
    if tuple(sorted(rows)) != DATASETS:
        raise V45ParityError(
            f"run CSV datasets differ: expected {DATASETS}, observed {tuple(sorted(rows))}"
        )
    return rows


class LazyFrameCache:
    """One-frame image cache; V45 requests frames in monotonically sorted order."""

    def __init__(self, image_array: zarr.Array) -> None:
        if image_array.ndim != 4 or any(int(size) <= 0 for size in image_array.shape):
            raise V45ParityError(
                f"test image array must have positive t/z/y/x shape, observed {image_array.shape}"
            )
        self.image_array = image_array
        self._frame_index: int | None = None
        self._frame: np.ndarray | None = None
        self.read_count = 0
        self.requested_frames: set[int] = set()
        self.peak_cached_bytes = 0

    def __call__(self, frame: int) -> np.ndarray:
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
            raise V45ParityError(f"image frame must be an integer, observed {frame!r}")
        index = int(frame)
        if not 0 <= index < int(self.image_array.shape[0]):
            raise V45ParityError(f"image frame {index} is outside {self.image_array.shape[0]}")
        self.requested_frames.add(index)
        if index != self._frame_index:
            try:
                loaded = np.asarray(self.image_array[index])
            except Exception as exc:
                raise V45ParityError(f"cannot lazily read test frame {index}: {exc}") from exc
            if loaded.ndim != 3 or loaded.shape != self.image_array.shape[1:]:
                raise V45ParityError(
                    f"test frame {index} shape {loaded.shape} != {self.image_array.shape[1:]}"
                )
            if loaded.dtype.kind not in "fiu" or not np.all(np.isfinite(loaded)):
                raise V45ParityError(f"test frame {index} is nonnumeric or nonfinite")
            self._frame_index = index
            self._frame = loaded
            self.read_count += 1
            self.peak_cached_bytes = max(self.peak_cached_bytes, int(loaded.nbytes))
        assert self._frame is not None
        return self._frame

    def release(self) -> None:
        self._frame_index = None
        self._frame = None


def public_node_tuple(node: GraphNode) -> tuple[int, int, int, int, int]:
    """Apply the notebook's only position rounding, at public CSV comparison."""

    return (
        int(node.node_id),
        int(node.t),
        max(0, int(round(float(node.z)))),
        max(0, int(round(float(node.y)))),
        max(0, int(round(float(node.x)))),
    )


def _division_sources(result: GraphResult) -> int:
    counts = Counter(edge.source_id for edge in result.edges)
    return sum(count >= 2 for count in counts.values())


def replay_counters(raw: RawGraph, result: GraphResult) -> dict[str, int | float]:
    """Translate only counters with identical public-notebook semantics."""

    stats = result.stats
    node_count = len(result.nodes)
    edge_count = len(result.edges)
    synth = int(stats["gap_synthesized_nodes"])
    gap_matches = int(stats["gap_matches"])
    short_nodes = int(stats["short_component_nodes_pruned"])
    short_components = int(stats["short_components_pruned"])
    return {
        "raw_nodes": len(raw.nodes),
        "nodes": node_count,
        "raw_edges": raw.raw_edge_count,
        "edges": edge_count,
        "division_like_sources": _division_sources(result),
        "edge_to_node_ratio": edge_count / max(node_count, 1),
        "gap_added_nodes_frac": synth / max(len(raw.nodes), 1),
        "dropped_nonconsecutive_edges": 0,
        "dropped_long_edges": raw.dropped_long_edges,
        "dropped_multi_parent_edges": 0,
        "dropped_multi_child_edges": 0,
        "dropped_division_edges": 0,
        "gap_candidates": int(stats["gap_candidate_pairs"]),
        "gap_pairs_selected": gap_matches,
        "gap_reused_existing": int(stats["gap_reused_nodes"]),
        "gap_inserted_synthetic": synth,
        "gap_added_nodes": synth,
        "gap_added_edges": 2 * gap_matches,
        "gap_skipped_node_cap": int(stats["gap_frame_cap_rejected"]),
        "gap_refined_synthetic": int(stats["gap_refinement_accepted"]),
        "gap_refine_failed": 0,
        "gap_refine_rejected_shift": int(stats["gap_refinement_rejected"]),
        "pruned_isolated_nodes": int(stats["isolated_nodes_pruned"]),
        "motion_relink_edges": int(stats["adjacent_edges"]),
        "motion_relink_tight_edges": int(stats["motion_tight_edges"]),
        "motion_relink_relaxed_edges": int(stats["motion_relaxed_edges"]),
        "motion_relink_frames": raw.adjacent_frame_pairs,
        "motion_relink_replaced_raw_edges": raw.raw_edge_count - raw.dropped_long_edges,
        "motion_relink_fallback_raw": 0,
        "motion_relink_skipped_large_frame": 0,
        "gap2_candidates": 0,
        "gap2_pairs_selected": 0,
        "gap2_added_nodes": 0,
        "gap2_added_edges": 0,
        "gap2_skipped_cap": 0,
        "safe_division_candidates": int(stats["division_candidate_count"]),
        "safe_divisions_added": int(stats["division_added"]),
        "deepcenter_gap_checked": 0,
        "deepcenter_gap_bypassed_strong_motion": 0,
        "deepcenter_gap_bypassed_synthetic_node": 0,
        "deepcenter_gap_accepted": 0,
        "deepcenter_gap_rejected": 0,
        "deepcenter_gap_missing": 0,
        "deepcenter_safe_div_checked": 0,
        "deepcenter_safe_div_accepted": 0,
        "deepcenter_safe_div_rejected": 0,
        "deepcenter_safe_div_missing": 0,
        "short_track_components_removed": short_components,
        "short_track_nodes_removed": short_nodes,
        "short_track_edges_removed": short_nodes - short_components,
        "short_track_filter_skipped_all": int(stats["short_component_filter_skipped_all"]),
        "short_track_rescue_triggered": 0,
        "short_track_rescue_components": 0,
        "short_track_rescue_nodes": 0,
        "short_track_rescue_budget": 0,
        "linefit_smoothed_nodes": int(stats["smoothed_nodes"]),
        "linefit_skipped_nodes": node_count - int(stats["smoothed_nodes"]),
        "gap_close_effective_max_gap": int(V45_GRAPH_CONFIG.gap_missing_frames),
    }


def compare_counters(
    actual: Mapping[str, int | float], expected_row: Mapping[str, str]
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    compared: dict[str, dict[str, int | float]] = {}
    for name in sorted(actual):
        if name not in expected_row:
            raise V45ParityError(f"run CSV lacks replay counter {name}")
        value = actual[name]
        try:
            expected: int | float
            expected = float(expected_row[name]) if isinstance(value, float) else int(expected_row[name])
        except (TypeError, ValueError) as exc:
            raise V45ParityError(
                f"run counter {name} is not numeric: {expected_row[name]!r}"
            ) from exc
        compared[name] = {"actual": value, "expected": expected}
        if value != expected:
            mismatches.append({"counter": name, "actual": value, "expected": expected})
    return {
        "compared": compared,
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "reference_only": {
            name: expected_row[name] for name in REFERENCE_ONLY_COUNTERS
        },
    }


def compare_graph(
    result: GraphResult,
    expected: ExpectedGraph,
) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_nodes = frozenset(public_node_tuple(node) for node in result.nodes)
    actual_edges = frozenset(
        (int(edge.source_id), int(edge.target_id)) for edge in result.edges
    )
    if len(actual_nodes) != len(result.nodes):
        raise V45ParityError("public node serialization produced duplicate records")
    if len(actual_edges) != len(result.edges):
        raise V45ParityError("calibration produced duplicate directed edges")
    missing_nodes = sorted(expected.nodes - actual_nodes)
    extra_nodes = sorted(actual_nodes - expected.nodes)
    missing_edges = sorted(expected.edges - actual_edges)
    extra_edges = sorted(actual_edges - expected.edges)
    return (
        {
            "actual_count": len(actual_nodes),
            "expected_count": len(expected.nodes),
            "missing": [list(row) for row in missing_nodes],
            "missing_count": len(missing_nodes),
            "extra": [list(row) for row in extra_nodes],
            "extra_count": len(extra_nodes),
        },
        {
            "actual_count": len(actual_edges),
            "expected_count": len(expected.edges),
            "missing": [list(row) for row in missing_edges],
            "missing_count": len(missing_edges),
            "extra": [list(row) for row in extra_edges],
            "extra_count": len(extra_edges),
        },
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            _plain(value),
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def write_report(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def _open_image_array(store: ZipStore, dataset: str) -> zarr.Array:
    path = f"test/{dataset}.zarr/0"
    try:
        array = zarr.open_array(store=store, path=path, mode="r")
    except Exception as exc:
        raise V45ParityError(f"cannot open test image array {path}: {exc}") from exc
    if array.ndim != 4:
        raise V45ParityError(f"test image array {path} is not t/z/y/x")
    return array


def replay_dataset(
    dataset: str,
    prediction_store: ZipStore,
    image_store: ZipStore,
    expected_run_row: Mapping[str, str],
    *,
    parity_mode: str = "sparse",
) -> dict[str, Any]:
    started = time.perf_counter()
    raw = load_raw_geff(prediction_store, dataset)
    expected = load_expected_submission(EXPECTED_SUBMISSION, dataset)
    image_array = _open_image_array(image_store, dataset)
    if int(image_array.shape[0]) <= max(node.t for node in raw.nodes):
        raise V45ParityError(
            f"{dataset}: image time axis {image_array.shape[0]} does not cover raw nodes"
        )
    frame_cache = LazyFrameCache(image_array)
    calibration_started = time.perf_counter()
    result = calibrate_v45_graph(
        raw.nodes,
        raw.learned_probabilities,
        images=frame_cache,
        max_component_matrix_dim=2 * V45_GRAPH_CONFIG.max_nodes_per_frame,
        parity_mode=parity_mode,
    )
    calibration_seconds = time.perf_counter() - calibration_started
    node_diff, edge_diff = compare_graph(result, expected)
    counter_diff = compare_counters(replay_counters(raw, result), expected_run_row)
    passed = not (
        node_diff["missing_count"]
        or node_diff["extra_count"]
        or edge_diff["missing_count"]
        or edge_diff["extra_count"]
        or counter_diff["mismatch_count"]
    )
    record = {
        "status": "pass" if passed else "fail",
        "raw": {
            "array_bytes": raw.array_bytes,
            "edge_count": raw.raw_edge_count,
            "learned_probability_count": len(raw.learned_probabilities),
            "node_count": len(raw.nodes),
        },
        "nodes": node_diff,
        "edges": edge_diff,
        "run_counters": counter_diff,
        "calibration_stats": _plain(result.stats),
        "calibration_provenance": _plain(result.provenance),
        "resources": {
            "calibration_seconds": round(calibration_seconds, 6),
            "dataset_seconds": round(time.perf_counter() - started, 6),
            "image_frame_reads": frame_cache.read_count,
            "image_frames_requested": sorted(frame_cache.requested_frames),
            "peak_cached_image_bytes": frame_cache.peak_cached_bytes,
        },
    }
    frame_cache.release()
    return record


def run_parity(
    datasets: Sequence[str],
    *,
    parity_mode: str = "sparse",
    report_path: Path | None = None,
) -> dict[str, Any]:
    if parity_mode not in REPORT_PATHS:
        raise V45ParityError(f"unsupported parity mode {parity_mode!r}")
    destination = report_path or REPORT_PATHS[parity_mode]
    started = time.perf_counter()
    hashes = validate_hashes(PINNED_SHA256)
    validate_archive_roots(PREDICTION_ZIP, IMAGE_ZIP)
    run_rows = load_expected_run(EXPECTED_RUN)
    records: dict[str, Any] = {}
    with ZipStore(PREDICTION_ZIP, mode="r") as prediction_store, ZipStore(
        IMAGE_ZIP, mode="r"
    ) as image_store:
        for dataset in datasets:
            records[dataset] = replay_dataset(
                dataset,
                prediction_store,
                image_store,
                run_rows[dataset],
                parity_mode=parity_mode,
            )
            gc.collect()

    passed = all(record["status"] == "pass" for record in records.values())
    totals = {
        "node_missing": sum(record["nodes"]["missing_count"] for record in records.values()),
        "node_extra": sum(record["nodes"]["extra_count"] for record in records.values()),
        "edge_missing": sum(record["edges"]["missing_count"] for record in records.values()),
        "edge_extra": sum(record["edges"]["extra_count"] for record in records.values()),
        "counter_mismatches": sum(
            record["run_counters"]["mismatch_count"] for record in records.values()
        ),
    }
    report = {
        "schema": "celltrack-alpha.v45-replay-parity.v1",
        "status": "pass" if passed else "fail",
        "solver_mode": parity_mode,
        "selection": list(datasets),
        "inputs": {
            "pinned_sha256": hashes,
            "image_zip": {
                "path": str(IMAGE_ZIP),
                "size_bytes": IMAGE_ZIP.stat().st_size,
                "sha256": None,
                "identity_note": "packet supplied no image-archive digest; exact test roots were validated",
            },
        },
        "bounds": {
            "dataset_count": len(datasets),
            "full_frame_dense_matrix": parity_mode == "dense_diagnostic",
            "hash_read_block_bytes": 1024 * 1024,
            "image_cache_frames": 1,
            "max_nodes_per_frame": V45_GRAPH_CONFIG.max_nodes_per_frame,
            "sparse_component_matrix_legacy_argument": 2
            * V45_GRAPH_CONFIG.max_nodes_per_frame,
        },
        "datasets": records,
        "totals": totals,
        "resources": {
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }
    write_report(destination, report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dataset", choices=DATASETS, help="replay exactly one dataset")
    selection.add_argument("--all", action="store_true", help="replay all four pinned datasets")
    parser.add_argument(
        "--solver-mode",
        choices=tuple(REPORT_PATHS),
        default="sparse",
        help="sparse is production; dense_diagnostic is the pinned LAPJV oracle",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = DATASETS if args.all else (args.dataset,)
    report_path = REPORT_PATHS[args.solver_mode]
    try:
        report = run_parity(
            datasets,
            parity_mode=args.solver_mode,
            report_path=report_path,
        )
    except Exception as exc:
        failure = {
            "schema": "celltrack-alpha.v45-replay-parity.v1",
            "status": "error",
            "solver_mode": args.solver_mode,
            "selection": list(datasets),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        write_report(report_path, failure)
        print(f"V45 parity error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    totals = report["totals"]
    print(
        f"V45 parity {report['status']}: datasets={len(datasets)} "
        f"node_missing={totals['node_missing']} node_extra={totals['node_extra']} "
        f"edge_missing={totals['edge_missing']} edge_extra={totals['edge_extra']} "
        f"counter_mismatches={totals['counter_mismatches']} report={report_path}"
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
