"""Deterministic fixed-detection benchmark for the 3D+t linking stack.

This deliberately isolates association from segmentation.  Passing it is
evidence of software behavior on an engineered sequence, not biological
accuracy or Cell Tracking Challenge performance.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from .solver import build_edge_candidates, select_edges, validate_lineage_graph
from .types import CandidateNode, RouteConfig


SYNTHETIC_SCALE_ZYX = (2.0, 1.0, 1.0)
EXPECTED_FIXTURE_DIGEST_SHA256 = "5b4f42026097f3adf4231a3e945509aad66c11acdf3621f359527262bc634c03"
EXPECTED_EDGE_DIGEST_SHA256 = "eac380a9a04d3155b222df8c557d9840d90274c9dd545912ad6364b1fbceb99b"


def _benchmark_config() -> RouteConfig:
    return RouteConfig(
        label="synthetic_fixed_detection",
        detector_sources=("truth",),
        threshold_quantiles=(0.99,),
        min_distance_um=1.0,
        fusion_radius_um=1.0,
        max_points_per_t=20,
        max_link_um=2.5,
        max_gap=2,
        gap_link_um=3.2,
        max_neighbors=6,
        edge_cost_threshold=0.95,
        node_score_weight=0.20,
        density_cost_weight=0.0,
        appearance_cost_weight=0.0,
        gap_motion_cost_weight=0.25,
    )


def make_synthetic_4d_case() -> tuple[
    np.ndarray,
    list[list[CandidateNode]],
    set[tuple[int, int]],
]:
    """Return a tiny anisotropic T/Z/Y/X sequence and exact association truth."""

    volume = np.zeros((6, 3, 16, 16), dtype=np.float32)
    nodes_by_t: list[list[CandidateNode]] = []
    truth_by_track: dict[str, list[CandidateNode]] = {"a": [], "b": [], "c": []}
    next_id = 1
    for t in range(volume.shape[0]):
        rows: list[CandidateNode] = []
        definitions = [
            ("a", 0, 2 + t, 2 + t, 100.0, t != 3),
            ("b", 1, 12 - t, 2 + t, 200.0, True),
            ("c", 0, 8, 8, 150.0, t == 0),
            ("c", 0, 8, 10, 150.0, t == 1),
            (None, 1, 9, 8, 150.0, t == 1),
        ]
        for track_name, z, y, x, intensity, visible in definitions:
            if not visible:
                continue
            volume[t, z, y, x] = intensity
            candidate = CandidateNode(
                dataset="synthetic_4d",
                node_id=next_id,
                t=t,
                z=z,
                y=y,
                x=x,
                score=1.0,
                sources=("truth",),
                radius_um=1.0,
                intensity=intensity,
                features={
                    "synthetic_track": float(
                        {"a": 0, "b": 1, "c": 2}.get(track_name, -1)
                    )
                },
            )
            next_id += 1
            rows.append(candidate)
            if track_name is not None:
                truth_by_track[track_name].append(candidate)
        nodes_by_t.append(rows)

    truth_edges: set[tuple[int, int]] = set()
    for track_nodes in truth_by_track.values():
        for previous, current in zip(track_nodes[:-1], track_nodes[1:], strict=True):
            truth_edges.add((previous.node_id, current.node_id))
    return volume, nodes_by_t, truth_edges


def _edge_metrics(
    predicted: set[tuple[int, int]], truth: set[tuple[int, int]]
) -> dict[str, float | int]:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
    }


def _fixture_manifest(
    volume: np.ndarray,
    nodes_by_t: list[list[CandidateNode]],
    truth_edges: set[tuple[int, int]],
    config: RouteConfig,
) -> dict[str, Any]:
    nodes = [node for frame in nodes_by_t for node in frame]
    return {
        "shape_tzyx": list(volume.shape),
        "scale_zyx": list(SYNTHETIC_SCALE_ZYX),
        "nodes": [
            {
                "node_id": node.node_id,
                "t": node.t,
                "z": node.z,
                "y": node.y,
                "x": node.x,
                "intensity": node.intensity,
            }
            for node in sorted(nodes, key=lambda value: value.node_id)
        ],
        "truth_edges": [list(pair) for pair in sorted(truth_edges)],
        "config": asdict(config),
    }


def run_synthetic_benchmark(repetitions: int = 5) -> dict[str, Any]:
    """Run the deterministic association benchmark and return JSON-safe evidence."""

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    volume, nodes_by_t, truth_edges = make_synthetic_4d_case()
    config = _benchmark_config()
    manifest = _fixture_manifest(volume, nodes_by_t, truth_edges, config)
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    all_nodes = [node for frame in nodes_by_t for node in frame]
    outputs: list[list[tuple[int, int]]] = []
    runtimes_ms: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        candidates = build_edge_candidates(nodes_by_t, SYNTHETIC_SCALE_ZYX, config)
        selected, forks = select_edges(nodes_by_t, candidates, SYNTHETIC_SCALE_ZYX, config)
        runtimes_ms.append((time.perf_counter() - started) * 1_000.0)
        if forks:
            raise AssertionError("synthetic no-division sequence produced a fork")
        validate_lineage_graph(all_nodes, selected, allow_divisions=False)
        outputs.append([(edge.source_id, edge.target_id) for edge in selected])

    deterministic = all(output == outputs[0] for output in outputs[1:])
    predicted = set(outputs[0])
    serialized = json.dumps(outputs[0], separators=(",", ":")).encode("utf-8")
    return {
        "scope": "fixed detections; segmentation bypassed",
        "seed": None,
        "shape_tzyx": list(volume.shape),
        "scale_zyx": list(SYNTHETIC_SCALE_ZYX),
        "num_nodes": len(all_nodes),
        "num_truth_edges": len(truth_edges),
        "num_predicted_edges": len(predicted),
        "gap_edges_recovered": sum(
            1
            for source, target in predicted
            if next(node.t for node in all_nodes if node.node_id == target)
            - next(node.t for node in all_nodes if node.node_id == source)
            > 1
        ),
        **_edge_metrics(predicted, truth_edges),
        "deterministic": deterministic,
        "fixture_digest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "edge_digest_sha256": hashlib.sha256(serialized).hexdigest(),
        "repetitions": repetitions,
        "runtime_ms_median": float(np.median(runtimes_ms)),
        "runtime_ms_max": float(np.max(runtimes_ms)),
        "config": asdict(config),
    }


def assert_synthetic_benchmark(result: dict[str, Any]) -> None:
    """Raise when the software-correctness acceptance gate is not met."""

    if result.get("edge_f1") != 1.0:
        raise AssertionError(f"synthetic edge F1 must be 1.0, got {result.get('edge_f1')!r}")
    if result.get("gap_edges_recovered") != 1:
        raise AssertionError("synthetic benchmark must recover exactly one gap edge")
    if result.get("deterministic") is not True:
        raise AssertionError("synthetic benchmark output changed across repetitions")
    if result.get("fixture_digest_sha256") != EXPECTED_FIXTURE_DIGEST_SHA256:
        raise AssertionError(
            "synthetic fixture/config changed without reviewing and updating its golden manifest"
        )
    if result.get("edge_digest_sha256") != EXPECTED_EDGE_DIGEST_SHA256:
        raise AssertionError("synthetic predicted edge graph changed from the reviewed golden")
    expected_manifest_fields = {
        "shape_tzyx": [6, 3, 16, 16],
        "scale_zyx": [2.0, 1.0, 1.0],
        "num_nodes": 14,
        "num_truth_edges": 10,
        "num_predicted_edges": 10,
    }
    for key, expected in expected_manifest_fields.items():
        if result.get(key) != expected:
            raise AssertionError(f"synthetic {key} changed: expected {expected!r}")
