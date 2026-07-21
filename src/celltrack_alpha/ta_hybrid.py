"""Parameter-free learned-motion reranking over frozen alpha graph rows.

The adapter never proposes a node or an edge.  It supplies only a child-owned
backward displacement and uncertainty for parent candidates already admitted
by the frozen alpha's physical cKDTree gate.  Division edges, out-of-gate
alpha edges, and edges whose child has no motion sample are copied verbatim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from .bh_graph import (
    GraphNode,
    PHYSICAL_SCALE_ZYX,
    V45_GRAPH_CONFIG,
    normalize_nodes,
    physical_distance,
    sparse_min_cost_max_cardinality,
)
from .ta_inference import FrameOutput, HybridAttestation


HYBRID_SCHEMA = "celltrack-alpha.alpha-node-motion-reranker.v1"
MOTION_OWNER = "child-at-t"
MOTION_DIRECTION = "parent_um-minus-child_um"


class HybridRerankError(ValueError):
    """Frozen alpha rows or sparse motion evidence are inconsistent."""


@dataclass(frozen=True, slots=True)
class ChildMotion:
    """One adapter sample at a frozen alpha child coordinate."""

    node_id: int
    backward_um: tuple[float, float, float]
    log_variance: float

    def __post_init__(self) -> None:
        if isinstance(self.node_id, bool) or not isinstance(self.node_id, int) or self.node_id < 0:
            raise HybridRerankError("child motion node_id must be a nonnegative integer")
        values = (*self.backward_um, self.log_variance)
        if len(self.backward_um) != 3 or any(not math.isfinite(float(value)) for value in values):
            raise HybridRerankError("child motion must contain finite z/y/x displacement and log variance")


@dataclass(frozen=True, slots=True)
class HybridRerankResult:
    """Frozen nodes, reranked edge rows, and sparse execution evidence."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[dict[str, int], ...]
    stats: Mapping[str, Any]
    provenance: Mapping[str, Any]


def _node_fingerprint(nodes: Sequence[GraphNode]) -> str:
    rows = [
        {
            "node_id": int(node.node_id),
            "t": int(node.t),
            "z": float(node.z),
            "y": float(node.y),
            "x": float(node.x),
            "score": float(node.score),
            "synthetic": bool(node.synthetic),
        }
        for node in nodes
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _positive_triplet(values: Sequence[float], label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HybridRerankError(f"{label} must contain three positive finite values") from exc
    if array.shape != (3,) or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise HybridRerankError(f"{label} must contain three positive finite values")
    return array


def _edge_pair(row: Mapping[str, Any] | Sequence[Any]) -> tuple[int, int]:
    try:
        if isinstance(row, Mapping):
            raw_source, raw_target = row["source_id"], row["target_id"]
        else:
            if isinstance(row, (str, bytes)) or len(row) != 2:
                raise HybridRerankError("alpha edge must contain source_id and target_id")
            raw_source, raw_target = row
        if isinstance(raw_source, bool) or isinstance(raw_target, bool):
            raise TypeError
        source, target = int(raw_source), int(raw_target)
        if float(raw_source) != source or float(raw_target) != target:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise HybridRerankError("alpha edge must contain exact integer source_id and target_id") from exc
    return source, target


def _normalize_alpha_graph(
    alpha_nodes: Iterable[Any],
    alpha_edges: Iterable[Mapping[str, Any] | Sequence[Any]],
) -> tuple[tuple[GraphNode, ...], tuple[tuple[int, int], ...]]:
    try:
        nodes = normalize_nodes(alpha_nodes, max_nodes_per_frame=V45_GRAPH_CONFIG.max_nodes_per_frame)
    except (TypeError, ValueError) as exc:
        raise HybridRerankError("alpha node table is invalid") from exc
    if not nodes:
        raise HybridRerankError("alpha node table is empty")
    node_by_id = {int(node.node_id): node for node in nodes}
    edges = tuple(_edge_pair(row) for row in alpha_edges)
    if len(edges) != len(set(edges)):
        raise HybridRerankError("alpha edge table contains duplicates")
    indegree: Counter[int] = Counter()
    outdegree: Counter[int] = Counter()
    for source_id, target_id in edges:
        source, target = node_by_id.get(source_id), node_by_id.get(target_id)
        if source is None or target is None:
            raise HybridRerankError("alpha edge references a missing frozen node")
        if int(target.t) != int(source.t) + 1:
            raise HybridRerankError("alpha reranking accepts adjacent-frame edges only")
        indegree[target_id] += 1
        outdegree[source_id] += 1
        if indegree[target_id] > 1 or outdegree[source_id] > 2:
            raise HybridRerankError("alpha graph violates lineage degree limits")
    return nodes, edges


def sample_child_motion(
    alpha_nodes: Iterable[Any],
    frame_outputs: Iterable[FrameOutput],
    *,
    downsample_zyx: Sequence[float] = (1.0, 4.0, 4.0),
) -> dict[int, ChildMotion]:
    """Sample only child-side association channels at frozen alpha nodes.

    Coordinates are mapped from original alpha voxels to the adapter output
    grid.  Center logits and source-side association channels are ignored.
    """

    nodes = normalize_nodes(alpha_nodes, max_nodes_per_frame=V45_GRAPH_CONFIG.max_nodes_per_frame)
    stride = _positive_triplet(downsample_zyx, "downsample_zyx")
    by_frame: dict[int, list[GraphNode]] = defaultdict(list)
    for node in nodes:
        by_frame[int(node.t)].append(node)
    seen_frames: set[int] = set()
    result: dict[int, ChildMotion] = {}
    for frame in frame_outputs:
        if not isinstance(frame, FrameOutput):
            raise HybridRerankError("frame output is not a FrameOutput")
        frame_index = int(frame.frame_index)
        if frame_index in seen_frames:
            raise HybridRerankError("adapter frame outputs contain a duplicate frame")
        seen_frames.add(frame_index)
        if not frame.association_valid:
            continue
        association = np.asarray(frame.association)
        if (
            association.ndim != 4
            or association.shape[0] != 4
            or min(association.shape) <= 0
            or not np.issubdtype(association.dtype, np.floating)
            or not np.isfinite(association).all()
        ):
            raise HybridRerankError("adapter association field must have finite shape (4,Z,Y,X)")
        for node in by_frame.get(frame_index, ()):
            point = np.asarray(node.coord_zyx, dtype=np.float64) / stride
            sampled = np.asarray(
                [
                    map_coordinates(
                        association[channel],
                        point.reshape(3, 1),
                        order=1,
                        mode="nearest",
                        prefilter=False,
                    )[0]
                    for channel in range(4)
                ],
                dtype=np.float64,
            )
            if not np.isfinite(sampled).all():
                raise HybridRerankError("sampled child motion is nonfinite")
            result[int(node.node_id)] = ChildMotion(
                node_id=int(node.node_id),
                backward_um=tuple(float(value) for value in sampled[:3]),
                log_variance=float(sampled[3]),
            )
    return result


def _validate_attestation(attestation: HybridAttestation | None, dataset: str | None) -> dict[str, Any]:
    if attestation is None:
        return {
            "attested": False,
            "weight_calibration": "none-parameter-free",
            "held_out_target_calibration_used": False,
        }
    if dataset is None or not dataset or ":" in dataset:
        raise HybridRerankError("an attested rerank requires one unprefixed held-out dataset")
    if attestation.held_out_dataset != dataset or attestation.held_out_video_id != f"train:{dataset}":
        raise HybridRerankError("hybrid attestation does not match the reranked dataset")
    fit_sources = set(attestation.candidate.source_video_ids) | set(attestation.pretrain.source_video_ids)
    if attestation.held_out_video_id in fit_sources or attestation.test_used or not attestation.audit_disjoint:
        raise HybridRerankError("hybrid attestation does not prove disjoint fold-fit provenance")
    return {
        "attested": True,
        "fold": int(attestation.fold),
        "held_out_video_id": attestation.held_out_video_id,
        "checkpoint_sha256": attestation.candidate.checkpoint_sha256,
        "temporal_pretrain_sha256": attestation.pretrain.checkpoint_sha256,
        "fit_video_ids": sorted(fit_sources),
        "split_sha256": attestation.split_sha256,
        "provenance_sha256": attestation.provenance_sha256,
        "registry_sha256": attestation.registry_sha256,
        "weight_calibration": "none-parameter-free",
        "held_out_target_calibration_used": False,
    }


def rerank_alpha_edges(
    alpha_nodes: Iterable[Any],
    alpha_edges: Iterable[Mapping[str, Any] | Sequence[Any]],
    child_motion: Mapping[int, ChildMotion],
    *,
    scale_zyx: Sequence[float] = PHYSICAL_SCALE_ZYX,
    gate_radius_um: float = V45_GRAPH_CONFIG.motion_relaxed_um,
    dataset: str | None = None,
    attestation: HybridAttestation | None = None,
) -> HybridRerankResult:
    """Rerank frozen alpha continuation slots with sparse backward agreement.

    The cardinality and target set of the frozen continuation edges remain
    unchanged.  Candidate parents are restricted to the corresponding alpha
    sources and the frozen physical radius.  A sparse component matcher avoids
    a full-frame ``N x N`` allocation.
    """

    nodes, alpha_pairs = _normalize_alpha_graph(alpha_nodes, alpha_edges)
    scale = _positive_triplet(scale_zyx, "scale_zyx")
    if not math.isfinite(float(gate_radius_um)) or float(gate_radius_um) <= 0.0:
        raise HybridRerankError("gate_radius_um must be positive and finite")
    if float(gate_radius_um) != float(V45_GRAPH_CONFIG.motion_relaxed_um):
        raise HybridRerankError("the hybrid gate is locked to the frozen alpha relaxed radius")
    provenance = _validate_attestation(attestation, dataset)
    node_by_id = {int(node.node_id): node for node in nodes}
    motions: dict[int, ChildMotion] = {}
    for raw_id, motion in child_motion.items():
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id != motion.node_id:
            raise HybridRerankError("child motion mapping key disagrees with its node_id")
        node = node_by_id.get(raw_id)
        if node is None:
            raise HybridRerankError("child motion references a node outside frozen alpha")
        if int(node.t) == min(int(value.t) for value in nodes):
            raise HybridRerankError("first-frame alpha nodes cannot own backward motion")
        motions[raw_id] = motion

    outdegree = Counter(source for source, _target in alpha_pairs)
    division_parents = {source for source, count in outdegree.items() if count == 2}
    eligible_by_frame: dict[int, list[tuple[int, int]]] = defaultdict(list)
    fixed_pairs: set[tuple[int, int]] = set()
    fallback_missing = 0
    fallback_outside = 0
    division_edges = 0
    for pair in alpha_pairs:
        source_id, target_id = pair
        source, target = node_by_id[source_id], node_by_id[target_id]
        distance = physical_distance(source, target, scale)
        if source_id in division_parents:
            fixed_pairs.add(pair)
            division_edges += 1
        elif target_id not in motions:
            fixed_pairs.add(pair)
            fallback_missing += 1
        elif distance > float(gate_radius_um) + 1.0e-12:
            fixed_pairs.add(pair)
            fallback_outside += 1
        else:
            eligible_by_frame[int(source.t)].append(pair)

    replacement_by_target: dict[int, int] = {}
    candidate_count = 0
    component_count = 0
    peak_component = 0
    for frame in sorted(eligible_by_frame):
        baseline = eligible_by_frame[frame]
        source_ids = tuple(sorted(source for source, _target in baseline))
        target_ids = tuple(sorted(target for _source, target in baseline))
        if len(source_ids) != len(set(source_ids)) or len(target_ids) != len(set(target_ids)):
            raise HybridRerankError("eligible alpha continuation slots are not one-to-one")
        source_physical = np.asarray(
            [node_by_id[source].coord_zyx for source in source_ids], dtype=np.float64
        ) * scale
        tree = cKDTree(source_physical)
        feasible: list[tuple[int, int, float]] = []
        for target_id in target_ids:
            target = node_by_id[target_id]
            child_physical = np.asarray(target.coord_zyx, dtype=np.float64) * scale
            motion = motions[target_id]
            predicted_parent = child_physical + np.asarray(motion.backward_um, dtype=np.float64)
            variance = math.exp(max(-10.0, min(10.0, float(motion.log_variance))))
            for source_index in tree.query_ball_point(
                child_physical, r=float(gate_radius_um), return_sorted=True
            ):
                source_id = source_ids[int(source_index)]
                residual = source_physical[int(source_index)] - predicted_parent
                agreement_cost = float(np.dot(residual, residual) / variance)
                feasible.append((source_id, target_id, agreement_cost))
        candidate_count += len(feasible)
        result = sparse_min_cost_max_cardinality(source_ids, target_ids, feasible, parity_mode="sparse")
        component_count += int(result.component_count)
        peak_component = max(peak_component, int(result.peak_component_matrix_dim))
        if len(result.matches) != len(baseline):
            raise HybridRerankError("sparse reranker failed to preserve alpha continuation cardinality")
        for match in result.matches:
            if match.target_id in replacement_by_target:
                raise HybridRerankError("sparse reranker assigned a target twice")
            replacement_by_target[int(match.target_id)] = int(match.source_id)

    hybrid_pairs = tuple(
        (replacement_by_target.get(target, source), target)
        for source, target in alpha_pairs
    )
    if len(hybrid_pairs) != len(alpha_pairs) or len(hybrid_pairs) != len(set(hybrid_pairs)):
        raise HybridRerankError("hybrid edge cardinality or uniqueness changed")
    hybrid_indegree = Counter(target for _source, target in hybrid_pairs)
    hybrid_outdegree = Counter(source for source, _target in hybrid_pairs)
    if any(value > 1 for value in hybrid_indegree.values()) or any(value > 2 for value in hybrid_outdegree.values()):
        raise HybridRerankError("hybrid rerank violated lineage degree limits")
    if any(pair not in hybrid_pairs for pair in fixed_pairs):
        raise HybridRerankError("hybrid rerank changed a frozen division or fallback edge")
    fingerprint = _node_fingerprint(nodes)
    changed = sum(first != second for first, second in zip(alpha_pairs, hybrid_pairs, strict=True))
    stats = {
        "schema": HYBRID_SCHEMA,
        "frozen_node_count": len(nodes),
        "alpha_edge_count": len(alpha_pairs),
        "hybrid_edge_count": len(hybrid_pairs),
        "eligible_continuation_edges": sum(len(value) for value in eligible_by_frame.values()),
        "candidate_pair_count": candidate_count,
        "changed_parent_count": changed,
        "division_parent_count": len(division_parents),
        "division_edges_preserved": division_edges,
        "fallback_missing_motion_edges": fallback_missing,
        "fallback_outside_gate_edges": fallback_outside,
        "sparse_component_count": component_count,
        "peak_component_matrix_dim": peak_component,
        "full_frame_dense_matrix": False,
        "node_fingerprint_before": fingerprint,
        "node_fingerprint_after": fingerprint,
        "nodes_preserved": True,
    }
    return HybridRerankResult(
        nodes=nodes,
        edges=tuple(
            {"source_id": int(source), "target_id": int(target)}
            for source, target in hybrid_pairs
        ),
        stats=stats,
        provenance={
            **provenance,
            "schema": HYBRID_SCHEMA,
            "alpha_nodes": "frozen",
            "alpha_edge_cardinality": "frozen",
            "division_policy": "frozen-alpha-edges-verbatim",
            "fallback_policy": "frozen-alpha-edge-verbatim-on-missing-motion-or-outside-gate",
            "candidate_gate": "physical-cKDTree-existing-alpha-nodes",
            "gate_radius_um": float(gate_radius_um),
            "motion_owner": MOTION_OWNER,
            "motion_direction": MOTION_DIRECTION,
            "agreement_cost": "squared-parent-residual-um2/exp(child-log-variance)",
            "learned_or_calibrated_rerank_weights": 0,
            "adapter_center_logits_used": False,
            "full_frame_dense_matrix": False,
        },
    )


__all__ = [
    "ChildMotion",
    "HYBRID_SCHEMA",
    "HybridRerankError",
    "HybridRerankResult",
    "MOTION_DIRECTION",
    "MOTION_OWNER",
    "rerank_alpha_edges",
    "sample_child_motion",
]
