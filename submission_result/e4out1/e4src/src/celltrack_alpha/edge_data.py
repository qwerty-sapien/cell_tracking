"""Sparse, label-safe association examples for compact edge calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import torch

from .bh_edge import SparseEdgeScorer, label_sparse_training_pairs
from .bh_graph import PHYSICAL_SCALE_ZYX, sparse_min_cost_max_cardinality
from .bh_model import FrameDet, PairContext, VideoInference


MATCH_RADIUS_UM = 7.0
FEATURE_COUNT = 16


class EdgeDataError(ValueError):
    """Raised when sparse association supervision is malformed or ambiguous."""


@dataclass(frozen=True)
class EdgeExamples:
    """One-positive source groups for a bounded candidate softmax objective."""

    features: np.ndarray
    checkpoint_logits: np.ndarray
    candidate_mask: np.ndarray
    target_index: np.ndarray
    source_count: int
    candidate_count: int
    positive_count: int
    skipped_ambiguous_sources: int
    skipped_unknown_sources: int

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        logits = np.asarray(self.checkpoint_logits, dtype=np.float32)
        mask = np.asarray(self.candidate_mask, dtype=bool)
        targets = np.asarray(self.target_index, dtype=np.int64)
        if (
            features.ndim != 3
            or features.shape[2] != FEATURE_COUNT
            or logits.shape != features.shape[:2]
            or mask.shape != features.shape[:2]
            or targets.shape != (features.shape[0],)
            or not np.isfinite(features).all()
            or not np.isfinite(logits).all()
        ):
            raise EdgeDataError("edge example tensor shapes or values are invalid")
        if features.shape[1] not in (2, 4, 8):
            raise EdgeDataError("edge example candidate width must be 2, 4, or 8")
        if np.any(mask.sum(axis=1) < 1) or np.any(targets < 0):
            raise EdgeDataError("edge examples require a positive supervised candidate")
        if np.any(targets >= mask.sum(axis=1)):
            raise EdgeDataError("edge example target lies outside its candidate rows")
        if any(type(value) is not int or value < 0 for value in (
            self.source_count,
            self.candidate_count,
            self.positive_count,
            self.skipped_ambiguous_sources,
            self.skipped_unknown_sources,
        )):
            raise EdgeDataError("edge example counters must be nonnegative integers")
        object.__setattr__(self, "features", np.ascontiguousarray(features))
        object.__setattr__(self, "checkpoint_logits", np.ascontiguousarray(logits))
        object.__setattr__(self, "candidate_mask", np.ascontiguousarray(mask))
        object.__setattr__(self, "target_index", np.ascontiguousarray(targets))

    def as_tensors(self, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features).to(device=device),
            torch.from_numpy(self.checkpoint_logits).to(device=device),
            torch.from_numpy(self.candidate_mask).to(device=device),
            torch.from_numpy(self.target_index).to(device=device),
        )


def _gt_by_frame(gt: Mapping[str, Any]) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], set[tuple[int, int]]]:
    nodes = gt.get("nodes")
    edges = gt.get("edges")
    if not isinstance(nodes, Mapping):
        raise EdgeDataError("GEFF node table is missing")
    try:
        ids = np.asarray(nodes["node_id"], dtype=np.int64)
        times = np.asarray(nodes["t"], dtype=np.int64)
        coords = np.column_stack((nodes["z"], nodes["y"], nodes["x"])).astype(np.float64, copy=False)
        raw_edges = np.asarray(edges, dtype=np.int64)
    except (KeyError, TypeError, ValueError) as exc:
        raise EdgeDataError("GEFF arrays have an unsupported node or edge schema") from exc
    if (
        ids.ndim != 1
        or times.shape != ids.shape
        or coords.shape != (len(ids), 3)
        or len(set(ids.tolist())) != len(ids)
        or np.any(ids < 0)
        or np.any(times < 0)
        or not np.isfinite(coords).all()
        or raw_edges.ndim != 2
        or raw_edges.shape[1:] != (2,)
    ):
        raise EdgeDataError("GEFF node or edge values are invalid")
    by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame in sorted(set(times.tolist())):
        rows = np.flatnonzero(times == frame)
        by_frame[int(frame)] = (ids[rows], coords[rows])
    edge_set = {(int(source), int(target)) for source, target in raw_edges.tolist()}
    if len(edge_set) != len(raw_edges):
        raise EdgeDataError("GEFF contains duplicate edges")
    return by_frame, edge_set


def match_detections_to_gt(
    detections: Sequence[FrameDet],
    gt: Mapping[str, Any],
    *,
    scale_zyx: tuple[float, float, float] = PHYSICAL_SCALE_ZYX,
    max_distance_um: float = MATCH_RADIUS_UM,
) -> tuple[dict[int, int | None], set[tuple[int, int]]]:
    """Match detections to GEFF nodes by the metric's physical 7-um gate.

    Candidate matches are KD-tree gated and then solved cardinality-first with
    the existing sparse assignment implementation.  No dense frame-by-frame
    distance matrix is formed.
    """

    if not isinstance(detections, (tuple, list)) or not all(isinstance(item, FrameDet) for item in detections):
        raise EdgeDataError("detections must be canonical FrameDet records")
    if (
        len(scale_zyx) != 3
        or any(not math.isfinite(value) or value <= 0.0 for value in scale_zyx)
        or not math.isfinite(max_distance_um)
        or max_distance_um <= 0.0
    ):
        raise EdgeDataError("physical matching scale or radius is invalid")
    frame_data, gt_edges = _gt_by_frame(gt)
    result: dict[int, int | None] = {}
    scale = np.asarray(scale_zyx, dtype=np.float64)
    for detection in detections:
        ids = [int(value) for value in detection.node_ids]
        if any(node_id in result for node_id in ids):
            raise EdgeDataError("detection IDs must be globally unique across frames")
        result.update({node_id: None for node_id in ids})
        gt_ids, gt_coords = frame_data.get(detection.frame_index, (np.empty(0, dtype=np.int64), np.empty((0, 3))))
        if not ids or len(gt_ids) == 0:
            continue
        pred_physical = np.asarray(detection.coords_zyx, dtype=np.float64) * scale
        gt_physical = np.asarray(gt_coords, dtype=np.float64) * scale
        tree = cKDTree(gt_physical)
        feasible: list[tuple[int, int, float]] = []
        for row, candidates in enumerate(tree.query_ball_point(pred_physical, r=max_distance_um)):
            for candidate in sorted(int(value) for value in candidates):
                distance = float(np.linalg.norm(pred_physical[row] - gt_physical[candidate]))
                if distance <= max_distance_um:
                    feasible.append((ids[row], int(gt_ids[candidate]), distance))
        if feasible:
            match = sparse_min_cost_max_cardinality(ids, gt_ids.tolist(), feasible)
            for row in match.matches:
                result[int(row.source_id)] = int(row.target_id)
    return result, gt_edges


def examples_from_video(
    scorer: SparseEdgeScorer,
    video: VideoInference,
    detection_to_gt: Mapping[int, int | None],
    gt_edges: set[tuple[int, int]],
) -> EdgeExamples:
    """Create only unambiguous fit examples from one canonical inferred video."""

    if not isinstance(scorer, SparseEdgeScorer) or not isinstance(video, VideoInference):
        raise EdgeDataError("examples require SparseEdgeScorer and VideoInference")
    if len(video.contexts) != len(video.detections) - 1:
        raise EdgeDataError("video pair contexts are incomplete")
    width = scorer.config.neighbor_cap
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[int] = []
    source_count = candidate_count = positive_count = ambiguous = unknown = 0
    for context, source, target in zip(video.contexts, video.detections[:-1], video.detections[1:], strict=True):
        candidates, base_logits, candidate_features = scorer.candidate_features(context, source, target)
        labels = label_sparse_training_pairs(candidates, detection_to_gt, gt_edges)
        rows_by_source: list[list[int]] = [[] for _ in range(len(source.node_ids))]
        for row, source_index in enumerate(candidates.source_index.tolist()):
            rows_by_source[int(source_index)].append(row)
        for rows in rows_by_source:
            if not rows:
                continue
            source_count += 1
            row_array = np.asarray(rows, dtype=np.int64)
            observed = labels.labels[row_array]
            positives = np.flatnonzero(observed == 1)
            if len(positives) != 1:
                ambiguous += 1
                continue
            if np.any(observed < 0):
                unknown += 1
                continue
            count = len(row_array)
            if count > width:
                raise EdgeDataError("candidate gate exceeded its configured width")
            feature_row = np.zeros((width, FEATURE_COUNT), dtype=np.float32)
            logit_row = np.zeros(width, dtype=np.float32)
            mask_row = np.zeros(width, dtype=bool)
            feature_row[:count] = candidate_features[row_array].detach().cpu().numpy()
            logit_row[:count] = base_logits[row_array].detach().cpu().numpy()
            mask_row[:count] = True
            features.append(feature_row)
            logits.append(logit_row)
            masks.append(mask_row)
            targets.append(int(positives[0]))
            candidate_count += count
            positive_count += 1
    if not features:
        return EdgeExamples(
            np.empty((0, width, FEATURE_COUNT), dtype=np.float32),
            np.empty((0, width), dtype=np.float32),
            np.empty((0, width), dtype=bool),
            np.empty((0,), dtype=np.int64),
            source_count,
            candidate_count,
            positive_count,
            ambiguous,
            unknown,
        )
    return EdgeExamples(
        np.stack(features),
        np.stack(logits),
        np.stack(masks),
        np.asarray(targets, dtype=np.int64),
        source_count,
        candidate_count,
        positive_count,
        ambiguous,
        unknown,
    )


__all__ = [
    "FEATURE_COUNT",
    "MATCH_RADIUS_UM",
    "EdgeDataError",
    "EdgeExamples",
    "examples_from_video",
    "match_detections_to_gt",
]
