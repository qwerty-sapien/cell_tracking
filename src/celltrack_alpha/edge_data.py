"""Sparse, label-safe association examples for compact edge calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
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
SOURCE_CONFLICT = 0
TARGET_CONFLICT = 1


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


@dataclass(frozen=True)
class ConflictPairExamples:
    """Reachable continuation misses paired with actual selected conflicts.

    Every row compares one identity-supported true continuation candidate with
    one edge selected by the frozen division-disabled alpha graph that consumes
    the same source or target.  The records retain only sparse cap-K candidate
    features; they never materialize or imply an object-by-object matrix.
    """

    true_features: np.ndarray
    competitor_features: np.ndarray
    true_checkpoint_logits: np.ndarray
    competitor_checkpoint_logits: np.ndarray
    conflict_kind: np.ndarray
    true_pairs: np.ndarray
    competitor_pairs: np.ndarray
    candidate_count: int
    reachable_miss_count: int
    source_conflict_count: int
    target_conflict_count: int
    skipped_division_truth: int
    skipped_without_candidate_conflict: int
    candidate_topology_sha256: str

    def __post_init__(self) -> None:
        true_features = np.asarray(self.true_features, dtype=np.float32)
        competitor_features = np.asarray(self.competitor_features, dtype=np.float32)
        true_logits = np.asarray(self.true_checkpoint_logits, dtype=np.float32)
        competitor_logits = np.asarray(self.competitor_checkpoint_logits, dtype=np.float32)
        kinds = np.asarray(self.conflict_kind, dtype=np.int8)
        true_pairs = np.asarray(self.true_pairs, dtype=np.int64)
        competitor_pairs = np.asarray(self.competitor_pairs, dtype=np.int64)
        count = int(true_features.shape[0]) if true_features.ndim else -1
        if (
            true_features.shape != (count, FEATURE_COUNT)
            or competitor_features.shape != (count, FEATURE_COUNT)
            or true_logits.shape != (count,)
            or competitor_logits.shape != (count,)
            or kinds.shape != (count,)
            or true_pairs.shape != (count, 2)
            or competitor_pairs.shape != (count, 2)
            or not np.isfinite(true_features).all()
            or not np.isfinite(competitor_features).all()
            or not np.isfinite(true_logits).all()
            or not np.isfinite(competitor_logits).all()
        ):
            raise EdgeDataError("conflict-pair tensor shapes or values are invalid")
        if np.any((kinds != SOURCE_CONFLICT) & (kinds != TARGET_CONFLICT)):
            raise EdgeDataError("conflict kind must identify a source or target conflict")
        if np.any(np.all(true_pairs == competitor_pairs, axis=1)):
            raise EdgeDataError("true and competitor edges must differ")
        if np.any((kinds == SOURCE_CONFLICT) & (true_pairs[:, 0] != competitor_pairs[:, 0])):
            raise EdgeDataError("source-conflict rows must share the true source")
        if np.any((kinds == TARGET_CONFLICT) & (true_pairs[:, 1] != competitor_pairs[:, 1])):
            raise EdgeDataError("target-conflict rows must share the true target")
        counters = (
            self.candidate_count,
            self.reachable_miss_count,
            self.source_conflict_count,
            self.target_conflict_count,
            self.skipped_division_truth,
            self.skipped_without_candidate_conflict,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise EdgeDataError("conflict-pair counters must be nonnegative integers")
        if self.source_conflict_count + self.target_conflict_count != count:
            raise EdgeDataError("conflict-pair kind counts do not reconcile")
        if (
            not isinstance(self.candidate_topology_sha256, str)
            or len(self.candidate_topology_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.candidate_topology_sha256)
        ):
            raise EdgeDataError("candidate topology identity must be a SHA-256")
        object.__setattr__(self, "true_features", np.ascontiguousarray(true_features))
        object.__setattr__(self, "competitor_features", np.ascontiguousarray(competitor_features))
        object.__setattr__(self, "true_checkpoint_logits", np.ascontiguousarray(true_logits))
        object.__setattr__(self, "competitor_checkpoint_logits", np.ascontiguousarray(competitor_logits))
        object.__setattr__(self, "conflict_kind", np.ascontiguousarray(kinds))
        object.__setattr__(self, "true_pairs", np.ascontiguousarray(true_pairs))
        object.__setattr__(self, "competitor_pairs", np.ascontiguousarray(competitor_pairs))

    def __len__(self) -> int:
        return int(self.true_features.shape[0])

    def as_tensors(
        self, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.true_features).to(device=device),
            torch.from_numpy(self.competitor_features).to(device=device),
            torch.from_numpy(self.true_checkpoint_logits).to(device=device),
            torch.from_numpy(self.competitor_checkpoint_logits).to(device=device),
        )


def _topology_sha256(pairs: Sequence[tuple[int, int]]) -> str:
    payload = json.dumps(
        [[int(source), int(target)] for source, target in sorted(pairs)],
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def conflict_examples_from_video(
    scorer: SparseEdgeScorer,
    video: VideoInference,
    detection_to_gt: Mapping[int, int | None],
    gt_edges: set[tuple[int, int]],
    selected_edges: Sequence[tuple[int, int]] | set[tuple[int, int]],
) -> ConflictPairExamples:
    """Build only true-vs-selected continuation conflict comparisons.

    A true candidate is eligible only when both detector endpoints have unique
    metric-radius GT identities and the GT edge is one-to-one (division and
    merge motifs are excluded).  A negative is never sampled: it must be the
    edge actually selected by the frozen graph and must itself be present in
    the same 10-um cap-K sparse candidate topology.
    """

    if not isinstance(scorer, SparseEdgeScorer) or not isinstance(video, VideoInference):
        raise EdgeDataError("conflict examples require SparseEdgeScorer and VideoInference")
    if len(video.contexts) != len(video.detections) - 1:
        raise EdgeDataError("video pair contexts are incomplete")
    try:
        selected = {(int(source), int(target)) for source, target in selected_edges}
    except (TypeError, ValueError) as exc:
        raise EdgeDataError("selected edges must be integer pairs") from exc
    if len(selected) != len(selected_edges):
        raise EdgeDataError("selected continuation edges contain duplicates")
    selected_out: dict[int, int] = {}
    selected_in: dict[int, int] = {}
    for source_id, target_id in sorted(selected):
        if source_id in selected_out or target_id in selected_in:
            raise EdgeDataError("selected continuation graph is not division/merge disabled")
        selected_out[source_id] = target_id
        selected_in[target_id] = source_id

    gt_out: dict[int, int] = {}
    gt_in: dict[int, int] = {}
    for source_gt, target_gt in gt_edges:
        gt_out[source_gt] = gt_out.get(source_gt, 0) + 1
        gt_in[target_gt] = gt_in.get(target_gt, 0) + 1

    detection_by_id: dict[int, tuple[int, np.ndarray]] = {}
    for detection in video.detections:
        for node_id, coordinate in zip(detection.node_ids, detection.coords_zyx, strict=True):
            key = int(node_id)
            if key in detection_by_id:
                raise EdgeDataError("detector node identities are not unique")
            detection_by_id[key] = (int(detection.frame_index), np.asarray(coordinate, dtype=np.float64))
    scale = np.asarray(scorer.config.scale_zyx, dtype=np.float64)

    feature_by_pair: dict[tuple[int, int], np.ndarray] = {}
    logit_by_pair: dict[tuple[int, int], float] = {}
    for context, source, target in zip(
        video.contexts, video.detections[:-1], video.detections[1:], strict=True
    ):
        previous_velocity: dict[int, np.ndarray | None] = {}
        for node_id, coordinate in zip(source.node_ids, source.coords_zyx, strict=True):
            source_id = int(node_id)
            parent_id = selected_in.get(source_id)
            parent = detection_by_id.get(parent_id) if parent_id is not None else None
            if parent is None or parent[0] != int(source.frame_index) - 1:
                previous_velocity[source_id] = None
            else:
                previous_velocity[source_id] = (
                    np.asarray(coordinate, dtype=np.float64) - parent[1]
                ) * scale
        candidates, checkpoint_logits, residual_features = scorer.candidate_features(
            context,
            source,
            target,
            previous_velocity_um=previous_velocity,
        )
        feature_rows = residual_features.detach().cpu().numpy()
        logit_rows = checkpoint_logits.detach().cpu().numpy()
        for row, (source_id, target_id) in enumerate(
            zip(candidates.source_ids, candidates.target_ids, strict=True)
        ):
            pair = (int(source_id), int(target_id))
            if pair in feature_by_pair:
                raise EdgeDataError("sparse candidate pair is duplicated across contexts")
            feature_by_pair[pair] = np.asarray(feature_rows[row], dtype=np.float32)
            logit_by_pair[pair] = float(logit_rows[row])

    true_features: list[np.ndarray] = []
    competitor_features: list[np.ndarray] = []
    true_logits: list[float] = []
    competitor_logits: list[float] = []
    kinds: list[int] = []
    true_pairs: list[tuple[int, int]] = []
    competitor_pairs: list[tuple[int, int]] = []
    reachable_misses = 0
    skipped_division = 0
    skipped_without_conflict = 0
    seen_rows: set[tuple[tuple[int, int], tuple[int, int], int]] = set()
    for true_pair in sorted(feature_by_pair):
        source_id, target_id = true_pair
        source_gt = detection_to_gt.get(source_id)
        target_gt = detection_to_gt.get(target_id)
        if source_gt is None or target_gt is None or (int(source_gt), int(target_gt)) not in gt_edges:
            continue
        if gt_out.get(int(source_gt), 0) != 1 or gt_in.get(int(target_gt), 0) != 1:
            skipped_division += 1
            continue
        if true_pair in selected:
            continue
        reachable_misses += 1
        local: list[tuple[tuple[int, int], int]] = []
        source_target = selected_out.get(source_id)
        if source_target is not None and source_target != target_id:
            local.append(((source_id, source_target), SOURCE_CONFLICT))
        target_source = selected_in.get(target_id)
        if target_source is not None and target_source != source_id:
            local.append(((target_source, target_id), TARGET_CONFLICT))
        retained = 0
        for competitor_pair, kind in local:
            if competitor_pair not in feature_by_pair or competitor_pair not in selected:
                continue
            row_identity = (true_pair, competitor_pair, kind)
            if row_identity in seen_rows:
                continue
            seen_rows.add(row_identity)
            true_features.append(feature_by_pair[true_pair])
            competitor_features.append(feature_by_pair[competitor_pair])
            true_logits.append(logit_by_pair[true_pair])
            competitor_logits.append(logit_by_pair[competitor_pair])
            kinds.append(kind)
            true_pairs.append(true_pair)
            competitor_pairs.append(competitor_pair)
            retained += 1
        if retained == 0:
            skipped_without_conflict += 1

    count = len(true_pairs)
    return ConflictPairExamples(
        true_features=(
            np.stack(true_features)
            if true_features
            else np.empty((0, FEATURE_COUNT), dtype=np.float32)
        ),
        competitor_features=(
            np.stack(competitor_features)
            if competitor_features
            else np.empty((0, FEATURE_COUNT), dtype=np.float32)
        ),
        true_checkpoint_logits=np.asarray(true_logits, dtype=np.float32),
        competitor_checkpoint_logits=np.asarray(competitor_logits, dtype=np.float32),
        conflict_kind=np.asarray(kinds, dtype=np.int8),
        true_pairs=np.asarray(true_pairs, dtype=np.int64).reshape(count, 2),
        competitor_pairs=np.asarray(competitor_pairs, dtype=np.int64).reshape(count, 2),
        candidate_count=len(feature_by_pair),
        reachable_miss_count=reachable_misses,
        source_conflict_count=sum(kind == SOURCE_CONFLICT for kind in kinds),
        target_conflict_count=sum(kind == TARGET_CONFLICT for kind in kinds),
        skipped_division_truth=skipped_division,
        skipped_without_candidate_conflict=skipped_without_conflict,
        candidate_topology_sha256=_topology_sha256(tuple(feature_by_pair)),
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
    "ConflictPairExamples",
    "FEATURE_COUNT",
    "MATCH_RADIUS_UM",
    "SOURCE_CONFLICT",
    "TARGET_CONFLICT",
    "EdgeDataError",
    "EdgeExamples",
    "conflict_examples_from_video",
    "examples_from_video",
    "match_detections_to_gt",
]
