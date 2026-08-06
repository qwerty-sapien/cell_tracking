#!/usr/bin/env python3
"""Deterministic label-only synthetic division model.

This module implements the frozen ``SYNTH-DIV-SIM2REAL-v2-REACHABLE`` two-stage
model: a reachable-only conditional daughter-pair ranker followed by an
importance-weighted parent division-event classifier over every eligible
parent.  Sequence volumes are deliberately never indexed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


SCHEMA = "celltrack-alpha.synth-div-sim2real-reachable-model.v2"
SCORE_SCHEMA = "celltrack-alpha.synth-div-sim2real-scores.v1"
REACHABILITY_SCHEMA = "celltrack-alpha.synth-div-sim2real-reachability-audit.v2"
NODE = "SYNTH-DIV-SIM2REAL-v2-REACHABLE"
AMENDMENT_2_SHA256 = "0389a6e81ef09150b98ebf3d274e4254acaf0f844656f82e79e016d076d428d5"
MINIMUM_REACHABILITY_COVERAGE = 0.95
VOXEL_UM_POOLED = np.asarray([1.625, 1.625, 1.625], dtype=np.float64)
MAX_NEXT_FRAME_CANDIDATES = 8
MEDIAN_STEP_UM = 1.86
LAG1_PERSISTENCE = 0.30
SISTER_SEPARATION_UM = 7.24
POSITIVE_IMPORTANCE_WEIGHT = 0.06381302740413997
NEGATIVE_IMPORTANCE_WEIGHT = 1.0397641847164156
REAL_TARGET_RATE = 0.0026
DEFAULT_RUNTIME_CAP_SECONDS = 4.0 * 60.0 * 60.0

PAIR_FEATURE_NAMES = (
    "near_daughter_step_um",
    "far_daughter_step_um",
    "mean_step_abs_residual_from_1p86_um",
    "daughter_step_balance_um",
    "sister_separation_um",
    "sister_separation_abs_residual_from_7p24_um",
    "daughter_branch_cosine",
    "daughter_branch_angle_rad",
    "daughter_midpoint_residual_from_persistence_0p30_um",
    "lag1_directional_persistence_residual_from_0p30",
    "local_next_frame_candidate_density",
    "parent_track_age",
    "parent_previous_available",
)

EVENT_FEATURE_NAMES = (
    "parent_lag1_step_um",
    "parent_lag1_step_abs_residual_from_1p86_um",
    "parent_track_age",
    "parent_previous_available",
    "local_next_frame_candidate_density",
    "best_pair_logit",
    "best_pair_logit_margin",
) + tuple(f"best_pair__{name}" for name in PAIR_FEATURE_NAMES)


class ContractError(ValueError):
    """Raised when an input or frozen model violates the lane contract."""


@dataclass(frozen=True)
class Deadline:
    started: float
    cap_seconds: float

    @classmethod
    def start(cls, cap_seconds: float) -> "Deadline":
        if not math.isfinite(cap_seconds) or cap_seconds <= 0.0:
            raise ContractError("runtime cap must be a finite positive number of seconds")
        return cls(time.monotonic(), float(cap_seconds))

    def check(self, context: str) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed > self.cap_seconds:
            raise TimeoutError(
                f"runtime cap exceeded during {context}: "
                f"{elapsed:.3f}s > {self.cap_seconds:.3f}s"
            )


@dataclass(frozen=True)
class SequenceLabels:
    sequence_id: str
    path: Path
    nodes: np.ndarray
    edges: np.ndarray
    divisions: np.ndarray
    voxel_um_pooled: np.ndarray
    label_sha256: str

    @property
    def times(self) -> np.ndarray:
        return self.nodes[:, 0].astype(np.int64, copy=False)

    @property
    def coords_zyx(self) -> np.ndarray:
        return self.nodes[:, 1:4].astype(np.float64, copy=False)

    @cached_property
    def physical_coords_zyx_um(self) -> np.ndarray:
        """One immutable realization of the frozen physical-coordinate view."""

        values = self.coords_zyx * self.voxel_um_pooled
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class LineageContext:
    previous: np.ndarray
    track_age: np.ndarray
    children: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ParentCandidates:
    parent_index: int
    next_indices: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    features: np.ndarray
    true_pair: tuple[int, int] | None


@dataclass(frozen=True)
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # intercept first

    def logits(self, features: np.ndarray) -> np.ndarray:
        matrix = _feature_matrix(features, self.mean.size, "logistic inference")
        standardized = (matrix - self.mean) / self.scale
        return self.coef[0] + standardized @ self.coef[1:]

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logits(features))


@dataclass(frozen=True)
class ModelBundle:
    pair: LogisticModel
    event: LogisticModel
    metadata: dict[str, Any]
    json_sha256: str | None = None
    arrays_sha256: str | None = None


def _require_exact_keys(mapping: Any, expected: set[str], context: str) -> None:
    if not isinstance(mapping, dict):
        raise ContractError(f"{context} must be a JSON object")
    actual = set(mapping)
    if actual != expected:
        raise ContractError(
            f"{context} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _integer_array(values: np.ndarray, context: str, ndim: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ContractError(f"{context} must have {ndim} dimensions, got {array.shape}")
    if array.dtype.kind not in "iu":
        raise ContractError(f"{context} must have an integer dtype, got {array.dtype}")
    return array.astype(np.int64, copy=False)


def _hash_label_arrays(arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(name.encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def split_residue(sequence_id: str) -> int:
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ContractError("sequence ID must be a non-empty string")
    return int(hashlib.sha256(sequence_id.encode("utf-8")).hexdigest(), 16) % 10


def split_name(sequence_id: str) -> str:
    residue = split_residue(sequence_id)
    if residue <= 7:
        return "train"
    return "validation" if residue == 8 else "synthetic_test"


def _coverage_counts(total: int, reachable: int) -> dict[str, Any]:
    if total < 0 or reachable < 0 or reachable > total:
        raise ContractError("invalid reachability counts")
    return {
        "total_division_parents": int(total),
        "reachable_division_parents": int(reachable),
        "unreachable_division_parents": int(total - reachable),
        "coverage_rate": float(reachable / total) if total else None,
    }


def load_sequence(path: Path | str, *, sequence_id: str | None = None) -> SequenceLabels:
    """Load and strictly validate only the four label members of a v3 NPZ.

    ``volumes`` may be present but is never accessed.  This is important for
    both the data firewall and peak memory use.
    """

    source = Path(path)
    sid = source.stem if sequence_id is None else sequence_id
    if not source.is_file():
        raise ContractError(f"sequence file does not exist: {source}")
    required = {"nodes", "edges", "divisions", "voxel_um_pooled"}
    try:
        with np.load(source, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ContractError(f"{source}: missing NPZ members {sorted(missing)}")
            # Deliberately use explicit member names; never iterate/access volumes.
            nodes_raw = archive["nodes"]
            edges_raw = archive["edges"]
            divisions_raw = archive["divisions"]
            voxel_raw = archive["voxel_um_pooled"]
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"failed to read label members from {source}: {exc}") from exc

    nodes = np.asarray(nodes_raw)
    if nodes.ndim != 2 or nodes.shape[1] != 5:
        raise ContractError(f"{source}: nodes must have shape [M,5], got {nodes.shape}")
    if nodes.dtype.kind not in "fiu" or not np.all(np.isfinite(nodes)):
        raise ContractError(f"{source}: nodes must be finite numeric values")
    nodes = nodes.astype(np.float64, copy=False)
    for column, name in ((0, "time"), (4, "track_id")):
        values = nodes[:, column]
        if np.any(values < 0) or not np.array_equal(values, np.rint(values)):
            raise ContractError(f"{source}: node {name} values must be nonnegative integers")

    edges = _integer_array(edges_raw, f"{source}: edges", 2)
    if edges.shape[1:] != (2,):
        raise ContractError(f"{source}: edges must have shape [E,2], got {edges.shape}")
    divisions = _integer_array(divisions_raw, f"{source}: divisions", 1)
    voxel = np.asarray(voxel_raw, dtype=np.float64)
    if voxel.shape != (3,) or not np.all(np.isfinite(voxel)):
        raise ContractError(f"{source}: voxel_um_pooled must be finite shape [3]")
    if not np.array_equal(voxel, VOXEL_UM_POOLED):
        raise ContractError(
            f"{source}: voxel_um_pooled must be exactly {VOXEL_UM_POOLED.tolist()}, "
            f"got {voxel.tolist()}"
        )

    count = nodes.shape[0]
    if edges.size and (np.any(edges < 0) or np.any(edges >= count)):
        raise ContractError(f"{source}: edge indices are out of node bounds")
    if divisions.size and (np.any(divisions < 0) or np.any(divisions >= count)):
        raise ContractError(f"{source}: division indices are out of node bounds")
    if len({tuple(row) for row in edges.tolist()}) != len(edges):
        raise ContractError(f"{source}: duplicate edges are forbidden")
    if np.unique(divisions).size != divisions.size:
        raise ContractError(f"{source}: duplicate division parent indices are forbidden")

    times = nodes[:, 0].astype(np.int64, copy=False)
    indegree = np.zeros(count, dtype=np.int16)
    outdegree = np.zeros(count, dtype=np.int16)
    for source_index, target_index in edges:
        if times[target_index] != times[source_index] + 1:
            raise ContractError(f"{source}: every edge must advance exactly one frame")
        indegree[target_index] += 1
        outdegree[source_index] += 1
    if np.any(indegree > 1):
        raise ContractError(f"{source}: lineage merges (indegree > 1) are forbidden")
    if np.any(outdegree > 2):
        raise ContractError(f"{source}: outdegree > 2 is forbidden")
    division_set = set(int(value) for value in divisions)
    actual_divisions = set(int(value) for value in np.flatnonzero(outdegree == 2))
    if division_set != actual_divisions:
        raise ContractError(
            f"{source}: divisions must exactly name outdegree-2 parents; "
            f"declared_only={sorted(division_set - actual_divisions)[:8]}, "
            f"edges_only={sorted(actual_divisions - division_set)[:8]}"
        )

    label_hash = _hash_label_arrays(
        (("nodes", nodes), ("edges", edges), ("divisions", divisions), ("voxel", voxel))
    )
    return SequenceLabels(sid, source, nodes, edges, divisions, voxel, label_hash)


def sequence_from_arrays(
    *,
    sequence_id: str,
    nodes: np.ndarray,
    edges: np.ndarray,
    divisions: np.ndarray | None = None,
    voxel_um_pooled: np.ndarray = VOXEL_UM_POOLED,
) -> SequenceLabels:
    """Build a validated label sequence from graph arrays without image data.

    This is the reusable real-transfer/integration seam.  Validation is kept
    identical to file-backed sequences by round-tripping only the small label
    arrays through an in-memory contract implementation.
    """

    if not isinstance(sequence_id, str) or not sequence_id:
        raise ContractError("sequence ID must be a non-empty string")
    node_values = np.asarray(nodes)
    if node_values.ndim != 2 or node_values.shape[1] != 5:
        raise ContractError(f"nodes must have shape [M,5], got {node_values.shape}")
    if node_values.dtype.kind not in "fiu" or not np.all(np.isfinite(node_values)):
        raise ContractError("nodes must be finite numeric values")
    node_values = node_values.astype(np.float64, copy=False)
    for column, name in ((0, "time"), (4, "track_id")):
        values = node_values[:, column]
        if np.any(values < 0) or not np.array_equal(values, np.rint(values)):
            raise ContractError(f"node {name} values must be nonnegative integers")
    edge_values = _integer_array(np.asarray(edges), "edges", 2)
    if edge_values.shape[1:] != (2,):
        raise ContractError(f"edges must have shape [E,2], got {edge_values.shape}")
    division_values = _integer_array(
        np.empty(0, dtype=np.int64) if divisions is None else np.asarray(divisions),
        "divisions",
        1,
    )
    voxel_values = np.asarray(voxel_um_pooled, dtype=np.float64)
    if voxel_values.shape != (3,) or not np.array_equal(voxel_values, VOXEL_UM_POOLED):
        raise ContractError(f"voxel_um_pooled must be exactly {VOXEL_UM_POOLED.tolist()}")
    count = node_values.shape[0]
    if edge_values.size and (np.any(edge_values < 0) or np.any(edge_values >= count)):
        raise ContractError("edge indices are out of node bounds")
    if division_values.size and (
        np.any(division_values < 0) or np.any(division_values >= count)
    ):
        raise ContractError("division indices are out of node bounds")
    if len({tuple(row) for row in edge_values.tolist()}) != len(edge_values):
        raise ContractError("duplicate edges are forbidden")
    if np.unique(division_values).size != division_values.size:
        raise ContractError("duplicate division parent indices are forbidden")
    times = node_values[:, 0].astype(np.int64, copy=False)
    indegree = np.zeros(count, dtype=np.int16)
    outdegree = np.zeros(count, dtype=np.int16)
    for source_index, target_index in edge_values:
        if times[target_index] != times[source_index] + 1:
            raise ContractError("every edge must advance exactly one frame")
        indegree[target_index] += 1
        outdegree[source_index] += 1
    if np.any(indegree > 1) or np.any(outdegree > 2):
        raise ContractError("lineage graph contains a merge or outdegree > 2")
    declared = set(int(value) for value in division_values)
    actual = set(int(value) for value in np.flatnonzero(outdegree == 2))
    if declared != actual:
        raise ContractError("divisions must exactly name outdegree-2 parents")
    label_hash = _hash_label_arrays(
        (
            ("nodes", node_values),
            ("edges", edge_values),
            ("divisions", division_values),
            ("voxel", voxel_values),
        )
    )
    return SequenceLabels(
        sequence_id,
        Path(f"<arrays:{sequence_id}>"),
        node_values,
        edge_values,
        division_values,
        voxel_values,
        label_hash,
    )


def lineage_context(sequence: SequenceLabels) -> LineageContext:
    count = sequence.nodes.shape[0]
    previous = np.full(count, -1, dtype=np.int64)
    children_lists: list[list[int]] = [[] for _ in range(count)]
    for parent, child in sequence.edges:
        previous[int(child)] = int(parent)
        children_lists[int(parent)].append(int(child))
    children = tuple(tuple(sorted(values)) for values in children_lists)
    age = np.ones(count, dtype=np.int64)
    # Edges are guaranteed to advance time, so ordering by time is topological.
    for index in np.argsort(sequence.times, kind="stable"):
        parent = previous[index]
        if parent >= 0:
            age[index] = age[parent] + 1
    return LineageContext(previous, age, children)


def eligible_parent_indices(sequence: SequenceLabels) -> np.ndarray:
    times = sequence.times
    if times.size == 0:
        return np.empty(0, dtype=np.int64)
    frame_counts = {int(t): int(np.count_nonzero(times == t)) for t in np.unique(times)}
    return np.asarray(
        [index for index, t in enumerate(times) if frame_counts.get(int(t) + 1, 0) >= 2],
        dtype=np.int64,
    )


def _unit_cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def nearest_next_frame_indices(
    sequence: SequenceLabels,
    parent_index: int,
    *,
    maximum_candidates: int = MAX_NEXT_FRAME_CANDIDATES,
) -> tuple[int, ...]:
    """Return the frozen deterministic nearest-next-frame candidate universe."""

    if maximum_candidates < 2 or maximum_candidates > MAX_NEXT_FRAME_CANDIDATES:
        raise ContractError(f"maximum_candidates must be in [2,{MAX_NEXT_FRAME_CANDIDATES}]")
    if parent_index < 0 or parent_index >= sequence.nodes.shape[0]:
        raise ContractError("parent index is out of bounds")
    next_frame = int(sequence.times[parent_index]) + 1
    available = np.flatnonzero(sequence.times == next_frame)
    if available.size < 2:
        return ()
    coordinates = sequence.physical_coords_zyx_um
    distances = np.linalg.norm(coordinates[available] - coordinates[parent_index], axis=1)
    order = np.lexsort((available, distances))
    return tuple(int(value) for value in available[order[:maximum_candidates]])


def pair_features(
    sequence: SequenceLabels,
    context: LineageContext,
    parent_index: int,
    daughter_a: int,
    daughter_b: int,
    *,
    local_density: int,
) -> np.ndarray:
    if daughter_a == daughter_b:
        raise ContractError("a daughter pair must contain two distinct nodes")
    coords = sequence.physical_coords_zyx_um
    parent = coords[parent_index]
    daughter_rows = [(int(daughter_a), coords[daughter_a]), (int(daughter_b), coords[daughter_b])]
    daughter_rows.sort(key=lambda row: (float(np.linalg.norm(row[1] - parent)), row[0]))
    near_index, near = daughter_rows[0]
    far_index, far = daughter_rows[1]
    del near_index, far_index
    near_vector = near - parent
    far_vector = far - parent
    near_step = float(np.linalg.norm(near_vector))
    far_step = float(np.linalg.norm(far_vector))
    mean_step = 0.5 * (near_step + far_step)
    sister = float(np.linalg.norm(near - far))
    branch_cosine = _unit_cosine(near_vector, far_vector)
    midpoint = 0.5 * (near + far)
    previous_index = int(context.previous[parent_index])
    has_previous = previous_index >= 0
    previous_velocity = parent - coords[previous_index] if has_previous else np.zeros(3)
    predicted_midpoint = parent + LAG1_PERSISTENCE * previous_velocity
    midpoint_motion = midpoint - parent
    directional = _unit_cosine(previous_velocity, midpoint_motion) if has_previous else 0.0
    values = np.asarray(
        [
            near_step,
            far_step,
            abs(mean_step - MEDIAN_STEP_UM),
            abs(far_step - near_step),
            sister,
            abs(sister - SISTER_SEPARATION_UM),
            branch_cosine,
            math.acos(branch_cosine),
            float(np.linalg.norm(midpoint - predicted_midpoint)),
            directional - LAG1_PERSISTENCE if has_previous else 0.0,
            float(local_density),
            float(context.track_age[parent_index]),
            float(has_previous),
        ],
        dtype=np.float64,
    )
    if values.shape != (len(PAIR_FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ContractError("pair feature computation produced invalid values")
    return values


def enumerate_parent_candidates(
    sequence: SequenceLabels,
    parent_index: int,
    *,
    context: LineageContext | None = None,
    maximum_candidates: int = MAX_NEXT_FRAME_CANDIDATES,
) -> ParentCandidates:
    lineage = lineage_context(sequence) if context is None else context
    selected = nearest_next_frame_indices(
        sequence, parent_index, maximum_candidates=maximum_candidates
    )
    if len(selected) < 2:
        return ParentCandidates(parent_index, (), (), np.empty((0, len(PAIR_FEATURE_NAMES))), None)
    pairs = tuple(tuple(sorted(pair)) for pair in itertools.combinations(selected, 2))
    features = np.vstack(
        [
            pair_features(
                sequence,
                lineage,
                parent_index,
                first,
                second,
                local_density=len(selected),
            )
            for first, second in pairs
        ]
    )
    children = lineage.children[parent_index]
    true_pair = tuple(sorted(children)) if len(children) == 2 else None
    return ParentCandidates(parent_index, selected, pairs, features, true_pair)


def _feature_matrix(values: np.ndarray, feature_count: int, context: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != feature_count:
        raise ContractError(
            f"{context}: expected feature matrix [N,{feature_count}], got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ContractError(f"{context}: features contain non-finite values")
    return matrix


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _penalized_objective(
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    coef: np.ndarray,
    l2: float,
) -> float:
    logits = design @ coef
    losses = np.logaddexp(0.0, logits) - labels * logits
    return float(np.dot(weights, losses) / weights.sum() + 0.5 * l2 * np.dot(coef[1:], coef[1:]))


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    l2: float = 1.0,
    maximum_iterations: int = 100,
    tolerance: float = 1e-10,
    deadline: Deadline | None = None,
) -> LogisticModel:
    """Fit deterministic standardized L2 logistic regression by damped Newton steps."""

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ContractError("logistic features must be a finite two-dimensional matrix")
    y = np.asarray(labels, dtype=np.float64)
    if y.shape != (matrix.shape[0],) or not np.all(np.isin(y, (0.0, 1.0))):
        raise ContractError("logistic labels must be a binary vector matching feature rows")
    if matrix.shape[0] == 0 or np.unique(y).size != 2:
        raise ContractError("logistic training requires nonempty examples from both classes")
    if not math.isfinite(l2) or l2 < 0.0:
        raise ContractError("logistic L2 regularization must be finite and nonnegative")
    if maximum_iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ContractError("invalid logistic iteration controls")
    if sample_weight is None:
        weights = np.ones(matrix.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != y.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ContractError("sample weights must be finite, positive, and match labels")

    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (matrix - mean) / scale
    design = np.column_stack((np.ones(matrix.shape[0]), standardized))
    coef = np.zeros(design.shape[1], dtype=np.float64)
    weighted_rate = float(np.dot(weights, y) / weights.sum())
    weighted_rate = min(max(weighted_rate, 1e-12), 1.0 - 1e-12)
    coef[0] = math.log(weighted_rate / (1.0 - weighted_rate))
    penalty = np.zeros(coef.size, dtype=np.float64)
    penalty[1:] = l2
    objective = _penalized_objective(design, y, weights, coef, l2)

    for iteration in range(maximum_iterations):
        if deadline is not None:
            deadline.check(f"logistic iteration {iteration}")
        probabilities = _sigmoid(design @ coef)
        gradient = design.T @ (weights * (probabilities - y)) / weights.sum()
        gradient += penalty * coef
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design / weights.sum()
        hessian.flat[:: hessian.shape[0] + 1] += penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            raise ContractError("logistic optimizer produced a non-finite Newton step")
        step_size = 1.0
        accepted = False
        for _ in range(30):
            proposal = coef - step_size * step
            proposed_objective = _penalized_objective(design, y, weights, proposal, l2)
            if proposed_objective <= objective + 1e-14:
                coef = proposal
                objective = proposed_objective
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break
        if float(np.max(np.abs(step_size * step))) <= tolerance:
            break
    if not np.all(np.isfinite(coef)):
        raise ContractError("logistic optimizer produced non-finite coefficients")
    return LogisticModel(mean, scale, coef)


def pair_conditional_probabilities(pair_model: LogisticModel, features: np.ndarray) -> np.ndarray:
    logits = pair_model.logits(features)
    if logits.size == 0:
        return np.empty(0, dtype=np.float64)
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum()
    if not np.all(np.isfinite(probabilities)) or not np.isclose(probabilities.sum(), 1.0):
        raise ContractError("conditional pair softmax produced invalid probabilities")
    return probabilities


def event_features_for_parent(
    sequence: SequenceLabels,
    context: LineageContext,
    candidates: ParentCandidates,
    pair_model: LogisticModel,
) -> np.ndarray:
    if not candidates.pairs:
        raise ContractError("event features require at least one daughter pair")
    pair_logits = pair_model.logits(candidates.features)
    ranking = np.lexsort(
        (
            np.asarray([pair[1] for pair in candidates.pairs]),
            np.asarray([pair[0] for pair in candidates.pairs]),
            -pair_logits,
        )
    )
    best = int(ranking[0])
    second_logit = float(pair_logits[ranking[1]]) if len(ranking) > 1 else float(pair_logits[best])
    parent = candidates.parent_index
    previous = int(context.previous[parent])
    has_previous = previous >= 0
    coordinates = sequence.physical_coords_zyx_um
    lag1_step = float(np.linalg.norm(coordinates[parent] - coordinates[previous])) if has_previous else 0.0
    prefix = np.asarray(
        [
            lag1_step,
            abs(lag1_step - MEDIAN_STEP_UM) if has_previous else 0.0,
            float(context.track_age[parent]),
            float(has_previous),
            float(len(candidates.next_indices)),
            float(pair_logits[best]),
            float(pair_logits[best] - second_logit),
        ],
        dtype=np.float64,
    )
    result = np.concatenate((prefix, candidates.features[best]))
    if result.shape != (len(EVENT_FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise ContractError("event feature computation produced invalid values")
    return result


def audit_reachability(
    sequence_paths: Sequence[Path | str],
    *,
    runtime_cap_seconds: float = DEFAULT_RUNTIME_CAP_SECONDS,
) -> dict[str, Any]:
    """Audit exact-daughter coverage in the unchanged nearest-8 universe."""

    deadline = Deadline.start(runtime_cap_seconds)
    paths = sorted((Path(path) for path in sequence_paths), key=lambda value: (value.stem, str(value)))
    if not paths:
        raise ContractError("reachability audit requires at least one sequence")
    identifiers = [path.stem for path in paths]
    if len(set(identifiers)) != len(identifiers):
        raise ContractError("reachability audit requires unique sequence IDs")
    split_totals = {
        name: {"total": 0, "reachable": 0}
        for name in ("train", "validation", "synthetic_test")
    }
    sequence_rows: list[dict[str, Any]] = []
    for path in paths:
        deadline.check(f"reachability audit loading {path.name}")
        sequence = load_sequence(path)
        context = lineage_context(sequence)
        reachable = 0
        for parent in sequence.divisions:
            deadline.check(f"reachability audit {sequence.sequence_id}")
            candidates = set(nearest_next_frame_indices(sequence, int(parent)))
            children = context.children[int(parent)]
            if len(children) != 2:
                raise ContractError("division parent does not have exactly two daughters")
            reachable += int(children[0] in candidates and children[1] in candidates)
        total = int(sequence.divisions.size)
        counts = _coverage_counts(total, reachable)
        split = split_name(sequence.sequence_id)
        split_totals[split]["total"] += total
        split_totals[split]["reachable"] += reachable
        sequence_rows.append(
            {
                "sequence_id": sequence.sequence_id,
                "split_residue": split_residue(sequence.sequence_id),
                "split": split,
                "label_sha256": sequence.label_sha256,
                **counts,
            }
        )
    by_split = {
        name: _coverage_counts(values["total"], values["reachable"])
        for name, values in split_totals.items()
    }
    total = sum(values["total"] for values in split_totals.values())
    reachable = sum(values["reachable"] for values in split_totals.values())
    return {
        "schema": REACHABILITY_SCHEMA,
        "node": NODE,
        "amendment2_sha256": AMENDMENT_2_SHA256,
        "maximum_next_frame_candidates": MAX_NEXT_FRAME_CANDIDATES,
        "minimum_coverage_rate": MINIMUM_REACHABILITY_COVERAGE,
        "sequence_count": len(sequence_rows),
        "sequences": sequence_rows,
        "by_split": by_split,
        "overall": _coverage_counts(total, reachable),
    }


def _validate_counts_object(value: Any, context: str) -> dict[str, Any]:
    expected = {
        "total_division_parents",
        "reachable_division_parents",
        "unreachable_division_parents",
        "coverage_rate",
    }
    _require_exact_keys(value, expected, context)
    total = value["total_division_parents"]
    reachable = value["reachable_division_parents"]
    unreachable = value["unreachable_division_parents"]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (total, reachable, unreachable)):
        raise ContractError(f"{context} counts must be integers")
    expected_counts = _coverage_counts(total, reachable)
    if value != expected_counts or unreachable != total - reachable:
        raise ContractError(f"{context} counts/rate are internally inconsistent")
    return value


def validate_reachability_report(
    report: Any,
    *,
    sequence_paths: Sequence[Path | str] | None = None,
    deadline: Deadline | None = None,
) -> dict[str, Any]:
    """Strictly validate an audit, optionally binding it to exact label arrays."""

    _require_exact_keys(
        report,
        {
            "schema",
            "node",
            "amendment2_sha256",
            "maximum_next_frame_candidates",
            "minimum_coverage_rate",
            "sequence_count",
            "sequences",
            "by_split",
            "overall",
        },
        "reachability report",
    )
    expected_scalars = (
        report["schema"] == REACHABILITY_SCHEMA,
        report["node"] == NODE,
        report["amendment2_sha256"] == AMENDMENT_2_SHA256,
        report["maximum_next_frame_candidates"] == MAX_NEXT_FRAME_CANDIDATES,
        report["minimum_coverage_rate"] == MINIMUM_REACHABILITY_COVERAGE,
    )
    if not all(expected_scalars):
        raise ContractError("reachability report frozen contract mismatch")
    if not isinstance(report["sequences"], list) or report["sequence_count"] != len(report["sequences"]):
        raise ContractError("reachability report sequence count mismatch")
    sequence_map: dict[str, dict[str, Any]] = {}
    expected_sequence_keys = {
        "sequence_id",
        "split_residue",
        "split",
        "label_sha256",
        "total_division_parents",
        "reachable_division_parents",
        "unreachable_division_parents",
        "coverage_rate",
    }
    aggregated = {
        name: {"total": 0, "reachable": 0}
        for name in ("train", "validation", "synthetic_test")
    }
    for index, row in enumerate(report["sequences"]):
        _require_exact_keys(row, expected_sequence_keys, f"reachability sequence {index}")
        sequence_id = row["sequence_id"]
        if not isinstance(sequence_id, str) or not sequence_id or sequence_id in sequence_map:
            raise ContractError("reachability report has an invalid or duplicate sequence ID")
        if row["split_residue"] != split_residue(sequence_id) or row["split"] != split_name(sequence_id):
            raise ContractError(f"reachability report split mismatch for {sequence_id}")
        if not isinstance(row["label_sha256"], str) or len(row["label_sha256"]) != 64:
            raise ContractError(f"reachability report label hash invalid for {sequence_id}")
        counts = _validate_counts_object(
            {key: row[key] for key in (
                "total_division_parents",
                "reachable_division_parents",
                "unreachable_division_parents",
                "coverage_rate",
            )},
            f"reachability sequence {sequence_id}",
        )
        split = row["split"]
        aggregated[split]["total"] += counts["total_division_parents"]
        aggregated[split]["reachable"] += counts["reachable_division_parents"]
        sequence_map[sequence_id] = row
    _require_exact_keys(report["by_split"], set(aggregated), "reachability by_split")
    for split, values in aggregated.items():
        expected = _coverage_counts(values["total"], values["reachable"])
        actual = _validate_counts_object(report["by_split"][split], f"reachability split {split}")
        if actual != expected:
            raise ContractError(f"reachability split aggregate mismatch for {split}")
    overall_expected = _coverage_counts(
        sum(value["total"] for value in aggregated.values()),
        sum(value["reachable"] for value in aggregated.values()),
    )
    if _validate_counts_object(report["overall"], "reachability overall") != overall_expected:
        raise ContractError("reachability overall aggregate mismatch")

    if sequence_paths is not None:
        paths = sorted((Path(path) for path in sequence_paths), key=lambda value: (value.stem, str(value)))
        if len(paths) != len(sequence_map) or {path.stem for path in paths} != set(sequence_map):
            raise ContractError("authenticated reachability report does not cover the input sequence set")
        for path in paths:
            if deadline is not None:
                deadline.check(f"binding reachability report to {path.name}")
            sequence = load_sequence(path)
            row = sequence_map[sequence.sequence_id]
            if row["label_sha256"] != sequence.label_sha256:
                raise ContractError(f"reachability label hash mismatch for {sequence.sequence_id}")
            if row["total_division_parents"] != int(sequence.divisions.size):
                raise ContractError(f"reachability division count mismatch for {sequence.sequence_id}")
    return report


def require_reachability_gate(report: Any) -> None:
    validated = validate_reachability_report(report)
    for population in ("overall", "train", "validation", "synthetic_test"):
        counts = validated["overall"] if population == "overall" else validated["by_split"][population]
        rate = counts["coverage_rate"]
        if rate is None:
            raise ContractError(f"reachability gate has no labelled divisions in {population}")
        if rate < MINIMUM_REACHABILITY_COVERAGE:
            raise ContractError(
                f"reachability gate failed for {population}: {rate:.12f} "
                f"< {MINIMUM_REACHABILITY_COVERAGE:.12f}"
            )


def _pair_training_rows(
    sequence: SequenceLabels, deadline: Deadline
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    context = lineage_context(sequence)
    matrices: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    reachable = 0
    for parent in sequence.divisions:
        deadline.check(f"pair examples for {sequence.sequence_id}")
        candidates = enumerate_parent_candidates(sequence, int(parent), context=context)
        if candidates.true_pair is None or candidates.true_pair not in candidates.pairs:
            # Amendment 2: exclude only this unreachable conditional-head row.
            # The same parent remains in the event population below.
            continue
        reachable += 1
        labels = np.asarray([pair == candidates.true_pair for pair in candidates.pairs], dtype=np.float64)
        if labels.sum() != 1:
            raise ContractError("each division parent must contribute exactly one positive pair")
        matrices.append(candidates.features)
        label_parts.append(labels)
    if not matrices:
        return (
            np.empty((0, len(PAIR_FEATURE_NAMES))),
            np.empty(0),
            _coverage_counts(int(sequence.divisions.size), reachable),
        )
    return (
        np.vstack(matrices),
        np.concatenate(label_parts),
        _coverage_counts(int(sequence.divisions.size), reachable),
    )


def _event_training_rows(
    sequence: SequenceLabels,
    pair_model: LogisticModel,
    deadline: Deadline,
) -> tuple[np.ndarray, np.ndarray]:
    context = lineage_context(sequence)
    division_set = set(int(value) for value in sequence.divisions)
    rows: list[np.ndarray] = []
    labels: list[float] = []
    for parent in eligible_parent_indices(sequence):
        deadline.check(f"event examples for {sequence.sequence_id}")
        candidates = enumerate_parent_candidates(sequence, int(parent), context=context)
        rows.append(event_features_for_parent(sequence, context, candidates, pair_model))
        labels.append(float(int(parent) in division_set))
    if not rows:
        return np.empty((0, len(EVENT_FEATURE_NAMES))), np.empty(0)
    return np.vstack(rows), np.asarray(labels, dtype=np.float64)


def train_model(
    sequence_paths: Sequence[Path | str],
    *,
    pair_l2: float = 1.0,
    event_l2: float = 1.0,
    maximum_iterations: int = 100,
    runtime_cap_seconds: float = DEFAULT_RUNTIME_CAP_SECONDS,
    train_split_only: bool = True,
    preregistration_sha256: str | None = None,
    coverage_report_path: Path | str | None = None,
    coverage_report_sha256: str | None = None,
    amendment2_sha256: str = AMENDMENT_2_SHA256,
) -> ModelBundle:
    deadline = Deadline.start(runtime_cap_seconds)
    paths = [Path(path) for path in sequence_paths]
    if not paths:
        raise ContractError("at least one sequence path is required")
    if amendment2_sha256 != AMENDMENT_2_SHA256:
        raise ContractError("Amendment 2 SHA-256 does not match the frozen v2-reachable contract")
    if coverage_report_path is None:
        if coverage_report_sha256 is not None:
            raise ContractError("coverage report SHA-256 was supplied without a coverage report")
        coverage_report = audit_reachability(paths, runtime_cap_seconds=runtime_cap_seconds)
        validated_coverage = validate_reachability_report(
            coverage_report, sequence_paths=paths, deadline=deadline
        )
        authenticated_coverage_sha256 = hashlib.sha256(_json_bytes(coverage_report)).hexdigest()
    else:
        if coverage_report_sha256 is None:
            raise ContractError("a supplied coverage report requires its authenticated SHA-256")
        report_path = Path(coverage_report_path)
        actual_coverage_hash = sha256_file(report_path)
        if actual_coverage_hash != coverage_report_sha256:
            raise ContractError("authenticated coverage report SHA-256 mismatch")
        try:
            coverage_report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ContractError(f"failed to parse coverage report: {exc}") from exc
        validated_coverage = validate_reachability_report(
            coverage_report, sequence_paths=paths, deadline=deadline
        )
        authenticated_coverage_sha256 = actual_coverage_hash
    require_reachability_gate(validated_coverage)
    deadline.check("reachability preflight gate")

    selected = [path for path in paths if not train_split_only or split_name(path.stem) == "train"]
    if not selected:
        raise ContractError("no input sequences belong to the frozen training split")

    pair_rows: list[np.ndarray] = []
    pair_labels: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    pair_division_total = 0
    pair_division_reachable = 0
    for path in sorted(selected, key=lambda value: (value.stem, str(value))):
        deadline.check(f"loading pair sequence {path.name}")
        sequence = load_sequence(path)
        rows, labels, reachability = _pair_training_rows(sequence, deadline)
        pair_division_total += reachability["total_division_parents"]
        pair_division_reachable += reachability["reachable_division_parents"]
        if rows.size:
            pair_rows.append(rows)
            pair_labels.append(labels)
        provenance.append(
            {
                "sequence_id": sequence.sequence_id,
                "split_residue": split_residue(sequence.sequence_id),
                "label_sha256": sequence.label_sha256,
            }
        )
    if not pair_rows:
        raise ContractError("training population contains no division pair examples")
    pair_x = np.vstack(pair_rows)
    pair_y = np.concatenate(pair_labels)
    pair_model = fit_logistic(
        pair_x,
        pair_y,
        l2=pair_l2,
        maximum_iterations=maximum_iterations,
        deadline=deadline,
    )
    pair_example_count = int(pair_y.size)
    del pair_x, pair_y, pair_rows, pair_labels

    event_rows: list[np.ndarray] = []
    event_labels: list[np.ndarray] = []
    for path in sorted(selected, key=lambda value: (value.stem, str(value))):
        deadline.check(f"loading event sequence {path.name}")
        sequence = load_sequence(path)
        rows, labels = _event_training_rows(sequence, pair_model, deadline)
        if rows.size:
            event_rows.append(rows)
            event_labels.append(labels)
    if not event_rows:
        raise ContractError("training population contains no eligible parent examples")
    event_x = np.vstack(event_rows)
    event_y = np.concatenate(event_labels)
    weights = np.where(
        event_y == 1.0, POSITIVE_IMPORTANCE_WEIGHT, NEGATIVE_IMPORTANCE_WEIGHT
    )
    event_model = fit_logistic(
        event_x,
        event_y,
        sample_weight=weights,
        l2=event_l2,
        maximum_iterations=maximum_iterations,
        deadline=deadline,
    )
    population = {
        "sequence_count": len(selected),
        "pair_examples": pair_example_count,
        "pair_division_reachability": _coverage_counts(
            pair_division_total, pair_division_reachable
        ),
        "event_examples": int(event_y.size),
        "event_positives": int(event_y.sum()),
    }
    metadata = {
        "training_sequences": provenance,
        "population": population,
        "pair_l2": float(pair_l2),
        "event_l2": float(event_l2),
        "maximum_iterations": int(maximum_iterations),
        "preregistration_sha256": preregistration_sha256,
        "amendment2_sha256": AMENDMENT_2_SHA256,
        "reachability_report_sha256": authenticated_coverage_sha256,
        "reachability_gate": {
            "minimum_coverage_rate": MINIMUM_REACHABILITY_COVERAGE,
            "overall": validated_coverage["overall"],
            "by_split": validated_coverage["by_split"],
            "sequence_count": validated_coverage["sequence_count"],
        },
    }
    return ModelBundle(pair_model, event_model, metadata)


def _model_document(bundle: ModelBundle, arrays_name: str, arrays_hash: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "node": NODE,
        "arrays_file": arrays_name,
        "arrays_sha256": arrays_hash,
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "event_feature_names": list(EVENT_FEATURE_NAMES),
        "priors": {
            "median_step_um_per_frame": MEDIAN_STEP_UM,
            "lag1_directional_persistence": LAG1_PERSISTENCE,
            "sister_separation_um": SISTER_SEPARATION_UM,
        },
        "calibration": {
            "real_target_rate": REAL_TARGET_RATE,
            "positive_importance_weight": POSITIVE_IMPORTANCE_WEIGHT,
            "negative_importance_weight": NEGATIVE_IMPORTANCE_WEIGHT,
            "additional_logit_shift": 0.0,
        },
        "training": bundle.metadata,
    }


def save_model(bundle: ModelBundle, json_path: Path | str) -> tuple[Path, Path]:
    destination = Path(json_path)
    if destination.suffix.lower() != ".json":
        raise ContractError("model manifest path must end in .json")
    arrays_path = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_arrays = arrays_path.with_name(f".{arrays_path.name}.{os.getpid()}.tmp.npz")
    temporary_json = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        np.savez_compressed(
            temporary_arrays,
            pair_mean=np.asarray(bundle.pair.mean, dtype=np.float64),
            pair_scale=np.asarray(bundle.pair.scale, dtype=np.float64),
            pair_coef=np.asarray(bundle.pair.coef, dtype=np.float64),
            event_mean=np.asarray(bundle.event.mean, dtype=np.float64),
            event_scale=np.asarray(bundle.event.scale, dtype=np.float64),
            event_coef=np.asarray(bundle.event.coef, dtype=np.float64),
        )
        arrays_hash = sha256_file(temporary_arrays)
        document = _model_document(bundle, arrays_path.name, arrays_hash)
        temporary_json.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_arrays, arrays_path)
        os.replace(temporary_json, destination)
    finally:
        temporary_arrays.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
    return destination, arrays_path


def _validate_logistic_arrays(
    mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, count: int, context: str
) -> LogisticModel:
    arrays = [np.asarray(value) for value in (mean, scale, coef)]
    if any(value.dtype != np.dtype("float64") for value in arrays):
        raise ContractError(f"{context} arrays must be float64")
    if arrays[0].shape != (count,) or arrays[1].shape != (count,) or arrays[2].shape != (count + 1,):
        raise ContractError(f"{context} model array shapes do not match its feature schema")
    if not all(np.all(np.isfinite(value)) for value in arrays) or np.any(arrays[1] <= 0.0):
        raise ContractError(f"{context} model arrays contain invalid values")
    return LogisticModel(*(value.copy() for value in arrays))


def load_model(json_path: Path | str) -> ModelBundle:
    source = Path(json_path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractError(f"failed to parse model JSON {source}: {exc}") from exc
    _require_exact_keys(
        document,
        {
            "schema",
            "node",
            "arrays_file",
            "arrays_sha256",
            "pair_feature_names",
            "event_feature_names",
            "priors",
            "calibration",
            "training",
        },
        "model manifest",
    )
    if document["schema"] != SCHEMA or document["node"] != NODE:
        raise ContractError("model schema or node does not match SYNTH-DIV-SIM2REAL-v2-REACHABLE")
    if document["pair_feature_names"] != list(PAIR_FEATURE_NAMES):
        raise ContractError("pair feature schema mismatch")
    if document["event_feature_names"] != list(EVENT_FEATURE_NAMES):
        raise ContractError("event feature schema mismatch")
    _require_exact_keys(
        document["priors"],
        {"median_step_um_per_frame", "lag1_directional_persistence", "sister_separation_um"},
        "model priors",
    )
    expected_priors = (MEDIAN_STEP_UM, LAG1_PERSISTENCE, SISTER_SEPARATION_UM)
    actual_priors = document["priors"]
    if actual_priors != {
        "median_step_um_per_frame": expected_priors[0],
        "lag1_directional_persistence": expected_priors[1],
        "sister_separation_um": expected_priors[2],
    }:
        raise ContractError("model published-prior constants mismatch")
    _require_exact_keys(
        document["calibration"],
        {
            "real_target_rate",
            "positive_importance_weight",
            "negative_importance_weight",
            "additional_logit_shift",
        },
        "model calibration",
    )
    calibration = document["calibration"]
    expected_calibration = {
        "real_target_rate": REAL_TARGET_RATE,
        "positive_importance_weight": POSITIVE_IMPORTANCE_WEIGHT,
        "negative_importance_weight": NEGATIVE_IMPORTANCE_WEIGHT,
        "additional_logit_shift": 0.0,
    }
    if calibration != expected_calibration:
        raise ContractError("model importance calibration mismatch or double-shift detected")
    arrays_name = document["arrays_file"]
    if not isinstance(arrays_name, str) or Path(arrays_name).name != arrays_name:
        raise ContractError("arrays_file must be a safe sibling basename")
    arrays_path = source.parent / arrays_name
    arrays_hash = sha256_file(arrays_path)
    if arrays_hash != document["arrays_sha256"]:
        raise ContractError("model NPZ SHA-256 mismatch")
    expected_arrays = {"pair_mean", "pair_scale", "pair_coef", "event_mean", "event_scale", "event_coef"}
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_arrays:
                raise ContractError("model NPZ member set mismatch")
            values = {name: archive[name] for name in expected_arrays}
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"failed to load model arrays: {exc}") from exc
    pair = _validate_logistic_arrays(
        values["pair_mean"], values["pair_scale"], values["pair_coef"], len(PAIR_FEATURE_NAMES), "pair"
    )
    event = _validate_logistic_arrays(
        values["event_mean"], values["event_scale"], values["event_coef"], len(EVENT_FEATURE_NAMES), "event"
    )
    if not isinstance(document["training"], dict):
        raise ContractError("model training metadata must be an object")
    if document["training"].get("amendment2_sha256") != AMENDMENT_2_SHA256:
        raise ContractError("model does not pin the frozen Amendment 2 SHA-256")
    if not isinstance(document["training"].get("reachability_report_sha256"), str):
        raise ContractError("model does not pin an authenticated reachability report")
    return ModelBundle(
        pair,
        event,
        document["training"],
        json_sha256=sha256_file(source),
        arrays_sha256=arrays_hash,
    )


def iter_score_sequence(
    sequence: SequenceLabels, model: ModelBundle
) -> Iterator[dict[str, Any]]:
    """Yield scored pairs one parent at a time without retaining prior rows."""

    context = lineage_context(sequence)
    for parent in eligible_parent_indices(sequence):
        candidates = enumerate_parent_candidates(sequence, int(parent), context=context)
        conditional = pair_conditional_probabilities(model.pair, candidates.features)
        event_row = event_features_for_parent(sequence, context, candidates, model.pair)
        event_probability = float(model.event.probabilities(event_row)[0])
        ranking = sorted(
            range(len(candidates.pairs)),
            key=lambda index: (-float(conditional[index]), candidates.pairs[index]),
        )
        for rank, index in enumerate(ranking, start=1):
            first, second = candidates.pairs[index]
            pair_probability = float(conditional[index])
            score = event_probability * pair_probability
            if not (0.0 <= score <= 1.0):
                raise ContractError("candidate score is not a probability")
            yield {
                "sequence_id": sequence.sequence_id,
                "parent_index": int(parent),
                "parent_track_id": int(round(sequence.nodes[int(parent), 4])),
                "daughter_a_index": int(first),
                "daughter_a_track_id": int(round(sequence.nodes[int(first), 4])),
                "daughter_b_index": int(second),
                "daughter_b_track_id": int(round(sequence.nodes[int(second), 4])),
                "pair_rank": rank,
                "event_probability": event_probability,
                "pair_probability": pair_probability,
                "score": score,
                "pair_features": {
                    name: float(value)
                    for name, value in zip(
                        PAIR_FEATURE_NAMES, candidates.features[index], strict=True
                    )
                },
                "event_features": {
                    name: float(value)
                    for name, value in zip(EVENT_FEATURE_NAMES, event_row, strict=True)
                },
            }


def score_sequence(sequence: SequenceLabels, model: ModelBundle) -> list[dict[str, Any]]:
    """Compatibility list API; prefer :func:`iter_score_sequence` for full graphs."""

    return list(iter_score_sequence(sequence, model))


def iter_score_graph_arrays(
    *,
    sequence_id: str,
    nodes: np.ndarray,
    edges: np.ndarray,
    model: ModelBundle,
    divisions: np.ndarray | None = None,
    voxel_um_pooled: np.ndarray = VOXEL_UM_POOLED,
) -> Iterator[dict[str, Any]]:
    """Return a streaming proposal iterator for real validation or compilation.

    Array validation happens before this function returns.  Iteration then
    retains only one parent's at-most-28 candidate rows and feature matrices.
    ``divisions`` is structural metadata only and is never a model feature.
    """

    sequence = sequence_from_arrays(
        sequence_id=sequence_id,
        nodes=nodes,
        edges=edges,
        divisions=divisions,
        voxel_um_pooled=voxel_um_pooled,
    )
    return iter_score_sequence(sequence, model)


def score_graph_arrays(
    *,
    sequence_id: str,
    nodes: np.ndarray,
    edges: np.ndarray,
    model: ModelBundle,
    divisions: np.ndarray | None = None,
    voxel_um_pooled: np.ndarray = VOXEL_UM_POOLED,
) -> list[dict[str, Any]]:
    """Score proposals from graph arrays for real validation or compilation.

    ``divisions`` is optional for division-free parent graphs.  It is used only
    for structural validation and never as a model feature.
    """

    return list(
        iter_score_graph_arrays(
            sequence_id=sequence_id,
            nodes=nodes,
            edges=edges,
            model=model,
            divisions=divisions,
            voxel_um_pooled=voxel_um_pooled,
        )
    )


def collect_sequence_paths(inputs: Sequence[Path | str]) -> list[Path]:
    result: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            result.update(candidate for candidate in path.rglob("*.npz") if candidate.is_file())
        elif path.is_file():
            result.add(path)
        else:
            raise ContractError(f"sequence input does not exist: {path}")
    return sorted(result, key=lambda path: (path.stem, str(path)))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_json_bytes(value))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_command(args: argparse.Namespace) -> int:
    deadline = Deadline.start(args.runtime_cap_seconds)
    rows = []
    for path in collect_sequence_paths(args.sequences):
        deadline.check(f"inspecting {path.name}")
        sequence = load_sequence(path)
        rows.append(
            {
                "sequence_id": sequence.sequence_id,
                "path": str(path),
                "split_residue": split_residue(sequence.sequence_id),
                "split": split_name(sequence.sequence_id),
                "nodes": int(sequence.nodes.shape[0]),
                "edges": int(sequence.edges.shape[0]),
                "divisions": int(sequence.divisions.size),
                "label_sha256": sequence.label_sha256,
            }
        )
    report = {"node": NODE, "sequences": rows}
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def coverage_audit_command(args: argparse.Namespace) -> int:
    paths = collect_sequence_paths(args.sequences)
    report = audit_reachability(paths, runtime_cap_seconds=args.runtime_cap_seconds)
    validate_reachability_report(report, sequence_paths=paths)
    _write_json(args.output, report)
    summary = {
        "schema": report["schema"],
        "node": report["node"],
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "sequence_count": report["sequence_count"],
        "overall": report["overall"],
        "by_split": report["by_split"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def train_command(args: argparse.Namespace) -> int:
    paths = collect_sequence_paths(args.sequences)
    prereg_hash = sha256_file(args.preregistration) if args.preregistration is not None else None
    amendment_hash = sha256_file(args.amendment2)
    if amendment_hash != AMENDMENT_2_SHA256:
        raise ContractError("Amendment 2 file SHA-256 does not match the frozen contract")
    bundle = train_model(
        paths,
        pair_l2=args.pair_l2,
        event_l2=args.event_l2,
        maximum_iterations=args.maximum_iterations,
        runtime_cap_seconds=args.runtime_cap_seconds,
        train_split_only=not args.all_splits,
        preregistration_sha256=prereg_hash,
        coverage_report_path=args.coverage_report,
        coverage_report_sha256=args.coverage_report_sha256,
        amendment2_sha256=amendment_hash,
    )
    json_path, arrays_path = save_model(bundle, args.model_json)
    report = {
        "node": NODE,
        "model_json": str(json_path),
        "model_json_sha256": sha256_file(json_path),
        "model_arrays": str(arrays_path),
        "model_arrays_sha256": sha256_file(arrays_path),
        "training": bundle.metadata,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def score_sequences_command(args: argparse.Namespace) -> int:
    deadline = Deadline.start(args.runtime_cap_seconds)
    model = load_model(args.model_json)
    rows: list[dict[str, Any]] = []
    sequences = []
    for path in collect_sequence_paths(args.sequences):
        deadline.check(f"scoring {path.name}")
        sequence = load_sequence(path)
        sequence_rows = score_sequence(sequence, model)
        rows.extend(sequence_rows)
        sequences.append(
            {
                "sequence_id": sequence.sequence_id,
                "label_sha256": sequence.label_sha256,
                "candidate_pairs": len(sequence_rows),
            }
        )
    report = {
        "schema": SCORE_SCHEMA,
        "node": NODE,
        "model_json_sha256": model.json_sha256,
        "model_arrays_sha256": model.arrays_sha256,
        "sequences": sequences,
        "candidates": rows,
    }
    _write_json(args.output, report)
    print(json.dumps({**report, "candidates": f"{len(rows)} rows"}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate and summarize label NPZs")
    inspect_parser.add_argument("sequences", nargs="+", type=Path)
    inspect_parser.add_argument("--output", type=Path)
    inspect_parser.add_argument("--runtime-cap-seconds", type=float, default=DEFAULT_RUNTIME_CAP_SECONDS)
    inspect_parser.set_defaults(function=inspect_command)

    coverage_parser = subparsers.add_parser(
        "coverage-audit", help="audit exact-daughter reachability in the frozen nearest-8 universe"
    )
    coverage_parser.add_argument("sequences", nargs="+", type=Path)
    coverage_parser.add_argument("--output", type=Path, required=True)
    coverage_parser.add_argument(
        "--runtime-cap-seconds", type=float, default=DEFAULT_RUNTIME_CAP_SECONDS
    )
    coverage_parser.set_defaults(function=coverage_audit_command)

    train_parser = subparsers.add_parser("train", help="fit and freeze the two-stage model")
    train_parser.add_argument("sequences", nargs="+", type=Path)
    train_parser.add_argument("--model-json", type=Path, required=True)
    train_parser.add_argument("--preregistration", type=Path)
    train_parser.add_argument(
        "--amendment2",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports/g04/lanes/synth_div_sim2real_v1/PREREGISTRATION_AMENDMENT_2.json",
    )
    train_parser.add_argument("--coverage-report", type=Path)
    train_parser.add_argument("--coverage-report-sha256")
    train_parser.add_argument("--pair-l2", type=float, default=1.0)
    train_parser.add_argument("--event-l2", type=float, default=1.0)
    train_parser.add_argument("--maximum-iterations", type=int, default=100)
    train_parser.add_argument("--runtime-cap-seconds", type=float, default=DEFAULT_RUNTIME_CAP_SECONDS)
    train_parser.add_argument(
        "--all-splits",
        action="store_true",
        help="test-fixture convenience only; production training must omit this flag",
    )
    train_parser.set_defaults(function=train_command)

    score_parser = subparsers.add_parser(
        "score-sequences", help="emit calibrated event-times-conditional-pair probabilities"
    )
    score_parser.add_argument("sequences", nargs="+", type=Path)
    score_parser.add_argument("--model-json", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--runtime-cap-seconds", type=float, default=DEFAULT_RUNTIME_CAP_SECONDS)
    score_parser.set_defaults(function=score_sequences_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.function(args))
    except (ContractError, TimeoutError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
