"""Bounded sparse association over released V45 checkpoint features.

This module deliberately does not expose an all-pairs scoring path.  Candidate
pairs are found in physical coordinates, retained as COO rows, and scored in
bounded candidate chunks.  The released transformer's projection, input/output
normalization, and pair MLP are reused read-only; its cross-attention blocks are
not registered or called by :class:`SparseEdgeScorer`.

The opt-in :class:`A0CheckpointScorer` additionally reuses all four frozen
cross-attention blocks, but only inside bounded connected candidate components.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bh_graph import PHYSICAL_SCALE_ZYX
from .bh_model import (
    DEFAULT_WEIGHT_SHA256,
    ArtifactBundle,
    FrameDet,
    PairContext,
    positional_features,
)
from .v45_head import V45AuxReadout


EDGE_SCHEMA = "celltrack-alpha.bh-sparse-edge.v1"
EDGE_ARTIFACT_SCHEMA = "celltrack-alpha.bh-sparse-edge-artifact.v2"
A0_EDGE_SCHEMA = "celltrack-alpha.bh-a0-component-edge.v1"
UNKNOWN_LABEL = -1
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FEATURE_NAMES = (
    "checkpoint_logit",
    "displacement_z_over_radius",
    "displacement_y_over_radius",
    "displacement_x_over_radius",
    "distance_over_radius",
    "source_confidence",
    "target_confidence",
    "source_log_local_density",
    "target_log_local_density",
    "log_local_density_delta",
    "previous_velocity_present",
    "velocity_residual_z_over_radius",
    "velocity_residual_y_over_radius",
    "velocity_residual_x_over_radius",
    "velocity_residual_distance_over_radius",
    "aux_motion_nll",
)


class SparseEdgeError(ValueError):
    """A sparse edge input, checkpoint view, or artifact is invalid."""


class EdgeArtifactError(SparseEdgeError):
    """A deterministic sparse-edge artifact failed closed validation."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise EdgeArtifactError("artifact metadata is not canonical JSON") from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SparseEdgeError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _finite_array(value: Any, shape_tail: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape_tail) + 1 or tuple(array.shape[1:]) != shape_tail:
        raise SparseEdgeError(f"{label} must have shape (N, {', '.join(map(str, shape_tail))})")
    if not np.isfinite(array).all():
        raise SparseEdgeError(f"{label} contains nonfinite values")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _readonly_1d(value: Any, dtype: str, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise SparseEdgeError(f"{label} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise SparseEdgeError(f"{label} must contain real numeric values")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise SparseEdgeError(f"{label} contains nonfinite values")
    target = np.dtype(dtype)
    if np.issubdtype(target, np.integer):
        if np.issubdtype(array.dtype, np.floating) and not np.equal(array, np.trunc(array)).all():
            raise SparseEdgeError(f"{label} must contain exact integers")
        if array.size:
            limits = np.iinfo(target)
            if np.min(array) < limits.min or np.max(array) > limits.max:
                raise SparseEdgeError(f"{label} exceeds its integer range")
    result = np.ascontiguousarray(array, dtype=target)
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise SparseEdgeError(f"{label} contains nonfinite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SparseEdgeConfig:
    """Small, serializable sparse-candidate and residual architecture config."""

    schema: str = EDGE_SCHEMA
    radius_um: float = 10.0
    neighbor_cap: int = 8
    pair_chunk_size: int = 256
    local_density_radius_um: float = 10.0
    residual_hidden: int = 32
    dummy_logit: float = 0.0
    scale_zyx: tuple[float, float, float] = PHYSICAL_SCALE_ZYX
    division_enabled: bool = False

    def __post_init__(self) -> None:
        if self.schema != EDGE_SCHEMA:
            raise SparseEdgeError(f"config schema must be {EDGE_SCHEMA!r}")
        if not math.isfinite(self.radius_um) or not 4.0 <= self.radius_um <= 10.0:
            raise SparseEdgeError("radius_um must be finite and in [4, 10]")
        if type(self.neighbor_cap) is not int or self.neighbor_cap not in (2, 4, 8):
            raise SparseEdgeError("neighbor_cap must be one of 2, 4, or 8")
        if type(self.pair_chunk_size) is not int or not 1 <= self.pair_chunk_size <= 65_536:
            raise SparseEdgeError("pair_chunk_size must be in [1, 65536]")
        if (
            not math.isfinite(self.local_density_radius_um)
            or not 4.0 <= self.local_density_radius_um <= 10.0
        ):
            raise SparseEdgeError("local_density_radius_um must be finite and in [4, 10]")
        if type(self.residual_hidden) is not int or not 1 <= self.residual_hidden <= 256:
            raise SparseEdgeError("residual_hidden must be in [1, 256]")
        if not math.isfinite(self.dummy_logit):
            raise SparseEdgeError("dummy_logit must be finite")
        if (
            not isinstance(self.scale_zyx, tuple)
            or len(self.scale_zyx) != 3
            or any(not math.isfinite(value) or value <= 0.0 for value in self.scale_zyx)
        ):
            raise SparseEdgeError("scale_zyx must contain three positive finite values")
        if type(self.division_enabled) is not bool:
            raise SparseEdgeError("division_enabled must be bool")
        if self.division_enabled:
            raise SparseEdgeError("division scoring remains disabled until an edge-only gate")
        trainable = (len(_FEATURE_NAMES) + 1) * self.residual_hidden + self.residual_hidden + 2
        if trainable > 50_000:
            raise SparseEdgeError("residual calibrator exceeds 50000 trainable parameters")

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            result[item.name] = list(value) if isinstance(value, tuple) else value
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "SparseEdgeConfig":
        if not isinstance(value, dict):
            raise EdgeArtifactError("artifact config must be an object")
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            raise EdgeArtifactError("artifact config has missing or unexpected fields")
        data = dict(value)
        scale = data.get("scale_zyx")
        if not isinstance(scale, list) or len(scale) != 3:
            raise EdgeArtifactError("artifact scale_zyx must be a three-value list")
        data["scale_zyx"] = tuple(scale)
        try:
            return cls(**data)
        except (TypeError, SparseEdgeError) as exc:
            raise EdgeArtifactError(f"invalid artifact config: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CandidateStats:
    source_count: int
    target_count: int
    candidate_count: int
    radius_um: float
    neighbor_cap: int
    peak_radius_neighbors: int
    peak_retained_neighbors: int

    def __post_init__(self) -> None:
        names = (
            "source_count",
            "target_count",
            "candidate_count",
            "neighbor_cap",
            "peak_radius_neighbors",
            "peak_retained_neighbors",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in names):
            raise SparseEdgeError("candidate counts must be nonnegative integers")
        if self.neighbor_cap not in (2, 4, 8):
            raise SparseEdgeError("candidate neighbor cap is invalid")
        if not math.isfinite(self.radius_um) or not 4.0 <= self.radius_um <= 10.0:
            raise SparseEdgeError("candidate radius is invalid")
        if self.candidate_count > self.source_count * self.neighbor_cap:
            raise SparseEdgeError("candidate cardinality cap was violated")
        if self.peak_retained_neighbors > self.neighbor_cap:
            raise SparseEdgeError("retained-neighbor peak exceeds the cap")
        if self.peak_retained_neighbors > self.peak_radius_neighbors:
            raise SparseEdgeError("retained-neighbor peak exceeds queried neighbors")


@dataclass(frozen=True, slots=True)
class SparseCandidateCOO:
    """Physical-coordinate candidate rows; every array has candidate length E."""

    source_index: np.ndarray
    target_index: np.ndarray
    source_ids: np.ndarray
    target_ids: np.ndarray
    displacement_um: np.ndarray
    distance_um: np.ndarray
    source_density: np.ndarray
    target_density: np.ndarray
    stats: CandidateStats

    def __post_init__(self) -> None:
        source_index = _readonly_1d(self.source_index, "<i8", "candidate source index")
        target_index = _readonly_1d(self.target_index, "<i8", "candidate target index")
        source_ids = _readonly_1d(self.source_ids, "<i8", "candidate source ID")
        target_ids = _readonly_1d(self.target_ids, "<i8", "candidate target ID")
        distance = _readonly_1d(self.distance_um, "<f8", "candidate distance")
        source_density = _readonly_1d(self.source_density, "<f8", "source density")
        target_density = _readonly_1d(self.target_density, "<f8", "target density")
        displacement = _finite_array(self.displacement_um, (3,), "candidate displacement")
        count = self.stats.candidate_count
        arrays = (
            source_index,
            target_index,
            source_ids,
            target_ids,
            distance,
            source_density,
            target_density,
            displacement,
        )
        if any(len(array) != count for array in arrays):
            raise SparseEdgeError("candidate COO lengths do not match candidate_count")
        if count:
            if np.any(source_index < 0) or np.any(source_index >= self.stats.source_count):
                raise SparseEdgeError("candidate source index is out of bounds")
            if np.any(target_index < 0) or np.any(target_index >= self.stats.target_count):
                raise SparseEdgeError("candidate target index is out of bounds")
            if np.any(source_ids < 0) or np.any(target_ids < 0):
                raise SparseEdgeError("candidate node IDs must be nonnegative")
            if np.any(distance < 0.0) or np.any(distance > self.stats.radius_um + 1e-9):
                raise SparseEdgeError("candidate distance lies outside the physical gate")
            if np.any(source_density < 0.0) or np.any(target_density < 0.0):
                raise SparseEdgeError("local densities must be nonnegative")
            if not np.allclose(np.linalg.norm(displacement, axis=1), distance, rtol=1e-12, atol=1e-12):
                raise SparseEdgeError("candidate displacement and distance disagree")
            pairs = set(zip(source_ids.tolist(), target_ids.tolist(), strict=True))
            if len(pairs) != count:
                raise SparseEdgeError("duplicate sparse candidate pair")
            source_counts = np.bincount(source_index, minlength=self.stats.source_count)
            if source_counts.size and int(source_counts.max(initial=0)) > self.stats.neighbor_cap:
                raise SparseEdgeError("per-source candidate cap was violated")
        object.__setattr__(self, "source_index", source_index)
        object.__setattr__(self, "target_index", target_index)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "displacement_um", displacement)
        object.__setattr__(self, "distance_um", distance)
        object.__setattr__(self, "source_density", source_density)
        object.__setattr__(self, "target_density", target_density)

    def __len__(self) -> int:
        return self.stats.candidate_count


def _frame_physical_coords(frame: FrameDet, scale_zyx: Sequence[float]) -> np.ndarray:
    if not isinstance(frame, FrameDet):
        raise SparseEdgeError("candidate gating requires FrameDet inputs")
    scale = np.asarray(scale_zyx, dtype=np.float64)
    coords = np.asarray(frame.coords_zyx, dtype=np.float64) * scale
    if coords.shape != (len(frame.node_ids), 3) or not np.isfinite(coords).all():
        raise SparseEdgeError("frame physical coordinates are invalid")
    return np.ascontiguousarray(coords)


def _local_neighbor_counts(points: np.ndarray, radius_um: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    tree = cKDTree(points)
    counts = np.empty(len(points), dtype=np.float64)
    for index in range(len(points)):
        # One point is queried at a time: no node-by-node index or distance object.
        count = int(tree.query_ball_point(points[index], radius_um, return_length=True))
        counts[index] = max(0, count - 1)
    return counts


def gate_sparse_candidates(
    source: FrameDet,
    target: FrameDet,
    *,
    config: SparseEdgeConfig = SparseEdgeConfig(),
) -> SparseCandidateCOO:
    """Gate adjacent detections with a physical cKDTree and deterministic cap.

    Equal-distance targets are retained by ascending target node ID (then input
    index), so tree traversal order cannot affect capped output.
    """

    if not isinstance(config, SparseEdgeConfig):
        raise SparseEdgeError("config must be SparseEdgeConfig")
    if not isinstance(source, FrameDet) or not isinstance(target, FrameDet):
        raise SparseEdgeError("candidate gating requires FrameDet inputs")
    if target.frame_index != source.frame_index + 1:
        raise SparseEdgeError("candidate gating requires adjacent forward frames")
    source_points = _frame_physical_coords(source, config.scale_zyx)
    target_points = _frame_physical_coords(target, config.scale_zyx)
    source_density_all = _local_neighbor_counts(source_points, config.local_density_radius_um)
    target_density_all = _local_neighbor_counts(target_points, config.local_density_radius_um)
    target_tree = cKDTree(target_points) if len(target_points) else None

    source_index: list[int] = []
    target_index: list[int] = []
    displacement_rows: list[np.ndarray] = []
    distance_rows: list[float] = []
    peak_radius_neighbors = 0
    peak_retained = 0
    if target_tree is not None:
        for source_row in range(len(source_points)):
            raw_neighbors = target_tree.query_ball_point(source_points[source_row], config.radius_um)
            neighbors = [int(value) for value in raw_neighbors]
            peak_radius_neighbors = max(peak_radius_neighbors, len(neighbors))
            if not neighbors:
                continue
            neighbor_array = np.asarray(neighbors, dtype=np.int64)
            delta = target_points[neighbor_array] - source_points[source_row]
            distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            ordered = sorted(
                range(len(neighbors)),
                key=lambda local: (
                    float(distances[local]),
                    int(target.node_ids[neighbors[local]]),
                    neighbors[local],
                ),
            )[: config.neighbor_cap]
            peak_retained = max(peak_retained, len(ordered))
            for local in ordered:
                source_index.append(source_row)
                target_index.append(neighbors[local])
                displacement_rows.append(delta[local].copy())
                distance_rows.append(float(distances[local]))

    source_index_array = np.asarray(source_index, dtype=np.int64)
    target_index_array = np.asarray(target_index, dtype=np.int64)
    displacement_array = (
        np.asarray(displacement_rows, dtype=np.float64).reshape(-1, 3)
        if displacement_rows
        else np.empty((0, 3), dtype=np.float64)
    )
    stats = CandidateStats(
        source_count=len(source.node_ids),
        target_count=len(target.node_ids),
        candidate_count=len(source_index),
        radius_um=float(config.radius_um),
        neighbor_cap=config.neighbor_cap,
        peak_radius_neighbors=peak_radius_neighbors,
        peak_retained_neighbors=peak_retained,
    )
    return SparseCandidateCOO(
        source_index=source_index_array,
        target_index=target_index_array,
        source_ids=source.node_ids[source_index_array],
        target_ids=target.node_ids[target_index_array],
        displacement_um=displacement_array,
        distance_um=np.asarray(distance_rows, dtype=np.float64),
        source_density=source_density_all[source_index_array],
        target_density=target_density_all[target_index_array],
        stats=stats,
    )


@dataclass(frozen=True, slots=True)
class ProbabilityStats:
    source_count: int
    target_count: int
    candidate_count: int
    pair_chunk_size: int
    peak_pair_chunk: int
    neighbor_cap: int
    explicit_dummy: bool = True

    def __post_init__(self) -> None:
        integer_names = (
            "source_count",
            "target_count",
            "candidate_count",
            "pair_chunk_size",
            "peak_pair_chunk",
            "neighbor_cap",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in integer_names):
            raise SparseEdgeError("probability counts must be nonnegative integers")
        if self.pair_chunk_size <= 0 or self.peak_pair_chunk > self.pair_chunk_size:
            raise SparseEdgeError("probability chunk accounting is invalid")
        if self.candidate_count > self.source_count * self.neighbor_cap:
            raise SparseEdgeError("probability cardinality cap was violated")
        if self.explicit_dummy is not True:
            raise SparseEdgeError("source normalization requires an explicit dummy")


@dataclass(frozen=True, slots=True)
class SparsePairProbabilities(Mapping[tuple[int, int], float]):
    """Pair-scoped sparse probabilities directly accepted by ``bh_graph``."""

    source_frame: int
    target_frame: int
    source_ids: np.ndarray
    target_ids: np.ndarray
    probabilities: np.ndarray
    no_link_source_ids: np.ndarray
    no_link_probabilities: np.ndarray
    stats: ProbabilityStats
    _mapping: dict[tuple[int, int], float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.source_frame) is not int
            or self.source_frame < 0
            or type(self.target_frame) is not int
            or self.target_frame != self.source_frame + 1
        ):
            raise SparseEdgeError("probabilities require adjacent forward frames")
        source_ids = _readonly_1d(self.source_ids, "<i8", "probability source IDs")
        target_ids = _readonly_1d(self.target_ids, "<i8", "probability target IDs")
        probabilities = _readonly_1d(self.probabilities, "<f8", "candidate probabilities")
        no_link_source_ids = _readonly_1d(
            self.no_link_source_ids, "<i8", "no-link source IDs"
        )
        no_link_probabilities = _readonly_1d(
            self.no_link_probabilities, "<f8", "no-link probabilities"
        )
        if not (len(source_ids) == len(target_ids) == len(probabilities) == self.stats.candidate_count):
            raise SparseEdgeError("probability COO lengths do not match candidate_count")
        if len(no_link_source_ids) != self.stats.source_count or len(no_link_probabilities) != self.stats.source_count:
            raise SparseEdgeError("no-link rows do not cover every source")
        if len(set(no_link_source_ids.tolist())) != len(no_link_source_ids):
            raise SparseEdgeError("duplicate no-link source ID")
        if (
            np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
            or np.any(no_link_probabilities < 0.0)
            or np.any(no_link_probabilities > 1.0)
        ):
            raise SparseEdgeError("probabilities must lie in [0, 1]")
        mapping = {
            (int(source_id), int(target_id)): float(probability)
            for source_id, target_id, probability in zip(
                source_ids, target_ids, probabilities, strict=True
            )
        }
        if len(mapping) != len(probabilities):
            raise SparseEdgeError("duplicate probability pair")
        no_link = {
            int(source_id): float(probability)
            for source_id, probability in zip(
                no_link_source_ids, no_link_probabilities, strict=True
            )
        }
        sums = dict(no_link)
        for (source_id, _target_id), probability in mapping.items():
            if source_id not in sums:
                raise SparseEdgeError("candidate probability has no source dummy row")
            sums[source_id] += probability
        if any(not math.isclose(value, 1.0, rel_tol=1e-7, abs_tol=1e-9) for value in sums.values()):
            raise SparseEdgeError("candidate plus no-link probabilities are not source-normalized")
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "no_link_source_ids", no_link_source_ids)
        object.__setattr__(self, "no_link_probabilities", no_link_probabilities)
        object.__setattr__(self, "_mapping", mapping)

    def __getitem__(self, key: tuple[int, int]) -> float:
        return self._mapping[key]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def as_graph_mapping(self) -> dict[tuple[int, int], float]:
        """Return a sparse copy suitable for ``learned_edge_probabilities``."""

        return dict(self._mapping)


@dataclass(frozen=True, slots=True)
class A0ScorerConfig:
    """Hard production budgets for opt-in component-local checkpoint scoring."""

    schema: str = A0_EDGE_SCHEMA
    max_component_nodes: int = 64
    neighbor_cap: int = 8
    attention_cell_budget: int = 65_536
    pair_chunk_size: int = 256

    def __post_init__(self) -> None:
        if self.schema != A0_EDGE_SCHEMA:
            raise SparseEdgeError(f"A0 config schema must be {A0_EDGE_SCHEMA!r}")
        if (
            type(self.max_component_nodes) is not int
            or not 2 <= self.max_component_nodes <= 64
        ):
            raise SparseEdgeError("A0 max_component_nodes must be in [2, 64]")
        if type(self.neighbor_cap) is not int or self.neighbor_cap not in (2, 4, 8):
            raise SparseEdgeError("A0 neighbor_cap must be one of 2, 4, or 8")
        if (
            type(self.attention_cell_budget) is not int
            or not 1 <= self.attention_cell_budget <= 65_536
        ):
            raise SparseEdgeError("A0 attention_cell_budget must be in [1, 65536]")
        if (
            type(self.pair_chunk_size) is not int
            or not 1 <= self.pair_chunk_size <= 256
        ):
            raise SparseEdgeError("A0 pair_chunk_size must be in [1, 256]")

    def as_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class A0AssociationStats:
    """Auditable component, fallback, batching, and allocation accounting."""

    source_count: int
    target_count: int
    candidate_count: int
    component_count: int
    attention_component_count: int
    fallback_component_count: int
    node_budget_fallback_component_count: int
    attention_budget_fallback_component_count: int
    unsupported_fallback_component_count: int
    attention_candidate_count: int
    fallback_candidate_count: int
    attention_component_fraction: float
    fallback_component_fraction: float
    attention_candidate_fraction: float
    fallback_candidate_fraction: float
    max_component_source_nodes: int
    max_component_target_nodes: int
    max_component_nodes: int
    max_component_candidates: int
    max_component_attention_cells: int
    batch_count: int
    max_batch_components: int
    max_padded_source_nodes: int
    max_padded_target_nodes: int
    peak_padding_nodes: int
    peak_attention_cells: int
    neighbor_cap: int
    max_component_nodes_budget: int
    attention_cell_budget: int
    pair_chunk_size: int
    peak_pair_chunk: int

    def __post_init__(self) -> None:
        integer_names = (
            "source_count",
            "target_count",
            "candidate_count",
            "component_count",
            "attention_component_count",
            "fallback_component_count",
            "node_budget_fallback_component_count",
            "attention_budget_fallback_component_count",
            "unsupported_fallback_component_count",
            "attention_candidate_count",
            "fallback_candidate_count",
            "max_component_source_nodes",
            "max_component_target_nodes",
            "max_component_nodes",
            "max_component_candidates",
            "max_component_attention_cells",
            "batch_count",
            "max_batch_components",
            "max_padded_source_nodes",
            "max_padded_target_nodes",
            "peak_padding_nodes",
            "peak_attention_cells",
            "neighbor_cap",
            "max_component_nodes_budget",
            "attention_cell_budget",
            "pair_chunk_size",
            "peak_pair_chunk",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 0
            for name in integer_names
        ):
            raise SparseEdgeError("A0 audit counts and maxima must be nonnegative integers")
        if self.attention_component_count + self.fallback_component_count != self.component_count:
            raise SparseEdgeError("A0 component accounting does not balance")
        if (
            self.node_budget_fallback_component_count
            + self.attention_budget_fallback_component_count
            + self.unsupported_fallback_component_count
            != self.fallback_component_count
        ):
            raise SparseEdgeError("A0 fallback reason accounting does not balance")
        if self.attention_candidate_count + self.fallback_candidate_count != self.candidate_count:
            raise SparseEdgeError("A0 candidate accounting does not balance")
        fraction_names = (
            "attention_component_fraction",
            "fallback_component_fraction",
            "attention_candidate_fraction",
            "fallback_candidate_fraction",
        )
        if any(
            not math.isfinite(getattr(self, name))
            or not 0.0 <= getattr(self, name) <= 1.0
            for name in fraction_names
        ):
            raise SparseEdgeError("A0 audit fractions must lie in [0, 1]")
        expected_component_denominator = max(self.component_count, 1)
        expected_candidate_denominator = max(self.candidate_count, 1)
        expected_fractions = (
            self.attention_component_count / expected_component_denominator,
            self.fallback_component_count / expected_component_denominator,
            self.attention_candidate_count / expected_candidate_denominator,
            self.fallback_candidate_count / expected_candidate_denominator,
        )
        if any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15)
            for observed, expected in zip(
                (getattr(self, name) for name in fraction_names),
                expected_fractions,
                strict=True,
            )
        ):
            raise SparseEdgeError("A0 audit fractions disagree with their counts")
        if self.neighbor_cap not in (2, 4, 8):
            raise SparseEdgeError("A0 audit neighbor cap is invalid")
        if not 2 <= self.max_component_nodes_budget <= 64:
            raise SparseEdgeError("A0 component-node budget is invalid")
        if not 1 <= self.attention_cell_budget <= 65_536:
            raise SparseEdgeError("A0 attention-cell budget is invalid")
        if self.peak_attention_cells > self.attention_cell_budget:
            raise SparseEdgeError("A0 attention-cell budget was exceeded")
        if not 1 <= self.pair_chunk_size <= 256 or self.peak_pair_chunk > self.pair_chunk_size:
            raise SparseEdgeError("A0 pair chunk accounting is invalid")


@dataclass(frozen=True, slots=True)
class A0PairProbabilities(Mapping[tuple[int, int], float]):
    """Original candidate COO rows after target-wise source softmax.

    A0 deliberately has no no-link row or dummy class.  For each target that
    has candidates, its retained source probabilities sum to one.
    """

    source_frame: int
    target_frame: int
    source_ids: np.ndarray
    target_ids: np.ndarray
    probabilities: np.ndarray
    stats: A0AssociationStats
    _mapping: dict[tuple[int, int], float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.source_frame) is not int
            or self.source_frame < 0
            or type(self.target_frame) is not int
            or self.target_frame != self.source_frame + 1
        ):
            raise SparseEdgeError("A0 probabilities require adjacent forward frames")
        if not isinstance(self.stats, A0AssociationStats):
            raise SparseEdgeError("A0 probabilities require A0AssociationStats")
        source_ids = _readonly_1d(self.source_ids, "<i8", "A0 probability source IDs")
        target_ids = _readonly_1d(self.target_ids, "<i8", "A0 probability target IDs")
        probabilities = _readonly_1d(
            self.probabilities, "<f8", "A0 candidate probabilities"
        )
        if not (
            len(source_ids)
            == len(target_ids)
            == len(probabilities)
            == self.stats.candidate_count
        ):
            raise SparseEdgeError("A0 probability COO lengths do not match candidate_count")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise SparseEdgeError("A0 probabilities must lie in [0, 1]")
        mapping = {
            (int(source_id), int(target_id)): float(probability)
            for source_id, target_id, probability in zip(
                source_ids, target_ids, probabilities, strict=True
            )
        }
        if len(mapping) != len(probabilities):
            raise SparseEdgeError("duplicate A0 probability pair")
        target_sums: dict[int, float] = {}
        for (_source_id, target_id), probability in mapping.items():
            target_sums[target_id] = target_sums.get(target_id, 0.0) + probability
        if any(
            not math.isclose(value, 1.0, rel_tol=1e-7, abs_tol=1e-9)
            for value in target_sums.values()
        ):
            raise SparseEdgeError("A0 probabilities are not target-wise source-normalized")
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "_mapping", mapping)

    def __getitem__(self, key: tuple[int, int]) -> float:
        return self._mapping[key]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def as_graph_mapping(self) -> dict[tuple[int, int], float]:
        return dict(self._mapping)


class _IdentityResidualCalibrator(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        # Fixed local initialization makes identity artifacts reproducible while
        # preserving a nonzero hidden basis for the first training update.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0xB4E45)
            self.input = nn.Linear(len(_FEATURE_NAMES), hidden)
            self.output = nn.Linear(hidden, 1)
        self.dummy_residual = nn.Parameter(torch.zeros((), dtype=torch.float32))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(F.gelu(self.input(features))).squeeze(-1)


def _resolve_transformer(value: Any) -> Any:
    if hasattr(value, "transformer") and not hasattr(value, "proj"):
        value = value.transformer
    required = ("proj", "norm_in", "norm_out", "pair_mlp")
    if any(not isinstance(getattr(value, name, None), nn.Module) for name in required):
        raise SparseEdgeError("checkpoint transformer lacks proj/norm_in/norm_out/pair_mlp")
    return value


def _tensor_state_hash(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    found = False
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        found = True
        if not isinstance(name, str) or not name:
            raise SparseEdgeError("tensor state contains an invalid name")
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise SparseEdgeError("tensor state must contain float32 tensors")
        detached = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(detached).all().item()):
            raise SparseEdgeError(f"tensor {name!r} contains nonfinite values")
        descriptor = {
            "dtype": "float32",
            "name": name,
            "shape": list(detached.shape),
        }
        digest.update(_canonical_json(descriptor))
        digest.update(b"\x00")
        digest.update(detached.numpy().tobytes(order="C"))
        digest.update(b"\x00")
    if not found:
        raise SparseEdgeError("tensor state is empty")
    return digest.hexdigest()


class SparseEdgeScorer(nn.Module):
    """Frozen checkpoint-local candidate scorer plus tiny residual calibrator."""

    def __init__(
        self,
        checkpoint_transformer: Any,
        *,
        config: SparseEdgeConfig = SparseEdgeConfig(),
        pretrained_weight_sha256: str = DEFAULT_WEIGHT_SHA256,
        auxiliary: V45AuxReadout | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, SparseEdgeConfig):
            raise SparseEdgeError("config must be SparseEdgeConfig")
        transformer = _resolve_transformer(checkpoint_transformer)
        self.config = config
        self.pretrained_weight_sha256 = _sha256(
            pretrained_weight_sha256, "pretrained_weight_sha256"
        )
        if auxiliary is not None:
            if not isinstance(auxiliary, V45AuxReadout):
                raise SparseEdgeError("auxiliary must be a validated V45AuxReadout or None")
            if auxiliary.identity.weight_sha256 != self.pretrained_weight_sha256:
                raise SparseEdgeError("auxiliary readout and sparse scorer checkpoint hashes disagree")
            if auxiliary.identity.division_enabled:
                raise SparseEdgeError("auxiliary division scoring remains disabled")
            if any(parameter.requires_grad for parameter in auxiliary.parameters()):
                raise SparseEdgeError("auxiliary readout must be fully frozen")
        # Register only the four released components used by sparse inference.
        self.proj = transformer.proj
        self.norm_in = transformer.norm_in
        self.norm_out = transformer.norm_out
        self.pair_mlp = transformer.pair_mlp
        self.calibrator = _IdentityResidualCalibrator(config.residual_hidden)
        self.auxiliary = auxiliary
        for module in self._frozen_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self._validate_checkpoint_view()
        self.calibrator.to(device=next(self.proj.parameters()).device)
        if self.auxiliary is not None:
            self.auxiliary.to(device=self._device())
            self.auxiliary.eval()
        if self.trainable_parameter_count > 50_000:
            raise SparseEdgeError("sparse edge scorer exceeds 50000 trainable parameters")
        self.eval()

    @classmethod
    def from_loaded_model(
        cls,
        model: Any,
        bundle: ArtifactBundle,
        *,
        config: SparseEdgeConfig = SparseEdgeConfig(),
    ) -> "SparseEdgeScorer":
        if not isinstance(bundle, ArtifactBundle):
            raise SparseEdgeError("from_loaded_model requires a validated ArtifactBundle")
        return cls(
            model,
            config=config,
            pretrained_weight_sha256=bundle.hashes.weight_sha256,
        )

    def _frozen_modules(self) -> tuple[nn.Module, ...]:
        return self.proj, self.norm_in, self.norm_out, self.pair_mlp

    def _frozen_named_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for module_name, module in (
            ("proj", self.proj),
            ("norm_in", self.norm_in),
            ("norm_out", self.norm_out),
            ("pair_mlp", self.pair_mlp),
        ):
            for name, value in module.state_dict().items():
                yield f"{module_name}.{name}", value

    def _validate_checkpoint_view(self) -> None:
        if getattr(self.proj, "in_features", None) != 64 or getattr(self.proj, "out_features", None) != 128:
            raise SparseEdgeError("checkpoint projection must be Linear(64, 128)")
        for name, module in (("norm_in", self.norm_in), ("norm_out", self.norm_out)):
            if tuple(getattr(module, "normalized_shape", ())) != (128,):
                raise SparseEdgeError(f"checkpoint {name} must normalize 128 features")
        parameters = [parameter for module in self._frozen_modules() for parameter in module.parameters()]
        if not parameters:
            raise SparseEdgeError("checkpoint scorer has no parameters")
        devices = {parameter.device for parameter in parameters}
        dtypes = {parameter.dtype for parameter in parameters}
        if len(devices) != 1 or dtypes != {torch.float32}:
            raise SparseEdgeError("checkpoint scorer must use one device and float32 weights")
        for name, value in self._frozen_named_tensors():
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all().item()):
                raise SparseEdgeError(f"checkpoint tensor {name!r} is not finite float32")
        device = parameters[0].device
        with torch.no_grad():
            probe = self.pair_mlp(torch.zeros((1, 259), dtype=torch.float32, device=device))
        if tuple(probe.shape) != (1, 1) or not bool(torch.isfinite(probe).all().item()):
            raise SparseEdgeError("checkpoint pair_mlp is not compatible with candidate rows")

    def train(self, mode: bool = True) -> "SparseEdgeScorer":
        super().train(mode)
        for module in self._frozen_modules():
            module.eval()
        if self.auxiliary is not None:
            self.auxiliary.eval()
        self.calibrator.train(mode)
        return self

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def frozen_subset_sha256(self) -> str:
        return _tensor_state_hash(self._frozen_named_tensors())

    @property
    def architecture(self) -> dict[str, Any]:
        schema: list[dict[str, Any]] = []
        for name, value in self._frozen_named_tensors():
            schema.append({"dtype": "float32", "name": name, "shape": list(value.shape)})
        auxiliary = None
        if self.auxiliary is not None:
            auxiliary = {
                "checkpoint_sha256": self.auxiliary.checkpoint_sha256,
                "identity": self.auxiliary.identity.as_dict(),
                "mode": "target-point-backward-motion-nll",
                "variance_log_clip": [-10.0, 10.0],
            }
        return {
            "name": "released-v45-candidate-coo-residual",
            "checkpoint_input_dim": 64,
            "checkpoint_hidden_dim": 128,
            "checkpoint_pair_input_dim": 259,
            "frozen_components": ["proj", "norm_in", "norm_out", "pair_mlp"],
            "frozen_tensor_schema": schema,
            "cross_attention_blocks": "bypassed-and-unregistered",
            "pair_mlp_scope": "candidate-coo-only",
            "position_encoding": "checkpoint-compatible-tzyx-sinusoidal-8-per-axis",
            "residual_features": list(_FEATURE_NAMES),
            "residual_hidden": self.config.residual_hidden,
            "residual_exit": "zero-initialized",
            "auxiliary": auxiliary,
            "dummy_logit": "explicit-per-source",
            "division_enabled": False,
        }

    @property
    def architecture_sha256(self) -> str:
        return _json_sha256(self.architecture)

    @property
    def config_sha256(self) -> str:
        return _json_sha256(self.config.as_dict())

    def _device(self) -> torch.device:
        return next(self.proj.parameters()).device

    def _validate_pair(
        self,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        candidates: SparseCandidateCOO,
    ) -> None:
        if not isinstance(context, PairContext):
            raise SparseEdgeError("scoring requires PairContext")
        if not isinstance(source, FrameDet) or not isinstance(target, FrameDet):
            raise SparseEdgeError("scoring requires FrameDet inputs")
        if context.frame_pair != (source.frame_index, target.frame_index):
            raise SparseEdgeError("pair context and detection frames disagree")
        if source.grid_shape_zyx != target.grid_shape_zyx or context.grid_shape_zyx != source.grid_shape_zyx:
            raise SparseEdgeError("pair feature grids must have equal shape")
        if set(context.source_ids.tolist()) != set(source.node_ids.tolist()):
            raise SparseEdgeError("pair context source IDs do not match detections")
        if set(context.target_ids.tolist()) != set(target.node_ids.tolist()):
            raise SparseEdgeError("pair context target IDs do not match detections")
        if (
            candidates.stats.source_count != len(source.node_ids)
            or candidates.stats.target_count != len(target.node_ids)
        ):
            raise SparseEdgeError("candidate counts do not match detection frames")
        if candidates.stats.neighbor_cap != self.config.neighbor_cap or not math.isclose(
            candidates.stats.radius_um, self.config.radius_um, rel_tol=0.0, abs_tol=0.0
        ):
            raise SparseEdgeError("candidates were produced with a different gate config")
        if len(candidates):
            if not np.array_equal(candidates.source_ids, source.node_ids[candidates.source_index]):
                raise SparseEdgeError("candidate source IDs and indices disagree")
            if not np.array_equal(candidates.target_ids, target.node_ids[candidates.target_index]):
                raise SparseEdgeError("candidate target IDs and indices disagree")
            source_points = _frame_physical_coords(source, self.config.scale_zyx)
            target_points = _frame_physical_coords(target, self.config.scale_zyx)
            displacement = target_points[candidates.target_index] - source_points[candidates.source_index]
            if not np.allclose(displacement, candidates.displacement_um, rtol=1e-12, atol=1e-12):
                raise SparseEdgeError("candidate physical displacements do not match detections")
            source_density = _local_neighbor_counts(
                source_points, self.config.local_density_radius_um
            )[candidates.source_index]
            target_density = _local_neighbor_counts(
                target_points, self.config.local_density_radius_um
            )[candidates.target_index]
            if not np.array_equal(source_density, candidates.source_density) or not np.array_equal(
                target_density, candidates.target_density
            ):
                raise SparseEdgeError("candidate local densities do not match detections")

    def _checkpoint_node_features(
        self,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_rows = {int(node_id): index for index, node_id in enumerate(context.source_ids)}
        target_rows = {int(node_id): index for index, node_id in enumerate(context.target_ids)}
        source_feat = context.source_feat[[source_rows[int(node_id)] for node_id in source.node_ids]]
        target_feat = context.target_feat[[target_rows[int(node_id)] for node_id in target.node_ids]]
        window_shape = (2,) + context.grid_shape_zyx
        source_pos = positional_features(
            np.column_stack((np.zeros(len(source.node_ids), dtype=np.int64), source.grid_coords_zyx)),
            window_shape,
        )
        target_pos = positional_features(
            np.column_stack((np.ones(len(target.node_ids), dtype=np.int64), target.grid_coords_zyx)),
            window_shape,
        )
        source_input = np.concatenate((source_feat, source_pos), axis=1)
        target_input = np.concatenate((target_feat, target_pos), axis=1)
        if source_input.shape != (len(source.node_ids), 64) or target_input.shape != (len(target.node_ids), 64):
            raise SparseEdgeError("checkpoint-local node inputs must have 64 features")
        device = self._device()
        source_tensor = torch.from_numpy(np.ascontiguousarray(source_input)).to(device=device, dtype=torch.float32)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target_input)).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            source_hidden = self.norm_out(self.norm_in(self.proj(source_tensor)))
            target_hidden = self.norm_out(self.norm_in(self.proj(target_tensor)))
        if (
            tuple(source_hidden.shape) != (len(source.node_ids), 128)
            or tuple(target_hidden.shape) != (len(target.node_ids), 128)
            or not bool(torch.isfinite(source_hidden).all().item())
            or not bool(torch.isfinite(target_hidden).all().item())
        ):
            raise SparseEdgeError("checkpoint-local node projection returned invalid features")
        return source_hidden, target_hidden

    def _checkpoint_candidate_logits(
        self,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        source: FrameDet,
        target: FrameDet,
        candidates: SparseCandidateCOO,
    ) -> torch.Tensor:
        device = self._device()
        source_index = torch.from_numpy(np.array(candidates.source_index, copy=True)).to(device=device)
        target_index = torch.from_numpy(np.array(candidates.target_index, copy=True)).to(device=device)
        source_coords = torch.from_numpy(np.ascontiguousarray(source.coords_zyx, dtype=np.float32)).to(device=device)
        target_coords = torch.from_numpy(np.ascontiguousarray(target.coords_zyx, dtype=np.float32)).to(device=device)
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(candidates), self.config.pair_chunk_size):
                end = min(start + self.config.pair_chunk_size, len(candidates))
                source_rows = source_index[start:end]
                target_rows = target_index[start:end]
                relative = (source_coords[source_rows] - target_coords[target_rows]) / 100.0
                pair_features = torch.cat(
                    (source_hidden[source_rows], target_hidden[target_rows], relative), dim=1
                )
                logits = self.pair_mlp(pair_features).squeeze(-1)
                if tuple(logits.shape) != (end - start,) or not bool(torch.isfinite(logits).all().item()):
                    raise SparseEdgeError("checkpoint pair_mlp returned invalid candidate logits")
                chunks.append(logits.float())
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32, device=device)

    def _velocity_residual(
        self,
        source: FrameDet,
        candidates: SparseCandidateCOO,
        previous_velocity_um: Mapping[int, Sequence[float] | None] | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        present = np.zeros(len(candidates), dtype=np.float64)
        velocity = np.zeros((len(candidates), 3), dtype=np.float64)
        if previous_velocity_um is None:
            return present, np.asarray(candidates.displacement_um, dtype=np.float64)
        if isinstance(previous_velocity_um, np.ndarray):
            array = np.asarray(previous_velocity_um, dtype=np.float64)
            if array.shape != (len(source.node_ids), 3) or not np.isfinite(array).all():
                raise SparseEdgeError("previous_velocity_um array must be finite with shape (N_source, 3)")
            velocity[:] = array[candidates.source_index]
            present[:] = 1.0
        elif isinstance(previous_velocity_um, Mapping):
            known_sources = set(int(node_id) for node_id in source.node_ids)
            normalized: dict[int, np.ndarray | None] = {}
            for raw_id, raw_velocity in previous_velocity_um.items():
                source_id = _exact_nonnegative_int(raw_id, "previous-velocity source ID")
                if source_id not in known_sources:
                    raise SparseEdgeError("previous velocity references an unknown source")
                if source_id in normalized:
                    raise SparseEdgeError("duplicate previous velocity source")
                if raw_velocity is not None:
                    row = np.asarray(raw_velocity, dtype=np.float64)
                    if row.shape != (3,) or not np.isfinite(row).all():
                        raise SparseEdgeError("previous velocities must contain three finite values")
                    normalized[source_id] = row
                else:
                    normalized[source_id] = None
            for candidate_index, source_id in enumerate(candidates.source_ids):
                raw = normalized.get(int(source_id))
                if raw is not None:
                    velocity[candidate_index] = raw
                    present[candidate_index] = 1.0
        else:
            raise SparseEdgeError("previous_velocity_um must be a mapping, array, or None")
        residual = np.asarray(candidates.displacement_um, dtype=np.float64) - velocity
        if not np.isfinite(residual).all():
            raise SparseEdgeError("velocity residual contains nonfinite values")
        return present, residual

    def _auxiliary_motion_nll(
        self,
        context: PairContext,
        target: FrameDet,
        candidates: SparseCandidateCOO,
    ) -> np.ndarray:
        """Evaluate the compact child-owned motion readout only at target nodes."""

        if self.auxiliary is None:
            return np.zeros(len(candidates), dtype=np.float64)
        target_rows = {int(node_id): index for index, node_id in enumerate(context.target_ids)}
        try:
            target_features = context.target_feat[
                [target_rows[int(node_id)] for node_id in target.node_ids]
            ]
        except KeyError as exc:
            raise SparseEdgeError("auxiliary target IDs do not match pair context") from exc
        feature_tensor = torch.from_numpy(np.ascontiguousarray(target_features, dtype=np.float32)).to(
            device=self._device(),
            dtype=torch.float32,
        )
        with torch.no_grad():
            target_outputs = self.auxiliary(feature_tensor)
        if (
            tuple(target_outputs.shape) != (len(target.node_ids), 5)
            or not bool(torch.isfinite(target_outputs).all().item())
        ):
            raise SparseEdgeError("auxiliary target readout is invalid")
        outputs = target_outputs.detach().cpu().numpy().astype(np.float64, copy=False)
        predicted_backward = outputs[candidates.target_index, 1:4]
        log_variance = np.clip(outputs[candidates.target_index, 4], -10.0, 10.0)
        candidate_backward = -np.asarray(candidates.displacement_um, dtype=np.float64)
        residual = candidate_backward - predicted_backward
        nll = 0.5 * (
            np.exp(-log_variance) * np.sum(np.square(residual), axis=1) + 3.0 * log_variance
        )
        if nll.shape != (len(candidates),) or not np.isfinite(nll).all():
            raise SparseEdgeError("auxiliary motion NLL is invalid")
        return nll

    def _residual_features(
        self,
        checkpoint_logits: torch.Tensor,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        candidates: SparseCandidateCOO,
        previous_velocity_um: Mapping[int, Sequence[float] | None] | np.ndarray | None,
    ) -> torch.Tensor:
        present, velocity_residual = self._velocity_residual(
            source, candidates, previous_velocity_um
        )
        radius = self.config.radius_um
        source_conf = np.asarray(source.scores[candidates.source_index], dtype=np.float64)
        target_conf = np.asarray(target.scores[candidates.target_index], dtype=np.float64)
        source_density = np.log1p(np.asarray(candidates.source_density, dtype=np.float64))
        target_density = np.log1p(np.asarray(candidates.target_density, dtype=np.float64))
        residual_distance = np.linalg.norm(velocity_residual, axis=1)
        auxiliary_motion_nll = self._auxiliary_motion_nll(context, target, candidates)
        checkpoint_np = checkpoint_logits.detach().cpu().numpy().astype(np.float64, copy=False)
        features = np.column_stack(
            (
                checkpoint_np,
                np.asarray(candidates.displacement_um, dtype=np.float64) / radius,
                np.asarray(candidates.distance_um, dtype=np.float64) / radius,
                source_conf,
                target_conf,
                source_density,
                target_density,
                target_density - source_density,
                present,
                velocity_residual / radius,
                residual_distance / radius,
                auxiliary_motion_nll,
            )
        )
        if features.shape != (len(candidates), len(_FEATURE_NAMES)) or not np.isfinite(features).all():
            raise SparseEdgeError("residual candidate features are invalid")
        return torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(
            device=self._device()
        )

    def candidate_logits(
        self,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        *,
        candidates: SparseCandidateCOO | None = None,
        previous_velocity_um: Mapping[int, Sequence[float] | None] | np.ndarray | None = None,
    ) -> tuple[SparseCandidateCOO, torch.Tensor]:
        """Return differentiable residual logits only for sparse candidate rows."""

        local_candidates, checkpoint_logits, residual_features = self.candidate_features(
            context,
            source,
            target,
            candidates=candidates,
            previous_velocity_um=previous_velocity_um,
        )
        residual = self.calibrator(residual_features)
        logits = checkpoint_logits.detach() + residual
        if tuple(logits.shape) != (len(local_candidates),) or not bool(torch.isfinite(logits).all().item()):
            raise SparseEdgeError("sparse candidate logits are invalid")
        return local_candidates, logits

    def candidate_features(
        self,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        *,
        candidates: SparseCandidateCOO | None = None,
        previous_velocity_um: Mapping[int, Sequence[float] | None] | np.ndarray | None = None,
    ) -> tuple[SparseCandidateCOO, torch.Tensor, torch.Tensor]:
        """Return frozen candidate logits and compact residual features.

        This is the fit-only interface for the small calibrator.  The returned
        checkpoint logits are detached from frozen checkpoint modules; gradients
        may flow only through :attr:`calibrator` when these features are used.
        """

        local_candidates = (
            gate_sparse_candidates(source, target, config=self.config)
            if candidates is None
            else candidates
        )
        if not isinstance(local_candidates, SparseCandidateCOO):
            raise SparseEdgeError("candidates must be SparseCandidateCOO")
        self._validate_pair(context, source, target, local_candidates)
        source_hidden, target_hidden = self._checkpoint_node_features(context, source, target)
        checkpoint_logits = self._checkpoint_candidate_logits(
            source_hidden, target_hidden, source, target, local_candidates
        )
        residual_features = self._residual_features(
            checkpoint_logits,
            context,
            source,
            target,
            local_candidates,
            previous_velocity_um,
        )
        if (
            tuple(checkpoint_logits.shape) != (len(local_candidates),)
            or tuple(residual_features.shape) != (len(local_candidates), len(_FEATURE_NAMES))
            or not bool(torch.isfinite(checkpoint_logits).all().item())
            or not bool(torch.isfinite(residual_features).all().item())
        ):
            raise SparseEdgeError("sparse candidate features are invalid")
        return local_candidates, checkpoint_logits.detach(), residual_features.detach()

    def forward(
        self,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        *,
        candidates: SparseCandidateCOO | None = None,
        previous_velocity_um: Mapping[int, Sequence[float] | None] | np.ndarray | None = None,
    ) -> torch.Tensor:
        return self.candidate_logits(
            context,
            source,
            target,
            candidates=candidates,
            previous_velocity_um=previous_velocity_um,
        )[1]

    def score_pair(
        self,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        *,
        candidates: SparseCandidateCOO | None = None,
        previous_velocity_um: Mapping[int, Sequence[float] | None] | np.ndarray | None = None,
    ) -> SparsePairProbabilities:
        """Score and source-normalize candidates with one explicit dummy each."""

        local_candidates, logits_tensor = self.candidate_logits(
            context,
            source,
            target,
            candidates=candidates,
            previous_velocity_um=previous_velocity_um,
        )
        logits = logits_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        probabilities = np.empty(len(local_candidates), dtype=np.float64)
        no_link = np.ones(len(source.node_ids), dtype=np.float64)
        dummy = float(self.config.dummy_logit + self.calibrator.dummy_residual.detach().cpu().item())
        if not math.isfinite(dummy):
            raise SparseEdgeError("dummy logit is nonfinite")
        rows_by_source: list[list[int]] = [[] for _ in range(len(source.node_ids))]
        for row, source_index in enumerate(local_candidates.source_index):
            rows_by_source[int(source_index)].append(row)
        for source_index, sparse_rows in enumerate(rows_by_source):
            if not sparse_rows:
                continue
            rows = np.asarray(sparse_rows, dtype=np.int64)
            local_logits = logits[rows]
            maximum = max(dummy, float(np.max(local_logits)))
            dummy_exp = math.exp(dummy - maximum)
            candidate_exp = np.exp(local_logits - maximum)
            denominator = dummy_exp + float(candidate_exp.sum())
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise SparseEdgeError("source probability normalizer is invalid")
            probabilities[rows] = candidate_exp / denominator
            no_link[source_index] = dummy_exp / denominator
        if not np.isfinite(probabilities).all() or not np.isfinite(no_link).all():
            raise SparseEdgeError("source-normalized probabilities contain nonfinite values")
        stats = ProbabilityStats(
            source_count=len(source.node_ids),
            target_count=len(target.node_ids),
            candidate_count=len(local_candidates),
            pair_chunk_size=self.config.pair_chunk_size,
            peak_pair_chunk=min(len(local_candidates), self.config.pair_chunk_size),
            neighbor_cap=self.config.neighbor_cap,
        )
        return SparsePairProbabilities(
            source_frame=source.frame_index,
            target_frame=target.frame_index,
            source_ids=local_candidates.source_ids,
            target_ids=local_candidates.target_ids,
            probabilities=probabilities,
            no_link_source_ids=source.node_ids,
            no_link_probabilities=no_link,
            stats=stats,
        )


@dataclass(frozen=True, slots=True)
class _A0Component:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    candidate_rows: tuple[int, ...]
    sort_key: tuple[tuple[int, ...], tuple[int, ...]]

    @property
    def node_count(self) -> int:
        return len(self.source_indices) + len(self.target_indices)

    @property
    def attention_cells(self) -> int:
        # Peak cells in one attention call.  The eight directional block calls
        # are sequential, so they do not coexist in memory.
        return len(self.source_indices) * len(self.target_indices)


@dataclass(slots=True)
class _A0Audit:
    attention_component_count: int = 0
    node_budget_fallback_component_count: int = 0
    attention_budget_fallback_component_count: int = 0
    unsupported_fallback_component_count: int = 0
    attention_candidate_count: int = 0
    fallback_candidate_count: int = 0
    batch_count: int = 0
    max_batch_components: int = 0
    max_padded_source_nodes: int = 0
    max_padded_target_nodes: int = 0
    peak_padding_nodes: int = 0
    peak_attention_cells: int = 0
    peak_pair_chunk: int = 0


def _a0_components(
    candidates: SparseCandidateCOO,
    source: FrameDet,
    target: FrameDet,
) -> list[_A0Component]:
    """Return deterministic connected components of the sparse bipartite COO."""

    NodeKey = tuple[int, int]
    parent: dict[NodeKey, NodeKey] = {}

    def find(node: NodeKey) -> NodeKey:
        root = parent.setdefault(node, node)
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            previous = parent[node]
            parent[node] = root
            node = previous
        return root

    def union(left: NodeKey, right: NodeKey) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Root choice is independent of candidate row order.
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for source_index, target_index in zip(
        candidates.source_index, candidates.target_index, strict=True
    ):
        union((0, int(source_index)), (1, int(target_index)))

    source_groups: dict[NodeKey, set[int]] = {}
    target_groups: dict[NodeKey, set[int]] = {}
    row_groups: dict[NodeKey, list[int]] = {}
    for row, (source_index, target_index) in enumerate(
        zip(candidates.source_index, candidates.target_index, strict=True)
    ):
        root = find((0, int(source_index)))
        source_groups.setdefault(root, set()).add(int(source_index))
        target_groups.setdefault(root, set()).add(int(target_index))
        row_groups.setdefault(root, []).append(row)

    components: list[_A0Component] = []
    for root, rows in row_groups.items():
        source_indices = tuple(
            sorted(
                source_groups[root],
                key=lambda index: (int(source.node_ids[index]), index),
            )
        )
        target_indices = tuple(
            sorted(
                target_groups[root],
                key=lambda index: (int(target.node_ids[index]), index),
            )
        )
        candidate_rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    int(candidates.source_ids[row]),
                    int(candidates.target_ids[row]),
                ),
            )
        )
        source_key = tuple(int(source.node_ids[index]) for index in source_indices)
        target_key = tuple(int(target.node_ids[index]) for index in target_indices)
        components.append(
            _A0Component(
                source_indices,
                target_indices,
                candidate_rows,
                (source_key, target_key),
            )
        )
    components.sort(key=lambda component: component.sort_key)
    return components


def _power_of_two_bucket(value: int) -> int:
    return 1 << (value - 1).bit_length()


class A0CheckpointScorer(nn.Module):
    """Production-bounded component-local view of the released checkpoint.

    Selecting this class is the explicit A0 opt-in.  It runs the checkpoint's
    exact projection, input norm, four bidirectional temporal blocks, output
    norm, and pair MLP only inside connected candidate components.  Components
    outside hard attention budgets, or batches rejected by the attention path,
    use the legacy sparse projection/norm-out/pair-MLP path instead.
    """

    def __init__(
        self,
        checkpoint_transformer: Any,
        *,
        config: A0ScorerConfig = A0ScorerConfig(),
        pretrained_weight_sha256: str = DEFAULT_WEIGHT_SHA256,
    ) -> None:
        super().__init__()
        if not isinstance(config, A0ScorerConfig):
            raise SparseEdgeError("A0 config must be A0ScorerConfig")
        transformer = _resolve_transformer(checkpoint_transformer)
        blocks = getattr(transformer, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != 4:
            raise SparseEdgeError("A0 checkpoint transformer must expose exactly four blocks")
        if any(not isinstance(block, nn.Module) for block in blocks):
            raise SparseEdgeError("A0 checkpoint blocks contain an unsupported object")
        self.config = config
        self.pretrained_weight_sha256 = _sha256(
            pretrained_weight_sha256, "pretrained_weight_sha256"
        )
        # These are the original loaded modules, not reconstructed equivalents.
        self.proj = transformer.proj
        self.norm_in = transformer.norm_in
        self.blocks = blocks
        self.norm_out = transformer.norm_out
        self.pair_mlp = transformer.pair_mlp
        for module in self._frozen_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self._validate_checkpoint_view()
        self.train(False)

    @classmethod
    def from_loaded_model(
        cls,
        model: Any,
        bundle: ArtifactBundle,
        *,
        config: A0ScorerConfig = A0ScorerConfig(),
    ) -> "A0CheckpointScorer":
        if not isinstance(bundle, ArtifactBundle):
            raise SparseEdgeError("A0 from_loaded_model requires a validated ArtifactBundle")
        return cls(
            model,
            config=config,
            pretrained_weight_sha256=bundle.hashes.weight_sha256,
        )

    def _frozen_modules(self) -> tuple[nn.Module, ...]:
        return self.proj, self.norm_in, self.blocks, self.norm_out, self.pair_mlp

    def _frozen_named_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for module_name, module in (
            ("proj", self.proj),
            ("norm_in", self.norm_in),
            ("blocks", self.blocks),
            ("norm_out", self.norm_out),
            ("pair_mlp", self.pair_mlp),
        ):
            for name, value in module.state_dict().items():
                yield f"{module_name}.{name}", value

    def _validate_checkpoint_view(self) -> None:
        if (
            getattr(self.proj, "in_features", None) != 64
            or getattr(self.proj, "out_features", None) != 128
        ):
            raise SparseEdgeError("A0 checkpoint projection must be Linear(64, 128)")
        for name, module in (("norm_in", self.norm_in), ("norm_out", self.norm_out)):
            if tuple(getattr(module, "normalized_shape", ())) != (128,):
                raise SparseEdgeError(f"A0 checkpoint {name} must normalize 128 features")
        parameters = [
            parameter
            for module in self._frozen_modules()
            for parameter in module.parameters()
        ]
        if not parameters:
            raise SparseEdgeError("A0 checkpoint scorer has no parameters")
        devices = {parameter.device for parameter in parameters}
        dtypes = {parameter.dtype for parameter in parameters}
        if len(devices) != 1 or dtypes != {torch.float32}:
            raise SparseEdgeError("A0 checkpoint scorer must use one device and float32 weights")
        for name, value in self._frozen_named_tensors():
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all().item()):
                raise SparseEdgeError(f"A0 checkpoint tensor {name!r} is not finite float32")
        with torch.no_grad():
            probe = self.pair_mlp(
                torch.zeros((1, 259), dtype=torch.float32, device=parameters[0].device)
            )
        if tuple(probe.shape) != (1, 1) or not bool(torch.isfinite(probe).all().item()):
            raise SparseEdgeError("A0 checkpoint pair_mlp is incompatible with candidate rows")

    def train(self, mode: bool = True) -> "A0CheckpointScorer":
        # A0 is a frozen inference-only checkpoint view; dropout must stay off.
        nn.Module.train(self, False)
        return self

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def frozen_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def frozen_subset_sha256(self) -> str:
        return _tensor_state_hash(self._frozen_named_tensors())

    @property
    def architecture(self) -> dict[str, Any]:
        return {
            "name": "a0-component-local-released-v45",
            "checkpoint_input_dim": 64,
            "checkpoint_hidden_dim": 128,
            "checkpoint_pair_input_dim": 259,
            "frozen_components": [
                "proj",
                "norm_in",
                "blocks.0",
                "blocks.1",
                "blocks.2",
                "blocks.3",
                "norm_out",
                "pair_mlp",
            ],
            "component_order": "node-id-lexicographic",
            "component_batching": "power-of-two-shape-buckets",
            "fallback": "sparse-projection-norm-out-pair-mlp",
            "normalization": "target-wise-source-softmax-no-dummy",
            "pair_scope": "original-candidate-coo-only",
            "full_frame_matrix": False,
            "budgets": self.config.as_dict(),
        }

    @property
    def architecture_sha256(self) -> str:
        return _json_sha256(self.architecture)

    def _device(self) -> torch.device:
        return next(self.proj.parameters()).device

    def _validate_pair(
        self,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        candidates: SparseCandidateCOO,
    ) -> tuple[dict[int, int], dict[int, int]]:
        if not isinstance(context, PairContext):
            raise SparseEdgeError("A0 scoring requires PairContext")
        if not isinstance(source, FrameDet) or not isinstance(target, FrameDet):
            raise SparseEdgeError("A0 scoring requires FrameDet inputs")
        if not isinstance(candidates, SparseCandidateCOO):
            raise SparseEdgeError("A0 candidates must be SparseCandidateCOO")
        if context.frame_pair != (source.frame_index, target.frame_index):
            raise SparseEdgeError("A0 context and detection frames disagree")
        if (
            source.grid_shape_zyx != target.grid_shape_zyx
            or context.grid_shape_zyx != source.grid_shape_zyx
        ):
            raise SparseEdgeError("A0 pair feature grids must have equal shape")
        source_rows = {
            int(node_id): index for index, node_id in enumerate(context.source_ids)
        }
        target_rows = {
            int(node_id): index for index, node_id in enumerate(context.target_ids)
        }
        if (
            len(source_rows) != len(source.node_ids)
            or any(int(node_id) not in source_rows for node_id in source.node_ids)
        ):
            raise SparseEdgeError("A0 context source IDs do not match detections")
        if (
            len(target_rows) != len(target.node_ids)
            or any(int(node_id) not in target_rows for node_id in target.node_ids)
        ):
            raise SparseEdgeError("A0 context target IDs do not match detections")
        if (
            candidates.stats.source_count != len(source.node_ids)
            or candidates.stats.target_count != len(target.node_ids)
        ):
            raise SparseEdgeError("A0 candidate counts do not match detection frames")
        if candidates.stats.neighbor_cap > self.config.neighbor_cap:
            raise SparseEdgeError("A0 candidates exceed the configured neighbor cap")
        source_degrees: dict[int, int] = {}
        for source_index in candidates.source_index:
            index = int(source_index)
            source_degrees[index] = source_degrees.get(index, 0) + 1
        if max(source_degrees.values(), default=0) > self.config.neighbor_cap:
            raise SparseEdgeError("A0 candidate source degree exceeds the neighbor cap")
        if len(candidates):
            if not np.array_equal(
                candidates.source_ids, source.node_ids[candidates.source_index]
            ):
                raise SparseEdgeError("A0 candidate source IDs and indices disagree")
            if not np.array_equal(
                candidates.target_ids, target.node_ids[candidates.target_index]
            ):
                raise SparseEdgeError("A0 candidate target IDs and indices disagree")
        return source_rows, target_rows

    def _component_node_inputs(
        self,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        component: _A0Component,
        source_rows: Mapping[int, int],
        target_rows: Mapping[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        source_indices = np.asarray(component.source_indices, dtype=np.int64)
        target_indices = np.asarray(component.target_indices, dtype=np.int64)
        source_feature_rows = np.asarray(
            [source_rows[int(source.node_ids[index])] for index in source_indices],
            dtype=np.int64,
        )
        target_feature_rows = np.asarray(
            [target_rows[int(target.node_ids[index])] for index in target_indices],
            dtype=np.int64,
        )
        window_shape = (2,) + context.grid_shape_zyx
        source_pos = positional_features(
            np.column_stack(
                (
                    np.zeros(len(source_indices), dtype=np.int64),
                    source.grid_coords_zyx[source_indices],
                )
            ),
            window_shape,
        )
        target_pos = positional_features(
            np.column_stack(
                (
                    np.ones(len(target_indices), dtype=np.int64),
                    target.grid_coords_zyx[target_indices],
                )
            ),
            window_shape,
        )
        source_input = np.concatenate(
            (context.source_feat[source_feature_rows], source_pos), axis=1
        ).astype(np.float32, copy=False)
        target_input = np.concatenate(
            (context.target_feat[target_feature_rows], target_pos), axis=1
        ).astype(np.float32, copy=False)
        if source_input.shape != (len(source_indices), 64) or target_input.shape != (
            len(target_indices),
            64,
        ):
            raise SparseEdgeError("A0 component node inputs must have 64 features")
        if not np.isfinite(source_input).all() or not np.isfinite(target_input).all():
            raise SparseEdgeError("A0 component node inputs contain nonfinite values")
        return np.ascontiguousarray(source_input), np.ascontiguousarray(target_input)

    def _write_component_logits(
        self,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        component: _A0Component,
        source: FrameDet,
        target: FrameDet,
        candidates: SparseCandidateCOO,
        output: torch.Tensor,
        audit: _A0Audit,
    ) -> None:
        self._write_batched_component_logits(
            source_hidden.unsqueeze(0),
            target_hidden.unsqueeze(0),
            (component,),
            source,
            target,
            candidates,
            output,
            audit,
        )

    def _write_batched_component_logits(
        self,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        components: Sequence[_A0Component],
        source: FrameDet,
        target: FrameDet,
        candidates: SparseCandidateCOO,
        output: torch.Tensor,
        audit: _A0Audit,
    ) -> None:
        records: list[tuple[int, int, int, int]] = []
        for batch_index, component in enumerate(components):
            source_local = {
                frame_index: local
                for local, frame_index in enumerate(component.source_indices)
            }
            target_local = {
                frame_index: local
                for local, frame_index in enumerate(component.target_indices)
            }
            records.extend(
                (
                    row,
                    batch_index,
                    source_local[int(candidates.source_index[row])],
                    target_local[int(candidates.target_index[row])],
                )
                for row in component.candidate_rows
            )
        device = self._device()
        for start in range(0, len(records), self.config.pair_chunk_size):
            chunk = records[start : start + self.config.pair_chunk_size]
            chunk_rows = [record[0] for record in chunk]
            source_frame_rows = np.asarray(
                [int(candidates.source_index[row]) for row in chunk_rows],
                dtype=np.int64,
            )
            target_frame_rows = np.asarray(
                [int(candidates.target_index[row]) for row in chunk_rows],
                dtype=np.int64,
            )
            batch_rows = torch.tensor(
                [record[1] for record in chunk],
                dtype=torch.int64,
                device=device,
            )
            source_local_rows = torch.tensor(
                [record[2] for record in chunk], dtype=torch.int64, device=device
            )
            target_local_rows = torch.tensor(
                [record[3] for record in chunk], dtype=torch.int64, device=device
            )
            source_coords = torch.from_numpy(
                np.ascontiguousarray(source.coords_zyx[source_frame_rows], dtype=np.float32)
            ).to(device=device)
            target_coords = torch.from_numpy(
                np.ascontiguousarray(target.coords_zyx[target_frame_rows], dtype=np.float32)
            ).to(device=device)
            pair_features = torch.cat(
                (
                    source_hidden[batch_rows, source_local_rows],
                    target_hidden[batch_rows, target_local_rows],
                    (source_coords - target_coords) / 100.0,
                ),
                dim=1,
            )
            logits = self.pair_mlp(pair_features).squeeze(-1)
            if tuple(logits.shape) != (len(chunk_rows),) or not bool(
                torch.isfinite(logits).all().item()
            ):
                raise SparseEdgeError("A0 checkpoint pair_mlp returned invalid logits")
            output_rows = torch.tensor(chunk_rows, dtype=torch.int64, device=device)
            output[output_rows] = logits.float()
            audit.peak_pair_chunk = max(audit.peak_pair_chunk, len(chunk_rows))

    def _score_projection_fallback(
        self,
        component: _A0Component,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        candidates: SparseCandidateCOO,
        source_rows: Mapping[int, int],
        target_rows: Mapping[int, int],
        output: torch.Tensor,
        audit: _A0Audit,
    ) -> None:
        source_input, target_input = self._component_node_inputs(
            context, source, target, component, source_rows, target_rows
        )
        device = self._device()
        source_tensor = torch.from_numpy(source_input).to(device=device)
        target_tensor = torch.from_numpy(target_input).to(device=device)
        # This is the same bounded checkpoint projection used by SparseEdgeScorer:
        # no attention, no pair matrix, and only candidate rows reach pair_mlp.
        source_hidden = self.norm_out(self.norm_in(self.proj(source_tensor)))
        target_hidden = self.norm_out(self.norm_in(self.proj(target_tensor)))
        if not bool(torch.isfinite(source_hidden).all().item()) or not bool(
            torch.isfinite(target_hidden).all().item()
        ):
            raise SparseEdgeError("A0 projection fallback returned nonfinite node features")
        self._write_component_logits(
            source_hidden,
            target_hidden,
            component,
            source,
            target,
            candidates,
            output,
            audit,
        )

    def _score_attention_batch(
        self,
        batch: Sequence[_A0Component],
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        candidates: SparseCandidateCOO,
        source_rows: Mapping[int, int],
        target_rows: Mapping[int, int],
        output: torch.Tensor,
        audit: _A0Audit,
    ) -> None:
        batch_size = len(batch)
        max_source = max(len(component.source_indices) for component in batch)
        max_target = max(len(component.target_indices) for component in batch)
        attention_cells = batch_size * max_source * max_target
        if attention_cells > self.config.attention_cell_budget:
            raise SparseEdgeError("A0 internal attention batch exceeds its hard budget")
        device = self._device()
        source_input = torch.zeros(
            (batch_size, max_source, 64), dtype=torch.float32, device=device
        )
        target_input = torch.zeros(
            (batch_size, max_target, 64), dtype=torch.float32, device=device
        )
        source_mask = torch.zeros((batch_size, max_source), dtype=torch.bool, device=device)
        target_mask = torch.zeros((batch_size, max_target), dtype=torch.bool, device=device)
        real_source_nodes = 0
        real_target_nodes = 0
        for batch_index, component in enumerate(batch):
            local_source, local_target = self._component_node_inputs(
                context, source, target, component, source_rows, target_rows
            )
            source_count = len(component.source_indices)
            target_count = len(component.target_indices)
            source_input[batch_index, :source_count] = torch.from_numpy(local_source).to(
                device=device
            )
            target_input[batch_index, :target_count] = torch.from_numpy(local_target).to(
                device=device
            )
            source_mask[batch_index, :source_count] = True
            target_mask[batch_index, :target_count] = True
            real_source_nodes += source_count
            real_target_nodes += target_count

        audit.batch_count += 1
        audit.max_batch_components = max(audit.max_batch_components, batch_size)
        audit.max_padded_source_nodes = max(audit.max_padded_source_nodes, max_source)
        audit.max_padded_target_nodes = max(audit.max_padded_target_nodes, max_target)
        audit.peak_padding_nodes = max(
            audit.peak_padding_nodes,
            batch_size * (max_source + max_target) - real_source_nodes - real_target_nodes,
        )
        audit.peak_attention_cells = max(audit.peak_attention_cells, attention_cells)

        q = self.norm_in(self.proj(source_input))
        k = self.norm_in(self.proj(target_input))
        for block in self.blocks:
            q = block(q, k, kv_mask=target_mask)
            k = block(k, q, kv_mask=source_mask)
        q = self.norm_out(q)
        k = self.norm_out(k)
        if not bool(torch.isfinite(q).all().item()) or not bool(torch.isfinite(k).all().item()):
            raise SparseEdgeError("A0 attention returned nonfinite node features")
        self._write_batched_component_logits(
            q, k, batch, source, target, candidates, output, audit
        )

    def _attention_batches(
        self, components: Sequence[_A0Component]
    ) -> list[list[_A0Component]]:
        ordered = sorted(
            components,
            key=lambda component: (
                _power_of_two_bucket(len(component.source_indices)),
                _power_of_two_bucket(len(component.target_indices)),
                component.sort_key,
            ),
        )
        batches: list[list[_A0Component]] = []
        current: list[_A0Component] = []
        current_bucket: tuple[int, int] | None = None
        current_max_source = 0
        current_max_target = 0
        for component in ordered:
            bucket = (
                _power_of_two_bucket(len(component.source_indices)),
                _power_of_two_bucket(len(component.target_indices)),
            )
            proposed_max_source = max(
                current_max_source, len(component.source_indices)
            )
            proposed_max_target = max(
                current_max_target, len(component.target_indices)
            )
            proposed_cells = (
                (len(current) + 1) * proposed_max_source * proposed_max_target
            )
            if current and (
                bucket != current_bucket
                or proposed_cells > self.config.attention_cell_budget
            ):
                batches.append(current)
                current = []
                current_max_source = 0
                current_max_target = 0
            current.append(component)
            current_max_source = max(
                current_max_source, len(component.source_indices)
            )
            current_max_target = max(
                current_max_target, len(component.target_indices)
            )
            current_bucket = bucket
        if current:
            batches.append(current)
        return batches

    def candidate_logits(
        self,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        *,
        candidates: SparseCandidateCOO,
    ) -> tuple[SparseCandidateCOO, torch.Tensor, A0AssociationStats]:
        """Return frozen A0 logits and audit data in the original COO order."""

        # The loaded model may be shared with another caller that toggled its
        # parent back to training mode.  Reassert eval mode before every A0 run.
        self.train(False)
        source_rows, target_rows = self._validate_pair(
            context, source, target, candidates
        )
        components = _a0_components(candidates, source, target)
        audit = _A0Audit()
        device = self._device()
        logits = torch.full(
            (len(candidates),), float("nan"), dtype=torch.float32, device=device
        )
        attention_components: list[_A0Component] = []
        with torch.inference_mode():
            for component in components:
                if component.node_count > self.config.max_component_nodes:
                    audit.node_budget_fallback_component_count += 1
                    audit.fallback_candidate_count += len(component.candidate_rows)
                    self._score_projection_fallback(
                        component,
                        context,
                        source,
                        target,
                        candidates,
                        source_rows,
                        target_rows,
                        logits,
                        audit,
                    )
                elif component.attention_cells > self.config.attention_cell_budget:
                    audit.attention_budget_fallback_component_count += 1
                    audit.fallback_candidate_count += len(component.candidate_rows)
                    self._score_projection_fallback(
                        component,
                        context,
                        source,
                        target,
                        candidates,
                        source_rows,
                        target_rows,
                        logits,
                        audit,
                    )
                else:
                    attention_components.append(component)

            for batch in self._attention_batches(attention_components):
                try:
                    self._score_attention_batch(
                        batch,
                        context,
                        source,
                        target,
                        candidates,
                        source_rows,
                        target_rows,
                        logits,
                        audit,
                    )
                except Exception:
                    # Invalid inputs/checkpoints were rejected before this point.
                    # A component-local runtime incompatibility cannot bypass
                    # scoring: overwrite every row with the bounded fallback.
                    audit.unsupported_fallback_component_count += len(batch)
                    for component in batch:
                        audit.fallback_candidate_count += len(component.candidate_rows)
                        self._score_projection_fallback(
                            component,
                            context,
                            source,
                            target,
                            candidates,
                            source_rows,
                            target_rows,
                            logits,
                            audit,
                        )
                else:
                    audit.attention_component_count += len(batch)
                    audit.attention_candidate_count += sum(
                        len(component.candidate_rows) for component in batch
                    )
        if tuple(logits.shape) != (len(candidates),) or not bool(
            torch.isfinite(logits).all().item()
        ):
            raise SparseEdgeError("A0 did not produce one finite logit per candidate")

        fallback_components = (
            audit.node_budget_fallback_component_count
            + audit.attention_budget_fallback_component_count
            + audit.unsupported_fallback_component_count
        )
        component_denominator = max(len(components), 1)
        candidate_denominator = max(len(candidates), 1)
        stats = A0AssociationStats(
            source_count=len(source.node_ids),
            target_count=len(target.node_ids),
            candidate_count=len(candidates),
            component_count=len(components),
            attention_component_count=audit.attention_component_count,
            fallback_component_count=fallback_components,
            node_budget_fallback_component_count=audit.node_budget_fallback_component_count,
            attention_budget_fallback_component_count=(
                audit.attention_budget_fallback_component_count
            ),
            unsupported_fallback_component_count=audit.unsupported_fallback_component_count,
            attention_candidate_count=audit.attention_candidate_count,
            fallback_candidate_count=audit.fallback_candidate_count,
            attention_component_fraction=(
                audit.attention_component_count / component_denominator
            ),
            fallback_component_fraction=fallback_components / component_denominator,
            attention_candidate_fraction=(
                audit.attention_candidate_count / candidate_denominator
            ),
            fallback_candidate_fraction=audit.fallback_candidate_count / candidate_denominator,
            max_component_source_nodes=max(
                (len(component.source_indices) for component in components), default=0
            ),
            max_component_target_nodes=max(
                (len(component.target_indices) for component in components), default=0
            ),
            max_component_nodes=max(
                (component.node_count for component in components), default=0
            ),
            max_component_candidates=max(
                (len(component.candidate_rows) for component in components), default=0
            ),
            max_component_attention_cells=max(
                (component.attention_cells for component in components), default=0
            ),
            batch_count=audit.batch_count,
            max_batch_components=audit.max_batch_components,
            max_padded_source_nodes=audit.max_padded_source_nodes,
            max_padded_target_nodes=audit.max_padded_target_nodes,
            peak_padding_nodes=audit.peak_padding_nodes,
            peak_attention_cells=audit.peak_attention_cells,
            neighbor_cap=self.config.neighbor_cap,
            max_component_nodes_budget=self.config.max_component_nodes,
            attention_cell_budget=self.config.attention_cell_budget,
            pair_chunk_size=self.config.pair_chunk_size,
            peak_pair_chunk=audit.peak_pair_chunk,
        )
        return candidates, logits, stats

    def forward(
        self,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        *,
        candidates: SparseCandidateCOO,
    ) -> torch.Tensor:
        return self.candidate_logits(
            context, source, target, candidates=candidates
        )[1]

    def score_pair(
        self,
        context: PairContext,
        source: FrameDet,
        target: FrameDet,
        *,
        candidates: SparseCandidateCOO,
    ) -> A0PairProbabilities:
        """Score original candidates with released target-wise softmax semantics."""

        local_candidates, logits_tensor, stats = self.candidate_logits(
            context, source, target, candidates=candidates
        )
        logits = logits_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        probabilities = np.empty(len(local_candidates), dtype=np.float64)
        rows_by_target: dict[int, list[int]] = {}
        for row, target_index in enumerate(local_candidates.target_index):
            rows_by_target.setdefault(int(target_index), []).append(row)
        for sparse_rows in rows_by_target.values():
            rows = np.asarray(
                sorted(
                    sparse_rows,
                    key=lambda row: (
                        int(local_candidates.source_ids[row]),
                        int(local_candidates.target_ids[row]),
                    ),
                ),
                dtype=np.int64,
            )
            local_logits = logits[rows]
            maximum = float(np.max(local_logits))
            exponentials = np.exp(local_logits - maximum)
            denominator = float(exponentials.sum())
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise SparseEdgeError("A0 target probability normalizer is invalid")
            probabilities[rows] = exponentials / denominator
        if not np.isfinite(probabilities).all():
            raise SparseEdgeError("A0 target-normalized probabilities contain nonfinite values")
        return A0PairProbabilities(
            source_frame=source.frame_index,
            target_frame=target.frame_index,
            source_ids=local_candidates.source_ids,
            target_ids=local_candidates.target_ids,
            probabilities=probabilities,
            stats=stats,
        )


def score_sparse_pair(
    scorer: SparseEdgeScorer,
    context: PairContext,
    source: FrameDet,
    target: FrameDet,
    **kwargs: Any,
) -> SparsePairProbabilities:
    if not isinstance(scorer, SparseEdgeScorer):
        raise SparseEdgeError("scorer must be SparseEdgeScorer")
    return scorer.score_pair(context, source, target, **kwargs)


def learned_edge_probability_map(
    pairs: Iterable[SparsePairProbabilities],
) -> dict[tuple[int, int], float]:
    """Merge pair-scoped sparse maps without changing their probabilities."""

    result: dict[tuple[int, int], float] = {}
    for pair in pairs:
        if not isinstance(pair, SparsePairProbabilities):
            raise SparseEdgeError("probability collection contains an unsupported record")
        for key, probability in pair.items():
            if key in result:
                raise SparseEdgeError("duplicate learned probability across frame pairs")
            result[key] = probability
    return result


@dataclass(frozen=True, slots=True)
class SparsePairLabels:
    """Tri-state labels: one/zero are known; minus one is always ignored."""

    source_ids: np.ndarray
    target_ids: np.ndarray
    labels: np.ndarray
    known: np.ndarray

    def __post_init__(self) -> None:
        source_ids = _readonly_1d(self.source_ids, "<i8", "label source IDs")
        target_ids = _readonly_1d(self.target_ids, "<i8", "label target IDs")
        labels = _readonly_1d(self.labels, "<i1", "sparse pair labels")
        known_input = np.asarray(self.known)
        if known_input.ndim != 1 or known_input.dtype != np.bool_:
            raise SparseEdgeError("known-label mask must be a one-dimensional boolean array")
        known_raw = np.ascontiguousarray(known_input)
        if not (len(source_ids) == len(target_ids) == len(labels) == len(known_raw)):
            raise SparseEdgeError("sparse label arrays must have equal length")
        if np.any(~np.isin(labels, (UNKNOWN_LABEL, 0, 1))):
            raise SparseEdgeError("sparse labels must be -1, 0, or 1")
        if np.any(~np.isin(known_raw, (0, 1))):
            raise SparseEdgeError("known-label mask must be boolean")
        known = np.ascontiguousarray(known_raw)
        known.setflags(write=False)
        if not np.array_equal(known, labels != UNKNOWN_LABEL):
            raise SparseEdgeError("unknown labels and known mask disagree")
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "known", known)


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise SparseEdgeError(f"{label} must be an integer, not bool")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SparseEdgeError(f"{label} must be an integer") from exc
    if not math.isfinite(numeric) or numeric != result or result < 0:
        raise SparseEdgeError(f"{label} must be an exact nonnegative integer")
    return result


def label_sparse_training_pairs(
    candidates: SparseCandidateCOO,
    detection_to_gt: Mapping[int, int | None],
    gt_edges: Iterable[Sequence[int]],
) -> SparsePairLabels:
    """Label only candidate pairs whose two detection endpoints are matched.

    A fully matched non-edge is a known negative.  Missing, ``None``, or
    partially matched endpoints remain ``UNKNOWN_LABEL`` and are never folded
    into the negative class.
    """

    if not isinstance(candidates, SparseCandidateCOO):
        raise SparseEdgeError("training labels require SparseCandidateCOO")
    if not isinstance(detection_to_gt, Mapping):
        raise SparseEdgeError("detection_to_gt must be a mapping")
    matches: dict[int, int | None] = {}
    for raw_detection, raw_gt in detection_to_gt.items():
        detection = _exact_nonnegative_int(raw_detection, "detection ID")
        if detection in matches:
            raise SparseEdgeError("duplicate detection match")
        matches[detection] = None if raw_gt is None else _exact_nonnegative_int(raw_gt, "GT ID")
    edge_set: set[tuple[int, int]] = set()
    for row in gt_edges:
        if isinstance(row, (str, bytes)):
            raise SparseEdgeError("GT edges must contain two IDs")
        try:
            if len(row) != 2:
                raise SparseEdgeError("GT edges must contain two IDs")
            edge = (
                _exact_nonnegative_int(row[0], "GT edge source"),
                _exact_nonnegative_int(row[1], "GT edge target"),
            )
        except TypeError as exc:
            raise SparseEdgeError("GT edges must contain two IDs") from exc
        if edge in edge_set:
            raise SparseEdgeError("duplicate sparse GT edge")
        edge_set.add(edge)
    labels = np.full(len(candidates), UNKNOWN_LABEL, dtype=np.int8)
    for index, (source_id, target_id) in enumerate(
        zip(candidates.source_ids, candidates.target_ids, strict=True)
    ):
        source_gt = matches.get(int(source_id))
        target_gt = matches.get(int(target_id))
        if source_gt is None or target_gt is None:
            continue
        labels[index] = 1 if (source_gt, target_gt) in edge_set else 0
    return SparsePairLabels(
        candidates.source_ids,
        candidates.target_ids,
        labels,
        labels != UNKNOWN_LABEL,
    )


@dataclass(frozen=True, slots=True)
class ExposureMetadata:
    """Training-data exposure identity stored beside residual weights."""

    fold: int
    source_groups: tuple[str, ...] = ()
    source_video_ids: tuple[str, ...] = ()
    split_manifest_sha256: str | None = None
    provenance_manifest_sha256: str | None = None
    audit_exposed: bool = False
    test_exposed: bool = False

    def __post_init__(self) -> None:
        if type(self.fold) is not int or self.fold < -1:
            raise SparseEdgeError("exposure fold must be an integer >= -1")
        for name in ("source_groups", "source_video_ids"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or any(not isinstance(value, str) or not value for value in values)
                or tuple(sorted(set(values))) != values
            ):
                raise SparseEdgeError(f"{name} must be sorted unique nonempty strings")
        for name in ("split_manifest_sha256", "provenance_manifest_sha256"):
            value = getattr(self, name)
            if value is not None:
                _sha256(value, name)
        if type(self.audit_exposed) is not bool or type(self.test_exposed) is not bool:
            raise SparseEdgeError("exposure flags must be bool")

    def as_dict(self) -> dict[str, Any]:
        return {
            "audit_exposed": self.audit_exposed,
            "fold": self.fold,
            "provenance_manifest_sha256": self.provenance_manifest_sha256,
            "source_groups": list(self.source_groups),
            "source_video_ids": list(self.source_video_ids),
            "split_manifest_sha256": self.split_manifest_sha256,
            "test_exposed": self.test_exposed,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ExposureMetadata":
        expected = {
            "audit_exposed",
            "fold",
            "provenance_manifest_sha256",
            "source_groups",
            "source_video_ids",
            "split_manifest_sha256",
            "test_exposed",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise EdgeArtifactError("artifact exposure metadata has missing or unexpected fields")
        if not isinstance(value["source_groups"], list) or not isinstance(value["source_video_ids"], list):
            raise EdgeArtifactError("artifact exposure lists are invalid")
        try:
            return cls(
                fold=value["fold"],
                source_groups=tuple(value["source_groups"]),
                source_video_ids=tuple(value["source_video_ids"]),
                split_manifest_sha256=value["split_manifest_sha256"],
                provenance_manifest_sha256=value["provenance_manifest_sha256"],
                audit_exposed=value["audit_exposed"],
                test_exposed=value["test_exposed"],
            )
        except (TypeError, SparseEdgeError) as exc:
            raise EdgeArtifactError(f"invalid exposure metadata: {exc}") from exc


@dataclass(frozen=True, slots=True)
class EdgeArtifactIdentity:
    architecture_sha256: str
    config_sha256: str
    pretrained_weight_sha256: str
    frozen_subset_sha256: str
    residual_weight_sha256: str
    auxiliary_checkpoint_sha256: str | None
    auxiliary_state_sha256: str | None
    exposure: ExposureMetadata

    def __post_init__(self) -> None:
        for name in (
            "architecture_sha256",
            "config_sha256",
            "pretrained_weight_sha256",
            "frozen_subset_sha256",
            "residual_weight_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (self.auxiliary_checkpoint_sha256 is None) != (self.auxiliary_state_sha256 is None):
            raise SparseEdgeError("auxiliary artifact identity must be fully present or absent")
        if self.auxiliary_checkpoint_sha256 is not None:
            _sha256(self.auxiliary_checkpoint_sha256, "auxiliary_checkpoint_sha256")
            _sha256(self.auxiliary_state_sha256, "auxiliary_state_sha256")
        if not isinstance(self.exposure, ExposureMetadata):
            raise SparseEdgeError("artifact identity requires ExposureMetadata")


def _calibrator_tensors(scorer: SparseEdgeScorer) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, tensor in sorted(scorer.calibrator.state_dict().items()):
        detached = tensor.detach().cpu().contiguous()
        if detached.dtype != torch.float32 or not bool(torch.isfinite(detached).all().item()):
            raise EdgeArtifactError(f"residual tensor {name!r} is not finite float32")
        result.append(
            {
                "data_base64": base64.b64encode(detached.numpy().tobytes(order="C")).decode("ascii"),
                "dtype": "float32",
                "name": name,
                "shape": list(detached.shape),
            }
        )
    return result


def _serialized_residual_hash(tensors: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(tensors)).hexdigest()


def _artifact_payload(
    scorer: SparseEdgeScorer,
    exposure: ExposureMetadata,
) -> tuple[dict[str, Any], EdgeArtifactIdentity]:
    if not isinstance(scorer, SparseEdgeScorer):
        raise EdgeArtifactError("artifact scorer must be SparseEdgeScorer")
    if not isinstance(exposure, ExposureMetadata):
        raise EdgeArtifactError("artifact exposure must be ExposureMetadata")
    tensors = _calibrator_tensors(scorer)
    residual_hash = _serialized_residual_hash(tensors)
    auxiliary = {"enabled": False}
    auxiliary_checkpoint_sha256: str | None = None
    auxiliary_state_sha256: str | None = None
    if scorer.auxiliary is not None:
        auxiliary_checkpoint_sha256 = scorer.auxiliary.checkpoint_sha256
        auxiliary_state_sha256 = scorer.auxiliary.identity.auxiliary_state_sha256
        auxiliary = {
            "checkpoint_sha256": auxiliary_checkpoint_sha256,
            "enabled": True,
            "identity": scorer.auxiliary.identity.as_dict(),
        }
    identity = EdgeArtifactIdentity(
        architecture_sha256=scorer.architecture_sha256,
        config_sha256=scorer.config_sha256,
        pretrained_weight_sha256=scorer.pretrained_weight_sha256,
        frozen_subset_sha256=scorer.frozen_subset_sha256,
        residual_weight_sha256=residual_hash,
        auxiliary_checkpoint_sha256=auxiliary_checkpoint_sha256,
        auxiliary_state_sha256=auxiliary_state_sha256,
        exposure=exposure,
    )
    payload = {
        "architecture": scorer.architecture,
        "architecture_sha256": identity.architecture_sha256,
        "config": scorer.config.as_dict(),
        "config_sha256": identity.config_sha256,
        "auxiliary": auxiliary,
        "exposure": exposure.as_dict(),
        "pretrained": {
            "checkpoint_weight_sha256": identity.pretrained_weight_sha256,
            "frozen_subset_sha256": identity.frozen_subset_sha256,
        },
        "schema": EDGE_ARTIFACT_SCHEMA,
        "weights": {
            "encoding": "base64-little-endian-float32",
            "residual_weight_sha256": residual_hash,
            "tensors": tensors,
        },
    }
    return payload, identity


def save_edge_artifact(
    path: str | os.PathLike[str],
    scorer: SparseEdgeScorer,
    exposure: ExposureMetadata,
) -> EdgeArtifactIdentity:
    """Atomically write deterministic JSON; no pickle loader is involved."""

    destination = Path(path)
    if destination.exists() and not destination.is_file():
        raise EdgeArtifactError("artifact destination is not a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload, identity = _artifact_payload(scorer, exposure)
    raw = _canonical_json(payload) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise EdgeArtifactError(f"failed to write sparse edge artifact: {exc}") from exc
    return identity


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EdgeArtifactError(f"duplicate artifact JSON key {key!r}")
        result[key] = value
    return result


def _load_artifact_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise EdgeArtifactError("artifact must be a regular JSON file no larger than 2 MB")
    try:
        value = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=_reject_duplicate_pairs)
    except EdgeArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EdgeArtifactError(f"artifact is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EdgeArtifactError("artifact top level must be an object")
    return value


def _decode_calibrator_state(
    value: Any,
    scorer: SparseEdgeScorer,
) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(value, dict) or set(value) != {
        "encoding",
        "residual_weight_sha256",
        "tensors",
    }:
        raise EdgeArtifactError("artifact weights have missing or unexpected fields")
    if value["encoding"] != "base64-little-endian-float32" or not isinstance(value["tensors"], list):
        raise EdgeArtifactError("artifact residual encoding is unsupported")
    recorded_hash = _sha256(value["residual_weight_sha256"], "residual_weight_sha256")
    if _serialized_residual_hash(value["tensors"]) != recorded_hash:
        raise EdgeArtifactError("residual weight hash mismatch")
    expected = scorer.calibrator.state_dict()
    state: dict[str, torch.Tensor] = {}
    previous_name: str | None = None
    for row in value["tensors"]:
        if not isinstance(row, dict) or set(row) != {"data_base64", "dtype", "name", "shape"}:
            raise EdgeArtifactError("artifact residual tensor schema is invalid")
        name = row["name"]
        if not isinstance(name, str) or name not in expected or (previous_name is not None and name <= previous_name):
            raise EdgeArtifactError("artifact residual tensor names are invalid or unsorted")
        previous_name = name
        reference = expected[name]
        if row["dtype"] != "float32" or row["shape"] != list(reference.shape):
            raise EdgeArtifactError(f"artifact residual tensor {name!r} has wrong shape or dtype")
        if not isinstance(row["data_base64"], str):
            raise EdgeArtifactError("artifact residual tensor bytes are invalid")
        try:
            raw = base64.b64decode(row["data_base64"], validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise EdgeArtifactError("artifact residual tensor is not strict base64") from exc
        if len(raw) != reference.numel() * 4:
            raise EdgeArtifactError(f"artifact residual tensor {name!r} byte count mismatch")
        array = np.frombuffer(raw, dtype="<f4").copy().reshape(tuple(reference.shape))
        if not np.isfinite(array).all():
            raise EdgeArtifactError(f"artifact residual tensor {name!r} contains nonfinite values")
        state[name] = torch.from_numpy(array)
    if set(state) != set(expected):
        raise EdgeArtifactError("artifact residual tensor set is incomplete")
    return state, recorded_hash


def _artifact_auxiliary(
    value: Any,
    auxiliary: V45AuxReadout | None,
) -> V45AuxReadout | None:
    if not isinstance(value, dict) or "enabled" not in value or type(value["enabled"]) is not bool:
        raise EdgeArtifactError("artifact auxiliary metadata is invalid")
    if value["enabled"] is False:
        if set(value) != {"enabled"}:
            raise EdgeArtifactError("disabled auxiliary metadata has unexpected fields")
        if auxiliary is not None:
            raise EdgeArtifactError("artifact disables the supplied auxiliary readout")
        return None
    if set(value) != {"checkpoint_sha256", "enabled", "identity"}:
        raise EdgeArtifactError("enabled auxiliary metadata has missing or unexpected fields")
    if not isinstance(auxiliary, V45AuxReadout):
        raise EdgeArtifactError("artifact requires a validated auxiliary readout")
    checkpoint_sha256 = _sha256(value["checkpoint_sha256"], "auxiliary_checkpoint_sha256")
    if checkpoint_sha256 != auxiliary.checkpoint_sha256:
        raise EdgeArtifactError("auxiliary checkpoint identity mismatch")
    if value["identity"] != auxiliary.identity.as_dict():
        raise EdgeArtifactError("auxiliary state identity mismatch")
    return auxiliary


def load_edge_artifact(
    path: str | os.PathLike[str],
    checkpoint_transformer: Any,
    *,
    expected_pretrained_weight_sha256: str = DEFAULT_WEIGHT_SHA256,
    auxiliary: V45AuxReadout | None = None,
) -> tuple[SparseEdgeScorer, EdgeArtifactIdentity]:
    """Restore JSON residual weights against the exact supplied frozen view."""

    expected_checkpoint = _sha256(
        expected_pretrained_weight_sha256, "expected_pretrained_weight_sha256"
    )
    payload = _load_artifact_json(Path(path))
    required_top = {
        "architecture",
        "architecture_sha256",
        "auxiliary",
        "config",
        "config_sha256",
        "exposure",
        "pretrained",
        "schema",
        "weights",
    }
    if set(payload) != required_top or payload.get("schema") != EDGE_ARTIFACT_SCHEMA:
        raise EdgeArtifactError("artifact top-level schema has missing or unexpected fields")
    config = SparseEdgeConfig.from_dict(payload["config"])
    artifact_auxiliary = _artifact_auxiliary(payload["auxiliary"], auxiliary)
    scorer = SparseEdgeScorer(
        checkpoint_transformer,
        config=config,
        pretrained_weight_sha256=expected_checkpoint,
        auxiliary=artifact_auxiliary,
    )
    architecture_hash = _sha256(payload["architecture_sha256"], "architecture_sha256")
    config_hash = _sha256(payload["config_sha256"], "config_sha256")
    if _json_sha256(payload["architecture"]) != architecture_hash or payload["architecture"] != scorer.architecture:
        raise EdgeArtifactError("artifact architecture hash or schema mismatch")
    if _json_sha256(payload["config"]) != config_hash or config_hash != scorer.config_sha256:
        raise EdgeArtifactError("artifact config hash mismatch")
    pretrained = payload["pretrained"]
    if not isinstance(pretrained, dict) or set(pretrained) != {
        "checkpoint_weight_sha256",
        "frozen_subset_sha256",
    }:
        raise EdgeArtifactError("artifact pretrained identity is invalid")
    recorded_checkpoint = _sha256(
        pretrained["checkpoint_weight_sha256"], "checkpoint_weight_sha256"
    )
    recorded_subset = _sha256(pretrained["frozen_subset_sha256"], "frozen_subset_sha256")
    if recorded_checkpoint != expected_checkpoint:
        raise EdgeArtifactError("pretrained checkpoint weight hash mismatch")
    if recorded_subset != scorer.frozen_subset_sha256:
        raise EdgeArtifactError("frozen checkpoint subset hash mismatch")
    exposure = ExposureMetadata.from_dict(payload["exposure"])
    state, residual_hash = _decode_calibrator_state(payload["weights"], scorer)
    try:
        incompatible = scorer.calibrator.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise EdgeArtifactError(f"strict residual restore failed: {exc}") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise EdgeArtifactError("strict residual restore returned incompatible keys")
    if any(
        parameter.dtype != torch.float32 or not bool(torch.isfinite(parameter).all().item())
        for parameter in scorer.calibrator.parameters()
    ):
        raise EdgeArtifactError("restored residual weights are nonfinite")
    identity = EdgeArtifactIdentity(
        architecture_sha256=architecture_hash,
        config_sha256=config_hash,
        pretrained_weight_sha256=recorded_checkpoint,
        frozen_subset_sha256=recorded_subset,
        residual_weight_sha256=residual_hash,
        auxiliary_checkpoint_sha256=(
            None if artifact_auxiliary is None else artifact_auxiliary.checkpoint_sha256
        ),
        auxiliary_state_sha256=(
            None
            if artifact_auxiliary is None
            else artifact_auxiliary.identity.auxiliary_state_sha256
        ),
        exposure=exposure,
    )
    return scorer, identity


# Compact aliases for callers that namespace the module rather than its types.
candidate_gate = gate_sparse_candidates
score_pair = score_sparse_pair
build_training_labels = label_sparse_training_pairs


__all__ = [
    "A0_EDGE_SCHEMA",
    "A0AssociationStats",
    "A0CheckpointScorer",
    "A0PairProbabilities",
    "A0ScorerConfig",
    "EDGE_ARTIFACT_SCHEMA",
    "EDGE_SCHEMA",
    "UNKNOWN_LABEL",
    "CandidateStats",
    "EdgeArtifactError",
    "EdgeArtifactIdentity",
    "ExposureMetadata",
    "ProbabilityStats",
    "SparseCandidateCOO",
    "SparseEdgeConfig",
    "SparseEdgeError",
    "SparseEdgeScorer",
    "SparsePairLabels",
    "SparsePairProbabilities",
    "build_training_labels",
    "candidate_gate",
    "gate_sparse_candidates",
    "label_sparse_training_pairs",
    "learned_edge_probability_map",
    "load_edge_artifact",
    "save_edge_artifact",
    "score_pair",
    "score_sparse_pair",
]
