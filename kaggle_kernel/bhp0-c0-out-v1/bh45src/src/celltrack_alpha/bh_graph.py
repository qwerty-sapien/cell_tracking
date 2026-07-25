"""Sparse semantic rewrite of exact scored public V45 postprocessing.

The source of truth is ``reports/g02/v45.ipynb`` at scriptVersionId
335810749.  This detector- and storage-agnostic rewrite preserves its scored
graph semantics while replacing full-frame dense assignments with sparse
candidate discovery and sparse assignment.  The notebook's 14 um threshold is
an input/output edge-validation limit; motion relinking itself uses raw
physical-distance gates of 6 um and 10 um.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
import math
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree


GRAPH_SCHEMA = "celltrack-alpha.bh-v45-graph.v1"
EXPERIMENTAL_SWEEP_SCHEMA = "celltrack-alpha.bh-v45-graph-sweep.v1"
PHYSICAL_SCALE_ZYX = (1.625, 0.40625, 0.40625)


class GraphCalibrationError(ValueError):
    """Base error for malformed input or an unsafe graph-calibration state."""


class SparseAssignmentError(GraphCalibrationError):
    """A sparse assignment input or augmented graph was malformed."""


class GraphInvariantError(GraphCalibrationError):
    """A graph violates the V45 lineage invariants."""


_V45_LOCKED_DEFAULTS: dict[str, Any] = {
    "schema": GRAPH_SCHEMA,
    "scale_zyx": PHYSICAL_SCALE_ZYX,
    "output_edge_max_um": 14.0,
    "motion_tight_um": 6.0,
    "motion_relaxed_um": 10.0,
    "velocity_weight": 0.5,
    "motion_direction": "backward",
    "motion_error_weight": 0.25,
    "learned_bonus": 1.0,
    "raw_distance_weight": 0.05,
    "max_nodes_per_frame": 2600,
    "repair_single_parent": True,
    "repair_single_child": False,
    "gap_missing_frames": 1,
    "gap_step_um": 5.8,
    "gap_total_um": 11.6,
    "spacing_neighbors": 3,
    "spacing_ref_um": 6.5,
    "density_gain": 0.0475,
    "density_delta_clip_um": 0.125,
    "midpoint_reuse_um": 3.2,
    "gap_add_fraction": 0.05,
    "gap_add_absolute": 2000,
    "outside_090_gate": False,
    "refine_z_radius": 1,
    "refine_yx_radius": 3,
    "refine_percentile": 20.0,
    "refine_max_shift_um": 3.2,
    "gap2_enabled": False,
    "division_candidate_child_um": 4.66,
    "division_existing_child_um": 7.65,
    "division_sister_um": 8.5,
    "division_sister_weight": 0.15,
    "division_frame_fraction": 0.0076,
    "division_global_fraction": 0.00375,
    "division_geometry_filter": False,
    "prune_isolated": True,
    "min_component_nodes": 6,
    "preserve_division_components": True,
    "smooth_window": 2,
    "smooth_weight": 0.8,
}


@dataclass(frozen=True, slots=True)
class V45GraphConfig:
    """Immutable, locked public V45 graph-calibration constants."""

    schema: str = GRAPH_SCHEMA
    scale_zyx: tuple[float, float, float] = PHYSICAL_SCALE_ZYX
    output_edge_max_um: float = 14.0
    motion_tight_um: float = 6.0
    motion_relaxed_um: float = 10.0
    velocity_weight: float = 0.5
    motion_direction: str = "backward"
    motion_error_weight: float = 0.25
    learned_bonus: float = 1.0
    raw_distance_weight: float = 0.05
    max_nodes_per_frame: int = 2600
    repair_single_parent: bool = True
    repair_single_child: bool = False
    gap_missing_frames: int = 1
    gap_step_um: float = 5.8
    gap_total_um: float = 11.6
    spacing_neighbors: int = 3
    spacing_ref_um: float = 6.5
    density_gain: float = 0.0475
    density_delta_clip_um: float = 0.125
    midpoint_reuse_um: float = 3.2
    gap_add_fraction: float = 0.05
    gap_add_absolute: int = 2000
    outside_090_gate: bool = False
    refine_z_radius: int = 1
    refine_yx_radius: int = 3
    refine_percentile: float = 20.0
    refine_max_shift_um: float = 3.2
    gap2_enabled: bool = False
    division_candidate_child_um: float = 4.66
    division_existing_child_um: float = 7.65
    division_sister_um: float = 8.5
    division_sister_weight: float = 0.15
    division_frame_fraction: float = 0.0076
    division_global_fraction: float = 0.00375
    division_geometry_filter: bool = False
    prune_isolated: bool = True
    min_component_nodes: int = 6
    preserve_division_components: bool = True
    smooth_window: int = 2
    smooth_weight: float = 0.8

    def __post_init__(self) -> None:
        for name, expected in _V45_LOCKED_DEFAULTS.items():
            if getattr(self, name) != expected:
                raise ValueError(f"V45 graph field {name!r} is locked to {expected!r}")

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            result[item.name] = list(value) if isinstance(value, tuple) else value
        return result


V45_GRAPH_CONFIG = V45GraphConfig()
# A short alias is convenient for callers that keep model and graph modules
# namespaced rather than importing both constants into the same namespace.
V45_CONFIG = V45_GRAPH_CONFIG


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One detector or synthesized node in voxel ``(z, y, x)`` coordinates."""

    node_id: int
    t: int
    z: float
    y: float
    x: float
    score: float = 1.0
    synthetic: bool = False

    @property
    def coord_zyx(self) -> tuple[float, float, float]:
        return float(self.z), float(self.y), float(self.x)

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "node_id": int(self.node_id),
            "t": int(self.t),
            "z": self.z,
            "y": self.y,
            "x": self.x,
            "score": float(self.score),
            "synthetic": bool(self.synthetic),
        }


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """One directed, next-frame lineage edge."""

    source_id: int
    target_id: int
    distance_um: float
    cost: float
    kind: str = "adjacent"
    learned_probability: float | None = None

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "source_id": int(self.source_id),
            "target_id": int(self.target_id),
            "distance_um": float(self.distance_um),
            "cost": float(self.cost),
            "kind": self.kind,
            "learned_probability": self.learned_probability,
        }


@dataclass(frozen=True, slots=True)
class ErlangDivisionPrior:
    """Fixed-shape Erlang prior for an experimental division-opening cost."""

    shape: int
    rate: float

    def __post_init__(self) -> None:
        if type(self.shape) is not int or self.shape <= 0:
            raise GraphCalibrationError("Erlang shape must be a positive integer")
        if not math.isfinite(self.rate) or self.rate <= 0.0:
            raise GraphCalibrationError("Erlang rate must be finite and positive")

    def negative_log_density(self, elapsed_frames: int) -> float:
        """Return ``-log p(dt)`` with the continuous Erlang density."""

        if type(elapsed_frames) is not int or elapsed_frames < 0:
            raise GraphCalibrationError("elapsed division lifetime must be a nonnegative integer")
        if elapsed_frames == 0:
            if self.shape == 1:
                return -math.log(self.rate)
            return math.inf
        value = (
            math.lgamma(self.shape)
            - self.shape * math.log(self.rate)
            - (self.shape - 1) * math.log(elapsed_frames)
            + self.rate * elapsed_frames
        )
        if not math.isfinite(value):
            raise GraphCalibrationError("Erlang division-opening cost is non-finite")
        return value


@dataclass(frozen=True, slots=True)
class LearnedEdgeProbability:
    """Primitive optional learned evidence for one adjacent candidate edge."""

    source_id: int
    target_id: int
    probability: float


@dataclass(frozen=True, slots=True)
class SparseMatch:
    source_id: int
    target_id: int
    cost: float


@dataclass(frozen=True, slots=True)
class SparseMatchResult:
    matches: tuple[SparseMatch, ...]
    feasible_pair_count: int
    component_count: int
    peak_component_matrix_dim: int


@dataclass(frozen=True, slots=True)
class GraphResult:
    """Calibrated graph plus auditable counters and algorithm provenance."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    stats: dict[str, int | float | bool]
    provenance: dict[str, Any]


NodeLike = GraphNode | Mapping[str, Any] | Any
ProbabilityLike = LearnedEdgeProbability | Mapping[str, Any] | Sequence[Any]
ImageSource = Mapping[int, np.ndarray] | Callable[[int], np.ndarray]
RefineCallback = Callable[[int, tuple[float, float, float]], Sequence[float] | None]


def _validated_experimental_sweep_provenance(value: Any) -> dict[str, Any]:
    """Validate a non-production graph-config unlock contract."""

    if not isinstance(value, Mapping):
        raise GraphCalibrationError("non-default graph config requires sweep provenance")
    required = {
        "schema",
        "experiment_id",
        "purpose",
        "varied_fields",
        "production_eligible",
        "submit_eligible",
    }
    if set(value) != required:
        raise GraphCalibrationError("experimental sweep provenance has missing or unexpected fields")
    varied = value["varied_fields"]
    purpose = value["purpose"]
    eligibility = {
        "offline-cached-cv57-sweep": False,
        "public-metric-parity-calibration": True,
    }
    if (
        value["schema"] != EXPERIMENTAL_SWEEP_SCHEMA
        or not isinstance(value["experiment_id"], str)
        or not value["experiment_id"]
        or purpose not in eligibility
        or not isinstance(varied, list)
        or not varied
        or set(varied) - {
            "candidate_top_k",
            "learned_bonus",
            "raw_distance_weight",
            "motion_error_weight",
        }
        or value["production_eligible"] is not False
        or value["submit_eligible"] is not eligibility[purpose]
    ):
        raise GraphCalibrationError("experimental sweep provenance has invalid eligibility")
    return dict(value)


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise GraphCalibrationError(f"{label} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GraphCalibrationError(f"{label} must be an integer") from exc
    try:
        exact = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GraphCalibrationError(f"{label} must be an integer") from exc
    if not math.isfinite(exact) or exact != result:
        raise GraphCalibrationError(f"{label} must be an exact finite integer")
    return result


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GraphCalibrationError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise GraphCalibrationError(f"{label} must be finite")
    return result


def _record_value(record: Any, names: Sequence[str]) -> Any:
    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return record[name]
    else:
        for name in names:
            if hasattr(record, name):
                return getattr(record, name)
    raise GraphCalibrationError(f"node is missing required field {names[0]!r}")


def _coerce_node(record: NodeLike) -> GraphNode:
    if isinstance(record, GraphNode):
        if not isinstance(record.synthetic, (bool, np.bool_)):
            raise GraphCalibrationError("node synthetic must be bool")
        node = GraphNode(
            node_id=_strict_int(record.node_id, "node_id"),
            t=_strict_int(record.t, "t"),
            z=_finite_float(record.z, "z"),
            y=_finite_float(record.y, "y"),
            x=_finite_float(record.x, "x"),
            score=_finite_float(record.score, "score"),
            synthetic=bool(record.synthetic),
        )
    else:
        node_id = _record_value(record, ("node_id", "id"))
        t = _record_value(record, ("t", "frame", "frame_index"))
        try:
            z = _record_value(record, ("z",))
            y = _record_value(record, ("y",))
            x = _record_value(record, ("x",))
        except GraphCalibrationError:
            coords = _record_value(record, ("coord_zyx", "coords_zyx"))
            try:
                coord_count = len(coords)
            except TypeError as exc:
                raise GraphCalibrationError("node coord_zyx must have exactly three values") from exc
            if isinstance(coords, (str, bytes)) or coord_count != 3:
                raise GraphCalibrationError("node coord_zyx must have exactly three values")
            z, y, x = coords
        if isinstance(record, Mapping):
            score = record.get("score", 1.0)
            synthetic = record.get("synthetic", False)
        else:
            score = getattr(record, "score", 1.0)
            synthetic = getattr(record, "synthetic", False)
        if not isinstance(synthetic, (bool, np.bool_)):
            raise GraphCalibrationError("node synthetic must be bool")
        node = GraphNode(
            node_id=_strict_int(node_id, "node_id"),
            t=_strict_int(t, "t"),
            z=_finite_float(z, "z"),
            y=_finite_float(y, "y"),
            x=_finite_float(x, "x"),
            score=_finite_float(score, "score"),
            synthetic=bool(synthetic),
        )
    if node.node_id < 0:
        raise GraphCalibrationError("node_id must be nonnegative")
    if node.t < 0:
        raise GraphCalibrationError("t must be nonnegative")
    for name, value in zip(("z", "y", "x"), node.coord_zyx, strict=True):
        if not math.isfinite(value):
            raise GraphCalibrationError(f"node {name} must be finite")
    if not math.isfinite(float(node.score)):
        raise GraphCalibrationError("node score must be finite")
    return node


def normalize_nodes(
    nodes: Iterable[NodeLike],
    *,
    max_nodes_per_frame: int = V45_GRAPH_CONFIG.max_nodes_per_frame,
) -> tuple[GraphNode, ...]:
    """Normalize generic records and fail closed on IDs or frame overflow."""

    limit = _strict_int(max_nodes_per_frame, "max_nodes_per_frame")
    if limit <= 0 or limit > V45_GRAPH_CONFIG.max_nodes_per_frame:
        raise GraphCalibrationError(
            f"max_nodes_per_frame must be in [1, {V45_GRAPH_CONFIG.max_nodes_per_frame}]"
        )
    result = tuple(sorted((_coerce_node(record) for record in nodes), key=lambda n: (n.t, n.node_id)))
    ids = [node.node_id for node in result]
    if len(ids) != len(set(ids)):
        raise GraphCalibrationError("duplicate node ID")
    counts: dict[int, int] = defaultdict(int)
    for node in result:
        counts[node.t] += 1
        if counts[node.t] > limit:
            raise GraphCalibrationError(
                f"frame {node.t} exceeds the {limit} node fail-closed cap"
            )
    return result


def _coord(value: GraphNode | Sequence[float]) -> np.ndarray:
    if isinstance(value, GraphNode):
        raw = value.coord_zyx
    else:
        raw = value
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise GraphCalibrationError("coordinates must be three finite z/y/x values")
    return array


def physical_distance(
    first: GraphNode | Sequence[float],
    second: GraphNode | Sequence[float],
    scale_zyx: Sequence[float] = PHYSICAL_SCALE_ZYX,
) -> float:
    """Return anisotropic Euclidean distance in micrometres."""

    scale = np.asarray(scale_zyx, dtype=np.float64)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise GraphCalibrationError("scale_zyx must contain three positive finite values")
    return float(np.linalg.norm((_coord(first) - _coord(second)) * scale))


distance_um = physical_distance


def _coerce_probability_map(
    values: Mapping[Any, Any] | Iterable[ProbabilityLike] | None,
    node_by_id: Mapping[int, GraphNode],
) -> dict[tuple[int, int], float]:
    if values is None:
        return {}
    rows: list[tuple[Any, Any, Any]] = []
    if isinstance(values, Mapping):
        for key, probability in values.items():
            if not isinstance(key, Sequence) or isinstance(key, (str, bytes)) or len(key) != 2:
                raise GraphCalibrationError("probability mapping keys must be (source_id, target_id)")
            rows.append((key[0], key[1], probability))
    else:
        for item in values:
            if isinstance(item, LearnedEdgeProbability):
                rows.append((item.source_id, item.target_id, item.probability))
            elif isinstance(item, Mapping):
                rows.append((item.get("source_id"), item.get("target_id"), item.get("probability")))
            else:
                try:
                    item_count = len(item)
                except TypeError as exc:
                    raise GraphCalibrationError("malformed learned edge probability record") from exc
                if isinstance(item, (str, bytes)) or item_count != 3:
                    raise GraphCalibrationError("malformed learned edge probability record")
                rows.append((item[0], item[1], item[2]))
    result: dict[tuple[int, int], float] = {}
    for raw_source, raw_target, raw_probability in rows:
        source = _strict_int(raw_source, "source_id")
        target = _strict_int(raw_target, "target_id")
        probability = _finite_float(raw_probability, "probability")
        if not 0.0 <= probability <= 1.0:
            raise GraphCalibrationError("learned edge probability must be in [0, 1]")
        pair = (source, target)
        if pair in result:
            raise GraphCalibrationError("duplicate learned edge probability")
        if source not in node_by_id or target not in node_by_id:
            raise GraphCalibrationError("learned edge probability references an unknown node")
        if node_by_id[target].t != node_by_id[source].t + 1:
            raise GraphCalibrationError("learned edge probability must be adjacent and forward")
        result[pair] = probability
    return result


def _coerce_no_link_probability_map(
    values: Mapping[Any, Any] | None,
    node_by_id: Mapping[int, GraphNode],
) -> dict[int, float]:
    """Validate source-owned dummy probabilities without inferring missing rows."""

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise GraphCalibrationError("source no-link probabilities must be a mapping")
    result: dict[int, float] = {}
    for raw_source, raw_probability in values.items():
        source = _strict_int(raw_source, "no-link source_id")
        probability = _finite_float(raw_probability, "no-link probability")
        if source not in node_by_id:
            raise GraphCalibrationError("no-link probability references an unknown node")
        if not 0.0 <= probability <= 1.0:
            raise GraphCalibrationError("no-link probability must be in [0, 1]")
        if source in result:
            raise GraphCalibrationError("duplicate no-link probability")
        result[source] = probability
    return result


def _coerce_feasible_pair(record: Any) -> tuple[int, int, float]:
    if isinstance(record, Mapping):
        source = record.get("source_id")
        target = record.get("target_id")
        cost = record.get("cost")
    else:
        try:
            record_count = len(record)
        except TypeError:
            record_count = -1
        if not isinstance(record, (str, bytes)) and record_count == 3:
            source, target, cost = record
        else:
            source = getattr(record, "source_id", None)
            target = getattr(record, "target_id", None)
            cost = getattr(record, "cost", None)
    return (
        _strict_int(source, "source_id"),
        _strict_int(target, "target_id"),
        _finite_float(cost, "cost"),
    )


def sparse_min_cost_max_cardinality(
    source_ids: Iterable[int],
    target_ids: Iterable[int],
    feasible_pairs: Iterable[Any],
    *,
    max_component_matrix_dim: int | None = None,
    parity_mode: str = "sparse",
    dense_unmatched_cost: float | None = None,
) -> SparseMatchResult:
    """Solve exact cardinality-first, cost-second sparse bipartite matching.

    Each connected feasible-pair component is represented by an augmented
    sparse square assignment graph.  Real source-to-target edges encode the
    supplied costs, own-dummy edges encode unmatched endpoints, and reciprocal
    dummy edges preserve feasibility without a dense dummy-to-dummy block.
    The unmatched penalty makes cardinality lexicographically dominant over
    cost.  ``max_component_matrix_dim`` remains as a compatibility argument;
    sparse components are never rejected because of that former dense cap.
    ``dense_diagnostic`` is an explicit full-frame LAPJV oracle for measuring
    implementation-dependent ties and is never selected by default.
    """

    if parity_mode not in {"sparse", "dense_diagnostic"}:
        raise SparseAssignmentError("parity_mode must be 'sparse' or 'dense_diagnostic'")
    if max_component_matrix_dim is not None:
        legacy_cap = _strict_int(max_component_matrix_dim, "max_component_matrix_dim")
        if legacy_cap <= 0:
            raise SparseAssignmentError("max_component_matrix_dim must be positive")
    sources = tuple(sorted(_strict_int(value, "source_id") for value in source_ids))
    targets = tuple(sorted(_strict_int(value, "target_id") for value in target_ids))
    if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
        raise SparseAssignmentError("duplicate source or target ID")
    source_set = set(sources)
    target_set = set(targets)
    source_rank = {node_id: index for index, node_id in enumerate(sources)}
    target_rank = {node_id: index for index, node_id in enumerate(targets)}
    pair_cost: dict[tuple[int, int], float] = {}
    source_neighbors: dict[int, set[int]] = defaultdict(set)
    target_neighbors: dict[int, set[int]] = defaultdict(set)
    for record in feasible_pairs:
        source, target, cost = _coerce_feasible_pair(record)
        if source not in source_set or target not in target_set:
            raise SparseAssignmentError("feasible pair references a node outside its bipartition")
        pair = (source, target)
        if pair in pair_cost:
            raise SparseAssignmentError("duplicate feasible pair")
        if abs(cost) > 1.0e100:
            raise SparseAssignmentError("feasible-pair cost magnitude is unsafe")
        pair_cost[pair] = cost
        source_neighbors[source].add(target)
        target_neighbors[target].add(source)

    components: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    visited_sources: set[int] = set()
    visited_targets: set[int] = set()
    for start in sources:
        if start in visited_sources or not source_neighbors.get(start):
            continue
        component_sources: set[int] = set()
        component_targets: set[int] = set()
        queue: deque[tuple[bool, int]] = deque([(True, start)])
        while queue:
            is_source, node_id = queue.popleft()
            if is_source:
                if node_id in visited_sources:
                    continue
                visited_sources.add(node_id)
                component_sources.add(node_id)
                for neighbor in sorted(source_neighbors[node_id]):
                    if neighbor not in visited_targets:
                        queue.append((False, neighbor))
            else:
                if node_id in visited_targets:
                    continue
                visited_targets.add(node_id)
                component_targets.add(node_id)
                for neighbor in sorted(target_neighbors[node_id]):
                    if neighbor not in visited_sources:
                        queue.append((True, neighbor))
        components.append((tuple(sorted(component_sources)), tuple(sorted(component_targets))))

    peak = max(
        (len(component_sources) + len(component_targets) for component_sources, component_targets in components),
        default=0,
    )
    if parity_mode == "dense_diagnostic":
        # This branch intentionally reproduces public LAPJV's full rectangular
        # tie behavior.  It is an explicit oracle only; every default caller
        # remains on the component-local sparse augmented solver below.
        from scipy.optimize import linear_sum_assignment

        max_cost = max(pair_cost.values(), default=0.0)
        if dense_unmatched_cost is None:
            cost_extent = max(1.0, max((abs(cost) for cost in pair_cost.values()), default=0.0))
            big = (min(len(sources), len(targets)) + 1.0) * (2.0 * cost_extent + 1.0)
        else:
            big = _finite_float(dense_unmatched_cost, "dense_unmatched_cost")
        if big <= max_cost:
            raise SparseAssignmentError("dense diagnostic unmatched cost must exceed every feasible cost")
        dense_cost = np.full((len(sources), len(targets)), big, dtype=np.float64)
        source_index = {node_id: index for index, node_id in enumerate(sources)}
        target_index = {node_id: index for index, node_id in enumerate(targets)}
        for (source, target), cost in pair_cost.items():
            dense_cost[source_index[source], target_index[target]] = cost
        row_indices, column_indices = linear_sum_assignment(dense_cost)
        dense_matches = tuple(
            SparseMatch(sources[row], targets[column], pair_cost[sources[row], targets[column]])
            for row, column in zip(row_indices.tolist(), column_indices.tolist(), strict=True)
            if (sources[row], targets[column]) in pair_cost
        )
        return SparseMatchResult(
            matches=dense_matches,
            feasible_pair_count=len(pair_cost),
            component_count=len(components),
            peak_component_matrix_dim=peak,
        )

    matches: list[SparseMatch] = []
    for component_sources, component_targets in components:
        left_count = len(component_sources)
        right_count = len(component_targets)
        dimension = left_count + right_count
        component_target_set = set(component_targets)
        component_pairs = [
            (source, target, pair_cost[(source, target)])
            for source in component_sources
            for target in sorted(source_neighbors[source])
            if target in component_target_set
        ]
        cost_scale = max(1.0, max(abs(cost) for _, _, cost in component_pairs))
        # For a fixed real-match cardinality, c/scale + 2 is a positive affine
        # transform of the original total cost.  Positive, nonzero weights are
        # required because sparse matrices treat explicit zero as no edge.
        base_real_weights = [
            cost / cost_scale + 2.0 for _, _, cost in component_pairs
        ]
        max_real_weight = max(base_real_weights)
        # Keep sparse ties deterministic while leaving near-but-distinct
        # learned/motion costs untouched.  Dense LAPJV's tie outcome also
        # depends on its full matrix of infeasible filler entries; exact replay
        # of that implementation artifact is isolated above as a diagnostic.
        tie_unit = 8.0 * float(np.spacing(max_real_weight))
        duplicate_costs = {
            cost
            for cost, count in Counter(
                cost for _, _, cost in component_pairs
            ).items()
            if count > 1
        }
        endpoint_priority_scale = len(sources) + len(targets) + 1
        reciprocal_weight = 1.0
        cardinality_bound = min(left_count, right_count)
        unmatched_penalty = (
            cardinality_bound + 1.0
        ) * (max_real_weight + reciprocal_weight + 1.0)
        if not math.isfinite(unmatched_penalty):
            raise SparseAssignmentError("component assignment penalty overflow")

        left_index = {node_id: index for index, node_id in enumerate(component_sources)}
        right_index = {node_id: index for index, node_id in enumerate(component_targets)}
        rows: list[int] = []
        columns: list[int] = []
        weights: list[float] = []
        for source, target, cost in component_pairs:
            source_row = left_index[source]
            target_column = right_index[target]
            rows.extend((source_row, left_count + target_column))
            columns.extend((target_column, right_count + source_row))
            tie_rank = 0
            if cost in duplicate_costs:
                if len(sources) > len(targets):
                    # LAPJV transposes tall rectangular inputs.  Mirror that
                    # target-row ordering, including scarce-target priority,
                    # instead of comparing unrelated source/target ranks.
                    tie_rank = abs(source_rank[source] - target_rank[target])
                    tie_rank += endpoint_priority_scale * (
                        source_rank[source]
                        + target_rank[target]
                        + endpoint_priority_scale
                        * (len(target_neighbors[target]) - 1)
                    )
                else:
                    tie_rank = abs(source_rank[source] - target_rank[target])
                    tie_rank += endpoint_priority_scale * source_rank[source]
            tie_cost = tie_unit * tie_rank
            weights.extend(
                (cost / cost_scale + 2.0 + tie_cost, reciprocal_weight)
            )
        for source_row in range(left_count):
            rows.append(source_row)
            columns.append(right_count + source_row)
            weights.append(unmatched_penalty)
        for target_column in range(right_count):
            rows.append(left_count + target_column)
            columns.append(target_column)
            weights.append(unmatched_penalty)

        matrix = coo_matrix(
            (np.asarray(weights, dtype=np.float64), (rows, columns)),
            shape=(dimension, dimension),
        ).tocsr()
        matrix.sort_indices()
        try:
            row_indices, column_indices = min_weight_full_bipartite_matching(matrix)
        except ValueError as exc:
            raise SparseAssignmentError("sparse augmented assignment has no full matching") from exc
        selected: list[SparseMatch] = []
        for row, column in zip(row_indices.tolist(), column_indices.tolist(), strict=True):
            if row >= left_count or column >= right_count:
                continue
            source = component_sources[row]
            target = component_targets[column]
            if (source, target) in pair_cost:
                selected.append(SparseMatch(source, target, pair_cost[(source, target)]))
        matches.extend(sorted(selected, key=lambda match: (match.source_id, match.target_id)))

    return SparseMatchResult(
        matches=tuple(sorted(matches, key=lambda match: (match.source_id, match.target_id))),
        feasible_pair_count=len(pair_cost),
        component_count=len(components),
        peak_component_matrix_dim=peak,
    )


def sparse_min_cost_partial_matching(
    source_ids: Iterable[int],
    target_ids: Iterable[int],
    feasible_pairs: Iterable[Any],
    no_link_probabilities: Mapping[int, float],
) -> SparseMatchResult:
    """Solve sparse one-to-one matching with an explicit source no-link cost.

    Unlike :func:`sparse_min_cost_max_cardinality`, this routine permits a
    source to choose its own dummy column.  The sparse reciprocal construction
    keeps every component feasible without allocating a dense dummy block.
    Candidate costs are normally ``-log(p_edge)`` and dummy costs are
    ``-log(p_no_link)``; all inputs must therefore be finite and nonnegative.
    """

    sources = tuple(sorted(_strict_int(value, "source_id") for value in source_ids))
    targets = tuple(sorted(_strict_int(value, "target_id") for value in target_ids))
    if not sources or not targets:
        return SparseMatchResult((), 0, 0, 0)
    if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
        raise SparseAssignmentError("duplicate source or target ID")
    source_set = set(sources)
    target_set = set(targets)
    dummy_cost: dict[int, float] = {}
    for source in sources:
        if source not in no_link_probabilities:
            raise SparseAssignmentError("partial assignment is missing a source no-link cost")
        cost = _finite_float(no_link_probabilities[source], "no-link cost")
        if cost < 0.0 or cost > 1.0e100:
            raise SparseAssignmentError("no-link cost is outside the safe range")
        dummy_cost[source] = cost

    pair_cost: dict[tuple[int, int], float] = {}
    source_neighbors: dict[int, set[int]] = defaultdict(set)
    target_neighbors: dict[int, set[int]] = defaultdict(set)
    for record in feasible_pairs:
        source, target, cost = _coerce_feasible_pair(record)
        if source not in source_set or target not in target_set:
            raise SparseAssignmentError("feasible pair references a node outside its bipartition")
        if cost < 0.0 or cost > 1.0e100:
            raise SparseAssignmentError("partial feasible-pair cost is outside the safe range")
        pair = (source, target)
        if pair in pair_cost:
            raise SparseAssignmentError("duplicate feasible pair")
        pair_cost[pair] = cost
        source_neighbors[source].add(target)
        target_neighbors[target].add(source)

    components: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    visited_sources: set[int] = set()
    visited_targets: set[int] = set()
    for start in sorted(source_neighbors):
        if start in visited_sources:
            continue
        queue: deque[tuple[bool, int]] = deque(((True, start),))
        component_sources: set[int] = set()
        component_targets: set[int] = set()
        while queue:
            is_source, node_id = queue.popleft()
            if is_source:
                if node_id in visited_sources:
                    continue
                visited_sources.add(node_id)
                component_sources.add(node_id)
                queue.extend((False, target) for target in sorted(source_neighbors[node_id]))
            else:
                if node_id in visited_targets:
                    continue
                visited_targets.add(node_id)
                component_targets.add(node_id)
                queue.extend((True, source) for source in sorted(target_neighbors[node_id]))
        components.append((tuple(sorted(component_sources)), tuple(sorted(component_targets))))

    matches: list[SparseMatch] = []
    peak = 0
    for component_sources, component_targets in components:
        left_count = len(component_sources)
        right_count = len(component_targets)
        dimension = left_count + right_count
        peak = max(peak, dimension)
        left_index = {node_id: index for index, node_id in enumerate(component_sources)}
        right_index = {node_id: index for index, node_id in enumerate(component_targets)}
        component_target_set = set(component_targets)
        component_pairs = [
            (source, target, pair_cost[source, target])
            for source in component_sources
            for target in sorted(source_neighbors[source])
            if target in component_target_set
        ]
        scale = max(
            1.0,
            max((cost for _, _, cost in component_pairs), default=0.0),
            max((dummy_cost[source] for source in component_sources), default=0.0),
        )
        tie_unit = 8.0 * float(np.spacing(scale))
        rows: list[int] = []
        columns: list[int] = []
        weights: list[float] = []
        for source, target, cost in component_pairs:
            source_row = left_index[source]
            target_column = right_index[target]
            tie_rank = source_row * (right_count + 1) + target_column + 1
            rows.extend((source_row, left_count + target_column))
            columns.extend((target_column, right_count + source_row))
            weights.extend((1.0 + cost + tie_unit * tie_rank, 1.0))
        for source, source_row in left_index.items():
            rows.append(source_row)
            columns.append(right_count + source_row)
            # A zero tie offset makes no-link the deterministic preference when
            # an edge and its dummy have exactly equal negative log likelihood.
            weights.append(1.0 + dummy_cost[source])
        for target_column in range(right_count):
            rows.append(left_count + target_column)
            columns.append(target_column)
            weights.append(1.0)
        matrix = coo_matrix(
            (np.asarray(weights, dtype=np.float64), (rows, columns)),
            shape=(dimension, dimension),
        ).tocsr()
        matrix.sort_indices()
        try:
            row_indices, column_indices = min_weight_full_bipartite_matching(matrix)
        except ValueError as exc:
            raise SparseAssignmentError("sparse partial assignment has no full matching") from exc
        for row, column in zip(row_indices.tolist(), column_indices.tolist(), strict=True):
            if row >= left_count or column >= right_count:
                continue
            source = component_sources[row]
            target = component_targets[column]
            if (source, target) in pair_cost:
                matches.append(SparseMatch(source, target, pair_cost[source, target]))

    return SparseMatchResult(
        matches=tuple(sorted(matches, key=lambda match: (match.source_id, match.target_id))),
        feasible_pair_count=len(pair_cost),
        component_count=len(components),
        peak_component_matrix_dim=peak,
    )


sparse_bipartite_match = sparse_min_cost_max_cardinality


def _frame_groups(nodes: Sequence[GraphNode]) -> dict[int, tuple[GraphNode, ...]]:
    groups: dict[int, list[GraphNode]] = defaultdict(list)
    for node in nodes:
        groups[node.t].append(node)
    return {frame: tuple(sorted(rows, key=lambda node: node.node_id)) for frame, rows in groups.items()}


def _physical_array(nodes: Sequence[GraphNode], scale_zyx: Sequence[float]) -> np.ndarray:
    coords = np.asarray([node.coord_zyx for node in nodes], dtype=np.float64).reshape((-1, 3))
    return coords * np.asarray(scale_zyx, dtype=np.float64)


def _radius_pairs(
    sources: Sequence[GraphNode],
    targets: Sequence[GraphNode],
    radius_um: float,
    scale_zyx: Sequence[float],
) -> list[tuple[GraphNode, GraphNode, float]]:
    if not sources or not targets:
        return []
    source_physical = _physical_array(sources, scale_zyx)
    target_physical = _physical_array(targets, scale_zyx)
    target_tree = cKDTree(target_physical)
    result: list[tuple[GraphNode, GraphNode, float]] = []
    for source_index, neighbor_indices in enumerate(
        target_tree.query_ball_point(source_physical, r=float(radius_um), return_sorted=True)
    ):
        source = sources[source_index]
        for target_index in neighbor_indices:
            target = targets[int(target_index)]
            distance = float(np.linalg.norm(source_physical[source_index] - target_physical[target_index]))
            if distance <= radius_um:
                result.append((source, target, distance))
    return result


def _top_k_pairs(
    sources: Sequence[GraphNode],
    targets: Sequence[GraphNode],
    top_k: int,
    scale_zyx: Sequence[float],
) -> list[tuple[GraphNode, GraphNode, float]]:
    """Return each source's nearest targets with deterministic boundary ties."""

    if not sources or not targets:
        return []
    neighbor_count = _strict_int(top_k, "candidate_top_k")
    if neighbor_count <= 0:
        raise GraphCalibrationError("candidate_top_k must be positive")
    neighbor_count = min(neighbor_count, len(targets))
    source_physical = _physical_array(sources, scale_zyx)
    target_physical = _physical_array(targets, scale_zyx)
    target_tree = cKDTree(target_physical)
    queried_distances, _ = target_tree.query(source_physical, k=neighbor_count)
    queried_distances = np.asarray(queried_distances, dtype=np.float64)
    if queried_distances.ndim == 1:
        queried_distances = queried_distances[:, None]
    result: list[tuple[GraphNode, GraphNode, float]] = []
    for source_index, source in enumerate(sources):
        boundary = float(queried_distances[source_index, -1])
        tolerance = 8.0 * float(np.spacing(max(1.0, abs(boundary))))
        neighbor_indices = target_tree.query_ball_point(
            source_physical[source_index],
            r=boundary + tolerance,
            return_sorted=True,
        )
        ranked = sorted(
            (
                float(
                    np.linalg.norm(
                        source_physical[source_index] - target_physical[int(target_index)]
                    )
                ),
                targets[int(target_index)].node_id,
                int(target_index),
            )
            for target_index in neighbor_indices
        )
        if len(ranked) < neighbor_count:
            raise GraphInvariantError("top-k query returned fewer candidates than requested")
        for distance, _target_id, target_index in ranked[:neighbor_count]:
            result.append((source, targets[target_index], distance))
    return result


def _candidate_components(
    pairs: Sequence[tuple[GraphNode, GraphNode, float]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]], ...]:
    """Return deterministic connected components of a sparse candidate graph."""

    source_neighbors: dict[int, set[int]] = defaultdict(set)
    target_neighbors: dict[int, set[int]] = defaultdict(set)
    for source, target, _distance in pairs:
        source_neighbors[source.node_id].add(target.node_id)
        target_neighbors[target.node_id].add(source.node_id)
    visited_sources: set[int] = set()
    visited_targets: set[int] = set()
    result: list[tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]]] = []
    for start in sorted(source_neighbors):
        if start in visited_sources:
            continue
        queue: deque[tuple[bool, int]] = deque(((True, start),))
        sources: set[int] = set()
        targets: set[int] = set()
        while queue:
            is_source, node_id = queue.popleft()
            if is_source:
                if node_id in visited_sources:
                    continue
                visited_sources.add(node_id)
                sources.add(node_id)
                queue.extend((False, target) for target in sorted(source_neighbors[node_id]))
            else:
                if node_id in visited_targets:
                    continue
                visited_targets.add(node_id)
                targets.add(node_id)
                queue.extend((True, source) for source in sorted(target_neighbors[node_id]))
        component_pairs = tuple(
            (source, target)
            for source in sorted(sources)
            for target in sorted(source_neighbors[source])
            if target in targets
        )
        result.append((tuple(sorted(sources)), tuple(sorted(targets)), component_pairs))
    return tuple(result)


def _peak_sparse_component_dim(
    sources: Sequence[GraphNode],
    targets: Sequence[GraphNode],
    pairs: Sequence[tuple[GraphNode, GraphNode, float]],
) -> int:
    """Return the largest incident source-plus-target component size."""

    source_index = {node.node_id: index for index, node in enumerate(sources)}
    target_index = {
        node.node_id: len(sources) + index for index, node in enumerate(targets)
    }
    parent = list(range(len(sources) + len(targets)))
    size = [1] * len(parent)
    incident: set[int] = set()

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for source, target, _ in pairs:
        left = source_index[source.node_id]
        right = target_index[target.node_id]
        incident.update((left, right))
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if size[left_root] < size[right_root]:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
            size[left_root] += size[right_root]
    return max((size[find(index)] for index in incident), default=0)


def _accumulate_match_stats(stats: dict[str, int], result: SparseMatchResult) -> None:
    stats["sparse_component_count"] += result.component_count
    stats["peak_component_matrix_dim"] = max(
        stats["peak_component_matrix_dim"], result.peak_component_matrix_dim
    )


def motion_only_edge_cost(
    distance_um: float,
    motion_error_um: float | None = None,
    *,
    config: V45GraphConfig = V45_GRAPH_CONFIG,
) -> float:
    """Public detector+motion ablation score with no learned contribution."""

    distance = _finite_float(distance_um, "distance_um")
    if distance < 0:
        raise GraphCalibrationError("distance_um must be nonnegative")
    cost = config.raw_distance_weight * distance
    if motion_error_um is not None:
        error = _finite_float(motion_error_um, "motion_error_um")
        if error < 0:
            raise GraphCalibrationError("motion_error_um must be nonnegative")
        cost += error
    return float(cost)


def learned_edge_score_bonus(
    probability: float | None,
    *,
    config: V45GraphConfig = V45_GRAPH_CONFIG,
) -> float:
    """Public optional learned hook; absent evidence contributes exactly zero."""

    if probability is None:
        return 0.0
    value = _finite_float(probability, "probability")
    if not 0.0 <= value <= 1.0:
        raise GraphCalibrationError("probability must be in [0, 1]")
    return float(config.learned_bonus * value)


def _edge_cost(
    distance: float,
    probability: float,
    config: V45GraphConfig,
    *,
    motion_error: float | None = None,
    motion_error_weight: float = 1.0,
) -> float:
    weight = _finite_float(motion_error_weight, "motion_error_weight")
    if weight < 0.0:
        raise GraphCalibrationError("motion_error_weight must be nonnegative")
    weighted_motion_error = (
        None if motion_error is None else weight * float(motion_error)
    )
    return motion_only_edge_cost(
        distance, weighted_motion_error, config=config
    ) - learned_edge_score_bonus(probability, config=config)


def build_adjacent_edges(
    nodes: Iterable[NodeLike],
    learned_edge_probabilities: Mapping[Any, Any] | Iterable[ProbabilityLike] | None = None,
    source_no_link_probabilities: Mapping[Any, Any] | None = None,
    *,
    max_component_matrix_dim: int | None = None,
    parity_mode: str = "sparse",
    topology_mode: str = "legacy",
    config: V45GraphConfig = V45_GRAPH_CONFIG,
    _experimental_sweep_provenance: Mapping[str, Any] | None = None,
    _experimental_motion_error_weight: float | None = None,
    _experimental_candidate_top_k: int | None = None,
) -> tuple[tuple[GraphEdge, ...], dict[str, int | bool]]:
    """Build adjacent edges with the notebook's two raw-distance motion passes."""

    if config is not V45_GRAPH_CONFIG and config != V45_GRAPH_CONFIG:
        _validated_experimental_sweep_provenance(_experimental_sweep_provenance)
    elif _experimental_sweep_provenance is not None:
        _validated_experimental_sweep_provenance(_experimental_sweep_provenance)
    if _experimental_motion_error_weight is None:
        motion_error_weight = config.motion_error_weight
    else:
        sweep = _validated_experimental_sweep_provenance(
            _experimental_sweep_provenance
        )
        if "motion_error_weight" not in sweep["varied_fields"]:
            raise GraphCalibrationError(
                "motion-error sweep provenance must vary motion_error_weight"
            )
        motion_error_weight = _finite_float(
            _experimental_motion_error_weight, "motion_error_weight"
        )
        if motion_error_weight < 0.0:
            raise GraphCalibrationError("motion_error_weight must be nonnegative")
    if _experimental_candidate_top_k is None:
        candidate_top_k = 0
    else:
        sweep = _validated_experimental_sweep_provenance(
            _experimental_sweep_provenance
        )
        if "candidate_top_k" not in sweep["varied_fields"]:
            raise GraphCalibrationError(
                "top-k gate provenance must vary candidate_top_k"
            )
        candidate_top_k = _strict_int(
            _experimental_candidate_top_k, "candidate_top_k"
        )
        if candidate_top_k <= 0:
            raise GraphCalibrationError("candidate_top_k must be positive")
    if parity_mode not in {"sparse", "dense_diagnostic"}:
        raise GraphCalibrationError("parity_mode must be 'sparse' or 'dense_diagnostic'")
    if topology_mode not in {"legacy", "scored_partial"}:
        raise GraphCalibrationError("topology_mode must be 'legacy' or 'scored_partial'")
    if topology_mode == "scored_partial" and parity_mode != "sparse":
        raise GraphCalibrationError("scored_partial topology requires sparse parity mode")
    if topology_mode == "scored_partial" and (
        learned_edge_probabilities is None or source_no_link_probabilities is None
    ):
        raise GraphCalibrationError("scored_partial topology requires edge and no-link probabilities")
    normalized = normalize_nodes(nodes, max_nodes_per_frame=config.max_nodes_per_frame)
    node_by_id = {node.node_id: node for node in normalized}
    probabilities = _coerce_probability_map(learned_edge_probabilities, node_by_id)
    no_link_probabilities = _coerce_no_link_probability_map(
        source_no_link_probabilities, node_by_id
    )
    frames = _frame_groups(normalized)
    edges: list[GraphEdge] = []
    incoming: dict[int, int] = {}
    outgoing: dict[int, int] = {}
    stats: dict[str, int | bool] = {
        "adjacent_candidate_pairs": 0,
        "motion_tight_candidate_pairs": 0,
        "motion_relaxed_candidate_pairs": 0,
        "adjacent_edges": 0,
        "motion_tight_edges": 0,
        "motion_relaxed_edges": 0,
        "motion_tight_raw_peak_component_dim": 0,
        "motion_relaxed_raw_peak_component_dim": 0,
        "motion_relaxed_prepass_raw_peak_component_dim": 0,
        "sparse_component_count": 0,
        "peak_component_matrix_dim": 0,
        "learned_probability_count": len(probabilities),
        "learned_probabilities_present": learned_edge_probabilities is not None,
        "no_link_probability_count": len(no_link_probabilities),
        "scored_partial_components": 0,
        "scored_partial_fallback_components": 0,
        "scored_partial_candidate_pairs": 0,
        "scored_partial_zero_probability_pairs": 0,
        "scored_partial_edges": 0,
        "scored_partial_tight_edges": 0,
        "scored_partial_relaxed_edges": 0,
        "motion_error_weight": motion_error_weight,
        "candidate_top_k": candidate_top_k,
        "radius_candidate_pairs_diagnostic": 0,
    }

    frame_order = sorted(frames, reverse=config.motion_direction == "backward")
    for frame in frame_order:
        if frame + 1 not in frames:
            continue
        sources = frames[frame]
        targets = frames[frame + 1]
        radius_geometric = _radius_pairs(
            sources, targets, config.motion_relaxed_um, config.scale_zyx
        )
        stats["radius_candidate_pairs_diagnostic"] = int(
            stats["radius_candidate_pairs_diagnostic"]
        ) + len(radius_geometric)
        geometric = (
            _top_k_pairs(sources, targets, candidate_top_k, config.scale_zyx)
            if candidate_top_k
            else radius_geometric
        )
        stats["motion_relaxed_prepass_raw_peak_component_dim"] = max(
            int(stats["motion_relaxed_prepass_raw_peak_component_dim"]),
            _peak_sparse_component_dim(sources, targets, geometric),
        )
        stats["adjacent_candidate_pairs"] = int(stats["adjacent_candidate_pairs"]) + len(geometric)
        unmatched_sources = {node.node_id for node in sources}
        unmatched_targets = {node.node_id for node in targets}
        pair_info = {(source.node_id, target.node_id): (source, target, distance) for source, target, distance in geometric}

        def run_pass(
            label: str,
            candidates: list[tuple[int, int, float]],
        ) -> None:
            if not candidates:
                return
            result = sparse_min_cost_max_cardinality(
                unmatched_sources,
                unmatched_targets,
                candidates,
                max_component_matrix_dim=max_component_matrix_dim,
                parity_mode=parity_mode,
                dense_unmatched_cost=(
                    config.motion_tight_um
                    if label == "motion_tight"
                    else config.motion_relaxed_um
                )
                * 1000.0
                + 1.0,
            )
            _accumulate_match_stats(stats, result)  # type: ignore[arg-type]
            component_stat = f"{label}_raw_peak_component_dim"
            stats[component_stat] = max(
                int(stats[component_stat]), result.peak_component_matrix_dim
            )
            for match in result.matches:
                if match.source_id not in unmatched_sources or match.target_id not in unmatched_targets:
                    raise GraphInvariantError("sparse pass returned an already-consumed endpoint")
                source, target, distance = pair_info[(match.source_id, match.target_id)]
                probability = probabilities.get((source.node_id, target.node_id), 0.0)
                edges.append(
                    GraphEdge(
                        source.node_id,
                        target.node_id,
                        distance,
                        match.cost,
                        label,
                        probability if learned_edge_probabilities is not None else None,
                    )
                )
                unmatched_sources.remove(source.node_id)
                unmatched_targets.remove(target.node_id)
                incoming[target.node_id] = source.node_id
                outgoing[source.node_id] = target.node_id
                stats[f"{label}_edges"] = int(stats[f"{label}_edges"]) + 1

        if config.repair_single_parent:
            motion_rows: list[tuple[int, int, float, float, float]] = []
            for source, target, distance in geometric:
                child_id = outgoing.get(target.node_id)
                if child_id is None:
                    predicted = _coord(target)
                else:
                    child = node_by_id[child_id]
                    predicted = _coord(target) + config.velocity_weight * (
                        _coord(target) - _coord(child)
                    )
                motion_error = physical_distance(predicted, source, config.scale_zyx)
                probability = probabilities.get((source.node_id, target.node_id), 0.0)
                cost = _edge_cost(
                    distance,
                    probability,
                    config,
                    motion_error=motion_error,
                    motion_error_weight=motion_error_weight,
                )
                motion_rows.append(
                    (source.node_id, target.node_id, cost, motion_error, distance)
                )
            motion_by_pair = {
                (source, target): (cost, raw_distance)
                for source, target, cost, _motion_error, raw_distance in motion_rows
            }
            if topology_mode == "scored_partial":
                for component_sources, component_targets, component_pairs in _candidate_components(geometric):
                    complete = (
                        all(pair in probabilities for pair in component_pairs)
                        and all(source in no_link_probabilities for source in component_sources)
                    )
                    if not complete:
                        stats["scored_partial_fallback_components"] = int(
                            stats["scored_partial_fallback_components"]
                        ) + 1
                        continue
                    stats["scored_partial_components"] = int(stats["scored_partial_components"]) + 1
                    stats["scored_partial_candidate_pairs"] = int(
                        stats["scored_partial_candidate_pairs"]
                    ) + len(component_pairs)
                    positive_pairs: list[tuple[int, int, float]] = []
                    for source, target in component_pairs:
                        probability = probabilities[source, target]
                        if probability <= 0.0:
                            stats["scored_partial_zero_probability_pairs"] = int(
                                stats["scored_partial_zero_probability_pairs"]
                            ) + 1
                            continue
                        positive_pairs.append((source, target, -math.log(probability)))
                    dummy_costs = {
                        source: -math.log(max(no_link_probabilities[source], np.finfo(np.float64).tiny))
                        for source in component_sources
                    }
                    result = sparse_min_cost_partial_matching(
                        component_sources,
                        component_targets,
                        positive_pairs,
                        dummy_costs,
                    )
                    _accumulate_match_stats(stats, result)  # type: ignore[arg-type]
                    unmatched_sources.difference_update(component_sources)
                    unmatched_targets.difference_update(component_targets)
                    for match in result.matches:
                        source, target, distance = pair_info[(match.source_id, match.target_id)]
                        geometry_cost, raw_distance = motion_by_pair[match.source_id, match.target_id]
                        label = (
                            "scored_partial_tight"
                            if raw_distance <= config.motion_tight_um
                            else "scored_partial_relaxed"
                        )
                        edges.append(
                            GraphEdge(
                                source.node_id,
                                target.node_id,
                                distance,
                                geometry_cost,
                                label,
                                probabilities[match.source_id, match.target_id],
                            )
                        )
                        incoming[target.node_id] = source.node_id
                        outgoing[source.node_id] = target.node_id
                        stats["scored_partial_edges"] = int(stats["scored_partial_edges"]) + 1
                        if label == "scored_partial_tight":
                            stats["scored_partial_tight_edges"] = int(
                                stats["scored_partial_tight_edges"]
                            ) + 1
                        else:
                            stats["scored_partial_relaxed_edges"] = int(
                                stats["scored_partial_relaxed_edges"]
                            ) + 1
            tight = [
                (source, target, cost)
                for source, target, cost, _, raw_distance in motion_rows
                if raw_distance <= config.motion_tight_um
                and source in unmatched_sources
                and target in unmatched_targets
            ]
            stats["motion_tight_candidate_pairs"] = int(stats["motion_tight_candidate_pairs"]) + len(tight)
            run_pass("motion_tight", tight)
            relaxed = [
                (source, target, cost)
                for source, target, cost, _, _ in motion_rows
                if source in unmatched_sources
                and target in unmatched_targets
            ]
            stats["motion_relaxed_candidate_pairs"] = int(stats["motion_relaxed_candidate_pairs"]) + len(relaxed)
            run_pass("motion_relaxed", relaxed)

    result_edges = tuple(sorted(edges, key=lambda edge: (node_by_id[edge.source_id].t, edge.source_id, edge.target_id)))
    stats["adjacent_edges"] = len(result_edges)
    validate_graph(normalized, result_edges)
    return result_edges, stats


def local_spacing_um(
    nodes: Iterable[NodeLike],
    *,
    neighbors: int = V45_GRAPH_CONFIG.spacing_neighbors,
    scale_zyx: Sequence[float] = PHYSICAL_SCALE_ZYX,
) -> dict[int, float]:
    """Median distance to up to three same-frame nearest neighbours."""

    normalized = normalize_nodes(nodes)
    count = _strict_int(neighbors, "neighbors")
    if count <= 0 or count > V45_GRAPH_CONFIG.spacing_neighbors:
        raise GraphCalibrationError("neighbors must be between one and the V45 cap")
    result: dict[int, float] = {}
    for frame_nodes in _frame_groups(normalized).values():
        if len(frame_nodes) <= 1:
            result[frame_nodes[0].node_id] = V45_GRAPH_CONFIG.spacing_ref_um
            continue
        physical = _physical_array(frame_nodes, scale_zyx)
        tree = cKDTree(physical)
        k = min(len(frame_nodes), count + 1)
        distances, indices = tree.query(physical, k=k)
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices)
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        for row, node in enumerate(frame_nodes):
            neighbor_distances = [
                float(distance)
                for distance, index in zip(distances[row], indices[row], strict=True)
                if int(index) != row and math.isfinite(float(distance))
            ][:count]
            result[node.node_id] = (
                float(np.median(np.asarray(neighbor_distances, dtype=np.float64)))
                if neighbor_distances
                else math.nan
            )
    return result


def gap_step_limit_um(
    source_spacing_um: float,
    target_spacing_um: float,
    *,
    config: V45GraphConfig = V45_GRAPH_CONFIG,
) -> tuple[float, float]:
    """Return the density-adjusted per-step gap limit and applied delta."""

    finite = [value for value in (source_spacing_um, target_spacing_um) if math.isfinite(value)]
    spacing = float(np.median(finite)) if finite else config.spacing_ref_um
    delta = float(
        np.clip(
            (spacing - config.spacing_ref_um) * config.density_gain,
            -config.density_delta_clip_um,
            config.density_delta_clip_um,
        )
    )
    return config.gap_step_um + delta, delta


def intensity_centroid_refine(
    image: np.ndarray,
    coord_zyx: Sequence[float],
    *,
    scale_zyx: Sequence[float] = PHYSICAL_SCALE_ZYX,
    z_radius: int = V45_GRAPH_CONFIG.refine_z_radius,
    yx_radius: int = V45_GRAPH_CONFIG.refine_yx_radius,
    percentile: float = V45_GRAPH_CONFIG.refine_percentile,
    max_shift_um: float = V45_GRAPH_CONFIG.refine_max_shift_um,
) -> tuple[float, float, float]:
    """Pure p20-subtracted local intensity-centroid refinement.

    A nonfinite/flat patch, empty clipped window, or shift beyond 3.2 um leaves
    the input coordinate unchanged.
    """

    volume = np.asarray(image)
    if volume.ndim != 3:
        raise GraphCalibrationError("refinement image must be a 3-D z/y/x array")
    original = _coord(coord_zyx)
    center = np.rint(original).astype(np.int64)
    radii = np.asarray((z_radius, yx_radius, yx_radius), dtype=np.int64)
    starts = np.maximum(0, center - radii)
    stops = np.minimum(np.asarray(volume.shape, dtype=np.int64), center + radii + 1)
    if np.any(stops <= starts):
        return tuple(float(value) for value in original)
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))
    patch = np.asarray(volume[slices], dtype=np.float64)
    finite = np.isfinite(patch)
    if not np.any(finite):
        return tuple(float(value) for value in original)
    background = float(np.percentile(patch[finite], percentile))
    weights = np.where(finite, np.maximum(patch - background, 0.0), 0.0)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        return tuple(float(value) for value in original)
    grids = np.indices(patch.shape, dtype=np.float64)
    refined = np.asarray(
        [starts[axis] + float((weights * grids[axis]).sum()) / total for axis in range(3)],
        dtype=np.float64,
    )
    shift = physical_distance(original, refined, scale_zyx)
    if shift > max_shift_um + 1.0e-12:
        return tuple(float(value) for value in original)
    return tuple(float(value) for value in refined)


refine_intensity_centroid = intensity_centroid_refine


def _image_for_frame(images: ImageSource, frame: int) -> np.ndarray:
    try:
        image = images(frame) if callable(images) else images[frame]
    except (KeyError, IndexError, TypeError) as exc:
        raise GraphCalibrationError(f"no refinement image for frame {frame}") from exc
    return np.asarray(image)


def close_one_frame_gaps(
    nodes: Iterable[NodeLike],
    edges: Iterable[GraphEdge],
    *,
    images: ImageSource | None = None,
    image_refine_callback: RefineCallback | None = None,
    max_component_matrix_dim: int | None = None,
    parity_mode: str = "sparse",
    config: V45GraphConfig = V45_GRAPH_CONFIG,
) -> tuple[tuple[GraphNode, ...], tuple[GraphEdge, ...], dict[str, int]]:
    """Close exactly one missing frame by midpoint reuse or bounded synthesis."""

    if parity_mode not in {"sparse", "dense_diagnostic"}:
        raise GraphCalibrationError("parity_mode must be 'sparse' or 'dense_diagnostic'")
    normalized = normalize_nodes(nodes, max_nodes_per_frame=config.max_nodes_per_frame)
    current_edges = list(edges)
    validate_graph(normalized, current_edges)
    node_by_id = {node.node_id: node for node in normalized}
    synth_cap = (
        min(
            config.gap_add_absolute,
            max(1, int(round(len(normalized) * config.gap_add_fraction))),
        )
        if config.gap_add_fraction > 0
        else 0
    )
    stats = {
        "gap_candidate_pairs": 0,
        "gap_density_rejected": 0,
        "gap_matches": 0,
        "gap_reused_nodes": 0,
        "gap_synthesized_nodes": 0,
        "gap_synthesis_cap": synth_cap,
        "gap_refinement_accepted": 0,
        "gap_refinement_rejected": 0,
        "gap_frame_cap_rejected": 0,
        "sparse_component_count": 0,
        "peak_component_matrix_dim": 0,
    }
    if not current_edges:
        return normalized, tuple(), stats

    frames = _frame_groups(normalized)
    incoming = {edge.target_id: edge.source_id for edge in current_edges}
    outgoing: dict[int, list[int]] = defaultdict(list)
    for edge in current_edges:
        outgoing[edge.source_id].append(edge.target_id)
    spacing = local_spacing_um(
        normalized, neighbors=config.spacing_neighbors, scale_zyx=config.scale_zyx
    )
    next_node_id = max(node_by_id, default=-1) + 1
    used_midpoints: set[int] = set()
    added_nodes: list[GraphNode] = []
    added_edges: list[GraphEdge] = []

    for frame in sorted(frames):
        if frame + 2 not in frames:
            continue
        sources = tuple(node for node in frames[frame] if not outgoing.get(node.node_id))
        targets = tuple(node for node in frames[frame + 2] if node.node_id not in incoming)
        if not sources or not targets:
            continue
        candidate_radius_um = 2.0 * (
            config.gap_step_um + config.density_delta_clip_um
        )
        radius_rows = _radius_pairs(
            sources, targets, candidate_radius_um, config.scale_zyx
        )
        candidates: list[tuple[int, int, float]] = []
        for source, target, total_distance in radius_rows:
            step_limit, _ = gap_step_limit_um(spacing[source.node_id], spacing[target.node_id], config=config)
            if total_distance / 2.0 > step_limit + 1.0e-12:
                stats["gap_density_rejected"] += 1
                continue
            candidates.append((source.node_id, target.node_id, total_distance))
        stats["gap_candidate_pairs"] += len(candidates)
        match_result = sparse_min_cost_max_cardinality(
            (node.node_id for node in sources),
            (node.node_id for node in targets),
            candidates,
            max_component_matrix_dim=max_component_matrix_dim,
            parity_mode=parity_mode,
            dense_unmatched_cost=candidate_radius_um * 1000.0 + 1.0,
        )
        _accumulate_match_stats(stats, match_result)
        middle_nodes = frames.get(frame + 1, ())
        middle_physical = _physical_array(middle_nodes, config.scale_zyx) if middle_nodes else np.empty((0, 3))
        middle_tree = cKDTree(middle_physical) if middle_nodes else None
        for match in match_result.matches:
            source = node_by_id[match.source_id]
            target = node_by_id[match.target_id]
            midpoint = (_coord(source) + _coord(target)) / 2.0
            reused: GraphNode | None = None
            if middle_tree is not None:
                midpoint_physical = midpoint * np.asarray(config.scale_zyx, dtype=np.float64)
                indices = middle_tree.query_ball_point(
                    midpoint_physical, r=config.midpoint_reuse_um, return_sorted=True
                )
                ranked = sorted(
                    (physical_distance(midpoint, middle_nodes[int(index)], config.scale_zyx), middle_nodes[int(index)].node_id, middle_nodes[int(index)])
                    for index in indices
                )
                for _, candidate_id, candidate in ranked:
                    if candidate_id in used_midpoints or candidate_id in incoming or outgoing.get(candidate_id):
                        continue
                    reused = candidate
                    break
            if reused is not None:
                middle = reused
                used_midpoints.add(middle.node_id)
                stats["gap_reused_nodes"] += 1
            else:
                if stats["gap_synthesized_nodes"] >= synth_cap:
                    continue
                refined = tuple(float(value) for value in midpoint)
                proposal: Sequence[float] | None = None
                if image_refine_callback is not None:
                    proposal = image_refine_callback(frame + 1, refined)
                elif images is not None:
                    proposal = intensity_centroid_refine(
                        _image_for_frame(images, frame + 1),
                        refined,
                        scale_zyx=config.scale_zyx,
                    )
                if proposal is not None:
                    candidate_coord = _coord(proposal)
                    if (
                        physical_distance(midpoint, candidate_coord, config.scale_zyx)
                        <= config.refine_max_shift_um + 1.0e-12
                        and np.all(candidate_coord >= 0)
                    ):
                        refined = tuple(float(value) for value in candidate_coord)
                        stats["gap_refinement_accepted"] += 1
                    else:
                        stats["gap_refinement_rejected"] += 1
                middle = GraphNode(
                    node_id=next_node_id,
                    t=frame + 1,
                    z=refined[0],
                    y=refined[1],
                    x=refined[2],
                    score=(float(source.score) + float(target.score)) / 2.0,
                    synthetic=True,
                )
                next_node_id += 1
                added_nodes.append(middle)
                node_by_id[middle.node_id] = middle
                stats["gap_synthesized_nodes"] += 1
            first_distance = physical_distance(source, middle, config.scale_zyx)
            second_distance = physical_distance(middle, target, config.scale_zyx)
            kind = "gap_reuse" if reused is not None else "gap_synth"
            added_edges.extend(
                (
                    GraphEdge(
                        source.node_id,
                        middle.node_id,
                        first_distance,
                        config.raw_distance_weight * first_distance,
                        kind,
                        None,
                    ),
                    GraphEdge(
                        middle.node_id,
                        target.node_id,
                        second_distance,
                        config.raw_distance_weight * second_distance,
                        kind,
                        None,
                    ),
                )
            )
            outgoing[source.node_id].append(middle.node_id)
            outgoing[middle.node_id].append(target.node_id)
            incoming[middle.node_id] = source.node_id
            incoming[target.node_id] = middle.node_id
            stats["gap_matches"] += 1

    result_nodes = tuple(sorted((*normalized, *added_nodes), key=lambda node: (node.t, node.node_id)))
    result_edges = tuple(
        sorted((*current_edges, *added_edges), key=lambda edge: (node_by_id[edge.source_id].t, edge.source_id, edge.target_id))
    )
    validate_graph(result_nodes, result_edges)
    return result_nodes, result_edges, stats


def add_safe_divisions(
    nodes: Iterable[NodeLike],
    edges: Iterable[GraphEdge],
    *,
    config: V45GraphConfig = V45_GRAPH_CONFIG,
    lifetime_prior: ErlangDivisionPrior | None = None,
) -> tuple[tuple[GraphEdge, ...], dict[str, int | float | bool]]:
    """Add a conservatively capped second child to safe one-child parents."""

    normalized = normalize_nodes(nodes, max_nodes_per_frame=config.max_nodes_per_frame)
    current_edges = list(edges)
    validate_graph(normalized, current_edges)
    node_by_id = {node.node_id: node for node in normalized}
    frames = _frame_groups(normalized)
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, int] = {}
    outdegree: dict[int, int] = defaultdict(int)
    indegree: dict[int, int] = defaultdict(int)
    for edge in current_edges:
        outgoing[edge.source_id].append(edge.target_id)
        incoming[edge.target_id] = edge.source_id
        outdegree[edge.source_id] += 1
        indegree[edge.target_id] += 1
    existing_edges = {
        (edge.source_id, edge.target_id) for edge in current_edges
    }
    global_cap = max(
        1,
        int(
            round(
                max(1, len(current_edges)) * config.division_global_fraction
            )
        ),
    )
    stats = {
        "division_candidate_count": 0,
        "division_geometry_rejected": 0,
        "division_added": 0,
        "division_frame_cap_hits": 0,
        "division_parent_cap_rejected": 0,
        "division_global_cap": global_cap,
    }
    selected: list[GraphEdge] = []
    used_targets: set[int] = set()

    def division_age(parent_id: int) -> int:
        """Trace the current lineage to its root or most recent division."""

        current_id = parent_id
        while current_id in incoming:
            predecessor_id = incoming[current_id]
            if outdegree[predecessor_id] == 2:
                break
            current_id = predecessor_id
        elapsed = node_by_id[parent_id].t - node_by_id[current_id].t
        if elapsed < 0:
            raise GraphInvariantError("division lineage birth occurs after its parent")
        return int(elapsed)

    lifetime_cost_min = math.inf
    lifetime_cost_max = -math.inf
    lifetime_zero_rejected = 0
    for frame in sorted(frames):
        if frame + 1 not in frames:
            continue
        source_ids = tuple(
            node.node_id
            for node in frames[frame]
            if outdegree[node.node_id] == 1
        )
        child_frame = frames[frame + 1]
        candidate_children = tuple(
            child
            for child in child_frame
            if indegree[child.node_id] < 1 and child.node_id not in used_targets
        )
        if not source_ids or not candidate_children:
            continue
        frame_cap = max(
            1, int(round(len(source_ids) * config.division_frame_fraction))
        )
        orphan_tree = cKDTree(
            _physical_array(candidate_children, config.scale_zyx)
        )
        candidates: list[tuple[float, int, int, float]] = []
        for parent_id in source_ids:
            parent = node_by_id[parent_id]
            children = outgoing[parent_id]
            existing = node_by_id.get(children[0])
            if existing is None or existing.t != frame + 1:
                continue
            existing_distance = physical_distance(parent, existing, config.scale_zyx)
            if existing_distance > config.division_existing_child_um + 1.0e-12:
                stats["division_geometry_rejected"] += 1
                continue
            parent_physical = _coord(parent) * np.asarray(config.scale_zyx, dtype=np.float64)
            for index in orphan_tree.query_ball_point(
                parent_physical, r=config.division_candidate_child_um, return_sorted=True
            ):
                candidate = candidate_children[int(index)]
                if (parent_id, candidate.node_id) in existing_edges:
                    continue
                parent_distance = physical_distance(parent, candidate, config.scale_zyx)
                sister_distance = physical_distance(existing, candidate, config.scale_zyx)
                if sister_distance > config.division_sister_um + 1.0e-12:
                    stats["division_geometry_rejected"] += 1
                    continue
                lifetime_cost = 0.0
                if lifetime_prior is not None:
                    lifetime_cost = lifetime_prior.negative_log_density(
                        division_age(parent_id)
                    )
                    if not math.isfinite(lifetime_cost):
                        lifetime_zero_rejected += 1
                        continue
                    lifetime_cost_min = min(lifetime_cost_min, lifetime_cost)
                    lifetime_cost_max = max(lifetime_cost_max, lifetime_cost)
                score = (
                    parent_distance
                    + config.division_sister_weight * sister_distance
                    + lifetime_cost
                )
                candidates.append((score, parent_id, candidate.node_id, parent_distance))
        stats["division_candidate_count"] += len(candidates)
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        added_this_frame = 0
        for score, parent_id, candidate_id, parent_distance in candidates:
            if stats["division_added"] >= global_cap:
                stats["division_frame_cap_hits"] += 1
                break
            if added_this_frame >= frame_cap:
                stats["division_frame_cap_hits"] += 1
                break
            if outdegree[parent_id] >= 2:
                stats["division_parent_cap_rejected"] += 1
                continue
            if indegree[candidate_id] >= 1 or candidate_id in used_targets:
                continue
            edge = GraphEdge(
                parent_id,
                candidate_id,
                parent_distance,
                score,
                "division",
                None,
            )
            selected.append(edge)
            outgoing[parent_id].append(candidate_id)
            incoming[candidate_id] = parent_id
            outdegree[parent_id] += 1
            indegree[candidate_id] += 1
            used_targets.add(candidate_id)
            added_this_frame += 1
            stats["division_added"] += 1

    stats.update(
        {
            "division_lifetime_prior_enabled": lifetime_prior is not None,
            "division_lifetime_shape": 0 if lifetime_prior is None else lifetime_prior.shape,
            "division_lifetime_rate": 0.0 if lifetime_prior is None else lifetime_prior.rate,
            "division_lifetime_zero_rejected": lifetime_zero_rejected,
            "division_lifetime_cost_min": (
                0.0 if lifetime_cost_min == math.inf else lifetime_cost_min
            ),
            "division_lifetime_cost_max": (
                0.0 if lifetime_cost_max == -math.inf else lifetime_cost_max
            ),
        }
    )

    result = tuple(
        sorted((*current_edges, *selected), key=lambda edge: (node_by_id[edge.source_id].t, edge.source_id, edge.target_id))
    )
    validate_graph(normalized, result)
    return result, stats


def validate_graph(nodes: Iterable[NodeLike], edges: Iterable[GraphEdge]) -> None:
    """Fail closed unless every edge is next-frame and lineage-safe."""

    normalized = normalize_nodes(nodes)
    node_by_id = {node.node_id: node for node in normalized}
    indegree: dict[int, int] = defaultdict(int)
    outdegree: dict[int, int] = defaultdict(int)
    seen: set[tuple[int, int]] = set()
    for edge in edges:
        if not isinstance(edge, GraphEdge):
            raise GraphInvariantError("edges must be GraphEdge records")
        try:
            source_id = _strict_int(edge.source_id, "edge source_id")
            target_id = _strict_int(edge.target_id, "edge target_id")
        except GraphCalibrationError as exc:
            raise GraphInvariantError(str(exc)) from exc
        if source_id != edge.source_id or target_id != edge.target_id:
            raise GraphInvariantError("edge IDs must be exact integers")
        if not isinstance(edge.kind, str) or not edge.kind:
            raise GraphInvariantError("edge kind must be a non-empty string")
        if edge.learned_probability is not None:
            probability = float(edge.learned_probability)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise GraphInvariantError("edge learned probability must be in [0, 1]")
        if edge.source_id not in node_by_id or edge.target_id not in node_by_id:
            raise GraphInvariantError("edge references an unknown node")
        pair = (edge.source_id, edge.target_id)
        if pair in seen:
            raise GraphInvariantError("duplicate edge")
        seen.add(pair)
        if node_by_id[edge.target_id].t != node_by_id[edge.source_id].t + 1:
            raise GraphInvariantError("edges must be forward and next-frame")
        if not math.isfinite(float(edge.distance_um)) or edge.distance_um < 0:
            raise GraphInvariantError("edge distance must be finite and nonnegative")
        if not math.isfinite(float(edge.cost)):
            raise GraphInvariantError("edge cost must be finite")
        indegree[edge.target_id] += 1
        outdegree[edge.source_id] += 1
        if indegree[edge.target_id] > 1:
            raise GraphInvariantError("merge detected: indegree exceeds one")
        if outdegree[edge.source_id] > 2:
            raise GraphInvariantError("outdegree exceeds two")


def _graph_components(
    node_ids: set[int],
    edges: Sequence[GraphEdge],
) -> list[set[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)
    components: list[set[int]] = []
    seen: set[int] = set()
    for start in sorted(node_ids):
        if start in seen:
            continue
        component: set[int] = set()
        queue = deque([start])
        while queue:
            node_id = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            component.add(node_id)
            queue.extend(sorted(neighbors[node_id] - seen))
        components.append(component)
    return components


def prune_graph(
    nodes: Iterable[NodeLike],
    edges: Iterable[GraphEdge],
    *,
    config: V45GraphConfig = V45_GRAPH_CONFIG,
) -> tuple[tuple[GraphNode, ...], tuple[GraphEdge, ...], dict[str, int]]:
    """Prune isolated nodes and short non-division components."""

    normalized = normalize_nodes(nodes)
    current_edges = tuple(edges)
    validate_graph(normalized, current_edges)
    node_by_id = {node.node_id: node for node in normalized}
    if not current_edges:
        return normalized, current_edges, {
            "isolated_nodes_pruned": 0,
            "short_component_nodes_pruned": 0,
            "short_components_pruned": 0,
            "short_component_filter_skipped_all": 0,
        }
    degree: dict[int, int] = defaultdict(int)
    outdegree: dict[int, int] = defaultdict(int)
    for edge in current_edges:
        degree[edge.source_id] += 1
        degree[edge.target_id] += 1
        outdegree[edge.source_id] += 1
    isolated = {node.node_id for node in normalized if degree[node.node_id] == 0}
    base_keep = set(node_by_id) - isolated if config.prune_isolated else set(node_by_id)
    retained_edges = tuple(
        edge
        for edge in current_edges
        if edge.source_id in base_keep and edge.target_id in base_keep
    )
    components = _graph_components(base_keep, retained_edges)
    short_keep: set[int] = set()
    for component in components:
        has_division = any(outdegree[node_id] == 2 for node_id in component)
        if len(component) >= config.min_component_nodes or (
            config.preserve_division_components and has_division
        ):
            short_keep.update(component)
    skipped_all = bool(base_keep and not short_keep)
    if skipped_all:
        keep = base_keep
        short_removed: set[int] = set()
    else:
        keep = short_keep
        short_removed = base_keep - keep
    result_nodes = tuple(node for node in normalized if node.node_id in keep)
    result_edges = tuple(
        edge for edge in retained_edges if edge.source_id in keep and edge.target_id in keep
    )
    stats = {
        "isolated_nodes_pruned": len(isolated),
        "short_component_nodes_pruned": len(short_removed),
        "short_components_pruned": sum(
            1
            for component in components
            if component and component <= short_removed
        ),
        "short_component_filter_skipped_all": int(skipped_all),
    }
    validate_graph(result_nodes, result_edges)
    return result_nodes, result_edges, stats


def line_fit_smooth(
    nodes: Iterable[NodeLike],
    edges: Iterable[GraphEdge],
    *,
    config: V45GraphConfig = V45_GRAPH_CONFIG,
) -> tuple[tuple[GraphNode, ...], tuple[GraphEdge, ...], dict[str, int]]:
    """Apply the notebook's floating window-2 local line-fit blend."""

    normalized = normalize_nodes(nodes)
    current_edges = tuple(edges)
    validate_graph(normalized, current_edges)
    node_by_id = {node.node_id: node for node in normalized}
    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for edge in current_edges:
        incoming[edge.target_id].append(edge.source_id)
        outgoing[edge.source_id].append(edge.target_id)

    def path_window(center_id: int) -> list[tuple[int, int]]:
        neighborhood: list[tuple[int, int]] = [(0, center_id)]
        current = center_id
        for step in range(1, config.smooth_window + 1):
            parents = incoming.get(current, [])
            if len(parents) != 1:
                break
            current = parents[0]
            if current not in node_by_id:
                break
            neighborhood.append((-step, current))
        current = center_id
        for step in range(1, config.smooth_window + 1):
            children = outgoing.get(current, [])
            if len(children) != 1:
                break
            current = children[0]
            if current not in node_by_id:
                break
            neighborhood.append((step, current))
        return neighborhood

    original_pos = {
        node.node_id: _coord(node) for node in normalized
    }
    updated: dict[int, np.ndarray] = {}
    for node in normalized:
        neighborhood = path_window(node.node_id)
        if len(neighborhood) < 3:
            continue
        deltas = np.asarray(
            [delta for delta, _ in neighborhood], dtype=np.float64
        )
        coordinates = np.stack(
            [original_pos[node_id] for _, node_id in neighborhood]
        )
        fitted = np.asarray(
            [
                np.polyval(np.polyfit(deltas, coordinates[:, axis], 1), 0.0)
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(fitted)):
            continue
        updated[node.node_id] = (
            (1.0 - config.smooth_weight) * original_pos[node.node_id]
            + config.smooth_weight * fitted
        )

    result_nodes: list[GraphNode] = []
    for node in normalized:
        position = updated.get(node.node_id, original_pos[node.node_id])
        result_nodes.append(
            GraphNode(
                node.node_id,
                node.t,
                float(position[0]),
                float(position[1]),
                float(position[2]),
                node.score,
                node.synthetic,
            )
        )
    result_nodes_tuple = tuple(sorted(result_nodes, key=lambda item: (item.t, item.node_id)))
    result_edges = tuple(
        sorted(
            current_edges,
            key=lambda edge: (
                node_by_id[edge.source_id].t,
                edge.source_id,
                edge.target_id,
            ),
        )
    )
    validate_graph(result_nodes_tuple, result_edges)
    return result_nodes_tuple, result_edges, {"smoothed_nodes": len(updated)}


def postprocess_graph(
    nodes: Iterable[NodeLike],
    edges: Iterable[GraphEdge],
    *,
    config: V45GraphConfig = V45_GRAPH_CONFIG,
) -> tuple[tuple[GraphNode, ...], tuple[GraphEdge, ...], dict[str, int]]:
    """Apply exact V45 component pruning and floating-position smoothing."""

    pruned_nodes, pruned_edges, prune_stats = prune_graph(nodes, edges, config=config)
    smooth_nodes, smooth_edges, smooth_stats = line_fit_smooth(
        pruned_nodes, pruned_edges, config=config
    )
    return smooth_nodes, smooth_edges, {**prune_stats, **smooth_stats}


def calibrate_v45_graph(
    nodes: Iterable[NodeLike],
    learned_edge_probabilities: Mapping[Any, Any] | Iterable[ProbabilityLike] | None = None,
    source_no_link_probabilities: Mapping[Any, Any] | None = None,
    *,
    images: ImageSource | None = None,
    image_refine_callback: RefineCallback | None = None,
    max_component_matrix_dim: int | None = None,
    parity_mode: str = "sparse",
    topology_mode: str = "legacy",
    division_enabled: bool = False,
    config: V45GraphConfig = V45_GRAPH_CONFIG,
    _experimental_sweep_provenance: Mapping[str, Any] | None = None,
    _experimental_motion_error_weight: float | None = None,
    _experimental_candidate_top_k: int | None = None,
) -> GraphResult:
    """Run V45 calibration, sparse by default with an explicit dense oracle."""

    if type(division_enabled) is not bool:
        raise GraphCalibrationError("division_enabled must be bool")
    sweep_provenance = None
    if config is not V45_GRAPH_CONFIG and config != V45_GRAPH_CONFIG:
        sweep_provenance = _validated_experimental_sweep_provenance(
            _experimental_sweep_provenance
        )
    elif _experimental_sweep_provenance is not None:
        sweep_provenance = _validated_experimental_sweep_provenance(
            _experimental_sweep_provenance
        )

    normalized = normalize_nodes(nodes, max_nodes_per_frame=config.max_nodes_per_frame)
    adjacent_edges, adjacent_stats = build_adjacent_edges(
        normalized,
        learned_edge_probabilities,
        source_no_link_probabilities,
        max_component_matrix_dim=max_component_matrix_dim,
        parity_mode=parity_mode,
        topology_mode=topology_mode,
        config=config,
        _experimental_sweep_provenance=sweep_provenance,
        _experimental_motion_error_weight=_experimental_motion_error_weight,
        _experimental_candidate_top_k=_experimental_candidate_top_k,
    )
    gap_nodes, gap_edges, gap_stats = close_one_frame_gaps(
        normalized,
        adjacent_edges,
        images=images,
        image_refine_callback=image_refine_callback,
        max_component_matrix_dim=max_component_matrix_dim,
        parity_mode=parity_mode,
        config=config,
    )
    if division_enabled:
        division_edges, division_stats = add_safe_divisions(gap_nodes, gap_edges, config=config)
    else:
        division_edges = gap_edges
        division_stats = {
            "division_candidate_count": 0,
            "division_geometry_rejected": 0,
            "division_added": 0,
            "division_frame_cap_hits": 0,
            "division_parent_cap_rejected": 0,
            "division_global_cap": 0,
        }
    final_nodes, final_edges, post_stats = postprocess_graph(
        gap_nodes, division_edges, config=config
    )
    peak = max(
        int(adjacent_stats["peak_component_matrix_dim"]),
        int(gap_stats["peak_component_matrix_dim"]),
    )
    stats: dict[str, int | float | bool] = {
        "input_node_count": len(normalized),
        "output_node_count": len(final_nodes),
        "output_edge_count": len(final_edges),
        **adjacent_stats,
        **{key: value for key, value in gap_stats.items() if key not in {"sparse_component_count", "peak_component_matrix_dim"}},
        **division_stats,
        **post_stats,
        "sparse_component_count": int(adjacent_stats["sparse_component_count"])
        + int(gap_stats["sparse_component_count"]),
        "peak_component_matrix_dim": peak,
    }
    if image_refine_callback is not None:
        refinement_mode = "callback"
    elif images is not None:
        refinement_mode = "p20-intensity-centroid"
    else:
        refinement_mode = "none"
    provenance = {
        "schema": GRAPH_SCHEMA,
        "calibration": "sparse-semantic-rewrite-of-exact-scored-public-v45",
        "source_notebook": "reports/g02/v45.ipynb",
        "source_script_version_id": 335810749,
        "scored_account_submission_id": 54633411,
        "scored_account_public_score": 0.902,
        "scored_account_submission_sha256": "8c73d776abca799a37c2bd24768a1bbb510418c3327192430730540d56cea698",
        "public_replay_displayed_score": 0.903,
        "public_replay_submission_sha256": "6f8d81f6e5d9eaa066e47f0cdd2d22df48eaa74853d2940d0bc957f3ef3a19e1",
        "deterministic": True,
        "config": config.as_dict(),
        "learned_probabilities": "provided" if learned_edge_probabilities is not None else "absent-zero-bonus",
        "source_no_link_probabilities": (
            "provided" if source_no_link_probabilities is not None else "absent"
        ),
        "assignment": (
            "full-frame-dense-lapjv-diagnostic-oracle"
            if parity_mode == "dense_diagnostic"
            else "component-local-sparse-partial-probability-matching"
            if topology_mode == "scored_partial"
            else "cKDTree-feasible/component-local-sparse-augmented-min-cost-max-cardinality"
        ),
        "full_frame_dense_matrix": parity_mode == "dense_diagnostic",
        "parity_mode": parity_mode,
        "topology_mode": topology_mode,
        "division_enabled": division_enabled,
        "motion_relink_raw_gates_um": [
            config.motion_tight_um,
            config.motion_relaxed_um,
        ],
        "motion_error_weight": (
            config.motion_error_weight
            if _experimental_motion_error_weight is None
            else float(_experimental_motion_error_weight)
        ),
        "motion_direction": config.motion_direction,
        "candidate_top_k": (
            0
            if _experimental_candidate_top_k is None
            else int(_experimental_candidate_top_k)
        ),
        "output_edge_validation_um": config.output_edge_max_um,
        "output_edge_validation_is_not_a_relink_pass": True,
        "image_refinement": refinement_mode,
        "gap2_enabled": False,
        "deepcenter_enabled": False,
        "adaptive_short_track_rescue": False,
        "outside_090_gate": False,
        "external_access": False,
        "production_eligible": sweep_provenance is None,
    }
    if sweep_provenance is not None:
        provenance["experimental_sweep"] = sweep_provenance
    validate_graph(final_nodes, final_edges)
    return GraphResult(final_nodes, final_edges, stats, provenance)


calibrate_graph = calibrate_v45_graph
build_graph = calibrate_v45_graph


__all__ = [
    "EXPERIMENTAL_SWEEP_SCHEMA",
    "GRAPH_SCHEMA",
    "PHYSICAL_SCALE_ZYX",
    "GraphCalibrationError",
    "GraphEdge",
    "ErlangDivisionPrior",
    "GraphInvariantError",
    "GraphNode",
    "GraphResult",
    "LearnedEdgeProbability",
    "SparseAssignmentError",
    "SparseMatch",
    "SparseMatchResult",
    "V45_CONFIG",
    "V45_GRAPH_CONFIG",
    "V45GraphConfig",
    "add_safe_divisions",
    "build_adjacent_edges",
    "build_graph",
    "calibrate_graph",
    "calibrate_v45_graph",
    "close_one_frame_gaps",
    "distance_um",
    "gap_step_limit_um",
    "intensity_centroid_refine",
    "line_fit_smooth",
    "local_spacing_um",
    "learned_edge_score_bonus",
    "motion_only_edge_cost",
    "normalize_nodes",
    "physical_distance",
    "postprocess_graph",
    "prune_graph",
    "refine_intensity_centroid",
    "sparse_bipartite_match",
    "sparse_min_cost_max_cardinality",
    "sparse_min_cost_partial_matching",
    "validate_graph",
]
