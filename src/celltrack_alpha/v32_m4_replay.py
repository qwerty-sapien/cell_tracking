"""Deterministic, outcome-free replay of V32 motion relinking with M4.

The module intentionally accepts only in-memory node and post-ILP edge records.
It does not know about replay caches, rankers, scorers, or campaign datasets.
M4 is the fixed forward-acceleration bonus layered onto the otherwise current
V32 two-pass Hungarian motion relinker.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
MAX_ACCEL_UM = 4.0
MAX_BONUS = 0.20
NEXT_STEP_GATE_UM = 10.0

_TIGHT_UM = 6.0
_RELAXED_UM = 10.0
_VELOCITY_WEIGHT = 0.5
_LEARNED_BONUS = 1.0
_RAW_DISTANCE_WEIGHT = 0.05
_MAX_FRAME_NODES = 2600
_OUTPUT_EDGE_MAX_UM = 14.0

_CACHE_NODE_DTYPE = np.dtype(
    [
        ("ordinal", "<i8"),
        ("node_id", "<i8"),
        ("t", "<i4"),
        ("z_vox", "<f8"),
        ("y_vox", "<f8"),
        ("x_vox", "<f8"),
        ("z_um", "<f8"),
        ("y_um", "<f8"),
        ("x_um", "<f8"),
    ],
    align=False,
)
_CACHE_POST_EDGE_DTYPE = np.dtype(
    [
        ("ordinal", "<i8"),
        ("source_id", "<i8"),
        ("target_id", "<i8"),
        ("edge_prob", "<f8"),
    ],
    align=False,
)


class MotionRelinkError(ValueError):
    """The standalone V32 M4 replay contract was violated."""


@dataclass(frozen=True, slots=True)
class MotionRelinkConfig:
    """The fully frozen V32/M4 scientific configuration.

    Fields remain explicit so serialized configurations bind every scientific
    value.  Any value other than the preregistered default is rejected; this
    object is a contract, not a sweep surface.
    """

    tight_um: float = _TIGHT_UM
    relaxed_um: float = _RELAXED_UM
    velocity_weight: float = _VELOCITY_WEIGHT
    learned_bonus: float = _LEARNED_BONUS
    raw_distance_weight: float = _RAW_DISTANCE_WEIGHT
    output_edge_max_um: float = _OUTPUT_EDGE_MAX_UM
    max_frame_nodes: int = _MAX_FRAME_NODES
    max_accel_um: float = MAX_ACCEL_UM
    max_bonus: float = MAX_BONUS
    next_step_gate_um: float = NEXT_STEP_GATE_UM

    def __post_init__(self) -> None:
        numeric = {
            "tight_um": self.tight_um,
            "relaxed_um": self.relaxed_um,
            "velocity_weight": self.velocity_weight,
            "learned_bonus": self.learned_bonus,
            "raw_distance_weight": self.raw_distance_weight,
            "output_edge_max_um": self.output_edge_max_um,
            "max_accel_um": self.max_accel_um,
            "max_bonus": self.max_bonus,
            "next_step_gate_um": self.next_step_gate_um,
        }
        for name, value in numeric.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MotionRelinkError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise MotionRelinkError(f"{name} must be a finite number")
        if isinstance(self.max_frame_nodes, bool) or not isinstance(self.max_frame_nodes, int):
            raise MotionRelinkError("max_frame_nodes must be an integer")
        fixed = (
            ("tight_um", float(self.tight_um), _TIGHT_UM),
            ("relaxed_um", float(self.relaxed_um), _RELAXED_UM),
            ("velocity_weight", float(self.velocity_weight), _VELOCITY_WEIGHT),
            ("learned_bonus", float(self.learned_bonus), _LEARNED_BONUS),
            (
                "raw_distance_weight",
                float(self.raw_distance_weight),
                _RAW_DISTANCE_WEIGHT,
            ),
            ("output_edge_max_um", float(self.output_edge_max_um), _OUTPUT_EDGE_MAX_UM),
            ("max_accel_um", float(self.max_accel_um), MAX_ACCEL_UM),
            ("max_bonus", float(self.max_bonus), MAX_BONUS),
            ("next_step_gate_um", float(self.next_step_gate_um), NEXT_STEP_GATE_UM),
        )
        for name, value, expected in fixed:
            if value != expected:
                raise MotionRelinkError(f"{name} is fixed at {expected}")
        if self.max_frame_nodes != _MAX_FRAME_NODES:
            raise MotionRelinkError(f"max_frame_nodes is fixed at {_MAX_FRAME_NODES}")


@dataclass(frozen=True, slots=True)
class MotionRelinkResult:
    """Normalized relinked edges and deterministic replay counters."""

    edges: tuple[dict[str, object], ...]
    stats: dict[str, int | float]


def _point3(value: object, name: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MotionRelinkError(f"{name} must contain three finite coordinates") from exc
    if point.shape != (3,) or not np.isfinite(point).all():
        raise MotionRelinkError(f"{name} must contain three finite coordinates")
    return point


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MotionRelinkError(f"{name} must be an integer")
    return int(value)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise MotionRelinkError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MotionRelinkError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise MotionRelinkError(f"{name} must be finite")
    return result


def _node_position_um(node: Mapping[str, object], index: int) -> np.ndarray:
    voxel_keys = ("z_vox", "y_vox", "x_vox")
    plain_keys = ("z", "y", "x")
    has_voxel = [key in node for key in voxel_keys]
    has_plain = [key in node for key in plain_keys]
    if any(has_voxel) and not all(has_voxel):
        raise MotionRelinkError(f"node {index} has an incomplete voxel coordinate triple")
    if any(has_plain) and not all(has_plain):
        raise MotionRelinkError(f"node {index} has an incomplete coordinate triple")
    if all(has_voxel):
        coordinates = [_finite_float(node[key], f"node {index} {key}") for key in voxel_keys]
    elif all(has_plain):
        coordinates = [_finite_float(node[key], f"node {index} {key}") for key in plain_keys]
    else:
        raise MotionRelinkError(f"node {index} is missing coordinates")
    return np.asarray(coordinates, dtype=np.float64) * np.asarray(VOXEL_SCALE_UM)


def _materialize_records(records: object, name: str) -> list[object]:
    if isinstance(records, (str, bytes, Mapping)):
        raise MotionRelinkError(f"{name} must be an iterable of mappings")
    try:
        return list(records)  # type: ignore[arg-type]
    except TypeError as exc:
        raise MotionRelinkError(f"{name} must be an iterable of mappings") from exc


def _normalize_inputs(
    nodes: Iterable[Mapping[str, object]],
    post_ilp_edges: Iterable[Mapping[str, object]],
) -> tuple[
    dict[int, int],
    dict[int, np.ndarray],
    dict[int, list[int]],
    tuple[dict[str, object], ...],
    dict[tuple[int, int], float],
]:
    node_times: dict[int, int] = {}
    position_um: dict[int, np.ndarray] = {}
    ids_by_t: dict[int, list[int]] = {}
    for index, value in enumerate(_materialize_records(nodes, "nodes")):
        if not isinstance(value, Mapping):
            raise MotionRelinkError(f"node {index} must be a mapping")
        if "node_id" not in value or "t" not in value:
            raise MotionRelinkError(f"node {index} is missing node_id or t")
        node_id = _strict_int(value["node_id"], f"node {index} node_id")
        t = _strict_int(value["t"], f"node {index} t")
        if node_id in node_times:
            raise MotionRelinkError(f"duplicate node_id {node_id}")
        node_times[node_id] = t
        position_um[node_id] = _node_position_um(value, index)
        ids_by_t.setdefault(t, []).append(node_id)
    for frame_ids in ids_by_t.values():
        frame_ids.sort()

    normalized_edges: list[dict[str, object]] = []
    learned_probabilities: dict[tuple[int, int], float] = {}
    seen_pairs: set[tuple[int, int]] = set()
    for index, value in enumerate(_materialize_records(post_ilp_edges, "post_ilp_edges")):
        if not isinstance(value, Mapping):
            raise MotionRelinkError(f"edge {index} must be a mapping")
        if "source_id" not in value or "target_id" not in value:
            raise MotionRelinkError(f"edge {index} is missing source_id or target_id")
        source_id = _strict_int(value["source_id"], f"edge {index} source_id")
        target_id = _strict_int(value["target_id"], f"edge {index} target_id")
        if source_id not in node_times or target_id not in node_times:
            raise MotionRelinkError(f"edge {index} references an unknown node")
        if node_times[target_id] != node_times[source_id] + 1:
            raise MotionRelinkError(f"edge {index} must connect consecutive frames")
        pair = (source_id, target_id)
        if pair in seen_pairs:
            raise MotionRelinkError(f"duplicate post-ILP edge {pair}")
        seen_pairs.add(pair)
        probability_value = value.get("edge_prob")
        probability: float | None
        if probability_value is None:
            probability = None
        else:
            probability = _finite_float(probability_value, f"edge {index} edge_prob")
            if probability < 0.0 or probability > 1.0:
                raise MotionRelinkError(f"edge {index} edge_prob must be in [0, 1]")
            learned_probabilities[pair] = probability
        normalized = dict(value)
        normalized["source_id"] = source_id
        normalized["target_id"] = target_id
        normalized["edge_prob"] = probability
        normalized_edges.append(normalized)
    return (
        node_times,
        position_um,
        ids_by_t,
        tuple(normalized_edges),
        learned_probabilities,
    )


def forward_acceleration_lookahead(
    source_pos: Sequence[float] | np.ndarray,
    target_pos: Sequence[float] | np.ndarray,
    target_t: int,
    ids_by_t: Mapping[int, Sequence[int]],
    position_um: Mapping[int, Sequence[float] | np.ndarray],
    next_step_gate_um: float = NEXT_STEP_GATE_UM,
) -> tuple[float | None, int]:
    """Return the best constant-velocity residual and gated continuation count.

    A continuation is any node in ``target_t + 1`` whose displacement from the
    target does not exceed the fixed next-step gate.  Among those candidates,
    the residual is the distance to the constant-velocity extrapolation
    ``target + (target - source)``.  No continuation is represented by
    ``(None, 0)`` rather than by a fabricated penalty or reward.
    """

    source = _point3(source_pos, "source_pos")
    target = _point3(target_pos, "target_pos")
    frame = _strict_int(target_t, "target_t")
    gate = _finite_float(next_step_gate_um, "next_step_gate_um")
    if gate != NEXT_STEP_GATE_UM:
        raise MotionRelinkError(f"next_step_gate_um is fixed at {NEXT_STEP_GATE_UM}")
    if not isinstance(ids_by_t, Mapping) or not isinstance(position_um, Mapping):
        raise MotionRelinkError("ids_by_t and position_um must be mappings")

    raw_ids = ids_by_t.get(frame + 1, ())
    if isinstance(raw_ids, (str, bytes, Mapping)):
        raise MotionRelinkError("ids_by_t frame entries must be ID sequences")
    try:
        next_ids = sorted(_strict_int(value, "continuation node_id") for value in raw_ids)
    except TypeError as exc:
        raise MotionRelinkError("ids_by_t frame entries must be ID sequences") from exc
    if len(next_ids) != len(set(next_ids)):
        raise MotionRelinkError("ids_by_t contains duplicate continuation IDs")

    predicted_next = target + (target - source)
    residuals: list[float] = []
    for node_id in next_ids:
        if node_id not in position_um:
            raise MotionRelinkError(f"position_um is missing node {node_id}")
        continuation = _point3(position_um[node_id], f"position_um[{node_id}]")
        if float(np.linalg.norm(continuation - target)) > gate:
            continue
        residuals.append(float(np.linalg.norm(continuation - predicted_next)))
    if not residuals:
        return None, 0
    return min(residuals), len(residuals)


def forward_acceleration_bonus(
    residual_um: float | None,
    max_accel_um: float = MAX_ACCEL_UM,
    max_bonus: float = MAX_BONUS,
) -> float:
    """Return the fixed linearly decaying M4 bonus for a residual."""

    maximum_acceleration = _finite_float(max_accel_um, "max_accel_um")
    maximum_bonus = _finite_float(max_bonus, "max_bonus")
    if maximum_acceleration != MAX_ACCEL_UM:
        raise MotionRelinkError(f"max_accel_um is fixed at {MAX_ACCEL_UM}")
    if maximum_bonus != MAX_BONUS:
        raise MotionRelinkError(f"max_bonus is fixed at {MAX_BONUS}")
    if residual_um is None:
        return 0.0
    residual = _finite_float(residual_um, "residual_um")
    if residual < 0.0:
        raise MotionRelinkError("residual_um must be nonnegative")
    if residual >= maximum_acceleration:
        return 0.0
    return maximum_bonus * (1.0 - residual / maximum_acceleration)


def _initial_stats(node_count: int, edge_count: int) -> dict[str, int | float]:
    return {
        "motion_relink_input_nodes": node_count,
        "motion_relink_input_edges": edge_count,
        "motion_relink_edges": 0,
        "motion_relink_tight_edges": 0,
        "motion_relink_relaxed_edges": 0,
        "motion_relink_frames": 0,
        "motion_relink_replaced_raw_edges": 0,
        "motion_relink_fallback_raw": 0,
        "motion_relink_skipped_large_frame": 0,
        "motion_relink_prefilter_dropped_long_edges": 0,
        "dropped_long_edges": 0,
        "motion_relink_accel_evaluated": 0,
        "motion_relink_accel_continuations": 0,
        "motion_relink_accel_bonus_edges": 0,
        "motion_relink_accel_bonus_sum": 0.0,
    }


def replay_motion_relink(
    nodes: Iterable[Mapping[str, object]],
    post_ilp_edges: Iterable[Mapping[str, object]],
    config: MotionRelinkConfig = MotionRelinkConfig(),
    *,
    apply_m4: bool = True,
) -> MotionRelinkResult:
    """Replay current V32 two-pass relinking plus the fixed M4 bonus.

    Learned probabilities are deliberately masked to pairs in
    ``post_ilp_edges``.  Geometry may introduce a different candidate pair,
    but that pair receives zero learned-probability bonus.  If relinking is
    Before relinking, V32's fixed 14-um output edge filter is applied. If
    relinking is skipped for the frame-size guard or produces no edge, that
    exact filtered post-ILP list is the deterministic fallback.
    ``apply_m4=False`` is the sole infrastructure control: it zeros only the
    acceleration bonus and therefore replays the frozen V32 baseline cost.
    """

    if not isinstance(config, MotionRelinkConfig):
        raise MotionRelinkError("config must be a MotionRelinkConfig")
    if type(apply_m4) is not bool:
        raise MotionRelinkError("apply_m4 must be a boolean")
    (
        node_times,
        position_um,
        ids_by_t,
        normalized_edges,
        learned_probabilities,
    ) = _normalize_inputs(nodes, post_ilp_edges)
    stats = _initial_stats(len(node_times), len(normalized_edges))
    if not node_times:
        stats["motion_relink_fallback_raw"] = 1
        return MotionRelinkResult((), stats)

    filtered_edges: list[dict[str, object]] = []
    for edge in normalized_edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        distance_um = float(np.linalg.norm(position_um[target_id] - position_um[source_id]))
        if distance_um > config.output_edge_max_um:
            stats["motion_relink_prefilter_dropped_long_edges"] += 1
            stats["dropped_long_edges"] += 1
            continue
        filtered = dict(edge)
        filtered["distance_um"] = distance_um
        filtered_edges.append(filtered)
    normalized_edges = tuple(filtered_edges)
    retained_pairs = {
        (int(edge["source_id"]), int(edge["target_id"])) for edge in normalized_edges
    }
    learned_probabilities = {
        pair: probability
        for pair, probability in learned_probabilities.items()
        if pair in retained_pairs
    }

    if max((len(frame_ids) for frame_ids in ids_by_t.values()), default=0) > config.max_frame_nodes:
        stats["motion_relink_skipped_large_frame"] = 1
        stats["motion_relink_fallback_raw"] = 1
        return MotionRelinkResult(normalized_edges, stats)

    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
    ) -> list[tuple[int, int, float, float, float, float | None, int, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        shape = (len(source_ids), len(target_ids))
        cost = np.full(shape, big, dtype=np.float64)
        raw_dist = np.full(shape, np.inf, dtype=np.float64)
        motion_dist = np.full(shape, np.inf, dtype=np.float64)
        probability = np.zeros(shape, dtype=np.float64)
        acceleration_residual = np.full(shape, np.nan, dtype=np.float64)
        continuation_count = np.zeros(shape, dtype=np.int64)
        acceleration_bonus = np.zeros(shape, dtype=np.float64)
        for row, source_id in enumerate(source_ids):
            source = position_um[source_id]
            previous = predecessor_position_um.get(source_id)
            predicted = (
                source
                if previous is None
                else source + float(config.velocity_weight) * (source - previous)
            )
            for column, target_id in enumerate(target_ids):
                target = position_um[target_id]
                raw = float(np.linalg.norm(target - source))
                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target - predicted))
                prob = learned_probabilities.get((source_id, target_id), 0.0)
                residual, continuations = forward_acceleration_lookahead(
                    source,
                    target,
                    node_times[target_id],
                    ids_by_t,
                    position_um,
                    config.next_step_gate_um,
                )
                m4_bonus = forward_acceleration_bonus(
                    residual,
                    config.max_accel_um,
                    config.max_bonus,
                )
                bonus = m4_bonus if apply_m4 else 0.0
                raw_dist[row, column] = raw
                motion_dist[row, column] = motion
                probability[row, column] = prob
                if residual is not None:
                    acceleration_residual[row, column] = residual
                continuation_count[row, column] = continuations
                acceleration_bonus[row, column] = bonus
                cost[row, column] = (
                    motion
                    + float(config.raw_distance_weight) * raw
                    - float(config.learned_bonus) * prob
                    - bonus
                )
        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int, float, float, float, float | None, int, float, float]] = []
        for row, column in zip(row_ind, col_ind):
            if cost[row, column] >= big:
                continue
            residual_value = float(acceleration_residual[row, column])
            matches.append(
                (
                    source_ids[int(row)],
                    target_ids[int(column)],
                    float(raw_dist[row, column]),
                    float(motion_dist[row, column]),
                    float(probability[row, column]),
                    residual_value if math.isfinite(residual_value) else None,
                    int(continuation_count[row, column]),
                    float(acceleration_bonus[row, column]),
                    float(cost[row, column]),
                )
            )
        return matches

    for t in sorted(ids_by_t):
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches: list[
            tuple[int, int, float, float, float, float | None, int, float, float, str]
        ] = []
        for pass_name, gate_um in (
            ("tight", float(config.tight_um)),
            ("relaxed", float(config.relaxed_um)),
        ):
            pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
            pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
            for match in assign_pass(pass_sources, pass_targets, gate_um):
                source_id, target_id = match[:2]
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append((*match, pass_name))
                stats[f"motion_relink_{pass_name}_edges"] += 1

        for (
            source_id,
            target_id,
            raw,
            motion,
            probability,
            residual,
            continuations,
            bonus,
            total_cost,
            pass_name,
        ) in frame_matches:
            selected_edges.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "edge_prob": probability,
                    "distance_um": raw,
                    "motion_distance_um": motion,
                    "motion_relinked": 1,
                    "motion_pass": pass_name,
                    "acceleration_residual_um": residual,
                    "acceleration_continuations": continuations,
                    "acceleration_bonus": bonus,
                    "motion_cost": total_cost,
                }
            )
            predecessor_position_um[target_id] = position_um[source_id]
            stats["motion_relink_accel_evaluated"] += 1
            stats["motion_relink_accel_continuations"] += continuations
            if bonus > 0.0:
                stats["motion_relink_accel_bonus_edges"] += 1
                stats["motion_relink_accel_bonus_sum"] += bonus
        stats["motion_relink_frames"] += 1

    if not selected_edges:
        stats["motion_relink_fallback_raw"] = 1
        return MotionRelinkResult(normalized_edges, stats)
    selected_edges.sort(
        key=lambda edge: (
            node_times[int(edge["source_id"])],
            int(edge["source_id"]),
            int(edge["target_id"]),
        )
    )
    stats["motion_relink_edges"] = len(selected_edges)
    stats["motion_relink_replaced_raw_edges"] = len(normalized_edges)
    return MotionRelinkResult(tuple(selected_edges), stats)


def replay_cached_arrays(
    post_ilp_nodes: np.ndarray,
    post_ilp_edges: np.ndarray,
    *,
    apply_m4: bool,
) -> MotionRelinkResult:
    """Replay exact structured arrays emitted by the V32 cache boundary.

    This is a deliberately narrow in-memory adapter.  It validates the durable
    cache dtypes directly and performs no path, cache-manifest, data, ranker, or
    scorer access.  Scientific parameters remain the frozen default config;
    only the explicit infrastructure control can disable the M4 bonus.
    """

    if not isinstance(post_ilp_nodes, np.ndarray) or post_ilp_nodes.ndim != 1:
        raise MotionRelinkError("post_ilp_nodes must be a one-dimensional structured array")
    if post_ilp_nodes.dtype != _CACHE_NODE_DTYPE:
        raise MotionRelinkError("post_ilp_nodes has the wrong cache dtype")
    if not isinstance(post_ilp_edges, np.ndarray) or post_ilp_edges.ndim != 1:
        raise MotionRelinkError("post_ilp_edges must be a one-dimensional structured array")
    if post_ilp_edges.dtype != _CACHE_POST_EDGE_DTYPE:
        raise MotionRelinkError("post_ilp_edges has the wrong cache dtype")
    nodes = (
        {
            "node_id": int(row["node_id"]),
            "t": int(row["t"]),
            "z_vox": float(row["z_vox"]),
            "y_vox": float(row["y_vox"]),
            "x_vox": float(row["x_vox"]),
        }
        for row in post_ilp_nodes
    )
    edges = (
        {
            "source_id": int(row["source_id"]),
            "target_id": int(row["target_id"]),
            "edge_prob": float(row["edge_prob"]),
        }
        for row in post_ilp_edges
    )
    return replay_motion_relink(nodes, edges, apply_m4=apply_m4)


__all__ = [
    "MAX_ACCEL_UM",
    "MAX_BONUS",
    "NEXT_STEP_GATE_UM",
    "VOXEL_SCALE_UM",
    "MotionRelinkConfig",
    "MotionRelinkError",
    "MotionRelinkResult",
    "forward_acceleration_bonus",
    "forward_acceleration_lookahead",
    "replay_cached_arrays",
    "replay_motion_relink",
]
