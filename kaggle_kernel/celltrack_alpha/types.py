"""Shared types for the low-compute scenario-routed tracking alpha."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from numbers import Integral, Real
from typing import Any


@dataclass
class CandidateNode:
    """One possible cell-center detection before final graph selection."""

    dataset: str
    node_id: int
    t: int
    z: int
    y: int
    x: int
    score: float
    sources: tuple[str, ...]
    radius_um: float
    uncertainty: float = 1.0
    local_density: float = 0.0
    intensity: float = 0.0
    features: dict[str, float] = field(default_factory=dict)

    @property
    def coord_zyx(self) -> tuple[int, int, int]:
        return int(self.z), int(self.y), int(self.x)

    def to_submission_node(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "node_id": int(self.node_id),
            "t": int(self.t),
            "z": int(self.z),
            "y": int(self.y),
            "x": int(self.x),
            "score": float(self.score),
            "sources": "|".join(self.sources),
        }


@dataclass(frozen=True)
class CandidateEdge:
    """A possible directed temporal link between two candidate nodes."""

    source_id: int
    target_id: int
    dt: int
    distance_um: float
    cost: float
    score: float
    kind: str = "continuation"
    features: dict[str, float] = field(default_factory=dict)

    def to_submission_edge(self) -> dict[str, int]:
        return {"source_id": int(self.source_id), "target_id": int(self.target_id)}


@dataclass(frozen=True)
class CandidateFork:
    """A one-parent/two-child division motif."""

    parent_id: int
    child_a_id: int
    child_b_id: int
    score: float
    parent_child_mean_um: float
    daughter_separation_um: float


@dataclass(frozen=True)
class RouteConfig:
    """Concrete tracker settings selected by the scenario router."""

    label: str
    detector_sources: tuple[str, ...]
    threshold_quantiles: tuple[float, ...]
    min_distance_um: float
    fusion_radius_um: float
    max_points_per_t: int
    max_link_um: float
    max_gap: int = 1
    gap_link_um: float = 10.0
    max_neighbors: int = 5
    edge_cost_threshold: float = 0.92
    birth_cost: float | None = None
    death_cost: float | None = None
    node_score_weight: float = 0.22
    density_cost_weight: float = 0.04
    appearance_cost_weight: float = 0.0
    gap_motion_cost_weight: float = 0.25
    enable_divisions: bool = False
    division_cost_threshold: float = 0.82
    division_requires_unlinked_parent: bool = True
    division_requires_child_future: bool = False
    division_child_max_um: float = 9.0
    division_child_min_sep_um: float = 2.0
    division_child_max_sep_um: float = 11.0
    division_allow_link_steal: bool = False
    division_steal_max_distance_um: float = 7.0
    division_steal_source_max_um: float = 5.0
    enable_second_choice_edge_rescue: bool = False
    edge_rescue_short_max_um: float = 2.2
    edge_rescue_long_min_um: float = 6.0
    edge_rescue_long_max_um: float = 7.1
    edge_rescue_margin_min_um: float = 4.5
    edge_rescue_intensity_ratio: float = 1.02
    edge_rescue_alt_cost_max: float = 0.45
    enable_localmax_refinement: bool = False
    localmax_refine_radius_um: float = 2.5
    localmax_refine_sources: tuple[str, ...] = ("dog_abs",)
    localmax_refine_center_offset_zyx: tuple[int, int, int] = (0, 0, 0)
    localmax_refine_center_offsets_zyx: tuple[tuple[int, int, int], ...] = ()
    enable_localmax_clone_layers: bool = False
    localmax_clone_radius_um: float = 2.5
    localmax_clone_sources: tuple[str, ...] = ("dog_abs",)
    localmax_clone_center_offsets_zyx: tuple[tuple[int, int, int], ...] = ()
    enable_localmax_clone_target_rewire: bool = False
    localmax_clone_rewire_target_radius_um: float = 4.0
    localmax_clone_rewire_source_radius_um: float = 4.0
    enable_base_chain_target_rewire: bool = False
    base_chain_rewire_target_radius_um: float = 5.5
    base_chain_rewire_source_radius_um: float = 4.0
    enable_offset_clone_layers: bool = False
    offset_clone_offsets_zyx: tuple[tuple[int, int, int], ...] = ()
    offset_clone_skip_clipped: bool = False
    offset_clone_filter_source_mode: str = "any"
    offset_clone_filter_min_component_timepoints: int = 0
    offset_clone_filter_component_max_x: int | None = None
    enable_endpoint_forward_clones: bool = False
    endpoint_forward_radius_um: float = 3.0
    endpoint_forward_sources: tuple[str, ...] = ("smooth",)
    endpoint_forward_max_per_t: int = 16
    endpoint_forward_requires_history: bool = False
    enable_endpoint_backward_clones: bool = False
    endpoint_backward_max_edge_um: float = 6.0
    endpoint_backward_min_edge_um: float = 0.0
    endpoint_backward_duplicate_um: float = 5.5
    endpoint_backward_max_per_t: int = 1
    endpoint_backward_axis_mode: str = "x_preferred"
    enable_refined_candidate_admission: bool = False
    refined_candidate_admission_route: str = "refined_dog_compact"
    refined_candidate_admission_max_per_t: int = 8
    refined_candidate_admission_min_dist_um: float = 1.0
    refined_candidate_admission_score_scale: float = 0.65
    enable_refined_chain_conflict_rescue: bool = False
    refined_chain_conflict_max_distance_um: float = 3.0
    refined_chain_conflict_max_cost: float = 0.42
    refined_chain_conflict_max_target_rank: int = 3
    refined_chain_conflict_min_temporal_count: float = 2.0
    enable_short_edge_pruning: bool = False
    short_edge_prune_min_distance_um: float = 0.0
    enable_free_edge_admission: bool = False
    free_edge_admission_max_cost: float = 0.55
    free_edge_admission_min_distance_um: float = 0.0
    free_edge_admission_max_distance_um: float = 4.5
    free_edge_admission_require_mutual_nearest: bool = True
    free_edge_admission_max_source_candidates: int = 1
    free_edge_admission_max_target_candidates: int = 1
    free_edge_admission_max_edges: int = 32
    enable_duplicate_endpoint_rescue: bool = False
    duplicate_endpoint_rescue_max_um: float = 2.0
    duplicate_endpoint_rescue_min_cost: float = 0.0
    duplicate_endpoint_rescue_max_cost: float = 0.45
    duplicate_endpoint_rescue_max_distance_um: float = 4.5
    duplicate_endpoint_rescue_max_swaps: int = 4
    enable_component_pruning: bool = False
    component_prune_min_edges: int = 0
    component_prune_min_x: int | None = None
    component_prune_max_x: int | None = None
    component_prune_x_offset_only: bool = False
    component_prune_drop_one_edge_offset: bool = False
    enable_motion_endpoint_admission: bool = False
    motion_endpoint_pred_max_um: float = 7.5
    motion_endpoint_source_max_um: float = 10.0
    motion_endpoint_max_edges_per_t: int = 4
    motion_endpoint_target_source_mode: str = "zshift"
    motion_endpoint_rank_mode: str = "source_order"
    motion_endpoint_requires_history: bool = True
    output_offset_zyx: tuple[int, int, int] = (0, 0, 0)
    prefix_output_offsets_zyx: tuple[tuple[str, int, int, int], ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        validate_route_config(self)


_LOCALMAX_SOURCES = frozenset(
    {"raw", "smooth", "local_contrast", "dog_abs", "dog_bright", "dog_dark"}
)
_PROPOSAL_DETECTOR_SOURCES = frozenset(
    {
        "components",
        "cover_contrast",
        "cover_dog_abs",
        "cover_dog_bright",
        "cover_dog_dark",
        "cover_raw",
        "cover_smooth",
        "dog_abs",
        "dog_bright",
        "dog_dark",
        "hmax",
        "local_contrast",
        "refine_local_score",
        "subtile_contrast",
        "subtile_dog_abs",
        "subtile_dog_bright",
        "subtile_dog_dark",
        "template_score",
        "tile_contrast",
        "tile_dog_abs",
        "tile_dog_bright",
        "tile_dog_dark",
        "tile_raw",
        "zshift_dog_bright_m3",
        "zshift_dog_bright_m3_mid",
        "zshift_dog_dark_m5",
        "zshift_dog_dark_m5_mid",
    }
)
_DEFAULT_ROUTE_LABELS = frozenset(
    {
        "clean_point",
        "clean_compact",
        "dense_fusion",
        "low_signal",
        "balanced_local",
        "balanced_compact",
        "balanced_midpoint",
        "balanced_tight_link",
        "balanced_axial_compact",
        "hybrid_rescue_compact",
        "balanced_dog_compact",
        "balanced_dog_tight",
        "coverage_dog_compact",
        "score_coverage_compact",
        "axial_rescue_compact",
        "template_point_compact",
        "subtile_dog_compact",
        "refined_dog_compact",
        "high_motion_gap",
        "division_likely",
    }
)
_OFFSET_CLONE_SOURCE_MODES = frozenset(
    {"any", "contrast_only", "raw_only", "raw_contrast", "contrast_any", "raw_any"}
)
_ENDPOINT_BACKWARD_AXIS_MODES = frozenset({"any", "x", "x_preferred", "interior_dup"})
_MOTION_ENDPOINT_TARGET_MODES = frozenset(
    {"", "any", "all", "same_source", "source_consistent", "zshift", "not_tile"}
)
_MOTION_ENDPOINT_RANK_MODES = frozenset({"source_order", "pred", "source_desc", "id"})


def validate_route_config(config: RouteConfig) -> None:
    """Validate every public ``RouteConfig`` field and its cross-field bounds."""

    if not isinstance(config, RouteConfig):
        raise TypeError(f"config must be a RouteConfig, got {type(config).__name__}")

    checked: set[str] = set()

    def value(name: str) -> Any:
        checked.add(name)
        return getattr(config, name)

    def require_bool(name: str) -> None:
        current = value(name)
        if not isinstance(current, bool):
            raise ValueError(f"{name} must be a boolean, got {current!r}")

    def require_string(name: str, *, allow_empty: bool = False) -> str:
        current = value(name)
        if not isinstance(current, str) or (not allow_empty and not current.strip()):
            qualifier = "a string" if allow_empty else "a non-empty string"
            raise ValueError(f"{name} must be {qualifier}, got {current!r}")
        return current

    def require_number(name: str, *, minimum: float | None = None, strict: bool = False) -> float:
        current = value(name)
        if isinstance(current, bool) or not isinstance(current, Real):
            raise ValueError(f"{name} must be a finite number, got {current!r}")
        resolved = float(current)
        if not math.isfinite(resolved):
            raise ValueError(f"{name} must be a finite number, got {current!r}")
        if minimum is not None and (resolved <= minimum if strict else resolved < minimum):
            relation = "greater than" if strict else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}, got {current!r}")
        return resolved

    def require_optional_number(name: str, *, minimum: float = 0.0) -> float | None:
        current = value(name)
        if current is None:
            return None
        if isinstance(current, bool) or not isinstance(current, Real):
            raise ValueError(f"{name} must be None or a finite number, got {current!r}")
        resolved = float(current)
        if not math.isfinite(resolved) or resolved < minimum:
            raise ValueError(
                f"{name} must be None or a finite number at least {minimum}, got {current!r}"
            )
        return resolved

    def require_integer(name: str, *, minimum: int) -> int:
        current = value(name)
        if isinstance(current, bool) or not isinstance(current, Integral) or int(current) < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}, got {current!r}")
        return int(current)

    def require_optional_integer(name: str, *, minimum: int) -> int | None:
        current = value(name)
        if current is None:
            return None
        if isinstance(current, bool) or not isinstance(current, Integral) or int(current) < minimum:
            raise ValueError(
                f"{name} must be None or an integer at least {minimum}, got {current!r}"
            )
        return int(current)

    def require_string_tuple(
        name: str,
        *,
        allowed: frozenset[str] | None = None,
    ) -> tuple[str, ...]:
        current = value(name)
        if (
            not isinstance(current, tuple)
            or not current
            or any(not isinstance(item, str) or not item.strip() for item in current)
        ):
            raise ValueError(f"{name} must be a non-empty tuple of non-empty strings")
        if allowed is not None:
            unknown = sorted(set(current) - allowed)
            if unknown:
                raise ValueError(f"{name} contains unsupported values: {unknown}")
        return current

    def require_integer_triple(name: str) -> tuple[int, int, int]:
        current = value(name)
        if (
            not isinstance(current, tuple)
            or len(current) != 3
            or any(isinstance(item, bool) or not isinstance(item, Integral) for item in current)
        ):
            raise ValueError(f"{name} must be a Z/Y/X tuple of exactly three integers")
        return tuple(int(item) for item in current)

    def require_integer_triples(name: str) -> tuple[tuple[int, int, int], ...]:
        current = value(name)
        if not isinstance(current, tuple):
            raise ValueError(f"{name} must be a tuple of Z/Y/X integer triples")
        for item in current:
            if (
                not isinstance(item, tuple)
                or len(item) != 3
                or any(isinstance(part, bool) or not isinstance(part, Integral) for part in item)
            ):
                raise ValueError(f"{name} must contain only Z/Y/X integer triples")
        return current

    require_string("label")
    detector_sources = require_string_tuple("detector_sources")

    quantiles = value("threshold_quantiles")
    if not isinstance(quantiles, tuple):
        raise ValueError("threshold_quantiles must be a tuple of finite values in [0, 1]")
    for quantile in quantiles:
        if (
            isinstance(quantile, bool)
            or not isinstance(quantile, Real)
            or not math.isfinite(float(quantile))
            or not 0.0 <= float(quantile) <= 1.0
        ):
            raise ValueError("threshold_quantiles must be a tuple of finite values in [0, 1]")
    if not quantiles and any(source in _PROPOSAL_DETECTOR_SOURCES for source in detector_sources):
        raise ValueError(
            "threshold_quantiles must not be empty when detector_sources contains a proposal source"
        )

    positive_numbers = (
        "max_link_um",
        "gap_link_um",
        "division_child_max_um",
        "division_child_max_sep_um",
        "localmax_refine_radius_um",
        "localmax_clone_radius_um",
        "localmax_clone_rewire_target_radius_um",
        "localmax_clone_rewire_source_radius_um",
        "base_chain_rewire_target_radius_um",
        "base_chain_rewire_source_radius_um",
        "endpoint_forward_radius_um",
        "endpoint_backward_max_edge_um",
        "refined_candidate_admission_score_scale",
        "refined_chain_conflict_max_distance_um",
        "free_edge_admission_max_distance_um",
        "duplicate_endpoint_rescue_max_distance_um",
        "motion_endpoint_pred_max_um",
        "motion_endpoint_source_max_um",
    )
    for name in positive_numbers:
        require_number(name, minimum=0.0, strict=True)

    nonnegative_numbers = (
        "min_distance_um",
        "fusion_radius_um",
        "edge_cost_threshold",
        "node_score_weight",
        "density_cost_weight",
        "appearance_cost_weight",
        "gap_motion_cost_weight",
        "division_cost_threshold",
        "division_child_min_sep_um",
        "division_steal_max_distance_um",
        "division_steal_source_max_um",
        "edge_rescue_short_max_um",
        "edge_rescue_long_min_um",
        "edge_rescue_long_max_um",
        "edge_rescue_margin_min_um",
        "edge_rescue_intensity_ratio",
        "edge_rescue_alt_cost_max",
        "endpoint_backward_min_edge_um",
        "endpoint_backward_duplicate_um",
        "refined_candidate_admission_min_dist_um",
        "refined_chain_conflict_min_temporal_count",
        "short_edge_prune_min_distance_um",
        "free_edge_admission_min_distance_um",
        "duplicate_endpoint_rescue_max_um",
    )
    for name in nonnegative_numbers:
        require_number(name, minimum=0.0)

    for name in (
        "refined_chain_conflict_max_cost",
        "free_edge_admission_max_cost",
        "duplicate_endpoint_rescue_min_cost",
        "duplicate_endpoint_rescue_max_cost",
    ):
        require_number(name)

    require_optional_number("birth_cost")
    require_optional_number("death_cost")

    for name in (
        "max_points_per_t",
        "max_gap",
        "max_neighbors",
        "endpoint_forward_max_per_t",
        "endpoint_backward_max_per_t",
        "refined_candidate_admission_max_per_t",
        "refined_chain_conflict_max_target_rank",
        "free_edge_admission_max_source_candidates",
        "free_edge_admission_max_target_candidates",
        "free_edge_admission_max_edges",
        "duplicate_endpoint_rescue_max_swaps",
        "motion_endpoint_max_edges_per_t",
    ):
        require_integer(name, minimum=1)
    require_integer("offset_clone_filter_min_component_timepoints", minimum=0)
    require_integer("component_prune_min_edges", minimum=0)
    require_optional_integer("offset_clone_filter_component_max_x", minimum=0)
    component_min_x = require_optional_integer("component_prune_min_x", minimum=0)
    component_max_x = require_optional_integer("component_prune_max_x", minimum=0)

    for name in (
        "enable_divisions",
        "division_requires_unlinked_parent",
        "division_requires_child_future",
        "division_allow_link_steal",
        "enable_second_choice_edge_rescue",
        "enable_localmax_refinement",
        "enable_localmax_clone_layers",
        "enable_localmax_clone_target_rewire",
        "enable_base_chain_target_rewire",
        "enable_offset_clone_layers",
        "offset_clone_skip_clipped",
        "enable_endpoint_forward_clones",
        "endpoint_forward_requires_history",
        "enable_endpoint_backward_clones",
        "enable_refined_candidate_admission",
        "enable_refined_chain_conflict_rescue",
        "enable_short_edge_pruning",
        "enable_free_edge_admission",
        "free_edge_admission_require_mutual_nearest",
        "enable_duplicate_endpoint_rescue",
        "enable_component_pruning",
        "component_prune_x_offset_only",
        "component_prune_drop_one_edge_offset",
        "enable_motion_endpoint_admission",
        "motion_endpoint_requires_history",
    ):
        require_bool(name)

    require_string_tuple("localmax_refine_sources", allowed=_LOCALMAX_SOURCES)
    require_integer_triple("localmax_refine_center_offset_zyx")
    require_integer_triples("localmax_refine_center_offsets_zyx")
    require_string_tuple("localmax_clone_sources", allowed=_LOCALMAX_SOURCES)
    localmax_clone_offsets = require_integer_triples("localmax_clone_center_offsets_zyx")
    offset_clone_offsets = require_integer_triples("offset_clone_offsets_zyx")
    require_string_tuple("endpoint_forward_sources", allowed=_LOCALMAX_SOURCES)
    require_integer_triple("output_offset_zyx")

    offset_mode = require_string("offset_clone_filter_source_mode").strip().lower()
    if offset_mode not in _OFFSET_CLONE_SOURCE_MODES:
        raise ValueError(f"offset_clone_filter_source_mode has unsupported value {offset_mode!r}")
    backward_mode = require_string("endpoint_backward_axis_mode").strip().lower()
    if backward_mode not in _ENDPOINT_BACKWARD_AXIS_MODES:
        raise ValueError(f"endpoint_backward_axis_mode has unsupported value {backward_mode!r}")
    refined_route = require_string("refined_candidate_admission_route")
    if config.enable_refined_candidate_admission and refined_route not in _DEFAULT_ROUTE_LABELS:
        raise ValueError(
            "refined_candidate_admission_route must name a registered default route when "
            "enable_refined_candidate_admission is true"
        )
    target_mode = require_string("motion_endpoint_target_source_mode", allow_empty=True).strip().lower()
    if target_mode not in _MOTION_ENDPOINT_TARGET_MODES:
        raise ValueError(f"motion_endpoint_target_source_mode has unsupported value {target_mode!r}")
    rank_mode = require_string("motion_endpoint_rank_mode").strip().lower()
    if rank_mode not in _MOTION_ENDPOINT_RANK_MODES:
        raise ValueError(f"motion_endpoint_rank_mode has unsupported value {rank_mode!r}")
    require_string("notes", allow_empty=True)

    prefix_offsets = value("prefix_output_offsets_zyx")
    if not isinstance(prefix_offsets, tuple):
        raise ValueError(
            "prefix_output_offsets_zyx must be a tuple of (prefix, dz, dy, dx) rows"
        )
    for row in prefix_offsets:
        if (
            not isinstance(row, tuple)
            or len(row) != 4
            or not isinstance(row[0], str)
            or not row[0]
            or any(isinstance(item, bool) or not isinstance(item, Integral) for item in row[1:])
        ):
            raise ValueError(
                "prefix_output_offsets_zyx must contain only (non-empty prefix, dz, dy, dx) rows"
            )

    if config.division_child_min_sep_um > config.division_child_max_sep_um:
        raise ValueError(
            "division_child_min_sep_um must not exceed division_child_max_sep_um"
        )
    if config.edge_rescue_long_min_um > config.edge_rescue_long_max_um:
        raise ValueError("edge_rescue_long_min_um must not exceed edge_rescue_long_max_um")
    if config.endpoint_backward_min_edge_um > config.endpoint_backward_max_edge_um:
        raise ValueError(
            "endpoint_backward_min_edge_um must not exceed endpoint_backward_max_edge_um"
        )
    if config.free_edge_admission_min_distance_um > config.free_edge_admission_max_distance_um:
        raise ValueError(
            "free_edge_admission_min_distance_um must not exceed "
            "free_edge_admission_max_distance_um"
        )
    if config.duplicate_endpoint_rescue_min_cost > config.duplicate_endpoint_rescue_max_cost:
        raise ValueError(
            "duplicate_endpoint_rescue_min_cost must not exceed "
            "duplicate_endpoint_rescue_max_cost"
        )
    if component_min_x is not None and component_max_x is not None and component_min_x > component_max_x:
        raise ValueError("component_prune_min_x must not exceed component_prune_max_x")
    if config.enable_localmax_clone_layers and not localmax_clone_offsets:
        raise ValueError(
            "localmax_clone_center_offsets_zyx must not be empty when "
            "enable_localmax_clone_layers is true"
        )
    if config.enable_offset_clone_layers and not offset_clone_offsets:
        raise ValueError(
            "offset_clone_offsets_zyx must not be empty when enable_offset_clone_layers is true"
        )

    expected = {item.name for item in fields(config)}
    if checked != expected:
        missing = sorted(expected - checked)
        extra = sorted(checked - expected)
        raise RuntimeError(
            f"RouteConfig validation coverage mismatch; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class ScenarioProfile:
    """Cheap per-video/per-window features used to choose a route."""

    dataset: str
    t_start: int
    t_end: int
    sampled_timepoints: tuple[int, ...]
    foreground_fraction: float
    bright_candidate_density: float
    dark_candidate_density: float
    both_candidate_density: float
    median_nn_um: float
    temporal_delta: float
    division_motif_rate: float
    intensity_p999: float
    estimated_total_nodes: float | None = None


@dataclass(frozen=True)
class RouteDecision:
    """The router output and its confidence."""

    label: str
    confidence: float
    reason: str
    alternatives: tuple[str, ...]
    config: RouteConfig


@dataclass
class TrackingRunResult:
    """Complete prediction payload for one dataset/window."""

    dataset: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, int]]
    profile: ScenarioProfile
    route: RouteDecision
    diagnostics: dict[str, Any] = field(default_factory=dict)
