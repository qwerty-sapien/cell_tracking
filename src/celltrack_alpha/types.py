"""Shared types for the low-compute scenario-routed tracking alpha."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    output_offset_zyx: tuple[int, int, int] = (0, 0, 0)
    prefix_output_offsets_zyx: tuple[tuple[str, int, int, int], ...] = ()
    notes: str = ""


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
