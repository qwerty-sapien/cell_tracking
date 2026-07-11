"""Lightweight candidate-graph linker and conservative division selector."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import replace
from numbers import Integral

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .types import CandidateEdge, CandidateFork, CandidateNode, RouteConfig


def _edge_radius(config: RouteConfig, dt: int) -> float:
    if dt <= 1:
        return float(config.max_link_um)
    return float(config.gap_link_um) * float(dt - 1)


def _validated_scale(scale_zyx: tuple[float, float, float]) -> np.ndarray:
    scale = np.asarray(scale_zyx, dtype=np.float64)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(
            "scale_zyx must contain exactly three positive finite values in z/y/x order"
        )
    return scale


def _validate_config(config: RouteConfig) -> None:
    numeric_nonnegative = {
        "edge_cost_threshold": config.edge_cost_threshold,
        "node_score_weight": config.node_score_weight,
        "density_cost_weight": config.density_cost_weight,
        "appearance_cost_weight": config.appearance_cost_weight,
        "gap_motion_cost_weight": config.gap_motion_cost_weight,
    }
    for name, value in numeric_nonnegative.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    for name, value in (("max_link_um", config.max_link_um), ("gap_link_um", config.gap_link_um)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    if (
        isinstance(config.max_gap, bool)
        or int(config.max_gap) != config.max_gap
        or int(config.max_gap) < 1
    ):
        raise ValueError(f"max_gap must be a positive integer, got {config.max_gap!r}")
    if (
        isinstance(config.max_neighbors, bool)
        or int(config.max_neighbors) != config.max_neighbors
        or int(config.max_neighbors) < 1
    ):
        raise ValueError(
            f"max_neighbors must be a positive integer, got {config.max_neighbors!r}"
        )
    for name, value in (("birth_cost", config.birth_cost), ("death_cost", config.death_cost)):
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError(f"{name} must be finite and non-negative when set, got {value!r}")


def _symmetric_relative_difference(a: float, b: float) -> float:
    """Return a bounded, scale-independent appearance difference."""

    if not np.isfinite(a) or not np.isfinite(b):
        return 1.0
    denominator = abs(a) + abs(b) + 1e-6
    return float(min(1.0, abs(a - b) / denominator))


def _validate_edge_candidates(
    candidates: list[CandidateEdge], nodes_by_id: dict[int, CandidateNode]
) -> None:
    for edge in candidates:
        source = nodes_by_id.get(int(edge.source_id))
        target = nodes_by_id.get(int(edge.target_id))
        pair = (int(edge.source_id), int(edge.target_id))
        if source is None or target is None:
            raise ValueError(f"candidate edge endpoint is missing for {pair}")
        if isinstance(edge.dt, bool) or not isinstance(edge.dt, Integral) or edge.dt <= 0:
            raise ValueError(f"candidate edge {pair} dt must be a positive integer")
        actual_dt = int(target.t) - int(source.t)
        if actual_dt <= 0 or int(edge.dt) != actual_dt:
            raise ValueError(
                f"candidate edge {pair} must advance by its declared positive dt={edge.dt}"
            )
        if not np.isfinite(edge.distance_um) or edge.distance_um < 0.0:
            raise ValueError(f"candidate edge {pair} has invalid physical distance")
        if not np.isfinite(edge.cost):
            raise ValueError(f"candidate edge {pair} has non-finite cost")
        if not np.isfinite(edge.score) or not 0.0 <= edge.score <= 1.0:
            raise ValueError(f"candidate edge {pair} score must be finite and in [0, 1]")


def build_edge_candidates(
    nodes_by_t: list[list[CandidateNode]],
    scale_zyx: tuple[float, float, float],
    config: RouteConfig,
) -> list[CandidateEdge]:
    """Build radius/kNN edge candidates over adjacent and optional gap frames."""

    _validate_config(config)
    scale = _validated_scale(scale_zyx)
    edges: list[CandidateEdge] = []
    max_gap = max(1, int(config.max_gap))
    for src_idx, src_nodes in enumerate(nodes_by_t):
        if not src_nodes:
            continue
        src_coords = np.asarray([n.coord_zyx for n in src_nodes], dtype=np.float32) * scale
        for dt in range(1, max_gap + 1):
            tgt_idx = src_idx + dt
            if tgt_idx >= len(nodes_by_t):
                continue
            tgt_nodes = nodes_by_t[tgt_idx]
            if not tgt_nodes:
                continue
            tgt_coords = np.asarray([n.coord_zyx for n in tgt_nodes], dtype=np.float32) * scale
            tree = cKDTree(tgt_coords)
            radius = _edge_radius(config, dt)
            for source_node, source_coord in zip(src_nodes, src_coords, strict=True):
                nearby = tree.query_ball_point(source_coord, r=radius)
                if not nearby:
                    continue
                ranked: list[tuple[float, int]] = []
                for target_i in nearby:
                    target_node = tgt_nodes[int(target_i)]
                    delta = tgt_coords[int(target_i)] - source_coord
                    dist = float(np.sqrt(np.dot(delta, delta)))
                    ranked.append((dist, int(target_i)))
                ranked.sort(key=lambda row: (row[0], tgt_nodes[row[1]].node_id))
                for dist, target_i in ranked[: config.max_neighbors]:
                    target_node = tgt_nodes[target_i]
                    density_penalty = config.density_cost_weight * (
                        source_node.local_density + target_node.local_density
                    ) / 10.0
                    gap_penalty = 0.18 * (dt - 1)
                    appearance_delta = _symmetric_relative_difference(
                        source_node.intensity, target_node.intensity
                    )
                    appearance_penalty = config.appearance_cost_weight * appearance_delta
                    node_bonus = config.node_score_weight * (source_node.score + target_node.score)
                    cost = (
                        (dist / max(radius, 1e-6))
                        + density_penalty
                        + gap_penalty
                        + appearance_penalty
                        - node_bonus
                    )
                    score = 1.0 / (1.0 + np.exp(4.0 * (cost - 0.55)))
                    edges.append(
                        CandidateEdge(
                            source_id=source_node.node_id,
                            target_id=target_node.node_id,
                            dt=dt,
                            distance_um=dist,
                            cost=float(cost),
                            score=float(score),
                            features={
                                "source_score": float(source_node.score),
                                "target_score": float(target_node.score),
                                "density_penalty": float(density_penalty),
                                "appearance_delta": float(appearance_delta),
                                "appearance_penalty": float(appearance_penalty),
                            },
                        )
                    )
    return edges


def _edge_order_key(edge: CandidateEdge) -> tuple[float, int, float, int, int]:
    return (edge.cost, edge.dt, -edge.score, edge.source_id, edge.target_id)


def _assignment_penalties(config: RouteConfig) -> tuple[float, float]:
    default = float(config.edge_cost_threshold) / 2.0
    birth = default if config.birth_cost is None else float(config.birth_cost)
    death = default if config.death_cost is None else float(config.death_cost)
    return birth, death


def _solve_gated_assignment(
    candidates: list[CandidateEdge],
    source_ids: list[int],
    target_ids: list[int],
    config: RouteConfig,
) -> list[CandidateEdge]:
    """Solve one globally optimal assignment with explicit birth/death dummies.

    Real links are hard-gated by ``edge_cost_threshold``.  A source may instead
    take its dedicated death column and a target its dedicated birth row.  The
    lower-right dummy block completes the square assignment without creating a
    biological edge.
    """

    if not source_ids or not target_ids or not candidates:
        return []
    source_ids = sorted(set(int(value) for value in source_ids))
    target_ids = sorted(set(int(value) for value in target_ids))
    source_index = {node_id: index for index, node_id in enumerate(source_ids)}
    target_index = {node_id: index for index, node_id in enumerate(target_ids)}

    best_by_pair: dict[tuple[int, int], CandidateEdge] = {}
    for edge in candidates:
        pair = (int(edge.source_id), int(edge.target_id))
        if pair[0] not in source_index or pair[1] not in target_index:
            continue
        if not np.isfinite(edge.cost) or edge.cost > config.edge_cost_threshold:
            continue
        previous = best_by_pair.get(pair)
        if previous is None or _edge_order_key(edge) < _edge_order_key(previous):
            best_by_pair[pair] = edge
    if not best_by_pair:
        return []

    n_sources = len(source_ids)
    n_targets = len(target_ids)
    size = n_sources + n_targets
    birth_cost, death_cost = _assignment_penalties(config)
    finite_values = [edge.cost for edge in best_by_pair.values()]
    finite_values.extend((birth_cost, death_cost, config.edge_cost_threshold))
    forbidden = max(1_000_000.0, max(abs(float(value)) for value in finite_values) + 1_000_000.0)
    costs = np.full((size, size), forbidden, dtype=np.float64)

    for pair, edge in best_by_pair.items():
        if edge.cost < birth_cost + death_cost:
            costs[source_index[pair[0]], target_index[pair[1]]] = float(edge.cost)
    for row in range(n_sources):
        costs[row, n_targets + row] = death_cost
    for col in range(n_targets):
        costs[n_sources + col, col] = birth_cost
    costs[n_sources:, n_targets:] = 0.0

    rows, cols = linear_sum_assignment(costs)
    selected: list[CandidateEdge] = []
    for row, col in zip(rows, cols, strict=True):
        if row >= n_sources or col >= n_targets:
            continue
        edge = best_by_pair.get((source_ids[row], target_ids[col]))
        if edge is not None:
            selected.append(edge)
    return sorted(selected, key=lambda edge: (edge.source_id, edge.target_id))


def _solve_assignment_components(
    candidates: list[CandidateEdge],
    source_ids: list[int],
    target_ids: list[int],
    config: RouteConfig,
) -> list[CandidateEdge]:
    """Solve independent sparse bipartite components exactly.

    Birth/death costs are node-local and there are no feasible links between
    components, so decomposing the augmented LAP preserves the global optimum
    while avoiding a dense whole-window matrix for gap closing.
    """

    allowed_sources = set(int(value) for value in source_ids)
    allowed_targets = set(int(value) for value in target_ids)
    birth_cost, death_cost = _assignment_penalties(config)
    source_to_targets: dict[int, set[int]] = defaultdict(set)
    target_to_sources: dict[int, set[int]] = defaultdict(set)
    eligible_by_source: dict[int, list[CandidateEdge]] = defaultdict(list)
    for edge in candidates:
        source_id = int(edge.source_id)
        target_id = int(edge.target_id)
        if source_id not in allowed_sources or target_id not in allowed_targets:
            continue
        if (
            not np.isfinite(edge.cost)
            or edge.cost > config.edge_cost_threshold
            or edge.cost >= birth_cost + death_cost
        ):
            continue
        eligible_by_source[source_id].append(edge)
        source_to_targets[source_id].add(target_id)
        target_to_sources[target_id].add(source_id)

    selected: list[CandidateEdge] = []
    unseen_sources = set(source_to_targets)
    while unseen_sources:
        first_source = min(unseen_sources)
        component_sources: set[int] = set()
        component_targets: set[int] = set()
        stack: list[tuple[str, int]] = [("source", first_source)]
        while stack:
            kind, node_id = stack.pop()
            if kind == "source":
                if node_id in component_sources:
                    continue
                component_sources.add(node_id)
                unseen_sources.discard(node_id)
                stack.extend(
                    ("target", target_id)
                    for target_id in sorted(source_to_targets[node_id], reverse=True)
                )
            else:
                if node_id in component_targets:
                    continue
                component_targets.add(node_id)
                stack.extend(
                    ("source", source_id)
                    for source_id in sorted(target_to_sources[node_id], reverse=True)
                )
        component_candidates = [
            edge
            for source_id in sorted(component_sources)
            for edge in eligible_by_source[source_id]
            if edge.target_id in component_targets
        ]
        selected.extend(
            _solve_gated_assignment(
                component_candidates,
                sorted(component_sources),
                sorted(component_targets),
                config,
            )
        )
    return sorted(selected, key=lambda edge: (edge.source_id, edge.target_id))


def _with_gap_motion_cost(
    edge: CandidateEdge,
    nodes_by_id: dict[int, CandidateNode],
    predecessor: dict[int, int],
    scale: np.ndarray,
    config: RouteConfig,
) -> CandidateEdge:
    """Add a bounded constant-velocity residual to a gap candidate."""

    source = nodes_by_id[edge.source_id]
    target = nodes_by_id[edge.target_id]
    previous_id = predecessor.get(source.node_id)
    if previous_id is None or config.gap_motion_cost_weight == 0.0:
        return edge
    previous = nodes_by_id[previous_id]
    history_dt = source.t - previous.t
    if history_dt <= 0:
        return edge
    previous_coord = np.asarray(previous.coord_zyx, dtype=np.float64) * scale
    source_coord = np.asarray(source.coord_zyx, dtype=np.float64) * scale
    target_coord = np.asarray(target.coord_zyx, dtype=np.float64) * scale
    velocity = (source_coord - previous_coord) / float(history_dt)
    predicted = source_coord + velocity * float(edge.dt)
    residual_um = float(np.linalg.norm(target_coord - predicted))
    normalized = min(2.0, residual_um / max(_edge_radius(config, edge.dt), 1e-6))
    motion_penalty = float(config.gap_motion_cost_weight) * normalized
    features = dict(edge.features)
    features.update(
        {
            "gap_motion_residual_um": residual_um,
            "gap_motion_penalty": motion_penalty,
        }
    )
    adjusted_cost = float(edge.cost + motion_penalty)
    adjusted_score = float(1.0 / (1.0 + np.exp(4.0 * (adjusted_cost - 0.55))))
    return replace(edge, cost=adjusted_cost, score=adjusted_score, features=features)


def validate_lineage_graph(
    nodes: list[CandidateNode],
    edges: list[CandidateEdge],
    *,
    allow_divisions: bool,
) -> dict[str, int]:
    """Validate graph identity, time direction, degree, and edge invariants."""

    node_ids = [int(node.node_id) for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("lineage graph contains duplicate node IDs")
    nodes_by_id = {int(node.node_id): node for node in nodes}
    indeg: Counter[int] = Counter()
    outdeg: Counter[int] = Counter()
    children: dict[int, list[int]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for edge in edges:
        pair = (int(edge.source_id), int(edge.target_id))
        if pair in seen:
            raise ValueError(f"lineage graph contains duplicate edge {pair}")
        seen.add(pair)
        source = nodes_by_id.get(pair[0])
        target = nodes_by_id.get(pair[1])
        if source is None or target is None:
            raise ValueError(f"lineage edge endpoint is missing for {pair}")
        actual_dt = int(target.t) - int(source.t)
        if actual_dt <= 0:
            raise ValueError(f"lineage edge {pair} does not advance in time")
        if isinstance(edge.dt, bool) or not isinstance(edge.dt, Integral) or edge.dt <= 0:
            raise ValueError(f"lineage edge {pair} dt must be a positive integer")
        if int(edge.dt) != actual_dt:
            raise ValueError(
                f"lineage edge {pair} declares dt={edge.dt}, but node times imply {actual_dt}"
            )
        indeg[pair[1]] += 1
        outdeg[pair[0]] += 1
        children[pair[0]].append(pair[1])
    if any(value > 1 for value in indeg.values()):
        raise ValueError("lineage graph contains a merge (in-degree greater than one)")
    max_out = 2 if allow_divisions else 1
    if any(value > max_out for value in outdeg.values()):
        raise ValueError(f"lineage graph contains out-degree greater than {max_out}")
    for parent_id, child_ids in children.items():
        if len(child_ids) != 2:
            continue
        parent_t = int(nodes_by_id[parent_id].t)
        child_times = {int(nodes_by_id[child_id].t) for child_id in child_ids}
        if child_times != {parent_t + 1}:
            raise ValueError(
                f"division parent {parent_id} must have exactly two children at the next timepoint"
            )
    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_divisions": sum(value == 2 for value in outdeg.values()),
    }


def _valid_fork(
    parent: CandidateNode,
    child_a: CandidateNode,
    child_b: CandidateNode,
    scale_zyx: tuple[float, float, float],
    config: RouteConfig,
) -> CandidateFork | None:
    scale = np.asarray(scale_zyx, dtype=np.float32)
    p = np.asarray(parent.coord_zyx, dtype=np.float32) * scale
    a = np.asarray(child_a.coord_zyx, dtype=np.float32) * scale
    b = np.asarray(child_b.coord_zyx, dtype=np.float32) * scale
    dist_a = float(np.linalg.norm(a - p))
    dist_b = float(np.linalg.norm(b - p))
    sep = float(np.linalg.norm(a - b))
    if max(dist_a, dist_b) > config.division_child_max_um:
        return None
    if sep < config.division_child_min_sep_um or sep > config.division_child_max_sep_um:
        return None
    score_symmetry = 1.0 - abs(child_a.score - child_b.score)
    distance_score = 1.0 - (dist_a + dist_b) / max(2.0 * config.division_child_max_um, 1e-6)
    separation_score = 1.0 - abs(sep - 5.0) / max(config.division_child_max_sep_um, 1e-6)
    score = 0.35 * parent.score + 0.25 * (child_a.score + child_b.score) + 0.20 * score_symmetry
    score += 0.15 * max(0.0, distance_score) + 0.05 * max(0.0, separation_score)
    if score < config.division_cost_threshold:
        return None
    return CandidateFork(
        parent_id=parent.node_id,
        child_a_id=child_a.node_id,
        child_b_id=child_b.node_id,
        score=float(score),
        parent_child_mean_um=float((dist_a + dist_b) / 2.0),
        daughter_separation_um=sep,
    )


def select_edges(
    nodes_by_t: list[list[CandidateNode]],
    edge_candidates: list[CandidateEdge],
    scale_zyx: tuple[float, float, float],
    config: RouteConfig,
) -> tuple[list[CandidateEdge], list[CandidateFork]]:
    """Select a coherent lineage graph with LAP links then tracklet gap closing."""

    _validate_config(config)
    scale = _validated_scale(scale_zyx)
    all_nodes = [node for nodes in nodes_by_t for node in nodes]
    validate_lineage_graph(all_nodes, [], allow_divisions=config.enable_divisions)
    nodes_by_id = {node.node_id: node for node in all_nodes}
    _validate_edge_candidates(edge_candidates, nodes_by_id)
    selected: list[CandidateEdge] = []
    selected_pairs: set[tuple[int, int]] = set()
    indeg: Counter[int] = Counter()
    outdeg: Counter[int] = Counter()

    # Stage 1: globally optimal one-to-one assignment for each adjacent pair.
    for source_nodes, target_nodes in zip(nodes_by_t[:-1], nodes_by_t[1:], strict=True):
        source_ids = [node.node_id for node in source_nodes]
        target_ids = [node.node_id for node in target_nodes]
        source_set = set(source_ids)
        target_set = set(target_ids)
        candidates = [
            edge
            for edge in edge_candidates
            if edge.dt == 1
            and edge.source_id in source_set
            and edge.target_id in target_set
        ]
        for edge in _solve_assignment_components(
            candidates, source_ids, target_ids, config
        ):
            selected.append(edge)
            selected_pairs.add((edge.source_id, edge.target_id))
            indeg[edge.target_id] += 1
            outdeg[edge.source_id] += 1

    # Stage 2: close gaps only between complete tracklet endpoints and starts.
    if int(config.max_gap) > 1:
        predecessor = {edge.target_id: edge.source_id for edge in selected}
        gap_candidates: list[CandidateEdge] = []
        for edge in edge_candidates:
            if edge.dt <= 1 or edge.dt > int(config.max_gap):
                continue
            source = nodes_by_id.get(edge.source_id)
            target = nodes_by_id.get(edge.target_id)
            if source is None or target is None or target.t - source.t != edge.dt:
                continue
            if outdeg[edge.source_id] or indeg[edge.target_id]:
                continue
            gap_candidates.append(
                _with_gap_motion_cost(edge, nodes_by_id, predecessor, scale, config)
            )
        gap_source_ids = sorted({edge.source_id for edge in gap_candidates})
        gap_target_ids = sorted({edge.target_id for edge in gap_candidates})
        for edge in _solve_assignment_components(
            gap_candidates, gap_source_ids, gap_target_ids, config
        ):
            if outdeg[edge.source_id] or indeg[edge.target_id]:
                continue
            selected.append(edge)
            selected_pairs.add((edge.source_id, edge.target_id))
            indeg[edge.target_id] += 1
            outdeg[edge.source_id] += 1

    forks: list[CandidateFork] = []
    if not config.enable_divisions:
        selected.sort(
            key=lambda edge: (
                nodes_by_id[edge.source_id].t,
                nodes_by_id[edge.target_id].t,
                edge.source_id,
                edge.target_id,
            )
        )
        validate_lineage_graph(all_nodes, selected, allow_divisions=False)
        return selected, forks

    max_t = max((node.t for node in nodes_by_id.values()), default=0)
    future_support: set[int] = set()
    if config.division_requires_child_future:
        for edge in edge_candidates:
            source = nodes_by_id.get(edge.source_id)
            if source is None:
                continue
            if edge.dt == 1 and edge.cost <= config.edge_cost_threshold and source.t < max_t:
                future_support.add(edge.source_id)

    children_by_parent: dict[int, list[int]] = defaultdict(list)
    for edge in edge_candidates:
        if edge.dt == 1 and edge.distance_um <= config.division_child_max_um:
            children_by_parent[edge.source_id].append(edge.target_id)

    for parent_id in sorted(children_by_parent):
        child_ids = sorted(set(children_by_parent[parent_id]))
        parent = nodes_by_id[parent_id]
        if outdeg[parent_id] >= 2:
            continue
        existing_children = sorted(
            target for source, target in selected_pairs if source == parent_id
        )
        if any(nodes_by_id[child_id].t != parent.t + 1 for child_id in existing_children):
            continue
        if config.division_requires_unlinked_parent and existing_children:
            continue
        candidates = [cid for cid in child_ids if cid not in existing_children]
        if existing_children and not config.division_requires_unlinked_parent:
            pairs = [(existing_children[0], cid) for cid in candidates]
        else:
            pairs = itertools.combinations(candidates, 2)
        valid: list[CandidateFork] = []
        for child_a_id, child_b_id in pairs:
            if child_a_id == child_b_id:
                continue
            if nodes_by_id[child_a_id].score < 0.85 or nodes_by_id[child_b_id].score < 0.85 or parent.score < 0.80:
                continue
            if config.division_requires_child_future:
                if nodes_by_id[child_a_id].t < max_t and child_a_id not in future_support:
                    continue
                if nodes_by_id[child_b_id].t < max_t and child_b_id not in future_support:
                    continue
            if indeg[child_a_id] >= 1 and child_a_id not in existing_children:
                continue
            if indeg[child_b_id] >= 1 and child_b_id not in existing_children:
                continue
            fork = _valid_fork(parent, nodes_by_id[child_a_id], nodes_by_id[child_b_id], scale_zyx, config)
            if fork is not None:
                valid.append(fork)
        if not valid:
            continue
        fork = min(
            valid,
            key=lambda value: (
                -value.score,
                value.parent_id,
                value.child_a_id,
                value.child_b_id,
            ),
        )
        for child_id in (fork.child_a_id, fork.child_b_id):
            pair = (fork.parent_id, child_id)
            if pair in selected_pairs:
                continue
            if indeg[child_id] >= 1 or outdeg[fork.parent_id] >= 2:
                continue
            child = nodes_by_id[child_id]
            dist = float(np.linalg.norm((np.asarray(parent.coord_zyx) - np.asarray(child.coord_zyx)) * np.asarray(scale_zyx)))
            selected.append(
                CandidateEdge(
                    source_id=fork.parent_id,
                    target_id=child_id,
                    dt=1,
                    distance_um=dist,
                    cost=1.0 - fork.score,
                    score=fork.score,
                    kind="division",
                )
            )
            selected_pairs.add(pair)
            indeg[child_id] += 1
            outdeg[fork.parent_id] += 1
        if outdeg[fork.parent_id] == 2:
            forks.append(fork)

    selected.sort(
        key=lambda edge: (
            nodes_by_id[edge.source_id].t,
            nodes_by_id[edge.target_id].t,
            edge.source_id,
            edge.target_id,
        )
    )
    validate_lineage_graph(all_nodes, selected, allow_divisions=True)
    return selected, forks


def graph_diagnostics(nodes: list[CandidateNode], edges: list[CandidateEdge]) -> dict[str, int | float]:
    indeg = Counter(edge.target_id for edge in edges)
    outdeg = Counter(edge.source_id for edge in edges)
    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_dividing_nodes": sum(1 for v in outdeg.values() if v >= 2),
        "num_out_gt2": sum(1 for v in outdeg.values() if v > 2),
        "num_merges": sum(1 for v in indeg.values() if v > 1),
        "mean_node_score": float(np.mean([node.score for node in nodes])) if nodes else float("nan"),
        "mean_edge_score": float(np.mean([edge.score for edge in edges])) if edges else float("nan"),
    }
