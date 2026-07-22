from __future__ import annotations

import csv
from collections import Counter
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
from pathlib import Path
import random
import unittest
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

from celltrack_alpha.bh_graph import (
    EXPERIMENTAL_SWEEP_SCHEMA,
    PHYSICAL_SCALE_ZYX,
    ErlangDivisionPrior,
    GraphEdge,
    GraphInvariantError,
    GraphNode,
    SparseAssignmentError,
    V45_GRAPH_CONFIG,
    add_safe_divisions,
    build_adjacent_edges,
    calibrate_v45_graph,
    close_one_frame_gaps,
    gap_step_limit_um,
    intensity_centroid_refine,
    learned_edge_score_bonus,
    line_fit_smooth,
    motion_only_edge_cost,
    physical_distance,
    postprocess_graph,
    prune_graph,
    sparse_min_cost_max_cardinality,
    sparse_min_cost_partial_matching,
    validate_graph,
)


ROOT = Path(__file__).resolve().parents[1]
X_SCALE = PHYSICAL_SCALE_ZYX[2]


def _node_um(node_id: int, t: int, x_um: float, y_um: float = 0.0) -> GraphNode:
    return GraphNode(node_id, t, 0.0, y_um / PHYSICAL_SCALE_ZYX[1], x_um / X_SCALE)


def _edge(source: GraphNode, target: GraphNode, kind: str = "adjacent") -> GraphEdge:
    distance = physical_distance(source, target)
    return GraphEdge(source.node_id, target.node_id, distance, distance, kind)


def _lineage_batch(
    count: int,
    frame: int,
    first_id: int,
    x_offset_um: float,
) -> tuple[list[GraphNode], list[GraphEdge], list[GraphNode]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    parents: list[GraphNode] = []
    for index in range(count):
        parent = _node_um(first_id + 2 * index, frame, x_offset_um + 20.0 * index)
        child = _node_um(first_id + 2 * index + 1, frame + 1, x_offset_um + 20.0 * index + 1.0)
        nodes.extend((parent, child))
        edges.append(_edge(parent, child))
        parents.append(parent)
    return nodes, edges, parents


def _division_batch(
    count: int,
    *,
    crowded_parents: int = 0,
) -> tuple[list[GraphNode], list[GraphEdge], list[GraphNode]]:
    """Build enough one-child lineages to exercise public division caps > 1."""

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    parents: list[GraphNode] = []
    for index in range(count):
        x_um = 0.0 if index < max(1, crowded_parents) else 100.0 + 20.0 * index
        parent = _node_um(2 * index, 0, x_um)
        child = _node_um(2 * index + 1, 1, x_um + 1.0)
        nodes.extend((parent, child))
        edges.append(_edge(parent, child))
        parents.append(parent)
    return nodes, edges, parents


def _degree_counts(edges: list[GraphEdge] | tuple[GraphEdge, ...]) -> tuple[Counter[int], Counter[int]]:
    indegree = Counter(edge.target_id for edge in edges)
    outdegree = Counter(edge.source_id for edge in edges)
    return indegree, outdegree


def _dense_public_motion_oracle(
    nodes: list[GraphNode],
    probabilities: dict[tuple[int, int], float],
) -> set[tuple[int, int, str]]:
    """Diagnostic-only full-frame replay of public V45's dense LAPJV passes."""

    node_by_id = {node.node_id: node for node in nodes}
    ids_by_t: dict[int, list[int]] = {}
    for node in nodes:
        ids_by_t.setdefault(node.t, []).append(node.node_id)
    for ids in ids_by_t.values():
        ids.sort()
    positions = {
        node.node_id: np.asarray(node.coord_zyx, dtype=np.float64)
        * np.asarray(PHYSICAL_SCALE_ZYX, dtype=np.float64)
        for node in nodes
    }
    predecessor_positions: dict[int, np.ndarray] = {}
    selected: set[tuple[int, int, str]] = set()
    for frame in sorted(ids_by_t):
        source_ids = ids_by_t.get(frame, [])
        target_ids = ids_by_t.get(frame + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches: list[tuple[int, int, str]] = []
        for label, gate_um in (("motion_tight", 6.0), ("motion_relaxed", 10.0)):
            pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
            pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
            if not pass_sources or not pass_targets:
                continue
            source_positions = np.stack([positions[node_id] for node_id in pass_sources])
            target_positions = np.stack([positions[node_id] for node_id in pass_targets])
            raw = np.linalg.norm(
                source_positions[:, None, :] - target_positions[None, :, :], axis=2
            )
            predicted = np.stack(
                [
                    source_position
                    if (previous := predecessor_positions.get(node_id)) is None
                    else source_position + 0.5 * (source_position - previous)
                    for node_id, source_position in zip(
                        pass_sources, source_positions, strict=True
                    )
                ]
            )
            motion = np.linalg.norm(
                predicted[:, None, :] - target_positions[None, :, :], axis=2
            )
            big = gate_um * 1000.0 + 1.0
            cost = np.full(raw.shape, big, dtype=np.float64)
            feasible_rows, feasible_columns = np.nonzero(raw <= gate_um)
            for row, column in zip(feasible_rows, feasible_columns, strict=True):
                source_id = pass_sources[int(row)]
                target_id = pass_targets[int(column)]
                cost[row, column] = (
                    motion[row, column]
                    + 0.05 * raw[row, column]
                    - probabilities.get((source_id, target_id), 0.0)
                )
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
                if cost[row, column] >= big:
                    continue
                source_id = pass_sources[row]
                target_id = pass_targets[column]
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append((source_id, target_id, label))
        for source_id, target_id, label in frame_matches:
            selected.add((source_id, target_id, label))
            predecessor_positions[target_id] = positions[source_id]
    return selected


class V45ConfigTests(unittest.TestCase):
    def test_locked_exact_public_defaults(self) -> None:
        config = V45_GRAPH_CONFIG
        self.assertEqual(config.scale_zyx, (1.625, 0.40625, 0.40625))
        self.assertEqual(config.output_edge_max_um, 14.0)
        self.assertEqual((config.motion_tight_um, config.motion_relaxed_um), (6.0, 10.0))
        self.assertEqual((config.velocity_weight, config.learned_bonus), (0.5, 1.0))
        self.assertEqual(config.motion_direction, "backward")
        self.assertEqual(config.motion_error_weight, 0.25)
        self.assertEqual(config.raw_distance_weight, 0.05)
        self.assertEqual((config.gap_step_um, config.gap_total_um), (5.8, 11.6))
        self.assertEqual((config.density_gain, config.density_delta_clip_um), (0.0475, 0.125))
        self.assertEqual((config.gap_add_fraction, config.gap_add_absolute), (0.05, 2000))
        self.assertEqual(config.midpoint_reuse_um, 3.2)
        self.assertFalse(config.gap2_enabled)
        self.assertEqual(
            (
                config.division_candidate_child_um,
                config.division_existing_child_um,
                config.division_sister_um,
            ),
            (4.66, 7.65, 8.5),
        )
        self.assertEqual((config.division_frame_fraction, config.division_global_fraction), (0.0076, 0.00375))
        self.assertFalse(config.division_geometry_filter)
        self.assertEqual((config.min_component_nodes, config.smooth_window), (6, 2))
        self.assertEqual(config.smooth_weight, 0.8)
        with self.assertRaisesRegex(ValueError, "locked"):
            replace(config, output_edge_max_um=13.0)
        with self.assertRaises(FrozenInstanceError):
            config.output_edge_max_um = 13.0  # type: ignore[misc]

    def test_nondefault_config_requires_explicit_offline_sweep_provenance(self) -> None:
        class SweepConfig:
            def __init__(self) -> None:
                for name, value in V45_GRAPH_CONFIG.as_dict().items():
                    setattr(self, name, tuple(value) if name == "scale_zyx" else value)
                self.learned_bonus = 2.0

            def as_dict(self) -> dict[str, object]:
                return {
                    name: list(value) if isinstance(value, tuple) else value
                    for name, value in vars(self).items()
                }

        nodes = [_node_um(0, 0, 0.0), _node_um(1, 1, 1.0)]
        with self.assertRaisesRegex(ValueError, "requires sweep provenance"):
            calibrate_v45_graph(nodes, config=SweepConfig())  # type: ignore[arg-type]
        result = calibrate_v45_graph(
            nodes,
            config=SweepConfig(),  # type: ignore[arg-type]
            _experimental_sweep_provenance={
                "schema": EXPERIMENTAL_SWEEP_SCHEMA,
                "experiment_id": "unit-test",
                "purpose": "offline-cached-cv57-sweep",
                "varied_fields": ["learned_bonus"],
                "production_eligible": False,
                "submit_eligible": False,
            },
        )
        self.assertFalse(result.provenance["production_eligible"])
        self.assertEqual(result.provenance["config"]["learned_bonus"], 2.0)

    def test_anisotropic_physical_distance(self) -> None:
        self.assertAlmostEqual(physical_distance((0, 0, 0), (1, 0, 0)), 1.625)
        self.assertAlmostEqual(physical_distance((0, 0, 0), (0, 4, 0)), 1.625)
        self.assertAlmostEqual(
            physical_distance((0, 0, 0), (1, 4, 0)), math.sqrt(2.0) * 1.625
        )


class SparseAssignmentTests(unittest.TestCase):
    def test_partial_assignment_can_choose_explicit_no_link(self) -> None:
        partial = sparse_min_cost_partial_matching(
            [1], [10], [(1, 10, -math.log(0.4))], {1: -math.log(0.6)}
        )
        legacy = sparse_min_cost_max_cardinality([1], [10], [(1, 10, 0.0)])
        self.assertEqual(partial.matches, ())
        self.assertEqual([(match.source_id, match.target_id) for match in legacy.matches], [(1, 10)])

    def test_partial_assignment_is_input_order_independent(self) -> None:
        pairs = [(2, 20, -math.log(0.6)), (1, 20, -math.log(0.3)), (1, 10, -math.log(0.6))]
        first = sparse_min_cost_partial_matching([1, 2], [10, 20], pairs, {1: -math.log(0.1), 2: -math.log(0.4)})
        second = sparse_min_cost_partial_matching([2, 1], [20, 10], reversed(pairs), {2: -math.log(0.4), 1: -math.log(0.1)})
        self.assertEqual(first.matches, second.matches)

    def test_maximum_cardinality_beats_greedy_counterexample(self) -> None:
        result = sparse_min_cost_max_cardinality(
            [1, 2],
            [10, 20],
            [(1, 10, 0.0), (1, 20, 1.0), (2, 10, 0.1)],
        )
        self.assertEqual(
            [(match.source_id, match.target_id) for match in result.matches],
            [(1, 20), (2, 10)],
        )

    def test_equal_cost_ties_are_input_order_independent(self) -> None:
        pairs = [(2, 20, 0.0), (1, 20, 0.0), (2, 10, 0.0), (1, 10, 0.0)]
        forward = sparse_min_cost_max_cardinality([2, 1], [20, 10], pairs)
        reverse = sparse_min_cost_max_cardinality([1, 2], [10, 20], reversed(pairs))
        self.assertEqual(forward.matches, reverse.matches)
        self.assertEqual(
            [(match.source_id, match.target_id) for match in forward.matches],
            [(1, 10), (2, 20)],
        )

    def test_single_source_tie_matches_dense_target_priority(self) -> None:
        pairs = [(12491, 13140, 8.75089281159357), (12491, 13156, 8.75089281159357)]
        sparse = sparse_min_cost_max_cardinality(
            [1, 2, 12491],
            [13140, 13156],
            pairs,
            dense_unmatched_cost=11851.0,
        )
        dense = sparse_min_cost_max_cardinality(
            [1, 2, 12491],
            [13140, 13156],
            pairs,
            parity_mode="dense_diagnostic",
            dense_unmatched_cost=11851.0,
        )
        self.assertEqual(sparse.matches, dense.matches)
        self.assertEqual(
            [(match.source_id, match.target_id) for match in sparse.matches],
            [(12491, 13140)],
        )
        self.assertEqual(sparse.peak_component_matrix_dim, 3)

    def test_randomized_sparse_parity_with_dense_lsa(self) -> None:
        rng = np.random.default_rng(335810749)
        for _ in range(80):
            left_count = int(rng.integers(1, 7))
            right_count = int(rng.integers(1, 7))
            sources = list(range(100, 100 + left_count))
            targets = list(range(900, 900 + right_count))
            mask = rng.random((left_count, right_count)) < 0.46
            if not mask.any():
                mask[int(rng.integers(left_count)), int(rng.integers(right_count))] = True
            raw_costs = rng.integers(-4, 5, size=(left_count, right_count)).astype(float)
            pairs = [
                (sources[row], targets[column], float(raw_costs[row, column]))
                for row, column in zip(*np.nonzero(mask), strict=True)
            ]
            sparse = sparse_min_cost_max_cardinality(sources, targets, reversed(pairs))

            max_abs = max(1.0, max(abs(cost) for _, _, cost in pairs))
            big = 100.0 * (min(left_count, right_count) + 1) * max_abs
            dense = np.full((left_count, right_count), big, dtype=np.float64)
            for source, target, cost in pairs:
                dense[sources.index(source), targets.index(target)] = cost
            rows, columns = linear_sum_assignment(dense)
            dense_selected = [
                (sources[row], targets[column], float(dense[row, column]))
                for row, column in zip(rows.tolist(), columns.tolist(), strict=True)
                if dense[row, column] < big
            ]
            self.assertEqual(len(sparse.matches), len(dense_selected))
            self.assertAlmostEqual(
                sum(match.cost for match in sparse.matches),
                sum(cost for _, _, cost in dense_selected),
            )

    def test_connected_component_larger_than_former_512_cap(self) -> None:
        count = 260
        sources = list(range(count))
        targets = list(range(1000, 1000 + count))
        pairs = [(index, 1000 + index, 0.0) for index in range(count)]
        pairs.extend((index, 1000 + index + 1, 1.0) for index in range(count - 1))
        result = sparse_min_cost_max_cardinality(
            sources,
            targets,
            pairs,
            max_component_matrix_dim=3,
        )
        self.assertEqual(len(result.matches), count)
        self.assertEqual(result.peak_component_matrix_dim, 520)
        self.assertEqual(
            [(match.source_id, match.target_id) for match in result.matches],
            [(index, 1000 + index) for index in range(count)],
        )

    def test_malformed_legacy_cap_still_fails(self) -> None:
        with self.assertRaises(SparseAssignmentError):
            sparse_min_cost_max_cardinality([1], [2], [(1, 2, 0.0)], max_component_matrix_dim=0)


class MotionTests(unittest.TestCase):
    def test_scored_partial_uses_no_link_and_all_distances_in_one_component(self) -> None:
        source_a = _node_um(1, 0, 0.0)
        source_b = _node_um(2, 0, 6.0)
        target_a = _node_um(10, 1, 0.0)
        target_b = _node_um(20, 1, 7.0)
        probabilities = {
            (1, 10): 0.4,
            (1, 20): 0.5,
            (2, 10): 0.6,
            (2, 20): 0.0,
        }
        legacy, _ = build_adjacent_edges([source_a, source_b, target_a, target_b], probabilities)
        partial, stats = build_adjacent_edges(
            [source_b, target_b, source_a, target_a],
            probabilities,
            {1: 0.1, 2: 0.4},
            topology_mode="scored_partial",
        )
        self.assertEqual(
            [(edge.source_id, edge.target_id) for edge in legacy], [(1, 10), (2, 20)]
        )
        self.assertEqual(
            [(edge.source_id, edge.target_id) for edge in partial], [(1, 20), (2, 10)]
        )
        self.assertEqual(stats["scored_partial_components"], 1)
        self.assertEqual(stats["scored_partial_fallback_components"], 0)
        self.assertEqual(stats["scored_partial_zero_probability_pairs"], 1)
        self.assertEqual(stats["scored_partial_edges"], 2)

    def test_scored_partial_falls_back_per_incomplete_component(self) -> None:
        source_a = _node_um(1, 0, 0.0)
        source_b = _node_um(2, 0, 6.0)
        target_a = _node_um(10, 1, 0.0)
        target_b = _node_um(20, 1, 7.0)
        complete = {(1, 10): 0.4, (1, 20): 0.5, (2, 10): 0.6, (2, 20): 0.0}
        incomplete = {key: value for key, value in complete.items() if key != (1, 20)}
        legacy, _ = build_adjacent_edges([source_a, source_b, target_a, target_b], incomplete)
        partial, stats = build_adjacent_edges(
            [source_a, source_b, target_a, target_b],
            incomplete,
            {1: 0.1, 2: 0.4},
            topology_mode="scored_partial",
        )
        self.assertEqual(partial, legacy)
        self.assertEqual(stats["scored_partial_components"], 0)
        self.assertEqual(stats["scored_partial_fallback_components"], 1)

    def test_single_pair_uses_banked_quarter_weight(self) -> None:
        source = _node_um(0, 0, 0.0)
        target = _node_um(1, 1, 5.5)
        edges, stats = build_adjacent_edges([source, target])
        self.assertEqual([(edge.source_id, edge.target_id, edge.kind) for edge in edges], [(0, 1, "motion_tight")])
        self.assertAlmostEqual(edges[0].cost, 0.25 * 5.5 + 0.05 * 5.5)
        self.assertEqual(stats["motion_tight_edges"], 1)

    def test_raw_distance_gates_while_velocity_changes_cost_only(self) -> None:
        parent = _node_um(0, 0, 0.0)
        source = _node_um(1, 1, 10.0)
        raw_inside = _node_um(2, 2, 1.0)
        edges, _ = build_adjacent_edges([parent, source, raw_inside])
        by_pair = {(edge.source_id, edge.target_id): edge for edge in edges}
        edge = by_pair[(1, 2)]
        self.assertEqual(edge.kind, "motion_relaxed")
        self.assertAlmostEqual(edge.distance_um, 9.0)
        self.assertAlmostEqual(edge.cost, 0.25 * 9.0 + 0.05 * 9.0)

    def test_low_motion_error_does_not_promote_raw_seven_um_to_tight(self) -> None:
        parent = _node_um(0, 0, 0.0)
        source = _node_um(1, 1, 10.0)
        target = _node_um(2, 2, 17.0)
        edges, _ = build_adjacent_edges([parent, source, target])
        by_pair = {(edge.source_id, edge.target_id): edge for edge in edges}
        self.assertEqual(by_pair[(1, 2)].kind, "motion_relaxed")
        self.assertAlmostEqual(by_pair[(1, 2)].cost, 0.25 * 7.0 + 0.35)

    def test_fourteen_um_is_output_validation_not_a_relink_pass(self) -> None:
        source = _node_um(0, 0, 0.0)
        target = _node_um(1, 1, 12.0)
        edges, stats = build_adjacent_edges([source, target], {(0, 1): 1.0})
        self.assertEqual(edges, ())
        self.assertEqual(stats["adjacent_candidate_pairs"], 0)

    def test_optional_learned_bonus_is_separate(self) -> None:
        self.assertAlmostEqual(motion_only_edge_cost(4.0), 0.2)
        self.assertEqual(learned_edge_score_bonus(None), 0.0)
        self.assertAlmostEqual(learned_edge_score_bonus(0.75), 0.75)


class GapClosingTests(unittest.TestCase):
    def test_no_input_edges_means_no_gap_pass(self) -> None:
        nodes = [_node_um(0, 0, 0.0), _node_um(1, 2, 8.0)]
        result_nodes, result_edges, stats = close_one_frame_gaps(nodes, [])
        self.assertEqual(result_nodes, tuple(nodes))
        self.assertEqual(result_edges, ())
        self.assertEqual(stats["gap_matches"], 0)

    def test_candidate_radius_covers_adaptive_11_85_um_limit(self) -> None:
        source = _node_um(0, 0, 0.0)
        target = _node_um(1, 2, 11.7)
        anchor_source = _node_um(2, 0, 100.0)
        anchor_child = _node_um(3, 1, 100.0)
        target_spacing = _node_um(4, 2, 100.0)
        result_nodes, result_edges, stats = close_one_frame_gaps(
            [source, target, anchor_source, anchor_child, target_spacing],
            [_edge(anchor_source, anchor_child)],
        )
        self.assertEqual(stats["gap_matches"], 1)
        self.assertEqual(stats["gap_synthesized_nodes"], 1)
        self.assertEqual(len(result_nodes), 6)
        self.assertEqual(len(result_edges), 3)

    def test_reuse_chooses_nearest_isolated_without_per_leg_rejection(self) -> None:
        source = _node_um(0, 0, 0.0)
        target = _node_um(1, 2, 8.0)
        isolated = _node_um(2, 1, 7.0)  # 3 um from midpoint; source leg is 7 um.
        anchor_source = _node_um(3, 0, 100.0)
        anchor_child = _node_um(4, 1, 100.0)
        result_nodes, result_edges, stats = close_one_frame_gaps(
            [source, target, isolated, anchor_source, anchor_child],
            [_edge(anchor_source, anchor_child)],
        )
        self.assertEqual(len(result_nodes), 5)
        self.assertEqual(stats["gap_reused_nodes"], 1)
        self.assertEqual(
            {(edge.source_id, edge.target_id, edge.kind) for edge in result_edges if edge.kind == "gap_reuse"},
            {(0, 2, "gap_reuse"), (2, 1, "gap_reuse")},
        )

    def test_synthesis_cap_uses_public_round_and_minimum_one(self) -> None:
        anchor_source = _node_um(0, 0, 0.0)
        anchor_child = _node_um(1, 1, 0.0)
        nodes = [anchor_source, anchor_child]
        nodes.extend(_node_um(2 + index, 1, 100.0 + index) for index in range(28))
        _, _, stats = close_one_frame_gaps(nodes, [_edge(anchor_source, anchor_child)])
        self.assertEqual(stats["gap_synthesis_cap"], 2)

    def test_refinement_shift_is_not_subject_to_extra_leg_gates(self) -> None:
        source = _node_um(0, 0, 0.0)
        target = _node_um(1, 2, 8.0)
        anchor_source = _node_um(2, 0, 100.0)
        anchor_child = _node_um(3, 1, 100.0)

        def shifted(frame: int, coord: tuple[float, float, float]) -> tuple[float, float, float]:
            self.assertEqual(frame, 1)
            return coord[0], coord[1], coord[2] + 3.0 / X_SCALE

        result_nodes, _, stats = close_one_frame_gaps(
            [source, target, anchor_source, anchor_child],
            [_edge(anchor_source, anchor_child)],
            image_refine_callback=shifted,
        )
        synthetic = next(node for node in result_nodes if node.synthetic)
        self.assertAlmostEqual(synthetic.x * X_SCALE, 7.0)
        self.assertEqual(stats["gap_refinement_accepted"], 1)

    def test_callback_shift_over_3_2_um_is_rejected(self) -> None:
        source = _node_um(0, 0, 0.0)
        target = _node_um(1, 2, 8.0)
        anchor_source = _node_um(2, 0, 100.0)
        anchor_child = _node_um(3, 1, 100.0)

        def far(frame: int, coord: tuple[float, float, float]) -> tuple[float, float, float]:
            return coord[0], coord[1], coord[2] + 4.0 / X_SCALE

        result_nodes, _, stats = close_one_frame_gaps(
            [source, target, anchor_source, anchor_child],
            [_edge(anchor_source, anchor_child)],
            image_refine_callback=far,
        )
        synthetic = next(node for node in result_nodes if node.synthetic)
        self.assertAlmostEqual(synthetic.x * X_SCALE, 4.0)
        self.assertEqual(stats["gap_refinement_rejected"], 1)

    def test_density_adjustment_and_p20_centroid(self) -> None:
        adjusted, delta = gap_step_limit_um(4.0625, 4.0625)
        self.assertLess(delta, 0.0)
        self.assertLess(adjusted, 5.8)
        image = np.zeros((3, 7, 7), dtype=np.float32)
        image[1, 3, 4] = 8.0
        before = image.copy()
        self.assertEqual(intensity_centroid_refine(image, (1, 3, 3)), (1.0, 3.0, 4.0))
        np.testing.assert_array_equal(image, before)


class DivisionTests(unittest.TestCase):
    def test_erlang_negative_log_density_matches_closed_form(self) -> None:
        prior = ErlangDivisionPrior(shape=3, rate=0.25)
        expected = (
            math.lgamma(3)
            - 3 * math.log(0.25)
            - 2 * math.log(8)
            + 0.25 * 8
        )
        self.assertAlmostEqual(prior.negative_log_density(8), expected)
        self.assertTrue(math.isinf(prior.negative_log_density(0)))
        self.assertAlmostEqual(
            ErlangDivisionPrior(shape=1, rate=0.25).negative_log_density(0),
            -math.log(0.25),
        )

    def test_erlang_opening_cost_can_change_capped_division_choice(self) -> None:
        young_root = _node_um(1, 9, 0.0)
        young_parent = _node_um(2, 10, 0.0)
        old_track = [_node_um(100 + frame, frame, 20.0) for frame in range(2, 11)]
        young_existing = _node_um(3, 11, 1.0)
        young_orphan = _node_um(4, 11, 1.5)
        old_existing = _node_um(200, 11, 21.0)
        old_orphan = _node_um(201, 11, 22.0)
        nodes = [
            young_root,
            young_parent,
            *old_track,
            young_existing,
            young_orphan,
            old_existing,
            old_orphan,
        ]
        edges = [_edge(young_root, young_parent), _edge(young_parent, young_existing)]
        edges.extend(_edge(old_track[index], old_track[index + 1]) for index in range(8))
        edges.append(_edge(old_track[-1], old_existing))

        without_prior, _ = add_safe_divisions(nodes, edges)
        with_prior, stats = add_safe_divisions(
            nodes,
            edges,
            lifetime_prior=ErlangDivisionPrior(shape=5, rate=0.5),
        )
        self.assertEqual(
            [(edge.source_id, edge.target_id) for edge in without_prior if edge.kind == "division"],
            [(young_parent.node_id, young_orphan.node_id)],
        )
        self.assertEqual(
            [(edge.source_id, edge.target_id) for edge in with_prior if edge.kind == "division"],
            [(old_track[-1].node_id, old_orphan.node_id)],
        )
        self.assertTrue(stats["division_lifetime_prior_enabled"])
        self.assertEqual(stats["division_lifetime_shape"], 5)

    def test_exact_geometry_without_disabled_angle_filter(self) -> None:
        parent = _node_um(0, 0, 0.0)
        existing = _node_um(1, 1, 7.65)
        candidate = _node_um(2, 1, 4.66)
        outside = _node_um(3, 1, 4.661)
        result, stats = add_safe_divisions(
            [parent, existing, candidate, outside], [_edge(parent, existing)]
        )
        divisions = [edge for edge in result if edge.kind == "division"]
        self.assertEqual([(edge.source_id, edge.target_id) for edge in divisions], [(0, 2)])
        self.assertAlmostEqual(divisions[0].cost, 4.66 + 0.15 * (7.65 - 4.66))
        self.assertEqual(stats["division_global_cap"], 1)

    def test_global_cap_uses_edge_count_and_public_round(self) -> None:
        nodes, edges, _ = _lineage_batch(400, 0, 0, 0.0)
        _, stats = add_safe_divisions(nodes, edges)
        self.assertEqual(stats["division_global_cap"], 2)

    def test_per_frame_cap_uses_one_child_source_count_and_round(self) -> None:
        first_nodes, first_edges, first_parents = _lineage_batch(200, 0, 0, 0.0)
        second_nodes, second_edges, _ = _lineage_batch(600, 10, 10000, 10000.0)
        candidates = [
            _node_um(30000 + index, 1, 20.0 * index + 2.0)
            for index in range(3)
        ]
        result, stats = add_safe_divisions(
            [*first_nodes, *second_nodes, *candidates],
            [*first_edges, *second_edges],
        )
        divisions = [edge for edge in result if edge.kind == "division"]
        self.assertEqual(stats["division_global_cap"], 3)
        self.assertEqual(len(first_parents), 200)
        self.assertEqual(len(divisions), 2)

    def test_frames_are_greedy_before_global_cost_order(self) -> None:
        early_nodes, early_edges, _ = _lineage_batch(1, 0, 0, 0.0)
        late_nodes, late_edges, late_parents = _lineage_batch(266, 2, 1000, 1000.0)
        early_candidate = _node_um(90000, 1, 4.0)
        late_candidate = _node_um(90001, 3, 1000.5)
        result, stats = add_safe_divisions(
            [*early_nodes, *late_nodes, early_candidate, late_candidate],
            [*early_edges, *late_edges],
        )
        divisions = [edge for edge in result if edge.kind == "division"]
        self.assertEqual(stats["division_global_cap"], 1)
        self.assertEqual((divisions[0].source_id, divisions[0].target_id), (0, 90000))
        self.assertEqual(late_parents[0].node_id, 1000)

    def test_bh9_v5_6bba_parent_residual_capacity_regression(self) -> None:
        evidence = json.loads(
            (ROOT / "kaggle_kernel/bh9o5/bh9fail.json").read_text(encoding="utf-8")
        )
        failure = evidence["error"]["message"]
        self.assertIn('"identity":"train:6bba_ebff6e76"', failure)
        self.assertIn("GraphInvariantError: outdegree exceeds two", failure)

        nodes, edges, parents = _division_batch(400)
        orphans = [_node_um(10000 + index, 1, 2.0 + index) for index in range(3)]
        result, stats = add_safe_divisions([*nodes, *orphans], edges)
        divisions = [edge for edge in result if edge.kind == "division"]
        self.assertGreater(stats["division_global_cap"], 1)
        self.assertEqual(
            [(edge.source_id, edge.target_id) for edge in divisions],
            [(parents[0].node_id, orphans[0].node_id)],
        )
        self.assertEqual(stats["division_parent_cap_rejected"], 2)
        self.assertLessEqual(
            stats["division_parent_cap_rejected"], stats["division_candidate_count"]
        )
        validate_graph([*nodes, *orphans], result)

    def test_multiple_parents_compete_without_reusing_targets_or_parents(self) -> None:
        nodes, edges, parents = _division_batch(400, crowded_parents=2)
        orphans = [_node_um(10000 + index, 1, 2.0) for index in range(3)]
        result, stats = add_safe_divisions([*nodes, *orphans], edges)
        divisions = [edge for edge in result if edge.kind == "division"]
        self.assertEqual(
            [(edge.source_id, edge.target_id) for edge in divisions],
            [(parents[0].node_id, orphans[0].node_id), (parents[1].node_id, orphans[1].node_id)],
        )
        self.assertEqual(len({edge.target_id for edge in divisions}), len(divisions))
        self.assertGreater(stats["division_parent_cap_rejected"], 0)
        indegree, outdegree = _degree_counts(result)
        self.assertLessEqual(max(indegree.values(), default=0), 1)
        self.assertLessEqual(max(outdegree.values(), default=0), 2)

    def test_parent_already_at_two_children_is_never_a_division_source(self) -> None:
        nodes, edges, parents = _division_batch(400)
        full_parent_child = _node_um(10000, 1, 2.0)
        available_parent_orphan = _node_um(10001, 1, 122.0)
        edges.append(_edge(parents[0], full_parent_child, "division"))
        result, _stats = add_safe_divisions(
            [*nodes, full_parent_child, available_parent_orphan], edges
        )
        added = [edge for edge in result if edge.target_id == available_parent_orphan.node_id]
        self.assertEqual(
            [(edge.source_id, edge.target_id) for edge in added],
            [(parents[1].node_id, available_parent_orphan.node_id)],
        )
        self.assertEqual(sum(edge.source_id == parents[0].node_id for edge in result), 2)
        validate_graph([*nodes, full_parent_child, available_parent_orphan], result)

    def test_equal_cost_selection_is_input_order_independent(self) -> None:
        nodes, edges, parents = _division_batch(400, crowded_parents=2)
        orphans = [_node_um(10000 + index, 1, 2.0) for index in range(3)]
        forward, _ = add_safe_divisions([*nodes, *orphans], edges)
        reverse, _ = add_safe_divisions(
            reversed([*nodes, *orphans]), reversed(edges)
        )
        forward_divisions = [edge for edge in forward if edge.kind == "division"]
        reverse_divisions = [edge for edge in reverse if edge.kind == "division"]
        self.assertEqual(forward_divisions, reverse_divisions)
        self.assertEqual(
            [(edge.source_id, edge.target_id) for edge in forward_divisions],
            [(parents[0].node_id, orphans[0].node_id), (parents[1].node_id, orphans[1].node_id)],
        )

    def test_dense_adversarial_candidates_preserve_degree_invariants(self) -> None:
        rng = random.Random(20260718)
        for trial in range(12):
            nodes, edges, _parents = _division_batch(800, crowded_parents=8)
            orphan_ids = list(range(10000, 10024))
            rng.shuffle(orphan_ids)
            orphans = [
                _node_um(node_id, 1, 1.25 + 0.125 * ((index + trial) % 24))
                for index, node_id in enumerate(orphan_ids)
            ]
            rng.shuffle(nodes)
            rng.shuffle(edges)
            result, stats = add_safe_divisions([*nodes, *orphans], edges)
            validate_graph([*nodes, *orphans], result)
            indegree, outdegree = _degree_counts(result)
            self.assertLessEqual(max(indegree.values(), default=0), 1)
            self.assertLessEqual(max(outdegree.values(), default=0), 2)
            self.assertLessEqual(stats["division_added"], stats["division_global_cap"])
            self.assertLessEqual(
                stats["division_parent_cap_rejected"], stats["division_candidate_count"]
            )


class PostprocessingTests(unittest.TestCase):
    def test_short_filter_skips_when_it_would_remove_everything(self) -> None:
        nodes = [_node_um(index, index, float(index)) for index in range(5)]
        edges = [_edge(nodes[index], nodes[index + 1]) for index in range(4)]
        result_nodes, result_edges, stats = prune_graph(nodes, edges)
        self.assertEqual(result_nodes, tuple(nodes))
        self.assertEqual(result_edges, tuple(edges))
        self.assertEqual(stats["short_component_filter_skipped_all"], 1)

    def test_six_member_component_is_kept(self) -> None:
        nodes = [_node_um(index, index, float(index)) for index in range(6)]
        edges = [_edge(nodes[index], nodes[index + 1]) for index in range(5)]
        result_nodes, _, stats = prune_graph(nodes, edges)
        self.assertEqual(len(result_nodes), 6)
        self.assertEqual(stats["short_component_filter_skipped_all"], 0)

    def test_short_division_component_kept_while_other_short_component_pruned(self) -> None:
        parent = _node_um(0, 0, 0.0)
        child_a = _node_um(1, 1, 1.0)
        child_b = GraphNode(2, 1, 0.0, 1.0, 0.0)
        line_a = _node_um(10, 0, 100.0)
        line_b = _node_um(11, 1, 101.0)
        nodes = [parent, child_a, child_b, line_a, line_b]
        edges = [_edge(parent, child_a), _edge(parent, child_b, "division"), _edge(line_a, line_b)]
        result_nodes, result_edges, stats = postprocess_graph(nodes, edges)
        self.assertEqual({node.node_id for node in result_nodes}, {0, 1, 2})
        self.assertEqual(len(result_edges), 2)
        self.assertEqual(stats["short_component_nodes_pruned"], 2)

    def test_smoothing_keeps_floats_until_csv_serialization(self) -> None:
        xs = [0.0, 1.0, 10.0, 3.0, 4.0]
        nodes = [GraphNode(index, index, 0.0, 0.0, x) for index, x in enumerate(xs)]
        edges = [_edge(nodes[index], nodes[index + 1]) for index in range(4)]
        result_nodes, result_edges, stats = line_fit_smooth(nodes, edges)
        by_id = {node.node_id: node for node in result_nodes}
        self.assertAlmostEqual(by_id[2].x, 4.88)
        self.assertNotEqual(by_id[2].x, round(by_id[2].x))
        self.assertEqual(result_edges, tuple(edges))
        self.assertEqual(stats["smoothed_nodes"], 5)

    def test_predecessor_traversal_does_not_stop_at_predecessor_branch(self) -> None:
        parent = GraphNode(0, 0, 0.0, 0.0, 0.0)
        center = GraphNode(1, 1, 0.0, 0.0, 10.0)
        sister = GraphNode(2, 1, 0.0, 1.0, 0.0)
        successor = GraphNode(3, 2, 0.0, 0.0, 2.0)
        edges = [_edge(parent, center), _edge(parent, sister, "division"), _edge(center, successor)]
        result_nodes, _, _ = line_fit_smooth([parent, center, sister, successor], edges)
        by_id = {node.node_id: node for node in result_nodes}
        # Public V45 includes the center in the polyfit even though its unique
        # predecessor is itself a branch point: fit(-1:0, 0:10, 1:2) -> 4.
        self.assertAlmostEqual(by_id[1].x, 5.2)


class InvariantAndProvenanceTests(unittest.TestCase):
    def test_calibration_defaults_divisions_off_but_retains_explicit_opt_in(self) -> None:
        parent = _node_um(0, 0, 0.0)
        child_a = _node_um(1, 1, 1.0)
        child_b = _node_um(2, 1, 2.0)

        default = calibrate_v45_graph([parent, child_a, child_b])
        opted_in = calibrate_v45_graph(
            [parent, child_a, child_b], division_enabled=True
        )

        self.assertFalse(default.provenance["division_enabled"])
        self.assertEqual(default.stats["division_added"], 0)
        self.assertEqual(max(_degree_counts(default.edges)[1].values()), 1)
        self.assertTrue(opted_in.provenance["division_enabled"])
        self.assertEqual(opted_in.stats["division_added"], 1)
        self.assertEqual(max(_degree_counts(opted_in.edges)[1].values()), 2)

    def test_merges_skips_and_third_children_fail_closed(self) -> None:
        nodes = [
            _node_um(0, 0, 0.0),
            _node_um(1, 0, 1.0),
            _node_um(2, 1, 0.0),
            _node_um(3, 1, 1.0),
            _node_um(4, 1, 2.0),
            _node_um(5, 2, 0.0),
        ]
        with self.assertRaisesRegex(GraphInvariantError, "merge"):
            validate_graph(nodes, [_edge(nodes[0], nodes[2]), _edge(nodes[1], nodes[2])])
        with self.assertRaisesRegex(GraphInvariantError, "outdegree"):
            validate_graph(nodes, [_edge(nodes[0], nodes[2]), _edge(nodes[0], nodes[3]), _edge(nodes[0], nodes[4])])
        with self.assertRaisesRegex(GraphInvariantError, "next-frame"):
            validate_graph(nodes, [_edge(nodes[0], nodes[5])])

    def test_exact_v45_provenance_distinguishes_replay_and_scored_hashes(self) -> None:
        nodes = [_node_um(frame, frame, float(frame)) for frame in range(6)]
        result = calibrate_v45_graph(nodes)
        provenance = result.provenance
        self.assertEqual(provenance["source_script_version_id"], 335810749)
        self.assertEqual(provenance["motion_relink_raw_gates_um"], [6.0, 10.0])
        self.assertEqual(provenance["motion_direction"], "backward")
        self.assertEqual(provenance["motion_error_weight"], 0.25)
        self.assertEqual(provenance["output_edge_validation_um"], 14.0)
        self.assertTrue(provenance["output_edge_validation_is_not_a_relink_pass"])
        self.assertFalse(provenance["full_frame_dense_matrix"])
        self.assertEqual(provenance["parity_mode"], "sparse")
        self.assertFalse(provenance["gap2_enabled"])
        self.assertFalse(provenance["deepcenter_enabled"])
        self.assertFalse(provenance["adaptive_short_track_rescue"])
        self.assertEqual(
            provenance["public_replay_submission_sha256"],
            "6f8d81f6e5d9eaa066e47f0cdd2d22df48eaa74853d2940d0bc957f3ef3a19e1",
        )
        self.assertEqual(
            provenance["scored_account_submission_sha256"],
            "8c73d776abca799a37c2bd24768a1bbb510418c3327192430730540d56cea698",
        )
        diagnostic = calibrate_v45_graph(nodes, parity_mode="dense_diagnostic")
        self.assertTrue(diagnostic.provenance["full_frame_dense_matrix"])
        self.assertEqual(diagnostic.provenance["parity_mode"], "dense_diagnostic")
        self.assertNotEqual(
            provenance["public_replay_submission_sha256"],
            provenance["scored_account_submission_sha256"],
        )


class LocalV45ArtifactParityTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "reports/g02/v45sub.csv").exists(), "local V45 artifacts absent")
    def test_public_replay_artifact_hash_and_row_count(self) -> None:
        path = ROOT / "reports/g02/v45sub.csv"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "6f8d81f6e5d9eaa066e47f0cdd2d22df48eaa74853d2940d0bc957f3ef3a19e1",
        )
        with path.open("rb") as handle:
            self.assertEqual(sum(1 for _ in handle), 252243)

    def test_banked_backward_quarter_cv57_ledger(self) -> None:
        report = json.loads(
            (
                ROOT
                / "reports/g02/phase1_e3_backward_motion_coefficient_cv57_v1/report.json"
            ).read_text(encoding="utf-8")
        )
        setting = next(
            row for row in report["settings"] if row["motion_error_weight"] == 0.25
        )
        self.assertAlmostEqual(
            setting["summary"]["adj_edge_jaccard"], 0.8731495935289003
        )
        self.assertEqual(
            setting["edge_counts"],
            {"edge_tp": 34205, "edge_fp": 2685, "edge_fn": 2063},
        )
        self.assertEqual(report["selection"]["video_count"], 57)

    @unittest.skipUnless((ROOT / "reports/g02/v45out.zip").exists(), "raw V45 GEFFs absent")
    def test_banked_sparse_ties_match_dense_diagnostic(self) -> None:
        """Sparse and dense solvers agree under the banked backward cost."""

        import zarr
        from zarr.storage import ZipStore

        archive_path = ROOT / "reports/g02/v45out.zip"
        wanted = {"44b6_0b24845f", "6bba_05db0fb1"}
        store = ZipStore(archive_path, mode="r")
        try:
            for dataset in sorted(wanted):
                root = (
                    "tracking_repo/predictions/unknown/unet_transformer/split_0/"
                    f"{dataset}.geff"
                )
                graph = zarr.open_group(store=store, path=root, mode="r")
                nodes = [
                    GraphNode(int(node_id), int(t), float(z), float(y), float(x))
                    for node_id, t, z, y, x in zip(
                        graph["nodes/ids"][:],
                        graph["nodes/props/t/values"][:],
                        graph["nodes/props/z/values"][:],
                        graph["nodes/props/y/values"][:],
                        graph["nodes/props/x/values"][:],
                        strict=True,
                    )
                ]
                node_by_id = {node.node_id: node for node in nodes}
                probabilities: dict[tuple[int, int], float] = {}
                for (source, target), probability in zip(
                    graph["edges/ids"][:],
                    graph["edges/props/edge_prob/values"][:],
                    strict=True,
                ):
                    source = int(source)
                    target = int(target)
                    if (
                        node_by_id[target].t == node_by_id[source].t + 1
                        and physical_distance(node_by_id[source], node_by_id[target]) <= 14.0
                    ):
                        probabilities[source, target] = max(
                            probabilities.get((source, target), float("-inf")),
                            float(probability),
                        )
                sparse_edges, _ = build_adjacent_edges(nodes, probabilities)
                sparse = {(edge.source_id, edge.target_id, edge.kind) for edge in sparse_edges}
                diagnostic_edges, _ = build_adjacent_edges(
                    nodes, probabilities, parity_mode="dense_diagnostic"
                )
                diagnostic = {
                    (edge.source_id, edge.target_id, edge.kind)
                    for edge in diagnostic_edges
                }
                self.assertEqual(sparse, diagnostic)

                sparse_gap_nodes, sparse_gap_edges, sparse_gap_stats = close_one_frame_gaps(
                    nodes, diagnostic_edges
                )
                dense_gap_nodes, dense_gap_edges, dense_gap_stats = close_one_frame_gaps(
                    nodes, diagnostic_edges, parity_mode="dense_diagnostic"
                )

                def gap_pairs(
                    gap_nodes: tuple[GraphNode, ...], gap_edges: tuple[GraphEdge, ...]
                ) -> set[tuple[int, int]]:
                    synthetic_ids = {node.node_id for node in gap_nodes if node.synthetic}
                    before = {
                        edge.target_id: edge.source_id
                        for edge in gap_edges
                        if edge.target_id in synthetic_ids
                    }
                    after = {
                        edge.source_id: edge.target_id
                        for edge in gap_edges
                        if edge.source_id in synthetic_ids
                    }
                    return {(before[node_id], after[node_id]) for node_id in synthetic_ids}

                sparse_pairs = gap_pairs(sparse_gap_nodes, sparse_gap_edges)
                dense_pairs = gap_pairs(dense_gap_nodes, dense_gap_edges)
                self.assertEqual(
                    (dense_gap_stats["gap_matches"], sparse_gap_stats["gap_matches"]),
                    (len(dense_pairs), len(sparse_pairs)),
                )
                self.assertEqual(sparse_pairs, dense_pairs)
                self.assertEqual(sparse_gap_stats, dense_gap_stats)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
