from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.solver import (  # noqa: E402
    _solve_assignment_components,
    _solve_gated_assignment,
    build_edge_candidates,
    select_edges,
    validate_lineage_graph,
)
from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig  # noqa: E402


def node(node_id: int, t: int, z: int = 0, y: int = 0, x: int = 0, intensity: float = 1.0) -> CandidateNode:
    return CandidateNode(
        dataset="synthetic",
        node_id=node_id,
        t=t,
        z=z,
        y=y,
        x=x,
        score=1.0,
        sources=("synthetic",),
        radius_um=1.0,
        intensity=intensity,
    )


def edge(source: int, target: int, dt: int, cost: float) -> CandidateEdge:
    return CandidateEdge(
        source_id=source,
        target_id=target,
        dt=dt,
        distance_um=cost,
        cost=cost,
        score=max(0.0, 1.0 - cost),
    )


def config(**overrides: object) -> RouteConfig:
    values = {
        "label": "synthetic",
        "detector_sources": ("synthetic",),
        "threshold_quantiles": (0.9,),
        "min_distance_um": 1.0,
        "fusion_radius_um": 1.0,
        "max_points_per_t": 20,
        "max_link_um": 10.0,
        "edge_cost_threshold": 0.5,
        "node_score_weight": 0.0,
        "density_cost_weight": 0.0,
        "appearance_cost_weight": 0.0,
        "gap_motion_cost_weight": 0.0,
    }
    values.update(overrides)
    return RouteConfig(**values)


def pairs(edges: list[CandidateEdge]) -> list[tuple[int, int]]:
    return [(value.source_id, value.target_id) for value in edges]


class AlphaSolverTests(unittest.TestCase):
    def test_global_assignment_avoids_greedy_dead_end(self) -> None:
        nodes_by_t = [
            [node(1, 0), node(2, 0)],
            [node(3, 1), node(4, 1)],
        ]
        candidates = [
            edge(1, 3, 1, 0.10),
            edge(1, 4, 1, 0.20),
            edge(2, 3, 1, 0.11),
            edge(2, 4, 1, 0.90),
        ]

        selected, forks = select_edges(nodes_by_t, candidates, (1.0, 1.0, 1.0), config())

        self.assertEqual(pairs(selected), [(1, 4), (2, 3)])
        self.assertEqual(forks, [])

    def test_assignment_is_deterministic_under_candidate_permutation(self) -> None:
        nodes_by_t = [[node(1, 0), node(2, 0)], [node(3, 1), node(4, 1)]]
        candidates = [
            edge(1, 3, 1, 0.20),
            edge(1, 4, 1, 0.20),
            edge(2, 3, 1, 0.20),
            edge(2, 4, 1, 0.20),
        ]
        expected: list[tuple[int, int]] | None = None
        for seed in range(12):
            shuffled = list(candidates)
            random.Random(seed).shuffle(shuffled)
            selected, _ = select_edges(nodes_by_t, shuffled, (1.0, 1.0, 1.0), config())
            current = pairs(selected)
            if expected is None:
                expected = current
            self.assertEqual(current, expected)

    def test_component_decomposition_preserves_assignment(self) -> None:
        candidates = [
            edge(1, 4, 1, 0.10),
            edge(1, 5, 1, 0.20),
            edge(2, 4, 1, 0.11),
            edge(2, 5, 1, 0.90),
            edge(3, 6, 1, 0.15),
        ]
        source_ids = [1, 2, 3]
        target_ids = [4, 5, 6]

        global_result = _solve_gated_assignment(
            candidates, source_ids, target_ids, config()
        )
        component_result = _solve_assignment_components(
            candidates, source_ids, target_ids, config()
        )

        self.assertEqual(pairs(component_result), pairs(global_result))

    def test_random_small_assignments_match_exhaustive_optimum(self) -> None:
        rng = random.Random(20260711)
        birth_cost = 0.35
        death_cost = 0.45
        threshold = 0.9
        cfg = config(
            edge_cost_threshold=threshold,
            birth_cost=birth_cost,
            death_cost=death_cost,
        )

        for case in range(200):
            n_sources = rng.randint(1, 4)
            n_targets = rng.randint(1, 4)
            source_ids = list(range(1, n_sources + 1))
            target_ids = list(range(101, 101 + n_targets))
            candidates: list[CandidateEdge] = []
            costs: dict[tuple[int, int], float] = {}
            for source_id in source_ids:
                for target_id in target_ids:
                    if rng.random() < 0.7:
                        value = round(rng.uniform(0.01, 1.1), 5)
                        costs[(source_id, target_id)] = value
                        candidates.append(edge(source_id, target_id, 1, value))

            best = float("inf")

            def visit(index: int, used: set[int], running: float) -> None:
                nonlocal best
                if running >= best:
                    return
                if index == len(source_ids):
                    best = min(best, running + birth_cost * (n_targets - len(used)))
                    return
                source_id = source_ids[index]
                visit(index + 1, used, running + death_cost)
                for target_id in target_ids:
                    value = costs.get((source_id, target_id))
                    if (
                        target_id in used
                        or value is None
                        or value > threshold
                        or value >= birth_cost + death_cost
                    ):
                        continue
                    visit(index + 1, used | {target_id}, running + value)

            visit(0, set(), 0.0)
            selected = _solve_assignment_components(
                candidates, source_ids, target_ids, cfg
            )
            selected_pairs = set(pairs(selected))
            objective = sum(costs[pair] for pair in selected_pairs)
            objective += death_cost * (n_sources - len(selected_pairs))
            objective += birth_cost * (n_targets - len(selected_pairs))
            self.assertAlmostEqual(objective, best, places=9, msg=f"case={case}")

    def test_sparse_components_bound_lap_matrix_size(self) -> None:
        count = 200
        candidates = [edge(i, 10_000 + i, 2, 0.1) for i in range(1, count + 1)]
        source_ids = list(range(1, count + 1))
        target_ids = list(range(10_001, 10_001 + count))

        with patch(
            "celltrack_alpha.solver.linear_sum_assignment",
            wraps=__import__("scipy.optimize", fromlist=["linear_sum_assignment"]).linear_sum_assignment,
        ) as mocked:
            selected = _solve_assignment_components(
                candidates, source_ids, target_ids, config(max_gap=2)
            )

        self.assertEqual(len(selected), count)
        self.assertEqual(len(mocked.call_args_list), count)
        self.assertEqual(
            max(call.args[0].shape[0] for call in mocked.call_args_list),
            2,
        )

    def test_gap_closing_runs_after_adjacent_assignment(self) -> None:
        nodes_by_t = [[node(1, 0)], [node(2, 1)], [], [node(3, 3)]]
        candidates = [edge(1, 2, 1, 0.10), edge(2, 3, 2, 0.15)]

        selected, _ = select_edges(
            nodes_by_t,
            candidates,
            (1.0, 1.0, 1.0),
            config(max_gap=2, gap_link_um=10.0),
        )

        self.assertEqual(pairs(selected), [(1, 2), (2, 3)])

    def test_gap_does_not_replace_an_adjacent_predecessor(self) -> None:
        nodes_by_t = [[node(1, 0)], [node(2, 1)], [node(3, 2)]]
        candidates = [
            edge(1, 2, 1, 0.10),
            edge(2, 3, 1, 0.10),
            edge(1, 3, 2, 0.01),
        ]

        selected, _ = select_edges(
            nodes_by_t,
            candidates,
            (1.0, 1.0, 1.0),
            config(max_gap=2, gap_link_um=10.0),
        )

        self.assertEqual(pairs(selected), [(1, 2), (2, 3)])

    def test_lineage_validator_rejects_merge(self) -> None:
        nodes = [node(1, 0), node(2, 0), node(3, 1)]
        with self.assertRaisesRegex(ValueError, "merge"):
            validate_lineage_graph(
                nodes,
                [edge(1, 3, 1, 0.1), edge(2, 3, 1, 0.1)],
                allow_divisions=False,
            )

    def test_lineage_validator_rejects_mixed_time_division(self) -> None:
        nodes = [node(1, 0), node(2, 1), node(3, 2)]
        with self.assertRaisesRegex(ValueError, "next timepoint"):
            validate_lineage_graph(
                nodes,
                [edge(1, 2, 1, 0.1), edge(1, 3, 2, 0.1)],
                allow_divisions=True,
            )

    def test_lineage_validator_full_invariant_table(self) -> None:
        n0 = node(1, 0)
        n1 = node(2, 1)
        n1b = node(3, 1)
        n1c = node(4, 1)
        fractional_dt = edge(1, 2, 1, 0.1)
        object.__setattr__(fractional_dt, "dt", 1.5)
        cases = [
            ("duplicate node IDs", [n0, node(1, 1)], [], False, "duplicate node"),
            (
                "duplicate edges",
                [n0, n1],
                [edge(1, 2, 1, 0.1), edge(1, 2, 1, 0.1)],
                False,
                "duplicate edge",
            ),
            ("missing endpoint", [n0], [edge(1, 99, 1, 0.1)], False, "endpoint is missing"),
            ("self edge", [n0], [edge(1, 1, 0, 0.1)], False, "does not advance"),
            ("backward edge", [n0, n1], [edge(2, 1, -1, 0.1)], False, "does not advance"),
            ("dt mismatch", [n0, n1], [edge(1, 2, 2, 0.1)], False, "times imply"),
            (
                "fractional dt",
                [n0, n1],
                [fractional_dt],
                False,
                "dt must be a positive integer",
            ),
            (
                "division disabled",
                [n0, n1, n1b],
                [edge(1, 2, 1, 0.1), edge(1, 3, 1, 0.1)],
                False,
                "out-degree greater than 1",
            ),
            (
                "more than two children",
                [n0, n1, n1b, n1c],
                [edge(1, 2, 1, 0.1), edge(1, 3, 1, 0.1), edge(1, 4, 1, 0.1)],
                True,
                "out-degree greater than 2",
            ),
        ]
        for label, nodes, edges, allow_divisions, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    validate_lineage_graph(
                        nodes,
                        edges,
                        allow_divisions=allow_divisions,
                    )

        result = validate_lineage_graph(
            [n0, n1, n1b],
            [edge(1, 2, 1, 0.1), edge(1, 3, 1, 0.1)],
            allow_divisions=True,
        )
        self.assertEqual(result["num_divisions"], 1)

    def test_birth_death_costs_control_unmatched_decision(self) -> None:
        nodes_by_t = [[node(1, 0)], [node(2, 1)]]
        candidates = [edge(1, 2, 1, 0.6)]

        unmatched, _ = select_edges(
            nodes_by_t,
            candidates,
            (1.0, 1.0, 1.0),
            config(edge_cost_threshold=1.0, birth_cost=0.2, death_cost=0.3),
        )
        linked, _ = select_edges(
            nodes_by_t,
            candidates,
            (1.0, 1.0, 1.0),
            config(edge_cost_threshold=1.0, birth_cost=0.4, death_cost=0.4),
        )
        boundary, _ = select_edges(
            nodes_by_t,
            [edge(1, 2, 1, 0.5)],
            (1.0, 1.0, 1.0),
            config(edge_cost_threshold=1.0, birth_cost=0.2, death_cost=0.3),
        )

        self.assertEqual(unmatched, [])
        self.assertEqual(pairs(linked), [(1, 2)])
        self.assertEqual(boundary, [])

    def test_global_gap_assignment_avoids_greedy_dead_end_and_is_deterministic(self) -> None:
        nodes_by_t = [
            [node(1, 0), node(2, 0)],
            [node(3, 1), node(4, 1)],
            [],
            [node(5, 3), node(6, 3)],
        ]
        adjacent = [edge(1, 3, 1, 0.1), edge(2, 4, 1, 0.1)]
        gaps = [
            edge(3, 5, 2, 0.10),
            edge(3, 6, 2, 0.20),
            edge(4, 5, 2, 0.11),
            edge(4, 6, 2, 0.90),
        ]
        cfg = config(max_gap=2, gap_link_um=10.0)
        expected = [(1, 3), (2, 4), (3, 6), (4, 5)]
        for seed in range(10):
            candidates = adjacent + gaps
            random.Random(seed).shuffle(candidates)
            selected, _ = select_edges(
                nodes_by_t, candidates, (1.0, 1.0, 1.0), cfg
            )
            self.assertEqual(pairs(selected), expected)

    def test_division_rescue_can_steal_child_from_near_duplicate_source(self) -> None:
        nodes_by_t = [
            [node(1, 0, y=10), node(2, 0, y=8)],
            [node(3, 1, y=14), node(4, 1, y=6)],
            [node(5, 2, y=15), node(6, 2, y=5)],
        ]
        candidates = [
            edge(1, 3, 1, 0.10),
            edge(1, 4, 1, 0.40),
            edge(2, 4, 1, 0.05),
            edge(3, 5, 1, 0.10),
            edge(4, 6, 1, 0.10),
        ]

        selected, forks = select_edges(
            nodes_by_t,
            candidates,
            (1.0, 1.0, 1.0),
            config(
                edge_cost_threshold=0.5,
                enable_divisions=True,
                division_requires_unlinked_parent=False,
                division_requires_child_future=True,
                division_allow_link_steal=True,
                division_child_max_um=5.0,
                division_steal_max_distance_um=5.0,
                division_steal_source_max_um=3.0,
                division_child_max_sep_um=10.0,
            ),
        )

        self.assertEqual(pairs(selected), [(1, 3), (1, 4), (3, 5), (4, 6)])
        self.assertEqual(len(forks), 1)
        self.assertEqual(forks[0].parent_id, 1)

    def test_division_rescue_does_not_steal_from_far_source(self) -> None:
        nodes_by_t = [
            [node(1, 0, y=10), node(2, 0, y=0)],
            [node(3, 1, y=14), node(4, 1, y=6)],
            [node(5, 2, y=15), node(6, 2, y=5)],
        ]
        candidates = [
            edge(1, 3, 1, 0.10),
            edge(1, 4, 1, 0.40),
            edge(2, 4, 1, 0.05),
            edge(3, 5, 1, 0.10),
            edge(4, 6, 1, 0.10),
        ]

        selected, forks = select_edges(
            nodes_by_t,
            candidates,
            (1.0, 1.0, 1.0),
            config(
                edge_cost_threshold=0.5,
                enable_divisions=True,
                division_requires_unlinked_parent=False,
                division_requires_child_future=True,
                division_allow_link_steal=True,
                division_child_max_um=5.0,
                division_steal_max_distance_um=5.0,
                division_steal_source_max_um=3.0,
                division_child_max_sep_um=10.0,
            ),
        )

        self.assertEqual(pairs(selected), [(1, 3), (2, 4), (3, 5), (4, 6)])
        self.assertEqual(forks, [])

    def test_second_choice_edge_rescue_replaces_short_duplicate_link(self) -> None:
        n0 = node(1, 0, y=10, intensity=100.0)
        short = node(2, 1, y=11, intensity=100.0)
        long = node(3, 1, y=16, intensity=130.0)
        nodes_by_t = [[n0], [short, long]]
        candidates = [
            CandidateEdge(
                source_id=1,
                target_id=2,
                dt=1,
                distance_um=1.0,
                cost=0.10,
                score=0.90,
                features={"source_out_rank": 1.0, "target_in_rank": 1.0},
            ),
            CandidateEdge(
                source_id=1,
                target_id=3,
                dt=1,
                distance_um=6.0,
                cost=0.40,
                score=0.60,
                features={"source_out_rank": 2.0, "target_in_rank": 1.0},
            ),
        ]

        selected, forks = select_edges(
            nodes_by_t,
            candidates,
            (1.0, 1.0, 1.0),
            config(
                edge_cost_threshold=0.5,
                enable_second_choice_edge_rescue=True,
                edge_rescue_short_max_um=2.2,
                edge_rescue_long_min_um=5.5,
                edge_rescue_long_max_um=7.1,
                edge_rescue_margin_min_um=4.5,
                edge_rescue_intensity_ratio=1.02,
                edge_rescue_alt_cost_max=0.45,
            ),
        )

        self.assertEqual(pairs(selected), [(1, 3)])
        self.assertEqual(forks, [])

    def test_second_choice_edge_rescue_requires_brighter_target(self) -> None:
        n0 = node(1, 0, y=10, intensity=100.0)
        short = node(2, 1, y=11, intensity=100.0)
        dim_long = node(3, 1, y=16, intensity=90.0)
        nodes_by_t = [[n0], [short, dim_long]]
        candidates = [
            CandidateEdge(1, 2, 1, 1.0, 0.10, 0.90, features={"source_out_rank": 1.0, "target_in_rank": 1.0}),
            CandidateEdge(1, 3, 1, 6.0, 0.40, 0.60, features={"source_out_rank": 2.0, "target_in_rank": 1.0}),
        ]

        selected, _ = select_edges(
            nodes_by_t,
            candidates,
            (1.0, 1.0, 1.0),
            config(
                edge_cost_threshold=0.5,
                enable_second_choice_edge_rescue=True,
                edge_rescue_short_max_um=2.2,
                edge_rescue_long_min_um=5.5,
                edge_rescue_long_max_um=7.1,
                edge_rescue_margin_min_um=4.5,
                edge_rescue_intensity_ratio=1.02,
                edge_rescue_alt_cost_max=0.45,
            ),
        )

        self.assertEqual(pairs(selected), [(1, 2)])

    def test_physical_scale_controls_candidate_gate(self) -> None:
        nodes_by_t = [[node(1, 0, z=0)], [node(2, 1, z=2)]]
        gated = build_edge_candidates(
            nodes_by_t,
            (3.0, 1.0, 1.0),
            config(max_link_um=5.0),
        )
        feasible = build_edge_candidates(
            nodes_by_t,
            (2.0, 1.0, 1.0),
            config(max_link_um=5.0),
        )

        self.assertEqual(gated, [])
        self.assertEqual(len(feasible), 1)
        self.assertAlmostEqual(feasible[0].distance_um, 4.0)

    def test_edge_candidates_include_association_context_features(self) -> None:
        nodes_by_t = [
            [node(1, 0, y=0), node(2, 0, y=5)],
            [node(3, 1, y=1), node(4, 1, y=4)],
        ]

        candidates = build_edge_candidates(
            nodes_by_t,
            (1.0, 1.0, 1.0),
            config(max_link_um=10.0, max_neighbors=2),
        )
        by_pair = {(edge.source_id, edge.target_id): edge for edge in candidates}

        self.assertEqual(by_pair[(1, 3)].features["source_out_rank"], 1.0)
        self.assertEqual(by_pair[(1, 3)].features["target_in_rank"], 1.0)
        self.assertEqual(by_pair[(1, 3)].features["mutual_nearest"], 1.0)
        self.assertEqual(by_pair[(1, 4)].features["source_out_rank"], 2.0)
        self.assertGreater(by_pair[(1, 4)].features["candidate_ambiguity"], 0.0)
        self.assertEqual(by_pair[(2, 4)].features["mutual_nearest"], 1.0)
        self.assertIn("min_best_margin_um", by_pair[(2, 3)].features)


    def test_appearance_cost_is_explicitly_opt_in(self) -> None:
        nodes_by_t = [[node(1, 0, intensity=0.0)], [node(2, 1, intensity=100.0)]]
        without = build_edge_candidates(
            nodes_by_t, (1.0, 1.0, 1.0), config(max_link_um=5.0)
        )[0]
        with_appearance = build_edge_candidates(
            nodes_by_t,
            (1.0, 1.0, 1.0),
            config(max_link_um=5.0, appearance_cost_weight=0.3),
        )[0]

        self.assertAlmostEqual(with_appearance.cost - without.cost, 0.3, places=6)
        self.assertAlmostEqual(with_appearance.features["appearance_delta"], 1.0)

    def test_invalid_scale_and_config_fail_before_linking(self) -> None:
        nodes_by_t = [[node(1, 0)], [node(2, 1)]]
        with self.assertRaisesRegex(ValueError, "scale_zyx"):
            build_edge_candidates(nodes_by_t, (0.0, 1.0, 1.0), config())
        with self.assertRaisesRegex(ValueError, "max_gap"):
            select_edges(nodes_by_t, [], (1.0, 1.0, 1.0), config(max_gap=1.5))
        invalid_configs = [
            ("edge_cost_threshold", {"edge_cost_threshold": float("nan")}),
            ("max_link_um", {"max_link_um": 0.0}),
            ("gap_link_um", {"gap_link_um": float("inf")}),
            ("max_neighbors", {"max_neighbors": 0}),
            ("birth_cost", {"birth_cost": -1.0}),
            ("appearance_cost_weight", {"appearance_cost_weight": -0.1}),
            ("division_steal_max_distance_um", {"division_steal_max_distance_um": -0.1}),
            ("division_steal_source_max_um", {"division_steal_source_max_um": -0.1}),
            ("edge_rescue_short_max_um", {"edge_rescue_short_max_um": -0.1}),
            ("edge_rescue_long_min_um", {"edge_rescue_long_min_um": -0.1}),
        ]
        for name, override in invalid_configs:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    select_edges(
                        nodes_by_t,
                        [],
                        (1.0, 1.0, 1.0),
                        config(**override),
                    )
        with self.assertRaisesRegex(ValueError, "edge_rescue_long_min_um"):
            select_edges(
                nodes_by_t,
                [],
                (1.0, 1.0, 1.0),
                config(edge_rescue_long_min_um=8.0, edge_rescue_long_max_um=7.0),
            )
        with self.assertRaisesRegex(ValueError, "endpoint is missing"):
            select_edges(
                nodes_by_t,
                [edge(1, 99, 1, 0.1)],
                (1.0, 1.0, 1.0),
                config(),
            )
        with self.assertRaisesRegex(ValueError, "declared positive dt"):
            select_edges(
                nodes_by_t,
                [edge(1, 2, 2, 0.1)],
                (1.0, 1.0, 1.0),
                config(),
            )
        fractional = edge(1, 2, 1, 0.1)
        object.__setattr__(fractional, "dt", 1.5)
        with self.assertRaisesRegex(ValueError, "dt must be a positive integer"):
            select_edges(
                nodes_by_t,
                [fractional],
                (1.0, 1.0, 1.0),
                config(),
            )


if __name__ == "__main__":
    unittest.main()
