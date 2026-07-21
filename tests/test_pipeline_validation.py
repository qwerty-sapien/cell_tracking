from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.pipeline import (  # noqa: E402
    _add_endpoint_backward_clones,
    _add_endpoint_forward_clones,
    _admit_free_candidate_edges,
    _admit_motion_endpoint_candidates,
    _clone_selected_nodes_by_localmax_layers,
    _clone_selected_nodes_by_offset_layers,
    _nodes_used_by_edges,
    _prune_submission_components,
    _prune_unlinked_submission_nodes,
    _prune_short_selected_edges,
    _rescue_duplicate_endpoint_conflicts,
    _rescue_refined_chain_conflicts,
    _refine_selected_nodes_by_localmax,
    _rewire_base_chain_targets,
    _rewire_localmax_clone_targets,
    _validate_array_window,
    predict_dataset_window,
)
from celltrack_alpha.proposals import _validate_volume  # noqa: E402


class PipelineValidationTests(unittest.TestCase):
    def test_valid_tzyx_window(self) -> None:
        array = np.zeros((3, 2, 4, 5), dtype=np.uint16)
        self.assertEqual(_validate_array_window("sample", array, 1, 3), (1, 3))
        self.assertEqual(_validate_array_window("sample", array, 0, None), (0, 3))

    def test_fractional_or_boolean_window_values_are_rejected(self) -> None:
        array = np.zeros((3, 2, 4, 5), dtype=np.uint16)
        for start, end in ((0, 1.5), (False, 2), (0, True)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    _validate_array_window("sample", array, start, end)

    def test_window_boundaries_are_rejected(self) -> None:
        array = np.zeros((3, 2, 4, 5), dtype=np.uint16)
        for start, end in ((-1, 2), (0, 0), (2, 2), (2, 1), (0, 4), (3, None)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    _validate_array_window("sample", array, start, end)

    def test_invalid_array_contract_is_rejected(self) -> None:
        cases = [
            np.zeros((2, 3, 4), dtype=np.uint16),
            np.zeros((1, 0, 3, 4), dtype=np.uint16),
            np.zeros((1, 2, 3, 4), dtype=object),
            np.zeros((1, 2, 3, 4), dtype=np.complex64),
        ]
        for array in cases:
            with self.subTest(shape=array.shape, dtype=array.dtype):
                with self.assertRaises(ValueError):
                    _validate_array_window("sample", array, 0, None)


    def test_node_emission_keeps_only_edge_endpoints_when_edges_exist(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode

        nodes = [
            CandidateNode("sample", 1, 0, 0, 0, 0, 0.5, ("a",), 1.0),
            CandidateNode("sample", 2, 1, 0, 1, 0, 0.5, ("a",), 1.0),
            CandidateNode("sample", 3, 1, 0, 2, 0, 0.5, ("a",), 1.0),
        ]
        filtered, stats = _nodes_used_by_edges(
            nodes, [CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.1, score=0.9)]
        )
        self.assertEqual([node.node_id for node in filtered], [1, 2])
        self.assertEqual(stats["policy"], "edge_endpoint_nodes")
        self.assertEqual(stats["before_nodes"], 3)
        self.assertEqual(stats["after_nodes"], 2)

    def test_node_emission_keeps_all_nodes_when_no_edges_exist(self) -> None:
        from celltrack_alpha.types import CandidateNode

        nodes = [CandidateNode("sample", 1, 0, 0, 0, 0, 0.5, ("a",), 1.0)]
        filtered, stats = _nodes_used_by_edges(nodes, [])
        self.assertIs(filtered, nodes)
        self.assertEqual(stats["policy"], "all_nodes_no_edges")

    def test_short_edge_pruning_drops_only_below_threshold_edges(self) -> None:
        from celltrack_alpha.types import CandidateEdge, RouteConfig

        edges = [
            CandidateEdge(1, 2, 1, distance_um=0.5, cost=0.1, score=0.9),
            CandidateEdge(2, 3, 1, distance_um=1.25, cost=0.1, score=0.9),
            CandidateEdge(3, 4, 1, distance_um=2.0, cost=0.1, score=0.9, kind="gap"),
        ]
        config = RouteConfig(
            label="synthetic_short_edge_prune",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=5,
            max_link_um=5.0,
            enable_short_edge_pruning=True,
            short_edge_prune_min_distance_um=1.25,
        )
        kept, stats = _prune_short_selected_edges(edges, config)
        self.assertEqual([(edge.source_id, edge.target_id) for edge in kept], [(2, 3), (3, 4)])
        self.assertEqual(stats["before_edges"], 3)
        self.assertEqual(stats["after_edges"], 2)
        self.assertEqual(stats["dropped_edges"], 1)
        self.assertEqual(stats["dropped_by_kind"], {"continuation": 1})


    def test_free_edge_admission_adds_only_graph_free_mutual_edges(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        nodes = [
            CandidateNode("sample", 1, 0, 0, 0, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 2, 1, 0, 1, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 3, 0, 0, 4, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 4, 1, 0, 5, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 5, 0, 0, 8, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 6, 1, 0, 9, 0, 1.0, ("a",), 1.0),
        ]
        selected = [CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.1, score=0.9)]
        admissible = CandidateEdge(
            3,
            4,
            1,
            distance_um=3.0,
            cost=0.4,
            score=0.7,
            features={
                "mutual_nearest": 1.0,
                "source_out_rank": 1.0,
                "target_in_rank": 1.0,
                "source_out_candidate_count": 1.0,
                "target_in_candidate_count": 1.0,
            },
        )
        blocked_by_target = CandidateEdge(
            5,
            2,
            1,
            distance_um=3.0,
            cost=0.4,
            score=0.7,
            features={
                "mutual_nearest": 1.0,
                "source_out_rank": 1.0,
                "target_in_rank": 1.0,
                "source_out_candidate_count": 1.0,
                "target_in_candidate_count": 1.0,
            },
        )
        too_ambiguous = CandidateEdge(
            5,
            6,
            1,
            distance_um=3.0,
            cost=0.4,
            score=0.7,
            features={
                "mutual_nearest": 1.0,
                "source_out_rank": 1.0,
                "target_in_rank": 1.0,
                "source_out_candidate_count": 2.0,
                "target_in_candidate_count": 1.0,
            },
        )
        config = RouteConfig(
            label="synthetic_free_admission",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=6,
            max_link_um=5.0,
            enable_free_edge_admission=True,
            free_edge_admission_max_cost=0.55,
            free_edge_admission_min_distance_um=1.25,
            free_edge_admission_max_distance_um=4.5,
            free_edge_admission_require_mutual_nearest=True,
            free_edge_admission_max_source_candidates=1,
            free_edge_admission_max_target_candidates=1,
            free_edge_admission_max_edges=4,
        )
        admitted, stats = _admit_free_candidate_edges(
            selected, [*selected, admissible, blocked_by_target, too_ambiguous], nodes, config
        )
        self.assertEqual(
            sorted((edge.source_id, edge.target_id) for edge in admitted),
            [(1, 2), (3, 4)],
        )
        self.assertEqual(stats["admitted_edges"], 1)
        self.assertEqual(stats["admitted"][0]["source_id"], 3)
        self.assertEqual(stats["admitted"][0]["target_id"], 4)


    def test_duplicate_endpoint_rescue_swaps_to_supported_duplicate_endpoint(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        nodes = [
            CandidateNode("sample", 1, 0, 0, 0, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 2, 1, 0, 1, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 3, 1, 0, 1, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 4, 2, 0, 2, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 5, 1, 2, 0, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 6, 1, 2, 1, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 7, 0, 2, 0, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 8, 2, 2, 2, 0, 1.0, ("a",), 1.0),
        ]
        selected = [
            CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(3, 4, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(7, 6, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(5, 8, 1, distance_um=2.0, cost=0.2, score=0.8),
        ]
        replacements = [
            CandidateEdge(1, 3, 1, distance_um=1.0, cost=0.3, score=0.9),
            CandidateEdge(6, 8, 1, distance_um=1.0, cost=0.3, score=0.9),
        ]
        config = RouteConfig(
            label="synthetic_duplicate_endpoint_rescue",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=8,
            max_link_um=5.0,
            enable_duplicate_endpoint_rescue=True,
            duplicate_endpoint_rescue_max_um=1.1,
            duplicate_endpoint_rescue_min_cost=0.25,
            duplicate_endpoint_rescue_max_cost=0.45,
            duplicate_endpoint_rescue_max_distance_um=4.5,
            duplicate_endpoint_rescue_max_swaps=4,
        )
        rescued, stats = _rescue_duplicate_endpoint_conflicts(
            selected, [*selected, *replacements], nodes, (1.0, 1.0, 1.0), config
        )
        self.assertEqual(
            sorted((edge.source_id, edge.target_id) for edge in rescued),
            [(1, 3), (3, 4), (6, 8), (7, 6)],
        )
        self.assertEqual(stats["candidate_swaps"], 2)
        self.assertEqual(stats["applied_swaps"], 2)
        self.assertEqual({swap["mode"] for swap in stats["swaps"]}, {"source", "target"})


    def test_motion_endpoint_admission_prefers_motion_ranked_zshift_target(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        nodes = [
            CandidateNode("sample", 1, 0, 0, 0, 0, 1.0, ("zshift_prev",), 1.0),
            CandidateNode("sample", 2, 1, 0, 2, 0, 1.0, ("zshift_source",), 1.0),
            CandidateNode("sample", 3, 2, 0, 2, 0, 1.0, ("tile_raw",), 1.0),
            CandidateNode("sample", 4, 2, 0, 4, 0, 1.0, ("zshift_dog_dark_m5_mid",), 1.0),
        ]
        selected = [CandidateEdge(1, 2, 1, distance_um=2.0, cost=0.1, score=0.9)]
        config = RouteConfig(
            label="synthetic_motion_endpoint",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=4,
            max_link_um=3.0,
            enable_motion_endpoint_admission=True,
            motion_endpoint_pred_max_um=1.0,
            motion_endpoint_source_max_um=3.0,
            motion_endpoint_max_edges_per_t=4,
            motion_endpoint_target_source_mode="zshift",
            motion_endpoint_requires_history=True,
        )
        admitted, stats = _admit_motion_endpoint_candidates(selected, nodes, (1.0, 1.0, 1.0), config)
        self.assertEqual(sorted((edge.source_id, edge.target_id) for edge in admitted), [(1, 2), (2, 4)])
        self.assertEqual(stats["admitted_edges"], 1)
        self.assertEqual(stats["admitted"][0]["source_id"], 2)
        self.assertEqual(stats["admitted"][0]["target_id"], 4)
        self.assertEqual(stats["per_t"], {"2": 1})


    def test_motion_endpoint_same_source_can_rank_long_continuation_first(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        nodes = [
            CandidateNode("sample", 1, 0, 0, 0, 0, 1.0, ("tile_contrast@q0.75",), 1.0),
            CandidateNode("sample", 2, 1, 0, 1, 0, 1.0, ("tile_contrast@q0.75",), 1.0),
            CandidateNode("sample", 3, 2, 0, 2, 0, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 4, 2, 0, 4, 0, 1.0, ("tile_contrast@q0.75",), 1.0),
            CandidateNode("sample", 5, 0, 0, 0, 5, 1.0, ("tile_contrast@q0.75",), 1.0),
            CandidateNode("sample", 6, 1, 0, 1, 5, 1.0, ("tile_contrast@q0.75",), 1.0),
            CandidateNode("sample", 7, 2, 0, 2, 5, 1.0, ("tile_contrast@q0.75",), 1.0),
        ]
        selected = [
            CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.1, score=0.9),
            CandidateEdge(5, 6, 1, distance_um=1.0, cost=0.1, score=0.9),
        ]
        config = RouteConfig(
            label="synthetic_same_source_motion_endpoint",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=7,
            max_link_um=3.0,
            enable_motion_endpoint_admission=True,
            motion_endpoint_pred_max_um=4.0,
            motion_endpoint_source_max_um=4.0,
            motion_endpoint_max_edges_per_t=1,
            motion_endpoint_target_source_mode="same_source",
            motion_endpoint_rank_mode="source_desc",
            motion_endpoint_requires_history=True,
        )
        admitted, stats = _admit_motion_endpoint_candidates(selected, nodes, (1.0, 1.0, 1.0), config)
        self.assertEqual(sorted((edge.source_id, edge.target_id) for edge in admitted), [(1, 2), (2, 4), (5, 6)])
        self.assertEqual(stats["admitted_edges"], 1)
        self.assertEqual(stats["rank_mode"], "source_desc")
        self.assertEqual(stats["target_source_mode"], "same_source")
        self.assertEqual(stats["admitted"][0]["source_id"], 2)
        self.assertEqual(stats["admitted"][0]["target_id"], 4)


    def test_refined_chain_conflict_rescue_swaps_unique_refined_chain(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        nodes = [
            CandidateNode("sample", 1, 0, 0, 0, 0, 1.0, ("base",), 1.0),
            CandidateNode("sample", 2, 1, 0, 1, 0, 1.0, ("base",), 1.0),
            CandidateNode("sample", 3, 0, 0, 4, 0, 1.0, ("base",), 1.0),
            CandidateNode(
                "sample",
                4,
                1,
                0,
                5,
                0,
                0.65,
                ("refined",),
                1.0,
                features={
                    "refined_admission": 1.0,
                    "refined_admission_temporal_two_sided": 1.0,
                    "refined_admission_temporal_count": 3.0,
                    "refined_admission_temporal_min_um": 1.0,
                },
            ),
            CandidateNode("sample", 5, 2, 0, 6, 0, 1.0, ("base",), 1.0),
        ]
        selected = [
            CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.1, score=0.9),
            CandidateEdge(3, 4, 1, distance_um=1.0, cost=0.1, score=0.9),
            CandidateEdge(2, 5, 1, distance_um=1.2, cost=0.05, score=0.9),
        ]
        rescue_edge = CandidateEdge(
            4,
            5,
            1,
            distance_um=2.0,
            cost=0.30,
            score=0.75,
            features={
                "source_out_rank": 1.0,
                "source_out_candidate_count": 1.0,
                "target_in_rank": 2.0,
            },
        )
        edge_candidates = [*selected, rescue_edge]
        config = RouteConfig(
            label="synthetic_refined_chain_rescue",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=5,
            max_link_um=5.0,
            enable_refined_chain_conflict_rescue=True,
            refined_chain_conflict_max_distance_um=3.0,
            refined_chain_conflict_max_cost=0.42,
            refined_chain_conflict_max_target_rank=3,
            refined_chain_conflict_min_temporal_count=2.0,
        )
        rescued, stats = _rescue_refined_chain_conflicts(
            selected, edge_candidates, nodes, config
        )
        self.assertEqual(
            sorted((edge.source_id, edge.target_id) for edge in rescued),
            [(1, 2), (3, 4), (4, 5)],
        )
        self.assertEqual(stats["replacements"], 1)
        self.assertEqual(
            stats["replaced_edges"],
            [
                {
                    "removed_source_id": 2,
                    "removed_target_id": 5,
                    "added_source_id": 4,
                    "added_target_id": 5,
                    "added_distance_um": 2.0,
                    "added_cost": 0.3,
                    "added_target_rank": 2.0,
                }
            ],
        )


    def test_localmax_refinement_moves_only_selected_edge_endpoints(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        array = np.zeros((2, 5, 7, 7), dtype=np.float32)
        array[0, 2, 4, 2] = 10.0
        array[1, 2, 4, 3] = 9.0
        nodes_by_t = [
            [
                CandidateNode("sample", 1, 0, 2, 2, 2, 0.5, ("a",), 1.0),
                CandidateNode("sample", 99, 0, 0, 0, 0, 0.5, ("a",), 1.0),
            ],
            [CandidateNode("sample", 2, 1, 2, 2, 3, 0.5, ("a",), 1.0)],
        ]
        config = RouteConfig(
            label="synthetic_refine",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=5,
            max_link_um=5.0,
            enable_localmax_refinement=True,
            localmax_refine_radius_um=2.1,
            localmax_refine_sources=("raw",),
        )
        stats = _refine_selected_nodes_by_localmax(
            nodes_by_t,
            [CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.1, score=0.9)],
            array,
            None,
            0,
            (1.0, 1.0, 1.0),
            config,
        )
        self.assertEqual(nodes_by_t[0][0].coord_zyx, (2, 4, 2))
        self.assertEqual(nodes_by_t[1][0].coord_zyx, (2, 4, 3))
        self.assertEqual(nodes_by_t[0][1].coord_zyx, (0, 0, 0))
        self.assertEqual(stats["endpoint_nodes"], 2)
        self.assertEqual(stats["refined_nodes"], 2)
        self.assertEqual(stats["sources"], ["raw"])


    def test_localmax_clone_layer_duplicates_selected_graph_with_new_ids(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        array = np.zeros((2, 5, 7, 7), dtype=np.float32)
        array[0, 2, 4, 2] = 10.0
        array[1, 2, 4, 3] = 9.0
        nodes = [
            CandidateNode("sample", 1, 0, 2, 2, 2, 0.5, ("a",), 1.0),
            CandidateNode("sample", 2, 1, 2, 2, 3, 0.5, ("a",), 1.0),
        ]
        edges = [CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.1, score=0.9)]
        config = RouteConfig(
            label="synthetic_clone",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=5,
            max_link_um=5.0,
            enable_localmax_clone_layers=True,
            localmax_clone_radius_um=2.1,
            localmax_clone_sources=("raw",),
            localmax_clone_center_offsets_zyx=((0, 0, 0),),
        )
        cloned_nodes, cloned_edges, stats = _clone_selected_nodes_by_localmax_layers(
            nodes, edges, array, None, 0, (1.0, 1.0, 1.0), config
        )
        self.assertEqual(len(cloned_nodes), 4)
        self.assertEqual(len(cloned_edges), 2)
        self.assertEqual(cloned_nodes[2].coord_zyx, (2, 4, 2))
        self.assertEqual(cloned_nodes[3].coord_zyx, (2, 4, 3))
        self.assertEqual((cloned_edges[1].source_id, cloned_edges[1].target_id), (3, 4))
        self.assertEqual(stats["cloned_nodes"], 2)
        self.assertEqual(stats["cloned_edges"], 1)




    def test_offset_clone_layer_duplicates_selected_graph_with_new_ids_and_clips(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        array = np.arange(2 * 4 * 6 * 8, dtype=np.float32).reshape(2, 4, 6, 8)
        nodes = [
            CandidateNode("sample", 1, 0, 1, 2, 3, 1.0, ("a",), 1.0),
            CandidateNode("sample", 2, 1, 1, 3, 4, 1.0, ("a",), 1.0),
        ]
        edges = [CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.2, score=0.8)]
        config = RouteConfig(
            label="synthetic_offset_clone",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=2,
            max_link_um=5.0,
            enable_offset_clone_layers=True,
            offset_clone_offsets_zyx=((-2, 1, 5),),
        )
        cloned_nodes, cloned_edges, stats = _clone_selected_nodes_by_offset_layers(
            nodes, edges, array, 0, config
        )
        self.assertEqual([node.node_id for node in cloned_nodes], [1, 2, 3, 4])
        self.assertEqual((cloned_nodes[2].z, cloned_nodes[2].y, cloned_nodes[2].x), (0, 3, 7))
        self.assertEqual((cloned_nodes[3].z, cloned_nodes[3].y, cloned_nodes[3].x), (0, 4, 7))
        self.assertIn("offset_clone_0", cloned_nodes[2].sources)
        self.assertEqual(cloned_nodes[2].features["offset_clone_dx"], 5.0)
        self.assertEqual([(edge.source_id, edge.target_id) for edge in cloned_edges], [(1, 2), (3, 4)])
        self.assertEqual(cloned_edges[1].kind, "offset_clone:continuation")
        self.assertEqual(stats["base_nodes"], 2)
        self.assertEqual(stats["base_edges"], 1)
        self.assertEqual(stats["cloned_nodes"], 2)
        self.assertEqual(stats["cloned_edges"], 1)
        self.assertEqual(stats["clipped_nodes"], 2)


    def test_offset_clone_component_filter_keeps_supported_contrast_chain(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        array = np.arange(3 * 4 * 6 * 8, dtype=np.float32).reshape(3, 4, 6, 8)
        nodes = [
            CandidateNode("sample", 1, 0, 1, 2, 1, 1.0, ("tile_contrast@q0.75",), 1.0),
            CandidateNode("sample", 2, 1, 1, 2, 1, 1.0, ("tile_contrast@q0.75",), 1.0),
            CandidateNode("sample", 3, 2, 1, 2, 1, 1.0, ("tile_contrast@q0.75",), 1.0),
            CandidateNode("sample", 4, 0, 1, 2, 1, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 5, 1, 1, 2, 1, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 6, 2, 1, 2, 1, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 7, 0, 1, 2, 7, 1.0, ("tile_contrast@q0.75",), 1.0),
            CandidateNode("sample", 8, 1, 1, 2, 7, 1.0, ("tile_contrast@q0.75",), 1.0),
        ]
        edges = [
            CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(2, 3, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(4, 5, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(5, 6, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(7, 8, 1, distance_um=1.0, cost=0.2, score=0.8),
        ]
        config = RouteConfig(
            label="synthetic_filtered_offset_clone",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=8,
            max_link_um=5.0,
            enable_offset_clone_layers=True,
            offset_clone_offsets_zyx=((0, 0, 0),),
            offset_clone_filter_source_mode="contrast_only",
            offset_clone_filter_min_component_timepoints=3,
            offset_clone_filter_component_max_x=5,
        )
        cloned_nodes, cloned_edges, stats = _clone_selected_nodes_by_offset_layers(
            nodes, edges, array, 0, config
        )
        cloned_only = [node for node in cloned_nodes if "offset_clone_0" in node.sources]
        cloned_pairs = [
            (edge.source_id, edge.target_id)
            for edge in cloned_edges
            if str(edge.kind).startswith("offset_clone:")
        ]
        self.assertEqual(len(cloned_only), 3)
        self.assertEqual(len(cloned_pairs), 2)
        self.assertTrue(all(any(source.startswith("tile_contrast@") for source in node.sources) for node in cloned_only))
        self.assertEqual(stats["filter"]["enabled"], True)
        self.assertEqual(stats["filter"]["layers"][0]["kept_nodes"], 3)
        self.assertEqual(stats["filter"]["layers"][0]["kept_edges"], 2)


    def test_submission_component_pruning_drops_short_and_low_x_components(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        nodes = [
            CandidateNode("sample", 1, 0, 0, 20, 20, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 2, 1, 0, 21, 20, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 3, 2, 0, 22, 20, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 4, 0, 0, 4, 4, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 5, 1, 0, 5, 4, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 6, 0, 0, 30, 30, 1.0, ("tile_raw@q0.75", "offset_clone_0"), 1.0),
            CandidateNode("sample", 7, 1, 0, 31, 30, 1.0, ("tile_raw@q0.75", "offset_clone_0"), 1.0),
        ]
        edges = [
            CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(2, 3, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(4, 5, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(6, 7, 1, distance_um=1.0, cost=0.2, score=0.8),
        ]
        config = RouteConfig(
            label="synthetic_component_prune",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=8,
            max_link_um=5.0,
            enable_component_pruning=True,
            component_prune_min_edges=2,
            component_prune_min_x=16,
            component_prune_drop_one_edge_offset=True,
        )
        pruned_nodes, pruned_edges, stats = _prune_submission_components(nodes, edges, config)
        self.assertEqual([node.node_id for node in pruned_nodes], [1, 2, 3])
        self.assertEqual([(edge.source_id, edge.target_id) for edge in pruned_edges], [(1, 2), (2, 3)])
        self.assertEqual(stats["removed_components"], 2)
        self.assertEqual(stats["removal_reasons"], {"min_edges": 2, "min_x": 1, "one_edge_offset": 1})

    def test_submission_component_pruning_can_apply_x_gate_to_offset_components_only(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        nodes = [
            CandidateNode("sample", 1, 0, 0, 4, 4, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 2, 1, 0, 5, 4, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 3, 0, 0, 6, 6, 1.0, ("tile_raw@q0.75", "offset_clone_0"), 1.0),
            CandidateNode("sample", 4, 1, 0, 7, 6, 1.0, ("tile_raw@q0.75", "offset_clone_0"), 1.0),
            CandidateNode("sample", 5, 0, 0, 24, 24, 1.0, ("tile_raw@q0.75", "offset_clone_1"), 1.0),
            CandidateNode("sample", 6, 1, 0, 25, 24, 1.0, ("tile_raw@q0.75", "offset_clone_1"), 1.0),
        ]
        edges = [
            CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(3, 4, 1, distance_um=1.0, cost=0.2, score=0.8),
            CandidateEdge(5, 6, 1, distance_um=1.0, cost=0.2, score=0.8),
        ]
        config = RouteConfig(
            label="synthetic_offset_only_component_prune",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=8,
            max_link_um=5.0,
            enable_component_pruning=True,
            component_prune_min_x=16,
            component_prune_x_offset_only=True,
        )
        pruned_nodes, pruned_edges, stats = _prune_submission_components(nodes, edges, config)
        self.assertEqual([node.node_id for node in pruned_nodes], [1, 2, 5, 6])
        self.assertEqual([(edge.source_id, edge.target_id) for edge in pruned_edges], [(1, 2), (5, 6)])
        self.assertEqual(stats["removed_components"], 1)
        self.assertEqual(stats["removal_reasons"], {"min_x": 1})
        self.assertTrue(stats["x_offset_only"])


    def test_localmax_clone_target_rewire_prefers_unlinked_base_chain_target(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        nodes = [
            CandidateNode("sample", 1, 0, 0, 0, 0, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 2, 1, 0, 3, 0, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 3, 2, 0, 4, 0, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 4, 0, 0, 1, 0, 1.0, ("tile_raw@q0.75", "localmax_clone_0"), 1.0),
            CandidateNode("sample", 5, 1, 0, 2, 0, 1.0, ("tile_raw@q0.75", "localmax_clone_0"), 1.0),
        ]
        edges = [
            CandidateEdge(4, 5, 1, distance_um=1.0, cost=0.2, score=0.8, kind="localmax_clone:continuation"),
            CandidateEdge(2, 3, 1, distance_um=1.0, cost=0.2, score=0.8),
        ]
        config = RouteConfig(
            label="synthetic_localmax_rewire",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=5,
            max_link_um=5.0,
            enable_localmax_clone_target_rewire=True,
            localmax_clone_rewire_target_radius_um=1.5,
            localmax_clone_rewire_source_radius_um=2.5,
        )
        rewired_nodes, rewired_edges, stats = _rewire_localmax_clone_targets(
            nodes, edges, (1.0, 1.0, 1.0), config
        )
        self.assertEqual(sorted((edge.source_id, edge.target_id) for edge in rewired_edges), [(2, 3), (4, 2)])
        self.assertEqual(sorted(node.node_id for node in rewired_nodes), [2, 3, 4])
        self.assertEqual(stats["replacements"], 1)
        self.assertEqual(stats["rewired"][0]["old_target_id"], 5)
        self.assertEqual(stats["rewired"][0]["new_target_id"], 2)

    def test_base_chain_target_rewire_prefers_unlinked_forward_chain_target(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        nodes = [
            CandidateNode("sample", 1, 0, 0, 0, 0, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 2, 1, 0, 4, 0, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 3, 2, 0, 5, 0, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 4, 1, 0, 2, 0, 1.0, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 5, 2, 0, 3, 0, 1.0, ("tile_raw@q0.75",), 1.0),
        ]
        edges = [
            CandidateEdge(1, 2, 1, distance_um=4.0, cost=0.2, score=0.8),
            CandidateEdge(4, 5, 1, distance_um=1.0, cost=0.2, score=0.8),
        ]
        config = RouteConfig(
            label="synthetic_base_chain_rewire",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=5,
            max_link_um=5.0,
            enable_base_chain_target_rewire=True,
            base_chain_rewire_target_radius_um=2.5,
            base_chain_rewire_source_radius_um=2.5,
        )
        rewired_nodes, rewired_edges, stats = _rewire_base_chain_targets(
            nodes, edges, (1.0, 1.0, 1.0), config
        )
        self.assertEqual(sorted((edge.source_id, edge.target_id) for edge in rewired_edges), [(1, 4), (4, 5)])
        self.assertEqual(sorted(node.node_id for node in rewired_nodes), [1, 4, 5])
        self.assertEqual(stats["replacements"], 1)
        self.assertEqual(stats["rewired"][0]["old_target_id"], 2)
        self.assertEqual(stats["rewired"][0]["new_target_id"], 4)

    def test_final_node_pruning_removes_only_unlinked_edge_endpoints(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode

        nodes = [
            CandidateNode("sample", 1, 0, 0, 0, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 2, 1, 0, 1, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 3, 1, 0, 9, 0, 1.0, ("unused",), 1.0),
        ]
        edges = [CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.1, score=0.9)]

        pruned_nodes, stats = _prune_unlinked_submission_nodes(nodes, edges)

        self.assertEqual([node.node_id for node in pruned_nodes], [1, 2])
        self.assertEqual(stats["before_nodes"], 3)
        self.assertEqual(stats["after_nodes"], 2)
        self.assertEqual(stats["removed_nodes"], 1)

    def test_final_node_pruning_keeps_nodes_when_edge_graph_is_empty(self) -> None:
        from celltrack_alpha.types import CandidateNode

        nodes = [
            CandidateNode("sample", 1, 0, 0, 0, 0, 1.0, ("a",), 1.0),
            CandidateNode("sample", 2, 0, 0, 2, 0, 1.0, ("b",), 1.0),
        ]

        pruned_nodes, stats = _prune_unlinked_submission_nodes(nodes, [])

        self.assertEqual([node.node_id for node in pruned_nodes], [1, 2])
        self.assertEqual(stats["removed_nodes"], 0)


    def test_endpoint_backward_clone_prefers_x_boundary_predecessor(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        array = np.zeros((3, 4, 8, 8), dtype=np.float32)
        array[0, 1, 3, 0] = 13.0
        nodes = [
            CandidateNode("sample", 1, 1, 1, 7, 3, 0.5, ("a",), 1.0),
            CandidateNode("sample", 2, 1, 1, 3, 1, 0.5, ("a",), 1.0),
            CandidateNode("sample", 3, 2, 1, 5, 3, 0.5, ("a",), 1.0),
            CandidateNode("sample", 4, 2, 1, 3, 4, 0.5, ("a",), 1.0),
        ]
        edges = [
            CandidateEdge(1, 3, 1, distance_um=2.0, cost=0.1, score=0.9),
            CandidateEdge(2, 4, 1, distance_um=3.0, cost=0.1, score=0.9),
        ]
        config = RouteConfig(
            label="synthetic_endpoint_backward",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=5,
            max_link_um=5.0,
            enable_endpoint_backward_clones=True,
            endpoint_backward_max_edge_um=4.0,
            endpoint_backward_duplicate_um=0.0,
            endpoint_backward_max_per_t=1,
            endpoint_backward_axis_mode="x_preferred",
        )
        cloned_nodes, cloned_edges, stats = _add_endpoint_backward_clones(
            nodes, edges, array, 0, (1.0, 1.0, 1.0), config
        )
        self.assertEqual(len(cloned_nodes), 5)
        self.assertEqual(len(cloned_edges), 3)
        self.assertEqual(cloned_nodes[-1].coord_zyx, (1, 3, 0))
        self.assertEqual((cloned_edges[-1].source_id, cloned_edges[-1].target_id), (5, 2))
        self.assertEqual(stats["candidate_count"], 2)
        self.assertEqual(stats["cloned_nodes"], 1)
        self.assertEqual(stats["new_per_t"], {"0": 1})


    def test_endpoint_backward_clone_interior_mode_ranks_by_duplicate_distance(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        array = np.zeros((3, 4, 16, 16), dtype=np.float32)
        array[0, 1, 9, 10] = 17.0
        nodes = [
            CandidateNode("sample", 1, 0, 1, 1, 2, 0.5, ("existing",), 1.0),
            CandidateNode("sample", 2, 1, 1, 2, 2, 0.5, ("a",), 1.0),
            CandidateNode("sample", 3, 2, 1, 3, 2, 0.5, ("a",), 1.0),
            CandidateNode("sample", 4, 1, 1, 10, 10, 0.5, ("b",), 1.0),
            CandidateNode("sample", 5, 2, 1, 11, 10, 0.5, ("b",), 1.0),
        ]
        edges = [
            CandidateEdge(2, 3, 1, distance_um=1.0, cost=0.1, score=0.9),
            CandidateEdge(4, 5, 1, distance_um=1.0, cost=0.1, score=0.9),
        ]
        config = RouteConfig(
            label="synthetic_endpoint_backward_interior",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=5,
            max_link_um=5.0,
            enable_endpoint_backward_clones=True,
            endpoint_backward_max_edge_um=2.0,
            endpoint_backward_min_edge_um=0.25,
            endpoint_backward_duplicate_um=1.0,
            endpoint_backward_max_per_t=1,
            endpoint_backward_axis_mode="interior_dup",
        )
        cloned_nodes, cloned_edges, stats = _add_endpoint_backward_clones(
            nodes, edges, array, 0, (1.0, 1.0, 1.0), config
        )
        self.assertEqual(len(cloned_nodes), 6)
        self.assertEqual(len(cloned_edges), 3)
        self.assertEqual(cloned_nodes[-1].coord_zyx, (1, 9, 10))
        self.assertEqual((cloned_edges[-1].source_id, cloned_edges[-1].target_id), (6, 4))
        self.assertEqual(stats["axis_mode"], "interior_dup")
        self.assertEqual(stats["min_edge_um"], 0.25)
        self.assertEqual(stats["candidate_count"], 1)
        self.assertEqual(stats["cloned_nodes"], 1)
        self.assertGreater(stats["clones"][0]["duplicate_um"], 1.0)


    def test_endpoint_forward_clone_extends_unlinked_endpoint(self) -> None:
        from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

        array = np.zeros((2, 5, 7, 7), dtype=np.float32)
        array[1, 2, 3, 2] = 11.0
        nodes = [
            CandidateNode("sample", 1, 0, 2, 2, 2, 0.5, ("a",), 1.0),
            CandidateNode("sample", 2, 0, 4, 4, 4, 0.5, ("a",), 1.0),
            CandidateNode("sample", 3, 1, 4, 4, 5, 0.5, ("a",), 1.0),
        ]
        edges = [CandidateEdge(2, 3, 1, distance_um=1.0, cost=0.1, score=0.9)]
        config = RouteConfig(
            label="synthetic_endpoint_forward",
            detector_sources=("raw",),
            threshold_quantiles=(0.9,),
            min_distance_um=1.0,
            fusion_radius_um=1.0,
            max_points_per_t=5,
            max_link_um=5.0,
            enable_endpoint_forward_clones=True,
            endpoint_forward_radius_um=2.1,
            endpoint_forward_sources=("raw",),
            endpoint_forward_max_per_t=1,
        )
        cloned_nodes, cloned_edges, stats = _add_endpoint_forward_clones(
            nodes, edges, array, None, 0, (1.0, 1.0, 1.0), config
        )
        self.assertEqual(len(cloned_nodes), 4)
        self.assertEqual(len(cloned_edges), 2)
        self.assertEqual(cloned_nodes[-1].coord_zyx, (2, 3, 2))
        self.assertEqual((cloned_edges[-1].source_id, cloned_edges[-1].target_id), (1, 4))
        self.assertEqual(stats["cloned_nodes"], 1)
        self.assertEqual(stats["cloned_edges"], 1)

    def test_balanced_local_route_keeps_spatially_isolated_moderate_peak(self) -> None:
        from celltrack_alpha.proposals import detect_candidates_one_timepoint
        from celltrack_alpha.routing import default_route_configs

        volume = np.zeros((8, 64, 64), dtype=np.float32)
        # Bright cluster in one tile.
        for i, value in enumerate((100.0, 98.0, 96.0, 94.0, 92.0, 90.0)):
            volume[2, 8 + i * 2, 8 + i * 2] = value
        # Moderate but spatially isolated peak in a different tile.
        volume[5, 52, 52] = 40.0
        config = default_route_configs()["balanced_local"]
        nodes, _ = detect_candidates_one_timepoint(
            volume=volume,
            dataset="synthetic",
            t=0,
            attrs=None,
            scale_zyx=(1.625, 0.40625, 0.40625),
            config=config,
            next_node_id=1,
        )
        coords = {node.coord_zyx for node in nodes}
        self.assertIn((5, 52, 52), coords)
        kept = next(node for node in nodes if node.coord_zyx == (5, 52, 52))
        self.assertTrue(any(source.startswith("tile_") for source in kept.sources))

    def test_score_coverage_route_uses_raw_coverage_sources(self) -> None:
        from celltrack_alpha.proposals import detect_candidates_one_timepoint
        from celltrack_alpha.routing import default_route_configs

        volume = np.zeros((8, 64, 64), dtype=np.float32)
        volume[4, 40, 48] = 50.0
        config = default_route_configs()["score_coverage_compact"]
        nodes, _ = detect_candidates_one_timepoint(
            volume=volume,
            dataset="synthetic",
            t=0,
            attrs=None,
            scale_zyx=(1.625, 0.40625, 0.40625),
            config=config,
            next_node_id=1,
        )
        self.assertTrue(nodes)
        all_sources = {source for node in nodes for source in node.sources}
        self.assertTrue(any(source.startswith("cover_raw") for source in all_sources))
        self.assertTrue(any(source.startswith("cover_smooth") for source in all_sources))

    def test_axial_rescue_route_emits_shifted_dog_sources(self) -> None:
        from celltrack_alpha.proposals import detect_candidates_one_timepoint
        from celltrack_alpha.routing import default_route_configs

        volume = np.zeros((12, 48, 48), dtype=np.float32)
        volume[8, 24, 24] = 100.0
        config = default_route_configs()["axial_rescue_compact"]
        nodes, _ = detect_candidates_one_timepoint(
            volume=volume,
            dataset="synthetic",
            t=0,
            attrs=None,
            scale_zyx=(1.0, 1.0, 1.0),
            config=config,
            next_node_id=1,
        )
        shifted_nodes = [
            node
            for node in nodes
            if any(source.startswith("zshift_dog_") for source in node.sources)
        ]
        self.assertTrue(shifted_nodes)
        self.assertTrue(any(node.z < 8 for node in shifted_nodes))

    def test_balanced_axial_route_keeps_tile_and_shifted_sources(self) -> None:
        from celltrack_alpha.proposals import detect_candidates_one_timepoint
        from celltrack_alpha.routing import default_route_configs

        volume = np.zeros((12, 48, 48), dtype=np.float32)
        volume[8, 24, 24] = 100.0
        volume[4, 36, 36] = 60.0
        config = default_route_configs()["balanced_axial_compact"]
        nodes, _ = detect_candidates_one_timepoint(
            volume=volume,
            dataset="synthetic",
            t=0,
            attrs=None,
            scale_zyx=(1.0, 1.0, 1.0),
            config=config,
            next_node_id=1,
        )
        all_sources = {source for node in nodes for source in node.sources}
        self.assertTrue(any(source.startswith("tile_") for source in all_sources))
        self.assertTrue(any(source.startswith("zshift_dog_") for source in all_sources))

    def test_template_point_route_uses_calibrated_template_source(self) -> None:
        from celltrack_alpha.proposals import detect_candidates_one_timepoint
        from celltrack_alpha.routing import default_route_configs

        volume = np.zeros((7, 21, 21), dtype=np.float32)
        volume[3, 10, 10] = 10.0
        kernel = np.zeros((3, 5, 5), dtype=float)
        kernel[1, 2, 2] = 1.0
        policy = {
            "templates": [
                {
                    "name": "single_peak",
                    "source": "template_score:single_peak",
                    "kernel": kernel.tolist(),
                    "score_quantiles": [0.99],
                    "max_points": 3,
                    "min_distance_um": 1.0,
                }
            ]
        }
        config = default_route_configs()["template_point_compact"]
        nodes, _ = detect_candidates_one_timepoint(
            volume=volume,
            dataset="synthetic",
            t=0,
            attrs=None,
            scale_zyx=(1.0, 1.0, 1.0),
            config=config,
            next_node_id=1,
            proposal_policy=policy,
        )
        template_nodes = [
            node
            for node in nodes
            if any(source.startswith("template_score:single_peak") for source in node.sources)
        ]
        self.assertTrue(template_nodes)

    def test_scalar_and_malformed_output_offsets_raise_clear_value_error(self) -> None:
        array = np.zeros((1, 2, 4, 5), dtype=np.uint16)
        for value in (1, "1,2,3", (1, 2), (1, 2, False), (1.0, 2, 3)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "exactly three integers"):
                    predict_dataset_window(
                        "sample",
                        array,
                        None,
                        output_offset_override=value,  # type: ignore[arg-type]
                    )

    def test_volume_validation_rejects_nonfinite_and_nonreal_data(self) -> None:
        valid = np.zeros((2, 3, 4), dtype=np.float32)
        self.assertIs(_validate_volume(valid), valid)
        invalid = valid.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            _validate_volume(invalid)
        for value in (np.inf, -np.inf):
            invalid = valid.copy()
            invalid[0, 0, 0] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                    _validate_volume(invalid)
        with self.assertRaisesRegex(ValueError, "real numeric"):
            _validate_volume(np.zeros((2, 3, 4), dtype=np.complex64))


if __name__ == "__main__":
    unittest.main()
