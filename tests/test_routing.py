from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.routing import default_route_configs, route_for_profile  # noqa: E402
from celltrack_alpha.types import ScenarioProfile  # noqa: E402


class RoutingTests(unittest.TestCase):
    def test_crowded_high_foreground_prefers_balanced_local(self) -> None:
        profile = ScenarioProfile(
            dataset="synthetic",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.18,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.5,
            temporal_delta=0.03,
            division_motif_rate=0.2,
            intensity_p999=2000.0,
            estimated_total_nodes=50000.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_compact")
        self.assertIn("spatially balanced", decision.reason)


    def test_low_foreground_saturated_peak_field_stays_clean_until_edge_scorer(self) -> None:
        profile = ScenarioProfile(
            dataset="synthetic",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.014,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.018,
            division_motif_rate=0.6,
            intensity_p999=756.0,
            estimated_total_nodes=6000.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "clean_point")


    def test_very_low_foreground_6bba_saturated_field_prefers_clean_compact(self) -> None:
        profile = ScenarioProfile(
            dataset="6bba_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.014,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.018,
            division_motif_rate=0.6,
            intensity_p999=756.0,
            estimated_total_nodes=None,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "clean_compact")

    def test_low_estimated_6bba_prefers_midpoint_after_endpoint_emission(self) -> None:
        profile = ScenarioProfile(
            dataset="6bba_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.047,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.027,
            division_motif_rate=0.54,
            intensity_p999=670.0,
            estimated_total_nodes=5534.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_midpoint")
        self.assertEqual(decision.config.max_link_um, 6.0)
        self.assertEqual(decision.config.max_neighbors, 1)
        self.assertEqual(decision.config.edge_cost_threshold, 0.45)
        self.assertEqual(decision.config.birth_cost, 0.18)
        self.assertEqual(decision.config.death_cost, 0.18)
        self.assertTrue(decision.config.enable_localmax_refinement)
        self.assertEqual(decision.config.localmax_refine_center_offsets_zyx, ((-3, 1, -2),))
        self.assertTrue(decision.config.enable_endpoint_forward_clones)
        self.assertEqual(decision.config.endpoint_forward_radius_um, 2.5)
        self.assertEqual(decision.config.endpoint_forward_sources, ("smooth",))
        self.assertEqual(decision.config.endpoint_forward_max_per_t, 4)
        self.assertFalse(decision.config.endpoint_forward_requires_history)
        self.assertTrue(decision.config.enable_offset_clone_layers)
        self.assertEqual(decision.config.offset_clone_offsets_zyx, ((3, 6, -24),))
        self.assertTrue(decision.config.offset_clone_skip_clipped)
        self.assertEqual(decision.config.offset_clone_filter_source_mode, "raw_only")
        self.assertEqual(decision.config.offset_clone_filter_component_max_x, 64)
        self.assertTrue(decision.config.enable_component_pruning)
        self.assertEqual(decision.config.component_prune_min_x, 16)
        self.assertIn("low-estimated-node", decision.reason)

    def test_very_low_foreground_low_estimated_6bba_prefers_midpoint(self) -> None:
        profile = ScenarioProfile(
            dataset="6bba_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.014,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.019,
            division_motif_rate=0.665,
            intensity_p999=756.0,
            estimated_total_nodes=6030.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_midpoint")
        self.assertEqual(decision.config.max_link_um, 3.5)
        self.assertEqual(decision.config.max_neighbors, 1)
        self.assertEqual(decision.config.edge_cost_threshold, 0.55)
        self.assertEqual(decision.config.birth_cost, 0.10)
        self.assertEqual(decision.config.death_cost, 0.38)
        self.assertFalse(decision.config.enable_localmax_refinement)
        self.assertTrue(decision.config.enable_refined_candidate_admission)
        self.assertEqual(decision.config.refined_candidate_admission_route, "refined_dog_compact")
        self.assertEqual(decision.config.refined_candidate_admission_max_per_t, 8)
        self.assertEqual(decision.config.refined_candidate_admission_min_dist_um, 1.0)
        self.assertEqual(decision.config.refined_candidate_admission_score_scale, 0.65)
        self.assertTrue(decision.config.enable_refined_chain_conflict_rescue)
        self.assertEqual(decision.config.refined_chain_conflict_max_distance_um, 3.0)
        self.assertEqual(decision.config.refined_chain_conflict_max_cost, 0.42)
        self.assertEqual(decision.config.refined_chain_conflict_max_target_rank, 3)
        self.assertEqual(decision.config.refined_chain_conflict_min_temporal_count, 2.0)
        self.assertTrue(decision.config.enable_offset_clone_layers)
        self.assertEqual(decision.config.offset_clone_offsets_zyx, ((0, 24, 18), (-3, 42, -12)))
        self.assertTrue(decision.config.offset_clone_skip_clipped)
        self.assertEqual(decision.config.offset_clone_filter_component_max_x, 192)
        self.assertTrue(decision.config.enable_component_pruning)
        self.assertEqual(decision.config.component_prune_min_x, 16)
        self.assertIn("very-low-foreground low-estimated", decision.reason)

    def test_very_low_foreground_high_motif_6bba_prefers_midpoint(self) -> None:
        profile = ScenarioProfile(
            dataset="6bba_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.021,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.011,
            division_motif_rate=0.96,
            intensity_p999=641.0,
            estimated_total_nodes=7021.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_midpoint")
        self.assertEqual(decision.config.max_link_um, 5.0)
        self.assertEqual(decision.config.max_neighbors, 1)
        self.assertEqual(decision.config.edge_cost_threshold, 0.55)
        self.assertTrue(decision.config.enable_localmax_refinement)
        self.assertEqual(decision.config.localmax_refine_center_offsets_zyx, ((1, -1, 4),))
        self.assertTrue(decision.config.enable_localmax_clone_layers)
        self.assertEqual(decision.config.localmax_clone_center_offsets_zyx, ((-3, -2, -5),))
        self.assertTrue(decision.config.enable_localmax_clone_target_rewire)
        self.assertEqual(decision.config.localmax_clone_rewire_target_radius_um, 4.0)
        self.assertEqual(decision.config.localmax_clone_rewire_source_radius_um, 4.0)
        self.assertTrue(decision.config.enable_offset_clone_layers)
        self.assertEqual(decision.config.offset_clone_offsets_zyx, ((3, 48, -30), (12, 13, -37)))
        self.assertTrue(decision.config.offset_clone_skip_clipped)
        self.assertEqual(decision.config.offset_clone_filter_source_mode, "contrast_only")
        self.assertEqual(decision.config.offset_clone_filter_min_component_timepoints, 3)
        self.assertEqual(decision.config.offset_clone_filter_component_max_x, 192)
        self.assertTrue(decision.config.enable_component_pruning)
        self.assertEqual(decision.config.component_prune_min_edges, 2)
        self.assertIn("very-low-foreground high-motif", decision.reason)

    def test_very_low_foreground_division_window_6bba_prefers_clean_rescue(self) -> None:
        profile = ScenarioProfile(
            dataset="6bba_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.003,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.004,
            division_motif_rate=0.75,
            intensity_p999=589.0,
            estimated_total_nodes=10836.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "clean_compact")
        self.assertTrue(decision.config.enable_divisions)
        self.assertTrue(decision.config.division_allow_link_steal)
        self.assertEqual(decision.config.division_child_min_sep_um, 9.0)
        self.assertIn("clean fork rescue", decision.reason)

    def test_d07_like_profile_sticks_to_clean_compact(self) -> None:
        profile = ScenarioProfile(
            dataset="6bba_0e7c0d07",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.0003254413604736328,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.003830095170997083,
            division_motif_rate=0.8475,
            intensity_p999=683.468,
            estimated_total_nodes=22467.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "clean_compact")
        self.assertFalse(decision.config.enable_divisions)
        self.assertFalse(decision.config.enable_component_pruning)

    def test_moderate_low_foreground_6bba_enables_second_choice_edge_rescue(self) -> None:
        profile = ScenarioProfile(
            dataset="6bba_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.0647,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.014,
            division_motif_rate=0.905,
            intensity_p999=1316.0,
            estimated_total_nodes=17909.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_compact")
        self.assertTrue(decision.config.enable_second_choice_edge_rescue)
        self.assertTrue(decision.config.enable_localmax_refinement)
        self.assertEqual(decision.config.localmax_refine_center_offsets_zyx, ((1, 2, 5),))
        self.assertTrue(decision.config.enable_endpoint_forward_clones)
        self.assertEqual(decision.config.endpoint_forward_sources, ("smooth",))
        self.assertEqual(decision.config.endpoint_forward_max_per_t, 16)
        self.assertTrue(decision.config.enable_offset_clone_layers)
        self.assertEqual(
            decision.config.offset_clone_offsets_zyx,
            (
                (0, 20, 17),
                (0, -36, 30),
                (-3, 48, -24),
                (-11, -19, -22),
                (4, 2, 19),
                (-9, -25, -18),
                (8, 0, 38),
                (-4, 32, 12),
            ),
        )
        self.assertTrue(decision.config.offset_clone_skip_clipped)
        self.assertEqual(decision.config.offset_clone_filter_min_component_timepoints, 3)
        self.assertEqual(decision.config.offset_clone_filter_component_max_x, 192)
        self.assertTrue(decision.config.enable_component_pruning)
        self.assertEqual(decision.config.component_prune_min_edges, 2)
        self.assertEqual(decision.config.component_prune_min_x, 32)
        self.assertTrue(decision.config.component_prune_x_offset_only)
        self.assertEqual(decision.config.edge_rescue_long_min_um, 6.0)
        self.assertEqual(decision.config.birth_cost, 0.15)
        self.assertEqual(decision.config.death_cost, 0.325)
        self.assertIn("second-choice edge rescue", decision.reason)

    def test_high_foreground_low_motif_44b6_balanced_compact_uses_strict_links(self) -> None:
        profile = ScenarioProfile(
            dataset="44b6_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.34,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.054,
            division_motif_rate=0.60,
            intensity_p999=2527.0,
            estimated_total_nodes=78644.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_compact")
        self.assertEqual(decision.config.max_link_um, 3.5)
        self.assertEqual(decision.config.max_neighbors, 1)
        self.assertEqual(decision.config.edge_cost_threshold, 0.55)
        self.assertEqual(decision.config.birth_cost, 0.10)
        self.assertEqual(decision.config.death_cost, 0.38)
        self.assertTrue(decision.config.enable_endpoint_backward_clones)
        self.assertEqual(decision.config.endpoint_backward_max_edge_um, 6.0)
        self.assertEqual(decision.config.endpoint_backward_duplicate_um, 5.5)
        self.assertEqual(decision.config.endpoint_backward_max_per_t, 1)
        self.assertEqual(decision.config.endpoint_backward_axis_mode, "x_preferred")
        self.assertIn("strict compact links", decision.reason)

    def test_moderate_high_foreground_44b6_compact_keeps_default_assignment_penalties(self) -> None:
        profile = ScenarioProfile(
            dataset="44b6_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.168,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.031,
            division_motif_rate=0.585,
            intensity_p999=2087.0,
            estimated_total_nodes=27958.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_compact")
        self.assertEqual(decision.config.max_link_um, 3.5)
        self.assertIsNone(decision.config.birth_cost)
        self.assertIsNone(decision.config.death_cost)
        self.assertFalse(decision.config.enable_endpoint_backward_clones)

    def test_endpoint_tuned_link_configs_are_stable(self) -> None:
        configs = default_route_configs()
        self.assertFalse(configs["clean_compact"].enable_divisions)
        self.assertFalse(configs["balanced_compact"].enable_second_choice_edge_rescue)
        midpoint = configs["balanced_midpoint"]
        self.assertEqual(midpoint.max_link_um, 6.0)
        self.assertEqual(midpoint.max_neighbors, 2)
        self.assertEqual(midpoint.edge_cost_threshold, 0.45)
        template = configs["template_point_compact"]
        self.assertEqual(template.max_link_um, 3.0)
        self.assertEqual(template.max_neighbors, 1)
        self.assertEqual(template.edge_cost_threshold, 0.25)

    def test_high_foreground_6bba_saturated_field_prefers_balanced_midpoint(self) -> None:
        profile = ScenarioProfile(
            dataset="6bba_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.26,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.028,
            division_motif_rate=0.9,
            intensity_p999=1000.0,
            estimated_total_nodes=None,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_midpoint")
        self.assertTrue(decision.config.enable_localmax_refinement)
        self.assertEqual(decision.config.localmax_refine_center_offsets_zyx, ((1, -5, -5),))
        self.assertTrue(decision.config.enable_endpoint_forward_clones)
        self.assertEqual(decision.config.endpoint_forward_radius_um, 2.5)
        self.assertEqual(decision.config.endpoint_forward_sources, ("smooth",))
        self.assertEqual(decision.config.endpoint_forward_max_per_t, 8)
        self.assertTrue(decision.config.enable_offset_clone_layers)
        self.assertEqual(decision.config.offset_clone_offsets_zyx, ((3, -24, -18), (3, 48, -30)))
        self.assertTrue(decision.config.offset_clone_skip_clipped)
        self.assertTrue(decision.config.enable_component_pruning)
        self.assertEqual(decision.config.component_prune_max_x, 224)
        self.assertIn("midpoint", decision.reason)

    def test_moderate_foreground_high_signal_6bba_prefers_balanced_dog_tight(self) -> None:
        profile = ScenarioProfile(
            dataset="6bba_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.13,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.025,
            division_motif_rate=0.77,
            intensity_p999=1500.0,
            estimated_total_nodes=None,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_dog_tight")
        self.assertIn("subtile_dog_abs", decision.config.detector_sources)
        self.assertTrue(decision.config.enable_localmax_refinement)
        self.assertEqual(decision.config.localmax_refine_radius_um, 2.5)
        self.assertEqual(decision.config.localmax_refine_sources, ("dog_abs",))
        self.assertEqual(decision.config.localmax_refine_center_offset_zyx, (-1, 3, 1))
        self.assertEqual(decision.config.localmax_refine_center_offsets_zyx, ((-1, 3, 1), (3, 6, -6)))
        self.assertEqual(decision.config.birth_cost, 0.18)
        self.assertEqual(decision.config.death_cost, 0.18)
        self.assertTrue(decision.config.enable_short_edge_pruning)
        self.assertEqual(decision.config.short_edge_prune_min_distance_um, 1.25)
        self.assertTrue(decision.config.enable_free_edge_admission)
        self.assertEqual(decision.config.free_edge_admission_max_cost, 0.55)
        self.assertEqual(decision.config.free_edge_admission_min_distance_um, 1.25)
        self.assertEqual(decision.config.free_edge_admission_max_distance_um, 4.5)
        self.assertTrue(decision.config.free_edge_admission_require_mutual_nearest)
        self.assertEqual(decision.config.free_edge_admission_max_source_candidates, 1)
        self.assertEqual(decision.config.free_edge_admission_max_target_candidates, 1)
        self.assertEqual(decision.config.free_edge_admission_max_edges, 32)
        self.assertTrue(decision.config.enable_second_choice_edge_rescue)
        self.assertEqual(decision.config.edge_rescue_short_max_um, 2.0)
        self.assertEqual(decision.config.edge_rescue_long_min_um, 3.0)
        self.assertEqual(decision.config.edge_rescue_long_max_um, 3.5)
        self.assertEqual(decision.config.edge_rescue_margin_min_um, 1.3)
        self.assertEqual(decision.config.edge_rescue_intensity_ratio, 0.90)
        self.assertEqual(decision.config.edge_rescue_alt_cost_max, 0.32)
        self.assertTrue(decision.config.enable_duplicate_endpoint_rescue)
        self.assertEqual(decision.config.duplicate_endpoint_rescue_max_um, 2.0)
        self.assertEqual(decision.config.duplicate_endpoint_rescue_min_cost, 0.25)
        self.assertEqual(decision.config.duplicate_endpoint_rescue_max_cost, 0.45)
        self.assertEqual(decision.config.duplicate_endpoint_rescue_max_distance_um, 4.5)
        self.assertEqual(decision.config.duplicate_endpoint_rescue_max_swaps, 4)
        self.assertTrue(decision.config.enable_base_chain_target_rewire)
        self.assertEqual(decision.config.base_chain_rewire_target_radius_um, 5.5)
        self.assertEqual(decision.config.base_chain_rewire_source_radius_um, 4.0)
        self.assertTrue(decision.config.enable_offset_clone_layers)
        self.assertEqual(decision.config.offset_clone_offsets_zyx, ((-3, 12, -24), (10, 12, 21)))
        self.assertTrue(decision.config.offset_clone_skip_clipped)
        self.assertTrue(decision.config.enable_component_pruning)
        self.assertEqual(decision.config.component_prune_min_x, 128)
        self.assertTrue(decision.config.component_prune_x_offset_only)
        self.assertTrue(decision.config.component_prune_drop_one_edge_offset)
        self.assertIn("tight balanced DoG", decision.reason)

    def test_low_foreground_high_motif_44b6_prefers_balanced_axial(self) -> None:
        profile = ScenarioProfile(
            dataset="44b6_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.071,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.011,
            division_motif_rate=0.95,
            intensity_p999=1721.0,
            estimated_total_nodes=24000.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_axial_compact")
        self.assertEqual(decision.config.max_link_um, 4.0)
        self.assertEqual(decision.config.max_neighbors, 5)
        self.assertEqual(decision.config.edge_cost_threshold, 0.65)
        self.assertTrue(decision.config.enable_motion_endpoint_admission)
        self.assertEqual(decision.config.motion_endpoint_pred_max_um, 7.5)
        self.assertEqual(decision.config.motion_endpoint_source_max_um, 10.0)
        self.assertEqual(decision.config.motion_endpoint_max_edges_per_t, 4)
        self.assertEqual(decision.config.motion_endpoint_target_source_mode, "zshift")
        self.assertEqual(decision.config.motion_endpoint_rank_mode, "source_order")
        self.assertTrue(decision.config.motion_endpoint_requires_history)
        self.assertIn("axial rescue", decision.reason)

    def test_low_estimated_axial_44b6_uses_strict_links(self) -> None:
        profile = ScenarioProfile(
            dataset="44b6_low_estimated",
            t_start=5,
            t_end=8,
            sampled_timepoints=(5, 6, 7),
            foreground_fraction=0.0665,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.0215,
            division_motif_rate=0.9125,
            intensity_p999=1362.0,
            estimated_total_nodes=18137.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_axial_compact")
        self.assertEqual(decision.config.max_link_um, 3.0)
        self.assertEqual(decision.config.max_neighbors, 1)
        self.assertEqual(decision.config.edge_cost_threshold, 0.25)
        self.assertEqual(decision.config.death_cost, 0.05)
        self.assertFalse(decision.config.enable_motion_endpoint_admission)
        self.assertTrue(decision.config.enable_endpoint_backward_clones)
        self.assertEqual(decision.config.endpoint_backward_max_edge_um, 4.0)
        self.assertEqual(decision.config.endpoint_backward_min_edge_um, 0.25)
        self.assertEqual(decision.config.endpoint_backward_duplicate_um, 2.0)
        self.assertEqual(decision.config.endpoint_backward_max_per_t, 1)
        self.assertEqual(decision.config.endpoint_backward_axis_mode, "interior_dup")
        self.assertIn("Strict axial links", decision.config.notes)

    def test_high_signal_high_motif_44b6_prefers_hybrid_rescue(self) -> None:
        profile = ScenarioProfile(
            dataset="44b6_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.133,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.017,
            division_motif_rate=0.86,
            intensity_p999=3434.0,
            estimated_total_nodes=32681.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "hybrid_rescue_compact")
        self.assertEqual(decision.config.max_link_um, 3.5)
        self.assertEqual(decision.config.max_neighbors, 1)
        self.assertEqual(decision.config.edge_cost_threshold, 0.55)
        self.assertEqual(decision.config.birth_cost, 0.10)
        self.assertEqual(decision.config.death_cost, 0.38)
        self.assertTrue(decision.config.enable_offset_clone_layers)
        self.assertEqual(decision.config.offset_clone_offsets_zyx, ((-6, 0, 38),))
        self.assertIn("hybrid rescue", decision.reason)

    def test_moderate_foreground_high_estimated_44b6_prefers_template_point(self) -> None:
        profile = ScenarioProfile(
            dataset="44b6_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.110,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.017,
            division_motif_rate=0.758,
            intensity_p999=2466.0,
            estimated_total_nodes=67688.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "template_point_compact")
        self.assertEqual(decision.config.max_link_um, 3.0)
        self.assertEqual(decision.config.max_neighbors, 1)
        self.assertEqual(decision.config.edge_cost_threshold, 0.25)
        self.assertTrue(decision.config.enable_motion_endpoint_admission)
        self.assertEqual(decision.config.motion_endpoint_pred_max_um, 7.0)
        self.assertEqual(decision.config.motion_endpoint_source_max_um, 7.0)
        self.assertEqual(decision.config.motion_endpoint_max_edges_per_t, 4)
        self.assertEqual(decision.config.motion_endpoint_target_source_mode, "same_source")
        self.assertEqual(decision.config.motion_endpoint_rank_mode, "source_desc")
        self.assertTrue(decision.config.motion_endpoint_requires_history)
        self.assertTrue(decision.config.enable_endpoint_backward_clones)
        self.assertEqual(decision.config.endpoint_backward_axis_mode, "interior_dup")
        self.assertEqual(decision.config.endpoint_backward_max_edge_um, 7.0)
        self.assertEqual(decision.config.endpoint_backward_duplicate_um, 2.0)
        self.assertEqual(decision.config.endpoint_backward_max_per_t, 24)
        self.assertIn("template point", decision.reason)

    def test_mid_foreground_saturated_field_prefers_balanced_compact(self) -> None:
        profile = ScenarioProfile(
            dataset="6bba_example",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.065,
            bright_candidate_density=400.0,
            dark_candidate_density=400.0,
            both_candidate_density=400.0,
            median_nn_um=0.40625,
            temporal_delta=0.014,
            division_motif_rate=0.9,
            intensity_p999=1300.0,
            estimated_total_nodes=None,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "balanced_compact")

    def test_clean_signal_still_prefers_clean_point(self) -> None:
        profile = ScenarioProfile(
            dataset="synthetic",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.02,
            bright_candidate_density=50.0,
            dark_candidate_density=45.0,
            both_candidate_density=60.0,
            median_nn_um=8.0,
            temporal_delta=0.01,
            division_motif_rate=0.0,
            intensity_p999=1000.0,
            estimated_total_nodes=1000.0,
        )
        decision = route_for_profile(profile, default_route_configs())
        self.assertEqual(decision.label, "clean_point")

    def test_auto_route_rejects_partial_config_map_with_clear_error(self) -> None:
        profile = ScenarioProfile(
            dataset="synthetic",
            t_start=0,
            t_end=3,
            sampled_timepoints=(0, 1, 2),
            foreground_fraction=0.02,
            bright_candidate_density=50.0,
            dark_candidate_density=45.0,
            both_candidate_density=60.0,
            median_nn_um=8.0,
            temporal_delta=0.01,
            division_motif_rate=0.0,
            intensity_p999=1000.0,
            estimated_total_nodes=1000.0,
        )
        config = default_route_configs()["clean_point"]
        with self.assertRaisesRegex(ValueError, "complete config map"):
            route_for_profile(profile, {"clean_point": config})
        forced = route_for_profile(
            profile, {"clean_point": config}, forced_label="clean_point"
        )
        self.assertEqual(forced.label, "clean_point")


if __name__ == "__main__":
    unittest.main()
