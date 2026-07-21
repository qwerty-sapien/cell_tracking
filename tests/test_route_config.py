from __future__ import annotations

import math
import re
import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.routing import default_route_configs  # noqa: E402
from celltrack_alpha.types import (  # noqa: E402
    RouteConfig,
    _DEFAULT_ROUTE_LABELS,
    validate_route_config,
)


def base_config() -> RouteConfig:
    return RouteConfig(
        label="synthetic",
        detector_sources=("synthetic",),
        threshold_quantiles=(0.9,),
        min_distance_um=1.0,
        fusion_radius_um=1.0,
        max_points_per_t=20,
        max_link_um=10.0,
    )


class RouteConfigTests(unittest.TestCase):
    def test_all_default_routes_validate(self) -> None:
        configs = default_route_configs()

        self.assertEqual(len(configs), 20)
        self.assertEqual(set(configs), _DEFAULT_ROUTE_LABELS)
        self.assertTrue(all(label == config.label for label, config in configs.items()))
        for config in configs.values():
            validate_route_config(config)

    def test_edge_only_config_allows_empty_threshold_quantiles(self) -> None:
        config = replace(base_config(), threshold_quantiles=())

        validate_route_config(config)

    def test_every_public_field_rejects_a_malformed_value(self) -> None:
        invalid_by_field: dict[str, object] = {
            "label": "",
            "detector_sources": [],
            "threshold_quantiles": (math.nan,),
            "min_distance_um": -1.0,
            "fusion_radius_um": -1.0,
            "max_points_per_t": 0,
            "max_link_um": 0.0,
            "max_gap": 0,
            "gap_link_um": 0.0,
            "max_neighbors": 0,
            "edge_cost_threshold": -1.0,
            "birth_cost": -1.0,
            "death_cost": -1.0,
            "node_score_weight": -1.0,
            "density_cost_weight": -1.0,
            "appearance_cost_weight": -1.0,
            "gap_motion_cost_weight": -1.0,
            "enable_divisions": 1,
            "division_cost_threshold": -1.0,
            "division_requires_unlinked_parent": 1,
            "division_requires_child_future": 1,
            "division_child_max_um": 0.0,
            "division_child_min_sep_um": -1.0,
            "division_child_max_sep_um": 0.0,
            "division_allow_link_steal": 1,
            "division_steal_max_distance_um": -1.0,
            "division_steal_source_max_um": -1.0,
            "enable_second_choice_edge_rescue": 1,
            "edge_rescue_short_max_um": -1.0,
            "edge_rescue_long_min_um": -1.0,
            "edge_rescue_long_max_um": -1.0,
            "edge_rescue_margin_min_um": -1.0,
            "edge_rescue_intensity_ratio": -1.0,
            "edge_rescue_alt_cost_max": -1.0,
            "enable_localmax_refinement": 1,
            "localmax_refine_radius_um": 0.0,
            "localmax_refine_sources": ("unsupported",),
            "localmax_refine_center_offset_zyx": (0, 0),
            "localmax_refine_center_offsets_zyx": ((0, 0),),
            "enable_localmax_clone_layers": 1,
            "localmax_clone_radius_um": 0.0,
            "localmax_clone_sources": ("unsupported",),
            "localmax_clone_center_offsets_zyx": ((0, 0),),
            "enable_localmax_clone_target_rewire": 1,
            "localmax_clone_rewire_target_radius_um": 0.0,
            "localmax_clone_rewire_source_radius_um": 0.0,
            "enable_base_chain_target_rewire": 1,
            "base_chain_rewire_target_radius_um": 0.0,
            "base_chain_rewire_source_radius_um": 0.0,
            "enable_offset_clone_layers": 1,
            "offset_clone_offsets_zyx": ((0, 0),),
            "offset_clone_skip_clipped": 1,
            "offset_clone_filter_source_mode": "unsupported",
            "offset_clone_filter_min_component_timepoints": -1,
            "offset_clone_filter_component_max_x": -1,
            "enable_endpoint_forward_clones": 1,
            "endpoint_forward_radius_um": 0.0,
            "endpoint_forward_sources": ("unsupported",),
            "endpoint_forward_max_per_t": 0,
            "endpoint_forward_requires_history": 1,
            "enable_endpoint_backward_clones": 1,
            "endpoint_backward_max_edge_um": 0.0,
            "endpoint_backward_min_edge_um": -1.0,
            "endpoint_backward_duplicate_um": -1.0,
            "endpoint_backward_max_per_t": 0,
            "endpoint_backward_axis_mode": "unsupported",
            "enable_refined_candidate_admission": 1,
            "refined_candidate_admission_route": "",
            "refined_candidate_admission_max_per_t": 0,
            "refined_candidate_admission_min_dist_um": -1.0,
            "refined_candidate_admission_score_scale": 0.0,
            "enable_refined_chain_conflict_rescue": 1,
            "refined_chain_conflict_max_distance_um": 0.0,
            "refined_chain_conflict_max_cost": math.nan,
            "refined_chain_conflict_max_target_rank": 0,
            "refined_chain_conflict_min_temporal_count": -1.0,
            "enable_short_edge_pruning": 1,
            "short_edge_prune_min_distance_um": -1.0,
            "enable_free_edge_admission": 1,
            "free_edge_admission_max_cost": math.nan,
            "free_edge_admission_min_distance_um": -1.0,
            "free_edge_admission_max_distance_um": 0.0,
            "free_edge_admission_require_mutual_nearest": 1,
            "free_edge_admission_max_source_candidates": 0,
            "free_edge_admission_max_target_candidates": 0,
            "free_edge_admission_max_edges": 0,
            "enable_duplicate_endpoint_rescue": 1,
            "duplicate_endpoint_rescue_max_um": -1.0,
            "duplicate_endpoint_rescue_min_cost": math.nan,
            "duplicate_endpoint_rescue_max_cost": math.nan,
            "duplicate_endpoint_rescue_max_distance_um": 0.0,
            "duplicate_endpoint_rescue_max_swaps": 0,
            "enable_component_pruning": 1,
            "component_prune_min_edges": -1,
            "component_prune_min_x": -1,
            "component_prune_max_x": -1,
            "component_prune_x_offset_only": 1,
            "component_prune_drop_one_edge_offset": 1,
            "enable_motion_endpoint_admission": 1,
            "motion_endpoint_pred_max_um": 0.0,
            "motion_endpoint_source_max_um": 0.0,
            "motion_endpoint_max_edges_per_t": 0,
            "motion_endpoint_target_source_mode": "unsupported",
            "motion_endpoint_rank_mode": "unsupported",
            "motion_endpoint_requires_history": 1,
            "output_offset_zyx": (0, 0),
            "prefix_output_offsets_zyx": (("", 0, 0, 0),),
            "notes": None,
        }
        field_names = {item.name for item in fields(RouteConfig)}
        self.assertEqual(set(invalid_by_field), field_names)

        config = base_config()
        for name, invalid in invalid_by_field.items():
            with self.subTest(name=name, invalid=invalid):
                with self.assertRaisesRegex(ValueError, re.escape(name)):
                    replace(config, **{name: invalid})

    def test_cross_field_invariants_fail_at_construction(self) -> None:
        cases = [
            (
                {"division_child_min_sep_um": 12.0, "division_child_max_sep_um": 11.0},
                "division_child_min_sep_um",
            ),
            (
                {"edge_rescue_long_min_um": 8.0, "edge_rescue_long_max_um": 7.0},
                "edge_rescue_long_min_um",
            ),
            (
                {"endpoint_backward_min_edge_um": 7.0, "endpoint_backward_max_edge_um": 6.0},
                "endpoint_backward_min_edge_um",
            ),
            (
                {
                    "free_edge_admission_min_distance_um": 5.0,
                    "free_edge_admission_max_distance_um": 4.0,
                },
                "free_edge_admission_min_distance_um",
            ),
            (
                {"duplicate_endpoint_rescue_min_cost": 0.6, "duplicate_endpoint_rescue_max_cost": 0.5},
                "duplicate_endpoint_rescue_min_cost",
            ),
            (
                {"component_prune_min_x": 10, "component_prune_max_x": 9},
                "component_prune_min_x",
            ),
            (
                {"enable_localmax_clone_layers": True, "localmax_clone_center_offsets_zyx": ()},
                "localmax_clone_center_offsets_zyx",
            ),
            (
                {"enable_offset_clone_layers": True, "offset_clone_offsets_zyx": ()},
                "offset_clone_offsets_zyx",
            ),
            (
                {"detector_sources": ("tile_raw",), "threshold_quantiles": ()},
                "threshold_quantiles",
            ),
            (
                {
                    "enable_refined_candidate_admission": True,
                    "refined_candidate_admission_route": "missing_route",
                },
                "refined_candidate_admission_route",
            ),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    replace(base_config(), **overrides)

    def test_explicit_validation_rechecks_a_low_level_mutation(self) -> None:
        config = base_config()
        object.__setattr__(config, "max_gap", 0)

        with self.assertRaisesRegex(ValueError, "max_gap"):
            validate_route_config(config)


if __name__ == "__main__":
    unittest.main()
