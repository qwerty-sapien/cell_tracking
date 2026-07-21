from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.scoring import (  # noqa: E402
    UnsafeScoringCalibrationError,
    correct_nodes_by_t,
    edge_feature_dict,
    load_scoring_calibration,
    node_feature_dict,
    prune_nodes_by_t,
    score_edge_candidates,
    scoring_for_dataset,
)
from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig  # noqa: E402


class ScoringCalibrationTests(unittest.TestCase):
    def write_split_and_calibration(self, tmp: Path, *, source: str = "train_a") -> tuple[Path, Path]:
        split = {
            "excluded_from_general_score": {"duplicate_public_ids": ["dup_public"]},
            "folds": {
                "0": {
                    "calibration_train": ["train_a", "train_b"],
                    "validation": ["val_a"],
                }
            },
        }
        split_path = tmp / "split.json"
        split_path.write_text(json.dumps(split), encoding="utf-8")
        calibration = {
            "kind": "candidate_graph_scoring",
            "provenance": {
                "status": "fold_safe",
                "fold_id": "0",
                "source_datasets": [source],
                "calibration_train": ["train_a", "train_b"],
                "validation_datasets": ["val_a"],
            },
            "node_policy": {
                "by_route": {
                    "balanced_compact": {
                        "weights": {"intensity": 0.01},
                        "bias": 0.0,
                        "keep_top_k_per_t": 1,
                    }
                }
            },
            "edge_policy": {
                "by_route": {
                    "balanced_compact": {
                        "weights": {"distance_norm": -2.0},
                        "bias": 1.0,
                        "drop_below_min_score": True,
                        "min_score": -1.0,
                        "probability_cost_weight": 0.5,
                        "replace_edge_score": True,
                    }
                }
            },
        }
        calibration_path = tmp / "scoring.json"
        calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
        return split_path, calibration_path

    def test_fold_safe_scoring_calibration_applies_only_to_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            split_path, calibration_path = self.write_split_and_calibration(Path(raw))
            calibration = load_scoring_calibration(
                calibration_path,
                split_manifest_path=split_path,
                expected_fold_id="0",
            )
            payload = scoring_for_dataset("val_a", calibration)
            self.assertIsNotNone(payload)
            with self.assertRaisesRegex(UnsafeScoringCalibrationError, "calibration dataset"):
                scoring_for_dataset("train_a", calibration)
            with self.assertRaisesRegex(UnsafeScoringCalibrationError, "not in the scoring validation"):
                scoring_for_dataset("other", calibration)

    def test_duplicate_source_dataset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            split_path, calibration_path = self.write_split_and_calibration(Path(raw), source="dup_public")
            with self.assertRaisesRegex(UnsafeScoringCalibrationError, "source_datasets must be a subset"):
                load_scoring_calibration(calibration_path, split_manifest_path=split_path)

    def test_proposal_policy_accepts_bounded_template_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            split_path, calibration_path = self.write_split_and_calibration(Path(raw))
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            payload["proposal_policy"] = {
                "by_route_prefix": {
                    "template_point_compact:val": {
                        "templates": [
                            {
                                "name": "tiny",
                                "source": "template_score:tiny",
                                "kernel": [
                                    [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                                    [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                                    [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                                ],
                                "score_quantiles": [0.99],
                                "max_points": 5,
                                "min_distance_um": 1.0,
                            }
                        ]
                    }
                }
            }
            calibration_path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = load_scoring_calibration(
                calibration_path,
                split_manifest_path=split_path,
                expected_fold_id="0",
            )
            resolved = scoring_for_dataset("val_a", calibration)
            self.assertIn("proposal_policy", resolved)

    def test_node_pruning_policy_is_deterministic_and_records_scores(self) -> None:
        nodes = [
            CandidateNode("sample", 1, 0, 0, 0, 0, 0.5, ("tile_raw@q0.75",), 1.0, intensity=20.0),
            CandidateNode("sample", 2, 0, 0, 1, 0, 0.5, ("tile_raw@q0.75",), 1.0, intensity=80.0),
            CandidateNode("sample", 3, 0, 0, 2, 0, 0.5, ("tile_raw@q0.75",), 1.0, intensity=40.0),
        ]
        payload = {
            "node_policy": {
                "by_route": {
                    "balanced_compact": {
                        "weights": {"intensity": 0.1},
                        "keep_top_k_per_t": 2,
                    }
                }
            }
        }
        pruned, stats = prune_nodes_by_t(
            [nodes],
            dataset="sample",
            route_label="balanced_compact",
            calibration_payload=payload,
        )
        self.assertEqual([[node.node_id for node in frame] for frame in pruned], [[2, 3]])
        self.assertIsNotNone(stats)
        self.assertEqual(stats.before_nodes, 3)
        self.assertEqual(stats.after_nodes, 2)
        self.assertIn("calibrated_node_score", nodes[0].features)


    def test_coordinate_policy_corrects_by_source_family_and_clamps(self) -> None:
        nodes = [
            CandidateNode("sample", 1, 0, 1, 2, 3, 0.5, ("tile_raw@q0.75",), 1.0),
            CandidateNode("sample", 2, 0, 0, 0, 0, 0.5, ("dog_dark@q0.99",), 1.0),
        ]
        payload = {
            "coordinate_policy": {
                "by_route_prefix": {
                    "balanced_compact:sample": {
                        "offset_zyx": [1, 1, 1],
                        "source_offsets_zyx": {"tile_raw": [2, -3, 4]},
                        "max_abs_offset_zyx": [2, 2, 2],
                    }
                }
            }
        }
        corrected, stats = correct_nodes_by_t(
            [nodes],
            dataset="sample_a",
            route_label="balanced_compact",
            calibration_payload=payload,
            spatial_shape_zyx=(5, 5, 5),
        )
        self.assertEqual([(node.z, node.y, node.x) for node in corrected[0]], [(3, 0, 5 - 1), (1, 1, 1)])
        self.assertIsNotNone(stats)
        self.assertEqual(stats.corrected_nodes, 2)
        self.assertIn("coordinate_correction_dz", corrected[0][0].features)

    def test_hybrid_source_families_are_exposed_as_node_and_edge_features(self) -> None:
        source = CandidateNode(
            "sample",
            1,
            0,
            0,
            0,
            0,
            0.5,
            ("cover_contrast@q0.70", "cover_dog_bright@q0.70"),
            1.0,
        )
        target = CandidateNode(
            "sample",
            2,
            1,
            0,
            1,
            0,
            0.5,
            ("tile_raw@q0.75", "dog_dark@q0.99", "zshift_dog_dark_m5:dog_dark@q0.9985"),
            1.0,
        )
        node_features = node_feature_dict(source)
        self.assertEqual(node_features["has_cover"], 1.0)
        self.assertEqual(node_features["has_cover_contrast"], 1.0)
        self.assertEqual(node_features["has_cover_dog_bright"], 1.0)
        self.assertEqual(node_features["has_dog"], 1.0)
        self.assertEqual(node_features["cover_only"], 1.0)
        self.assertEqual(node_features["rescue_source_count"], 2.0)

        edge_features = edge_feature_dict(
            CandidateEdge(1, 2, 1, distance_um=3.0, cost=0.5, score=0.5),
            source,
            target,
            max_link_um=10.0,
        )
        self.assertEqual(edge_features["source_has_cover"], 1.0)
        self.assertEqual(edge_features["target_has_cover"], 0.0)
        self.assertEqual(edge_features["source_cover_only"], 1.0)
        self.assertEqual(edge_features["source_rescue_source_count"], 2.0)
        self.assertEqual(edge_features["target_has_tile"], 1.0)
        self.assertEqual(edge_features["target_has_zshift"], 1.0)
        target_features = node_feature_dict(target)
        self.assertEqual(target_features["has_zshift"], 1.0)
        self.assertEqual(target_features["has_zshift_dog_dark_m5"], 1.0)
        self.assertEqual(target_features["has_dog"], 1.0)


    def test_edge_scoring_policy_adjusts_costs_and_can_drop_edges(self) -> None:
        nodes_by_t = [
            [CandidateNode("sample", 1, 0, 0, 0, 0, 0.5, ("tile_raw@q0.75",), 1.0, intensity=20.0)],
            [
                CandidateNode("sample", 2, 1, 0, 1, 0, 0.5, ("tile_raw@q0.75",), 1.0, intensity=22.0),
                CandidateNode("sample", 3, 1, 0, 20, 0, 0.5, ("tile_raw@q0.75",), 1.0, intensity=80.0),
            ],
        ]
        edges = [
            CandidateEdge(1, 2, 1, distance_um=1.0, cost=0.5, score=0.5),
            CandidateEdge(1, 3, 1, distance_um=20.0, cost=0.5, score=0.5),
        ]
        route_config = RouteConfig(
            label="balanced_compact",
            detector_sources=("tile_raw",),
            threshold_quantiles=(0.75,),
            min_distance_um=2.0,
            fusion_radius_um=1.8,
            max_points_per_t=10,
            max_link_um=10.0,
        )
        payload = {
            "edge_policy": {
                "by_route": {
                    "balanced_compact": {
                        "weights": {"distance_norm": -4.0},
                        "bias": 1.0,
                        "min_score": -2.0,
                        "drop_below_min_score": True,
                        "base_cost_weight": 0.0,
                        "probability_cost_weight": 1.0,
                        "replace_edge_score": True,
                    }
                }
            }
        }
        adjusted, stats = score_edge_candidates(
            edges,
            nodes_by_t,
            dataset="sample",
            route_label="balanced_compact",
            route_config=route_config,
            calibration_payload=payload,
        )
        self.assertEqual([(edge.source_id, edge.target_id) for edge in adjusted], [(1, 2)])
        self.assertIsNotNone(stats)
        self.assertEqual(stats.before_edges, 2)
        self.assertEqual(stats.after_edges, 1)
        self.assertEqual(stats.dropped_edges, 1)
        self.assertLess(adjusted[0].cost, 0.5)
        self.assertIn("calibrated_edge_probability", adjusted[0].features)


if __name__ == "__main__":
    unittest.main()
