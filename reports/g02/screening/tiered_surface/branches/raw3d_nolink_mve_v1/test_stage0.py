#!/usr/bin/env python3
"""Synthetic, outcome-free tests for RAW3D-NOLINK-MVE Stage 0."""

from __future__ import annotations

from pathlib import Path
import time
import unittest

import numpy as np

import raw3d_features as features
import run_stage0


class Raw3DFeatureTests(unittest.TestCase):
    def test_workspace_root_resolution(self) -> None:
        self.assertEqual(run_stage0.ROOT, Path(__file__).resolve().parents[6])

    def test_endpoint_features_are_finite_at_boundaries(self) -> None:
        frame = np.arange(9 * 17 * 17, dtype=np.uint16).reshape(9, 17, 17)
        coordinates = np.asarray([[0, 0, 0], [4, 8, 8], [8, 16, 16]])
        observed = features.endpoint_features(
            frame, coordinates, time_fraction=0.25, batch_size=2
        )
        self.assertEqual(observed.shape, (3, features.ENDPOINT_DIMENSION))
        self.assertEqual(observed.dtype, np.float32)
        self.assertTrue(np.isfinite(observed).all())
        self.assertFalse(np.array_equal(observed[0], observed[1]))

    def test_bright_blob_changes_crude_3d_summary(self) -> None:
        frame = np.zeros((9, 17, 17), dtype=np.uint16)
        frame[3:6, 6:11, 6:11] = 1000
        observed = features.endpoint_features(
            frame, np.asarray([[4, 8, 8], [0, 0, 0]]), time_fraction=0.5
        )
        self.assertGreater(observed[0, 1], observed[1, 1])
        self.assertGreater(observed[0, 13], observed[1, 13])

    def test_action_map_matches_frozen_eight_summary_shape(self) -> None:
        generator = np.random.default_rng(7)
        source = generator.normal(size=(11, features.ENDPOINT_DIMENSION)).astype(np.float32)
        target = generator.normal(size=(11, features.ENDPOINT_DIMENSION)).astype(np.float32)
        observed = features.action_features(source, target)
        self.assertEqual(observed.shape, (11, 8))
        self.assertTrue(np.isfinite(observed).all())
        identical = features.action_features(source, source)
        np.testing.assert_allclose(identical[:, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(identical[:, 1:3], 0.0, atol=1e-6)

    def test_permutation_is_deterministic_and_preserves_node_vectors(self) -> None:
        values = np.arange(
            4 * features.ENDPOINT_DIMENSION, dtype=np.float32
        ).reshape(4, features.ENDPOINT_DIMENSION)
        node_ids = np.asarray([11, 12, 13, 11])
        kwargs = {
            "identity": "train:44b6_fixture",
            "frame": 3,
            "preregistration_sha256": "a" * 64,
        }
        first = features.within_frame_permutation(values, node_ids, **kwargs)
        second = features.within_frame_permutation(values, node_ids, **kwargs)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], first[3])
        original_unique = {tuple(row) for row in values[:3]}
        permuted_unique = {tuple(row) for row in first[:3]}
        self.assertEqual(original_unique, permuted_unique)


class AdvanceRuleTests(unittest.TestCase):
    @staticmethod
    def class_result(aucs: list[float]) -> dict:
        folds = []
        for held_fold, auc in enumerate(aucs):
            folds.append(
                {
                    "held_fold": held_fold,
                    "arms": {
                        "A1": {
                            "auc": auc,
                            "retention": 0.03,
                            "selected_beneficial": 5,
                            "selected_harmful": 2,
                        },
                        "A0": {"auc": 0.51, "retention": 0.0},
                    },
                }
            )
        return {"folds": folds}

    def test_both_mean_and_fold_floor_are_binding(self) -> None:
        passing = run_stage0.summarize_result(
            self.class_result([0.69] * 5),
            preregistration_sha256="b" * 64,
            step0={},
            feature_receipts=[],
            started=time.monotonic(),
        )
        self.assertTrue(passing["advance_rule"]["pass"])
        self.assertTrue(passing["downstream_authority"]["stage1_asset_provenance_open"])

        floor_failure = run_stage0.summarize_result(
            self.class_result([0.72, 0.72, 0.72, 0.72, 0.67]),
            preregistration_sha256="c" * 64,
            step0={},
            feature_receipts=[],
            started=time.monotonic(),
        )
        self.assertGreater(floor_failure["observed"]["A1_auc_mean"], 0.68)
        self.assertFalse(floor_failure["advance_rule"]["pass"])
        self.assertEqual(
            floor_failure["verdict"], "PARK_C_RAW3D_AND_CONSOLIDATE_CAMPAIGN"
        )
        self.assertFalse(
            floor_failure["downstream_authority"]["stage1_asset_provenance_open"]
        )

    def test_feature_module_has_no_outcome_interface(self) -> None:
        source = Path(features.__file__).read_text(encoding="utf-8").lower()
        for forbidden in ("oracle_label", "beneficial", "ground_truth", "graph_score"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

