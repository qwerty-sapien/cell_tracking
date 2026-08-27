from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("dinov2_xy_action_featurizer.py")
SPEC = importlib.util.spec_from_file_location("dinov2_xy_action_featurizer", MODULE_PATH)
assert SPEC and SPEC.loader
features = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(features)


class ActionFeaturizerTests(unittest.TestCase):
    def test_identity_vectors_have_expected_scalars(self):
        source = np.ones((2, 768), dtype=np.float32)
        result = features.action_features(source, source.copy())
        self.assertEqual(result.shape, (2, 8))
        self.assertTrue(np.allclose(result[:, 0], 1.0))
        self.assertTrue(np.allclose(result[:, 1:3], 0.0))
        self.assertTrue(np.allclose(result[:, 3:6], 1.0))
        self.assertTrue(np.allclose(result[:, 6:], 0.0))

    def test_bad_shape_fails_closed(self):
        with self.assertRaises(features.FeatureContractError):
            features.action_features(
                np.zeros((1, 384), dtype=np.float32),
                np.zeros((1, 384), dtype=np.float32),
            )

    def test_nonfinite_fails_closed(self):
        source = np.zeros((1, 768), dtype=np.float32)
        source[0, 0] = np.nan
        with self.assertRaises(features.FeatureContractError):
            features.action_features(source, np.zeros_like(source))

    def test_permutation_is_deterministic_and_preserves_multiset(self):
        values = np.arange(4 * 768, dtype=np.float32).reshape(4, 768)
        node_ids = np.array([10, 11, 12, 13], dtype=np.int64)
        kwargs = {
            "identity": "train:44b6_example",
            "frame": 7,
            "supplier_contract_sha256": "a" * 64,
        }
        first = features.within_frame_permutation(values, node_ids, **kwargs)
        second = features.within_frame_permutation(values, node_ids, **kwargs)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(
            sorted(map(tuple, first.tolist())),
            sorted(map(tuple, values.tolist())),
        )

    def test_duplicate_node_ids_share_one_permuted_vector(self):
        values = np.vstack(
            (
                np.zeros((1, 768), dtype=np.float32),
                np.zeros((1, 768), dtype=np.float32),
                np.ones((1, 768), dtype=np.float32),
            )
        )
        result = features.within_frame_permutation(
            values,
            np.array([10, 10, 11], dtype=np.int64),
            identity="train:44b6_example",
            frame=7,
            supplier_contract_sha256="b" * 64,
        )
        self.assertTrue(np.array_equal(result[0], result[1]))

    def test_invalid_contract_hash_fails_closed(self):
        with self.assertRaises(features.FeatureContractError):
            features.within_frame_permutation(
                np.zeros((1, 768), dtype=np.float32),
                np.array([1], dtype=np.int64),
                identity="train:44b6_example",
                frame=0,
                supplier_contract_sha256="not-a-sha",
            )


if __name__ == "__main__":
    unittest.main()
