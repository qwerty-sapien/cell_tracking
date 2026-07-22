from __future__ import annotations

import unittest

import numpy as np

from scripts.falsify_e4_e5 import (
    MorphologyRecord,
    PATCH_SHAPE,
    balanced_accuracy,
    block_mean_3d,
    block_mean_lateral,
    extract_patch,
    balanced_group_folds,
    make_identity_folds,
    morphology_feature_vector,
    roc_auc,
    sphere_moments,
)


class TestFalsifyE4E5(unittest.TestCase):
    def test_boundary_patch_is_fixed_shape_and_finite(self) -> None:
        frame = np.arange(8 * 20 * 20, dtype=np.uint16).reshape(8, 20, 20)
        patch = extract_patch(frame, (0, 1, 19))
        self.assertEqual(patch.shape, PATCH_SHAPE)
        self.assertTrue(np.isfinite(patch).all())
        self.assertEqual(block_mean_3d(patch).shape, (256,))
        self.assertEqual(block_mean_lateral(np.max(patch, axis=0)).shape, (256,))

    def test_sphere_inertia_detects_anisotropy(self) -> None:
        shape = (25, 41, 41)
        zz, yy, xx = np.indices(shape)
        center = np.asarray((12.0, 20.0, 20.0))
        isotropic = 100.0 + 2_000.0 * np.exp(
            -((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) / 18.0
        )
        elongated = 100.0 + 2_000.0 * np.exp(
            -((zz - center[0]) ** 2 / 4.0 + (yy - center[1]) ** 2 / 30.0 + (xx - center[2]) ** 2 / 4.0)
        )
        iso = sphere_moments(isotropic, center, (1.0, 1.0, 1.0))
        long = sphere_moments(elongated, center, (1.0, 1.0, 1.0))
        self.assertGreater(iso["eig_min_over_max"], long["eig_min_over_max"])
        self.assertLess(iso["anisotropy"], long["anisotropy"])

    def test_morphology_vector_is_finite(self) -> None:
        moments = {
            "raw_sum": 100.0,
            "excess_sum": 40.0,
            "variance": 9.0,
            "relative_variance": 0.2,
            "eig_min_over_max": 0.4,
            "eig_mid_over_max": 0.7,
            "anisotropy": 0.3,
        }
        vector = morphology_feature_vector(moments, moments)
        self.assertEqual(len(vector), 14)
        self.assertTrue(np.isfinite(vector).all())

    def test_metrics_have_expected_values(self) -> None:
        y = np.asarray([0, 0, 1, 1])
        perfect = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc(y, perfect), 1.0)
        self.assertAlmostEqual(balanced_accuracy(y, perfect), 1.0)

    def test_group_folds_never_split_an_identity(self) -> None:
        identities = [f"train:video_{index}" for index in range(10)]
        assignment = make_identity_folds(identities + identities)
        self.assertEqual(set(assignment), set(identities))
        self.assertEqual(set(assignment.values()), set(range(5)))

        records = []
        for identity in identities[:5]:
            records.append(MorphologyRecord(identity, 1, 1, 0, (0.0,) * 14))
            records.append(MorphologyRecord(identity, 0, 2, 0, (0.0,) * 14))
        grouped = balanced_group_folds(records)
        self.assertEqual(set(grouped), set(identities[:5]))
        self.assertEqual(set(grouped.values()), set(range(5)))


if __name__ == "__main__":
    unittest.main()
