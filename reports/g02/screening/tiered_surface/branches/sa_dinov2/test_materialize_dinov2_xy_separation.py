from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("materialize_dinov2_xy_separation.py")
SPEC = importlib.util.spec_from_file_location("materialize_dinov2_xy_separation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MaterializerTests(unittest.TestCase):
    def test_exact_histogram_quantile_matches_numpy_linear(self):
        values = np.asarray([0, 0, 1, 4, 4, 9, 12, 65535], dtype=np.uint16)
        histogram = np.bincount(values, minlength=65536)
        for quantile in (0.0, 0.01, 0.5, 0.99, 1.0):
            self.assertEqual(
                MODULE.exact_uint16_quantile(histogram, quantile),
                float(np.quantile(values, quantile, method="linear")),
            )

    def test_pseudo_rgb_shape_range_and_border_replication(self):
        z = np.arange(64, dtype=np.uint16)[:, None, None]
        y = np.arange(256, dtype=np.uint16)[None, :, None]
        x = np.arange(256, dtype=np.uint16)[None, None, :]
        frame = (z * 100 + y + x).astype(np.uint16)
        coordinates = np.asarray([[0, 0, 0], [32, 128, 128]], dtype=np.int64)
        batches = list(
            MODULE.pseudo_rgb_batches(
                frame, coordinates, low=0.0, high=7000.0, batch_size=2
            )
        )
        self.assertEqual(len(batches), 1)
        _, pseudo = batches[0]
        self.assertEqual(pseudo.shape, (2, 3, 33, 33))
        self.assertTrue(np.isfinite(pseudo).all())
        self.assertGreaterEqual(float(pseudo.min()), 0.0)
        self.assertLessEqual(float(pseudo.max()), 1.0)
        self.assertEqual(float(pseudo[0, 2, 0, 0]), 0.0)

    def test_row_mapping_is_exact_and_deterministic(self):
        rows = np.asarray(
            [[1, 2, 3, 4], [0, 9, 8, 7], [1, 2, 3, 4], [2, 0, 0, 0]],
            dtype=np.int64,
        )
        unique = np.unique(rows, axis=0)
        indices = MODULE.map_rows(unique, rows)
        np.testing.assert_array_equal(unique[indices], rows)


if __name__ == "__main__":
    unittest.main()
