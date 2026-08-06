#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "reverse_cycle", HERE / "materialize_reverse_cycle.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReverseCycleTests(unittest.TestCase):
    def test_return_uses_reverse_target_source_entry(self) -> None:
        association = np.asarray([[0.1, 0.2], [0.7, 0.4]])
        self.assertEqual(
            MODULE.returned_score(association, {10: 0, 20: 1}, (10, 20)),
            0.7,
        )
        self.assertNotEqual(
            MODULE.returned_score(association, {10: 0, 20: 1}, (10, 20)),
            -0.2,
        )

    def test_null_uses_all_original_target_frame_nodes(self) -> None:
        association = np.asarray(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
        )
        observed = MODULE.returned_null(
            association,
            {1: 0, 2: 1, 3: 2},
            np.asarray([4, 5, 5]),
            source=1,
            target_t=5,
        )
        self.assertAlmostEqual(observed, 0.5)

    def test_null_fails_outside_numeric_guard(self) -> None:
        association = np.asarray([[0.0, 0.0], [1.1, 0.0]])
        with self.assertRaises(MODULE.ReverseSupplierError):
            MODULE.returned_null(
                association,
                {1: 0, 2: 1},
                np.asarray([4, 5]),
                source=1,
                target_t=5,
            )

    def test_packing_is_deterministic_and_preserves_required(self) -> None:
        by_node = {
            node: (node // 10, np.asarray([node, 0, 0], dtype=np.float32))
            for node in (10, 11, 20, 21)
        }
        frames = {1: (11, 10), 2: (21, 20)}
        first, cropped = MODULE.select_window_nodes(
            by_node=by_node,
            frame_nodes=frames,
            source_frame=1,
            required={11, 20},
        )
        second, _ = MODULE.select_window_nodes(
            by_node=by_node,
            frame_nodes=frames,
            source_frame=1,
            required={11, 20},
        )
        self.assertFalse(cropped)
        self.assertEqual(first, (10, 11, 20, 21))
        self.assertEqual(first, second)

    def test_deterministic_npz_is_byte_identical(self) -> None:
        arrays = {
            "b": np.asarray([1.5], dtype=np.float32),
            "a": np.asarray([[1, 2]], dtype=np.int64),
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            MODULE.write_deterministic_npz(first, arrays)
            MODULE.write_deterministic_npz(second, arrays)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with np.load(first, allow_pickle=False) as loaded:
                self.assertTrue(np.array_equal(loaded["a"], arrays["a"]))


if __name__ == "__main__":
    unittest.main()
