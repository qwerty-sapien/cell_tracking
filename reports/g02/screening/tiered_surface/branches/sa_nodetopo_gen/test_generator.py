#!/usr/bin/env python3
"""Focused synthetic tests for BLOCKPEAK-RESTORE-V1."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "build_authentic_universe", HERE / "build_authentic_universe.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from celltrack_alpha.action_compiler import compile_action, compile_batch  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
)


class GeneratorTest(unittest.TestCase):
    def test_block_peak_is_deterministic_and_tie_uses_c_order(self) -> None:
        volume = np.zeros((4, 16, 16), dtype=np.uint16)
        volume[0, 0, 0] = 9
        volume[3, 15, 15] = 9
        self.assertEqual(MODULE.block_peaks(volume, 9), [(9, 0, 0, 0)])
        self.assertEqual(MODULE.block_peaks(volume, 9), MODULE.block_peaks(volume, 9))

    def test_candidate_identity_changes_with_raw_intensity(self) -> None:
        first = MODULE.Candidate("44b6_x", 1, 0, 1, 2, 3, 10)
        second = MODULE.Candidate("44b6_x", 1, 0, 1, 2, 3, 11)
        self.assertNotEqual(first.identity_sha256, second.identity_sha256)

    def test_canonical_gzip_is_byte_identical(self) -> None:
        rows = [{"b": 2, "a": 1}, {"b": 4, "a": 3}]
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.gz"
            second = Path(tmp) / "b.gz"
            self.assertEqual(MODULE.gzip_rows(first, rows), MODULE.gzip_rows(second, rows))

    def test_positive_and_false_partition(self) -> None:
        rows = [
            MODULE.Candidate("44b6_x", 1, 1, 0, 0, 0, 10),
            MODULE.Candidate("44b6_x", 2, 1, 0, 100, 0, 10),
        ]
        positive, false, nearest = MODULE.positive_false_masks(
            rows, np.asarray([[0, 0, 0]], dtype=np.float64), (1.0, 1.0, 1.0)
        )
        self.assertEqual(positive.tolist(), [True, False])
        self.assertEqual(false.tolist(), [False, True])
        self.assertTrue(np.all(np.isfinite(nearest)))

    def test_restoration_compiles_single_and_batch(self) -> None:
        graph = SubmissionGraph(
            dataset="44b6_x",
            nodes={
                1: SubmissionNode(1, 0, 0, 0, 0),
                2: SubmissionNode(2, 1, 0, 0, 0),
                3: SubmissionNode(3, 2, 0, 0, 0),
                4: SubmissionNode(4, 0, 0, 30, 0),
                5: SubmissionNode(5, 1, 0, 30, 0),
                6: SubmissionNode(6, 2, 0, 30, 0),
            },
            edges=[
                SubmissionEdge(1, 2),
                SubmissionEdge(2, 3),
                SubmissionEdge(4, 5),
                SubmissionEdge(5, 6),
            ],
        )
        indices = MODULE.graph_indices(graph, (1.0, 1.0, 1.0))
        first = MODULE.build_action(
            "train:44b6_x",
            MODULE.Candidate("44b6_x", 10, 1, 0, 1, 0, 10),
            graph,
            (1.0, 1.0, 1.0),
            indices,
        )
        second = MODULE.build_action(
            "train:44b6_x",
            MODULE.Candidate("44b6_x", 11, 1, 0, 29, 0, 10),
            graph,
            (1.0, 1.0, 1.0),
            indices,
        )
        assert first and second
        self.assertEqual(first.action.kind.value, "restore_topology")
        self.assertNotEqual(
            compile_action(graph, first.action).before_graph_sha256,
            compile_action(graph, first.action).after_graph_sha256,
        )
        population, rejected = MODULE.compatible_population(
            sorted([first, second], key=lambda row: row.order_key), 2
        )
        self.assertEqual(len(population), 2)
        self.assertFalse(rejected)
        self.assertNotEqual(
            compile_batch(graph, [row.action for row in population]).before_graph_sha256,
            compile_batch(graph, [row.action for row in population]).after_graph_sha256,
        )

    def test_fixed_contract_constants(self) -> None:
        self.assertEqual(MODULE.BLOCK, (4, 16, 16))
        self.assertEqual(MODULE.MAX_PER_FRAME, 64)
        self.assertEqual(MODULE.BUDGETS, (0, 1, 2, 5, 10, 20, 50))
        self.assertEqual(MODULE.FAMILY, "44b6")


if __name__ == "__main__":
    unittest.main()
