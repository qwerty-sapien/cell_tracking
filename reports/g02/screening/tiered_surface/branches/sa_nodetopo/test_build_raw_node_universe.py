#!/usr/bin/env python3
"""Synthetic-only focused tests for the raw node-universe runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "build_raw_node_universe", HERE / "build_raw_node_universe.py"
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


class RawNodeUniverseTest(unittest.TestCase):
    def test_canonical_gzip_is_byte_deterministic(self) -> None:
        rows = [{"z": 2, "a": 1}, {"z": 4, "a": 3}]
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.gz"
            second = Path(tmp) / "second.gz"
            sha1, count1 = MODULE.gzip_jsonl(first, rows)
            sha2, count2 = MODULE.gzip_jsonl(second, rows)
        self.assertEqual((sha1, count1), (sha2, count2))
        self.assertEqual(count1, 2)

    def test_positive_assignment_and_hard_false_are_disjoint(self) -> None:
        candidates = [
            MODULE.CandidateRow("44b6_x", 1, 1, 0, 0, 0, "0x0p+0", ("raw",)),
            MODULE.CandidateRow("44b6_x", 2, 1, 0, 4, 0, "0x0p+0", ("raw",)),
            MODULE.CandidateRow("44b6_x", 3, 1, 0, 100, 0, "0x0p+0", ("raw",)),
        ]
        positive, hard_false, nearest = MODULE.positive_and_hard_false_masks(
            candidates,
            np.asarray([[0, 0, 0]], dtype=np.float64),
            (1.0, 1.0, 1.0),
        )
        self.assertEqual(positive.tolist(), [True, False, False])
        self.assertEqual(hard_false.tolist(), [False, False, True])
        self.assertTrue(np.all(np.isfinite(nearest)))

    def test_topology_action_compiles_singly_and_in_disjoint_batch(self) -> None:
        graph = SubmissionGraph(
            dataset="44b6_synthetic",
            nodes={
                1: SubmissionNode(1, 0, 0, 0, 0),
                2: SubmissionNode(2, 1, 0, 0, 0),
                3: SubmissionNode(3, 2, 0, 0, 0),
                4: SubmissionNode(4, 0, 0, 20, 0),
                5: SubmissionNode(5, 1, 0, 20, 0),
                6: SubmissionNode(6, 2, 0, 20, 0),
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
            "train:44b6_synthetic",
            MODULE.CandidateRow(
                "44b6_synthetic", 10, 1, 0, 1, 0, "0x1p+0", ("raw",)
            ),
            graph,
            (1.0, 1.0, 1.0),
            indices,
        )
        second = MODULE.build_action(
            "train:44b6_synthetic",
            MODULE.CandidateRow(
                "44b6_synthetic", 11, 1, 0, 19, 0, "0x1p+0", ("raw",)
            ),
            graph,
            (1.0, 1.0, 1.0),
            indices,
        )
        assert first and second
        self.assertNotEqual(
            compile_action(graph, first.action).before_graph_sha256,
            compile_action(graph, first.action).after_graph_sha256,
        )
        accepted, rejected = MODULE.mutually_compatible(
            sorted([first, second], key=lambda value: value.order_key), 2
        )
        self.assertEqual(len(accepted), 2)
        self.assertFalse(rejected)
        compiled = compile_batch(graph, [value.action for value in accepted])
        self.assertNotEqual(compiled.before_graph_sha256, compiled.after_graph_sha256)

    def test_conflicting_actions_are_rejected_before_batch(self) -> None:
        graph = SubmissionGraph(
            dataset="44b6_synthetic",
            nodes={
                1: SubmissionNode(1, 0, 0, 0, 0),
                2: SubmissionNode(2, 1, 0, 0, 0),
                3: SubmissionNode(3, 2, 0, 0, 0),
            },
            edges=[SubmissionEdge(1, 2), SubmissionEdge(2, 3)],
        )
        indices = MODULE.graph_indices(graph, (1.0, 1.0, 1.0))
        rows = [
            MODULE.CandidateRow(
                "44b6_synthetic", node_id, 1, 0, y, 0, "0x1p+0", ("raw",)
            )
            for node_id, y in ((10, 1), (11, 2))
        ]
        records = [
            MODULE.build_action(
                "train:44b6_synthetic", row, graph, (1.0, 1.0, 1.0), indices
            )
            for row in rows
        ]
        valid = [value for value in records if value is not None]
        accepted, rejected = MODULE.mutually_compatible(
            sorted(valid, key=lambda value: value.order_key), 2
        )
        self.assertEqual(len(accepted), 1)
        self.assertGreater(sum(rejected.values()), 0)

    def test_constants_match_preregistered_budgets(self) -> None:
        self.assertEqual(MODULE.BUDGETS, (0, 1, 2, 5, 10, 20, 50))
        self.assertEqual(MODULE.ROUTE, "balanced_compact")
        self.assertEqual(MODULE.FAMILY, "44b6")


if __name__ == "__main__":
    unittest.main()
