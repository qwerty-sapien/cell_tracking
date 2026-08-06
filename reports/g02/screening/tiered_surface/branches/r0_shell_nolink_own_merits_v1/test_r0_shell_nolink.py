#!/usr/bin/env python3
"""Unit tests for the frozen R0 shell no-link early falsifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "evaluate_r0_shell_nolink", HERE / "evaluate_r0_shell_nolink.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class R0ShellNoLinkTests(unittest.TestCase):
    def test_outer_fold_is_stable_and_bounded(self) -> None:
        identities = ["train:44b6_a", "train:44b6_b", "train:44b6_c"]
        observed = [MODULE.outer_fold(value) for value in identities]
        self.assertEqual(observed, [MODULE.outer_fold(value) for value in identities])
        self.assertTrue(all(0 <= value < 5 for value in observed))

    def test_balanced_ridge_separates_toy_classes(self) -> None:
        features = np.asarray([[-3.0], [-2.0], [2.0], [3.0]])
        positive = np.asarray([0, 0, 1, 1])
        model = MODULE.fit_balanced_ridge(features, positive)
        scores = model.score(features)
        self.assertGreater(float(np.min(scores[positive == 1])), float(np.max(scores[positive == 0])))
        self.assertAlmostEqual(float(np.mean(model.mean)), 0.0, places=12)

    def test_balanced_ridge_rejects_one_class(self) -> None:
        with self.assertRaises(MODULE.R0Error):
            MODULE.fit_balanced_ridge(np.ones((2, 1)), np.ones(2))

    def test_standard_deviation_floor_is_literal(self) -> None:
        model = MODULE.fit_balanced_ridge(
            np.asarray([[7.0], [7.0], [7.0], [7.0]]),
            np.asarray([0, 0, 1, 1]),
        )
        self.assertEqual(model.scale, (MODULE.STD_FLOOR,))

    def test_empirical_upper_tail_orientation_and_ties(self) -> None:
        observed = MODULE.empirical_upper_tail(
            np.asarray([1.0, 2.0, 2.0, 4.0]),
            np.asarray([0.0, 2.0, 3.0, 5.0]),
        )
        np.testing.assert_allclose(observed, [1.0, 0.8, 0.4, 0.2])

    def test_greedy_selection_is_order_invariant_and_node_disjoint(self) -> None:
        rows = [
            {"tail_probability": 0.1, "raw_score": 2.0, "identity": "x", "source_id": 1, "target_id": 2},
            {"tail_probability": 0.2, "raw_score": 9.0, "identity": "x", "source_id": 1, "target_id": 3},
            {"tail_probability": 0.3, "raw_score": 1.0, "identity": "x", "source_id": 4, "target_id": 5},
        ]
        forward = MODULE.greedy_node_disjoint(rows, budget=2)
        reverse = MODULE.greedy_node_disjoint(list(reversed(rows)), budget=2)
        self.assertEqual(forward, reverse)
        self.assertEqual([(row["source_id"], row["target_id"]) for row in forward], [(1, 2), (4, 5)])

    def test_node_ids_are_scoped_by_video(self) -> None:
        rows = [
            {"tail_probability": 0.1, "raw_score": 2.0, "identity": "x", "source_id": 1, "target_id": 2},
            {"tail_probability": 0.2, "raw_score": 1.0, "identity": "y", "source_id": 1, "target_id": 2},
        ]
        selected = MODULE.greedy_node_disjoint(rows, budget=2)
        self.assertEqual(len(selected), 2)

    def test_marginal_value_gate_closes_graph(self) -> None:
        bound = MODULE.marginal_value_bound()
        self.assertEqual(bound["edge_denominator"], 11972)
        self.assertAlmostEqual(bound["maximum_raw_edge_jaccard_delta"], 4 / 11972)
        self.assertAlmostEqual(bound["maximum_adjusted_edge_jaccard_delta"], 4.4 / 11972)
        self.assertFalse(bound["graph_physically_capable_of_floor"])
        self.assertEqual(bound["maximum_node_recall_delta"], 0.0)

    def test_truth_blind_shard_round_trip_and_truth_rejection(self) -> None:
        identity = "train:44b6_a"
        fingerprint = "f" * 64
        candidate = {
            "identity": identity,
            "held_fold": MODULE.outer_fold(identity),
            "source_id": 1,
            "target_id": 2,
            "source_frame": 3,
            "features": {
                "pair_ncc": 0.7,
                "source_no_link_probability": 0.8,
                "distance_um": 10.5,
            },
            "raw_score": 0.9,
            "tail_probability": 0.1,
            "topology": 0,
            "action": "add_edge_remove_none",
        }
        shard = {
            "schema": "g02.r0-shell-freefree.truth-blind-video-shard.v1",
            "mechanism": MODULE.MECHANISM,
            "scientific_contract_sha256": MODULE.SCIENTIFIC_CONTRACT_SHA256,
            "identity": identity,
            "model_fingerprint": fingerprint,
            "candidates": [candidate],
            "video_report": {
                "identity": identity,
                "held_fold": MODULE.outer_fold(identity),
                "eligible_topology_0_rows": 1,
                "full_shell_rows": 10,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.json"
            MODULE._atomic_create_json(path, shard)
            candidates, report = MODULE._load_truth_blind_shard(
                path,
                identity=identity,
                expected_rows=1,
                expected_full_shell_rows=10,
                expected_model_fingerprint=fingerprint,
            )
            self.assertEqual(candidates, [candidate])
            self.assertEqual(report["identity"], identity)
            leaked = dict(shard)
            leaked["precision"] = 1.0
            leaked_path = Path(directory) / "leaked.json"
            MODULE._atomic_create_json(leaked_path, leaked)
            with self.assertRaises(MODULE.R0Error):
                MODULE._load_truth_blind_shard(
                    leaked_path,
                    identity=identity,
                    expected_rows=1,
                    expected_full_shell_rows=10,
                    expected_model_fingerprint=fingerprint,
                )

    def test_model_fingerprint_authenticates_calibration(self) -> None:
        model = MODULE.RidgeModel((0.0,), (1.0,), (2.0,), 0.5)
        first = MODULE.model_fingerprint(model, np.asarray([0.1, 0.2]))
        second = MODULE.model_fingerprint(model, np.asarray([0.1, 0.3]))
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
