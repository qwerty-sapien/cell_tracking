#!/usr/bin/env python3
"""Focused fixture tests for the frozen SA-DELETE implementation."""

from __future__ import annotations

import unittest

import numpy as np

import sa_delete as target
from celltrack_alpha.action_compiler import ActionKind, GraphAction, compile_batch
from celltrack_alpha.kaggle_calibration import (
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
)


class FakeFrame:
    def __init__(self, node_ids: list[int], scores: list[float]) -> None:
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.scores = np.asarray(scores, dtype=np.float32)


class FakeCache:
    def __init__(self) -> None:
        self.detections = (
            FakeFrame([1, 2], [0.99, 0.98]),
            FakeFrame([3], [0.97]),
        )


def graph() -> SubmissionGraph:
    return SubmissionGraph(
        dataset="44b6_fixture",
        nodes={
            1: SubmissionNode(1, 0, 0, 0, 0),
            2: SubmissionNode(2, 0, 0, 4, 0),
            3: SubmissionNode(3, 1, 0, 1, 0),
            4: SubmissionNode(4, 1, 0, 8, 0),
        },
        edges=[SubmissionEdge(1, 3), SubmissionEdge(2, 4)],
    )


class DeleteTests(unittest.TestCase):
    def test_mandated_audit_named_fold_manifest_is_the_only_exemption(self) -> None:
        target.safe_runtime_path(target.INPUT_PINS["folds"][0])
        with self.assertRaises(target.DeleteGateError):
            target.safe_runtime_path(target.ROOT / "reports/g02/audit/forbidden.json")

    def test_nested_partitions_are_complete_and_disjoint(self) -> None:
        identities = tuple(f"train:44b6_{index:08x}" for index in range(100))
        partitions = target.nested_partitions(identities)
        self.assertEqual(len(partitions), 5)
        for fit, calibration, held in partitions:
            self.assertEqual(set((*fit, *calibration, *held)), set(identities))
            self.assertFalse(set(fit) & set(calibration))
            self.assertFalse(set(fit) & set(held))
            self.assertFalse(set(calibration) & set(held))

    def test_feature_extraction_is_finite_and_canonical(self) -> None:
        edges, features = target.edge_features(graph(), FakeCache())
        np.testing.assert_array_equal(edges, np.asarray([[1, 3], [2, 4]]))
        self.assertEqual(features.shape, (2, len(target.FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(float(features[0, 5]), 1.0)
        self.assertEqual(float(features[1, 6]), 0.5)

    def test_linear_fit_separates_fixture(self) -> None:
        x = np.asarray(
            [
                [0.0] * len(target.FEATURE_NAMES),
                [0.1] * len(target.FEATURE_NAMES),
                [1.0] * len(target.FEATURE_NAMES),
                [1.1] * len(target.FEATURE_NAMES),
            ],
            dtype=np.float32,
        )
        y = np.asarray([0, 0, 1, 1], dtype=np.int8)
        fitted = target.fit_linear_retention([x], [y])
        scores = target.retention_scores(fitted, x)
        self.assertGreater(float(scores[2:].min()), float(scores[:2].max()))
        self.assertEqual(fitted.parameter_count, len(target.FEATURE_NAMES) + 1)

    def test_conservative_threshold_retains_lower_quantile(self) -> None:
        scores = np.linspace(0.0, 1.0, 1001)
        labels = np.ones(1001, dtype=np.int8)
        threshold, diagnostic = target.conservative_threshold([scores], [labels])
        self.assertEqual(threshold, float(scores[1]))
        self.assertEqual(diagnostic["deleted_true_edges"], 1)

    def test_n1_delete_bundle_compiles_as_final_batch(self) -> None:
        baseline = graph()
        action = GraphAction(
            action_id="delete-o0:fixture",
            kind=ActionKind.DELETE_EDGES,
            delete_edges=((2, 4),),
        )
        compiled = compile_batch(baseline, [action])
        self.assertEqual(compiled.removed_edges, ((2, 4),))
        self.assertEqual(
            {(edge.source_id, edge.target_id) for edge in compiled.graph.edges},
            {(1, 3)},
        )


if __name__ == "__main__":
    unittest.main()
