"""Focused invariants for the frozen-alpha learned-motion reranker."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external/kaggle-cell-tracking-competition/src"))

from celltrack_alpha.bh_graph import GraphNode  # noqa: E402
from celltrack_alpha.official_eval import official_metric_row  # noqa: E402
from celltrack_alpha.ta_hybrid import (  # noqa: E402
    ChildMotion,
    MOTION_DIRECTION,
    MOTION_OWNER,
    rerank_alpha_edges,
    sample_child_motion,
)
from celltrack_alpha.ta_inference import FrameOutput  # noqa: E402


def _node(node_id: int, t: int, x: float) -> GraphNode:
    return GraphNode(node_id, t, 0.0, 0.0, x)


class AlphaMotionHybridTest(unittest.TestCase):
    def test_frozen_nodes_and_edge_cardinality_are_preserved_without_dense_nxn(self) -> None:
        nodes = (_node(1, 0, 0.0), _node(2, 0, 4.0), _node(3, 1, 1.0), _node(4, 1, 5.0))
        edges = ({"source_id": 1, "target_id": 3}, {"source_id": 2, "target_id": 4})
        motion = {
            3: ChildMotion(3, (0.0, 0.0, 3.0), 0.0),
            4: ChildMotion(4, (0.0, 0.0, -5.0), 0.0),
        }
        result = rerank_alpha_edges(nodes, edges, motion, scale_zyx=(1.0, 1.0, 1.0))
        self.assertEqual(result.nodes, nodes)
        self.assertEqual(len(result.edges), len(edges))
        self.assertTrue(result.stats["nodes_preserved"])
        self.assertEqual(result.stats["node_fingerprint_before"], result.stats["node_fingerprint_after"])
        self.assertFalse(result.stats["full_frame_dense_matrix"])
        self.assertGreater(result.stats["candidate_pair_count"], 0)

    def test_child_owned_parent_minus_child_direction_selects_expected_parents(self) -> None:
        nodes = (_node(1, 0, 0.0), _node(2, 0, 4.0), _node(3, 1, 1.0), _node(4, 1, 5.0))
        edges = ({"source_id": 1, "target_id": 3}, {"source_id": 2, "target_id": 4})
        result = rerank_alpha_edges(
            nodes,
            edges,
            {
                # child(1) + backward(+3) = parent(4)
                3: ChildMotion(3, (0.0, 0.0, 3.0), 0.0),
                # child(5) + backward(-5) = parent(0)
                4: ChildMotion(4, (0.0, 0.0, -5.0), 0.0),
            },
            scale_zyx=(1.0, 1.0, 1.0),
        )
        self.assertEqual(
            {(edge["source_id"], edge["target_id"]) for edge in result.edges},
            {(2, 3), (1, 4)},
        )
        self.assertEqual(result.provenance["motion_owner"], MOTION_OWNER)
        self.assertEqual(result.provenance["motion_direction"], MOTION_DIRECTION)
        self.assertEqual(result.stats["changed_parent_count"], 2)

    def test_divisions_and_missing_or_outside_gate_fallbacks_are_verbatim(self) -> None:
        nodes = (
            _node(1, 0, 0.0),
            _node(2, 0, 30.0),
            _node(3, 1, 1.0),
            _node(4, 1, 2.0),
            _node(5, 1, 50.0),
            _node(6, 0, 70.0),
            _node(7, 1, 71.0),
        )
        edges = (
            {"source_id": 1, "target_id": 3},
            {"source_id": 1, "target_id": 4},
            {"source_id": 2, "target_id": 5},
            {"source_id": 6, "target_id": 7},
        )
        result = rerank_alpha_edges(
            nodes,
            edges,
            {
                3: ChildMotion(3, (0.0, 0.0, 29.0), 0.0),
                4: ChildMotion(4, (0.0, 0.0, 28.0), 0.0),
                5: ChildMotion(5, (0.0, 0.0, -20.0), 0.0),
                # node 7 deliberately lacks motion.
            },
            scale_zyx=(1.0, 1.0, 1.0),
        )
        self.assertEqual(result.edges, edges)
        self.assertEqual(result.stats["division_edges_preserved"], 2)
        self.assertEqual(result.stats["fallback_outside_gate_edges"], 1)
        self.assertEqual(result.stats["fallback_missing_motion_edges"], 1)

    def test_sampling_uses_only_valid_child_frame_association(self) -> None:
        nodes = (_node(1, 0, 0.0), _node(2, 1, 4.0))
        invalid = FrameOutput(
            0,
            np.full((1, 1, 2), 99.0, dtype=np.float32),
            np.full((4, 1, 1, 2), 99.0, dtype=np.float32),
            False,
        )
        field = np.zeros((4, 1, 1, 2), dtype=np.float32)
        field[:, 0, 0, 1] = np.asarray([1.0, 2.0, -4.0, 0.5], dtype=np.float32)
        valid = FrameOutput(1, np.full((1, 1, 2), -99.0, dtype=np.float32), field, True)
        sampled = sample_child_motion(nodes, (invalid, valid), downsample_zyx=(1.0, 1.0, 4.0))
        self.assertNotIn(1, sampled)
        self.assertEqual(sampled[2].backward_um, (1.0, 2.0, -4.0))
        self.assertEqual(sampled[2].log_variance, 0.5)

    def test_hybrid_rows_are_accepted_by_unchanged_official_scorer(self) -> None:
        nodes = (_node(1, 0, 0.0), _node(2, 0, 4.0), _node(3, 1, 1.0), _node(4, 1, 5.0))
        result = rerank_alpha_edges(
            nodes,
            ({"source_id": 1, "target_id": 3}, {"source_id": 2, "target_id": 4}),
            {
                3: ChildMotion(3, (0.0, 0.0, 3.0), 0.0),
                4: ChildMotion(4, (0.0, 0.0, -5.0), 0.0),
            },
            scale_zyx=(1.0, 1.0, 1.0),
        )
        public_nodes = [
            {
                "dataset": "fixture",
                "node_id": node.node_id,
                "t": node.t,
                "z": node.z,
                "y": node.y,
                "x": node.x,
            }
            for node in result.nodes
        ]
        gt = {
            "nodes": {
                "node_id": np.asarray([1, 2, 3, 4], dtype=np.int64),
                "t": np.asarray([0, 0, 1, 1], dtype=np.int64),
                "z": np.asarray([0.0, 0.0, 0.0, 0.0]),
                "y": np.asarray([0.0, 0.0, 0.0, 0.0]),
                "x": np.asarray([0.0, 4.0, 1.0, 5.0]),
            },
            "edges": np.asarray([[2, 3], [1, 4]], dtype=np.int64),
            "estimated_number_of_nodes": 4.0,
        }
        score = official_metric_row(public_nodes, list(result.edges), gt, (1.0, 1.0, 1.0))
        self.assertEqual(score["edge_jaccard"], 1.0)
        self.assertEqual(score["node_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
