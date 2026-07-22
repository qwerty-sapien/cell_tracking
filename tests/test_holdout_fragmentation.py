from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_holdout_fragmentation as audit  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
)


SCALE = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


def node(node_id: int, t: int, x: int) -> SubmissionNode:
    return SubmissionNode(node_id=node_id, t=t, z=0, y=0, x=x)


class BreakClassificationTests(unittest.TestCase):
    def classify(
        self,
        graph: SubmissionGraph,
        *,
        pred_edges: set[tuple[int, int]] | None = None,
        gt_out: dict[int, list[int]] | None = None,
        gt_to_pred: dict[int, int] | None = None,
    ) -> str:
        edges = pred_edges or set()
        pred_out: dict[int, list[int]] = {}
        pred_in: dict[int, list[int]] = {}
        for source, target in edges:
            pred_out.setdefault(source, []).append(target)
            pred_in.setdefault(target, []).append(source)
        by_frame: dict[int, list[int]] = {}
        for node_id, value in graph.nodes.items():
            by_frame.setdefault(value.t, []).append(node_id)
        cause, _ = audit.classify_break(
            gt_source=100,
            gt_target=101,
            pred_source=0,
            pred_target=1,
            graph=graph,
            pred_edges=edges,
            pred_out=pred_out,
            pred_in=pred_in,
            gt_out=gt_out or {100: [101]},
            gt_to_pred=gt_to_pred or {100: 0, 101: 1},
            by_frame=by_frame,
            scale=SCALE,
            input_node_count=len(graph.nodes),
        )
        return cause

    def test_final_pair_above_radius_is_observed_before_conflict(self) -> None:
        graph = SubmissionGraph("x", {0: node(0, 0, 0), 1: node(1, 1, 40)}, [])
        self.assertEqual(
            self.classify(graph), "final_pair_geometry_above_gate_radius"
        )

    def test_final_rank_above_cap_is_separate_from_radius(self) -> None:
        nodes = {0: node(0, 0, 0)}
        for index in range(1, 11):
            nodes[index] = node(index, 1, index)
        graph = SubmissionGraph("x", nodes, [])
        cause, observation = audit.classify_break(
            gt_source=100,
            gt_target=101,
            pred_source=0,
            pred_target=9,
            graph=graph,
            pred_edges=set(),
            pred_out={},
            pred_in={},
            gt_out={100: [101]},
            gt_to_pred={100: 0, 101: 9},
            by_frame={0: [0], 1: list(range(1, 11))},
            scale=SCALE,
            input_node_count=len(nodes),
        )
        self.assertEqual(cause, "final_pair_rank_above_cap8")
        self.assertEqual(observation["final_node_distance_rank"], 9)

    def test_occupied_endpoint_is_an_exact_conflict_observation(self) -> None:
        graph = SubmissionGraph(
            "x",
            {0: node(0, 0, 0), 1: node(1, 1, 4), 2: node(2, 1, 8)},
            [SubmissionEdge(0, 2)],
        )
        self.assertEqual(
            self.classify(graph, pred_edges={(0, 2)}),
            "occupied_endpoint_conflict",
        )

    def test_selected_sister_exposes_division_degree_constraint(self) -> None:
        graph = SubmissionGraph(
            "x",
            {0: node(0, 0, 0), 1: node(1, 1, 4), 2: node(2, 1, 8)},
            [SubmissionEdge(0, 2)],
        )
        self.assertEqual(
            self.classify(
                graph,
                pred_edges={(0, 2)},
                gt_out={100: [101, 102]},
                gt_to_pred={100: 0, 101: 1, 102: 2},
            ),
            "solver_outdegree_division",
        )


class FrozenArtifactTests(unittest.TestCase):
    ARTIFACT = (
        ROOT
        / "reports/g02/source_softmax_edge_f0_fit117_holdout_complete57_v2"
        / "fragmentation.json"
    )

    @unittest.skipUnless(ARTIFACT.is_file(), "requires the completed local 57-video audit")
    def test_prefix_edge_budget_and_scope_are_reproduced(self) -> None:
        report = json.loads(self.ARTIFACT.read_text(encoding="utf-8"))
        self.assertEqual(report["scope"]["video_count"], 57)
        self.assertEqual(report["scope"]["excluded_identity"], audit.EXCLUDED_IDENTITY)
        self.assertEqual(
            report["prefix_52"]["edge_totals"],
            {"edge_tp": 29676, "edge_fp": 2660, "edge_fn": 1810},
        )
        self.assertEqual(sum(report["prefix_52"]["fn_causes"].values()), 1810)
        self.assertEqual(sum(report["prefix_52"]["fp_direct_classes"].values()), 2660)


if __name__ == "__main__":
    unittest.main()
