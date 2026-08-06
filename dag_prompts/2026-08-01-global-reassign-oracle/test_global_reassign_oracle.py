#!/usr/bin/env python3
"""Outcome-free focused tests for the global reassignment harness."""

from __future__ import annotations

from contextlib import redirect_stderr
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("global_reassign_oracle", HERE / "global_reassign_oracle.py")
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)

from celltrack_alpha import bh_model
from celltrack_alpha.kaggle_calibration import SubmissionEdge, SubmissionGraph, SubmissionNode


def graph(edges: set[tuple[int, int]]) -> SubmissionGraph:
    nodes = {
        1: SubmissionNode(1, 0, 0, 0, 0),
        2: SubmissionNode(2, 0, 0, 10, 0),
        3: SubmissionNode(3, 1, 0, 0, 0),
        4: SubmissionNode(4, 1, 0, 10, 0),
        5: SubmissionNode(5, 2, 0, 0, 0),
        6: SubmissionNode(6, 2, 0, 10, 0),
    }
    return SubmissionGraph("synthetic", nodes, [SubmissionEdge(*edge) for edge in sorted(edges)])


def frame(frame_index: int, node_ids: list[int], coords: list[tuple[int, int, int]]) -> bh_model.FrameDet:
    count = len(node_ids)
    array = np.asarray(coords, dtype=np.int64)
    return bh_model.FrameDet(
        frame_index=frame_index,
        node_ids=np.asarray(node_ids, dtype=np.int64),
        coords_zyx=array,
        scores=np.ones(count, dtype=np.float32),
        grid_coords_zyx=array,
        grid_shape_zyx=(128, 128, 128),
    )


class ChainTests(unittest.TestCase):
    @staticmethod
    def metric_row(**updates: float) -> dict[str, float]:
        row = {
            "edge_tp": 10.0,
            "edge_fp": 1.0,
            "edge_fn": 2.0,
            "division_tp": 0.0,
            "division_fp": 0.0,
            "division_fn": 0.0,
            "num_pred_nodes": 12.0,
            "node_recall": 0.9,
            "total_node_ratio": 0.0,
            "edge_jaccard": 10.0 / 13.0,
            "adj_edge_jaccard": 10.0 / 13.0,
            "division_jaccard": float("nan"),
            "estimated_number_of_nodes": 12.0,
        }
        row.update(updates)
        return row

    def test_division_free_diagnostic_uses_strict_json_null(self) -> None:
        row = self.metric_row()
        encoded = oracle.serializable_metric_row(row)
        self.assertIsNone(encoded["division_jaccard"])
        self.assertEqual(encoded["division_tp"], 0.0)
        self.assertIn('"division_jaccard": null', json.dumps(encoded, allow_nan=False))
        self.assertTrue(math.isnan(row["division_jaccard"]))

    def test_opened_zip_store_closes_when_unused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.zip"
            with zipfile.ZipFile(path, mode="w"):
                pass
            with oracle.opened_zip_store(path):
                pass

    def test_fail_closed_output_contains_full_traceback(self) -> None:
        output = io.StringIO()
        with redirect_stderr(output):
            try:
                raise RuntimeError("outcome-free traceback sentinel")
            except RuntimeError as exc:
                self.assertEqual(oracle.fail_closed(exc), 2)
        captured = output.getvalue()
        self.assertIn("Traceback (most recent call last):", captured)
        self.assertIn("RuntimeError: outcome-free traceback sentinel", captured)
        self.assertIn("GLOBAL-REASSIGN-ORACLE failed closed", captured)

    def test_nonfinite_scientific_metric_fails_closed(self) -> None:
        with self.assertRaisesRegex(oracle.OracleError, "non-finite scientific quantity"):
            oracle.serializable_metric_row(self.metric_row(adj_edge_jaccard=float("nan")))

    def test_nonfinite_division_ratio_with_nonzero_counts_fails_closed(self) -> None:
        with self.assertRaisesRegex(oracle.OracleError, "non-finite scientific quantity"):
            oracle.serializable_metric_row(self.metric_row(division_fn=1.0))

    def test_null_diagnostic_preserves_frozen_summary_inputs(self) -> None:
        encoded = oracle.serializable_metric_row(self.metric_row())
        summary = oracle.official_summary(
            [encoded, oracle.serializable_metric_row(self.metric_row(division_fn=1.0, division_jaccard=0.0))]
        )
        self.assertEqual(summary["division_fn"], 1.0)
        self.assertEqual(summary["division_jaccard"], 0.0)

    def test_control_diagnostics_aggregate_per_video_count_schema(self) -> None:
        baseline_rows = [
            self.metric_row(division_fn=1.0, division_jaccard=0.0),
            self.metric_row(edge_tp=20.0, edge_fp=3.0, edge_fn=4.0),
        ]
        candidate_rows = [
            self.metric_row(
                edge_tp=11.0,
                edge_fp=0.0,
                division_fn=1.0,
                division_jaccard=0.0,
            ),
            self.metric_row(edge_tp=22.0, edge_fp=1.0, edge_fn=2.0),
        ]
        fragments = [
            {
                "baseline_metric": oracle.serializable_metric_row(baseline),
                "candidate_metric": oracle.serializable_metric_row(candidate),
            }
            for baseline, candidate in zip(
                baseline_rows, candidate_rows, strict=True
            )
        ]
        summary = oracle.official_summary(
            [dict(fragment["candidate_metric"]) for fragment in fragments]
        )
        self.assertNotIn("edge_tp", summary)
        counts = oracle.aggregate_fragment_metric_counts(fragments)
        self.assertEqual(counts["baseline_metric"]["edge_tp"], 30)
        self.assertEqual(counts["candidate_metric"]["edge_tp"], 33)
        self.assertEqual(counts["baseline_metric"]["edge_fp"], 4)
        self.assertEqual(counts["candidate_metric"]["edge_fp"], 1)

    def test_recursive_chains_remain_transaction_independent(self) -> None:
        baseline = {(1, 4), (2, 3), (3, 6), (4, 5)}
        truth = {(1, 3), (2, 4), (3, 5), (4, 6)}
        chains = oracle.build_chains(baseline, set(), truth)
        self.assertEqual([chain.length for chain in chains], [2, 2])
        self.assertEqual(
            set().union(*(set(chain.conflicts) for chain in chains)), baseline
        )

    def test_independent_chains_and_limits(self) -> None:
        baseline = {(1, 4), (2, 3), (3, 5)}
        truth = {(1, 3), (2, 4), (3, 6)}
        chains = oracle.build_chains(baseline, set(), truth)
        self.assertEqual(sorted(chain.length for chain in chains), [1, 2])
        self.assertEqual(sum(chain.length for chain in chains if chain.length <= 2), 3)

    def test_true_division_anchor_is_retained(self) -> None:
        baseline = {(1, 3), (1, 4), (2, 5)}
        baseline_true = {(1, 3)}
        truth = {(1, 3), (1, 5), (2, 4)}
        chains = oracle.build_chains(baseline, baseline_true, truth)
        merged = max(chains, key=lambda chain: chain.length)
        self.assertIn((1, 3), merged.add_edges)
        self.assertNotIn((1, 3), set(merged.conflicts) - set(merged.add_edges))

    def test_true_anchor_recursively_closes_components(self) -> None:
        baseline = {(1, 3), (2, 4), (3, 6)}
        baseline_true = {(1, 3)}
        truth = {(1, 3), (1, 4), (2, 3), (3, 5)}
        chains = oracle.build_chains(baseline, baseline_true, truth)
        self.assertEqual(sorted(chain.length for chain in chains), [1, 2])
        anchored = next(chain for chain in chains if (1, 4) in chain.missing_edges)
        self.assertIn((2, 3), anchored.missing_edges)
        self.assertIn((1, 3), anchored.add_edges)

    def test_frozen_compiler_expresses_multi_edge_chain(self) -> None:
        baseline = {(1, 4), (2, 3)}
        truth = {(1, 3), (2, 4)}
        chains = oracle.build_chains(baseline, set(), truth)
        candidate, finding = oracle.apply_chains(graph(baseline), "train:synthetic", chains)
        self.assertTrue(finding["expressible"])
        self.assertEqual(
            {(edge.source_id, edge.target_id) for edge in candidate.edges}, truth
        )

    def test_decision_arithmetic_boundary(self) -> None:
        self.assertLess(oracle.COLLAPSE_RATIO * 0.020416, oracle.ACTIONABILITY)
        self.assertGreaterEqual(oracle.COLLAPSE_RATIO * (oracle.ACTIONABILITY / oracle.COLLAPSE_RATIO), oracle.ACTIONABILITY)

    def test_candidate_rank_is_deterministic(self) -> None:
        synthetic = type(
            "SyntheticCache",
            (),
            {
                "detections": (
                    frame(0, [1, 2], [(0, 0, 0), (0, 10, 0)]),
                    frame(1, [3, 4], [(0, 0, 0), (0, 10, 0)]),
                    frame(2, [5, 6], [(0, 0, 0), (0, 10, 0)]),
                )
            },
        )()
        pairs = oracle.candidate_tensor(synthetic)
        self.assertIn((1, 3), pairs)
        self.assertIn((2, 4), pairs)
        self.assertIn((3, 5), pairs)
        self.assertNotIn((1, 5), pairs)

    def test_banked_fitcache_pool_regression(self) -> None:
        cache = oracle.candidate_cache("train:44b6_cf8fed6b")
        self.assertEqual(len(oracle.candidate_tensor(cache)), 117_995)


if __name__ == "__main__":
    unittest.main()
