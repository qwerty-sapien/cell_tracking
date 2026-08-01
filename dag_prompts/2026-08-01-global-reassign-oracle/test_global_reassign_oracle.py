#!/usr/bin/env python3
"""Outcome-free focused tests for the global reassignment harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("global_reassign_oracle", HERE / "global_reassign_oracle.py")
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)

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


class ChainTests(unittest.TestCase):
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
        synthetic = graph(set())
        pairs = oracle.candidate_tensor(synthetic, (1.0, 1.0, 1.0))
        self.assertIn((1, 3), pairs)
        self.assertIn((2, 4), pairs)
        self.assertIn((3, 5), pairs)
        self.assertNotIn((1, 5), pairs)


if __name__ == "__main__":
    unittest.main()
