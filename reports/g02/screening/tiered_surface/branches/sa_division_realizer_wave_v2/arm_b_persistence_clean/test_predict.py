#!/usr/bin/env python3
"""Target-free structural tests for predict.py."""

from __future__ import annotations

import unittest

from predict import (
    Candidate,
    FailClosed,
    Graph,
    Node,
    apply_candidate,
    assert_zero_swapouts,
    generate_candidates,
    graph_csv_bytes,
    parse_graph_bytes,
    select_candidate,
    verify_only_parent_changed,
)


def node(node_id: int, t: int, x: int, y: int = 0, z: int = 0) -> Node:
    return Node(node_id=node_id, t=t, z=z, y=y, x=x)


class PredictorTests(unittest.TestCase):
    def clean_graph(self) -> Graph:
        nodes = {
            1: node(1, 0, 0),
            2: node(2, 1, 1),
            3: node(3, 1, -1),
            4: node(4, 2, 2),
            5: node(5, 2, -2),
            6: node(6, 0, 30),
            7: node(7, 1, 35),
            8: node(8, 2, 40),
        }
        return Graph(
            "synthetic",
            nodes,
            frozenset(((1, 2), (2, 4), (3, 5), (6, 7), (7, 8))),
        )

    def test_clean_persistence_event_is_canonical_early_fork(self) -> None:
        graph = self.clean_graph()
        candidates, _scales, _upper = generate_candidates(graph)
        selected = select_candidate(candidates)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual((selected.p, selected.a, selected.b), (1, 2, 3))
        result = apply_candidate(graph, selected)
        self.assertIn((1, 2), result.edges)
        self.assertIn((1, 3), result.edges)
        self.assertIn((2, 4), result.edges)
        self.assertIn((3, 5), result.edges)
        verify_only_parent_changed(graph, result, selected)

    def test_no_root_daughter_means_no_event(self) -> None:
        graph = self.clean_graph()
        edges = set(graph.edges)
        edges.add((6, 3))
        changed = Graph(graph.dataset, graph.nodes, frozenset(edges))
        candidates, _scales, _upper = generate_candidates(changed)
        self.assertIsNone(select_candidate(candidates))

    def test_exact_global_tie_suppresses_video(self) -> None:
        candidates = [
            Candidate(1, 2, 3, 4, 5, 1.0),
            Candidate(11, 12, 13, 14, 15, 1.0),
        ]
        self.assertIsNone(select_candidate(candidates))

    def test_shared_node_conflict_keeps_unique_minimum(self) -> None:
        candidates = [
            Candidate(1, 2, 3, 4, 5, 1.0),
            Candidate(1, 2, 6, 4, 7, 2.0),
        ]
        self.assertEqual(select_candidate(candidates), candidates[0])

    def test_duplicate_edge_input_fails_closed(self) -> None:
        data = (
            b"id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n"
            b"0,s,node,1,0,0,0,0,-1,-1\n"
            b"1,s,node,2,1,0,0,1,-1,-1\n"
            b"2,s,edge,-1,-1,-1,-1,-1,1,2\n"
            b"3,s,edge,-1,-1,-1,-1,-1,1,2\n"
        )
        with self.assertRaises(FailClosed):
            parse_graph_bytes(data, "synthetic.csv")

    def test_csv_round_trip_has_unique_ids_nodes_and_edges(self) -> None:
        graph = self.clean_graph()
        candidates, _scales, _upper = generate_candidates(graph)
        selected = select_candidate(candidates)
        result = apply_candidate(graph, selected)
        reparsed = parse_graph_bytes(graph_csv_bytes(result), "synthetic.csv")
        self.assertEqual(reparsed.nodes, result.nodes)
        self.assertEqual(reparsed.edges, result.edges)

    def test_guard_ignores_swapins_but_rejects_swapouts(self) -> None:
        assert_zero_swapouts(
            {"swapins": 10, "swapouts": 20},
            {"swapins": 99, "swapouts": 20},
        )
        with self.assertRaises(FailClosed):
            assert_zero_swapouts(
                {"swapins": 10, "swapouts": 20},
                {"swapins": 10, "swapouts": 21},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
