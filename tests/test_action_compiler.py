from __future__ import annotations

import unittest

from celltrack_alpha.action_compiler import (
    ActionCompileError,
    ActionKind,
    GraphAction,
    action_sha256,
    compile_action,
    compile_batch,
    graph_sha256,
    interface_schema,
)
from celltrack_alpha.kaggle_calibration import SubmissionEdge, SubmissionGraph, SubmissionNode


def node(node_id: int, t: int) -> SubmissionNode:
    return SubmissionNode(node_id=node_id, t=t, z=0, y=node_id, x=0)


def fixture() -> SubmissionGraph:
    nodes = {value: node(value, time) for value, time in (
        (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 0), (7, 1),
        (8, 0), (9, 1), (10, 2), (11, 0), (12, 1), (13, 1),
    )}
    edges = [
        SubmissionEdge(1, 2),
        SubmissionEdge(2, 4),
        SubmissionEdge(6, 7),
        SubmissionEdge(8, 9),
        SubmissionEdge(11, 12),
        SubmissionEdge(11, 13),
    ]
    return SubmissionGraph("fixture", nodes, edges)


def valid_actions() -> dict[str, GraphAction]:
    return {
        "noop": GraphAction("a.noop", ActionKind.NOOP),
        "add": GraphAction(
            "b.add",
            ActionKind.ADD_EDGE,
            add_edges=((3, 5),),
        ),
        "delete": GraphAction(
            "c.delete",
            ActionKind.DELETE_EDGES,
            delete_edges=((2, 4),),
        ),
        "swap": GraphAction(
            "d.swap",
            ActionKind.SWAP,
            add_edges=((6, 9),),
            replace_outgoing_from=(6,),
            replace_incoming_to=(9,),
            expected_conflict_edges=((6, 7), (8, 9)),
        ),
        "fork": GraphAction(
            "e.fork",
            ActionKind.DIVISION_FORK,
            add_edges=((11, 4), (11, 5)),
            replace_outgoing_from=(11,),
            replace_incoming_to=(4, 5),
            expected_conflict_edges=((2, 4), (11, 12), (11, 13)),
        ),
        "restore": GraphAction(
            "f.restore",
            ActionKind.RESTORE_TOPOLOGY,
            add_nodes=(node(20, 2), node(21, 3)),
            add_edges=((3, 20), (20, 21)),
        ),
    }


class ActionCompilerTest(unittest.TestCase):
    def test_every_action_kind_compiles_alone(self) -> None:
        baseline = fixture()
        before = graph_sha256(baseline)
        for name, action in valid_actions().items():
            with self.subTest(name=name):
                compiled = compile_action(baseline, action)
                self.assertEqual(compiled.before_graph_sha256, before)
                self.assertEqual(len(compiled.action_sha256), 64)
                self.assertEqual(len(compiled.after_graph_sha256), 64)

    def test_incident_conflict_omission_fails_closed(self) -> None:
        action = GraphAction(
            "bad.swap",
            ActionKind.SWAP,
            add_edges=((6, 9),),
            replace_outgoing_from=(6,),
            replace_incoming_to=(9,),
            expected_conflict_edges=((6, 7),),
        )
        with self.assertRaisesRegex(ActionCompileError, "incident-conflict proof mismatch"):
            compile_action(fixture(), action)

    def test_missing_endpoint_fails_closed(self) -> None:
        action = GraphAction(
            "bad.endpoint",
            ActionKind.ADD_EDGE,
            add_edges=((3, 999),),
        )
        with self.assertRaisesRegex(ActionCompileError, "missing final endpoint"):
            compile_action(fixture(), action)

    def test_temporal_direction_fails_closed(self) -> None:
        action = GraphAction(
            "bad.time",
            ActionKind.ADD_EDGE,
            add_edges=((3, 1),),
        )
        with self.assertRaisesRegex(ValueError, "non-forward edge"):
            compile_action(fixture(), action)

    def test_merge_fails_closed(self) -> None:
        action = GraphAction(
            "bad.merge",
            ActionKind.ADD_EDGE,
            add_edges=((3, 7),),
        )
        with self.assertRaisesRegex(ActionCompileError, "free source and target"):
            compile_action(fixture(), action)

    def test_outdegree_greater_than_two_fails_closed(self) -> None:
        action = GraphAction(
            "bad.degree",
            ActionKind.RESTORE_TOPOLOGY,
            add_nodes=(node(21, 1),),
            add_edges=((11, 21),),
        )
        with self.assertRaisesRegex(ValueError, "out-degree greater than two"):
            compile_action(fixture(), action)

    def test_valid_batch_is_order_independent(self) -> None:
        actions = valid_actions()
        selected = (actions["delete"], actions["swap"], actions["restore"])
        left = compile_batch(fixture(), selected)
        right = compile_batch(fixture(), tuple(reversed(selected)))
        self.assertEqual(left.batch_sha256, right.batch_sha256)
        self.assertEqual(left.after_graph_sha256, right.after_graph_sha256)
        self.assertEqual(
            tuple(action.action_id for action in left.actions),
            ("c.delete", "d.swap", "f.restore"),
        )

    def test_batch_rejects_shared_conflict_ownership(self) -> None:
        swap = valid_actions()["swap"]
        restore = GraphAction(
            "g.restore",
            ActionKind.RESTORE_TOPOLOGY,
            add_nodes=(node(22, 1),),
            add_edges=((8, 22), (22, 10)),
            replace_outgoing_from=(8,),
            replace_incoming_to=(10,),
            expected_conflict_edges=((8, 9),),
        )
        with self.assertRaisesRegex(ActionCompileError, "removed by both"):
            compile_batch(fixture(), (swap, restore))

    def test_action_hash_ignores_tuple_order(self) -> None:
        first = GraphAction(
            "ordered",
            ActionKind.DELETE_EDGES,
            delete_edges=((1, 2), (2, 4)),
        )
        second = GraphAction(
            "ordered",
            ActionKind.DELETE_EDGES,
            delete_edges=((2, 4), (1, 2)),
        )
        self.assertEqual(action_sha256(first), action_sha256(second))

    def test_interface_lists_all_kinds_and_invariants(self) -> None:
        schema = interface_schema()
        self.assertEqual(set(schema["action_kinds"]), {kind.value for kind in ActionKind})
        self.assertEqual(len(schema["invariants"]), 10)


if __name__ == "__main__":
    unittest.main()
