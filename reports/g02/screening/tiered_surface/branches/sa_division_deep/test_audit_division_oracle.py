from __future__ import annotations

import unittest

from audit_division_oracle import (
    Node,
    classify_roles,
    components,
    internal_to_submission_ids,
    suppress_oracle_actions,
)


class ComponentAuditTests(unittest.TestCase):
    def test_present_fragmented(self) -> None:
        component = components([1, 2, 3, 4, 5], [(1, 2), (3, 4)])
        label, evidence = classify_roles(True, [1], [[3, 4], [5]], component)
        self.assertEqual(label, "PRESENT_FRAGMENTED")
        self.assertEqual(evidence["spanning_component_count"], 0)
        self.assertEqual(evidence["role_component_count"], 3)

    def test_present_connected_without_local_fork(self) -> None:
        component = components([1, 2, 3, 4], [(1, 2), (2, 3), (2, 4)])
        label, evidence = classify_roles(True, [1], [[3], [4]], component)
        self.assertEqual(label, "PRESENT_CONNECTED_NO_LOCAL_FORK")
        self.assertEqual(evidence["spanning_component_count"], 1)

    def test_role_missing(self) -> None:
        label, evidence = classify_roles(False, [], [[], []], {})
        self.assertEqual(label, "ROLE_MISSING")
        self.assertIsNone(evidence["role_component_count"])

    def test_component_identity_is_order_independent(self) -> None:
        first = components([1, 2, 3, 4], [(3, 4), (1, 2), (2, 3)])
        second = components([4, 3, 2, 1], [(2, 3), (1, 2), (3, 4)])
        self.assertEqual(first, second)

    def test_missing_role_node_fails_closed(self) -> None:
        with self.assertRaisesRegex(AssertionError, "missing prediction nodes"):
            classify_roles(True, [1], [[2], [99]], components([1, 2], [(1, 2)]))

    def test_internal_id_conversion_matches_official_sort(self) -> None:
        nodes = {
            30: Node(30, 2, 0, 0, 0),
            10: Node(10, 0, 0, 0, 0),
            20: Node(20, 0, 0, 0, 0),
        }
        self.assertEqual(internal_to_submission_ids(nodes), {0: 10, 1: 20, 2: 30})

    def test_suppression_is_exact(self) -> None:
        def event(
            action_id: str,
            divider: int,
            *,
            persistent: bool,
            parent: int,
        ) -> dict[str, object]:
            return {
                "identity": "train:44b6_fixture",
                "gt_divider_id": divider,
                "event_index": divider,
                "oracle_action": {
                    "action_id": action_id,
                    "action_sha256": action_id.zfill(64),
                    "persistent_both_branches": persistent,
                    "selected_parent": {"node_id": parent},
                    "selected_children": [{"node_id": parent + 1}, {"node_id": parent + 2}],
                    "persistence_evidence": [
                        {"node_id": parent + 3},
                        {"node_id": parent + 4},
                    ],
                },
            }

        selected, suppressed = suppress_oracle_actions(
            [
                event("a", 1, persistent=True, parent=10),
                event("b", 2, persistent=True, parent=10),
                event("c", 3, persistent=False, parent=30),
            ]
        )
        self.assertEqual([row["action_id"] for row in selected], ["a"])
        self.assertEqual(
            [row["reason"] for row in suppressed],
            ["one_to_one_node_conflict", "missing_branch_persistence"],
        )


if __name__ == "__main__":
    unittest.main()
