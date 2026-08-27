from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_sa_swap.py")
SPEC = importlib.util.spec_from_file_location("sa_swap_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
lane = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lane
SPEC.loader.exec_module(lane)

from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionKind,
    GraphAction,
    compile_action,
    compile_batch,
)
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
)


def graph() -> SubmissionGraph:
    return SubmissionGraph(
        dataset="44b6_fixture",
        nodes={
            1: SubmissionNode(1, 0, 0, 0, 0),
            2: SubmissionNode(2, 0, 0, 4, 0),
            3: SubmissionNode(3, 1, 0, 0, 0),
            4: SubmissionNode(4, 1, 0, 4, 0),
        },
        edges=[SubmissionEdge(1, 3), SubmissionEdge(2, 4)],
    )


def multi_graph() -> SubmissionGraph:
    return SubmissionGraph(
        dataset="44b6_multi_fixture",
        nodes={
            1: SubmissionNode(1, 0, 0, 0, 0),
            2: SubmissionNode(2, 0, 0, 4, 0),
            3: SubmissionNode(3, 1, 0, 0, 0),
            4: SubmissionNode(4, 1, 0, 4, 0),
            5: SubmissionNode(5, 0, 0, 8, 0),
            6: SubmissionNode(6, 0, 0, 12, 0),
            7: SubmissionNode(7, 1, 0, 8, 0),
            8: SubmissionNode(8, 1, 0, 12, 0),
        },
        edges=[
            SubmissionEdge(1, 3),
            SubmissionEdge(2, 4),
            SubmissionEdge(5, 7),
            SubmissionEdge(6, 8),
        ],
    )


def swap_action(
    source: int = 1,
    target: int = 4,
    conflicts: tuple[tuple[int, int], ...] = ((1, 3), (2, 4)),
) -> GraphAction:
    return GraphAction(
        action_id=f"swap:{source}:{target}",
        kind=ActionKind.SWAP,
        add_edges=((source, target),),
        replace_outgoing_from=(source,),
        replace_incoming_to=(target,),
        expected_conflict_edges=conflicts,
    )


def action_record(
    source: int,
    target: int,
    conflicts: tuple[tuple[int, int], ...],
    source_index: int,
) -> object:
    return lane.ActionRecord(
        action=swap_action(source, target, conflicts),
        feature=np.full(49, float(source_index), dtype=np.float64),
        target=float(source_index),
        source_index=source_index,
        removed_edges=conflicts,
    )


class SwapLaneTests(unittest.TestCase):
    def test_fold_is_deterministic(self) -> None:
        identity = "train:44b6_cf8fed6b"
        self.assertEqual(lane.fold(identity), lane.fold(identity))
        self.assertIn(lane.fold(identity), range(5))

    def test_ridge_recovers_finite_scores(self) -> None:
        rng = np.random.default_rng(20260728)
        x = rng.normal(size=(200, 49))
        y = 3.0 * x[:, 0] - 2.0 * x[:, 48]
        head = lane.fit_ridge(x, y)
        score = head.score(x)
        self.assertEqual(score.shape, (200,))
        self.assertTrue(np.isfinite(score).all())
        self.assertGreater(np.corrcoef(score, y)[0, 1], 0.99)

    def test_complete_swap_removes_both_incidents(self) -> None:
        baseline = graph()
        action = GraphAction(
            action_id="swap:1:4",
            kind=ActionKind.SWAP,
            add_edges=((1, 4),),
            replace_outgoing_from=(1,),
            replace_incoming_to=(4,),
            expected_conflict_edges=((1, 3), (2, 4)),
        )
        compiled = compile_action(baseline, action)
        self.assertEqual(compiled.removed_edges, ((1, 3), (2, 4)))
        self.assertEqual(
            {(edge.source_id, edge.target_id) for edge in compiled.graph.edges},
            {(1, 4)},
        )

    def test_incomplete_conflict_proof_fails_closed(self) -> None:
        baseline = graph()
        action = GraphAction(
            action_id="swap:1:4",
            kind=ActionKind.SWAP,
            add_edges=((1, 4),),
            replace_outgoing_from=(1,),
            replace_incoming_to=(4,),
            expected_conflict_edges=((1, 3),),
        )
        with self.assertRaises(ValueError):
            compile_action(baseline, action)

    def test_complete_local_witness_calls_n1(self) -> None:
        baseline = graph()
        edges, outgoing, incoming = lane._edge_sets(baseline)
        lane._compile_local_witness(
            baseline,
            swap_action(),
            {(1, 3), (2, 4)},
            edges,
            outgoing,
            incoming,
        )

    def test_local_witness_omitted_conflict_fails_closed(self) -> None:
        baseline = graph()
        edges, outgoing, incoming = lane._edge_sets(baseline)
        with self.assertRaises(lane.SwapRunError):
            lane._compile_local_witness(
                baseline,
                swap_action(),
                {(1, 3)},
                edges,
                outgoing,
                incoming,
            )

    def test_local_witness_extra_conflict_fails_closed(self) -> None:
        baseline = graph()
        edges, outgoing, incoming = lane._edge_sets(baseline)
        with self.assertRaises(lane.SwapRunError):
            lane._compile_local_witness(
                baseline,
                swap_action(conflicts=((1, 3), (2, 4), (3, 4))),
                {(1, 3), (2, 4), (3, 4)},
                edges,
                outgoing,
                incoming,
            )

    def test_local_witness_incomplete_action_proof_fails_closed(self) -> None:
        baseline = graph()
        edges, outgoing, incoming = lane._edge_sets(baseline)
        with self.assertRaises(lane.SwapRunError):
            lane._compile_local_witness(
                baseline,
                swap_action(conflicts=((1, 3),)),
                {(1, 3), (2, 4)},
                edges,
                outgoing,
                incoming,
            )

    def test_local_witness_missing_endpoint_fails_closed(self) -> None:
        baseline = graph()
        edges, outgoing, incoming = lane._edge_sets(baseline)
        with self.assertRaises(lane.SwapRunError):
            lane._compile_local_witness(
                baseline,
                swap_action(target=99, conflicts=((1, 3),)),
                {(1, 3)},
                edges,
                outgoing,
                incoming,
            )

    def test_local_witness_time_mismatch_fails_closed(self) -> None:
        baseline = graph()
        edges, outgoing, incoming = lane._edge_sets(baseline)
        with self.assertRaises(lane.SwapRunError):
            lane._compile_local_witness(
                baseline,
                swap_action(target=2, conflicts=((1, 3),)),
                {(1, 3)},
                edges,
                outgoing,
                incoming,
            )

    def test_local_witness_degree_violation_fails_closed(self) -> None:
        baseline = graph()
        baseline.nodes[5] = SubmissionNode(5, 1, 0, 8, 0)
        baseline.edges.append(SubmissionEdge(2, 5))
        edges, outgoing, incoming = lane._edge_sets(baseline)
        conflicts = {(1, 3), (2, 4)}
        with self.assertRaises(lane.SwapRunError):
            lane._compile_local_witness(
                baseline,
                swap_action(conflicts=tuple(sorted(conflicts))),
                conflicts,
                edges,
                outgoing,
                incoming,
            )

    def test_local_witness_remote_nonboundary_edge_is_irrelevant(self) -> None:
        baseline = graph()
        baseline.nodes[5] = SubmissionNode(5, 2, 0, 8, 0)
        baseline.edges.append(SubmissionEdge(3, 5))
        edges, outgoing, incoming = lane._edge_sets(baseline)
        lane._compile_local_witness(
            baseline,
            swap_action(),
            {(1, 3), (2, 4)},
            edges,
            outgoing,
            incoming,
        )

    def test_local_witness_altered_node_payload_fails_closed(self) -> None:
        baseline = graph()
        edges, outgoing, incoming = lane._edge_sets(baseline)
        witness = SubmissionGraph(
            dataset=baseline.dataset,
            nodes=dict(baseline.nodes),
            edges=list(baseline.edges),
        )
        witness.nodes[1] = SubmissionNode(1, 0, 99, 0, 0)
        with self.assertRaises(lane.SwapRunError):
            lane._validate_local_witness(
                baseline,
                swap_action(),
                {(1, 3), (2, 4)},
                edges,
                outgoing,
                incoming,
                witness,
            )

    def test_overlapping_batch_fails_closed(self) -> None:
        baseline = graph()
        first = GraphAction(
            action_id="swap:1:4",
            kind=ActionKind.SWAP,
            add_edges=((1, 4),),
            replace_outgoing_from=(1,),
            replace_incoming_to=(4,),
            expected_conflict_edges=((1, 3), (2, 4)),
        )
        second = GraphAction(
            action_id="swap:2:3",
            kind=ActionKind.SWAP,
            add_edges=((2, 3),),
            replace_outgoing_from=(2,),
            replace_incoming_to=(3,),
            expected_conflict_edges=((1, 3), (2, 4)),
        )
        with self.assertRaises(ValueError):
            compile_batch(baseline, (first, second))

    def test_edge_class_matches_true_valid_and_ignored(self) -> None:
        mapping = {1: 10, 2: 20, 3: 30}
        truth = {(10, 20)}
        outgoing = {10: {20}}
        incoming = {20: {10}}
        self.assertEqual(
            lane._edge_class((1, 2), mapping, truth, outgoing, incoming),
            (True, True),
        )
        self.assertEqual(
            lane._edge_class((1, 3), mapping, truth, outgoing, incoming),
            (False, True),
        )
        self.assertEqual(
            lane._edge_class((3, 3), mapping, truth, outgoing, incoming),
            (False, False),
        )

    def test_selection_uses_n1_batch_and_noop_threshold(self) -> None:
        baseline = graph()
        action = GraphAction(
            action_id="swap:1:4",
            kind=ActionKind.SWAP,
            add_edges=((1, 4),),
            replace_outgoing_from=(1,),
            replace_incoming_to=(4,),
            expected_conflict_edges=((1, 3), (2, 4)),
        )
        record = lane.ActionRecord(
            action=action,
            feature=np.zeros(49),
            target=1.0,
            source_index=0,
            removed_edges=((1, 3), (2, 4)),
        )
        noop_graph, noop = lane._select(
            baseline, [record], np.asarray([0.0]), 16
        )
        self.assertEqual(noop["selected"], 0)
        self.assertEqual(
            {(edge.source_id, edge.target_id) for edge in noop_graph.edges},
            {(1, 3), (2, 4)},
        )
        swapped_graph, swapped = lane._select(
            baseline, [record], np.asarray([0.1]), 16
        )
        self.assertEqual(swapped["selected"], 1)
        self.assertEqual(
            {(edge.source_id, edge.target_id) for edge in swapped_graph.edges},
            {(1, 4)},
        )

    def test_selection_tie_conflict_and_cap_ordering(self) -> None:
        baseline = multi_graph()
        independent_late = action_record(
            5, 8, ((5, 7), (6, 8)), source_index=2
        )
        first_by_tie = action_record(
            1, 4, ((1, 3), (2, 4)), source_index=0
        )
        overlapping = action_record(
            2, 3, ((1, 3), (2, 4)), source_index=1
        )
        records = [independent_late, overlapping, first_by_tie]
        scores = np.asarray([0.5, 0.5, 0.5])
        selected_graph, selected = lane._select(
            baseline, records, scores, cap=2
        )
        self.assertEqual(
            selected["selected_action_ids"], ["swap:1:4", "swap:5:8"]
        )
        self.assertEqual(selected["batch_conflict_rejected"], 1)
        self.assertEqual(
            {(edge.source_id, edge.target_id) for edge in selected_graph.edges},
            {(1, 4), (5, 8)},
        )
        capped_graph, capped = lane._select(baseline, records, scores, cap=1)
        self.assertEqual(capped["selected_action_ids"], ["swap:1:4"])
        self.assertTrue(capped["cap_active"])
        self.assertEqual(
            {(edge.source_id, edge.target_id) for edge in capped_graph.edges},
            {(1, 4), (5, 7), (6, 8)},
        )

    def test_streamed_execution_is_exactly_in_memory_semantics(self) -> None:
        rng = np.random.default_rng(19850910)
        identities = ("train:44b6_fixture_a", "train:44b6_fixture_b")
        records_by_identity = {}
        for identity_index, identity in enumerate(identities):
            records = []
            for row in range(40):
                unique_index = identity_index * 40 + row
                feature = rng.normal(size=49).astype(np.float64)
                target = float(
                    2.5 * feature[0]
                    - 0.75 * feature[17]
                    + 1.25 * feature[48]
                )
                source = 1000 + 2 * unique_index
                target_node = 1001 + 2 * unique_index
                conflict = (source, target_node + 1000)
                action = GraphAction(
                    action_id=f"swap:{source}:{target_node}",
                    kind=ActionKind.SWAP,
                    add_edges=((source, target_node),),
                    replace_outgoing_from=(source,),
                    replace_incoming_to=(target_node,),
                    expected_conflict_edges=(conflict,),
                )
                records.append(
                    lane.ActionRecord(
                        action=action,
                        feature=feature,
                        target=target,
                        source_index=unique_index,
                        removed_edges=(conflict,),
                    )
                )
            records_by_identity[identity] = records

        reference_alpha_x, reference_y = lane._fit_arrays_in_memory(
            records_by_identity, identities, beta=False
        )
        reference_beta_x, reference_beta_y = lane._fit_arrays_in_memory(
            records_by_identity, identities, beta=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            data = {}
            for index, identity in enumerate(identities):
                store = work / f"records_{index}"
                original = records_by_identity[identity]
                lane._write_record_store(store, original)
                stored_arrays = lane._load_record_arrays(store)
                original_arrays = lane._record_arrays(original)
                for name in original_arrays:
                    np.testing.assert_array_equal(
                        stored_arrays[name], original_arrays[name]
                    )
                roundtrip = lane._load_records(store)
                self.assertEqual(
                    [record.action.action_id for record in roundtrip],
                    [record.action.action_id for record in original],
                )
                self.assertEqual(
                    [record.source_index for record in roundtrip],
                    [record.source_index for record in original],
                )
                np.testing.assert_array_equal(
                    np.stack([record.feature for record in roundtrip]),
                    np.stack([record.feature for record in original]),
                )
                np.testing.assert_array_equal(
                    np.asarray([record.target for record in roundtrip]),
                    np.asarray([record.target for record in original]),
                )
                self.assertEqual(
                    [record.removed_edges for record in roundtrip],
                    [record.removed_edges for record in original],
                )
                data[identity] = lane.VideoData(
                    identity=identity,
                    baseline_path=Path("unused.csv"),
                    baseline_sha256="0" * 64,
                    baseline_metrics={},
                    baseline_graph_sha256="1" * 64,
                    feature_sha256="2" * 64,
                    record_store=store,
                    record_count=len(original),
                    witness_compile_count=len(original),
                    witness_reject_count=0,
                    exclusions=lane.Counter(),
                    source_proposals=len(original),
                )

            streamed_x, streamed_y = lane._assemble_streamed_fold(
                data,
                identities,
                work / "fold_features.npy",
                work / "fold_targets.npy",
            )
            np.testing.assert_array_equal(streamed_x, reference_alpha_x)
            np.testing.assert_array_equal(streamed_y, reference_y)
            reference_alpha = lane.fit_ridge(reference_alpha_x, reference_y)
            streamed_alpha = lane.fit_ridge(streamed_x, streamed_y)
            np.testing.assert_array_equal(
                streamed_alpha.weights, reference_alpha.weights
            )
            np.testing.assert_array_equal(
                streamed_alpha.mean, reference_alpha.mean
            )
            np.testing.assert_array_equal(
                streamed_alpha.scale, reference_alpha.scale
            )
            self.assertEqual(
                streamed_alpha.intercept, reference_alpha.intercept
            )
            np.testing.assert_array_equal(
                streamed_alpha.score(reference_alpha_x),
                reference_alpha.score(reference_alpha_x),
            )

            streamed_x[:, lane.TRACKASTRA_COLUMN] = 0.0
            streamed_x.flush()
            np.testing.assert_array_equal(streamed_x, reference_beta_x)
            np.testing.assert_array_equal(streamed_y, reference_beta_y)
            reference_beta = lane.fit_ridge(reference_beta_x, reference_beta_y)
            streamed_beta = lane.fit_ridge(streamed_x, streamed_y)
            np.testing.assert_array_equal(
                streamed_beta.weights, reference_beta.weights
            )
            self.assertEqual(streamed_beta.intercept, reference_beta.intercept)
            np.testing.assert_array_equal(
                streamed_beta.score(reference_beta_x),
                reference_beta.score(reference_beta_x),
            )

            selection_records = [
                action_record(5, 8, ((5, 7), (6, 8)), 2),
                action_record(2, 3, ((1, 3), (2, 4)), 1),
                action_record(1, 4, ((1, 3), (2, 4)), 0),
            ]
            selection_store = work / "selection_records"
            lane._write_record_store(selection_store, selection_records)
            streamed_selection_records = lane._load_records(selection_store)
            selection_scores = np.asarray([0.5, 0.5, 0.5])
            original_graph, original_selection = lane._select(
                multi_graph(), selection_records, selection_scores, 2
            )
            streamed_graph, streamed_selection = lane._select(
                multi_graph(), streamed_selection_records, selection_scores, 2
            )
            self.assertEqual(
                streamed_selection["selected_action_ids"],
                original_selection["selected_action_ids"],
            )
            self.assertEqual(
                streamed_selection["batch_conflict_rejected"],
                original_selection["batch_conflict_rejected"],
            )
            self.assertEqual(
                streamed_selection["cap_active"],
                original_selection["cap_active"],
            )
            self.assertEqual(
                lane.graph_sha256(streamed_graph),
                lane.graph_sha256(original_graph),
            )


if __name__ == "__main__":
    unittest.main()
