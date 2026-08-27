from __future__ import annotations

import copy
import sys
from pathlib import Path

import polars as pl
import tracksdata as td

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "external/kaggle-cell-tracking-competition/src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import instrument_division_forks as idf  # noqa: E402


def _build_graph(nodes: dict, edges: list) -> tuple[td.graph.InMemoryGraph, dict[str, int]]:
    """Build an InMemoryGraph, returning (graph, name->node_id mapping).

    Mirrors external/kaggle-cell-tracking-competition/tests/test_division_metrics.py's
    own helper exactly, so fixtures here are directly comparable to the official tests.
    """
    g = td.graph.InMemoryGraph()
    g.add_node_attr_key("z", pl.Float64, 0.0)
    g.add_node_attr_key("y", pl.Float64, 0.0)
    g.add_node_attr_key("x", pl.Float64, 0.0)
    ids = {}
    for name, attrs in nodes.items():
        ids[name] = g.add_node(attrs=copy.deepcopy(attrs))
    for src, tgt in edges:
        g.add_edge(ids[src], ids[tgt], {})
    return g, ids


def _empty_graph() -> td.graph.InMemoryGraph:
    g = td.graph.InMemoryGraph()
    g.add_node_attr_key("z", pl.Float64, 0.0)
    g.add_node_attr_key("y", pl.Float64, 0.0)
    g.add_node_attr_key("x", pl.Float64, 0.0)
    return g


class TestClassifyPopulation:
    def test_zero_gt_division_video_forks_are_labeled_not_skipped(self) -> None:
        """A video with zero GT divisions must still have its forks classified.

        This is the exact gap the old diagnose_division_failure.py has: it restricts
        its video loop to identities with division_fn > 0 in a prior score artifact,
        so a zero-GT-division video's forks are never even visited. R2 found this is
        precisely where 7 of cv57's 11 division FPs live (2 zero-GT videos holding
        216/368 forks). A fork with a far-away, unrelated GT node nearby (no local
        topology/branch evidence at all) must be reported with channel "invisible",
        not silently dropped from the population report.
        """
        # A single dividing node with two children, isolated in space -- no GT nodes
        # anywhere near it, so it can carry no division evidence at all.
        pred, pred_ids = _build_graph(
            {
                "P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "C1": {"t": 1, "z": 0.0, "y": 5.0, "x": 0.0},
                "C2": {"t": 1, "z": 0.0, "y": -5.0, "x": 0.0},
            },
            [("P", "C1"), ("P", "C2")],
        )
        gt = _empty_graph()

        result = idf.classify_population(
            [("zero_gt_video", pred, gt)], scale=(1.0, 1.0, 1.0)
        )

        assert result["per_video"][0]["identity"] == "zero_gt_video"
        assert result["per_video"][0]["gt_division_count"] == 0
        fork_records = [f for f in result["forks"] if f["identity"] == "zero_gt_video"]
        assert len(fork_records) == 1
        assert fork_records[0]["fork_id"] == pred_ids["P"]
        assert fork_records[0]["channel"] == "invisible"
        assert fork_records[0]["video_gt_division_count"] == 0

    def test_division_bearing_video_reuses_official_classification_unmodified(self) -> None:
        """A video with a GT division must be classified via the existing 7-way
        taxonomy (classify_division), reusing the frozen scorer's own outputs --
        this test only proves the new population loop calls it, not that the
        taxonomy logic itself is new (it is unmodified, imported from
        diagnose_division_failure).
        """
        # A division the prediction fully recovers: same structure the official test
        # suite uses (test_division_metrics.py's own single-division fixture shape).
        shared_nodes = {
            "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
            "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
            "D": {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
            "E": {"t": 3, "z": 0.0, "y": 5.0, "x": 0.0},
        }
        shared_edges = [("A", "B"), ("B", "C1"), ("B", "C2"), ("C2", "D"), ("C1", "E")]
        gt, _ = _build_graph(shared_nodes, shared_edges)
        pred, _ = _build_graph(shared_nodes, shared_edges)

        result = idf.classify_population([("recovered_video", pred, gt)], scale=(1.0, 1.0, 1.0))

        assert result["per_video"][0]["gt_division_count"] == 1
        assert result["divisions"][0]["classification"] == "officially_recovered"
        assert result["summary"]["classification_counts"] == {"officially_recovered": 1}

    def test_summary_aggregates_across_multiple_videos(self) -> None:
        """Population-level summary must sum gt_division_count and forks across
        every video passed in, not just the first."""
        pred_zero, _ = _build_graph(
            {"P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
             "C1": {"t": 1, "z": 0.0, "y": 5.0, "x": 0.0},
             "C2": {"t": 1, "z": 0.0, "y": -5.0, "x": 0.0}},
            [("P", "C1"), ("P", "C2")],
        )
        gt_zero = _empty_graph()

        shared_nodes = {
            "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
            "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
            "D": {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
            "E": {"t": 3, "z": 0.0, "y": 5.0, "x": 0.0},
        }
        shared_edges = [("A", "B"), ("B", "C1"), ("B", "C2"), ("C2", "D"), ("C1", "E")]
        gt_div, _ = _build_graph(shared_nodes, shared_edges)
        pred_div, _ = _build_graph(shared_nodes, shared_edges)

        result = idf.classify_population(
            [("zero_gt_video", pred_zero, gt_zero), ("recovered_video", pred_div, gt_div)],
            scale=(1.0, 1.0, 1.0),
        )

        assert result["summary"]["gt_divisions"] == 1
        assert len(result["per_video"]) == 2
        # one fork from the zero-GT video ("P") + one from the recovered-division video ("B")
        assert len(result["forks"]) == 2


class TestProposeAdditionalDaughters:
    def test_proposes_only_nodes_within_scorer_distance_of_the_fork(self) -> None:
        """DIV-WIDESAT (R2 §3.2): additional daughter candidates must be restricted
        to the same physical radius (7um, the scorer's own DistanceMatching bound)
        the frozen scorer already uses -- no new free parameter. A same-frame node
        just inside 7um is proposed; one just outside is not; an already-attached
        existing daughter is never re-proposed.
        """
        pred, ids = _build_graph(
            {
                "P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "D1": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},   # existing daughter
                "NEAR": {"t": 1, "z": 0.0, "y": 6.9, "x": 0.0},  # 6.9um away: in range
                "FAR": {"t": 1, "z": 0.0, "y": 7.1, "x": 0.0},   # 7.1um away: out of range
            },
            [("P", "D1")],
        )
        proposals = idf.propose_additional_daughters(
            pred, fork_id=ids["P"], scale=(1.0, 1.0, 1.0), max_distance=7.0
        )
        assert proposals == {ids["NEAR"]}


class TestSweepWidesat:
    def test_widening_recovers_a_division_missed_at_width_two(self) -> None:
        """DIV-WIDESAT (R2 SS3.2): a fork whose TRUE second daughter is farther
        from the parent than an already-attached wrong node should be recoverable
        once the width cap allows a 3rd branch -- exercising the actual mechanism
        DIV-WIDESAT bets on (extra branches are edge-FP-free; N1's DIV-K3 finding).
        """
        # GT: parent P divides into true daughters T1, T2.
        gt, gt_ids = _build_graph(
            {
                "P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "T1": {"t": 1, "z": 0.0, "y": 3.0, "x": 0.0},
                "T2": {"t": 1, "z": 0.0, "y": -3.0, "x": 0.0},
                "G1": {"t": 2, "z": 0.0, "y": 3.0, "x": 0.0},
                "G2": {"t": 2, "z": 0.0, "y": -3.0, "x": 0.0},
            },
            [("P", "T1"), ("P", "T2"), ("T1", "G1"), ("T2", "G2")],
        )
        # Prediction: P correctly placed and correctly linked to T1, but its
        # *second* child slot went to a spurious nearby node instead of T2. T2
        # itself is separately, correctly detected+placed by the pipeline as an
        # unattached same-frame node (out_degree 2 cap already used by T1+decoy).
        pred, pred_ids = _build_graph(
            {
                "P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "T1": {"t": 1, "z": 0.0, "y": 3.0, "x": 0.0},
                "DECOY": {"t": 1, "z": 0.0, "y": 0.5, "x": 0.0},
                "T2": {"t": 1, "z": 0.0, "y": -3.0, "x": 0.0},
                "G1": {"t": 2, "z": 0.0, "y": 3.0, "x": 0.0},
            },
            [("P", "T1"), ("P", "DECOY"), ("T1", "G1")],
        )
        scale = (1.0, 1.0, 1.0)

        result = idf.sweep_widesat(
            [("v", pred, gt)], scale=scale, widths=(2, 3), max_distance=7.0
        )

        assert result[2]["division_tp"] == 0  # baseline: DECOY blocks the true pairing
        assert result[3]["division_tp"] == 1  # widened: T2 gets attached too, now recoverable


class TestGeometricForkGenerator:
    def test_creates_a_second_child_for_a_forkless_matched_parent_and_recovers_tp(self) -> None:
        """DIV-CHANGEPOINT's cheapest-first premise test (R2 kill line #3): before
        any change-point sophistication, check whether the SAME geometric proximity
        test DIV-WIDESAT uses, applied as a GENERATOR on local_parent_fork_missing
        divisions (propose a second child for an out_degree<2 matched parent, not
        just widen an already-out_degree==2 fork), recovers anything at all. If a
        plausible second child exists nearby and is spliced in, the division should
        become officially recoverable.
        """
        gt, gt_ids = _build_graph(
            {
                "P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "T1": {"t": 1, "z": 0.0, "y": 3.0, "x": 0.0},
                "T2": {"t": 1, "z": 0.0, "y": -3.0, "x": 0.0},
                "G1": {"t": 2, "z": 0.0, "y": 3.0, "x": 0.0},
                "G2": {"t": 2, "z": 0.0, "y": -3.0, "x": 0.0},
            },
            [("P", "T1"), ("P", "T2"), ("T1", "G1"), ("T2", "G2")],
        )
        # Prediction: P is correctly matched but only ever linked to T1 -- no fork
        # anywhere near P (out_degree(P) == 1). T2 exists as an unattached node.
        pred, pred_ids = _build_graph(
            {
                "P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "T1": {"t": 1, "z": 0.0, "y": 3.0, "x": 0.0},
                "T2": {"t": 1, "z": 0.0, "y": -3.0, "x": 0.0},
                "G1": {"t": 2, "z": 0.0, "y": 3.0, "x": 0.0},
            },
            [("P", "T1"), ("T1", "G1")],
        )
        scale = (1.0, 1.0, 1.0)

        before = idf.classify_population([("v", pred, gt)], scale=scale)
        assert before["divisions"][0]["classification"] == "local_parent_fork_missing"

        widened = idf.generate_forks_for_forkless_parents(
            pred, gt, scale=scale, max_distance=7.0
        )
        after_scores = idf.division_metrics.score_divisions(widened, gt, scale, 7.0)
        assert after_scores.scores[gt_ids["P"]] == 1
