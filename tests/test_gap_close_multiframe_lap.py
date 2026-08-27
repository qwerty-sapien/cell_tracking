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

import gap_close_multiframe_lap as gcl  # noqa: E402
from tracking_cellmot import metrics  # noqa: E402


def _build_graph(nodes: dict, edges: list) -> tuple[td.graph.InMemoryGraph, dict[str, int]]:
    """Mirrors tests/test_instrument_division_forks.py's helper exactly."""
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


class TestFindBrokenEndpoints:
    def test_finds_the_tail_and_head_of_a_one_frame_gap(self) -> None:
        # A(t=0) -> [missing t=1] -> C(t=2): no predicted edge across the gap.
        pred, ids = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 20.0},
            },
            [],
        )
        tails, heads = gcl.find_broken_endpoints(pred)
        assert tails == [ids["A"]]
        assert heads == [ids["C"]]

    def test_track_ending_at_the_video_boundary_is_not_a_tail(self) -> None:
        # A(t=0) -> B(t=1) is a complete track that legitimately ends at the last
        # frame of the video; B must not be treated as a broken tail.
        pred, ids = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 5.0},
            },
            [("A", "B")],
        )
        tails, heads = gcl.find_broken_endpoints(pred)
        assert tails == []
        assert heads == []


class TestCloseGapsInGraph:
    def test_bridges_a_one_missing_frame_gap_and_recovers_matched_edges(self) -> None:
        """The core mechanism: a track broken by one missing detection must be
        bridged with a synthetic intermediate node so the scorer's own
        consecutive-frame-only edge filter (metrics.py's `_target_t - _source_t == 1`
        check) accepts the recovered edges as unit-time-step, not skipped.
        """
        pred, pred_ids = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 20.0},
            },
            [],
        )
        # GT: a continuous three-node lineage A-B-C; predicted A and C sit exactly
        # on the GT track (so they match), and predicted has no node near GT's B
        # until gap-closing inserts one at the interpolated midpoint (10,0,0).
        gt, gt_ids = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 10.0},
                "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 20.0},
            },
            [("A", "B"), ("B", "C")],
        )

        before = metrics.evaluate(pred, gt, scale=(1.0, 1.0, 1.0), max_distance=7.0)
        assert before.edge_tp == 0
        assert before.edge_fn == 2

        closed = gcl.close_gaps_in_graph(
            pred, scale=(1.0, 1.0, 1.0), max_gap_frames=2, max_speed_um_per_frame=15.0
        )
        after = metrics.evaluate(closed, gt, scale=(1.0, 1.0, 1.0), max_distance=7.0)

        assert after.edge_tp == 2
        assert after.edge_fp == 0
        assert after.edge_fn == 0

    def test_does_not_bridge_a_gap_that_exceeds_the_speed_bound(self) -> None:
        """Two endpoints far enough apart that bridging them would require an
        implausible speed must be left alone (declared a birth/death instead),
        not spliced -- ties directly to the u-track/Jaqaman LAP's birth/death
        alternative that keeps the assignment from firing on noise.
        """
        pred, pred_ids = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 500.0},
            },
            [],
        )
        gt = _build_graph({}, [])[0]

        closed = gcl.close_gaps_in_graph(
            pred, scale=(1.0, 1.0, 1.0), max_gap_frames=2, max_speed_um_per_frame=15.0
        )
        assert closed.num_edges() == 0
        assert closed.num_nodes() == 2

    def test_global_assignment_picks_the_correct_pairing_over_greedy_nearest(self) -> None:
        """Two tails and two heads where a greedy nearest-first bridge would
        mis-pair (double-booking the closer head to both tails); the global LAP
        must produce the one-to-one assignment that minimizes total cost instead.
        """
        pred, ids = _build_graph(
            {
                "A1": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "A2": {"t": 0, "z": 0.0, "y": 0.0, "x": 100.0},
                "C1": {"t": 2, "z": 0.0, "y": 0.0, "x": 4.0},
                "C2": {"t": 2, "z": 0.0, "y": 0.0, "x": 104.0},
            },
            [],
        )
        gt, gt_ids = _build_graph(
            {
                "A1": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B1": {"t": 1, "z": 0.0, "y": 0.0, "x": 2.0},
                "C1": {"t": 2, "z": 0.0, "y": 0.0, "x": 4.0},
                "A2": {"t": 0, "z": 0.0, "y": 0.0, "x": 100.0},
                "B2": {"t": 1, "z": 0.0, "y": 0.0, "x": 102.0},
                "C2": {"t": 2, "z": 0.0, "y": 0.0, "x": 104.0},
            },
            [("A1", "B1"), ("B1", "C1"), ("A2", "B2"), ("B2", "C2")],
        )

        closed = gcl.close_gaps_in_graph(
            pred, scale=(1.0, 1.0, 1.0), max_gap_frames=2, max_speed_um_per_frame=15.0
        )
        after = metrics.evaluate(closed, gt, scale=(1.0, 1.0, 1.0), max_distance=7.0)

        assert after.edge_tp == 4
        assert after.edge_fp == 0


class TestSweepPopulation:
    def test_reports_pooled_edge_counts_before_and_after(self) -> None:
        pred, pred_ids = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 20.0},
            },
            [],
        )
        gt, gt_ids = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 10.0},
                "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 20.0},
            },
            [("A", "B"), ("B", "C")],
        )

        result = gcl.sweep_population(
            [("v1", pred, gt)],
            scale=(1.0, 1.0, 1.0),
            max_gap_frames=2,
            max_speed_um_per_frame=15.0,
        )

        assert result["before"]["edge_tp"] == 0
        assert result["before"]["edge_fn"] == 2
        assert result["after"]["edge_tp"] == 2
        assert result["after"]["edge_fp"] == 0
