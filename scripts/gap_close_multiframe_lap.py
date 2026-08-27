#!/usr/bin/env python3
"""Global multi-frame gap-closing LAP -- CPU-only, banked-graph-only.

Implements R1's Candidate 3 (dag_prompts/2026-08-07-score-ceiling-research/runs/
R1_EDGE_ALPHA_CEILING.md, Phase 4/5): a Jaqaman/u-track-style global linear
assignment over ALL broken track endpoints in a video simultaneously (not a
greedy nearest-first bridge), with an explicit birth/death "no-link"
alternative cost, operating purely on the already-banked predicted graph's
node timestamps and positions -- no re-inference, no GPU.

Deliberately does not implement u-track's merge/split costs (out of scope for
this first, cheapest falsification pass per the report's own staged
discipline); it is a genuine global one-to-one assignment, not a
simplification into independent greedy pairwise choices.

The frozen scorer (external/kaggle-cell-tracking-competition/src/
tracking_cellmot/metrics.py) drops any edge whose endpoints are not exactly
one frame apart (`_target_t - _source_t == 1`), so a bridge across a gap of
missing frames must be realized as a chain of synthetic intermediate nodes at
each skipped frame, linked by unit-time-step edges -- a single long edge
across the gap would be silently discarded by the scorer, not counted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import tracksdata as td
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts", ROOT / "external/kaggle-cell-tracking-competition/src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from tracking_cellmot import metrics  # noqa: E402

_NODE_ID = td.DEFAULT_ATTR_KEYS.NODE_ID
_T = td.DEFAULT_ATTR_KEYS.T
_POS_KEYS = ("z", "y", "x")


def _node_table(graph: td.graph.BaseGraph) -> pl.DataFrame:
    return graph.node_attrs(attr_keys=[_NODE_ID, _T, *_POS_KEYS])


def find_broken_endpoints(graph: td.graph.BaseGraph) -> tuple[list[int], list[int]]:
    """Return (tails, heads): track endpoints broken mid-video, not at the
    video's actual first/last frame (a track legitimately ending at the last
    frame is not a gap, it is a real end)."""
    node_ids = graph.node_ids()
    if len(node_ids) == 0:
        return [], []
    table = _node_table(graph)
    t_min = int(table[_T].min())
    t_max = int(table[_T].max())
    out_deg = dict(zip(node_ids, graph.out_degree(node_ids)))
    in_deg = dict(zip(node_ids, graph.in_degree(node_ids)))
    t_by_node = dict(zip(table[_NODE_ID].to_list(), table[_T].to_list()))

    tails = sorted(
        nid for nid in node_ids if out_deg[nid] == 0 and t_by_node[nid] < t_max
    )
    heads = sorted(
        nid for nid in node_ids if in_deg[nid] == 0 and t_by_node[nid] > t_min
    )
    return tails, heads


def _positions(table: pl.DataFrame) -> dict[int, tuple[int, np.ndarray]]:
    out: dict[int, tuple[int, np.ndarray]] = {}
    for row in table.iter_rows(named=True):
        pos = np.array([row[k] for k in _POS_KEYS], dtype=float)
        out[row[_NODE_ID]] = (int(row[_T]), pos)
    return out


def estimate_max_speed(graph: td.graph.BaseGraph, *, scale: Sequence[float]) -> float:
    """95th percentile physical displacement of the graph's own existing
    consecutive-frame edges -- a data-derived speed reference rather than an
    arbitrary constant, so the gap-closing radius is not hand-tuned to win."""
    scale_arr = np.array(scale, dtype=float)
    table = _node_table(graph)
    pos = _positions(table)
    edge_table = graph.edge_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
    )
    displacements: list[float] = []
    for row in edge_table.iter_rows(named=True):
        src = row[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]
        tgt = row[td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
        t_src, p_src = pos[src]
        t_tgt, p_tgt = pos[tgt]
        if t_tgt - t_src != 1:
            continue
        displacements.append(float(np.linalg.norm((p_tgt - p_src) * scale_arr)))
    if not displacements:
        return 0.0
    return float(np.percentile(displacements, 95))


def build_gap_candidates(
    graph: td.graph.BaseGraph,
    tails: Sequence[int],
    heads: Sequence[int],
    *,
    scale: Sequence[float],
    max_gap_frames: int,
    max_speed_um_per_frame: float,
) -> dict[tuple[int, int], float]:
    """Cost of bridging each feasible (tail, head) pair: physical distance
    divided by gap length (so a longer, proportionally-consistent bridge is
    not penalized relative to a short one), restricted to gap lengths in
    [1, max_gap_frames - 1] missing frames and to a speed-plausible radius."""
    scale_arr = np.array(scale, dtype=float)
    table = _node_table(graph)
    pos = _positions(table)
    costs: dict[tuple[int, int], float] = {}
    for tail in tails:
        t_tail, p_tail = pos[tail]
        for head in heads:
            t_head, p_head = pos[head]
            gap = t_head - t_tail  # number of frames spanned, including both ends
            missing = gap - 1
            if missing < 1 or missing > max_gap_frames - 1:
                continue
            distance = float(np.linalg.norm((p_head - p_tail) * scale_arr))
            if distance > max_speed_um_per_frame * gap:
                continue
            costs[(tail, head)] = distance / gap
    return costs


def solve_gap_assignment(
    tails: Sequence[int],
    heads: Sequence[int],
    costs: dict[tuple[int, int], float],
    *,
    no_link_cost: float,
) -> list[tuple[int, int]]:
    """Global one-to-one assignment via a single Hungarian solve over the
    augmented (tails+dustbin) x (heads+dustbin) matrix -- the u-track/Jaqaman
    LAP structure: every tail may instead pay `no_link_cost` to stay a real
    track end (death), every head may instead pay it to stay a real track
    start (birth), so a bridge only fires when it beats declaring a
    birth/death, not merely when it is the nearest option."""
    n_tails = len(tails)
    n_heads = len(heads)
    if n_tails == 0 or n_heads == 0:
        return []
    size = n_tails + n_heads
    cost_matrix = np.full((size, size), no_link_cost, dtype=float)
    for i, tail in enumerate(tails):
        for j, head in enumerate(heads):
            if (tail, head) in costs:
                cost_matrix[i, j] = costs[(tail, head)]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    accepted: list[tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if r < n_tails and c < n_heads and (tails[r], heads[c]) in costs:
            accepted.append((tails[r], heads[c]))
    return accepted


def bridge_gap(
    graph: td.graph.BaseGraph,
    tail: int,
    head: int,
) -> None:
    """Insert one synthetic node per skipped frame, linearly interpolated
    between tail and head, chained by unit-time-step edges -- the scorer's
    consecutive-frame-only edge filter requires this instead of one long
    edge across the gap."""
    table = _node_table(graph)
    pos = _positions(table)
    t_tail, p_tail = pos[tail]
    t_head, p_head = pos[head]
    gap = t_head - t_tail
    prev_id = tail
    for step in range(1, gap):
        frac = step / gap
        interp = p_tail + frac * (p_head - p_tail)
        attrs = {"t": t_tail + step}
        attrs.update(dict(zip(_POS_KEYS, (float(v) for v in interp))))
        new_id = graph.add_node(attrs=attrs)
        graph.add_edge(prev_id, new_id, {})
        prev_id = new_id
    graph.add_edge(prev_id, head, {})


def close_gaps_in_graph(
    graph: td.graph.BaseGraph,
    *,
    scale: Sequence[float],
    max_gap_frames: int,
    max_speed_um_per_frame: float,
) -> td.graph.BaseGraph:
    closed = graph.copy()
    tails, heads = find_broken_endpoints(closed)
    costs = build_gap_candidates(
        closed,
        tails,
        heads,
        scale=scale,
        max_gap_frames=max_gap_frames,
        max_speed_um_per_frame=max_speed_um_per_frame,
    )
    if not costs:
        return closed
    no_link_cost = max(costs.values()) * 10.0 + 1.0
    accepted = solve_gap_assignment(tails, heads, costs, no_link_cost=no_link_cost)
    for tail, head in accepted:
        bridge_gap(closed, tail, head)
    return closed


def sweep_population(
    graphs: Sequence[tuple[str, td.graph.BaseGraph, td.graph.BaseGraph]],
    *,
    scale: Sequence[float],
    max_gap_frames: int,
    max_speed_um_per_frame: float | None = None,
    max_distance: float = 7.0,
) -> dict[str, Any]:
    before_totals = {"edge_tp": 0, "edge_fp": 0, "edge_fn": 0}
    after_totals = {"edge_tp": 0, "edge_fp": 0, "edge_fn": 0}
    per_video: list[dict[str, Any]] = []

    for identity, pred_graph, gt_graph in graphs:
        speed = (
            max_speed_um_per_frame
            if max_speed_um_per_frame is not None
            else estimate_max_speed(pred_graph, scale=scale)
        )
        before = metrics.evaluate(
            pred_graph.copy(), gt_graph, scale=scale, max_distance=max_distance
        )
        closed = close_gaps_in_graph(
            pred_graph,
            scale=scale,
            max_gap_frames=max_gap_frames,
            max_speed_um_per_frame=speed,
        )
        after = metrics.evaluate(closed, gt_graph, scale=scale, max_distance=max_distance)

        for key in before_totals:
            before_totals[key] += getattr(before, key)
            after_totals[key] += getattr(after, key)
        per_video.append(
            {
                "identity": identity,
                "max_speed_um_per_frame": speed,
                "before": before._asdict(),
                "after": after._asdict(),
            }
        )

    def _jaccard(counts: dict[str, int]) -> float:
        denom = counts["edge_tp"] + counts["edge_fp"] + counts["edge_fn"]
        return counts["edge_tp"] / denom if denom > 0 else float("nan")

    return {
        "schema": "celltrack-alpha.gap-close-multiframe-lap.sweep.v1",
        "max_gap_frames": max_gap_frames,
        "before": {**before_totals, "edge_jaccard": _jaccard(before_totals)},
        "after": {**after_totals, "edge_jaccard": _jaccard(after_totals)},
        "per_video": per_video,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--identities", required=True, type=Path, help="JSON list of identities")
    parser.add_argument("--max-gap-frames", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    from instrument_division_forks import load_population_from_submission, load_scale

    identities = json.loads(args.identities.read_text(encoding="utf-8"))
    if not isinstance(identities, list) or not identities:
        raise SystemExit("--identities must be a non-empty JSON list")
    graphs = load_population_from_submission(args.submission, args.zip, identities)
    scale = load_scale(args.zip, identities[0].split(":", 1)[1])
    result = sweep_population(graphs, scale=scale, max_gap_frames=args.max_gap_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "per_video"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
