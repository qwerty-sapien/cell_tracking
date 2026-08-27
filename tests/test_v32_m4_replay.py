"""Synthetic breaker tests for the standalone fixed-default V32 M4 replay."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, asdict
import inspect
import math

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import celltrack_alpha.v32_m4_replay as m4
from celltrack_alpha import v32_preilp_cache as replay_cache
from celltrack_alpha.v32_m4_replay import (
    MAX_ACCEL_UM,
    MAX_BONUS,
    NEXT_STEP_GATE_UM,
    VOXEL_SCALE_UM,
    MotionRelinkConfig,
    MotionRelinkError,
    forward_acceleration_bonus,
    forward_acceleration_lookahead,
    replay_cached_arrays,
    replay_motion_relink,
)


def _node(node_id: int, t: int, x_um: float, *, y_um: float = 0.0, z_um: float = 0.0) -> dict:
    return {
        "node_id": node_id,
        "t": t,
        "z_vox": z_um / VOXEL_SCALE_UM[0],
        "y_vox": y_um / VOXEL_SCALE_UM[1],
        "x_vox": x_um / VOXEL_SCALE_UM[2],
    }


def _edge(source_id: int, target_id: int, probability: float | None = None, **extra: object) -> dict:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "edge_prob": probability,
        **extra,
    }


def _pairs(result: m4.MotionRelinkResult) -> list[tuple[int, int]]:
    return [(int(edge["source_id"]), int(edge["target_id"])) for edge in result.edges]


def _accepted_v32_oracle(nodes: list[dict], post_ilp_edges: list[dict]) -> list[dict]:
    """Test-local port of accepted V32 notebook ``motion_relink_edges``.

    This deliberately does not call implementation helpers.  Constants and
    control flow mirror the immutable notebook: 6/10 um passes, 0.5 velocity,
    and ``motion + .05 * raw - 1.0 * post-ILP probability``.
    """

    positions = {
        int(node["node_id"]): np.asarray(
            [
                float(node.get("z_vox", node.get("z"))) * 1.625,
                float(node.get("y_vox", node.get("y"))) * 0.40625,
                float(node.get("x_vox", node.get("x"))) * 0.40625,
            ]
        )
        for node in nodes
    }
    times = {int(node["node_id"]): int(node["t"]) for node in nodes}
    ids_by_t: dict[int, list[int]] = {}
    for node_id, t in times.items():
        ids_by_t.setdefault(t, []).append(node_id)
    for frame_ids in ids_by_t.values():
        frame_ids.sort()
    filtered_edges: list[dict] = []
    for edge in post_ilp_edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        distance_um = float(np.linalg.norm(positions[target_id] - positions[source_id]))
        if distance_um <= 14.0:
            retained = dict(edge)
            retained["distance_um"] = distance_um
            filtered_edges.append(retained)
    probabilities = {
        (int(edge["source_id"]), int(edge["target_id"])): float(edge["edge_prob"])
        for edge in filtered_edges
        if edge.get("edge_prob") is not None
    }
    predecessor: dict[int, np.ndarray] = {}
    selected: list[dict] = []

    def assign(source_ids: list[int], target_ids: list[int], gate_um: float) -> list[dict]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        shape = (len(source_ids), len(target_ids))
        costs = np.full(shape, big)
        raw_distances = np.full(shape, np.inf)
        motion_distances = np.full(shape, np.inf)
        probs = np.zeros(shape)
        for row, source_id in enumerate(source_ids):
            source = positions[source_id]
            previous = predecessor.get(source_id)
            predicted = source if previous is None else source + 0.5 * (source - previous)
            for column, target_id in enumerate(target_ids):
                target = positions[target_id]
                raw = float(np.linalg.norm(target - source))
                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target - predicted))
                probability = probabilities.get((source_id, target_id), 0.0)
                raw_distances[row, column] = raw
                motion_distances[row, column] = motion
                probs[row, column] = probability
                costs[row, column] = motion + 0.05 * raw - probability
        rows, columns = linear_sum_assignment(costs)
        return [
            {
                "source_id": source_ids[int(row)],
                "target_id": target_ids[int(column)],
                "edge_prob": float(probs[row, column]),
                "distance_um": float(raw_distances[row, column]),
                "motion_distance_um": float(motion_distances[row, column]),
                "motion_pass": "",
                "motion_cost": float(costs[row, column]),
            }
            for row, column in zip(rows, columns)
            if costs[row, column] < big
        ]

    for t in sorted(ids_by_t):
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches: list[dict] = []
        for pass_name, gate_um in (("tight", 6.0), ("relaxed", 10.0)):
            matches = assign(
                [value for value in source_ids if value in unmatched_sources],
                [value for value in target_ids if value in unmatched_targets],
                gate_um,
            )
            for match in matches:
                source_id = int(match["source_id"])
                target_id = int(match["target_id"])
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                match["motion_pass"] = pass_name
                frame_matches.append(match)
        for match in frame_matches:
            selected.append(match)
            predecessor[int(match["target_id"])] = positions[int(match["source_id"])]
    if selected:
        return sorted(
            selected,
            key=lambda edge: (
                times[int(edge["source_id"])], edge["source_id"], edge["target_id"]
            ),
        )
    return filtered_edges


def test_fixed_defaults_and_voxel_scale_contract() -> None:
    config = MotionRelinkConfig()
    assert VOXEL_SCALE_UM == (1.625, 0.40625, 0.40625)
    assert (MAX_ACCEL_UM, MAX_BONUS, NEXT_STEP_GATE_UM) == (4.0, 0.20, 10.0)
    assert config == MotionRelinkConfig(
        tight_um=6.0,
        relaxed_um=10.0,
        velocity_weight=0.5,
        learned_bonus=1.0,
        raw_distance_weight=0.05,
        output_edge_max_um=14.0,
        max_frame_nodes=2600,
        max_accel_um=4.0,
        max_bonus=0.20,
        next_step_gate_um=10.0,
    )
    assert asdict(config) == {
        "tight_um": 6.0,
        "relaxed_um": 10.0,
        "velocity_weight": 0.5,
        "learned_bonus": 1.0,
        "raw_distance_weight": 0.05,
        "output_edge_max_um": 14.0,
        "max_frame_nodes": 2600,
        "max_accel_um": 4.0,
        "max_bonus": 0.20,
        "next_step_gate_um": 10.0,
    }
    with pytest.raises(FrozenInstanceError):
        config.max_bonus = 1.0  # type: ignore[misc]

    result = replay_motion_relink(
        [
            {"node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            {"node_id": 2, "t": 1, "z": 1.0, "y": 1.0, "x": 1.0},
        ],
        [],
    )
    assert len(result.edges) == 1
    assert result.edges[0]["distance_um"] == pytest.approx(
        math.sqrt(1.625**2 + 0.40625**2 + 0.40625**2)
    )


def test_constant_velocity_has_zero_residual_and_full_bonus() -> None:
    positions = {
        1: np.asarray([0.0, 0.0, 2.0]),
        2: np.asarray([0.0, 0.0, 9.0]),
    }
    residual, count = forward_acceleration_lookahead(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        1,
        {2: [2, 1]},
        positions,
    )
    assert residual == pytest.approx(0.0)
    assert count == 2
    assert forward_acceleration_bonus(residual) == pytest.approx(0.20)


def test_no_or_gated_continuation_has_no_residual_and_no_bonus() -> None:
    assert forward_acceleration_lookahead(
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1, {}, {}
    ) == (None, 0)
    assert forward_acceleration_lookahead(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        1,
        {2: [9]},
        {9: [0.0, 0.0, 11.0000001]},
    ) == (None, 0)
    assert forward_acceleration_bonus(None) == 0.0


def test_residual_at_or_above_four_has_zero_bonus() -> None:
    residual, count = forward_acceleration_lookahead(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        1,
        {2: [3]},
        {3: [0.0, 0.0, 6.0]},
    )
    assert residual == pytest.approx(4.0)
    assert count == 1
    assert forward_acceleration_bonus(residual) == 0.0
    assert forward_acceleration_bonus(100.0) == 0.0
    assert forward_acceleration_bonus(2.0) == pytest.approx(0.1)


def test_replay_cost_is_current_v32_cost_minus_fixed_acceleration_bonus() -> None:
    result = replay_motion_relink(
        [_node(1, 0, 0.0), _node(2, 1, 1.0), _node(3, 2, 2.0)],
        [_edge(1, 2, 0.5), _edge(2, 3, 0.0)],
    )
    first = result.edges[0]
    assert (first["source_id"], first["target_id"]) == (1, 2)
    assert first["motion_distance_um"] == pytest.approx(1.0)
    assert first["distance_um"] == pytest.approx(1.0)
    assert first["edge_prob"] == pytest.approx(0.5)
    assert first["acceleration_residual_um"] == pytest.approx(0.0)
    assert first["acceleration_bonus"] == pytest.approx(0.2)
    assert first["motion_cost"] == pytest.approx(1.0 + 0.05 - 0.5 - 0.2)


@pytest.mark.parametrize(
    "nodes, post_ilp_edges",
    [
        (
            [_node(1, 0, 0.0), _node(2, 1, 1.0), _node(3, 2, 2.0)],
            [_edge(1, 2, 0.8), _edge(2, 3, 0.6)],
        ),
        (
            [_node(1, 0, 0.0), _node(2, 0, 0.0), _node(3, 1, 1.0), _node(4, 1, 1.0)],
            [_edge(1, 4, 1.0), _edge(2, 3, 1.0)],
        ),
        (
            [_node(1, 0, 0.0), _node(2, 0, 20.0), _node(3, 1, 5.0), _node(4, 1, 27.0)],
            [_edge(1, 3), _edge(2, 4)],
        ),
        (
            [_node(1, 0, 0.0), _node(2, 1, 2.0), _node(3, 2, 3.0)],
            [_edge(1, 2), _edge(2, 3)],
        ),
    ],
)
def test_apply_m4_false_differential_matches_accepted_v32_oracle(
    nodes: list[dict], post_ilp_edges: list[dict]
) -> None:
    expected = _accepted_v32_oracle(nodes, post_ilp_edges)
    observed = replay_motion_relink(nodes, post_ilp_edges, apply_m4=False)
    assert _pairs(observed) == [
        (int(edge["source_id"]), int(edge["target_id"])) for edge in expected
    ]
    assert len(observed.edges) == len(expected)
    for actual, oracle in zip(observed.edges, expected):
        assert actual["edge_prob"] == pytest.approx(oracle["edge_prob"])
        assert actual["distance_um"] == pytest.approx(oracle["distance_um"])
        assert actual["motion_distance_um"] == pytest.approx(oracle["motion_distance_um"])
        assert actual["motion_pass"] == oracle["motion_pass"]
        assert actual["motion_cost"] == pytest.approx(oracle["motion_cost"])
        assert actual["acceleration_bonus"] == 0.0
    assert observed.stats["motion_relink_accel_bonus_edges"] == 0
    assert observed.stats["motion_relink_accel_bonus_sum"] == 0.0


def test_learned_probability_is_masked_to_post_ilp_pairs() -> None:
    nodes = [
        _node(1, 0, 0.0),
        _node(2, 0, 10.0),
        _node(3, 1, 1.0),
        _node(4, 1, 9.0),
    ]
    result = replay_motion_relink(nodes, [_edge(1, 4, 1.0)])
    assert _pairs(result) == [(1, 3), (2, 4)]
    assert [edge["edge_prob"] for edge in result.edges] == [0.0, 0.0]
    assert result.stats["motion_relink_replaced_raw_edges"] == 1

    tied = replay_motion_relink(
        [_node(1, 0, 0.0), _node(2, 0, 0.0), _node(3, 1, 1.0), _node(4, 1, 1.0)],
        [_edge(1, 4, 1.0), _edge(2, 3, 1.0)],
    )
    assert _pairs(tied) == [(1, 4), (2, 3)]
    assert [edge["edge_prob"] for edge in tied.edges] == [1.0, 1.0]


def test_tight_pass_precedes_relaxed_pass() -> None:
    result = replay_motion_relink(
        [
            _node(1, 0, 0.0),
            _node(2, 0, 20.0),
            _node(3, 1, 5.0),
            _node(4, 1, 27.0),
        ],
        [_edge(1, 3), _edge(2, 4)],
    )
    assert _pairs(result) == [(1, 3), (2, 4)]
    assert [edge["motion_pass"] for edge in result.edges] == ["tight", "relaxed"]
    assert result.stats["motion_relink_tight_edges"] == 1
    assert result.stats["motion_relink_relaxed_edges"] == 1


def test_predecessor_position_propagates_into_next_frame_prediction() -> None:
    result = replay_motion_relink(
        [_node(1, 0, 0.0), _node(2, 1, 2.0), _node(3, 2, 3.0)],
        [_edge(1, 2), _edge(2, 3)],
    )
    assert _pairs(result) == [(1, 2), (2, 3)]
    assert result.edges[0]["motion_distance_um"] == pytest.approx(2.0)
    assert result.edges[1]["motion_distance_um"] == pytest.approx(0.0)


def test_equal_cost_assignment_and_output_are_id_ordered() -> None:
    forward = replay_motion_relink(
        [_node(20, 0, 0.0), _node(10, 0, 0.0), _node(40, 1, 1.0), _node(30, 1, 1.0)],
        [_edge(20, 30), _edge(10, 40)],
    )
    reverse = replay_motion_relink(
        reversed([_node(20, 0, 0.0), _node(10, 0, 0.0), _node(40, 1, 1.0), _node(30, 1, 1.0)]),
        reversed([_edge(20, 30), _edge(10, 40)]),
    )
    assert _pairs(forward) == [(10, 30), (20, 40)]
    assert forward == reverse


def test_more_than_2600_nodes_preserves_raw_edge_fallback_order() -> None:
    nodes = [_node(node_id, 0, float(node_id % 5)) for node_id in range(1, 2602)]
    nodes.append(_node(9000, 1, 0.0))
    result = replay_motion_relink(
        nodes,
        [_edge(2, 9000, 0.4, provenance="post-ilp"), _edge(1, 9000, None)],
    )
    assert _pairs(result) == [(2, 9000), (1, 9000)]
    assert result.edges[0]["provenance"] == "post-ilp"
    assert result.stats["motion_relink_skipped_large_frame"] == 1
    assert result.stats["motion_relink_fallback_raw"] == 1
    assert result.stats["motion_relink_edges"] == 0


def test_no_feasible_relink_falls_back_to_post_ilp_edges() -> None:
    result = replay_motion_relink(
        [_node(1, 0, 0.0), _node(2, 1, 12.0)],
        [_edge(1, 2, 0.8, marker="raw")],
    )
    assert _pairs(result) == [(1, 2)]
    assert result.edges[0]["marker"] == "raw"
    assert result.stats["motion_relink_fallback_raw"] == 1


def test_v32_prefilter_drops_post_ilp_edges_over_fourteen_um_before_fallback() -> None:
    nodes = [_node(1, 0, 0.0), _node(2, 1, 15.0)]
    edge = _edge(1, 2, 0.8, marker="must-drop")
    assert _accepted_v32_oracle(nodes, [edge]) == []
    result = replay_motion_relink(nodes, [edge], apply_m4=False)
    assert result.edges == ()
    assert result.stats["motion_relink_prefilter_dropped_long_edges"] == 1
    assert result.stats["motion_relink_fallback_raw"] == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"tight_um": 5.0}, "tight_um.*fixed"),
        ({"relaxed_um": 9.0}, "relaxed_um.*fixed"),
        ({"velocity_weight": 0.4}, "velocity_weight.*fixed"),
        ({"learned_bonus": 0.9}, "learned_bonus.*fixed"),
        ({"raw_distance_weight": 0.04}, "raw_distance_weight.*fixed"),
        ({"output_edge_max_um": 13.0}, "output_edge_max_um.*fixed"),
        ({"learned_bonus": float("nan")}, "learned_bonus"),
        ({"max_frame_nodes": True}, "max_frame_nodes"),
        ({"max_frame_nodes": 2599}, "max_frame_nodes.*fixed"),
        ({"max_accel_um": 4.01}, "fixed"),
        ({"max_bonus": 0.21}, "fixed"),
        ({"next_step_gate_um": 9.0}, "fixed"),
    ],
)
def test_config_rejects_malformed_or_tuned_values(kwargs: dict, message: str) -> None:
    with pytest.raises(MotionRelinkError, match=message):
        MotionRelinkConfig(**kwargs)


@pytest.mark.parametrize(
    "nodes, edges, message",
    [
        ([{"node_id": 1, "t": 0, "z": 0.0, "y": 0.0}], [], "incomplete"),
        ([_node(1, 0, 0.0), _node(1, 1, 1.0)], [], "duplicate node_id"),
        ([{**_node(1, 0, 0.0), "x_vox": float("inf")}], [], "finite"),
        ([{**_node(1, 0, 0.0), "t": 0.5}], [], "integer"),
        ([_node(1, 0, 0.0)], [{"source_id": 1}], "missing"),
        ([_node(1, 0, 0.0), _node(2, 1, 1.0)], [_edge(1, 3)], "unknown"),
        ([_node(1, 0, 0.0), _node(2, 2, 1.0)], [_edge(1, 2)], "consecutive"),
        (
            [_node(1, 0, 0.0), _node(2, 1, 1.0)],
            [_edge(1, 2), _edge(1, 2)],
            "duplicate post-ILP",
        ),
        ([_node(1, 0, 0.0), _node(2, 1, 1.0)], [_edge(1, 2, float("nan"))], "finite"),
        ([_node(1, 0, 0.0), _node(2, 1, 1.0)], [_edge(1, 2, 1.1)], r"\[0, 1\]"),
    ],
)
def test_replay_rejects_malformed_and_nonfinite_records(
    nodes: list[dict], edges: list[dict], message: str
) -> None:
    with pytest.raises(MotionRelinkError, match=message):
        replay_motion_relink(nodes, edges)


def test_helpers_fail_closed_on_malformed_and_nonfinite_values() -> None:
    invalid_calls = (
        lambda: forward_acceleration_lookahead([0.0, 0.0], [0.0, 0.0, 1.0], 1, {}, {}),
        lambda: forward_acceleration_lookahead([0.0, 0.0, 0.0], [0.0, np.nan, 1.0], 1, {}, {}),
        lambda: forward_acceleration_lookahead([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], True, {}, {}),
        lambda: forward_acceleration_lookahead(
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1, {2: [9]}, {}
        ),
        lambda: forward_acceleration_lookahead(
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1, {2: [9, 9]}, {9: [0.0, 0.0, 2.0]}
        ),
        lambda: forward_acceleration_lookahead(
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1, {}, {}, float("inf")
        ),
        lambda: forward_acceleration_lookahead(
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1, {}, {}, 9.0
        ),
        lambda: forward_acceleration_bonus(-0.1),
        lambda: forward_acceleration_bonus(float("nan")),
        lambda: forward_acceleration_bonus(0.0, max_accel_um=5.0),
        lambda: forward_acceleration_bonus(0.0, max_bonus=0.1),
    )
    for call in invalid_calls:
        with pytest.raises(MotionRelinkError):
            call()


@pytest.mark.parametrize("value", [0, 1, None, "false", np.bool_(False)])
def test_apply_m4_is_a_strict_boolean_infrastructure_control(value: object) -> None:
    with pytest.raises(MotionRelinkError, match="apply_m4 must be a boolean"):
        replay_motion_relink([], [], apply_m4=value)  # type: ignore[arg-type]


def test_cache_array_adapter_consumes_exact_cache_builder_arrays() -> None:
    nodes = replay_cache.make_nodes(
        [
            {"node_id": 10, "t": 0, "z": 1.0, "y": 2.0, "x": 3.0},
            {"node_id": 20, "t": 1, "z": 1.0, "y": 3.0, "x": 3.0},
            {"node_id": 30, "t": 2, "z": 1.0, "y": 4.0, "x": 3.0},
        ]
    )
    edges = replay_cache.make_post_edges(
        [
            {"source_id": 10, "target_id": 20, "edge_prob": 0.9},
            {"source_id": 20, "target_id": 30, "edge_prob": 0.7},
        ]
    )
    cached = replay_cached_arrays(nodes, edges, apply_m4=False)
    mapped = replay_motion_relink(
        [
            {"node_id": 10, "t": 0, "z": 1.0, "y": 2.0, "x": 3.0},
            {"node_id": 20, "t": 1, "z": 1.0, "y": 3.0, "x": 3.0},
            {"node_id": 30, "t": 2, "z": 1.0, "y": 4.0, "x": 3.0},
        ],
        [
            {"source_id": 10, "target_id": 20, "edge_prob": 0.9},
            {"source_id": 20, "target_id": 30, "edge_prob": 0.7},
        ],
        apply_m4=False,
    )
    assert cached == mapped


def test_cache_array_adapter_rejects_noncanonical_structured_dtypes() -> None:
    nodes = replay_cache.make_nodes(
        [{"node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0}]
    )
    edges = replay_cache.make_post_edges([])
    wrong_nodes = nodes[["node_id", "t", "z_vox", "y_vox", "x_vox"]]
    wrong_edges = np.empty(0, dtype=[("source_id", "<i8"), ("target_id", "<i8")])
    with pytest.raises(MotionRelinkError, match="wrong cache dtype"):
        replay_cached_arrays(wrong_nodes, edges, apply_m4=True)
    with pytest.raises(MotionRelinkError, match="wrong cache dtype"):
        replay_cached_arrays(nodes, wrong_edges, apply_m4=True)


def test_empty_inputs_are_stable_and_module_has_no_campaign_or_data_imports() -> None:
    result = replay_motion_relink(iter(()), iter(()))
    assert result.edges == ()
    assert result.stats["motion_relink_input_nodes"] == 0
    assert result.stats["motion_relink_fallback_raw"] == 1

    tree = ast.parse(inspect.getsource(m4))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imports <= {"__future__", "dataclasses", "math", "typing", "numpy", "scipy"}
