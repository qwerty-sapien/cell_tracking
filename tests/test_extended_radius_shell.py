from __future__ import annotations

from scripts import prepare_extended_radius_shell as shell


def test_topology_code_preserves_exact_conflicts() -> None:
    selected_out = {1: 10, 2: 20}
    selected_in = {10: 1, 20: 2}
    assert shell.topology_code(3, 30, selected_out, selected_in) == (0, ())
    assert shell.topology_code(1, 30, selected_out, selected_in) == (
        1,
        ((1, 10),),
    )
    assert shell.topology_code(3, 20, selected_out, selected_in) == (
        2,
        ((2, 20),),
    )
    assert shell.topology_code(1, 20, selected_out, selected_in) == (
        3,
        ((1, 10), (2, 20)),
    )


def test_label_pair_distinguishes_truth_negative_unknown_and_division() -> None:
    matching = {1: 101, 2: 102, 3: None}
    gt_edges = {(101, 102), (101, 103)}
    outdegree = {101: 1, 104: 1}
    indegree = {102: 1, 103: 1}
    assert shell.label_pair(
        (1, 2), matching, gt_edges, outdegree, indegree
    ) == 1
    assert shell.label_pair(
        (2, 1), matching, gt_edges, outdegree, indegree
    ) == 0
    assert shell.label_pair(
        (1, 3), matching, gt_edges, outdegree, indegree
    ) == -1
    assert shell.label_pair(
        (1, 2), matching, gt_edges, {101: 2}, indegree
    ) == -2


def test_unknown_sampling_is_deterministic_and_bounded() -> None:
    rows = [
        {
            "pair": [index, index + 1],
            "features": [
                11.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                float(index),
                0.0,
                20.0,
                0.5,
                0.0,
                -1.0,
                20.0,
                9.0,
            ],
        }
        for index in range(20)
    ]
    first = shell._sample_unknown(
        rows, identity="train:tiny", hard_limit=3, uniform_limit=4
    )
    second = shell._sample_unknown(
        list(reversed(rows)),
        identity="train:tiny",
        hard_limit=3,
        uniform_limit=4,
    )
    assert [row["pair"] for row in first] == [row["pair"] for row in second]
    assert len(first) <= 7


def test_bounded_push_keeps_largest_inverted_keys() -> None:
    heap: list[tuple[int, str]] = []
    for value in (-8, -2, -5, -1, -9):
        shell._bounded_push(heap, (value, str(value)), limit=3)
    assert sorted(value for value, _label in heap) == [-5, -2, -1]
