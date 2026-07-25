from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import audit_selected_graph_appearance as appearance
from scripts import audit_selected_graph_appearance_support as support
from scripts import evaluate_appearance_direct_policy_fit as policy_fit
from scripts import evaluate_selected_graph_direct_swap as direct
from celltrack_alpha import appearance_swap
from celltrack_alpha import bh_graph


def test_extract_patch_and_unit_vector_are_deterministic_at_boundary() -> None:
    frame = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
    first = appearance.extract_patch(frame, (0.0, 0.0, 0.0))
    second = appearance.extract_patch(frame, (0.0, 0.0, 0.0))
    assert first.shape == appearance.PATCH_SHAPE
    assert np.array_equal(first, second)
    vector = appearance.unit_vector(first)
    assert vector.shape == (np.prod(appearance.PATCH_SHAPE),)
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert abs(float(vector.mean())) < 1.0e-6


def test_ncc_orders_identical_above_reversed_patch() -> None:
    source = appearance.unit_vector(np.arange(16, dtype=np.float32))
    same = appearance.unit_vector(np.arange(16, dtype=np.float32))
    reverse = appearance.unit_vector(np.arange(15, -1, -1, dtype=np.float32))
    assert appearance.ncc(source, same) > appearance.ncc(source, reverse)


def test_proposal_metrics_require_every_conflict_row() -> None:
    groups = [
        {"rows": [0, 1], "label": 1},
        {"rows": [2], "label": 0},
    ]
    metrics = appearance.proposal_metrics(
        np.asarray([0.9, 0.4, 0.8]), groups, threshold=0.5
    )
    assert metrics == {
        "accepted": 1,
        "correct_rescues": 0,
        "harmful_swaps": 1,
        "occupied_correct_edges": 0,
        "occupied_harmful_edges": 1,
        "precision": 0.0,
        "conservative_utility": -2,
    }


def test_threshold_is_fit_from_admissible_proposals() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, -0.5])
    groups = [
        {"rows": [0], "label": 1},
        {"rows": [1], "label": 1},
        {"rows": [2], "label": 0},
        {"rows": [3], "label": 0},
    ]
    threshold, sweep = appearance._select_threshold(
        scores, groups, minimum_accepted=2, minimum_precision=1.0
    )
    assert threshold is not None
    assert all(np.isfinite(row["threshold"]) for row in sweep)
    metrics = appearance.proposal_metrics(scores, groups, threshold=threshold)
    assert metrics["accepted"] == 2
    assert metrics["precision"] == 1.0


def test_reconstruct_winner_pair_for_source_and_target_conflicts() -> None:
    detection_by_id = {
        1: (0, np.asarray((1.0, 1.0, 1.0))),
        2: (0, np.asarray((2.0, 1.0, 1.0))),
        3: (1, np.asarray((1.0, 2.0, 1.0))),
        4: (1, np.asarray((1.0, 3.0, 1.0))),
    }
    by_frame = {}
    for frame in (0, 1):
        ids = np.asarray(
            [node_id for node_id, (time, _coord) in detection_by_id.items() if time == frame]
        )
        physical = np.stack(
            [detection_by_id[int(node_id)][1] * support.SCALE_ZYX for node_id in ids]
        )
        by_frame[frame] = (ids, physical, support.cKDTree(physical))

    challenger = (1, 3)
    challenger_displacement = (
        detection_by_id[3][1] - detection_by_id[1][1]
    ) * support.SCALE_ZYX
    source_winner_displacement = (
        detection_by_id[4][1] - detection_by_id[1][1]
    ) * support.SCALE_ZYX
    assert support.reconstruct_winner_pair(
        challenger,
        challenger_displacement - source_winner_displacement,
        detection_by_id,
        by_frame,
        tolerance_um=1.0e-6,
    ) == (1, 4)

    target_winner_displacement = (
        detection_by_id[3][1] - detection_by_id[2][1]
    ) * support.SCALE_ZYX
    assert support.reconstruct_winner_pair(
        challenger,
        challenger_displacement - target_winner_displacement,
        detection_by_id,
        by_frame,
        tolerance_um=1.0e-6,
    ) == (2, 3)


def test_support_threshold_enforces_every_video_budget() -> None:
    rows = [
        {"confidence": [0.9, 0.8, 0.7], "budget": 1},
        {"confidence": [0.95, 0.2], "budget": 1},
    ]
    threshold = support._support_threshold(rows, lower_bound=-0.5)
    assert sum(value >= threshold for value in rows[0]["confidence"]) == 1
    assert sum(value >= threshold for value in rows[1]["confidence"]) == 1


def test_direct_swap_realizes_challenger_and_removes_conflict() -> None:
    nodes = tuple(
        bh_graph.GraphNode(node_id=node_id, t=t, z=0.0, y=y, x=0.0)
        for node_id, t, y in (
            (1, 0, 0.0),
            (2, 0, 1.0),
            (3, 1, 0.0),
            (4, 1, 1.0),
        )
    )
    edges = (
        bh_graph.GraphEdge(1, 3, 0.0, 0.0),
        bh_graph.GraphEdge(2, 4, 0.0, 0.0),
    )
    swapped, counts = direct.swap_adjacent_edges(
        nodes,
        edges,
        [{"challenger": [1, 4], "conflicts": [[1, 3], [2, 4]]}],
        {(1, 4): 0.9},
    )
    assert {(edge.source_id, edge.target_id) for edge in swapped} == {(1, 4)}
    assert counts["removed_conflicts"] == 2
    assert counts["added_challengers"] == 1


def test_runtime_contract_loads_exact_frozen_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = root / "reports/g02/appearance_direct_swap_contract_v1.json"
    value, gate = appearance_swap.load_contract(
        contract,
        expected_sha256=appearance_swap.sha256_file(contract),
        root=root,
    )
    assert value["contract_sha256"] == (
        "cee547ed7a82d1bf7f6afe8cbbc3cc9d554181f9e19e87d319a5350863718cf7"
    )
    assert value["appearance"]["patch_shape"] == [16, 32, 32]
    assert gate.parameter_count == 825
    assert all(not parameter.requires_grad for parameter in gate.parameters())


def test_runtime_swap_matches_evaluation_swap() -> None:
    nodes = tuple(
        bh_graph.GraphNode(node_id=node_id, t=t, z=0.0, y=y, x=0.0)
        for node_id, t, y in (
            (1, 0, 0.0),
            (2, 0, 1.0),
            (3, 1, 0.0),
            (4, 1, 1.0),
        )
    )
    edges = (
        bh_graph.GraphEdge(1, 3, 0.0, 0.0),
        bh_graph.GraphEdge(2, 4, 0.0, 0.0),
    )
    proposals = [
        {"challenger": [1, 4], "conflicts": [[1, 3], [2, 4]]}
    ]
    probabilities = {(1, 4): 0.9}
    expected = direct.swap_adjacent_edges(
        nodes, edges, proposals, probabilities
    )
    observed = appearance_swap.swap_post_gap_edges(
        nodes, edges, proposals, probabilities
    )
    assert observed == expected


def test_direct_swap_decision_reference_uses_parent_c0p25_baseline() -> None:
    role, score, margin = direct.decision_reference(
        {"baseline": {"summary": {"adj_edge_jaccard": 0.8731495935}}},
        {},
        temporal_family_gate=False,
    )
    assert role == "parent-c0p25-baseline"
    assert score == pytest.approx(0.8731495935)
    assert margin == pytest.approx(0.005)


def test_temporal_hybrid_decision_reference_uses_banked_candidate() -> None:
    role, score, margin = direct.decision_reference(
        {},
        {"local_confirmation": {"candidate_score": 0.8815220699597315}},
        temporal_family_gate=True,
    )
    assert role == "banked-appearance-direct-swap"
    assert score == pytest.approx(0.8815220699597315)
    assert margin == pytest.approx(0.0)


def test_fit_policy_manifest_is_hash_pinned_and_lowers_only_threshold() -> None:
    root = Path(__file__).resolve().parents[1]
    value = policy_fit.load_policy(
        root / "reports/g02/appearance_direct_policy_fit_v1.json",
        policy_fit.POLICY_SHA256,
    )
    control = policy_fit._policy(value["control"], "control")
    candidate = policy_fit._policy(value["candidate"], "candidate")
    assert candidate[0] < control[0]
    assert candidate[1:] == control[1:]
