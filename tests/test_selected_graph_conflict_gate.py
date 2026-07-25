"""Focused contracts for selected-graph rescue/control construction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from celltrack_alpha.bh_edge import SparseEdgeConfig, SparseEdgeScorer
from celltrack_alpha.bh_model import FrameDet, PairContext, VideoInference
from celltrack_alpha.edge_data import (
    CONTROL_LABEL,
    RESCUE_LABEL,
    SOURCE_CONFLICT,
    TARGET_CONFLICT,
    selected_graph_conflict_examples_from_video,
)
from scripts.fit_selected_graph_conflict_gate import (
    _proposal_metrics,
    _select_threshold,
)
from scripts.evaluate_selected_graph_conflict_gate import (
    _cap_proposals,
    _greedy_nonconflicting,
)
from scripts.fit_selected_graph_background_gate import _weighted_loss
from scripts.prepare_selected_graph_background import _select_groups


class _PairMLP(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(nn.Linear(259, 16), nn.GELU(), nn.Linear(16, 1))


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91)
            self.proj = nn.Linear(64, 128)
            self.norm_in = nn.LayerNorm(128)
            self.norm_out = nn.LayerNorm(128)
            self.pair_mlp = _PairMLP()


def _frame(frame: int, ids: list[int], x: list[int]) -> FrameDet:
    points = np.asarray([(0, 10, value) for value in x], dtype=np.int64)
    return FrameDet(
        frame_index=frame,
        node_ids=np.asarray(ids, dtype=np.int64),
        coords_zyx=points,
        scores=np.full(len(ids), 0.95, dtype=np.float32),
        grid_coords_zyx=points,
        grid_shape_zyx=(8, 64, 64),
    )


def _video() -> VideoInference:
    source = _frame(0, [1, 2], [10, 14])
    target = _frame(1, [3, 4], [11, 13])
    context = PairContext(
        source_frame=0,
        target_frame=1,
        source_ids=source.node_ids,
        target_ids=target.node_ids,
        source_feat=np.arange(64, dtype=np.float32).reshape(2, 32) / 37.0,
        target_feat=np.arange(64, dtype=np.float32).reshape(2, 32) / 41.0,
        grid_shape_zyx=source.grid_shape_zyx,
    )
    return VideoInference((source, target), (context,), d4_views=(0,))


def _scorer() -> SparseEdgeScorer:
    return SparseEdgeScorer(
        _Transformer(), config=SparseEdgeConfig(radius_um=10.0, neighbor_cap=8)
    )


def test_missed_truth_is_paired_with_actual_selected_endpoint_conflicts() -> None:
    examples = selected_graph_conflict_examples_from_video(
        _scorer(),
        _video(),
        {1: 101, 2: 102, 3: 103, 4: 104},
        {(101, 104), (102, 103)},
        {(1, 3), (2, 4)},
    )
    observed = {
        (tuple(challenger), tuple(selected), int(kind), int(label))
        for challenger, selected, kind, label in zip(
            examples.challenger_pairs,
            examples.selected_pairs,
            examples.conflict_kind,
            examples.labels,
            strict=True,
        )
    }
    assert observed == {
        ((1, 4), (1, 3), SOURCE_CONFLICT, RESCUE_LABEL),
        ((1, 4), (2, 4), TARGET_CONFLICT, RESCUE_LABEL),
        ((2, 3), (2, 4), SOURCE_CONFLICT, RESCUE_LABEL),
        ((2, 3), (1, 3), TARGET_CONFLICT, RESCUE_LABEL),
    }
    assert examples.rescue_count == 4
    assert examples.control_count == 0
    assert examples.reachable_miss_count == 2


def test_correct_selected_edges_generate_known_negative_controls() -> None:
    examples = selected_graph_conflict_examples_from_video(
        _scorer(),
        _video(),
        {1: 101, 2: 102, 3: 103, 4: 104},
        {(101, 103), (102, 104)},
        {(1, 3), (2, 4)},
    )
    assert examples.rescue_count == 0
    assert examples.control_count == 4
    assert set(examples.labels.tolist()) == {CONTROL_LABEL}
    assert {
        (tuple(challenger), tuple(selected), int(kind))
        for challenger, selected, kind in zip(
            examples.challenger_pairs,
            examples.selected_pairs,
            examples.conflict_kind,
            strict=True,
        )
    } == {
        ((1, 4), (1, 3), SOURCE_CONFLICT),
        ((2, 3), (2, 4), SOURCE_CONFLICT),
        ((2, 3), (1, 3), TARGET_CONFLICT),
        ((1, 4), (2, 4), TARGET_CONFLICT),
    }


def test_unmatched_challenger_is_never_used_as_a_negative_control() -> None:
    examples = selected_graph_conflict_examples_from_video(
        _scorer(),
        _video(),
        {1: 101, 2: 102, 3: 103, 4: None},
        {(101, 103)},
        {(1, 3), (2, 4)},
    )
    observed = {
        tuple(pair)
        for pair, label in zip(examples.challenger_pairs, examples.labels, strict=True)
        if int(label) == CONTROL_LABEL
    }
    assert (1, 4) not in observed


def test_proposal_requires_every_occupied_endpoint_to_pass() -> None:
    groups = [
        {"rows": [0, 1], "label": 1, "occupied_endpoints": 2},
        {"rows": [2], "label": 0, "occupied_endpoints": 1},
    ]
    metrics = _proposal_metrics(
        np.asarray([0.99, 0.80, 0.70]), groups, threshold=0.9
    )
    assert metrics["accepted"] == 0
    assert metrics["available_rescues"] == 1


def test_threshold_selection_enforces_high_precision_controls() -> None:
    probabilities = np.asarray(
        [0.99] * 12 + [0.80] * 2 + [0.60] * 2, dtype=np.float64
    )
    groups = [
        {"rows": [index], "label": 1, "occupied_endpoints": 1}
        for index in range(12)
    ] + [
        {"rows": [12], "label": 0, "occupied_endpoints": 1},
        {"rows": [13], "label": 0, "occupied_endpoints": 1},
        {"rows": [14], "label": 0, "occupied_endpoints": 1},
        {"rows": [15], "label": 0, "occupied_endpoints": 1},
    ]
    threshold, _rows = _select_threshold(probabilities, groups)
    assert threshold > 0.8
    metrics = _proposal_metrics(probabilities, groups, threshold=threshold)
    assert metrics["correct_rescues"] == 12
    assert metrics["harmful_swaps"] == 0


def test_greedy_proposals_are_confidence_ordered_and_node_disjoint() -> None:
    proposals = [
        {
            "challenger": [1, 4],
            "conflicts": [[1, 3], [2, 4]],
            "confidence": 0.91,
            "row_probabilities": [0.95, 0.91],
        },
        {
            "challenger": [2, 3],
            "conflicts": [[2, 4], [1, 3]],
            "confidence": 0.99,
            "row_probabilities": [0.99, 0.99],
        },
        {
            "challenger": [10, 11],
            "conflicts": [[10, 12]],
            "confidence": 0.92,
            "row_probabilities": [0.92],
        },
    ]
    accepted = _greedy_nonconflicting(proposals)
    assert [row["challenger"] for row in accepted] == [[2, 3], [10, 11]]
    capped = _cap_proposals(accepted, 1)
    assert [row["challenger"] for row in capped] == [[2, 3]]
    assert _cap_proposals(accepted, None) == accepted


class _FirstFeatureGate(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0]


def test_background_sampling_excludes_labels_and_mixes_hard_with_uniform() -> None:
    groups = []
    for index in range(10):
        features = np.zeros((1, 44), dtype=np.float32)
        features[0, 0] = float(index)
        groups.append(
            {
                "challenger": [index, 100 + index],
                "conflicts": [[index, 200 + index]],
                "relative_features": features,
            }
        )
    first, counts = _select_groups(
        groups,
        identity="train:fixture",
        known={(9, 109)},
        gate=_FirstFeatureGate(),
        seed=17,
        sample_rate=1.0,
        minimum=4,
        maximum=4,
        hard_fraction=0.5,
    )
    second, _ = _select_groups(
        groups,
        identity="train:fixture",
        known={(9, 109)},
        gate=_FirstFeatureGate(),
        seed=17,
        sample_rate=1.0,
        minimum=4,
        maximum=4,
        hard_fraction=0.5,
    )
    assert counts == {
        "unlabeled_proposals": 9,
        "selected_proposals": 4,
        "selected_hard": 2,
        "selected_uniform": 2,
    }
    assert [row["challenger"] for row in first] == [
        row["challenger"] for row in second
    ]
    assert {(7, 107), (8, 108)}.issubset(
        {tuple(row["challenger"]) for row in first}
    )
    assert (9, 109) not in {tuple(row["challenger"]) for row in first}


def test_background_loss_is_explicitly_downweighted() -> None:
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 0.0])
    equal = _weighted_loss(
        logits, labels, torch.tensor([1.0, 1.0]), positive_weight=1.0
    )
    downweighted = _weighted_loss(
        logits, labels, torch.tensor([1.0, 0.05]), positive_weight=1.0
    )
    expected = torch.tensor(np.log(2.0), dtype=torch.float32)
    assert torch.isclose(equal, expected, atol=1e-6)
    assert torch.isclose(downweighted, expected, atol=1e-6)

    asymmetric_logits = torch.tensor([2.0, 2.0])
    equal_asymmetric = _weighted_loss(
        asymmetric_logits,
        labels,
        torch.tensor([1.0, 1.0]),
        positive_weight=1.0,
    )
    downweighted_asymmetric = _weighted_loss(
        asymmetric_logits,
        labels,
        torch.tensor([1.0, 0.05]),
        positive_weight=1.0,
    )
    assert downweighted_asymmetric < equal_asymmetric
