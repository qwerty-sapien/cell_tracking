from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_appearance_residual",
    ROOT / "scripts/analyze_appearance_residual.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_minimum_corrections_reaches_target() -> None:
    result = MODULE.minimum_corrections(
        edge_tp=90,
        edge_fp=5,
        edge_fn=5,
        node_adjustment=1.0,
        target_score=0.95,
    )
    assert result["fn_to_tp_only"] == 5
    assert result["fp_prune_only"] == 6
    assert result["perfect_wrong_to_right_swaps"] == 3
    swaps = int(result["perfect_wrong_to_right_swaps"])
    assert (90 + swaps) / (100 - swaps) >= 0.95


def _sample(identity: str, tp: int, fp: int, fn: int) -> dict[str, object]:
    return {
        "identity": identity,
        "edge_tp": float(tp),
        "edge_fp": float(fp),
        "edge_fn": float(fn),
        "adj_edge_jaccard": tp / (tp + fp + fn),
    }


def test_summarize_videos_reconciles_effect_and_concentration() -> None:
    baseline = []
    candidate = []
    run = []
    for index in range(57):
        identity = f"train:6bba_{index:08d}"
        baseline.append(_sample(identity, 10, 2, 2))
        if index == 0:
            candidate.append(_sample(identity, 11, 1, 1))
        elif index == 1:
            candidate.append(_sample(identity, 9, 3, 3))
        else:
            candidate.append(_sample(identity, 10, 2, 2))
        run.append({"identity": identity, "proposal_count": 2})
    result = MODULE.summarize_videos(baseline, candidate, run)
    assert result["helpful_videos"] == 1
    assert result["harmful_videos"] == 1
    assert result["neutral_videos"] == 55
    assert result["proposal_count"] == 114
    assert result["event_utility"] == 0
    assert result["candidate_residual"] == 228
    assert math.isclose(
        result["residual_concentration"]["1"]["share"], 6 / 228
    )


def test_minimum_corrections_rejects_unreachable_swap_target() -> None:
    with pytest.raises(MODULE.ResidualAuditError, match="available perfect"):
        MODULE.minimum_corrections(
            edge_tp=1,
            edge_fp=0,
            edge_fn=1,
            node_adjustment=1.0,
            target_score=0.99,
        )
