from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_cgl_d14_fastfail import (  # noqa: E402
    CGLFastFailError,
    FEATURE_DIM,
    _hard_topk_decision,
    _one_to_one_gt_edges,
    _ranking_decision,
    assert_safe_fit_only_path,
    compact_pair_features,
)


def test_forbidden_paths_are_rejected_before_read() -> None:
    for value in ("reports/private_cv57.json", "reports/HOLDOUT/data.json"):
        with pytest.raises(CGLFastFailError, match="before read"):
            assert_safe_fit_only_path(Path(value))


def test_compact_pair_feature_contract() -> None:
    rng = np.random.default_rng(7)
    count = 5
    observed = compact_pair_features(
        source_features=rng.normal(size=(count, 32)).astype(np.float32),
        target_features=rng.normal(size=(count, 32)).astype(np.float32),
        displacement_um=rng.normal(size=(count, 3)).astype(np.float32),
        distance_um=np.abs(rng.normal(size=count)).astype(np.float32),
        source_scores=rng.uniform(size=count).astype(np.float32),
        target_scores=rng.uniform(size=count).astype(np.float32),
        checkpoint_logits=rng.normal(size=count).astype(np.float32),
    )
    assert observed.shape == (count, FEATURE_DIM)
    assert observed.dtype == np.float16
    assert np.isfinite(observed).all()


def test_ranking_metric_exhausts_only_the_background_budget() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = np.asarray([1, 0, 1, 0, 1], dtype=np.int8)
    accepted = _ranking_decision(scores, labels, budget=1)
    assert accepted.tolist() == [True, True, True, False, False]
    assert int(np.sum(accepted & (labels == 0))) == 1


def test_one_to_one_gt_edges_excludes_divisions_and_merges() -> None:
    observed = _one_to_one_gt_edges(
        [(1, 2), (2, 3), (2, 4), (5, 6), (7, 6), (8, 9)]
    )
    assert observed == {(1, 2), (8, 9)}


def test_hard_topk_is_a_count_safe_lower_bound() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    accepted = _hard_topk_decision(scores, budget=2)
    assert accepted.tolist() == [True, True, False, False]
