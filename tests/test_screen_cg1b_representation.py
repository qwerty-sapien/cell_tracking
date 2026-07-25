from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from screen_cg1b_representation import (  # noqa: E402
    CG1bError,
    _json_safe,
    _rank01,
    _topk_video,
    assert_safe_fit_only_path,
)


def test_forbidden_paths_are_rejected_before_read() -> None:
    for value in ("reports/private_cv57.json", "reports/HOLDOUT/data.json"):
        with pytest.raises(CG1bError, match="before read"):
            assert_safe_fit_only_path(Path(value))


def test_rank01_is_deterministic_with_ties() -> None:
    observed = _rank01(np.asarray([4.0, 1.0, 4.0, 2.0]))
    np.testing.assert_allclose(observed, [0.875, 0.25, 0.875, 0.5])


def test_hard_topk_never_admits_more_than_the_video_budget() -> None:
    video = {
        "scores": {"signal": np.asarray([0.1, 0.9, 0.4, 0.4])},
        "challengers": np.asarray([[3, 4], [1, 2], [2, 3], [1, 3]]),
        "budget": 2,
    }
    accepted = _topk_video(video, "signal")
    assert int(accepted.sum()) == 2
    assert accepted.tolist() == [False, True, False, True]


def test_json_safe_replaces_infinite_thresholds() -> None:
    value = _json_safe({"threshold": math.inf, "nested": [-math.inf, 0.5]})
    assert value == {"threshold": None, "nested": [None, 0.5]}
