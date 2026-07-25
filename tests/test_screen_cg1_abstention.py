from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from screen_cg1_abstention import (  # noqa: E402
    CG1ScreenError,
    _json_safe,
    _upper_binomial_rate,
    assert_safe_fit_only_path,
)


def test_forbidden_paths_are_rejected_before_read() -> None:
    for value in ("reports/private_cv57.json", "reports/HOLDOUT/data.json"):
        with pytest.raises(CG1ScreenError, match="forbidden"):
            assert_safe_fit_only_path(Path(value))


def test_zero_success_upper_bound_is_nonzero() -> None:
    upper = _upper_binomial_rate(0, 8668)
    assert 0.0 < upper < 0.001


def test_json_safe_replaces_infinite_abstention_threshold() -> None:
    value = _json_safe({"threshold": math.inf, "nested": [-math.inf, 0.5]})
    assert value == {"threshold": None, "nested": [None, 0.5]}
