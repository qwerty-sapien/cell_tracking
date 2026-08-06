from __future__ import annotations

from decimal import Decimal

import pytest

from scripts.validate_g03_wave_r import (
    BEST_KNOWN_SCORE,
    NEIGHBORHOOD_FLOOR,
    expected_branch,
    validate,
)


def test_exact_owner_floor_and_boundary_classification() -> None:
    assert NEIGHBORHOOD_FLOOR == Decimal("0.868299238910600")
    assert expected_branch(NEIGHBORHOOD_FLOOR) == "A"
    assert expected_branch(NEIGHBORHOOD_FLOOR - Decimal("0.000000000001")) == "B"
    assert expected_branch(BEST_KNOWN_SCORE + Decimal("0.000000000001")) == "A"


def test_promoted_wave_r_evidence_validates_after_r1_finalization() -> None:
    report = validate(
        __import__("pathlib").Path(__file__).resolve().parents[1],
        allow_pending_r1=True,
    )
    assert report["status"] == "PASS_COMPLETE_WAVE_R_EVIDENCE"
    assert report["r1"]["branch"] == "B"
    assert report["r1"]["score"] == "0.7637338804169099"
    assert report["r2"]["verdict"] == "BRANCH_B_FAIL"
    assert report["r3"]["real_wrfeat_reusable"] is False
    assert report["r4"]["total_configurations"] == 18_145


@pytest.mark.parametrize(
    ("score", "branch"),
    [
        (Decimal("0.868299238910600"), "A"),
        (Decimal("0.860000000000"), "B"),
        (Decimal("0.900000000000"), "A"),
    ],
)
def test_policy_branch_examples(score: Decimal, branch: str) -> None:
    assert expected_branch(score, new_best=score > BEST_KNOWN_SCORE) == branch
