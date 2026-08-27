"""Fail-closed tests for the measured-requirement separation verdict layer.

Run: python3 -m pytest test_separation_verdict.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import separation_verdict as sv

HERE = Path(__file__).resolve().parent
AMENDMENT = HERE / "SEPARATION_GATE_MEASURED_AMENDMENT_V4.json"
REQUIREMENT_SURFACE = (
    HERE.parents[7]
    / "reports/g02/screening/selector_requirement_44b6/run_v6/requirement_surface.json"
)

# Exact measured values this gate binds. Pinned so an edit to the amendment
# cannot silently drift the bar.
BINDING = {
    "delete": {"auc": 0.85, "retention": 0.02039355992844365},
    "occupied_swap": {"auc": 0.9, "retention": 0.06791044776119404},
}
LEGACY = {"auc": 0.65, "retention": 0.15}


def arm(auc: float, retention: float, budget_pass: bool = True) -> dict:
    return {
        "auc": auc,
        "retention": retention,
        "videos": [{"identity": "v0", "budget_pass": budget_pass}],
    }


def fold(held: int, a1_auc: float, a1_ret: float, **kw) -> dict:
    """A fold whose B0/A0 sit far enough below A1 to clear every lift condition."""
    return {
        "held_fold": held,
        "arms": {
            "B0": arm(kw.get("b0_auc", a1_auc - 0.20), kw.get("b0_ret", a1_ret - 0.10)),
            "A0": arm(kw.get("a0_auc", a1_auc - 0.10), kw.get("a0_ret", a1_ret - 0.05)),
            "A1": arm(a1_auc, a1_ret, kw.get("budget_pass", True)),
        },
    }


def class_result(action_class: str, folds: list[dict]) -> dict:
    return {
        "action_class": action_class,
        "actions": 1000,
        "beneficial": 100,
        "harmful": 900,
        "folds": folds,
    }


def result_doc(delete_folds: list[dict], swap_folds: list[dict]) -> dict:
    return {
        "schema": sv.RESULT_SCHEMA,
        "status": "COMPLETE",
        "both_class_reporting_complete": True,
        "cross_class_conjunction": False,
        "results": {
            "delete": class_result("delete", delete_folds),
            "occupied_swap": class_result("occupied_swap", swap_folds),
        },
    }


def write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(doc))
    return path


def verdict_for(tmp_path: Path, doc: dict) -> dict:
    return sv.build_verdict(write(tmp_path, doc), AMENDMENT, "TEST-SUPPLIER")


def clearing(action_class: str, margin: float = 0.02) -> list[dict]:
    """Five folds that clear the binding bar for the class."""
    b = BINDING[action_class]
    return [fold(i, b["auc"] + margin, b["retention"] + margin) for i in range(5)]


# --- the amendment itself -------------------------------------------------


def test_amendment_binds_exactly_the_measured_surface_values():
    surface = json.loads(REQUIREMENT_SURFACE.read_text())
    amendment = sv.load_amendment(AMENDMENT)
    for action_class, expected in BINDING.items():
        measured = surface["requirements"][action_class]["target_plus_0_005"]["mean_requirement"]
        assert measured["target_auc"] == expected["auc"]
        assert measured["retention"]["mean"] == expected["retention"]
        bound = amendment["binding_floors"][action_class]
        assert bound["auc_absolute_floor"] == expected["auc"]
        assert bound["retention_absolute_floor"] == expected["retention"]


def test_amendment_records_that_retention_relaxed_and_auc_tightened():
    amendment = sv.load_amendment(AMENDMENT)
    for action_class in sv.CLASSES:
        direction = amendment["binding_floors"][action_class]["direction_of_change"]
        assert direction["auc"] == "TIGHTENED"
        assert direction["retention"] == "RELAXED"
        assert BINDING[action_class]["auc"] > LEGACY["auc"]
        assert BINDING[action_class]["retention"] < LEGACY["retention"]


def test_lift_conditions_carried_forward_unchanged():
    unchanged = sv.load_amendment(AMENDMENT)["unchanged_from_v3"]
    assert unchanged["auc_min_absolute_lift_over_B0"] == 0.05
    assert unchanged["auc_min_absolute_lift_over_A0"] == 0.03
    assert unchanged["retention_min_absolute_lift_over_B0"] == 0.05
    assert unchanged["retention_min_absolute_lift_over_A0"] == 0.03


def test_rejects_an_unfrozen_amendment(tmp_path):
    doc = json.loads(AMENDMENT.read_text())
    doc["status"] = "DRAFT"
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(sv.VerdictError, match="not frozen"):
        sv.load_amendment(path)


# --- the bar actually bites ----------------------------------------------


def test_supplier_clearing_the_binding_bar_passes_both_classes(tmp_path):
    out = verdict_for(tmp_path, result_doc(clearing("delete"), clearing("occupied_swap")))
    assert out["verdicts"] == {
        "delete": sv.PASS_TOKEN,
        "occupied_swap": sv.PASS_TOKEN,
    }


def test_auc_between_legacy_and_measured_now_fails_and_flags_divergence(tmp_path):
    """AUC 0.70: cleared the old 0.65 floor, misses the measured 0.85/0.90.

    This is the case the amendment exists to catch.
    """
    folds = [fold(i, 0.70, 0.30) for i in range(5)]
    out = verdict_for(tmp_path, result_doc(folds, folds))
    for action_class in sv.CLASSES:
        unit = out["units"][action_class]
        assert unit["verdict"] == sv.FAIL_TOKEN
        assert unit["outcomes_by_bar"]["reference_legacy_frozen_v3"]["pass"] is True
        assert unit["divergence_from_legacy_bar"] is True
        for failing in unit["outcomes_by_bar"]["binding_measured_mean"][
            "failing_conditions_by_fold"
        ].values():
            assert "auc_absolute_floor" in failing


def test_low_retention_now_passes_because_the_measured_bar_relaxed_it(tmp_path):
    """Retention 0.10: fails the old 0.15 floor, clears the measured floors.

    Documents the relaxation direction so it stays visible rather than implicit.
    """
    folds_del = [fold(i, 0.90, 0.10) for i in range(5)]
    folds_swap = [fold(i, 0.95, 0.10) for i in range(5)]
    out = verdict_for(tmp_path, result_doc(folds_del, folds_swap))
    for action_class in sv.CLASSES:
        unit = out["units"][action_class]
        assert unit["verdict"] == sv.PASS_TOKEN
        assert unit["outcomes_by_bar"]["reference_legacy_frozen_v3"]["pass"] is False
        assert unit["divergence_from_legacy_bar"] is True


def test_robust_p05_bar_is_reported_alongside_the_binding_bar(tmp_path):
    """Swap AUC 0.92 clears the mean bar 0.90 but misses the robust bar 0.95."""
    folds = clearing("occupied_swap")
    folds = [fold(i, 0.92, 0.20) for i in range(5)]
    out = verdict_for(tmp_path, result_doc(clearing("delete"), folds))
    unit = out["units"]["occupied_swap"]
    assert unit["verdict"] == sv.PASS_TOKEN
    assert unit["outcomes_by_bar"]["reference_measured_robust_p05"]["pass"] is False
    assert unit["divergence_from_robust_bar"] is True


def test_classes_are_judged_independently(tmp_path):
    out = verdict_for(tmp_path, result_doc(clearing("delete"), [fold(i, 0.70, 0.30) for i in range(5)]))
    assert out["units"]["delete"]["verdict"] == sv.PASS_TOKEN
    assert out["units"]["occupied_swap"]["verdict"] == sv.FAIL_TOKEN
    assert out["cross_class_conjunction"] is False


def test_a_single_failing_fold_fails_the_unit(tmp_path):
    folds = clearing("delete")
    folds[3] = fold(3, 0.70, 0.30)
    out = verdict_for(tmp_path, result_doc(folds, clearing("occupied_swap")))
    assert out["units"]["delete"]["verdict"] == sv.FAIL_TOKEN


def test_missing_lift_over_permuted_control_fails(tmp_path):
    """A1 separates, but A0 matches it: capacity, not representation."""
    folds = [
        fold(i, 0.90, 0.30, a0_auc=0.895, a0_ret=0.295) for i in range(5)
    ]
    out = verdict_for(tmp_path, result_doc(folds, clearing("occupied_swap")))
    unit = out["units"]["delete"]
    assert unit["verdict"] == sv.FAIL_TOKEN
    for failing in unit["outcomes_by_bar"]["binding_measured_mean"][
        "failing_conditions_by_fold"
    ].values():
        assert "auc_lift_over_A0" in failing


# --- fail-closed ----------------------------------------------------------


def test_missing_fold_is_invalid_not_pass(tmp_path):
    out = verdict_for(tmp_path, result_doc(clearing("delete")[:4], clearing("occupied_swap")))
    unit = out["units"]["delete"]
    assert unit["verdict"] == sv.INVALID_TOKEN
    assert unit["pass"] is False
    assert "fold coverage" in unit["reason"]


def test_missing_arm_is_invalid(tmp_path):
    folds = clearing("delete")
    del folds[0]["arms"]["A0"]
    out = verdict_for(tmp_path, result_doc(folds, clearing("occupied_swap")))
    assert out["units"]["delete"]["verdict"] == sv.INVALID_TOKEN


def test_nonfinite_metric_is_invalid(tmp_path):
    folds = clearing("delete")
    folds[2]["arms"]["A1"]["auc"] = float("nan")
    path = tmp_path / "result.json"
    # json.dumps emits bare NaN, which json.loads accepts; assert we reject it.
    path.write_text(json.dumps(result_doc(folds, clearing("occupied_swap"))))
    out = sv.build_verdict(path, AMENDMENT, "TEST-SUPPLIER")
    assert out["units"]["delete"]["verdict"] == sv.INVALID_TOKEN


def test_per_video_budget_violation_is_invalid(tmp_path):
    folds = clearing("delete")
    folds[1]["arms"]["A1"]["videos"] = [{"identity": "v7", "budget_pass": False}]
    out = verdict_for(tmp_path, result_doc(folds, clearing("occupied_swap")))
    unit = out["units"]["delete"]
    assert unit["verdict"] == sv.INVALID_TOKEN
    assert "budget" in unit["reason"]


def test_incomplete_result_is_rejected_outright(tmp_path):
    doc = result_doc(clearing("delete"), clearing("occupied_swap"))
    doc["status"] = "PARTIAL"
    with pytest.raises(sv.VerdictError, match="not COMPLETE"):
        verdict_for(tmp_path, doc)


def test_single_class_result_is_rejected_outright(tmp_path):
    doc = result_doc(clearing("delete"), clearing("occupied_swap"))
    del doc["results"]["occupied_swap"]
    with pytest.raises(sv.VerdictError, match="exactly the two gate classes"):
        verdict_for(tmp_path, doc)


def test_wrong_schema_is_rejected_outright(tmp_path):
    doc = result_doc(clearing("delete"), clearing("occupied_swap"))
    doc["schema"] = "something-else.v1"
    with pytest.raises(sv.VerdictError, match="schema mismatch"):
        verdict_for(tmp_path, doc)


def test_cross_class_conjunction_is_rejected(tmp_path):
    doc = result_doc(clearing("delete"), clearing("occupied_swap"))
    doc["cross_class_conjunction"] = True
    with pytest.raises(sv.VerdictError, match="cross-class conjunction"):
        verdict_for(tmp_path, doc)


def test_verdict_records_no_downstream_action(tmp_path):
    out = verdict_for(tmp_path, result_doc(clearing("delete"), clearing("occupied_swap")))
    assert out["graph_run"] is False
    assert out["family_6bba_run"] is False
    assert out["cv57_run"] is False
    assert out["tier_move"] is False
    assert out["submission"] is False
    assert out["barrier"].startswith("HOLD_AFTER_PER_CLASS_VERDICTS")
