from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANES = ROOT / "reports/g02/screening/tiered_surface/branches/g03_orthogonal_lanes_20260803"


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="ascii"))
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_freeze_hashes_are_still_exact() -> None:
    freeze = load(LANES / "FREEZE_RECEIPT.json")
    assert freeze["status"] == "SEALED_BEFORE_NEW_LANE_OUTCOMES"
    assert freeze["outcomes_opened_at_freeze"] is False
    for row in freeze["artifacts"].values():
        path = ROOT / row["path"]
        assert path.is_file()
        assert sha256(path) == row["sha256"]


def test_h005_gate_is_same_population_and_boundary_is_stable() -> None:
    prereg = load(LANES / "lane0_h005/PREREGISTRATION.json")
    numerical = load(LANES / "lane0_h005/FIXED_SCORE_NUMERICAL_AMENDMENT.json")
    assert prereg["checkpoint"]["start_population"] == "held-fit36"
    assert prereg["scoring"]["population"] == "held-fit36"
    assert prereg["gate"]["reference_name"] == "APNCC held-fit36"
    assert "CV57 access" in prereg["scoring"]["forbidden"]
    h000 = prereg["checkpoint"]["start_score"]
    score = prereg["gate"]["exactly_5h_minimum_h005_score"]
    projection = h000 + 30.0 * (score - h000) / 5.0
    tolerance = numerical["rule"]["absolute_comparison_tolerance"]
    assert projection + tolerance >= prereg["gate"]["reference_score"]
    assert numerical["h005_score_known_at_freeze"] is False


def test_fixed_checkpoint_evaluation_is_remote_score_blind() -> None:
    contract = load(LANES / "lane0_h005/FIXED_CHECKPOINT_EVALUATION_CONTRACT.json")
    remote = contract["remote_score_blind_bank"]
    local = contract["local_compile_and_score"]
    assert contract["status"] == "FROZEN_BEFORE_H005_CHECKPOINT_OR_SCORE_EXISTS"
    assert remote["population"] == "held-fit36"
    assert remote["identity_count"] == 36
    assert remote["gt_access"] is False
    assert remote["score_access"] is False
    assert remote["cv57_access"] is False
    assert local["cross_population_subtraction"] is False
    assert local["cv57_access"] is False
    assert contract["allocation"]["h030_not_automatically_launched"] is True


def test_h005_terminal_failure_stops_before_checkpoint_evaluation_and_h030() -> None:
    diagnosis = load(LANES / "lane0_h005/V2_TERMINAL_DIAGNOSIS.json")
    terminal = load(LANES / "TERMINAL_RESULT.json")
    assert diagnosis["status"] == "INVALID_PRETRAINING_REMOTE_MOUNT_INTEGRITY_FAILURE"
    assert diagnosis["failure"]["exception"] == "RuntimeError: runtime dependency manifest mismatch"
    assert diagnosis["negative_evidence"]["training_main_entered"] is False
    assert diagnosis["negative_evidence"]["cuda_model_allocation"] is False
    assert diagnosis["negative_evidence"]["optimizer_steps"] == 0
    assert diagnosis["negative_evidence"]["h005_checkpoint"] is False
    assert diagnosis["negative_evidence"]["held_fit36_access"] is False
    assert diagnosis["negative_evidence"]["cv57_access"] is False
    lane0 = terminal["lane0_h005"]
    assert lane0["population"] == "held-fit36"
    assert lane0["h005_score"] is None
    assert lane0["slope_adjusted_graph_score_per_t4_hour"] is None
    assert lane0["h030_gate_evaluable"] is False
    assert lane0["h030_funded"] is False
    assert lane0["h030_launched"] is False
    assert lane0["version_3_authorized"] is False


def test_dense_motion_is_distinct_and_uses_frozen_consumer() -> None:
    prereg = load(LANES / "lane1_dense_motion/PREREGISTRATION.json")
    assert "dense" in prereg["orthogonality"]["supplier"].lower()
    assert "rigid global phase-correlation" in prereg["orthogonality"]["distinct_from_A7"]
    assert prereg["immutable_inputs"]["population"] == "fit-only frozen 44b6"
    assert prereg["immutable_inputs"]["identity_count"] == 41
    assert prereg["consumer"]["arms_in_order"] == ["sealed B0", "A1", "A0"]
    assert "unchanged" in prereg["consumer"]["probe"].lower()
    assert "human review only" in prereg["ceiling_and_kill"]["pass"]


def test_dense_motion_terminal_result_closes_learned_stage() -> None:
    result = load(LANES / "lane1_dense_motion/run_v1/SEPARATION_RESULT.json")
    features = load(LANES / "lane1_dense_motion/run_v1/FEATURE_MANIFEST.json")
    assert result["status"] == "COMPLETE"
    assert result["execution_order"] == ["B0_SEALED", "A1", "A0"]
    assert result["sealed_B0"]["sha256"] == "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2"
    assert features["aggregate"]["identity_count"] == 41
    assert features["aggregate"]["field_pair_count"] == 4059
    assert features["aggregate"]["action_count"] == 1_391_458
    assert features["gpu_hours"] == 0.0
    assert features["graph"] is False
    assert all(value["verdict"] == "FAIL_PARK_C" for value in result["results"].values())
    assert all(not fold["pass"] for value in result["results"].values() for fold in value["folds"])
    assert result["execution"]["graph"] is False
    assert result["execution"]["family_6bba"] is False
    assert result["execution"]["cv57"] is False


def test_trained_fork_stopped_cross_family_before_final_38() -> None:
    result = load(LANES / "lane2_trained_fork/run_v1/CROSS_FAMILY_RESULT.json")
    assert result["status"] == "FAIL_CROSS_FAMILY_STOP_BEFORE_FINAL_38"
    assert result["train"]["family"] == "6bba"
    assert result["validation"]["family"] == "44b6"
    assert result["final_38_access"] is False
    assert result["graph_integration"] is False
    assert result["gpu_hours"] == 0.0
    metrics = result["validation"]["metrics"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (5, 6, 4, 30)
    assert math.isclose(metrics["precision"], 5 / 11)
    assert math.isclose(metrics["recall"], 5 / 9)
    report = (LANES / "lane2_trained_fork/TERMINAL_REPORT.md").read_text(encoding="utf-8")
    assert "hand-crafted geometry, persistence, and decode rules" in report
    assert "not a re-screen" in report
