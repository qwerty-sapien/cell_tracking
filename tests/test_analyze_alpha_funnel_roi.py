import json
from pathlib import Path

import pytest

from scripts.analyze_alpha_funnel_roi import (
    adjusted_edge_jaccard,
    compute_analysis,
    minimum_beneficial_swaps,
    sha256_file,
    swap_delta,
)


def _evidence():
    return {
        "study_id": "test",
        "claude_scope_audit": {
            "requested_population_movies": 6,
            "launched_population_movies": 57,
            "requested_artifacts": ["a", "b", "c", "d", "e"],
            "launched_exporter_payload": "cache",
            "launched_receipt_says_still_requires_model_forward": ["D", "E"],
            "scientific_outcome": "INVALID",
            "candidate_scores_opened": 0,
        },
        "score_evidence": {
            "baseline": {"tp": 5200, "fp": 402, "fn": 313},
            "minimum_effect": 0.005,
            "perfect_repair_classes": {"swap": 167},
        },
        "relevant_learned_graph_precedents": [
            {"adjusted_delta": -0.0111480766, "cleared_full_gate": False},
            {"adjusted_delta": 0.000133826958, "cleared_full_gate": False},
            {"adjusted_delta": -0.0003102875275311501, "cleared_full_gate": False},
            {"adjusted_delta": 0.006949544395, "cleared_full_gate": False},
        ],
        "scenario_probability_assumptions": {
            "worst": 0.025,
            "base": 0.10,
            "best_before_target_learnability_pilot": 0.25,
            "status": "ASSUMPTIONS_NOT_POSTERIOR_PROBABILITIES",
        },
        "provider_prices_usd_per_gpu_hour": {
            "rtx_a6000": 0.35,
            "a100_80gb": 1.09,
        },
        "substrate_and_reuse_audit": {
            "cache_supported_frozen_candidates": {"count": 34},
            "model_forward_required_frozen_candidates": {"count": 14},
            "prior_feature_bank": {"current_six_movie_overlap": 0},
            "peft_capacity_probe": {
                "selected_oof_auc": 0.49649547544786093,
                "selected_beneficial_at_budget": 3,
                "total_beneficial": 1118,
            },
        },
    }


def test_swap_accounting_reproduces_baseline_and_minimum_gate():
    assert adjusted_edge_jaccard(5200, 402, 313) == 0.8791208791208791
    assert minimum_beneficial_swaps(5200, 402, 313, 0.005) == 16
    assert swap_delta(5200, 402, 313, 15) < 0.005
    assert swap_delta(5200, 402, 313, 16) >= 0.005


def test_harmful_swap_precision_examples_are_exact():
    expected = {0: 16, 1: 17, 2: 18, 5: 21, 10: 26, 20: 36}
    observed = {
        harmful: minimum_beneficial_swaps(5200, 402, 313, 0.005, harmful)
        for harmful in expected
    }
    assert observed == expected


def test_analysis_parks_paid_compute_and_costs_the_two_configurations():
    result = compute_analysis(_evidence(), [])
    assert result["decision"]["paid_thunder_gpu_hours_now"] == 0
    assert result["decision"]["verdict"] == "PARK_PAID_TRAINING"
    assert result["locked_f6_gate_audit"]["config_space_exhausted"]["f3_coverage"] == "0/48"
    assert result["prior_outcomes"]["full_gate_successes"] == 0
    assert result["thunder_cost_sensitivity_usd"]["8"] == {
        "rtx_a6000": 2.8,
        "a100_80gb": 8.72,
    }
    assert result["scenario_sensitivity"]["status"] == "ASSUMPTIONS_NOT_POSTERIOR_PROBABILITIES"


def test_f6_receipt_copies_roi_values_without_rounding_drift():
    root = Path(__file__).resolve().parents[1]
    runs = root / "dag_prompts/2026-08-20-alpha-funnel-v1/runs"
    analysis = json.loads((runs / "F2B_ROI_ANALYSIS.json").read_text())
    f6 = json.loads((runs / "F6_TRAINING_JUSTIFICATION.json").read_text())
    sensitivity = analysis["scenario_sensitivity"]

    assert f6["status"] == "PARK"
    assert f6["thunder_decision"]["gpu_hours_authorized_now"] == 0
    for binding in f6["evidence"].values():
        if not isinstance(binding, dict):
            continue
        assert sha256_file(runs / binding["path"]) == binding["sha256"]
    assert f6["effect_arithmetic"]["conditional_best_delta_at_24_49_percent_of_oracle"] == pytest.approx(
        sensitivity["conditional_best_delta"], abs=0.0
    )
    assert f6["probability_sensitivity_not_posterior"]["p_required_for_expected_delta_at_least_0_005"] == pytest.approx(
        sensitivity["probability_needed_for_expected_delta_at_least_0_005"], abs=0.0
    )
    name_map = {
        "worst": "worst",
        "base": "base",
        "best_before_target_pilot": "best_before_target_learnability_pilot",
    }
    for f6_name, analysis_name in name_map.items():
        assert f6["probability_sensitivity_not_posterior"][f6_name]["score_ev"] == pytest.approx(
            sensitivity["scenarios"][analysis_name]["expected_score_delta"], abs=0.0
        )
