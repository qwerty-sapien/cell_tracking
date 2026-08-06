#!/usr/bin/env python3
"""Independent structural and arithmetic validation of N0--N3 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
GT = "GT-DEPENDENT"
NONGT = "NON-GT-IDENTIFIABLE"


def load(name: str) -> dict:
    value = json.loads((HERE / name).read_text(encoding="ascii"))
    assert isinstance(value, dict)
    return value


def sha(name: str) -> str:
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def assert_all_numbers_labeled(value: object, inherited_label: str | None = None) -> int:
    """Require every scientific numeric leaf to inherit an explicit framing label."""
    if isinstance(value, dict):
        own_label = value.get("information_label")
        label = own_label if own_label in (GT, NONGT) else inherited_label
        return sum(
            assert_all_numbers_labeled(child, label)
            for child in value.values()
        )
    if isinstance(value, list):
        return sum(
            assert_all_numbers_labeled(child, inherited_label)
            for child in value
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        assert inherited_label in (GT, NONGT)
        return 1
    return 0


def main() -> None:
    results = load("N0_N3_RESULTS.json")
    dependency = load("GT_DEPENDENCY_TABLE.json")
    curves = load("FP_SENSITIVITY.json")
    provenance = load("RUN_PROVENANCE.json")
    guard = load("resource_guard.json")
    preflight = load("idle_preflight.json")

    assert results["status"] == "N0_N3_COMPLETE"
    assert guard["passed"] is True and guard["cpu_only"] is True
    assert guard["swap_growth_mib"] <= 0.0
    assert guard["peak_process_rss_mib"] <= 8192.0
    assert preflight["passed"] is True and preflight["swap_growth_mib"] <= 0.0
    assert provenance["cpu_only"] is True
    assert provenance["gpu_used"] is False
    assert provenance["kaggle_used"] is False
    assert provenance["budget_debit"] is False
    assert provenance["fitting"] is False
    assert provenance["bundle_derivation"] is False

    count_records = results["n1_gt_dependency"]["tier_counts"]
    counts = {tier: record["count"] for tier, record in count_records.items()}
    assert counts == {"a": 1, "b": 37, "c": 0}
    assert count_records["a"]["information_label"] == NONGT
    assert count_records["b"]["information_label"] == NONGT
    assert count_records["c"]["information_label"] == GT
    assert sum(counts.values()) == 38
    assert len(dependency["events"]) == 38
    assert {row["tier"] for row in dependency["events"]} == {"a", "b"}
    for event in dependency["events"]:
        assert event["information_label"] in (GT, NONGT)
        audit = event["candidate_audit"]
        assert abs(audit["probability_sum_including_no_link"] - 1.0) <= 1e-10
    assert sum(row["event_count"] for row in dependency["tiers"]) == 38

    ceiling = results["n0_ceiling"]["coalition"]["pooled"]["delta"]
    assert ceiling["information_label"] == GT
    assert ceiling["division_tp"] == 38
    assert ceiling["division_fp"] == 0
    assert ceiling["division_fn"] == 4
    assert math.isclose(ceiling["division_jaccard"], 38 / 42, abs_tol=1e-15)
    assert math.isclose(
        ceiling["combined_score_delta"],
        ceiling["adjusted_edge_jaccard_delta"] + 0.1 * ceiling["division_jaccard"],
        abs_tol=1e-15,
    )
    assert results["n0_ceiling"]["coalition"]["families"]["44b6"]["delta"][
        "division_tp"
    ] == 9
    assert results["n0_ceiling"]["coalition"]["families"]["6bba"]["delta"][
        "division_tp"
    ] == 29

    current = results["n2_realizable_range"]["a"]["pooled"]["delta"]
    expanded = results["n2_realizable_range"]["a_plus_b"]["pooled"]["delta"]
    assert current["information_label"] == NONGT and current["division_tp"] == 1
    assert expanded["information_label"] == NONGT and expanded["division_tp"] == 38
    assert math.isclose(
        current["combined_score_delta"],
        current["adjusted_edge_jaccard_delta"] + 0.1 * current["division_jaccard"],
        abs_tol=1e-15,
    )
    shapley = results["n1_gt_dependency"]["shapley"]
    assert math.isclose(
        sum(row["combined_score_share"] for row in shapley.values()),
        ceiling["combined_score_delta"],
        abs_tol=1e-12,
    )
    labeled_numeric_leaves = sum(
        assert_all_numbers_labeled(artifact)
        for artifact in (results, dependency, curves)
    )
    assert labeled_numeric_leaves > 0

    for key, true_count in (("a", 1), ("ab", 38)):
        curve = curves["curves"][key]
        points = curve["points"]
        assert points[0]["false_forks"] == 0
        for point in points:
            assert point["information_label"] == NONGT
            assert point["true_forks"] == true_count
            assert math.isclose(
                point["precision"],
                true_count / (true_count + point["false_forks"]),
                abs_tol=1e-15,
            )
            assert math.isclose(
                point["division_jaccard"],
                true_count / (42 + point["false_forks"]),
                abs_tol=1e-15,
            )
        target = curve["requirements"]["target_combined_delta_plus_0p005"]
        if key == "a":
            assert target["status"] == "INFEASIBLE_AT_PERFECT_SELECTION"
            assert target["minimum_precision"] is None
            assert target["maximum_false_forks"] is None
            assert target["perfect_precision_payoff"] < 0.005
        else:
            assert target["status"] == "EDGE_NEUTRAL_UPPER_BOUND"
            boundary_fp = target["maximum_false_forks"]
            assert boundary_fp == points[-2]["false_forks"]
            assert points[-2]["edge_neutral_combined_score_delta"] >= 0.005
            assert points[-1]["false_forks"] == boundary_fp + 1
            assert points[-1]["edge_neutral_combined_score_delta"] < 0.005
            assert math.isclose(
                target["minimum_precision"],
                points[-2]["precision"],
                abs_tol=1e-15,
            )

    receipt = {
        "schema": "celltrack-alpha.division-value-n0-n3-validation.v1",
        "passed": True,
        "checks": 54,
        "labeled_numeric_leaves": labeled_numeric_leaves,
        "artifact_sha256": {
            name: sha(name)
            for name in (
                "N0_N3_RESULTS.json",
                "GT_DEPENDENCY_TABLE.json",
                "FP_SENSITIVITY.json",
                "RUN_PROVENANCE.json",
                "idle_preflight.json",
                "resource_guard.json",
            )
        },
    }
    (HERE / "N0_N3_VALIDATION.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
