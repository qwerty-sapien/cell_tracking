#!/usr/bin/env python3
"""Refine the banked P0 division oracle without fitting or held execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("/Users/agent/Desktop/cell_tracking")
BRANCH = Path(__file__).resolve().parent

PINNED = {
    "n1_interface": (
        ROOT
        / "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json",
        "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    ),
    "n1_implementation": (
        ROOT / "src/celltrack_alpha/action_compiler.py",
        "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
    ),
    "grouped_folds": (
        ROOT / "reports/g02/grouped_3fold_audit20_v1.json",
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    ),
    "p0": (
        ROOT
        / "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json",
        "8ad548c134c419e2e76b8661848e90493d674731ceb34ab231c5f18c8889c06",
    ),
    "d1_preregistration": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/"
        "d1_persistent_fork/N0_preregistration.json",
        "34b2aa020099e14ad7a73820d97e9baefadec386e0e2d9acbd7ab5c1cb07620f",
    ),
    "d1_arm_manifest": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/"
        "d1_persistent_fork/N1_arm_manifest.json",
        "327af0d29dbde19172a61bf396b951d150040bb357c3fbba90e79249035e4776",
    ),
    "dependency_audit": (
        BRANCH / "guarded_results.json",
        "38990d30d4b30ea7afbf99e0a655b0150c34a73fdd4918f93178316d9d9e3c76",
    ),
}

EXPECTED_ARMS = [
    "UNCHANGED_GRAPH",
    "D0_PARENTAL_SOFTMAX_GREEDY_FORK",
    "D1_LATE_PERSISTENT_ONE_TO_ONE_FORK",
    "D1_PERSISTENCE_ABLATED",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError(f"{label} drift: {observed} != {expected}")


def authenticate() -> dict[str, dict[str, Any]]:
    authenticated: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in PINNED.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError(f"pin mismatch for {name}: {observed}")
        authenticated[name] = {
            "path": str(path),
            "sha256": observed,
            "match": True,
        }
    return authenticated


def assert_dependency_closed() -> dict[str, Any]:
    audit = load_json(PINNED["dependency_audit"][0])
    dependency = audit["dependency"]
    if (
        dependency["qualifying_arm_count"] != 0
        or dependency["verdict"] != "PARK_DEPENDENCY"
        or dependency["held_family_execution"]
        or audit["execution"]["model_fitted"]
        or audit["execution"]["family_44b6_ran"]
        or audit["execution"]["family_6bba_ran"]
        or audit["execution"]["cv57_used"]
    ):
        raise AssertionError("association-continuity dependency is not safely closed")
    return {
        "qualifying_arm_count": 0,
        "verdict": "PARK_DEPENDENCY",
        "fitting_gate_open": False,
    }


def assert_d1_unchanged() -> dict[str, Any]:
    prereg = load_json(PINNED["d1_preregistration"][0])
    manifest = load_json(PINNED["d1_arm_manifest"][0])
    thresholds = prereg["thresholds"]
    if prereg["arms"] != EXPECTED_ARMS or list(manifest["arms"]) != EXPECTED_ARMS:
        raise AssertionError("D1 arm identities drifted")
    if (
        thresholds["dependency_min_adjusted_delta_each_family"] != 0.005
        or thresholds["dependency_min_raw_delta_each_family"] != 0.0
        or thresholds["dependency_min_node_recall_delta_each_family"] != 0.0
        or not thresholds["dependency_requires_positive_connected_fork_opportunity_delta"]
        or thresholds["d1_go_min_division_jaccard"] != 0.30
        or thresholds["d1_dead_max_division_jaccard"] != 0.15
        or thresholds["max_duplicate_survivors"] != 0
        or prereg["budget"]["held_graph_runs_before_dependency_go"] != 0
        or prereg["budget"]["gpu"]
        or prereg["budget"]["cv57"]
        or manifest["training_performed"]
        or manifest["held_fit_graph_access"]
        or manifest["cv57_access"]
        or manifest["gpu_action"]
    ):
        raise AssertionError("D1 frozen threshold or execution boundary drifted")
    return {
        "arms": EXPECTED_ARMS,
        "go_min_division_jaccard": 0.30,
        "null_max_division_jaccard": 0.15,
        "max_duplicate_survivors": 0,
        "unchanged": True,
    }


def refine_family(
    family: str,
    family_row: dict[str, Any],
    go_threshold: float,
    null_threshold: float,
) -> dict[str, Any]:
    control = family_row["control"]
    action = family_row["actions"]["division_fork"]
    oracle = action["oracle"]
    reported_delta = action["delta_vs_control"]
    counts = action["action_counts"]

    recomputed_delta = {
        "adjusted_edge_jaccard": (
            oracle["adj_edge_jaccard"] - control["adj_edge_jaccard"]
        ),
        "raw_edge_jaccard": oracle["edge_jaccard"] - control["edge_jaccard"],
        "node_recall": oracle["node_recall"] - control["node_recall"],
        "division_jaccard": (
            oracle["division_jaccard"] - control["division_jaccard"]
        ),
        "score": (
            oracle["adj_edge_jaccard"] + 0.1 * oracle["division_jaccard"]
        )
        - (control["adj_edge_jaccard"] + 0.1 * control["division_jaccard"]),
    }
    report_keys = {
        "adjusted_edge_jaccard": "adj_edge_jaccard",
        "raw_edge_jaccard": "edge_jaccard",
        "node_recall": "node_recall",
        "division_jaccard": "division_jaccard",
        "score": "score",
    }
    for key, report_key in report_keys.items():
        assert_close(
            recomputed_delta[key],
            reported_delta[report_key],
            f"{family}.{key}",
        )
    assert_close(
        oracle["score"],
        oracle["adj_edge_jaccard"] + 0.1 * oracle["division_jaccard"],
        f"{family}.oracle_score",
    )
    assert_close(
        control["score"],
        control["adj_edge_jaccard"] + 0.1 * control["division_jaccard"],
        f"{family}.control_score",
    )

    positive = counts["positive_actions"]
    background = counts["raw_background_actions"]
    if (
        counts["background_definition"]
        != "non-GT daughter pairs among each source's r10/k8 next-frame candidates"
        or positive <= 0
        or background <= 0
    ):
        raise AssertionError(f"{family} division population drifted")

    tp = oracle["division_tp"]
    fp = oracle["division_fp"]
    fn = oracle["division_fn"]
    maximum_fp_for_go = math.floor(tp / go_threshold - tp - fn)
    minimum_fp_reduction_for_go = max(0, fp - maximum_fp_for_go)

    return {
        "family": family,
        "control": {
            "adjusted_edge_jaccard": control["adj_edge_jaccard"],
            "raw_edge_jaccard": control["edge_jaccard"],
            "node_recall": control["node_recall"],
            "division_tp": control["division_tp"],
            "division_fp": control["division_fp"],
            "division_fn": control["division_fn"],
            "division_jaccard": control["division_jaccard"],
            "score": control["score"],
        },
        "oracle": {
            "adjusted_edge_jaccard": oracle["adj_edge_jaccard"],
            "raw_edge_jaccard": oracle["edge_jaccard"],
            "node_recall": oracle["node_recall"],
            "division_tp": tp,
            "division_fp": fp,
            "division_fn": fn,
            "division_jaccard": oracle["division_jaccard"],
            "score": oracle["score"],
        },
        "delta_vs_control_recomputed": recomputed_delta,
        "population_support": {
            "positive_exact_actions": positive,
            "non_gt_daughter_pair_background": background,
            "positive_fraction_diagnostic_only": positive / (positive + background),
            "probability_or_promotion_gate": False,
        },
        "d1_threshold_envelope": {
            "go_min_division_jaccard": go_threshold,
            "null_max_division_jaccard": null_threshold,
            "oracle_meets_go": oracle["division_jaccard"] >= go_threshold,
            "oracle_within_null": oracle["division_jaccard"] <= null_threshold,
            "maximum_integer_division_fp_for_go_holding_tp_fn": maximum_fp_for_go,
            "minimum_division_fp_reduction_for_go_holding_tp_fn": (
                minimum_fp_reduction_for_go
            ),
            "counterfactual_is_diagnostic_only": True,
        },
    }


def run_refinement() -> dict[str, Any]:
    authenticated = authenticate()
    dependency = assert_dependency_closed()
    d1 = assert_d1_unchanged()

    p0 = load_json(PINNED["p0"][0])
    if (
        p0["control"]["name"] != "fit117 c=0.25 fragment graph used by CG0"
        or p0["control"]["apncc_incumbent"]
        or p0["control"]["common_across_actions"] is not True
        or set(p0["families"]) != {"44b6", "6bba"}
    ):
        raise AssertionError("P0 control or family identity drifted")

    family_results = [
        refine_family(
            family,
            p0["families"][family],
            d1["go_min_division_jaccard"],
            d1["null_max_division_jaccard"],
        )
        for family in ("44b6", "6bba")
    ]

    return {
        "schema": "celltrack-alpha.sa-division-oracle-refinement.v1",
        "status": "PASS_DIAGNOSTIC_PARK_DEPENDENCY",
        "dependency": dependency,
        "d1_preregistered_arm": d1,
        "p0_control": {
            "name": p0["control"]["name"],
            "apncc_incumbent": False,
            "diagnostic_not_forecast": True,
        },
        "families": family_results,
        "conclusion": {
            "both_family_oracle_meets_d1_go": all(
                row["d1_threshold_envelope"]["oracle_meets_go"]
                for row in family_results
            ),
            "fitting_authorized": False,
            "tier_move_proposed": False,
            "diagnostics_promote": False,
            "verdict": "PARK_DEPENDENCY",
        },
        "authenticated_inputs": authenticated,
        "execution": {
            "model_fitted": False,
            "training": False,
            "held_graph_execution": False,
            "family_44b6_ran": False,
            "family_6bba_ran": False,
            "cv57_used": False,
            "gpu_action": False,
            "kaggle_action": False,
            "submission_action": False,
            "runpod_action": False,
            "external_action": False,
            "plan_amendment": False,
            "dead_action": False,
        },
        "memo_quarantine": True,
    }


def self_test(result: dict[str, Any]) -> dict[str, int]:
    families = {row["family"]: row for row in result["families"]}
    checks = [
        result["status"] == "PASS_DIAGNOSTIC_PARK_DEPENDENCY",
        result["dependency"]["qualifying_arm_count"] == 0,
        not result["conclusion"]["fitting_authorized"],
        not result["conclusion"]["tier_move_proposed"],
        set(families) == {"44b6", "6bba"},
        families["44b6"]["d1_threshold_envelope"][
            "minimum_division_fp_reduction_for_go_holding_tp_fn"
        ]
        == 39,
        families["6bba"]["d1_threshold_envelope"][
            "minimum_division_fp_reduction_for_go_holding_tp_fn"
        ]
        == 82,
        not result["conclusion"]["both_family_oracle_meets_d1_go"],
        not any(result["execution"].values()),
        result["memo_quarantine"],
    ]
    if not all(checks):
        failed = [index for index, passed in enumerate(checks, 1) if not passed]
        raise AssertionError(f"self-test failures: {failed}")
    return {"passed": len(checks), "total": len(checks)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    result = run_refinement()
    result["tests"] = self_test(result)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({"status": result["status"], "tests": result["tests"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
