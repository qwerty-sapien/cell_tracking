#!/usr/bin/env python3
"""Validate the target-blind replacement generator preregistration."""

from __future__ import annotations

import json
from pathlib import Path


CONTRACT = Path(__file__).with_name("PREREGISTRATION.json")


def main() -> None:
    contract = json.loads(CONTRACT.read_text())
    acceptance = contract["acceptance_test"]
    forbidden = " ".join(contract["generation_inputs_before_freeze"]["forbidden"]).lower()
    stops = contract["hard_stops"]
    checks = {
        "distinct_named_mechanism": contract["mechanism_id"] != contract["predecessor_not_retried"],
        "p0_targets_61": contract["p0_target_reference"]["gt_restorable_nodes"] == 61,
        "target_comparison_only_after_freeze": contract["freeze_boundary"]["target_access_after_freeze_only"] is True,
        "gt_targets_forbidden_during_generation": "gt node coordinates" in forbidden and "p0 action labels" in forbidden,
        "literal_acceptance_test": acceptance["literal"] == "generated population contains > 0 beneficial actions with a positive oracle ceiling",
        "positive_beneficial_required": acceptance["beneficial_action_count"] == "> 0",
        "positive_oracle_required": acceptance["oracle_ceiling_adjusted_delta"] == "> 0",
        "no_selector_before_pass": stops["selector_built"] is False and acceptance["pass_consequence"] == "HOLD_FOR_HUMAN_REVIEW_BEFORE_ANY_SELECTOR_PREREGISTRATION",
        "hard_stops_preserved": all(stops[key] is False for key in ("model_fit", "graph_scoring", "family_6bba", "cv57", "kaggle", "gpu", "runpod", "submission", "tier_change", "dead_promotion")),
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
