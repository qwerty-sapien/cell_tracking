"""Fail-closed integrity checks for the 2026-07-31 CPU barrier merge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def main() -> int:
    merge_path = HERE / "BARRIER_MERGE.json"
    merge = json.loads(merge_path.read_text())
    reverse = load(merge["reverse_supplier"]["artifact"])
    reverse_validation = load(merge["reverse_supplier"]["validation_receipt"])
    p5 = load(merge["ae_p5"]["result"])
    p7 = load(merge["ae_p7"]["result"])
    selector = load(merge["dag2_selector"]["result"])
    surface = load(merge["dag2_selector"]["requirement_surface"])
    review_validation = load(merge["dag2_selector"]["review_validation"])
    review = load(merge["dag2_selector"]["review"])
    census = load("reports/g02/screening/tiered_surface/tier_census.json")
    nodes = {row["id"]: row for row in census["nodes"]}
    checks = {
        "merge_status": merge["status"] == "COMPLETE_NO_TIER_MOVE"
        and merge["proposed_tier_moves"] == [],
        "reverse_result_pin": sha256(ROOT / merge["reverse_supplier"]["artifact"])
        == merge["reverse_supplier"]["sha256"],
        "reverse_source_receipt": sha256(
            ROOT / merge["reverse_supplier"]["validation_receipt"]
        )
        == merge["reverse_supplier"]["validation_receipt_sha256"]
        and reverse_validation["status"] == "PASS"
        and reverse_validation["checks_passed"] == 19,
        "reverse_both_classes_fail": reverse["status"] == "COMPLETE"
        and all(value["verdict"] == "FAIL_PARK_C" for value in reverse["results"].values()),
        "p5_result_pin": sha256(ROOT / merge["ae_p5"]["result"])
        == merge["ae_p5"]["result_sha256"],
        "p5_complete_ownership_falsifier": p5["totals"]["actions"] == 45961
        and p5["totals"]["components"] == p5["totals"]["components_fallback"]
        and p5["totals"]["actions_selected"] == 0
        and p5["totals"]["n1_batch_compiles"] == 361
        and p5["totals"]["n1_batch_rejections"] == 0,
        "p7_result_pin": sha256(ROOT / merge["ae_p7"]["result"])
        == merge["ae_p7"]["result_sha256"],
        "p7_interface_only": p7["status"] == "PASS_INTERFACE_ONLY"
        and p7["sample_count"] == 3
        and p7["scope"]["graph"] is False,
        "selector_result_pin": sha256(ROOT / merge["dag2_selector"]["result"])
        == merge["dag2_selector"]["result_sha256"],
        "selector_surface_pin_and_complete": sha256(
            ROOT / merge["dag2_selector"]["requirement_surface"]
        )
        == merge["dag2_selector"]["requirement_surface_sha256"]
        and surface["status"] == "COMPLETE"
        and selector["status"] == "HUMAN_REVIEW_REQUIRED",
        "selector_independent_review": review_validation["status"] == "PASS"
        and review_validation["checks_passed"] == 8
        and review["status"] == "PASS"
        and review_validation["tier_moves"] == 0,
        "census_no_tier_move": {
            key: nodes[key]["tier"] for key in merge["current_tiers"]
        }
        == merge["current_tiers"],
    }
    payload = {
        "schema": "g02.cpu-barrier-merge-validation.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "merge_sha256": sha256(merge_path),
        "checks": checks,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
