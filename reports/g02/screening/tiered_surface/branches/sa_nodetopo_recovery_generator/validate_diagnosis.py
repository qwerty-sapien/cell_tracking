#!/usr/bin/env python3
"""Validate that NODETOPO's causal correction is grounded in run_v6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
DIAGNOSIS = HERE / "NODETOPO_GENERATOR_DIAGNOSIS_20260731.json"
RUN = ROOT / "reports/g02/screening/selector_requirement_44b6/run_v6/RESULT.json"
CENSUS = ROOT / "reports/g02/screening/tiered_surface/tier_census.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    diagnosis = json.loads(DIAGNOSIS.read_text())
    run = json.loads(RUN.read_text())
    census = json.loads(CENSUS.read_text())
    node = next(item for item in census["nodes"] if item["id"] == "NODETOPO-O0")
    restored = run["population"]["restore_topology"]
    checks = {
        "run_v6_result_byte_pinned": sha256(RUN) == diagnosis["selector_requirement_counterevidence"]["sha256"],
        "generated_population_has_zero_beneficial": restored["actions"] == 50 and restored["beneficial"] == 0,
        "oracle_ceiling_is_exactly_zero": diagnosis["selector_requirement_counterevidence"]["oracle_ceiling_adjusted_delta"] == 0.0,
        "diagnosis_is_generator_not_selector": diagnosis["status"] == "GENERATOR_FAILURE_NOT_SELECTOR_FAILURE",
        "census_uses_diagnosis": node["evidence_artifact"] == "reports/g02/screening/tiered_surface/branches/sa_nodetopo_recovery_generator/NODETOPO_GENERATOR_DIAGNOSIS_20260731.json" and node["evidence_sha256"] == sha256(DIAGNOSIS),
        "tier_unchanged": node["tier"] == "PARK_A",
        "selector_stays_closed": diagnosis["consequence"]["selector_authorized"] is False,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
