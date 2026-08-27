#!/usr/bin/env python3
"""Validate the completed AE-P7 image-interface smoke without scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    contract = HERE / "AE_P7_FIT105_SMOKE_CONTRACT.json"
    result = HERE / "AE_P7_FIT105_SMOKE_RESULT.json"
    value = load(result)
    checks = {
        "frozen_contract": load(contract)["status"] == "FROZEN_BEFORE_EXECUTION",
        "contract_pin": value["contract_sha256"] == sha256(contract),
        "interface_pass": value["status"] == "PASS_INTERFACE_ONLY",
        "three_samples": value["sample_count"] == 3 and len(value["samples"]) == 3,
        "crop_identity": all(row["archive_extracted_equal"] and len(row["crop_sha256"]) == 2 for row in value["samples"]),
        "scope_no_score": all(value["scope"][key] is False for key in ("held_labels_read", "apncc_graph_read", "action_population_read", "fitting", "graph", "family_6bba", "cv57", "kaggle_action", "gpu_action", "budget_debit")),
        "finite_features": all(all(isinstance(item, (int, float)) for item in row["phase_features"]["shift_zyx_voxels"]) for row in value["samples"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"validation failed: {[key for key, passed in checks.items() if not passed]}")
    print(json.dumps({"schema": "celltrack-alpha.ae-p7-fit105-smoke-validation.v1", "status": "PASS", "checks": checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
