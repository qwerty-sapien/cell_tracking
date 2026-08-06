#!/usr/bin/env python3
"""Fail-closed validator for the authentic-reverse N0 terminal bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent

EXPECTED_PINS = {
    "/Users/agent/Desktop/cell_tracking/reports/g02/screening/tiered_surface/ALIGNMENT.md":
        "eda3840ca4decf1fbc9943dcd872800852cf1ff52e0923136adb51b4f2ea6b37",
    "/Users/agent/Desktop/cell_tracking/reports/g02/screening/tiered_surface/SCOPE_MANIFEST.md":
        "066fb71b8ead255ff061a7aa75283df27c367e08aa8fd0a0b91677537a0a7d49",
    str(ROOT / "reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md"):
        "f1d2720e55cc806a3db5d036fa43e9972344e32c47c65a4720032868c79f2638",
    str(ROOT / "reports/g02/screening/tiered_surface/branches/sa_reverse/ENDPOINT_RETURN_CONTRACT.json"):
        "4eaf9d8f13eb0121d804be4252e6b53667ae65f7955c3eadf858d257804fd294",
    "/Users/agent/Desktop/cell_tracking/reports/g02/trackastra_h1_feature_cache/report.json":
        "d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf",
    "/private/tmp/trackastra-h1-fit105-cache-v1/cache_dataset_manifest.json":
        "ff7b205f58ad70c69954fed1c3e60a100d246c1e374a849c6ea7183934e98f28",
    "/private/tmp/trackastra-h1-full-cache-output/trackastra-source/LICENSE":
        "cc9ea92a72a63101ff425cc525faeace28ec9fc9c658a0fc01daf236e06326c9",
    "/Users/agent/Desktop/cell_tracking/reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/PREREGISTRATION_v3_per_class_authenticated_weight.json":
        "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
    "/Users/agent/Desktop/cell_tracking/reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_v1/population_manifest.json":
        "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    "/Users/agent/Desktop/cell_tracking/reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_v3/B0_SEALED.json":
        "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


checks: list[dict[str, object]] = []

for raw_path, expected in EXPECTED_PINS.items():
    path = Path(raw_path)
    observed = sha256(path) if path.is_file() else None
    checks.append({
        "name": f"frozen_pin:{path.name}",
        "pass": observed == expected,
        "expected": expected,
        "observed": observed,
        "path": str(path),
    })

search = json.loads((HERE / "CHECKPOINT_SEARCH_RECEIPT.json").read_text())
auth = json.loads((HERE / "CHECKPOINT_AUTHENTICATION.json").read_text())
contract = json.loads((HERE / "SUPPLIER_CONTRACT.json").read_text())
det = json.loads((HERE / "DETERMINISM_AUDIT.json").read_text())
sep = json.loads((HERE / "SEPARATION_RESULT.json").read_text())

semantic = {
    "n0_not_found": search["status"] == "EXACT_CHECKPOINT_NOT_FOUND",
    "authentication_failed_closed": (
        auth["status"] == "FAIL_EXACT_CHECKPOINT_NOT_LOCAL"
        and auth["authenticated_checkpoint"] is False
        and auth["hard_stop"]["triggered"] is True
    ),
    "supplier_not_materialized": (
        contract["materialization"]["supplier_available"] is False
        and contract["materialization"]["reverse_calls_executed"] == 0
        and contract["materialization"]["reverse_rows_emitted"] == 0
    ),
    "no_fabricated_reverse": all(
        contract["anti_fabrication"][key] is False
        for key in (
            "negate_forward_margin",
            "transpose_forward_matrix",
            "duplicate_forward_evidence",
            "scalar_recalibration",
        )
    ),
    "determinism_not_claimed": (
        det["status"] == "NOT_RUN_N0_HARD_STOP"
        and det["byte_identical_repeat"] is None
    ),
    "separation_not_claimed": (
        sep["status"] == "NOT_RUN_N0_HARD_STOP"
        and sep["all_five_folds_run"] is False
        and sep["tier_move_proposed"] is False
        and set(sep["per_class"]) == {"delete", "occupied_swap"}
    ),
    "forbidden_execution_absent": (
        search["controls"]["model_loaded"] is False
        and search["controls"]["gpu_used"] is False
        and search["controls"]["kaggle_used"] is False
        and search["controls"]["network_used"] is False
    ),
}
for name, passed in semantic.items():
    checks.append({"name": name, "pass": passed})

failed = [check for check in checks if not check["pass"]]
receipt = {
    "schema": "g02.sa-reverse-authentic.validation.v1",
    "status": "PASS" if not failed else "FAIL",
    "checks_total": len(checks),
    "checks_passed": len(checks) - len(failed),
    "checks_failed": len(failed),
    "checks": checks,
}
print(json.dumps(receipt, indent=2, sort_keys=True))
raise SystemExit(1 if failed else 0)
