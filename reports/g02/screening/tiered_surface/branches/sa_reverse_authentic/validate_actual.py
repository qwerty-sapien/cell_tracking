#!/usr/bin/env python3
"""Fail-closed terminal validator for the authentic reverse Trackastra lane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
EXPECTED = {
    "model": "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f",
    "archive": "2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e",
    "endpoint": "4eaf9d8f13eb0121d804be4252e6b53667ae65f7955c3eadf858d257804fd294",
    "supplier_manifest": "654cb499d65db137d71e3915850d4bc94a5373660cb2026507ad070275fb6fe1",
    "feature_manifest": "a1ede36b2f2e081709c711e7c34da0c5ca2cf4168be29fc7426f39985b13871c",
    "population": "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    "prereg": "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
    "b0": "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
    "raw_result": "bd1c3d553bdfbf8ce3547ca87e346a485d96c3698d1918540263eadf68bbf46e",
    "tier_census": "4c7b6735cb2b03abdae3e1780e7b207a6e14ea237e199126ca47e7a9de944aa7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AssertionError(f"not an object: {path}")
    return value


def main() -> int:
    checks: dict[str, bool] = {}

    checkpoint = load(HERE / "CHECKPOINT_AUTHENTICATION.json")
    checks["n0_exact_assets"] = (
        checkpoint["status"] == "PASS_EXACT_HISTORICAL_ASSETS"
        and checkpoint["required_checkpoint_sha256"] == EXPECTED["model"]
        and checkpoint["required_release_archive_sha256"] == EXPECTED["archive"]
        and checkpoint["content_hash_resolution"]["archive_member_equals_standalone"]
        and not checkpoint["list_dataset_files_called"]
    )

    manifest_path = HERE / "reverse_cache/manifest.json"
    manifest = load(manifest_path)
    checks["supplier_manifest_pin"] = sha256(manifest_path) == EXPECTED["supplier_manifest"]
    checks["supplier_exact_scope"] = (
        manifest["status"] == "COMPLETE"
        and manifest["classification"] == "NON-GT-IDENTIFIABLE"
        and manifest["totals"]
        == {
            "cropped_calls": 2430,
            "inference_calls": 3539,
            "peak_tokens": 1024,
            "proposal_count": 84883,
            "row_count": 121534,
            "video_count": 36,
        }
    )
    checks["supplier_repeat"] = (
        manifest["determinism"]["identity"] == "train:44b6_0db75fae"
        and manifest["determinism"]["inference_calls"] == 99
        and manifest["determinism"]["all_array_equal"] is True
    )
    total_rows = total_proposals = total_transitions = 0
    supplier_files_ok = len(manifest["files"]) == 36
    for row in manifest["files"]:
        path = HERE / "reverse_cache" / row["path"]
        supplier_files_ok &= path.is_file() and sha256(path) == row["sha256"]
        with np.load(path, allow_pickle=False) as z:
            supplier_files_ok &= (
                len(z["forward_margin"]) == row["row_count"]
                and len(z["challengers"]) == row["proposal_count"]
                and len(z["transition_target_t"]) == row["inference_calls"]
                and np.isfinite(z["forward_margin"]).all()
                and np.isfinite(z["reverse_margin"]).all()
                and np.isfinite(z["reverse_challenger_null"]).all()
                and np.isfinite(z["reverse_conflict_null"]).all()
            )
        total_rows += int(row["row_count"])
        total_proposals += int(row["proposal_count"])
        total_transitions += int(row["inference_calls"])
    checks["supplier_files"] = supplier_files_ok
    checks["supplier_file_totals"] = (
        total_rows,
        total_proposals,
        total_transitions,
    ) == (121534, 84883, 3539)
    checks["authentic_reverse_contract"] = (
        manifest["pins"]["endpoint_contract_sha256"] == EXPECTED["endpoint"]
        and manifest["prohibited_shortcuts_used"] is False
        and "A_reverse[index(v),index(u)]" in manifest["endpoint_return"]
    )
    checks["supplier_forbidden_actions"] = manifest["execution"] == {
        "cpu_only": True,
        "cv57": False,
        "device": "cpu",
        "elapsed_seconds": manifest["execution"]["elapsed_seconds"],
        "family_6bba": False,
        "gpu": False,
        "graph": False,
        "kaggle_kernel": False,
        "labels_read": False,
        "training": False,
    }

    feature_path = HERE / "separation_features/feature_manifest.json"
    feature = load(feature_path)
    checks["feature_manifest_pin"] = sha256(feature_path) == EXPECTED["feature_manifest"]
    checks["feature_scope"] = (
        feature["status"] == "COMPLETE"
        and feature["classification"] == "NON-GT-IDENTIFIABLE"
        and feature["amendment_sha256"] == EXPECTED["prereg"]
        and feature["population_manifest_sha256"] == EXPECTED["population"]
        and feature["supplier_manifest_sha256"] == EXPECTED["supplier_manifest"]
        and feature["labels_read"] is False
        and feature["totals"]["delete"]
        == {
            "actions": 1345497,
            "supported": 109569,
            "unsupported_zero_evidence": 1235928,
        }
        and feature["totals"]["occupied_swap"]
        == {
            "actions": 45961,
            "supported": 45961,
            "unsupported_zero_evidence": 0,
        }
    )
    feature_files_ok = len(feature["files"]) == 77
    for row in feature["files"]:
        path = HERE / "separation_features" / row["path"]
        feature_files_ok &= path.is_file() and sha256(path) == row["sha256"]
        with np.load(path, allow_pickle=False) as z:
            feature_files_ok &= (
                z["a1"].shape == (row["count"], 8)
                and z["a0"].shape == (row["count"], 8)
                and np.isfinite(z["a1"]).all()
                and np.isfinite(z["a0"]).all()
                and int(z["a1"][:, -1].sum()) == row["supported"]
            )
    checks["feature_files"] = feature_files_ok

    raw_path = HERE / "SEPARATION_RESULT_ACTUAL.json"
    raw = load(raw_path)
    checks["raw_result_pin"] = sha256(raw_path) == EXPECTED["raw_result"]
    checks["frozen_gate"] = (
        raw["status"] == "COMPLETE"
        and raw["amendment_sha256"] == EXPECTED["prereg"]
        and raw["sealed_B0"]["sha256"] == EXPECTED["b0"]
        and raw["feature_manifest"]["sha256"] == EXPECTED["feature_manifest"]
        and raw["both_class_reporting_complete"] is True
        and raw["cross_class_conjunction"] is False
    )
    checks["both_classes_fail_all_folds"] = all(
        result["verdict"] == "FAIL_PARK_C"
        and result["pass"] is False
        and len(result["folds"]) == 5
        and all(fold["pass"] is False for fold in result["folds"])
        for result in raw["results"].values()
    ) and set(raw["results"]) == {"delete", "occupied_swap"}
    checks["no_separation_graph"] = raw["execution"] == {
        "cv57": False,
        "elapsed_seconds": raw["execution"]["elapsed_seconds"],
        "family_6bba": False,
        "full_pool_cache": False,
        "graph": False,
        "runpod": False,
        "submission": False,
    }

    state = load(HERE / "state_fragment.json")
    checks["state_no_tier_move"] = (
        state["verdict"] == "SUPPLIER_PASS_SEPARATION_FAIL_BOTH_HALT_AT_BARRIER"
        and state["tier_moves_proposed"] == 0
        and state["tier_moves_applied"] == 0
        and state["graph_unlock"] is False
        and state["resource_and_action_accounting"]["tier_census_write"] is False
    )
    checks["root_live_state"] = (
        (ROOT / "reports/g02/screening/LIVE_STATE.md")
        .read_text()
        .count("AUTHENTIC REVERSE TRACKASTRA TERMINAL / NO GRAPH UNLOCK")
        == 1
    )
    checks["root_ledger"] = (
        (ROOT / "reports/g02/screening/LEDGER.md")
        .read_text()
        .count("REVERSE-AUTHENTIC-N0-N3-20260729")
        == 1
    )
    checks["tier_census_unchanged"] = (
        sha256(ROOT / "reports/g02/screening/tiered_surface/tier_census.json")
        == EXPECTED["tier_census"]
    )

    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "g02.sa-reverse-authentic.validation.v2",
        "status": "PASS" if not failed else "FAIL",
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failed),
        "checks_failed": len(failed),
        "checks": checks,
        "failed": failed,
    }
    output = HERE / "VALIDATION_RECEIPT_ACTUAL.json"
    if output.exists():
        output.unlink()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
