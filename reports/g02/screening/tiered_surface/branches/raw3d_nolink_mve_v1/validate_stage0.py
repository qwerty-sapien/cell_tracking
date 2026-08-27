#!/usr/bin/env python3
"""Independent arithmetic and authority validation for RAW3D Stage 0."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def main() -> int:
    result_path = HERE / "STAGE0_RESULT.json"
    prereg_path = HERE / "PREREGISTRATION.json"
    freeze_path = HERE / "PREREGISTRATION_FREEZE.json"
    selection_path = ROOT / "reports/g02/screening/KAGGLE_FINAL_SELECTION_20260731.json"
    result = json.loads(result_path.read_text())
    prereg = json.loads(prereg_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    selection = json.loads(selection_path.read_text())
    checks: dict[str, bool] = {}

    checks["result_sha"] = sha256(result_path) == "e651688be773a91cb7ac838a64e349951d38b450dfb607a8c0e36e25616a238a"
    checks["prereg_sha"] = sha256(prereg_path) == "5b76da90c67b412e174152965475209aa3000d633f008724fbe8e1146c8588dd"
    checks["freeze_links_prereg"] = freeze["preregistration"]["sha256"] == sha256(prereg_path)
    checks["preoutcome_status"] = prereg["status"] == "FROZEN_BEFORE_STAGE0_OUTCOME"
    checks["population"] = result["population"] == {
        "action_class": "delete",
        "actions": 1345497,
        "beneficial": 1118,
        "fold_actions": [244465, 173496, 311859, 238772, 376905],
        "fold_beneficial": [306, 99, 167, 195, 351],
        "video_count": 41,
    }
    receipts = result["feature_extraction"]["video_receipts"]
    checks["feature_receipts"] = len(receipts) == 41 and sum(
        int(row["action_count"]) for row in receipts
    ) == 1345497
    checks["feature_hashes"] = all(
        len(row["feature_sha256"]) == 64 and int(row["feature_bytes"]) > 0
        for row in receipts
    )

    folds = sorted(
        result["frozen_falsifier_result"]["folds"], key=lambda row: row["held_fold"]
    )
    aucs = [float(row["arms"]["A1"]["auc"]) for row in folds]
    retentions = [float(row["arms"]["A1"]["retention"]) for row in folds]
    a0_aucs = [float(row["arms"]["A0"]["auc"]) for row in folds]
    checks["fold_coverage"] = [row["held_fold"] for row in folds] == list(range(5))
    checks["auc_array"] = all(
        close(a, b) for a, b in zip(aucs, result["observed"]["A1_auc_by_fold"])
    )
    checks["retention_array"] = all(
        close(a, b)
        for a, b in zip(retentions, result["observed"]["A1_retention_by_fold"])
    )
    checks["auc_mean"] = close(statistics.fmean(aucs), result["observed"]["A1_auc_mean"])
    checks["auc_floor"] = close(min(aucs), result["observed"]["A1_auc_fold_floor"])
    checks["retention_mean"] = close(
        statistics.fmean(retentions), result["observed"]["A1_retention_mean"]
    )
    checks["A0_mean"] = close(statistics.fmean(a0_aucs), result["observed"]["A0_auc_mean"])
    checks["selected_beneficial"] = sum(
        int(row["arms"]["A1"]["selected_beneficial"]) for row in folds
    ) == 5
    checks["full_retention"] = close(
        5 / 1118, result["observed"]["A1_retention_full_population"]
    )

    threshold = float(prereg["advance_rule"]["binding_mean_auc_threshold"])
    checks["threshold_identity"] = close(threshold, 0.6804117434526917)
    checks["mean_fails"] = result["observed"]["A1_auc_mean"] < threshold
    checks["floor_fails"] = result["observed"]["A1_auc_fold_floor"] < threshold
    checks["advance_false"] = result["advance_rule"]["pass"] is False
    checks["verdict"] = result["verdict"] == "PARK_C_RAW3D_AND_CONSOLIDATE_CAMPAIGN"
    checks["stage1_closed"] = result["downstream_authority"]["stage1_asset_provenance_open"] is False
    checks["all_downstream_closed"] = not any(result["downstream_authority"].values())
    checks["hard_stops"] = not any(result["hard_stops"].values())
    checks["no_persistent_cache"] = (
        result["feature_extraction"]["temporary_feature_shards_deleted_after_evaluation"]
        and not result["feature_extraction"]["raw_voxel_or_crop_cache_written"]
        and not result["feature_extraction"]["full_pool_cache"]
    )
    checks["step0"] = result["step0"]["status"] == "PASS" and all(
        result["step0"]["hard_checks"].values()
    )
    checks["within_timebox"] = result["resource"]["elapsed_seconds"] < result["resource"]["timebox_seconds"]

    checks["selection_explicit"] = (
        selection["status"] == "EXPLICITLY_SELECTED_1_OF_2"
        and selection["selection"]["explicit_final_selection"]
        and not selection["selection"]["platform_default_relied_on"]
    )
    checks["selection_candidate"] = (
        selection["candidate"]["submission_ref"] == 54942864
        and selection["candidate"]["public_score"] == 0.881
        and close(selection["candidate"]["local_score"], 0.8815220699597315)
    )
    checks["selection_not_submission"] = (
        not selection["action"]["new_submission_created"]
        and selection["action"]["existing_submission_selected_as_final"]
    )
    checks["deadline"] = selection["deadline"]["kaggle_api_value"] == "2026-09-29T23:59:00"

    passed = all(checks.values())
    print(
        json.dumps(
            {
                "schema": "celltrack-alpha.raw3d-stage0-validation.v1",
                "status": "PASS" if passed else "FAIL",
                "checks_passed": sum(checks.values()),
                "checks_total": len(checks),
                "checks": checks,
                "result_sha256": sha256(result_path),
                "preregistration_sha256": sha256(prereg_path),
                "selection_receipt_sha256": sha256(selection_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
