#!/usr/bin/env python3
"""Independent local replay of the terminal PEFT result invariants."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
RUN = HERE / "run_v1"
KERNEL = ROOT / "kaggle_kernel/g02-peft-delete-v1/g02_peft_delete_v1.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    spec = importlib.util.spec_from_file_location("peft_validation_kernel", KERNEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    result_path = RUN / "kaggle_output/PEFT_RESULT.json"
    scores_path = RUN / "kaggle_output/OOF_SCORES.npz"
    terminal_path = RUN / "TERMINAL.json"
    step0_path = RUN / "kaggle_output/STEP0_RECEIPT.json"
    result = json.loads(result_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    step0 = json.loads(step0_path.read_text())
    data = module._load_population(RUN / "kaggle_output/peft_assets")
    sample, sample_sha = module.fixed_negative_sample(data["records"])
    with np.load(scores_path, allow_pickle=False) as payload:
        selected = np.asarray(payload["selected_scores"], dtype=np.float32)
        epochs = np.asarray(payload["epoch_scores"], dtype=np.float32)
        selected_mask = np.asarray(payload["selected_mask"], dtype=np.uint8).astype(bool)
        fold = np.asarray(payload["fold"], dtype=np.int8)

    checks: dict[str, bool] = {}
    checks["terminal_complete"] = terminal["status"] == "COMPLETE"
    checks["version_delta_one"] = terminal["version_count_delta"] == 1
    checks["submission_false"] = terminal["submission"] is False
    checks["step0_pass"] = step0["status"] == "PASS"
    checks["step0_zero_competition_visits"] = (
        step0["competition_tree_visits"] == 0
    )
    checks["step0_under_warning"] = (
        step0["elapsed_seconds"] < step0["warning_only_envelope_seconds"]
    )
    checks["sample_exact"] = (
        len(sample) == 112_918
        and sample_sha
        == "a1e1f638884112b6637c0930dd8b9345e62028d2f37b13c3d00c1f8510e51571"
    )
    checks["full_score_shapes"] = (
        selected.shape == (1_345_497,)
        and epochs.shape == (8, 1_345_497)
        and selected_mask.shape == (1_345_497,)
        and np.array_equal(fold, data["fold"])
    )
    selected_auc = module.empirical_auc(data["y"], selected)
    trajectory_auc = [
        module.empirical_auc(data["y"], epochs[index]) for index in range(8)
    ]
    checks["selected_auc_exact"] = (
        abs(selected_auc - result["capacity_probe"]["selected_auc"]) < 1e-15
    )
    checks["trajectory_auc_exact"] = all(
        abs(observed - expected["auc"]) < 1e-15
        for observed, expected in zip(
            trajectory_auc,
            result["capacity_probe"]["trajectory"],
        )
    )
    retention = module.retention_at_budget(
        data["y"], selected, data["video"], data["budgets"]
    )
    checks["retention_exact"] = (
        retention["selected_beneficial"]
        == result["capacity_probe"]["selected_retention"]["selected_beneficial"]
        and retention["selected_harmful"]
        == result["capacity_probe"]["selected_retention"]["selected_harmful"]
        and retention["retention"]
        == result["capacity_probe"]["selected_retention"]["retention"]
    )
    checks["selected_mask_count"] = (
        int(selected_mask.sum())
        == result["graph"]["aggregate"]["selected_actions"]
    )
    removed_true = int((data["delta"][selected_mask] < 0.0).sum())
    removed_false = int((data["delta"][selected_mask] > 0.0).sum())
    result_removed_true = sum(
        row["removed_true"] for row in result["graph"]["aggregate"]["videos"]
    )
    result_removed_false = sum(
        row["removed_false"] for row in result["graph"]["aggregate"]["videos"]
    )
    checks["selected_effects_exact"] = (
        removed_true == result_removed_true == 0
        and removed_false == result_removed_false == 2
    )
    delta = result["graph"]["aggregate"]["delta"]
    checks["44b6_kill_fails_only_adjusted"] = (
        delta["adjusted_edge_jaccard"] < 0.005
        and delta["raw_edge_jaccard"] >= 0.0
        and delta["node_recall"] >= 0.0
        and result["graph"]["kill_checks"]["aggregate_adjusted"] is False
        and result["graph"]["pass"] is False
        and result["status"] == "FAIL_PARK_C"
    )
    checks["capacity_case_b_frozen_rule"] = (
        selected_auc < 0.552
        and result["capacity_probe"]["case"]
        == "B_NEGLIGIBLE_MOVEMENT_BACKBONE_STOP_SIGNAL"
    )
    validation = {
        "schema": "celltrack-alpha.peft-delete-run-validation.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "identities": {
            "result_sha256": sha256(result_path),
            "oof_scores_sha256": sha256(scores_path),
            "terminal_sha256": sha256(terminal_path),
            "step0_sha256": sha256(step0_path),
        },
        "replayed": {
            "selected_auc": selected_auc,
            "trajectory_auc": trajectory_auc,
            "selected_retention": retention["retention"],
            "selected_beneficial": retention["selected_beneficial"],
            "selected_harmful": retention["selected_harmful"],
            "selected_actions": int(selected_mask.sum()),
            "removed_true": removed_true,
            "removed_false": removed_false,
            "adjusted_delta": delta["adjusted_edge_jaccard"],
            "raw_delta": delta["raw_edge_jaccard"],
            "node_delta": delta["node_recall"],
        },
        "downstream": {
            "family_6bba": "CLOSED",
            "cv57": "CLOSED",
            "transport_package": "NOT_TRIGGERED",
            "submission": "FORBIDDEN_WITHOUT_EXPLICIT_HUMAN_APPROVAL",
        },
    }
    output = HERE / "RUN_V1_VALIDATION.json"
    output.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
