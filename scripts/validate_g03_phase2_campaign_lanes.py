#!/usr/bin/env python3
"""Fail-closed terminal validator for the owner-amended G-03 three-lane run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LANES = ROOT / "reports/g02/screening/tiered_surface/branches/g03_orthogonal_lanes_20260803"
H005 = ROOT / "reports/g03/screening/phase2_zero_init_v1/h005"
H000_SCORE = 0.7630876414338513
HELD_FIT36_APNCC = 0.8665345260361912


class ValidationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"not a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def close(actual: Any, expected: float, *, tolerance: float = 1e-12) -> bool:
    return isinstance(actual, (int, float)) and math.isfinite(float(actual)) and math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=tolerance
    )


def validate_h005_infrastructure_terminal(checks: list[str]) -> None:
    diagnosis = load(LANES / "lane0_h005/V2_TERMINAL_DIAGNOSIS.json")
    require(
        diagnosis.get("status") == "INVALID_PRETRAINING_REMOTE_MOUNT_INTEGRITY_FAILURE",
        "h005 infrastructure-terminal status drift",
    )
    failure = diagnosis["failure"]
    require(
        failure.get("exception") == "RuntimeError: runtime dependency manifest mismatch",
        "h005 V2 failure signature drift",
    )
    require(
        failure.get("expected_manifest_sha256")
        == "7c03e1fe562e81f4d883c4b245b32550e62bd3f104f6ce1bc9037e6c5eff8781",
        "h005 V2 runtime manifest binding drift",
    )
    negative = diagnosis["negative_evidence"]
    require(negative.get("training_main_entered") is False, "h005 entered training")
    require(negative.get("cuda_model_allocation") is False, "h005 allocated CUDA model")
    require(negative.get("raw_or_geff_data_access") is False, "h005 opened training data")
    require(negative.get("optimizer_steps") == 0, "h005 performed optimizer steps")
    require(negative.get("h005_checkpoint") is False, "h005 claims a checkpoint")
    require(negative.get("held_fit36_access") is False, "h005 opened held-fit36")
    require(negative.get("cv57_access") is False, "h005 opened CV57")
    require(negative.get("remote_output_file_count") == 0, "h005 remote output drift")
    quota = diagnosis["quota"]
    require(close(quota.get("used_t4h"), 1.02), "terminal quota used drift")
    require(close(quota.get("remaining_t4h"), 28.98), "terminal quota remaining drift")
    require(close(quota.get("v2_rounded_debit_t4h"), 0.0), "V2 debit drift")
    forbidden = (
        H005 / "h005.pt",
        H005 / "PUBLICATION_RECEIPT.json",
        H005 / "evaluation_v1/BANK_SEAL.json",
        H005 / "fixed_fragments/FRAGMENT_SEAL.json",
        H005 / "FIXED_SCORE.json",
    )
    require(not any(path.exists() for path in forbidden), "post-failure h005 artifact exists")
    terminal = load(LANES / "TERMINAL_RESULT.json")
    lane0 = terminal["lane0_h005"]
    require(
        lane0.get("status") == "INFRASTRUCTURE_TERMINAL_NO_CHECKPOINT_NO_SCORE",
        "lane0 terminal join drift",
    )
    require(lane0.get("population") == "held-fit36", "lane0 population drift")
    require(close(lane0.get("h000_score"), H000_SCORE), "lane0 h000 drift")
    require(
        close(lane0.get("held_fit36_apncc_reference"), HELD_FIT36_APNCC),
        "lane0 same-population reference drift",
    )
    require(lane0.get("checkpoint_published") is False, "lane0 checkpoint claim drift")
    require(lane0.get("fixed_evaluation_launched") is False, "lane0 evaluation claim drift")
    require(lane0.get("held_fit36_access") is False, "lane0 held access drift")
    require(lane0.get("cv57_access") is False, "lane0 CV57 access drift")
    require(lane0.get("h005_score") is None and lane0.get("actual_cumulative_t4_hours") is None, "lane0 invented h005 measurement")
    require(lane0.get("slope_adjusted_graph_score_per_t4_hour") is None, "lane0 invented slope")
    require(lane0.get("h030_gate_evaluable") is False, "lane0 gate unexpectedly evaluable")
    require(lane0.get("h030_funded") is False and lane0.get("h030_launched") is False, "h030 opened")
    require(lane0.get("version_3_authorized") is False, "V3 unexpectedly authorized")
    checks.append("h005 V2 terminal before training with no checkpoint/evaluation/score")
    checks.append("same-population h030 gate correctly not evaluable and continuation closed")


def validate_h005_success(checks: list[str]) -> None:
    checkpoint = H005 / "h005.pt"
    publication = load(H005 / "PUBLICATION_RECEIPT.json")
    require(checkpoint.is_file(), "immutable h005 checkpoint absent")
    require(publication.get("status") == "PUBLISHED_IMMUTABLE_H005", "h005 publication status drift")
    require(publication["checkpoint"]["sha256"] == sha256(checkpoint), "h005 checkpoint hash drift")
    hours = float(publication["checkpoint"]["cumulative_t4_hours"])
    require(4.75 <= hours <= 5.25, "h005 cumulative hours outside fixed window")
    checks.append("immutable fixed h005 checkpoint and cumulative-hour window")

    bank_seal = load(H005 / "evaluation_v1/BANK_SEAL.json")
    require(bank_seal.get("status") == "SEALED_ALL_36_SCORE_BLIND", "h005 bank seal incomplete")
    require(bank_seal.get("population", {}).get("name") == "held-fit36", "h005 bank population drift")
    require(bank_seal.get("gt_access") is False and bank_seal.get("score_access") is False, "remote score leak")
    require(bank_seal.get("cv57_access") is False, "CV57 opened in h005 evaluation")
    require(bank_seal["checkpoint"]["sha256"] == sha256(checkpoint), "evaluation checkpoint drift")
    require(len(bank_seal.get("results", [])) == 36, "h005 bank count drift")
    checks.append("36/36 score-blind CUDA banks bound to h005")

    fragment_seal = load(H005 / "fixed_fragments/FRAGMENT_SEAL.json")
    require(
        fragment_seal.get("status") == "SEALED_36_FIXED_H005_FRAGMENTS_SCORE_BLIND",
        "fixed fragment seal incomplete",
    )
    require(fragment_seal.get("held_fit_score_read") is False, "fragment seal was not score blind")
    require(fragment_seal.get("cv57_access") is False, "fragment seal touched CV57")
    require(len(fragment_seal.get("videos", [])) == 36, "fragment count drift")
    for row in fragment_seal["videos"]:
        path = ROOT / row["fragment_path"] if not Path(row["fragment_path"]).is_absolute() else Path(row["fragment_path"])
        require(path.is_file() and sha256(path) == row["fragment_sha256"], f"fragment hash drift: {row['identity']}")
    checks.append("36/36 fixed graph fragments independently rehashed")

    score = load(H005 / "FIXED_SCORE.json")
    require(score.get("status") == "COMPLETE_FIXED_H005_HELD_FIT36_MEASUREMENT", "h005 score status drift")
    require(score.get("population") == {"identity_count": 36, "name": "held-fit36"}, "score population drift")
    gate = score["trajectory_gate"]
    observed = float(score["candidate"]["score"])
    require(close(gate["h000_score"], H000_SCORE), "h000 score mixed or drifted")
    require(close(gate["h005_score"], observed), "h005 gate score drift")
    require(close(gate["actual_cumulative_t4_hours"], hours), "h005 hours drift")
    slope = (observed - H000_SCORE) / hours
    projection = H000_SCORE + 30.0 * slope
    require(close(gate["slope_adjusted_graph_score_per_t4_hour"], slope), "slope arithmetic drift")
    require(close(gate["linear_projection_at_30_t4_hours"], projection), "projection arithmetic drift")
    require(gate["reference"] == {"name": "APNCC held-fit36", "score": HELD_FIT36_APNCC}, "reference drift")
    passed = projection + float(gate["projection_absolute_tolerance"]) >= HELD_FIT36_APNCC
    require(gate["passes_h030_funding_gate"] is passed, "h030 gate decision drift")
    require(gate.get("cross_population_subtraction") is False and gate.get("cv57_access") is False, "population mixing")
    checks.append("same-population h000-to-h005 slope and held-fit36 APNCC decision recomputed")


def main() -> int:
    started = time.perf_counter()
    checks: list[str] = []

    freeze = load(LANES / "FREEZE_RECEIPT.json")
    require(freeze.get("status") == "SEALED_BEFORE_NEW_LANE_OUTCOMES", "freeze status drift")
    for name, row in freeze["artifacts"].items():
        path = ROOT / row["path"]
        require(path.is_file() and sha256(path) == row["sha256"], f"freeze hash drift: {name}")
    checks.append("owner amendment plus three preregistrations pre-outcome hash exact")

    lane1 = load(LANES / "lane1_dense_motion/run_v1/SEPARATION_RESULT.json")
    features = load(LANES / "lane1_dense_motion/run_v1/FEATURE_MANIFEST.json")
    require(lane1.get("status") == "COMPLETE", "lane1 is not terminal")
    require(lane1.get("execution_order") == ["B0_SEALED", "A1", "A0"], "lane1 order drift")
    require(features.get("aggregate", {}).get("identity_count") == 41, "lane1 population drift")
    require(features.get("aggregate", {}).get("action_count") == 1_391_458, "lane1 action drift")
    require(lane1.get("sealed_B0", {}).get("sha256") == "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2", "lane1 B0 drift")
    require(set(lane1.get("results", {})) == {"delete", "occupied_swap"}, "lane1 class drift")
    for action_class, result in lane1["results"].items():
        require(result.get("verdict") == "FAIL_PARK_C" and result.get("pass") is False, f"lane1 {action_class} verdict drift")
        require(len(result.get("folds", [])) == 5, f"lane1 {action_class} fold drift")
        require(all(fold.get("pass") is False for fold in result["folds"]), f"lane1 {action_class} unexpected fold pass")
    execution = lane1["execution"]
    require(features.get("gpu_hours") == 0.0, "lane1 used GPU")
    require(execution.get("graph") is False, "lane1 opened graph integration")
    require(execution.get("family_6bba") is False and execution.get("cv57") is False, "lane1 population leak")
    checks.append("dense motion terminal on frozen 41-video B0/A1/A0 consumer")

    lane2 = load(LANES / "lane2_trained_fork/run_v1/CROSS_FAMILY_RESULT.json")
    require(lane2.get("status") == "FAIL_CROSS_FAMILY_STOP_BEFORE_FINAL_38", "lane2 status drift")
    require(lane2["train"]["family"] == "6bba", "lane2 train family drift")
    require(lane2["validation"]["family"] == "44b6", "lane2 validation family drift")
    require(lane2.get("final_38_access") is False, "lane2 accessed final 38")
    require(lane2.get("graph_integration") is False, "lane2 integrated graph")
    require(lane2.get("gpu_hours") == 0.0, "lane2 used GPU")
    metrics = lane2["validation"]["metrics"]
    require(
        (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (5, 6, 4, 30),
        "lane2 operating point drift",
    )
    checks.append("trained fork cross-family kill before final 38/graph/GPU")

    if (LANES / "lane0_h005/V2_TERMINAL_DIAGNOSIS.json").is_file():
        validate_h005_infrastructure_terminal(checks)
    else:
        validate_h005_success(checks)

    report = load(LANES / "TERMINAL_RESULT.json")
    require(report.get("standalone_scoring_only") is True, "standalone constraint drift")
    require(report.get("hybridization") is False, "forbidden hybridization")
    require(report.get("runpod") is False and report.get("submission") is False, "forbidden external action")
    checks.append("terminal join preserves standalone/no-hybrid/no-submission constraints")

    result = {
        "schema": "celltrack-alpha.g03-phase2-campaign-lanes-validation.v1",
        "status": "PASS",
        "checks": checks,
        "check_count": len(checks),
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
