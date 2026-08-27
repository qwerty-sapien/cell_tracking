"""Verdict layer for the per-class 44b6 separation gate under measured requirements.

Consumes an unmodified `evaluate_cellect_separation.py` result artifact
(schema `celltrack-alpha.cellect-separation-result.v1`) and applies the frozen
v4 measured-requirement amendment to produce a per supplier x action_class
verdict.

This module measures nothing. It re-derives a decision from banked per-fold
B0/A1/A0 AUC and CG0-budget retention. It is supplier-agnostic: DINOv2 R5,
CELLECT R7, and any later supplier emit the same result schema.

Fail-closed by construction: any missing class, missing fold, missing arm,
non-finite metric, budget violation, or identity mismatch yields INVALID for the
affected unit and never a PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

RESULT_SCHEMA = "celltrack-alpha.cellect-separation-result.v1"
AMENDMENT_SCHEMA = "celltrack-alpha.separation-gate-measured-requirement-amendment.v4"
CLASSES = ("delete", "occupied_swap")
FOLDS = tuple(range(5))
ARMS = ("B0", "A1", "A0")

PASS_TOKEN = "PASS_UNLOCK_GRAPH_PREREG_REVIEW"
FAIL_TOKEN = "FAIL_PARK_C"
INVALID_TOKEN = "INVALID_NO_SCIENTIFIC_VERDICT"


class VerdictError(Exception):
    """Raised when the inputs cannot support any scientific verdict."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VerdictError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise VerdictError(f"{label} is not finite")
    return number


def load_amendment(path: Path) -> dict[str, Any]:
    amendment = json.loads(path.read_text())
    if amendment.get("schema") != AMENDMENT_SCHEMA:
        raise VerdictError("amendment schema mismatch")
    if amendment.get("status") != "FROZEN_BEFORE_ANY_SUPPLIER_A1_A0_OUTCOME":
        raise VerdictError("amendment is not frozen pre-outcome")
    binding = amendment.get("binding_floors") or {}
    if set(binding) != set(CLASSES):
        raise VerdictError("amendment does not bind exactly the two gate classes")
    unchanged = amendment.get("unchanged_from_v3") or {}
    for key in (
        "auc_min_absolute_lift_over_B0",
        "auc_min_absolute_lift_over_A0",
        "retention_min_absolute_lift_over_B0",
        "retention_min_absolute_lift_over_A0",
    ):
        if key not in unchanged:
            raise VerdictError(f"amendment is missing carried-forward lift condition {key}")
    return amendment


def _bars(amendment: dict[str, Any], action_class: str) -> dict[str, dict[str, float]]:
    """Binding bar plus the two reported-only reference bars."""
    binding = amendment["binding_floors"][action_class]
    legacy = amendment["reference_only_alternative_bars"]["legacy_frozen_v3"]
    robust = binding["reference_only_robust_p05"]
    return {
        "binding_measured_mean": {
            "auc_absolute_floor": _finite(binding["auc_absolute_floor"], "binding auc floor"),
            "retention_absolute_floor": _finite(
                binding["retention_absolute_floor"], "binding retention floor"
            ),
        },
        "reference_legacy_frozen_v3": {
            "auc_absolute_floor": _finite(legacy["auc_absolute_floor"], "legacy auc floor"),
            "retention_absolute_floor": _finite(
                legacy["retention_absolute_floor"], "legacy retention floor"
            ),
        },
        "reference_measured_robust_p05": {
            "auc_absolute_floor": _finite(robust["auc_absolute_floor"], "robust auc floor"),
            "retention_absolute_floor": _finite(
                robust["retention_absolute_floor"], "robust retention floor"
            ),
        },
    }


def _fold_metrics(fold: dict[str, Any], action_class: str) -> dict[str, dict[str, float]]:
    arms = fold.get("arms")
    if not isinstance(arms, dict) or set(ARMS) - set(arms):
        raise VerdictError(f"{action_class} fold {fold.get('held_fold')} is missing an arm")
    out = {}
    for arm in ARMS:
        block = arms[arm]
        if not isinstance(block, dict):
            raise VerdictError(f"{action_class} arm {arm} is malformed")
        out[arm] = {
            "auc": _finite(block.get("auc"), f"{action_class}/{arm}/auc"),
            "retention": _finite(block.get("retention"), f"{action_class}/{arm}/retention"),
        }
    videos = arms["A1"].get("videos")
    if not isinstance(videos, list) or not videos:
        raise VerdictError(f"{action_class} A1 has no per-video budget rows")
    for row in videos:
        if not isinstance(row, dict) or "budget_pass" not in row:
            raise VerdictError(f"{action_class} A1 per-video row is malformed")
        if not row["budget_pass"]:
            raise VerdictError(
                f"{action_class} A1 violated the per-video CG0 budget on {row.get('identity')}"
            )
    return out


def _checks(
    metrics: dict[str, dict[str, float]],
    bar: dict[str, float],
    lifts: dict[str, float],
) -> dict[str, bool]:
    a1, b0, a0 = metrics["A1"], metrics["B0"], metrics["A0"]
    return {
        "auc_absolute_floor": a1["auc"] >= bar["auc_absolute_floor"],
        "auc_lift_over_B0": a1["auc"] - b0["auc"] >= lifts["auc_over_B0"],
        "auc_lift_over_A0": a1["auc"] - a0["auc"] >= lifts["auc_over_A0"],
        "retention_absolute_floor": a1["retention"] >= bar["retention_absolute_floor"],
        "retention_lift_over_B0": a1["retention"] - b0["retention"] >= lifts["retention_over_B0"],
        "retention_lift_over_A0": a1["retention"] - a0["retention"] >= lifts["retention_over_A0"],
    }


def evaluate_class(
    class_result: dict[str, Any],
    action_class: str,
    amendment: dict[str, Any],
) -> dict[str, Any]:
    if class_result.get("action_class") != action_class:
        raise VerdictError(f"{action_class} result identity mismatch")
    folds = class_result.get("folds")
    if not isinstance(folds, list):
        raise VerdictError(f"{action_class} has no folds")
    seen = [fold.get("held_fold") for fold in folds]
    if sorted(seen) != list(FOLDS):
        raise VerdictError(
            f"{action_class} fold coverage is {sorted(seen)}, expected {list(FOLDS)}"
        )

    unchanged = amendment["unchanged_from_v3"]
    lifts = {
        "auc_over_B0": _finite(unchanged["auc_min_absolute_lift_over_B0"], "auc lift B0"),
        "auc_over_A0": _finite(unchanged["auc_min_absolute_lift_over_A0"], "auc lift A0"),
        "retention_over_B0": _finite(
            unchanged["retention_min_absolute_lift_over_B0"], "retention lift B0"
        ),
        "retention_over_A0": _finite(
            unchanged["retention_min_absolute_lift_over_A0"], "retention lift A0"
        ),
    }
    bars = _bars(amendment, action_class)

    per_fold = []
    for fold in sorted(folds, key=lambda row: row["held_fold"]):
        metrics = _fold_metrics(fold, action_class)
        per_fold.append(
            {
                "held_fold": int(fold["held_fold"]),
                "arms": metrics,
                "lifts": {
                    "A1_minus_B0_auc": metrics["A1"]["auc"] - metrics["B0"]["auc"],
                    "A1_minus_A0_auc": metrics["A1"]["auc"] - metrics["A0"]["auc"],
                    "A1_minus_B0_retention": metrics["A1"]["retention"] - metrics["B0"]["retention"],
                    "A1_minus_A0_retention": metrics["A1"]["retention"] - metrics["A0"]["retention"],
                },
                "checks": {
                    name: _checks(metrics, bar, lifts) for name, bar in bars.items()
                },
            }
        )

    outcomes = {}
    for name in bars:
        all_pass = all(all(row["checks"][name].values()) for row in per_fold)
        failing = {
            row["held_fold"]: sorted(k for k, v in row["checks"][name].items() if not v)
            for row in per_fold
            if not all(row["checks"][name].values())
        }
        outcomes[name] = {
            "pass": all_pass,
            "verdict": PASS_TOKEN if all_pass else FAIL_TOKEN,
            "failing_conditions_by_fold": failing,
        }

    binding = outcomes["binding_measured_mean"]
    return {
        "action_class": action_class,
        "actions": class_result.get("actions"),
        "beneficial": class_result.get("beneficial"),
        "harmful": class_result.get("harmful"),
        "bars": bars,
        "folds": per_fold,
        "outcomes_by_bar": outcomes,
        "verdict": binding["verdict"],
        "pass": binding["pass"],
        "divergence_from_legacy_bar": (
            binding["pass"] != outcomes["reference_legacy_frozen_v3"]["pass"]
        ),
        "divergence_from_robust_bar": (
            binding["pass"] != outcomes["reference_measured_robust_p05"]["pass"]
        ),
    }


def build_verdict(
    result_path: Path,
    amendment_path: Path,
    supplier: str,
) -> dict[str, Any]:
    amendment = load_amendment(amendment_path)
    result = json.loads(result_path.read_text())
    if result.get("schema") != RESULT_SCHEMA:
        raise VerdictError("separation result schema mismatch")
    if result.get("status") != "COMPLETE":
        raise VerdictError("separation result is not COMPLETE")
    if not result.get("both_class_reporting_complete"):
        raise VerdictError("separation result does not report both classes")
    if result.get("cross_class_conjunction"):
        raise VerdictError("cross-class conjunction is forbidden by the frozen decision")
    results = result.get("results")
    if not isinstance(results, dict) or set(results) != set(CLASSES):
        raise VerdictError("separation result must carry exactly the two gate classes")

    units = {}
    for action_class in CLASSES:
        try:
            units[action_class] = evaluate_class(
                results[action_class], action_class, amendment
            )
        except VerdictError as error:
            units[action_class] = {
                "action_class": action_class,
                "verdict": INVALID_TOKEN,
                "pass": False,
                "reason": str(error),
            }

    return {
        "schema": "celltrack-alpha.separation-measured-verdict.v1",
        "status": "COMPLETE",
        "supplier": supplier,
        "amendment": {"path": str(amendment_path), "sha256": sha256(amendment_path)},
        "separation_result": {"path": str(result_path), "sha256": sha256(result_path)},
        "binding_bar": "binding_measured_mean",
        "verdict_unit": "supplier x action_class",
        "cross_class_conjunction": False,
        "units": units,
        "verdicts": {name: unit["verdict"] for name, unit in units.items()},
        "graph_run": False,
        "family_6bba_run": False,
        "cv57_run": False,
        "tier_move": False,
        "submission": False,
        "barrier": "HOLD_AFTER_PER_CLASS_VERDICTS_BEFORE_ANY_GRAPH_PREREGISTRATION_EXECUTION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--separation-result", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--supplier", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    verdict = build_verdict(args.separation_result, args.amendment, args.supplier)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict["verdicts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
