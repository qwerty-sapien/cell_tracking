"""Freeze the measured-requirement separation gate amendment (v4).

Human authority, 2026-07-30: adopt the run_v6 measured `+0.005` mean requirements
as the binding absolute floors of the per-class separation gate, replacing the
preregistered v3 absolute floors of AUC 0.65 / retention 0.15.

This script only reads frozen artifacts and writes the amendment record. It opens
no supplier feature, B0, A1, A0, metric, verdict, graph, or tier outcome.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE.parent
ROOT = GATE.parents[6]

REQUIREMENT_SURFACE = ROOT / "reports/g02/screening/selector_requirement_44b6/run_v6/requirement_surface.json"
PREREG_V3 = GATE / "PREREGISTRATION_v3_per_class_authenticated_weight.json"
EVALUATOR = GATE / "evaluate_cellect_separation.py"

# Frozen v3 absolute floors that this amendment supersedes.
FROZEN_AUC_FLOOR = 0.65
FROZEN_RETENTION_FLOOR = 0.15

CLASSES = ("delete", "occupied_swap")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    surface = json.loads(REQUIREMENT_SURFACE.read_text())
    if surface.get("status") != "COMPLETE":
        raise SystemExit("requirement surface is not COMPLETE")
    if not surface.get("grid_only") or surface.get("interpolation"):
        raise SystemExit("requirement surface must be grid-only without interpolation")

    binding = {}
    for action_class in CLASSES:
        entry = surface["requirements"][action_class]["target_plus_0_005"]
        if entry.get("status") != "MEASURED_ON_TESTED_GRID":
            raise SystemExit(f"{action_class} +0.005 requirement is not measured on the tested grid")
        if entry.get("threshold") != 0.005:
            raise SystemExit(f"{action_class} requirement threshold is not +0.005")
        mean = entry["mean_requirement"]
        robust = entry["robust_p05_requirement"]
        binding[action_class] = {
            "auc_absolute_floor": float(mean["target_auc"]),
            "retention_absolute_floor": float(mean["retention"]["mean"]),
            "measured_at_candidate_prefix_N": int(mean["candidate_prefix_N"]),
            "oracle_ceiling": float(entry["oracle_ceiling"]),
            "oracle_best_support": int(entry["oracle_best_support"]),
            "reference_only_robust_p05": {
                "auc_absolute_floor": float(robust["target_auc"]),
                "retention_absolute_floor": float(robust["retention"]["mean"]),
            },
            "superseded_frozen_floors": {
                "auc_absolute_floor": FROZEN_AUC_FLOOR,
                "retention_absolute_floor": FROZEN_RETENTION_FLOOR,
            },
            "direction_of_change": {
                "auc": "TIGHTENED",
                "retention": (
                    "RELAXED"
                    if float(mean["retention"]["mean"]) < FROZEN_RETENTION_FLOOR
                    else "TIGHTENED"
                ),
            },
        }

    amendment = {
        "schema": "celltrack-alpha.separation-gate-measured-requirement-amendment.v4",
        "status": "FROZEN_BEFORE_ANY_SUPPLIER_A1_A0_OUTCOME",
        "frozen_utc": "2026-07-30T00:00:00Z",
        "name": "R-SEPARATION-MEASURED-REQUIREMENT-44B6-V4",
        "authority": {
            "source": "human decision, 2026-07-30",
            "decision": "Apply the run_v6 measured +0.005 mean requirements as the binding absolute floors.",
            "rationale": (
                "The preregistered v3 absolute floors were chosen before any requirement "
                "surface existed. run_v6 measured, on the frozen fit-only 44b6 population, "
                "what a selector must actually achieve to reach the +0.005 actionability "
                "target. The frozen floors carry verdict TOO_LENIENT for both feasible "
                "classes. Binding the measured values makes the gate calibrated rather "
                "than arbitrary, and it is fixed before any supplier outcome exists."
            ),
        },
        "supersedes": {
            "preregistration": {
                "path": str(PREREG_V3.relative_to(ROOT)),
                "sha256": sha256(PREREG_V3),
                "superseded_fields": [
                    "pass_gate.auc_absolute_floor",
                    "pass_gate.retention_absolute_floor",
                ],
            }
        },
        "unchanged_from_v3": {
            "verdict_unit": "supplier x action_class",
            "conjunction_within_unit": "Every condition must pass in every one of five held OOF folds.",
            "cross_class_conjunction": False,
            "both_class_reporting_required": True,
            "auc_min_absolute_lift_over_B0": 0.05,
            "auc_min_absolute_lift_over_A0": 0.03,
            "retention_min_absolute_lift_over_B0": 0.05,
            "retention_min_absolute_lift_over_A0": 0.03,
            "per_video_fp_budget_violations": 0,
            "population_coverage": 1.0,
            "nonfinite_feature_or_score_count": 0,
            "duplicate_or_invalid_action_count": 0,
            "provenance_and_license_pass": True,
            "frozen_population_manifest_sha256": "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
            "arms_and_execution_order": ["B0_SEALED", "A1", "A0"],
            "note": (
                "The measurement constrains absolute floors only. Lift conditions over "
                "sealed B0 and permuted A0 guard against separation explained by the "
                "incumbent or by capacity alone, and are untouched."
            ),
        },
        "requirement_surface": {
            "path": str(REQUIREMENT_SURFACE.relative_to(ROOT)),
            "sha256": sha256(REQUIREMENT_SURFACE),
            "target": "target_plus_0_005",
            "basis": "mean_requirement",
        },
        "evaluator": {
            "path": str(EVALUATOR.relative_to(ROOT)),
            "sha256": sha256(EVALUATOR),
            "role": (
                "Unmodified. It remains the sole producer of per-class per-fold "
                "B0/A1/A0 AUC and CG0-budget retention. This amendment is applied by a "
                "separate verdict layer over its emitted result artifact."
            ),
        },
        "binding_floors": binding,
        "retention_comparability": {
            "verified": True,
            "finding": (
                "The requirement surface's retention (selector_requirement.retention_metrics) "
                "and the separation evaluator's retention (evaluate_cellect_separation.retained) "
                "are the same quantity: beneficial-action retention under the frozen CG0 "
                "per-video harmful-action budget, with denominator total beneficial actions. "
                "The measured floors are therefore directly comparable to the frozen floor."
            ),
            "residual_caveat": (
                "Tie handling differs. retention_metrics selects on a strict score cutoff "
                "above the kth-largest harmful score; retained selects score-tied blocks "
                "atomically and only when the whole block fits the budget. On heavily tied "
                "scores the two can differ marginally. The binding floors are stated on the "
                "evaluator's tie rule, which is the stricter of the two."
            ),
        },
        "explicit_consequence": {
            "auc": (
                "Tightened for both classes: delete 0.65 -> 0.85, occupied_swap 0.65 -> 0.90. "
                "This is the change that carried the TOO_LENIENT verdict."
            ),
            "retention": (
                "RELAXED for both classes: delete 0.15 -> 0.020394, occupied_swap 0.15 -> 0.067910. "
                "The measured requirement to reach +0.005 is well below the preregistered guess. "
                "Adopting the measured surface verbatim therefore loosens this condition. "
                "It is recorded explicitly so it is a visible decision, not a side effect."
            ),
            "net": (
                "The amended gate is not uniformly stricter. It is calibrated: each floor is "
                "set to what run_v6 measured as necessary for the +0.005 target."
            ),
            "reversal": (
                "To keep the frozen retention floor instead, set retention_absolute_floor back "
                "to 0.15 per class in this artifact and rerun the verdict layer. No supplier "
                "measurement changes."
            ),
        },
        "reference_only_alternative_bars": {
            "legacy_frozen_v3": {
                "auc_absolute_floor": FROZEN_AUC_FLOOR,
                "retention_absolute_floor": FROZEN_RETENTION_FLOOR,
                "role": "Reported for continuity with all prior campaign records. Not binding.",
            },
            "measured_robust_p05": {
                "delete": binding["delete"]["reference_only_robust_p05"],
                "occupied_swap": binding["occupied_swap"]["reference_only_robust_p05"],
                "role": (
                    "The p05-robust variant of the same measurement. Reported so a PASS on "
                    "the mean bar that would fail the robust bar is visible. Not binding."
                ),
            },
        },
        "restore_topology": {
            "status": "NOT_APPLICABLE_POPULATION_LIMIT",
            "reason": (
                "Infeasible on the frozen population: oracle ceiling 0 at 50 actions / 0 "
                "beneficial. It is not one of the two separation-gate classes and no floor "
                "is defined for it."
            ),
        },
        "chronology_at_freeze": {
            "supplier_feature_manifest_opened": False,
            "B0_outcome_opened": False,
            "A1_outcome_opened": False,
            "A0_outcome_opened": False,
            "supplier_separation_result_artifact_exists": False,
            "verified_by": (
                "No file under reports/ matches schema "
                "celltrack-alpha.cellect-separation-result.v1 at freeze time."
            ),
            "dinov2_gpu_v1_state": "RUNNING, no local result artifact",
            "dinov2_cpu_promoted_state": "RUNNING, no local result artifact",
            "cellect_r7_state": "launch-closed, no result artifact",
            "graph_run": False,
            "family_6bba_run": False,
            "cv57_run": False,
            "tier_move": False,
            "submission": False,
            "runpod": False,
        },
        "verdict_semantics": {
            "PASS": (
                "For that supplier x action_class only, unlock a newly frozen 44b6 graph-gate "
                "preregistration for human review. Do not execute the graph or promote a tier."
            ),
            "FAIL": "Park that supplier x action_class instantiation PARK-C.",
            "INVALID": (
                "Any population, fold, label, baseline, feature, metric, provenance, resource, "
                "ordering, or integrity violation produces no scientific verdict for that unit."
            ),
            "shared_failure_rule": (
                "A shared checkpoint, license, loader, representation, leakage, or integrity "
                "failure invalidates both class evaluations for that supplier."
            ),
        },
        "barrier": "HOLD_AFTER_PER_CLASS_VERDICTS_BEFORE_ANY_GRAPH_PREREGISTRATION_EXECUTION",
    }

    out = HERE / "SEPARATION_GATE_MEASURED_AMENDMENT_V4.json"
    out.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"written": out.name, "sha256": sha256(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
