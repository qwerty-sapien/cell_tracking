#!/usr/bin/env python3
"""Post-hoc audit of calibration-only family thresholds for the crop pilot."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import fit_crop_linker_pilot as pilot  # noqa: E402
import run_bh  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha.crop_linker import (  # noqa: E402
    SiameseCropLinker,
    TemperatureCalibration,
)
from fit_selected_graph_temporal_rank_gate import _metrics, _select  # noqa: E402


SCHEMA = "celltrack-alpha.crop-linker-family-threshold-audit.v1"
PILOT_REPORT_SHA256 = (
    "082e682b74a78ae74005ac0d7b86ee7d76b446bae909b3cdb9a8bd93a29bf333"
)
PILOT_ARTIFACT_SHA256 = (
    "fc3c36475a6424e0b5d382650714dc7c6d0c052d92178da80d61d2b33d03f20f"
)


class FamilyThresholdAuditError(RuntimeError):
    """The immutable pilot or threshold-audit contract drifted."""


def family_rows(
    scored: Sequence[Mapping[str, Any]], family: str
) -> list[Mapping[str, Any]]:
    result = [
        row
        for row in scored
        if str(row["identity"]).split(":", 1)[1].startswith(family + "_")
    ]
    if not result:
        raise FamilyThresholdAuditError(f"family is absent: {family}")
    return result


def combined_metrics(
    scored: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    family_results = {
        family: _metrics(family_rows(scored, family), threshold)
        for family, threshold in thresholds.items()
    }
    correct = sum(
        int(value["correct_rescues"]) for value in family_results.values()
    )
    harmful = sum(
        int(value["harmful_swaps"]) for value in family_results.values()
    )
    accepted = correct + harmful
    return {
        "accepted_labeled": accepted,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "precision": correct / max(accepted, 1),
        "conservative_utility": correct - 2 * harmful,
        "families": family_results,
    }


def select_family_thresholds(
    scored: Sequence[Mapping[str, Any]],
    *,
    minimum_accepted: int,
    minimum_precision: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds = {}
    calibration = {}
    for family in ("44b6", "6bba"):
        threshold, metrics = _select(
            family_rows(scored, family),
            minimum_accepted=minimum_accepted,
            minimum_precision=minimum_precision,
        )
        thresholds[family] = threshold
        calibration[family] = metrics
    return thresholds, calibration


def report_thresholds(
    thresholds: Mapping[str, float],
) -> dict[str, float | None]:
    return {
        family: float(threshold) if np.isfinite(threshold) else None
        for family, threshold in thresholds.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FamilyThresholdAuditError(
            f"refusing to overwrite output: {args.output}"
        )
    for path, expected, name in (
        (args.pilot_report, args.pilot_report_sha256, "pilot report"),
        (args.pilot_artifact, args.pilot_artifact_sha256, "pilot artifact"),
        (
            args.dataset_report,
            pilot.DATASET_REPORT_SHA256,
            "dataset report",
        ),
        (args.prepare, pilot.PREPARE_SHA256, "prepare report"),
    ):
        if sweep._sha256(path) != expected:
            raise FamilyThresholdAuditError(f"{name} SHA-256 mismatch")
    report = json.loads(args.pilot_report.read_text(encoding="ascii"))
    dataset = json.loads(args.dataset_report.read_text(encoding="ascii"))
    artifact = torch.load(
        args.pilot_artifact, map_location="cpu", weights_only=False
    )
    if (
        report.get("schema") != pilot.SCHEMA
        or report.get("status") != "PARK"
        or artifact.get("schema") != pilot.SCHEMA
        or artifact.get("dataset_report_sha256")
        != pilot.DATASET_REPORT_SHA256
    ):
        raise FamilyThresholdAuditError("pilot identity drifted")

    calibration = pilot.load_examples(
        dataset, partition="calibration", verify_hashes=True
    )
    validation = pilot.load_examples(
        dataset, partition="validation", verify_hashes=True
    )
    model = SiameseCropLinker(
        context_features=48,
        embedding_width=int(artifact["embedding_width"]),
        hidden=int(artifact["hidden"]),
    )
    model.load_state_dict(artifact["state_dict"], strict=True)
    temperature = TemperatureCalibration(
        temperature=float(artifact["temperature"]),
        before={},
        after={},
    )
    device = torch.device("cpu")
    calibration_logits = pilot.model_logits(
        model,
        calibration,
        context_mean=np.asarray(artifact["context_mean"]),
        context_scale=np.asarray(artifact["context_scale"]),
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    validation_logits = pilot.model_logits(
        model,
        validation,
        context_mean=np.asarray(artifact["context_mean"]),
        context_scale=np.asarray(artifact["context_scale"]),
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    selected_edges = {
        str(row["identity"]): int(row["graph"]["raw_selected_edges"])
        for row in json.loads(
            args.prepare.read_text(encoding="ascii")
        )["videos"]
    }
    candidate_calibration = pilot.score_records(
        calibration,
        torch.sigmoid(temperature.apply(calibration_logits)).numpy(),
        selected_edges=selected_edges,
    )
    candidate_validation = pilot.score_records(
        validation,
        torch.sigmoid(temperature.apply(validation_logits)).numpy(),
        selected_edges=selected_edges,
    )
    control_calibration = pilot.score_records(
        calibration,
        calibration.context[:, 44].astype(np.float64),
        selected_edges=selected_edges,
    )
    control_validation = pilot.score_records(
        validation,
        validation.context[:, 44].astype(np.float64),
        selected_edges=selected_edges,
    )

    candidate_thresholds, candidate_calibration_metrics = (
        select_family_thresholds(
            candidate_calibration,
            minimum_accepted=args.minimum_accepted_per_family,
            minimum_precision=args.minimum_precision,
        )
    )
    control_thresholds, control_calibration_metrics = (
        select_family_thresholds(
            control_calibration,
            minimum_accepted=args.minimum_accepted_per_family,
            minimum_precision=args.minimum_precision,
        )
    )
    candidate_validation_metrics = combined_metrics(
        candidate_validation, candidate_thresholds
    )
    control_validation_metrics = combined_metrics(
        control_validation, control_thresholds
    )
    result = {
        "schema": SCHEMA,
        "status": "EXPLORATORY-NONCONFIRMATORY",
        "scope": {
            "post_hoc_after_validation_unseal": True,
            "cv57_access": False,
            "training": False,
            "calibration_only_threshold_selection": True,
        },
        "selection": {
            "minimum_accepted_per_family": args.minimum_accepted_per_family,
            "minimum_precision": args.minimum_precision,
        },
        "candidate": {
            "thresholds": report_thresholds(candidate_thresholds),
            "threshold_admissible": {
                family: bool(np.isfinite(threshold))
                for family, threshold in candidate_thresholds.items()
            },
            "calibration": candidate_calibration_metrics,
            "validation": candidate_validation_metrics,
        },
        "control": {
            "thresholds": report_thresholds(control_thresholds),
            "threshold_admissible": {
                family: bool(np.isfinite(threshold))
                for family, threshold in control_thresholds.items()
            },
            "calibration": control_calibration_metrics,
            "validation": control_validation_metrics,
        },
        "comparison": {
            "validation_utility_gain": int(
                candidate_validation_metrics["conservative_utility"]
            )
            - int(control_validation_metrics["conservative_utility"]),
            "candidate_passes_pilot_precision_floor": all(
                float(value["precision"]) >= args.pilot_precision_floor
                for value in candidate_validation_metrics[
                    "families"
                ].values()
            ),
            "interpretation": (
                "mechanism diagnostic only; adaptive reuse of validation12 "
                "cannot promote the model or unlock CV57"
            ),
        },
        "provenance": {
            "pilot_report": str(args.pilot_report),
            "pilot_report_sha256": args.pilot_report_sha256,
            "pilot_artifact": str(args.pilot_artifact),
            "pilot_artifact_sha256": args.pilot_artifact_sha256,
            "dataset_report": str(args.dataset_report),
            "dataset_report_sha256": pilot.DATASET_REPORT_SHA256,
            "prepare": str(args.prepare),
            "prepare_sha256": pilot.PREPARE_SHA256,
        },
        "execution": {
            "device": "cpu",
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
        },
    }
    run_bh.atomic_create_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--pilot-report",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_pilot_v1/report.json",
    )
    value.add_argument(
        "--pilot-report-sha256", default=PILOT_REPORT_SHA256
    )
    value.add_argument(
        "--pilot-artifact",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_pilot_v1/pilot.pt",
    )
    value.add_argument(
        "--pilot-artifact-sha256", default=PILOT_ARTIFACT_SHA256
    )
    value.add_argument(
        "--dataset-report",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_dataset_v1/report.json",
    )
    value.add_argument(
        "--prepare",
        type=Path,
        default=(
            ROOT
            / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
        ),
    )
    value.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reports/g02/crop_linker_family_threshold_audit_v1.json"
        ),
    )
    value.add_argument("--evaluation-batch-size", type=int, default=128)
    value.add_argument(
        "--minimum-accepted-per-family", type=int, default=20
    )
    value.add_argument("--minimum-precision", type=float, default=0.95)
    value.add_argument("--pilot-precision-floor", type=float, default=0.93)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))
