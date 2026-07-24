#!/usr/bin/env python3
"""Screen a deterministic constant-velocity gate on the fit-only 10--12 um shell."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from fit_selected_graph_temporal_rank_gate import _metrics as capped_metrics  # noqa: E402


SCHEMA = "celltrack-alpha.extended-radius-shell-motion-gate.v1"
SOURCE_SCHEMA = "celltrack-alpha.extended-radius-shell-appearance.v1"
DEFAULT_INPUT = ROOT / "reports/g02/extended_radius_shell_appearance_v1/report.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/extended_radius_shell_motion_gate_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/g02/screening/extended_radius_shell_motion_gate_v1.md"
DEFAULT_CONTRACT = ROOT / "reports/g02/appearance_direct_swap_contract_v1.json"
EXPECTED = {
    "input": "8183f347bd89f3405dfdb7e14e564f2b285ef624ab6bd6f5e12d7460b43f7aa8",
    "prepare": "cb882835d9b33bcad016684f988ea81e2fff3713432cbde0dadc9b6879cd6386",
    "reachability": "22a70e0b750d3d59bdd92ca5a82967d35c5d866fafe1c867dbb09b936efc2803",
    "contract": "cee547ed7a82d1bf7f6afe8cbbc3cc9d554181f9e19e87d319a5350863718cf7",
}
TOLERANCES_UM = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


class MotionShellError(RuntimeError):
    """A motion-shell identity, split, or accounting contract failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MotionShellError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise MotionShellError(f"{label} must be a JSON object")
    return value


def motion_residual(
    features: np.ndarray, feature_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return a symmetric local constant-velocity residual in physical microns."""

    names = tuple(feature_names)
    required = (
        "source_history_available",
        "source_acceleration_residual_um",
        "target_future_available",
        "target_continuation_residual_um",
    )
    if any(name not in names for name in required):
        raise MotionShellError("motion features are incomplete")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(names) or not np.isfinite(x).all():
        raise MotionShellError("motion feature matrix is invalid")
    source_available = x[:, names.index(required[0])] > 0.5
    source_residual = x[:, names.index(required[1])]
    target_available = x[:, names.index(required[2])] > 0.5
    target_residual = x[:, names.index(required[3])]
    available = source_available | target_available
    residual = np.full(len(x), np.inf, dtype=np.float64)
    residual[source_available & ~target_available] = source_residual[
        source_available & ~target_available
    ]
    residual[target_available & ~source_available] = target_residual[
        target_available & ~source_available
    ]
    both = source_available & target_available
    # Requiring the worse of forward and backward consistency is deterministic
    # and conservative; no learned or fitted weighting is introduced.
    residual[both] = np.maximum(source_residual[both], target_residual[both])
    if np.any(residual[available] < 0.0):
        raise MotionShellError("physical motion residual cannot be negative")
    return residual, available


def ncc_confidence(features: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Mirror the banked pair-vs-incumbent NCC confidence on shell rows."""

    names = tuple(feature_names)
    required = (
        "pair_ncc",
        "conflict_ncc_available",
        "pair_minus_maximum_conflict_ncc",
    )
    if any(name not in names for name in required):
        raise MotionShellError("NCC features are incomplete")
    x = np.asarray(features, dtype=np.float64)
    pair = x[:, names.index("pair_ncc")]
    conflict_available = x[:, names.index("conflict_ncc_available")] > 0.5
    margin = x[:, names.index("pair_minus_maximum_conflict_ncc")]
    result = np.where(conflict_available, margin, pair)
    if not np.isfinite(result).all():
        raise MotionShellError("NCC confidence is non-finite")
    return result


def _family(identity: str) -> str:
    if identity.startswith("train:44b6_"):
        return "44b6"
    if identity.startswith("train:6bba_"):
        return "6bba"
    raise MotionShellError(f"unknown family identity: {identity}")


def _video_scores(
    video: Mapping[str, Any],
    *,
    tolerance_um: float,
    ncc_threshold: float,
) -> dict[str, Any]:
    residual = np.asarray(video["residual"], dtype=np.float64)
    labels = np.asarray(video["labels"], dtype=np.int8)
    ncc = np.asarray(video["ncc"], dtype=np.float64)
    available = np.asarray(video["available"], dtype=bool)
    eligible = (
        available
        & (ncc >= ncc_threshold)
        & (residual <= float(tolerance_um))
    )
    score = np.where(eligible, -residual, -np.inf)
    labeled = labels != -1
    background = labels == -1
    return {
        "identity": video["identity"],
        "selected_edges": int(video["selected_edges"]),
        "labeled_confidence": score[labeled],
        "labels": labels[labeled],
        "background_confidence": score[background],
        "eligible_rows": int(np.sum(eligible)),
        "eligible_positive": int(np.sum(eligible & (labels == 1))),
        "eligible_known_negative": int(np.sum(eligible & (labels == 0))),
        "eligible_background": int(np.sum(eligible & (labels == -1))),
        "total_positive": int(np.sum(labels == 1)),
        "total_rows": len(labels),
        "threshold": -float(tolerance_um),
    }


def evaluate(
    videos: Sequence[Mapping[str, Any]],
    *,
    tolerance_um: float,
    ncc_threshold: float,
) -> dict[str, Any]:
    """Evaluate one tolerance using the existing deployed cap/utility helper."""

    scored = [
        _video_scores(
            video,
            tolerance_um=tolerance_um,
            ncc_threshold=ncc_threshold,
        )
        for video in videos
    ]
    cap_input = [
        {
            "identity": row["identity"],
            "selected_edges": row["selected_edges"],
            "labeled_confidence": row["labeled_confidence"],
            "labels": row["labels"],
            "background_confidence": row["background_confidence"],
        }
        for row in scored
    ]
    metrics = capped_metrics(cap_input, -float(tolerance_um))
    eligible_rows = sum(int(row["eligible_rows"]) for row in scored)
    positive_total = sum(int(row["total_positive"]) for row in scored)
    eligible_positive = sum(int(row["eligible_positive"]) for row in scored)
    total_rows = sum(int(row["total_rows"]) for row in scored)
    families = {}
    for family in ("44b6", "6bba"):
        subset = [
            item for item in cap_input if _family(str(item["identity"])) == family
        ]
        families[family] = capped_metrics(subset, -float(tolerance_um))
    result = {
        "tolerance_um": float(tolerance_um),
        "capped": metrics,
        "families": families,
        "pre_cap": {
            "rows": total_rows,
            "eligible_rows": eligible_rows,
            "candidate_reduction_fraction": 1.0
            - eligible_rows / max(total_rows, 1),
            "positive_rows": positive_total,
            "eligible_positive_rows": eligible_positive,
            "shell_positive_recall": eligible_positive / max(positive_total, 1),
            "eligible_known_negative": sum(
                int(row["eligible_known_negative"]) for row in scored
            ),
            "eligible_sampled_background": sum(
                int(row["eligible_background"]) for row in scored
            ),
        },
    }
    return result


def _admissible(row: Mapping[str, Any], *, minimum_correct: int) -> bool:
    capped = row["capped"]
    return (
        int(capped["correct_rescues"]) >= minimum_correct
        and float(capped["precision"]) >= 0.95
        and int(capped["background_accepted"]) == 0
        and all(
            (
                int(row["families"][family]["accepted_labeled"]) == 0
                or float(row["families"][family]["precision"]) >= 0.93
            )
            for family in ("44b6", "6bba")
        )
    )


def select_tolerance(grid: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select only from train105, with deterministic PARK fallback."""

    rows = [dict(row) for row in grid]
    for row in rows:
        row["train_admissible"] = _admissible(row, minimum_correct=25)
    admissible = [row for row in rows if row["train_admissible"]]
    supported = [
        row for row in rows if int(row["capped"]["correct_rescues"]) > 0
    ]
    pool = admissible or supported or rows
    selected = max(
        pool,
        key=lambda row: (
            int(row["capped"]["conservative_utility"]),
            float(row["capped"]["precision"]),
            int(row["capped"]["correct_rescues"]),
            -int(row["capped"]["background_accepted"]),
            -float(row["tolerance_um"]),
        ),
    )
    return {"admissible": bool(admissible), "selected": selected, "grid": rows}


def validate_inputs(
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    input_path: Path,
    contract_path: Path,
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    """Fail closed on the exact fit-only source and banked APNCC contract."""

    scope = report.get("scope")
    if (
        sha256(input_path) != EXPECTED["input"]
        or report.get("schema") != SOURCE_SCHEMA
        or not isinstance(scope, Mapping)
        or scope.get("fit_only") is not True
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise MotionShellError("appearance source is not the exact fit-only input")
    train = tuple(str(value) for value in scope.get("train_video_ids", []))
    validation = tuple(
        str(value) for value in scope.get("validation_video_ids", [])
    )
    if (
        len(train) != 105
        or len(validation) != 12
        or set(train).intersection(validation)
    ):
        raise MotionShellError("fixed grouped 105/12 split is invalid")
    provenance = report.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("prepare_sha256") != EXPECTED["prepare"]
        or sha256(contract_path) != EXPECTED["contract"]
        or contract.get("schema")
        != "celltrack-alpha.appearance-direct-swap-contract.v1"
    ):
        raise MotionShellError("source or APNCC provenance drifted")
    appearance = contract.get("appearance")
    geometry = contract.get("geometry_gate")
    cap = contract.get("proposal_cap")
    if (
        not isinstance(appearance, Mapping)
        or not isinstance(geometry, Mapping)
        or not isinstance(cap, Mapping)
        or int(cap.get("minimum_per_video", 0)) != 16
        or float(cap.get("selected_edge_fraction", 0.0)) != 0.005
    ):
        raise MotionShellError("banked APNCC contract is malformed")
    return float(appearance["threshold"]), train, validation


def _load_videos(
    report: Mapping[str, Any],
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = set(train_ids) | set(validation_ids)
    seen = set()
    result = {"train": [], "validation": []}
    for row in report["videos"]:
        identity = str(row["identity"])
        if identity not in expected or identity in seen:
            raise MotionShellError("video identity coverage drifted")
        path = ROOT / str(row["path"])
        if sha256(path) != row["sha256"]:
            raise MotionShellError(f"example SHA-256 mismatch: {identity}")
        value = torch.load(path, map_location="cpu", weights_only=False)
        names = tuple(str(name) for name in value["feature_names"])
        if names != tuple(str(name) for name in report["feature_names"]):
            raise MotionShellError("example feature identity drifted")
        features = np.asarray(value["features"], dtype=np.float32)
        labels = np.asarray(value["labels"], dtype=np.int8)
        if (
            features.shape != (len(labels), len(names))
            or np.any(~np.isin(labels, (-1, 0, 1)))
        ):
            raise MotionShellError("example labels/features are malformed")
        residual, available = motion_residual(features, names)
        record = {
            "identity": identity,
            "selected_edges": int(row["rows"])
            if "graph" not in row
            else int(row["graph"]["raw_selected_edges"]),
            "labels": labels,
            "residual": residual,
            "available": available,
            "ncc": ncc_confidence(features, names),
        }
        # Appearance rows omit the graph count, so recover it from the
        # hash-pinned prepare manifest joined below in run().
        result[str(row["partition"])].append(record)
        seen.add(identity)
    if seen != expected:
        raise MotionShellError("video coverage is incomplete")
    return result["train"], result["validation"]


def _markdown(report: Mapping[str, Any]) -> str:
    selected = report["selection"]["train"]
    validation = report["selection"]["validation"]
    return (
        "# Deterministic motion-gated extended-radius shell\n\n"
        f"Decision: **{report['status']}**.\n\n"
        "The screen uses the exact fit117 10–12 µm/cap-8 shell, the banked "
        "APNCC threshold, and a symmetric constant-velocity residual. Only "
        "the motion tolerance was selected on train105; validation12 was "
        "evaluated once after freezing it.\n\n"
        "| partition | tolerance (µm) | correct | harmful | sampled bg | "
        "precision | utility | shell recall | reduction |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| train105 | {selected['tolerance_um']:.3f} | "
        f"{selected['capped']['correct_rescues']} | "
        f"{selected['capped']['harmful_swaps']} | "
        f"{selected['capped']['background_accepted']} | "
        f"{selected['capped']['precision']:.4f} | "
        f"{selected['capped']['conservative_utility']} | "
        f"{selected['pre_cap']['shell_positive_recall']:.4f} | "
        f"{selected['pre_cap']['candidate_reduction_fraction']:.4f} |\n"
        f"| validation12 | {validation['tolerance_um']:.3f} | "
        f"{validation['capped']['correct_rescues']} | "
        f"{validation['capped']['harmful_swaps']} | "
        f"{validation['capped']['background_accepted']} | "
        f"{validation['capped']['precision']:.4f} | "
        f"{validation['capped']['conservative_utility']} | "
        f"{validation['pre_cap']['shell_positive_recall']:.4f} | "
        f"{validation['pre_cap']['candidate_reduction_fraction']:.4f} |\n\n"
        "No model training, CV57, Kaggle, GPU, RunPod, or submission action "
        "occurred.\n"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.markdown.exists():
        raise MotionShellError("refusing to overwrite deterministic outputs")
    report = load_json(args.input, "appearance shell report")
    contract = load_json(args.contract, "APNCC contract")
    ncc_threshold, train_ids, validation_ids = validate_inputs(
        report,
        contract,
        input_path=args.input,
        contract_path=args.contract,
    )
    prepare_path = ROOT / "reports/g02/extended_radius_shell_prepare_v1/prepare.json"
    reachability_path = ROOT / "reports/g02/extended_radius_reachability_v1.json"
    if (
        sha256(prepare_path) != EXPECTED["prepare"]
        or sha256(reachability_path) != EXPECTED["reachability"]
    ):
        raise MotionShellError("shell/reachability provenance drifted")
    prepare = load_json(prepare_path, "shell prepare report")
    graph_counts = {
        str(row["identity"]): int(row["graph"]["raw_selected_edges"])
        for row in prepare["videos"]
    }
    train, validation = _load_videos(report, train_ids, validation_ids)
    for video in (*train, *validation):
        video["selected_edges"] = graph_counts[str(video["identity"])]
    train_grid = [
        evaluate(train, tolerance_um=value, ncc_threshold=ncc_threshold)
        for value in TOLERANCES_UM
    ]
    selection = select_tolerance(train_grid)
    tolerance = float(selection["selected"]["tolerance_um"])
    validation_result = evaluate(
        validation, tolerance_um=tolerance, ncc_threshold=ncc_threshold
    )
    validation_passed = _admissible(validation_result, minimum_correct=5)
    result = {
        "schema": SCHEMA,
        "status": "BANK" if selection["admissible"] and validation_passed else "PARK",
        "scope": {
            "fit_only": True,
            "train_video_ids": list(train_ids),
            "validation_video_ids": list(validation_ids),
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
            "cv57_access": False,
        },
        "mechanism": {
            "geometry": "exact prepared 10<distance_um<=12, cap-8 shell",
            "appearance_confidence": (
                "pair NCC for free rows; pair-minus-maximum-conflict NCC "
                "for occupied rows"
            ),
            "appearance_threshold": ncc_threshold,
            "motion": (
                "maximum available forward/backward constant-velocity "
                "residual in physical microns"
            ),
            "learned_weights": False,
        },
        "selection_contract": {
            "selection_partition": "fit-train105 only",
            "validation_partition": "fixed fit-validation12, evaluated once",
            "only_tuned_parameter": "motion_tolerance_um",
            "tolerances_um": list(TOLERANCES_UM),
            "train_minimum_correct": 25,
            "validation_minimum_correct": 5,
            "minimum_precision": 0.95,
            "minimum_family_precision": 0.93,
            "maximum_sampled_background": 0,
            "cap_implementation": (
                "fit_selected_graph_temporal_rank_gate._metrics and "
                "capped_video_metrics"
            ),
        },
        "selection": {
            "train_admissible": selection["admissible"],
            "train": selection["selected"],
            "validation": validation_result,
            "validation_passed": validation_passed,
        },
        "grid": selection["grid"],
        "provenance": {
            "input": str(args.input),
            "input_sha256": sha256(args.input),
            "prepare": str(prepare_path),
            "prepare_sha256": sha256(prepare_path),
            "reachability": str(reachability_path),
            "reachability_sha256": sha256(reachability_path),
            "appearance_contract": str(args.contract),
            "appearance_contract_sha256": sha256(args.contract),
        },
        "execution": {
            "device": "cpu",
            "deterministic": True,
            "model_training": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
        },
    }
    run_bh.atomic_create_json(args.output, result)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(result), encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    value.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "train_admissible": result["selection"]["train_admissible"],
                "validation_passed": result["selection"]["validation_passed"],
                "tolerance_um": result["selection"]["train"]["tolerance_um"],
            },
            sort_keys=True,
        )
    )
