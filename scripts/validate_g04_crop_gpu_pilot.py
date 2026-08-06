#!/usr/bin/env python3
"""Authenticate and adjudicate the bounded G04 crop-wrfeat CUDA pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_SHA256 = "4b588eef4ff45778dae7c6f09b6f5e31eb825fddefce58bfc9c605083bd01f5d"
EXPECTED_DETECTIONS_SHA256 = "33c902a9d0fd2b7b9a8a13283f6048c327a114ff692e051208bd833e4b6dcb30"
EXPECTED_CODE_SHA256 = "ae5dbf00fbb5c42f000138ff4f628feef45f718c2a041e792238bbdab32381fd"
EXPECTED_FRAMES = (
    ("train:44b6_40c45f5a", 0, 243, "5d908c6b92e925aa7134ab7df64095bee4f21cfcb5823581c223300eb8a8e914", "design"),
    ("train:44b6_40c45f5a", 50, 407, "a1abfcfe2b773739679fba24ac2a1f2aca924d3217aec8c87b807ca35a816139", "independent"),
    ("train:6bba_debd7bfa", 0, 266, "2343f03a76d7a0125ea086d10b11f7e3e171a6b87a4246dc5ee90a6e3b754a7c", "independent"),
)


class ValidationError(RuntimeError):
    """The downloaded pilot or a provenance receipt drifted."""


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
        raise ValidationError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root is not an object: {path}")
    return value


def close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValidationError(f"{label} drifted: {observed} != {expected}")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ValidationError(f"refusing to overwrite adjudication: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    result_path = args.remote_output / "G04_CROP_WRFEAT_GPU_PILOT.json"
    step0_path = args.remote_output / "G04_CROP_WRFEAT_GPU_PILOT_STEP0.json"
    supervisor_path = args.remote_output / "G04_CROP_WRFEAT_GPU_PILOT_SUPERVISOR.json"
    result, step0, supervisor = map(load, (result_path, step0_path, supervisor_path))
    prelaunch = load(args.prelaunch)
    launch = load(args.launch_receipt)
    terminal = load(args.terminal_receipt)
    ledger = load(args.gpu_ledger)

    if (
        step0.get("schema") != "celltrack-alpha.g04-crop-wrfeat-gpu-pilot-step0.v1"
        or step0.get("status") != "PASS_PURE_STDLIB_BEFORE_SCIENTIFIC_IMPORTS"
        or step0.get("third_party_imports") is not False
        or step0.get("competition_raw_bytes_authenticated") is not True
        or step0.get("pilot_source_sha256") != EXPECTED_SOURCE_SHA256
        or step0.get("detection_manifest_sha256") != EXPECTED_DETECTIONS_SHA256
        or step0.get("pilot_frame_count") != 3
        or step0.get("pilot_detection_count") != 916
        or step0.get("gt_access") is not False
        or step0.get("submission") is not False
    ):
        raise ValidationError("STEP0 contract failed")
    result_sha = sha256(result_path)
    if (
        supervisor.get("status") != "PASS_WORKER_TERMINAL"
        or supervisor.get("worker_returncode") != 0
        or supervisor.get("worker_timed_out") is not False
        or supervisor.get("result_sha256") != result_sha
        or supervisor.get("full_run_authorized") is not False
        or result.get("schema") != "celltrack-alpha.g04-crop-wrfeat-gpu-pilot.v2"
        or result.get("status") != "PASS_BOUNDED_GPU_CROP_PILOT"
        or result.get("production_manifest_eligible") is not False
        or result.get("full_run_authorized") is not False
        or result.get("full_run_launch_performed") is not False
    ):
        raise ValidationError("terminal supervisor/result contract failed")
    if (
        prelaunch.get("status") != "PASS"
        or prelaunch.get("eligible") is not True
        or prelaunch.get("code_sha256") != EXPECTED_CODE_SHA256
        or launch.get("kernel_push_predicate", {}).get("code_sha256")
        != EXPECTED_CODE_SHA256
        or launch.get("returncode") != 0
        or terminal.get("stdout_sha256")
        != "e804bfd491d875c5958ceb53db4f5f174541f683b11f4ca647264192bd2feb92"
    ):
        raise ValidationError("prelaunch/launch/terminal provenance failed")

    parity = result["serial_batch_one_parity"]
    if (
        parity.get("status") != "PASS_EXACT_LABEL_BYTES"
        or parity.get("serial_labels_sha256") != parity.get("batch_labels_sha256")
    ):
        raise ValidationError("serial/batch-size-one parity failed")
    configuration = result["configuration"]
    if (
        configuration.get("work_unit") != "per_detection_crop_batched_across_frame"
        or configuration.get("feature_scope") != "checkpoint_native_wrfeat12_only"
        or configuration.get("descriptive_qhull_or_regionprops") is not False
        or configuration.get("primary_shape_zyx") != [32, 64, 64]
        or configuration.get("fallback_shape_zyx") != [48, 96, 96]
        or configuration.get("maximum_recenter_um") != 4.0
        or configuration.get("crop_batch_size") != 4
        or configuration.get("network_plane_batch_size") != 64
    ):
        raise ValidationError("GPU crop configuration drifted")

    rows = result.get("frames")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_FRAMES):
        raise ValidationError("pilot frame coverage drifted")
    for row, expected in zip(rows, EXPECTED_FRAMES, strict=True):
        observed = (
            row.get("identity"),
            row.get("timepoint"),
            row.get("detection_count"),
            row.get("raw_logical_sha256"),
            row.get("role"),
        )
        if observed != expected:
            raise ValidationError(f"pilot frame identity/input drifted: {observed}")
        primary = row["batch_runtime"]["primary"]
        fallback = row["batch_runtime"]["fallback"]
        if primary.get("genuine_cross_detection_gpu_batching") is not True:
            raise ValidationError("primary did not use genuine CUDA batching")
        if fallback is not None and fallback.get("genuine_cross_detection_gpu_batching") is not True:
            raise ValidationError("fallback did not use genuine CUDA batching")

    independent = [row for row in rows if row["role"] == "independent"]
    independent_count = sum(int(row["detection_count"]) for row in independent)
    independent_valid = sum(int(row["valid_nonboundary_count"]) for row in independent)
    independent_boundary = sum(int(row["boundary_contact_count"]) for row in independent)
    independent_empty = sum(int(row["empty_selected_mask_count"]) for row in independent)
    gate = result["independent_coverage_gate"]
    if (
        independent_count != 673
        or independent_valid != 673
        or independent_boundary != 0
        or independent_empty != 0
        or gate.get("pass") is not True
        or gate.get("valid_fraction") != 1.0
        or gate.get("boundary_contact_fraction") != 0.0
        or gate.get("empty_selected_mask_fraction") != 0.0
    ):
        raise ValidationError("independent coverage gate did not reproduce")

    total_count = sum(int(row["detection_count"]) for row in rows)
    total_wall = sum(float(row["wall_seconds"]) for row in rows)
    seconds_per_detection = total_wall / total_count
    independent_seconds = sorted(float(row["wall_seconds"]) for row in independent)
    p95_frame_seconds = independent_seconds[0] + 0.95 * (
        independent_seconds[1] - independent_seconds[0]
    )
    initialization = float(result["initialization_seconds"])
    cv_detection_hours = (
        initialization + seconds_per_detection * 1_124_771
    ) / 3600.0
    cv_frame_hours = (initialization + p95_frame_seconds * 5_700) / 3600.0
    cv_conservative = 1.10 * max(cv_detection_hours, cv_frame_hours)
    combined_detection_hours = (
        initialization + seconds_per_detection * 3_748_493
    ) / 3600.0
    combined_frame_hours = (
        initialization + p95_frame_seconds * 16_200
    ) / 3600.0
    combined_conservative = 1.10 * max(
        combined_detection_hours, combined_frame_hours
    )
    projections = result["projections"]
    close(
        seconds_per_detection,
        float(projections["cv57"]["all_pilot_seconds_per_detection"]),
        "seconds per detection",
    )
    close(
        cv_conservative,
        float(projections["cv57"]["projected_t4_hours_conservative"]),
        "CV57 conservative projection",
    )
    close(
        combined_conservative,
        float(
            projections["held_fit105_plus_cv57"][
                "projected_t4_hours_conservative"
            ]
        ),
        "combined conservative projection",
    )
    if (
        projections["quota_reference"]["cv57_fits_unreserved_reference_remaining"]
        is not False
        or projections["quota_reference"][
            "combined_fits_unreserved_reference_remaining"
        ]
        is not False
    ):
        raise ValidationError("budget gate unexpectedly opened")

    active = ledger.get("active_reservations")
    quota = ledger.get("latest_authoritative_quota", {})
    reconciles = [
        row
        for row in ledger.get("events", [])
        if row.get("type") == "terminal_reconcile"
        and row.get("kernel_id") == "onebitbrain/g04-crop-wrfeat-gpu-pilot-v1"
    ]
    if (
        not isinstance(active, list)
        or any(row.get("kernel_id") == "onebitbrain/g04-crop-wrfeat-gpu-pilot-v1" for row in active)
        or len(reconciles) != 1
        or quota.get("used_h") != 2.76
        or quota.get("remaining_h") != 27.24
    ):
        raise ValidationError("shared GPU reservation/quota did not reconcile")

    totals = {
        "detection_count": total_count,
        "valid_nonboundary_count": sum(
            int(row["valid_nonboundary_count"]) for row in rows
        ),
        "boundary_contact_count": sum(
            int(row["boundary_contact_count"]) for row in rows
        ),
        "empty_selected_mask_count": sum(
            int(row["empty_selected_mask_count"]) for row in rows
        ),
        "fallback_attempted_count": sum(
            int(row["fallback_attempted_count"]) for row in rows
        ),
        "fallback_recovered_count": sum(
            int(row["fallback_recovered_count"]) for row in rows
        ),
    }
    return {
        "schema": "celltrack-alpha.g04-crop-wrfeat-gpu-pilot-terminal-adjudication.v1",
        "status": "PASS_COVERAGE_FAIL_BUDGET_FULL_RUN_CLOSED",
        "coverage_gate": "PASS",
        "budget_gate": "FAIL",
        "full_run_authorized": False,
        "score_run_authorized": False,
        "remote": {
            "result_sha256": result_sha,
            "step0_sha256": sha256(step0_path),
            "supervisor_sha256": sha256(supervisor_path),
            "serial_batch_labels_sha256": parity["serial_labels_sha256"],
            "worker_elapsed_seconds": result["elapsed_seconds"],
            "kernel_elapsed_seconds": supervisor["elapsed_seconds"],
        },
        "coverage": {
            "all_frames": totals,
            "independent": gate,
        },
        "throughput": {
            "initialization_seconds": initialization,
            "all_pilot_seconds_per_detection": seconds_per_detection,
            "independent_p95_seconds_per_frame": p95_frame_seconds,
        },
        "projections_t4_hours": {
            "cv57_detection_scaled_before_margin": cv_detection_hours,
            "cv57_frame_scaled_before_margin": cv_frame_hours,
            "cv57_preregistered_conservative": cv_conservative,
            "held_fit105_plus_cv57_detection_scaled_before_margin": combined_detection_hours,
            "held_fit105_plus_cv57_frame_scaled_before_margin": combined_frame_hours,
            "held_fit105_plus_cv57_preregistered_conservative": combined_conservative,
        },
        "quota": {
            "used_h": quota["used_h"],
            "remaining_h": quota["remaining_h"],
            "pilot_rounded_debit_h": 0.25,
            "reservation_released_h": reconciles[0]["released_max_gpu_h"],
        },
        "interpretation": (
            "The crop-size/recentering coverage defect and CPU execution defect are "
            "repaired, but per-detection three-orientation Cellpose work remains far "
            "more expensive than whole-frame reuse. No full feature or score run."
        ),
        "score_effect": {
            "crop_wrfeat": "NOT_MEASURED_FULL_RUN_CLOSED",
            "v32_global_jump": "PENDING_IMMUTABLE_V32_BASE_AUTHENTICATION",
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    lane = ROOT / "reports/g04/lanes/global_jump_crop_wrfeat_v2"
    value.add_argument(
        "--remote-output",
        type=Path,
        default=lane / "crop_gpu_pilot_remote_output_v1",
    )
    value.add_argument(
        "--prelaunch", type=Path, default=lane / "CROP_GPU_PILOT_PRELAUNCH_INLINE.json"
    )
    value.add_argument(
        "--launch-receipt",
        type=Path,
        default=lane / "cmux_receipts/kernel_push_2026-08-03T102648666724Z.json",
    )
    value.add_argument(
        "--terminal-receipt",
        type=Path,
        default=lane / "cmux_receipts/kernel_status_2026-08-03T104518397451Z.json",
    )
    value.add_argument(
        "--gpu-ledger",
        type=Path,
        default=ROOT / "reports/g02/screening/tiered_surface/KAGGLE_SHARED_GPU_LEDGER.json",
    )
    value.add_argument(
        "--output", type=Path, default=lane / "CROP_GPU_PILOT_TERMINAL_ADJUDICATION.json"
    )
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    report = run(arguments)
    atomic_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
