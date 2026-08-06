#!/usr/bin/env python3
"""Offline owner-approval and measured-cost gates for G-04 RunPod work.

This module has no RunPod client, network code, credential handling, upload,
launch, or submission capability. It can only (1) turn an explicit owner
approval into exact-frame benchmark job contracts and (2) validate terminal
benchmark/price receipts into a local production-unlock receipt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCHEMA = "celltrack-alpha.g04-runpod-owner-approval-package.v1"
APPROVAL_SCHEMA = "celltrack-alpha.g04-runpod-owner-approval.v1"
BENCHMARK_SCHEMA = "celltrack-alpha.g04-runpod-exact-frame-benchmark.v1"
BENCHMARK_RESULT_SCHEMA = (
    "celltrack-alpha.g04-runpod-exact-frame-benchmark-result.v1"
)
PRICING_SCHEMA = "celltrack-alpha.g04-runpod-production-pricing-receipt.v1"
BENCHMARK_PLAN_SCHEMA = "celltrack-alpha.g04-runpod-benchmark-plan.v1"
UNLOCK_SCHEMA = "celltrack-alpha.g04-runpod-production-unlock.v1"
OFFICIAL_PRICING_URL = "https://www.runpod.io/pricing"


class GateError(RuntimeError):
    """An owner, scientific, cost, timing, or atomicity gate failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise GateError(f"{label} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise GateError(f"{label} must be a finite positive number")
    return number


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise GateError(f"{label} must be a finite nonnegative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise GateError(f"{label} must be a finite nonnegative number")
    return number


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise GateError(f"{label} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise GateError(f"{label} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise GateError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _sha_field(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise GateError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _negative_scope_is_exact(value: Any) -> bool:
    required = {
        "geff_access": False,
        "learned_detection_access": False,
        "gt_access": False,
        "metric_access": False,
        "submission": False,
    }
    return isinstance(value, Mapping) and dict(value) == required


def validate_package(path: Path) -> dict[str, Any]:
    package = load_json(path, "owner-approval package")
    if (
        package.get("schema") != PACKAGE_SCHEMA
        or package.get("status") != "PREPARED_UNAPPROVED_NO_EXTERNAL_ACTION"
        or package.get("owner_approval_required") is not True
        or package.get("external_actions")
        != {
            "runpod_sign_in": False,
            "runpod_upload": False,
            "runpod_launch": False,
            "competition_submission": False,
        }
    ):
        raise GateError("owner-approval package status/scope drifted")

    pricing = package.get("pricing_snapshot")
    candidates = (
        pricing.get("candidate_gpu_classes")
        if isinstance(pricing, Mapping)
        else None
    )
    if (
        not isinstance(pricing, Mapping)
        or pricing.get("source_url") != OFFICIAL_PRICING_URL
        or pricing.get("currency") != "USD"
        or pricing.get("pod_mode") != "on_demand"
        or pricing.get("time_sensitive") is not True
        or not isinstance(candidates, list)
        or not candidates
    ):
        raise GateError("dated official pricing snapshot is malformed")
    gpu_names: list[str] = []
    for row in candidates:
        if not isinstance(row, Mapping):
            raise GateError("GPU candidate row is malformed")
        name = str(row.get("gpu_class", ""))
        if not name or name in gpu_names:
            raise GateError("GPU candidate names must be nonempty and unique")
        _finite_positive(
            row.get("planning_usd_per_gpu_hour"), f"{name} planning price"
        )
        vram = row.get("vram_gb")
        if type(vram) is not int or vram <= 0:
            raise GateError(f"{name} VRAM is invalid")
        gpu_names.append(name)

    scientific = package.get("scientific_identity")
    required_hashes = (
        "raw_frame_sha256",
        "raw_exact_file_contract_sha256",
        "model_sha256",
        "license_sha256",
        "runtime_manifest_sha256",
        "feature_contract_sha256",
        "benchmark_code_sha256",
    )
    if (
        not isinstance(scientific, Mapping)
        or scientific.get("identity") != "train:44b6_24264f12"
        or scientific.get("timepoint") != 0
        or scientific.get("provenance_label") != "COMPUTED_SEGMENTATIONS"
        or scientific.get("authentic_masks") is not False
        or scientific.get("organizer_provided") is not False
        or scientific.get("h3_authentic_masks") is not False
    ):
        raise GateError("exact scientific benchmark identity drifted")
    for field in required_hashes:
        _sha_field(scientific.get(field), f"scientific_identity.{field}")
    outputs = scientific.get("reference_outputs")
    if not isinstance(outputs, Mapping) or outputs.get("mask_count") != 207:
        raise GateError("reference output contract is malformed")
    for field in (
        "labels_sha256",
        "native_wrfeat12_sha256",
        "regionprops_payload_sha256",
    ):
        _sha_field(outputs.get(field), f"reference_outputs.{field}")

    benchmark = package.get("benchmark_protocol")
    if (
        not isinstance(benchmark, Mapping)
        or benchmark.get("schema") != BENCHMARK_SCHEMA
        or benchmark.get("one_exact_frame") is not True
        or benchmark.get("model_initializations") != 1
        or benchmark.get("post_initialization_repeats") != 2
        or benchmark.get("no_speedup_assumption") is not True
        or benchmark.get("every_authorized_class_must_finish") is not True
        or benchmark.get("production_manifest_eligible") is not False
        or not _negative_scope_is_exact(benchmark.get("negative_scope"))
    ):
        raise GateError("exact-frame benchmark protocol drifted")
    max_billed = benchmark.get("max_billed_seconds_per_class")
    if type(max_billed) is not int or max_billed <= 0:
        raise GateError("benchmark billed-time hard kill is invalid")

    budget = package.get("budget")
    if not isinstance(budget, Mapping):
        raise GateError("owner budget contract is absent")
    benchmark_cap = _finite_positive(
        budget.get("benchmark_cap_usd"), "benchmark cap"
    )
    production_cap = _finite_positive(
        budget.get("production_compute_cap_usd"), "production compute cap"
    )
    reserve = _finite_nonnegative(
        budget.get("noncompute_reserve_usd"), "noncompute reserve"
    )
    total_cap = _finite_positive(budget.get("total_cap_usd"), "total cap")
    if benchmark_cap + production_cap + reserve > total_cap + 1e-9:
        raise GateError("subcaps exceed the total owner cap")

    production = package.get("production_protocol")
    populations = (
        production.get("populations")
        if isinstance(production, Mapping)
        else None
    )
    hard_kills = (
        production.get("hard_kills")
        if isinstance(production, Mapping)
        else None
    )
    if (
        not isinstance(production, Mapping)
        or production.get("requires_local_gate_receipt") is not True
        or production.get("single_measured_gpu_class_only") is not True
        or production.get("spot_or_preemptible_allowed") is not False
        or production.get("partial_shards_manifest_eligible") is not False
        or production.get("contingency_fraction") != 0.05
        or not _negative_scope_is_exact(production.get("remote_negative_scope"))
        or populations != {"cv57": 5700, "held_fit105": 10500}
        or not isinstance(hard_kills, Mapping)
        or type(hard_kills.get("process_seconds")) is not int
        or hard_kills["process_seconds"] <= 0
        or type(hard_kills.get("per_shard_finalize_margin_seconds")) is not int
        or hard_kills["per_shard_finalize_margin_seconds"] < 0
        or hard_kills.get("frame_bound_rule")
        != "ceil(max(two_repeat_seconds) * 1.25)"
        or hard_kills.get("initialization_bound_rule")
        != "ceil(initialization_seconds * 1.25)"
    ):
        raise GateError("atomic production protocol drifted")
    freshness = production.get("fresh_price_max_age_seconds")
    if type(freshness) is not int or freshness <= 0:
        raise GateError("fresh production-price window is invalid")

    sources = package.get("reused_repository_sources")
    if not isinstance(sources, list) or not sources:
        raise GateError("reused source inventory is empty")
    seen_paths: set[str] = set()
    for row in sources:
        if not isinstance(row, Mapping):
            raise GateError("reused source row is malformed")
        relative = str(row.get("path", ""))
        relative_path = Path(relative)
        if (
            not relative
            or relative in seen_paths
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise GateError(f"unsafe reused source path: {relative!r}")
        seen_paths.add(relative)
        expected = _sha_field(row.get("sha256"), f"source {relative}")
        local = ROOT / relative_path
        if not local.is_file() or sha256(local) != expected:
            raise GateError(f"reused repository source drifted: {relative}")
    return package


def validate_approval(
    approval: Mapping[str, Any], package: Mapping[str, Any], package_sha256: str
) -> tuple[list[str], dict[str, float]]:
    if (
        approval.get("schema") != APPROVAL_SCHEMA
        or approval.get("status") != "OWNER_APPROVED"
        or approval.get("owner_approved") is not True
        or approval.get("package_file_sha256") != package_sha256
        or approval.get("benchmark_exact_frame_first") is not True
        or approval.get("conditional_production_only_after_local_gate") is not True
        or approval.get("spot_or_preemptible_allowed") is not False
    ):
        raise GateError(
            "explicit owner approval is absent or does not bind this package"
        )
    if not str(approval.get("owner_identity", "")).strip():
        raise GateError("owner identity is absent")
    _parse_utc(approval.get("owner_approved_at_utc"), "owner_approved_at_utc")
    approved = approval.get("approved_gpu_classes")
    if (
        not isinstance(approved, list)
        or not approved
        or approved != sorted(set(str(item) for item in approved))
    ):
        raise GateError(
            "approved GPU classes must be a sorted nonempty unique list"
        )
    candidates = {
        str(row["gpu_class"])
        for row in package["pricing_snapshot"]["candidate_gpu_classes"]
    }
    if any(name not in candidates for name in approved):
        raise GateError("approval names a GPU class outside the package")
    budget = package["budget"]
    approved_budget = approval.get("approved_budget")
    if not isinstance(approved_budget, Mapping):
        raise GateError("owner-approved budget is absent")
    normalized: dict[str, float] = {}
    for field in (
        "benchmark_cap_usd",
        "production_compute_cap_usd",
        "noncompute_reserve_usd",
        "total_cap_usd",
    ):
        number = _finite_nonnegative(approved_budget.get(field), field)
        if number != float(budget[field]):
            raise GateError(
                f"owner-approved {field} does not match the package"
            )
        normalized[field] = number
    return [str(item) for item in approved], normalized


def build_benchmark_plan(
    package_path: Path, approval_path: Path
) -> dict[str, Any]:
    package = validate_package(package_path)
    package_sha = sha256(package_path)
    approval = load_json(approval_path, "owner approval")
    approved, budget = validate_approval(approval, package, package_sha)
    planning_prices = {
        row["gpu_class"]: float(row["planning_usd_per_gpu_hour"])
        for row in package["pricing_snapshot"]["candidate_gpu_classes"]
    }
    jobs = []
    for gpu_class in approved:
        jobs.append(
            {
                "benchmark_id": (
                    "g04-exact-frame-"
                    + gpu_class.lower().replace(" ", "-")
                ),
                "gpu_class": gpu_class,
                "planning_usd_per_gpu_hour_time_sensitive": planning_prices[
                    gpu_class
                ],
                "protocol": package["benchmark_protocol"],
                "scientific_identity": package["scientific_identity"],
                "terminal_result_schema": BENCHMARK_RESULT_SCHEMA,
            }
        )
    return {
        "schema": BENCHMARK_PLAN_SCHEMA,
        "status": "PASS_OWNER_APPROVAL_BENCHMARKS_ONLY",
        "package_file_sha256": package_sha,
        "approval_file_sha256": sha256(approval_path),
        "approved_gpu_classes": approved,
        "benchmark_cap_usd": budget["benchmark_cap_usd"],
        "jobs": jobs,
        "production_authorized": False,
        "external_actions_performed": False,
    }


def _validate_benchmark_billing(
    receipt: Mapping[str, Any],
    package: Mapping[str, Any],
    gpu_class: str,
) -> float:
    price = receipt.get("price_at_benchmark")
    billing = receipt.get("billing")
    if (
        not isinstance(price, Mapping)
        or price.get("source_url") != OFFICIAL_PRICING_URL
        or price.get("currency") != "USD"
        or price.get("pod_mode") != "on_demand"
        or price.get("spot") is not False
        or not isinstance(billing, Mapping)
    ):
        raise GateError(
            f"{gpu_class} benchmark price/billing receipt is malformed"
        )
    _parse_utc(
        price.get("observed_at_utc"), f"{gpu_class} benchmark price time"
    )
    rate = _finite_positive(
        price.get("usd_per_gpu_hour"), f"{gpu_class} rate"
    )
    billed_seconds = _finite_positive(
        billing.get("gpu_billed_seconds"), f"{gpu_class} billed seconds"
    )
    if (
        billed_seconds
        > package["benchmark_protocol"]["max_billed_seconds_per_class"]
    ):
        raise GateError(
            f"{gpu_class} benchmark exceeded its billed-time hard kill"
        )
    gpu_cost = _finite_nonnegative(
        billing.get("gpu_cost_usd"), f"{gpu_class} GPU cost"
    )
    expected = billed_seconds * rate / 3600.0
    one_second_cost = rate / 3600.0
    if abs(gpu_cost - expected) > max(1e-9, one_second_cost):
        raise GateError(
            f"{gpu_class} GPU cost does not match receipted billing"
        )
    storage_cost = _finite_nonnegative(
        billing.get("storage_cost_usd"), f"{gpu_class} storage cost"
    )
    total_cost = _finite_nonnegative(
        billing.get("total_cost_usd"), f"{gpu_class} total cost"
    )
    if abs(total_cost - (gpu_cost + storage_cost)) > 1e-6:
        raise GateError(
            f"{gpu_class} benchmark total cost does not reconcile"
        )
    return total_cost


def validate_benchmark_receipt(
    receipt: Mapping[str, Any],
    package: Mapping[str, Any],
    expected_gpu_class: str,
) -> dict[str, Any]:
    status = receipt.get("status")
    if (
        receipt.get("schema") != BENCHMARK_RESULT_SCHEMA
        or receipt.get("terminal") is not True
        or receipt.get("gpu_class") != expected_gpu_class
        or receipt.get("scientific_identity")
        != package["scientific_identity"]
        or receipt.get("benchmark_only") is not True
        or receipt.get("production_manifest_eligible") is not False
        or receipt.get("production_launched") is not False
        or receipt.get("submission") is not False
        or not _negative_scope_is_exact(receipt.get("negative_scope"))
        or status
        not in {
            "PASS_EXACT_FRAME_BENCHMARK",
            "FAIL_EXACT_FRAME_BENCHMARK",
        }
    ):
        raise GateError(
            f"{expected_gpu_class} terminal benchmark receipt drifted"
        )
    device = receipt.get("device")
    if (
        not isinstance(device, Mapping)
        or not str(device.get("name", "")).strip()
        or not isinstance(device.get("cuda_capability"), list)
        or len(device["cuda_capability"]) != 2
        or type(device.get("vram_bytes")) is not int
        or device["vram_bytes"] <= 0
    ):
        raise GateError(
            f"{expected_gpu_class} device identity is incomplete"
        )
    cost = _validate_benchmark_billing(
        receipt, package, expected_gpu_class
    )
    if status == "FAIL_EXACT_FRAME_BENCHMARK":
        if not str(receipt.get("terminal_failure", "")).strip():
            raise GateError(
                f"{expected_gpu_class} failed without a terminal reason"
            )
        return {
            "gpu_class": expected_gpu_class,
            "passed": False,
            "cost_usd": cost,
            "terminal_failure": str(receipt["terminal_failure"]),
        }

    initialization = _finite_positive(
        receipt.get("initialization_seconds"),
        f"{expected_gpu_class} initialization seconds",
    )
    repetitions = receipt.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != 2:
        raise GateError(
            f"{expected_gpu_class} must contain exactly two repeats"
        )
    expected_outputs = package["scientific_identity"]["reference_outputs"]
    elapsed: list[float] = []
    for index, row in enumerate(repetitions):
        if not isinstance(row, Mapping) or row.get("repeat") != index:
            raise GateError(
                f"{expected_gpu_class} repeat order drifted"
            )
        seconds = _finite_positive(
            row.get("elapsed_seconds"),
            f"{expected_gpu_class} repeat {index} duration",
        )
        elapsed.append(seconds)
        if (
            row.get("labels_sha256")
            != expected_outputs["labels_sha256"]
            or row.get("native_wrfeat12_sha256")
            != expected_outputs["native_wrfeat12_sha256"]
            or row.get("regionprops_payload_sha256")
            != expected_outputs["regionprops_payload_sha256"]
            or row.get("mask_count") != expected_outputs["mask_count"]
            or row.get("region_count") != expected_outputs["mask_count"]
            or row.get("all_features_finite") is not True
        ):
            raise GateError(
                f"{expected_gpu_class} changed the exact labels/features "
                "signature"
            )
    if receipt.get("repeats_byte_identical") is not True:
        raise GateError(
            f"{expected_gpu_class} repeats are not byte-identical"
        )
    measured = initialization + sum(elapsed)
    if (
        float(receipt["billing"]["gpu_billed_seconds"]) + 1e-9
        < measured
    ):
        raise GateError(
            f"{expected_gpu_class} billed time is shorter than measured work"
        )
    return {
        "gpu_class": expected_gpu_class,
        "passed": True,
        "cost_usd": cost,
        "initialization_seconds": initialization,
        "repeat_seconds": elapsed,
        "worst_seconds_per_frame": max(elapsed),
    }


def validate_production_pricing(
    receipt: Mapping[str, Any],
    authorized: Sequence[str],
    package: Mapping[str, Any],
    now: datetime,
) -> dict[str, float]:
    if (
        receipt.get("schema") != PRICING_SCHEMA
        or receipt.get("status")
        != "RECORDED_PRELAUNCH_PRICE_NO_PURCHASE"
        or receipt.get("source_url") != OFFICIAL_PRICING_URL
        or receipt.get("currency") != "USD"
        or receipt.get("pod_mode") != "on_demand"
        or receipt.get("spot") is not False
        or receipt.get("purchase_made") is not False
    ):
        raise GateError("fresh production-pricing receipt drifted")
    observed = _parse_utc(
        receipt.get("observed_at_utc"), "pricing observed_at_utc"
    )
    age = (now.astimezone(timezone.utc) - observed).total_seconds()
    if (
        age < -60
        or age
        > package["production_protocol"]["fresh_price_max_age_seconds"]
    ):
        raise GateError("production pricing is stale or future-dated")
    rows = receipt.get("rates")
    if not isinstance(rows, list):
        raise GateError("production pricing has no rate rows")
    rates: dict[str, float] = {}
    availability: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise GateError("production price row is malformed")
        name = str(row.get("gpu_class", ""))
        if name in rates:
            raise GateError(f"duplicate production price row: {name}")
        rates[name] = _finite_positive(
            row.get("usd_per_gpu_hour"),
            f"{name} production rate",
        )
        if type(row.get("available")) is not bool:
            raise GateError(f"{name} availability must be boolean")
        availability[name] = bool(row["available"])
    if set(rates) != set(authorized):
        raise GateError(
            "fresh production pricing must cover every authorized class"
        )
    return {
        name: rates[name]
        for name in authorized
        if availability[name]
    }


def project_production(
    benchmark: Mapping[str, Any],
    hourly_rate: float,
    benchmark_spend_usd: float,
    package: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = package["production_protocol"]
    hard = protocol["hard_kills"]
    worst = float(benchmark["worst_seconds_per_frame"])
    initialization = float(benchmark["initialization_seconds"])
    frame_bound = int(math.ceil(worst * 1.25))
    init_bound = int(math.ceil(initialization * 1.25))
    process_seconds = int(hard["process_seconds"])
    finalize = int(hard["per_shard_finalize_margin_seconds"])
    usable = process_seconds - init_bound - finalize
    max_frames = usable // frame_bound
    if max_frames <= 0:
        raise GateError(
            f"{benchmark['gpu_class']} cannot admit one frame per shard"
        )
    populations = protocol["populations"]
    shard_counts = {
        name: int(math.ceil(int(frames) / max_frames))
        for name, frames in populations.items()
    }
    population_projections: dict[str, dict[str, Any]] = {}
    for name, frames_value in populations.items():
        frames = int(frames_value)
        frame_seconds = frames * worst
        overhead_seconds = shard_counts[name] * (init_bound + finalize)
        subtotal_seconds = frame_seconds + overhead_seconds
        risk_adjusted_seconds = subtotal_seconds * (
            1.0 + float(protocol["contingency_fraction"])
        )
        population_projections[name] = {
            "frame_count": frames,
            "atomic_shard_count": shard_counts[name],
            "measured_worst_frame_gpu_hours": frame_seconds / 3600.0,
            "shard_overhead_upper_gpu_hours": overhead_seconds / 3600.0,
            "risk_adjusted_projected_gpu_hours": (
                risk_adjusted_seconds / 3600.0
            ),
            "risk_adjusted_projected_compute_cost_usd": (
                risk_adjusted_seconds * hourly_rate / 3600.0
            ),
        }
    shard_count = sum(shard_counts.values())
    frame_count = sum(int(value) for value in populations.values())
    measured_worst_compute_seconds = frame_count * worst
    shard_overhead_upper_seconds = shard_count * (init_bound + finalize)
    projected_gpu_seconds = (
        measured_worst_compute_seconds + shard_overhead_upper_seconds
    )
    risk_adjusted_gpu_seconds = projected_gpu_seconds * (
        1.0 + float(protocol["contingency_fraction"])
    )
    projected_compute_cost = (
        risk_adjusted_gpu_seconds * hourly_rate / 3600.0
    )
    budget = package["budget"]
    total_owner_exposure = (
        benchmark_spend_usd
        + projected_compute_cost
        + float(budget["noncompute_reserve_usd"])
    )
    compute_cap_pass = projected_compute_cost <= float(
        budget["production_compute_cap_usd"]
    ) + 1e-12
    total_cap_pass = total_owner_exposure <= float(
        budget["total_cap_usd"]
    ) + 1e-12
    return {
        "gpu_class": benchmark["gpu_class"],
        "fresh_usd_per_gpu_hour": hourly_rate,
        "initialization_seconds_measured": initialization,
        "repeat_seconds_measured": list(benchmark["repeat_seconds"]),
        "worst_seconds_per_frame_measured": worst,
        "frame_hard_bound_seconds": frame_bound,
        "initialization_hard_bound_seconds": init_bound,
        "max_frames_per_atomic_shard": max_frames,
        "shard_counts": shard_counts,
        "population_projections": population_projections,
        "total_shard_count": shard_count,
        "measured_worst_frame_gpu_hours": (
            measured_worst_compute_seconds / 3600.0
        ),
        "shard_overhead_upper_gpu_hours": (
            shard_overhead_upper_seconds / 3600.0
        ),
        "risk_adjusted_projected_gpu_hours": (
            risk_adjusted_gpu_seconds / 3600.0
        ),
        "risk_adjusted_production_compute_cost_usd": (
            projected_compute_cost
        ),
        "benchmark_spend_usd": benchmark_spend_usd,
        "noncompute_reserve_usd": float(
            budget["noncompute_reserve_usd"]
        ),
        "total_owner_exposure_usd": total_owner_exposure,
        "production_compute_cap_pass": compute_cap_pass,
        "total_owner_cap_pass": total_cap_pass,
    }


def build_production_unlock(
    package_path: Path,
    approval_path: Path,
    benchmark_paths: Sequence[Path],
    pricing_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    package = validate_package(package_path)
    package_sha = sha256(package_path)
    approval = load_json(approval_path, "owner approval")
    authorized, budget = validate_approval(
        approval, package, package_sha
    )
    indexed_paths: dict[str, Path] = {}
    indexed_receipts: dict[str, dict[str, Any]] = {}
    for path in benchmark_paths:
        receipt = load_json(path, "benchmark result")
        name = str(receipt.get("gpu_class", ""))
        if not name or name in indexed_receipts:
            raise GateError(
                f"duplicate or unnamed benchmark result: {path}"
            )
        indexed_paths[name] = path
        indexed_receipts[name] = receipt
    if set(indexed_receipts) != set(authorized):
        raise GateError(
            "every and only owner-authorized GPU class must finish a benchmark"
        )
    results = [
        validate_benchmark_receipt(
            indexed_receipts[name], package, name
        )
        for name in authorized
    ]
    benchmark_spend = sum(float(row["cost_usd"]) for row in results)
    if benchmark_spend > budget["benchmark_cap_usd"] + 1e-12:
        raise GateError("terminal benchmark spend exceeds the owner cap")
    current = now or datetime.now(timezone.utc)
    rates = validate_production_pricing(
        load_json(pricing_path, "production pricing"),
        authorized,
        package,
        current,
    )
    projections = []
    for row in results:
        name = str(row["gpu_class"])
        if row["passed"] and name in rates:
            projections.append(
                project_production(
                    row, rates[name], benchmark_spend, package
                )
            )
    eligible = [
        row
        for row in projections
        if row["production_compute_cap_pass"]
        and row["total_owner_cap_pass"]
    ]
    if not eligible:
        raise GateError(
            "no exact-parity measured GPU class passes both production cost "
            "caps"
        )
    selected = min(
        eligible,
        key=lambda row: (
            row["total_owner_exposure_usd"],
            row["risk_adjusted_projected_gpu_hours"],
            row["gpu_class"],
        ),
    )
    benchmark_receipts = [
        {
            "gpu_class": name,
            "path": str(indexed_paths[name]),
            "sha256": sha256(indexed_paths[name]),
            "passed": bool(
                next(
                    row
                    for row in results
                    if row["gpu_class"] == name
                )["passed"]
            ),
        }
        for name in authorized
    ]
    return {
        "schema": UNLOCK_SCHEMA,
        "status": "PASS_LOCAL_PRODUCTION_GATE_NO_EXTERNAL_ACTION",
        "package_file_sha256": package_sha,
        "approval_file_sha256": sha256(approval_path),
        "production_pricing_file_sha256": sha256(pricing_path),
        "benchmark_receipts": benchmark_receipts,
        "benchmark_spend_usd": benchmark_spend,
        "all_authorized_classes_benchmarked": True,
        "no_speedup_assumed": True,
        "selected_gpu_class": selected["gpu_class"],
        "selected_projection": selected,
        "all_passing_projections": projections,
        "atomic_production_contract": {
            "population_frames": package["production_protocol"][
                "populations"
            ],
            "hard_kills": package["production_protocol"]["hard_kills"],
            "frame_schema": package["production_protocol"][
                "frame_schema"
            ],
            "shard_receipt_schema": package["production_protocol"][
                "shard_receipt_schema"
            ],
            "partial_shards_manifest_eligible": False,
            "assembler": package["production_protocol"]["assembler"],
            "single_gpu_class": selected["gpu_class"],
        },
        "external_actions_performed": False,
        "runpod_launch_performed": False,
        "submission_performed": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("benchmark-plan")
    plan.add_argument("--package", required=True, type=Path)
    plan.add_argument("--approval", required=True, type=Path)
    plan.add_argument("--output", type=Path)
    unlock = subparsers.add_parser("production-gate")
    unlock.add_argument("--package", required=True, type=Path)
    unlock.add_argument("--approval", required=True, type=Path)
    unlock.add_argument(
        "--benchmark-result",
        action="append",
        required=True,
        type=Path,
    )
    unlock.add_argument(
        "--production-pricing", required=True, type=Path
    )
    unlock.add_argument("--as-of-utc")
    unlock.add_argument("--output", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "benchmark-plan":
            result = build_benchmark_plan(args.package, args.approval)
        else:
            now = (
                _parse_utc(args.as_of_utc, "as-of-utc")
                if args.as_of_utc
                else datetime.now(timezone.utc)
            )
            result = build_production_unlock(
                args.package,
                args.approval,
                args.benchmark_result,
                args.production_pricing,
                now=now,
            )
    except GateError as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_FAIL_CLOSED",
                    "reason": str(exc),
                    "external_actions_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
