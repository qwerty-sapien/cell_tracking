from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from scripts.g04_runpod_gate import (
    APPROVAL_SCHEMA,
    BENCHMARK_RESULT_SCHEMA,
    GateError,
    OFFICIAL_PRICING_URL,
    PACKAGE_SCHEMA,
    PRICING_SCHEMA,
    build_benchmark_plan,
    build_production_unlock,
    canonical_sha256,
    project_production,
    sha256,
    validate_package,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATH = (
    ROOT
    / "reports/g04/lanes/computed_segmentation/"
    "RUNPOD_OWNER_APPROVAL_PACKAGE.json"
)
APPROVAL_TEMPLATE_PATH = (
    ROOT
    / "reports/g04/lanes/computed_segmentation/"
    "RUNPOD_OWNER_APPROVAL_TEMPLATE.json"
)
NOW = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
NEGATIVE_SCOPE = {
    "geff_access": False,
    "gt_access": False,
    "learned_detection_access": False,
    "metric_access": False,
    "submission": False,
}


def write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    return path


def owner_approval(classes: list[str]) -> dict[str, object]:
    package = json.loads(PACKAGE_PATH.read_text(encoding="ascii"))
    return {
        "schema": APPROVAL_SCHEMA,
        "status": "OWNER_APPROVED",
        "owner_approved": True,
        "owner_identity": "test-owner",
        "owner_approved_at_utc": "2026-08-03T11:00:00Z",
        "package_file_sha256": sha256(PACKAGE_PATH),
        "approved_gpu_classes": sorted(classes),
        "benchmark_exact_frame_first": True,
        "conditional_production_only_after_local_gate": True,
        "spot_or_preemptible_allowed": False,
        "approved_budget": deepcopy(package["budget"]),
    }


def benchmark_receipt(
    package: dict[str, object],
    gpu_class: str,
    *,
    rate: float,
    initialization: float = 30.0,
    repeats: tuple[float, float] = (43.0, 42.0),
    billed_seconds: float = 120.0,
    storage_cost: float = 0.0,
    status: str = "PASS_EXACT_FRAME_BENCHMARK",
) -> dict[str, object]:
    gpu_cost = billed_seconds * rate / 3600.0
    receipt: dict[str, object] = {
        "schema": BENCHMARK_RESULT_SCHEMA,
        "status": status,
        "terminal": True,
        "gpu_class": gpu_class,
        "scientific_identity": deepcopy(package["scientific_identity"]),
        "benchmark_only": True,
        "production_manifest_eligible": False,
        "production_launched": False,
        "submission": False,
        "negative_scope": deepcopy(NEGATIVE_SCOPE),
        "device": {
            "name": f"RunPod {gpu_class}",
            "cuda_capability": [8, 6],
            "vram_bytes": 24_000_000_000,
        },
        "price_at_benchmark": {
            "source_url": OFFICIAL_PRICING_URL,
            "currency": "USD",
            "pod_mode": "on_demand",
            "spot": False,
            "observed_at_utc": "2026-08-03T11:00:00Z",
            "usd_per_gpu_hour": rate,
        },
        "billing": {
            "gpu_billed_seconds": billed_seconds,
            "gpu_cost_usd": gpu_cost,
            "storage_cost_usd": storage_cost,
            "total_cost_usd": gpu_cost + storage_cost,
        },
    }
    if status == "FAIL_EXACT_FRAME_BENCHMARK":
        receipt["terminal_failure"] = "exact output parity failed"
        return receipt
    reference = package["scientific_identity"]["reference_outputs"]
    receipt.update(
        {
            "initialization_seconds": initialization,
            "repeats_byte_identical": True,
            "repetitions": [
                {
                    "repeat": index,
                    "elapsed_seconds": elapsed,
                    "labels_sha256": reference["labels_sha256"],
                    "native_wrfeat12_sha256": reference[
                        "native_wrfeat12_sha256"
                    ],
                    "regionprops_payload_sha256": reference[
                        "regionprops_payload_sha256"
                    ],
                    "mask_count": reference["mask_count"],
                    "region_count": reference["mask_count"],
                    "all_features_finite": True,
                }
                for index, elapsed in enumerate(repeats)
            ],
        }
    )
    return receipt


def pricing_receipt(
    rows: dict[str, tuple[float, bool]],
    *,
    observed_at: str = "2026-08-03T11:30:00Z",
) -> dict[str, object]:
    return {
        "schema": PRICING_SCHEMA,
        "status": "RECORDED_PRELAUNCH_PRICE_NO_PURCHASE",
        "source_url": OFFICIAL_PRICING_URL,
        "currency": "USD",
        "pod_mode": "on_demand",
        "spot": False,
        "purchase_made": False,
        "observed_at_utc": observed_at,
        "rates": [
            {
                "gpu_class": name,
                "usd_per_gpu_hour": rate,
                "available": available,
            }
            for name, (rate, available) in rows.items()
        ],
    }


def test_package_is_exact_and_unsigned_template_fails_closed() -> None:
    package = validate_package(PACKAGE_PATH)
    assert package["schema"] == PACKAGE_SCHEMA
    assert package["scientific_identity"]["reference_outputs"]["mask_count"] == 207
    assert (
        canonical_sha256(package["production_protocol"]["populations"])
        == "013447ccf582aff5929a271db4ddbc48199903eead6da3b5b827c068251096e9"
    )
    with pytest.raises(GateError, match="explicit owner approval"):
        build_benchmark_plan(PACKAGE_PATH, APPROVAL_TEMPLATE_PATH)


def test_owner_approval_emits_benchmarks_only_for_every_authorized_class(
    tmp_path: Path,
) -> None:
    classes = ["A40", "L40S", "RTX 4090"]
    approval_path = write_json(
        tmp_path / "approval.json", owner_approval(classes)
    )
    result = build_benchmark_plan(PACKAGE_PATH, approval_path)
    assert result["status"] == "PASS_OWNER_APPROVAL_BENCHMARKS_ONLY"
    assert [row["gpu_class"] for row in result["jobs"]] == sorted(classes)
    assert result["production_authorized"] is False
    assert result["external_actions_performed"] is False
    assert all(row["protocol"]["no_speedup_assumption"] for row in result["jobs"])


def test_measured_a40_parity_projection_passes_and_unlock_is_local_only(
    tmp_path: Path,
) -> None:
    package = validate_package(PACKAGE_PATH)
    approval_path = write_json(
        tmp_path / "approval.json", owner_approval(["A40"])
    )
    benchmark_path = write_json(
        tmp_path / "a40.json",
        benchmark_receipt(package, "A40", rate=0.44),
    )
    price_path = write_json(
        tmp_path / "price.json",
        pricing_receipt({"A40": (0.44, True)}),
    )
    result = build_production_unlock(
        PACKAGE_PATH,
        approval_path,
        [benchmark_path],
        price_path,
        now=NOW,
    )
    projection = result["selected_projection"]
    assert result["status"] == "PASS_LOCAL_PRODUCTION_GATE_NO_EXTERNAL_ACTION"
    assert result["selected_gpu_class"] == "A40"
    assert projection["shard_counts"] == {"cv57": 9, "held_fit105": 17}
    assert projection["population_projections"]["cv57"]["frame_count"] == 5700
    assert (
        projection["population_projections"]["held_fit105"]["frame_count"]
        == 10500
    )
    assert projection["risk_adjusted_production_compute_cost_usd"] < 95
    assert projection["total_owner_exposure_usd"] < 100
    assert result["runpod_launch_performed"] is False
    assert result["submission_performed"] is False
    assert result["atomic_production_contract"][
        "partial_shards_manifest_eligible"
    ] is False


def test_every_authorized_class_must_have_a_terminal_benchmark(
    tmp_path: Path,
) -> None:
    package = validate_package(PACKAGE_PATH)
    approval_path = write_json(
        tmp_path / "approval.json", owner_approval(["A40", "L40S"])
    )
    benchmark_path = write_json(
        tmp_path / "a40.json",
        benchmark_receipt(package, "A40", rate=0.44),
    )
    price_path = write_json(
        tmp_path / "price.json",
        pricing_receipt(
            {"A40": (0.44, True), "L40S": (0.99, True)}
        ),
    )
    with pytest.raises(GateError, match="every and only"):
        build_production_unlock(
            PACKAGE_PATH,
            approval_path,
            [benchmark_path],
            price_path,
            now=NOW,
        )


def test_failed_authorized_class_is_terminal_but_not_selectable(
    tmp_path: Path,
) -> None:
    package = validate_package(PACKAGE_PATH)
    approval_path = write_json(
        tmp_path / "approval.json", owner_approval(["A40", "L40S"])
    )
    a40 = write_json(
        tmp_path / "a40.json",
        benchmark_receipt(package, "A40", rate=0.44),
    )
    l40s = write_json(
        tmp_path / "l40s.json",
        benchmark_receipt(
            package,
            "L40S",
            rate=0.99,
            status="FAIL_EXACT_FRAME_BENCHMARK",
        ),
    )
    price = write_json(
        tmp_path / "price.json",
        pricing_receipt(
            {"A40": (0.44, True), "L40S": (0.99, True)}
        ),
    )
    result = build_production_unlock(
        PACKAGE_PATH,
        approval_path,
        [a40, l40s],
        price,
        now=NOW,
    )
    assert result["selected_gpu_class"] == "A40"
    assert {row["gpu_class"]: row["passed"] for row in result["benchmark_receipts"]} == {
        "A40": True,
        "L40S": False,
    }


def test_changed_label_signature_fails_closed(
    tmp_path: Path,
) -> None:
    package = validate_package(PACKAGE_PATH)
    approval_path = write_json(
        tmp_path / "approval.json", owner_approval(["A40"])
    )
    receipt = benchmark_receipt(package, "A40", rate=0.44)
    receipt["repetitions"][1]["labels_sha256"] = "0" * 64
    benchmark_path = write_json(tmp_path / "a40.json", receipt)
    price_path = write_json(
        tmp_path / "price.json",
        pricing_receipt({"A40": (0.44, True)}),
    )
    with pytest.raises(GateError, match="changed the exact"):
        build_production_unlock(
            PACKAGE_PATH,
            approval_path,
            [benchmark_path],
            price_path,
            now=NOW,
        )


def test_stale_price_receipt_fails_closed(tmp_path: Path) -> None:
    package = validate_package(PACKAGE_PATH)
    approval_path = write_json(
        tmp_path / "approval.json", owner_approval(["A40"])
    )
    benchmark_path = write_json(
        tmp_path / "a40.json",
        benchmark_receipt(package, "A40", rate=0.44),
    )
    price_path = write_json(
        tmp_path / "price.json",
        pricing_receipt(
            {"A40": (0.44, True)},
            observed_at="2026-08-03T09:00:00Z",
        ),
    )
    with pytest.raises(GateError, match="pricing is stale"):
        build_production_unlock(
            PACKAGE_PATH,
            approval_path,
            [benchmark_path],
            price_path,
            now=NOW,
        )


def test_no_speedup_a100_fails_preregistered_cost_cap(
    tmp_path: Path,
) -> None:
    package = validate_package(PACKAGE_PATH)
    approval_path = write_json(
        tmp_path / "approval.json", owner_approval(["A100 PCIe"])
    )
    benchmark_path = write_json(
        tmp_path / "a100.json",
        benchmark_receipt(package, "A100 PCIe", rate=1.39),
    )
    price_path = write_json(
        tmp_path / "price.json",
        pricing_receipt({"A100 PCIe": (1.39, True)}),
    )
    with pytest.raises(GateError, match="passes both production cost caps"):
        build_production_unlock(
            PACKAGE_PATH,
            approval_path,
            [benchmark_path],
            price_path,
            now=NOW,
        )


def test_frame_hard_bound_cannot_fit_one_frame() -> None:
    package = validate_package(PACKAGE_PATH)
    with pytest.raises(GateError, match="cannot admit one frame"):
        project_production(
            {
                "gpu_class": "synthetic",
                "initialization_seconds": 10.0,
                "repeat_seconds": [40_000.0, 40_000.0],
                "worst_seconds_per_frame": 40_000.0,
            },
            0.44,
            0.0,
            package,
        )


def test_benchmark_spend_and_unavailable_class_fail_closed(
    tmp_path: Path,
) -> None:
    package = validate_package(PACKAGE_PATH)
    approval_path = write_json(
        tmp_path / "approval.json", owner_approval(["A40"])
    )
    costly = benchmark_receipt(
        package,
        "A40",
        rate=0.44,
        storage_cost=2.01,
    )
    benchmark_path = write_json(tmp_path / "costly.json", costly)
    price_path = write_json(
        tmp_path / "price.json",
        pricing_receipt({"A40": (0.44, True)}),
    )
    with pytest.raises(GateError, match="benchmark spend"):
        build_production_unlock(
            PACKAGE_PATH,
            approval_path,
            [benchmark_path],
            price_path,
            now=NOW,
        )

    normal_path = write_json(
        tmp_path / "normal.json",
        benchmark_receipt(package, "A40", rate=0.44),
    )
    unavailable_path = write_json(
        tmp_path / "unavailable.json",
        pricing_receipt({"A40": (0.44, False)}),
    )
    with pytest.raises(GateError, match="passes both production cost caps"):
        build_production_unlock(
            PACKAGE_PATH,
            approval_path,
            [normal_path],
            unavailable_path,
            now=NOW,
        )
