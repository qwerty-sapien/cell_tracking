#!/usr/bin/env python3
"""Audit the owner-downloaded CELLECT v6 payload and the current shared kernel."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

from content_addressed_step0 import (
    DiscoveryError,
    DiscoveryPins,
    discover_requests_by_content,
    sha256,
)


PINS = DiscoveryPins(
    bundle_manifest_sha256=(
        "ae3bc9b2bbf5e7b4e544bcb44159a2fc0d6250b3178a81192f2d8e9f867fecd3"
    ),
    request_manifest_sha256=(
        "82111af54e92a4118efbaea4ed0e5aef770f08a252df8e9f868ecea03b81bc83"
    ),
    population_manifest_sha256=(
        "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
    ),
    request_file_count=77,
    request_count=1_391_458,
)
CHECKPOINT_SHA256 = (
    "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5"
)
WHEEL_SHA256 = {
    "donfig-0.8.1.post1-py3-none-any.whl": (
        "2a3175ce74a06109ff9307d90a230f81215cbac9a751f4d1c6194644b8204f9d"
    ),
    "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": (
        "c3a09e22140f2c691f7df26303ff8fa2dadcf26d7d0828398c0bc09b69e5efa3"
    ),
    "zarr-3.2.1-py3-none-any.whl": (
        "f78cdd3d9687ad0e9f9cba2c5683b64f0c52589c19f685eeabe872e93cc0d2c7"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--shared-step0", type=Path, required=True)
    parser.add_argument("--shared-kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path):
    specification = importlib.util.spec_from_file_location(
        "audited_shared_cellect_step0", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load shared step-0 module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def static_kernel_audit(step0_source: str, kernel_source: str) -> dict[str, object]:
    order_markers = {
        "step0_call": kernel_source.find("STEP0 = ("),
        "numpy_import": kernel_source.find("import numpy as np"),
        "torch_import": kernel_source.find("import torch"),
        "pip_call": kernel_source.find("subprocess.check_call"),
        "model_load": kernel_source.find("load_published_backbone"),
    }
    pre_heavy = (
        order_markers["step0_call"] >= 0
        and all(
            order_markers["step0_call"] < order_markers[name]
            for name in ("numpy_import", "torch_import", "pip_call", "model_load")
            if order_markers[name] >= 0
        )
    )
    exact_manifest_name = (
        "_one(input_root, BUNDLE_MANIFEST_NAME)" in step0_source
    )
    manifest_relative_payload = (
        'path = bundle_root / str(record["path"])' in step0_source
    )
    path_derived_class = "action_class = request_path.parent.name" in kernel_source
    path_derived_identity = (
        'identity = request_path.stem.replace("__", ":", 1)' in kernel_source
    )
    return {
        "pre_heavy_order_pass": pre_heavy,
        "order_markers": order_markers,
        "bundle_manifest_discovery_by_expected_content_hash": not exact_manifest_name,
        "request_payload_resolution_by_expected_content_hash": (
            not manifest_relative_payload
        ),
        "consumer_uses_structured_action_class": not path_derived_class,
        "consumer_uses_structured_identity": not path_derived_identity,
        "observed_exact_manifest_filename_dependency": exact_manifest_name,
        "observed_manifest_relative_payload_dependency": manifest_relative_payload,
        "observed_path_derived_action_class": path_derived_class,
        "observed_path_derived_identity": path_derived_identity,
    }


def make_opaque_layout(source: Path, target: Path) -> None:
    for ordinal, path in enumerate(
        candidate for candidate in sorted(source.rglob("*")) if candidate.is_file()
    ):
        bucket = target / f"bucket_{ordinal % 7:02d}"
        bucket.mkdir(parents=True, exist_ok=True)
        (bucket / f"object_{ordinal:04d}.blob").symlink_to(path)


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    prototype = discover_requests_by_content(args.download_root, PINS)
    step0_source = args.shared_step0.read_text()
    kernel_source = args.shared_kernel.read_text()
    static_audit = static_kernel_audit(step0_source, kernel_source)

    shared_step0 = load_module(args.shared_step0)
    with tempfile.TemporaryDirectory(prefix="cellect-v6-shared-step0-") as directory:
        current_result = shared_step0.run_cellect_step0(
            args.download_root,
            Path(directory),
            expected_bundle_manifest_sha256=PINS.bundle_manifest_sha256,
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
            expected_population_manifest_sha256=PINS.population_manifest_sha256,
            expected_wheels=WHEEL_SHA256,
        )

    with tempfile.TemporaryDirectory(prefix="cellect-v6-opaque-") as directory:
        opaque_root = Path(directory)
        make_opaque_layout(args.download_root, opaque_root)
        opaque_prototype = discover_requests_by_content(opaque_root, PINS)
        shared_mangled_error = None
        try:
            with tempfile.TemporaryDirectory(
                prefix="cellect-v6-shared-mangled-work-"
            ) as work:
                shared_step0.run_cellect_step0(
                    opaque_root,
                    Path(work),
                    expected_bundle_manifest_sha256=PINS.bundle_manifest_sha256,
                    expected_checkpoint_sha256=CHECKPOINT_SHA256,
                    expected_population_manifest_sha256=PINS.population_manifest_sha256,
                    expected_wheels=WHEEL_SHA256,
                )
        except shared_step0.Step0InputError as error:
            shared_mangled_error = str(error)
        if shared_mangled_error is None:
            raise DiscoveryError(
                "shared step-0 unexpectedly accepted fully opaque filenames"
            )

    kernel_ready = bool(
        static_audit["pre_heavy_order_pass"]
        and static_audit["bundle_manifest_discovery_by_expected_content_hash"]
        and static_audit["request_payload_resolution_by_expected_content_hash"]
        and static_audit["consumer_uses_structured_action_class"]
        and static_audit["consumer_uses_structured_identity"]
    )
    status = (
        "PRELAUNCH_DISCOVERY_PASS"
        if kernel_ready
        else "FAIL_CLOSED_SHARED_KERNEL_NOT_CONTENT_ADDRESSED"
    )
    result = {
        "schema": "celltrack-alpha.cellect-v6-independent-discovery-audit.v1",
        "status": status,
        "audited_utc": "2026-07-29",
        "elapsed_seconds": time.monotonic() - started,
        "inputs": {
            "download_root": str(args.download_root),
            "shared_step0_path": str(args.shared_step0),
            "shared_step0_sha256": sha256(args.shared_step0),
            "shared_kernel_path": str(args.shared_kernel),
            "shared_kernel_sha256": sha256(args.shared_kernel),
            "pins": {
                "bundle_manifest_sha256": PINS.bundle_manifest_sha256,
                "request_manifest_sha256": PINS.request_manifest_sha256,
                "population_manifest_sha256": PINS.population_manifest_sha256,
                "request_file_count": PINS.request_file_count,
                "request_count": PINS.request_count,
            },
        },
        "prototype_content_addressed_discovery": {
            "status": prototype["status"],
            "inventory_file_count": prototype["inventory_file_count"],
            "request_file_count": prototype["request_file_count"],
            "request_count": prototype["request_count"],
            "class_file_counts": prototype["class_file_counts"],
            "class_action_counts": prototype["class_action_counts"],
            "structured_record_fields": sorted(
                prototype["request_records"][0].keys()
            ),
            "heavy_work_started": prototype["heavy_work_started"],
        },
        "opaque_name_parent_invariance": {
            "prototype_status": opaque_prototype["status"],
            "request_file_count": opaque_prototype["request_file_count"],
            "request_count": opaque_prototype["request_count"],
            "shared_step0_status": "FAIL_CLOSED",
            "shared_step0_error": shared_mangled_error,
        },
        "current_shared_step0_actual_download": {
            "status": current_result["status"],
            "request_file_count": current_result["request_file_count"],
            "request_count": current_result["request_count"],
            "heavy_work_started": current_result["heavy_work_started"],
        },
        "shared_kernel_contract_audit": static_audit,
        "root_cause": (
            "The v6 payload is complete and uniquely content-addressable, but the "
            "shared step-0 locates the bundle manifest by canonical filename and "
            "request payloads by manifest-relative paths. The kernel consumer then "
            "reconstructs action_class and identity from transport paths instead of "
            "using frozen manifest metadata."
        ),
        "orchestrator_patch_plan": [
            "Port this prototype's unique SHA-256 plus byte-count resolver into the shared pure-stdlib step 0.",
            "Pin the request-manifest SHA-256 in the kernel and discover it by content, not filename.",
            "Return structured request records carrying action_class, identity, action_count, bytes, sha256, and resolved_path.",
            "Change the kernel consumer to use record action_class/identity and resolved_path; never infer semantics from parent/stem.",
            "Add opaque-name/parent, missing, empty, duplicate, and wrong-hash tests to the shared 21-test suite.",
            "Run the repaired shared step 0 on the owner-downloaded v6 payload before any authorization or launch.",
        ],
        "projection": {
            "status": "NOT_AVAILABLE_NO_V6_RUNTIME_PROJECTION",
            "hours": None,
            "note": (
                "The historical 0.0545-second step-0 pass is discovery timing, not "
                "an authenticated end-to-end GPU projection."
            ),
        },
        "decision": {
            "n2_clears_n3_request": kernel_ready,
            "kernel_launch_authorized": False,
            "required_next_barrier": "ORCHESTRATOR_SHARED_KERNEL_REPAIR_AND_REVERIFY",
        },
        "forbidden_actions": {
            "authenticated_or_network_command": False,
            "kernel_launch": False,
            "authorization": False,
            "gpu_debit": False,
            "training": False,
            "full_pool_cache": False,
            "graph": False,
            "family_6bba": False,
            "cv57": False,
            "submission": False,
            "runpod": False,
            "tier_move": False,
            "dead": False,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if kernel_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
