#!/usr/bin/env python3
"""Reproduce the CELLECT v7 targeted step-0 and launch-control acceptance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent / "cellect_attempt_v7"
EXPANDED = Path("/private/tmp/cellect-r7-v6-cmux-roundtrip.20260729/download")
EXPECTED_POPULATION = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)


def load_kernel():
    path = HERE / "kaggle_cellect_v7.py"
    spec = importlib.util.spec_from_file_location("verify_cellect_v7", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load CELLECT v7 kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    build = json.loads((HERE / "BUILD_RECEIPT.json").read_text())
    controls = json.loads((HERE / "TASKS_1_3_VALIDATION.json").read_text())
    remote = json.loads((HERE / "REMOTE_PREFLIGHT_AUDIT.json").read_text())
    budget = json.loads(
        (
            ROOT / "reports/g02/screening/tiered_surface/kaggle_budget.json"
        ).read_text()
    )
    if (
        build.get("status") != "PASS"
        or build.get("population_manifest_sha256") != EXPECTED_POPULATION
        or build.get("request_count") != 1_391_458
        or build.get("request_file_count") != 77
        or build.get("checkpoint_container_roundtrip_required") is not False
        or controls.get("status") != "PASS_BEFORE_ANY_PUSH"
        or remote.get("status") != "PASS_EXCLUSIVE_ZERO_EXISTING_VERSION"
        or remote.get("campaign_inflight") != []
        or remote.get("target_version_count") != 0
        or budget.get("accounting_authority", {}).get("local_file_role")
        != "derived mirror plus named-run audit trail"
    ):
        raise SystemExit("CELLECT v7 frozen control receipt drift")
    kernel = load_kernel()
    with tempfile.TemporaryDirectory() as directory:
        step0 = kernel.run_cellect_step0(EXPANDED, Path(directory))
    if (
        step0["elapsed_seconds"] >= 5
        or step0["declared_paths_hashed"] != 6
        or step0["competition_paths_visited"] != 0
        or step0["request_count"] <= 0
        or step0["population_manifest_sha256"] != EXPECTED_POPULATION
        or not step0["checkpoint_member_sha256_verified"]
        or step0["checkpoint_container_sha256_verified"]
    ):
        raise SystemExit("CELLECT v7 targeted step-0 verification failed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "elapsed_seconds": step0["elapsed_seconds"],
                "declared_paths_hashed": step0["declared_paths_hashed"],
                "competition_paths_visited": step0["competition_paths_visited"],
                "request_count": step0["request_count"],
                "authoritative_remaining_h": budget["remaining_kernel_h"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
