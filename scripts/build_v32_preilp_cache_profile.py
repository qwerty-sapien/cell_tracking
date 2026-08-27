#!/usr/bin/env python3
"""Build the fail-closed ship profile for the V32 CV57 cache successor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "dag_prompts/2026-08-20-alpha-funnel-v1/runs"
PARENT_PROFILE = RUNS / "F2B_RELEASE12_PREFLIGHT_PROFILE.json"
EXPECTED_PARENT_PROFILE_SHA256 = (
    "bb173cbabd3f4199411c93ed46fd9d99fc54c41cf4d1c59c04495edfe2da22bc"
)
EXPECTED_ENTRYPOINT = "v32_preilp_cache_kernel.py"
EXPECTED_KERNEL_ID = "onebitbrain/v32-pre-ilp-cache-alpha-funnel-v2"


class ProfileBuildError(RuntimeError):
    """Raised when an exact profile input drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProfileBuildError(f"expected a JSON object: {path}")
    return value


def build_profile(*, kernel_dir: Path, initialized_profile: Path) -> dict[str, Any]:
    if sha256_file(PARENT_PROFILE) != EXPECTED_PARENT_PROFILE_SHA256:
        raise ProfileBuildError("release-12 parent profile hash drift")
    parent = _load_json(PARENT_PROFILE)
    initialized = _load_json(initialized_profile)
    metadata_path = kernel_dir / "kernel-metadata.json"
    source_path = kernel_dir / EXPECTED_ENTRYPOINT
    freeze_path = kernel_dir / "CACHE_SOURCE_FREEZE.json"
    metadata = _load_json(metadata_path)
    freeze = _load_json(freeze_path)
    if metadata.get("id") != EXPECTED_KERNEL_ID:
        raise ProfileBuildError("release-13 kernel id drift")
    if metadata.get("kernel_type") != "script":
        raise ProfileBuildError("V32 successor must be a Kaggle script kernel")
    if freeze.get("role") != "cv57" or len(freeze.get("identities", [])) != 57:
        raise ProfileBuildError("release-13 CV57 source freeze drift")
    if freeze.get("instrumented_kernel_source_sha256") != sha256_file(source_path):
        raise ProfileBuildError("release-13 source hash differs from its freeze")

    profile = copy.deepcopy(parent)
    initialized_kernel = initialized.get("kernel")
    if not isinstance(initialized_kernel, dict):
        raise ProfileBuildError("initialized profile has no kernel binding")
    profile["kernel"] = copy.deepcopy(initialized_kernel)
    profile["kernel"].update(
        {
            "command": ["{python}", "{entrypoint}"],
            "execution_control_env": [
                "KAGGLE_PREFLIGHT_MODE",
                "KAGGLE_PREFLIGHT_UNITS",
            ],
            "expected_metadata": metadata,
            "forbidden_source_fragments": [
                "/kaggle/input/biohub-cell-tracking-during-development",
                "/kaggle/input/competitions/biohub-cell-tracking-during-development/test",
                'COMP_DIR / "test"',
                'inventory_input("competition_test"',
            ],
            "self_contained": True,
            "step0": {
                "assignment_marker": "SHIP_STEP0 = _v32_ship_step0()",
                "heavy_imports": ["numpy", "pandas", "scipy", "torch", "zarr"],
                "heavy_work_marker": "V32_HEAVY_WORK_BEGIN",
            },
            "upload_transport": "kaggle-cli-save-kernel.v1",
        }
    )

    ladder = {
        "tiny": ("tiny_sparse_first_three", "tiny_sparse", 3, 3),
        "stratified": (
            "stratified_both_first_three",
            "stratified_both",
            3,
            6,
        ),
        "stress": ("crowded_stress_first_twelve", "stress_crowded", 12, 12),
        "expansive": (
            "expansive_both_first_fifty_seven",
            "expansive_both",
            57,
            114,
        ),
    }
    scenarios = []
    for original in parent["scenarios"]:
        role = str(original["role"])
        name, mode, frames, workload_units = ladder[role]
        scenario = copy.deepcopy(original)
        scenario["name"] = name
        scenario["workload_units"] = workload_units
        scenario["env"] = {
            "KAGGLE_PREFLIGHT_MODE": mode,
            "KAGGLE_PREFLIGHT_UNITS": str(frames),
        }
        scenario["stdout_markers"] = list(scenario["stdout_markers"])
        scenario["stdout_markers"].append("V32_CPU_PREFLIGHT_ILP_SKIPPED")
        scenarios.append(scenario)
    profile["scenarios"] = scenarios
    profile["runtime"] = copy.deepcopy(parent["runtime"])
    profile["runtime"].update(
        {
            "calibration": [],
            "fixed_remote_overhead_seconds": 300,
            "memory_limit_bytes": 8589934592,
            "min_largest_sample_fraction": 0.02,
            "remote_multiplier": 1.5,
            "total_workload_units": 5700,
        }
    )
    return profile


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--kernel-dir", type=Path, required=True)
    value.add_argument("--initialized-profile", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    profile = build_profile(
        kernel_dir=args.kernel_dir.resolve(),
        initialized_profile=args.initialized_profile.resolve(),
    )
    args.output.write_bytes(canonical_bytes(profile))
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "profile_sha256": sha256_file(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
