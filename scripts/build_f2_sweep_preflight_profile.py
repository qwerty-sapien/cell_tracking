#!/usr/bin/env python3
"""Derive an exact ship-certifier profile for a frozen F2 D/E sweep package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_PROFILE = (
    ROOT
    / "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
    "F2B_RELEASE12_PREFLIGHT_PROFILE.json"
)
V9_D_REJECTION_REL = (
    "f2_d_v9_native_interactive_cert/downloaded_native_t4/extracted/"
    "F2_DIRECT_D_SWEEP_V9_NATIVE_CERTIFICATION.json"
)
V9_D_REJECTION_SHA256 = (
    "5370c55e18d0ec25ec673258e44a9b8ef40e15ff75123c53ecd10362bcfc174c"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def inventory_sha256(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"package inventory contains symlink: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return hashlib.sha256(canonical_json(rows)).hexdigest()


def build_profile(
    *,
    package_dir: Path,
    profile_path: Path,
    family: str,
    bounded_sequential_v10: bool = False,
) -> dict[str, Any]:
    profile = copy.deepcopy(json.loads(BASE_PROFILE.read_text(encoding="utf-8")))
    metadata_path = package_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entrypoint = package_dir / metadata["code_file"]
    first_candidate = "D_det0.9375" if family == "D" else "E_sew0.10"
    profile["kernel"].update(
        {
            "artifact_sha256": sha256_file(entrypoint),
            "directory": package_dir.relative_to(profile_path.parent).as_posix(),
            "entrypoint": entrypoint.name,
            "expected_metadata": metadata,
            "metadata_sha256": sha256_file(metadata_path),
            "package_inventory_sha256": inventory_sha256(package_dir),
        }
    )
    profile["kernel"]["execution_control_env"] = [
        "KAGGLE_PREFLIGHT_MODE",
        "KAGGLE_PREFLIGHT_UNITS",
    ]
    if bounded_sequential_v10:
        profile["kernel"]["execution_control_env"].append(
            "KAGGLE_PREFLIGHT_CANDIDATES"
        )
    total_candidates = 7 if family == "D" else 9
    profile["runtime"]["total_workload_units"] = total_candidates * 6 * 100
    profile["runtime"]["fixed_remote_overhead_seconds"] = 600
    # Preflight deliberately executes only the first candidate because the
    # dependency/export/ILP transport is shared by every frozen candidate.
    # Two-frame prefixes on both canaries legitimately produce zero
    # postprocessed nodes. Three frames are the smallest empirically viable
    # prefix; the largest case honestly covers at least 2% of the full E sweep.
    preflight_frames = (
        {
            "tiny_sparse": 3,
            "stratified_both": 3,
            "stress_crowded": 100,
            "expansive_both": 100,
        }
        if bounded_sequential_v10
        else {
            "tiny_sparse": 3,
            "stratified_both": 3,
            "stress_crowded": 12,
            "expansive_both": 54,
        }
    )
    preflight_identities = {
        "tiny_sparse": 1,
        "stratified_both": 2,
        "stress_crowded": 1,
        "expansive_both": 2,
    }
    preflight_candidates = (
        {
            "tiny_sparse": 1,
            "stratified_both": 1,
            "stress_crowded": 1,
            "expansive_both": total_candidates,
        }
        if bounded_sequential_v10
        else {
            "tiny_sparse": 1,
            "stratified_both": 1,
            "stress_crowded": 1,
            "expansive_both": 1,
        }
    )
    for scenario in profile["scenarios"]:
        mode = scenario["env"]["KAGGLE_PREFLIGHT_MODE"]
        frames = preflight_frames[mode]
        candidate_count = preflight_candidates[mode]
        scenario["env"]["KAGGLE_PREFLIGHT_UNITS"] = str(frames)
        if bounded_sequential_v10:
            scenario["env"]["KAGGLE_PREFLIGHT_CANDIDATES"] = str(candidate_count)
        scenario["name"] = (
            f"{mode}_first_{frames}_candidates_{candidate_count}"
            if bounded_sequential_v10
            else f"{mode}_first_{frames}"
        )
        # The orchestrator selects CANDIDATES[:1] under preflight, so receipt
        # units are exactly frames x selected identities, never projected
        # across candidates that were not executed.
        scenario["workload_units"] = (
            frames * preflight_identities[mode] * candidate_count
        )
        if bounded_sequential_v10:
            scenario["repeats"] = 3 if mode == "tiny_sparse" else 2
            scenario["timeout_seconds"] = {
                "tiny_sparse": 1800,
                "stratified_both": 2700,
                "stress_crowded": 7200,
                "expansive_both": 21600,
            }[mode]
        scenario["completion"] = {
            "path": "F2_SWEEP_COMPLETION.json",
            "status_pointer": "/status",
            "success_values": ["PASS"],
            "units_pointer": "/completed_units",
        }
        dataset = (
            "44b6_18ced818"
            if scenario["env"]["KAGGLE_PREFLIGHT_MODE"] == "stress_crowded"
            else "44b6_d754aa59"
        )
        scenario["required_outputs"] = [
            {
                "kind": "json",
                "path": "F2_SWEEP_COMPLETION.json",
                "repeatability": "none",
            },
            {
                "kind": "json",
                "path": (
                    f"f2_direct_{family.lower()}_sweep_v1/{first_candidate}/"
                    "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json"
                ),
                "repeatability": "none",
            },
            {
                "kind": "csv",
                "path": (
                    f"f2_direct_{family.lower()}_sweep_v1/{first_candidate}/"
                    "V32_FINAL_GRAPH_ROWS.csv"
                ),
                "repeatability": "byte",
                "header": [
                    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
                    "source_id", "target_id",
                ],
                "min_rows": 1,
            },
            {
                "kind": "json",
                "path": (
                    f"f2_direct_{family.lower()}_sweep_v1/{first_candidate}/"
                    f"v32_preilp_cache_v1/datasets/{dataset}/PRE_ILP_RECEIPT.json"
                ),
                "repeatability": "none",
            },
        ]
        scenario["stdout_markers"] = [
            "V32_RELEASE_IMPORT_PROBE_COMPLETE",
            "V32_HEAVY_WORK_BEGIN",
            "V32_STANDALONE_DISPLAY_SINK",
            "V32_SHIP_PREFLIGHT_COMPLETE",
            "V32_CACHE_NO_SUBMISSION_CONFIRMED",
            "V32_FINAL_GRAPH_ROWS_PRESERVED",
            "F2_SWEEP_CANDIDATE_COMPLETE",
            "F2_DIRECT_SWEEP_COMPLETE",
        ]
    profile["adversarial"]["success_markers"] = ["F2_DIRECT_SWEEP_COMPLETE"]
    if bounded_sequential_v10:
        freeze_path = package_dir / "F2_SWEEP_SOURCE_FREEZE.json"
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        proof = freeze.get("runtime_envelope_proof")
        if not isinstance(proof, dict) or proof.get("status") != "PASS_SOURCE_BOUND":
            raise ValueError("v10 package lacks the source-bound runtime envelope proof")
        expansive = next(
            row for row in profile["scenarios"] if row["role"] == "expansive"
        )
        stress = next(row for row in profile["scenarios"] if row["role"] == "stress")
        profile["runtime"].update(
            {
                "min_runs": 9,
                "projection_model": {
                    "schema": "celltrack-alpha.f2-v10-runtime-projection.v1",
                    "mode": "bounded_sequential_replay.v1",
                    "warm_min_repeats_per_scenario": 2,
                    "startup": {
                        "stdout_marker": (
                            "V32_FULL_SUPPORT_INSTALL_PLAN_BOUND="
                            "44d759547070cca311a89b3676e3a949045f6b242443f2263d39b8593c3553d1"
                        ),
                        "max_observed_occurrences": 1,
                        "frozen_upper_seconds": 97.45017849700002,
                        "parent_receipt": {
                            "path": V9_D_REJECTION_REL,
                            "sha256": V9_D_REJECTION_SHA256,
                            "run_index": 0,
                            "cold_elapsed_seconds": 97.45017849700002,
                            "warm_elapsed_seconds": 22.775516248999907,
                        },
                    },
                    "expansive_replay": {
                        "scenario": expansive["name"],
                        "observed_workload_units": expansive["workload_units"],
                        "full_workload_multiplier": 3,
                        "candidate_count": total_candidates,
                        "identity_count": 2,
                        "frames_per_identity": 100,
                    },
                    "bounded_memory": {
                        "mode": "max_observed_source_bound.v1",
                        "stress_scenario": stress["name"],
                        "stress_candidate_count": 1,
                        "stress_identity_count": 1,
                        "stress_frames_per_identity": 100,
                        "source_freeze_path": freeze_path.relative_to(
                            profile_path.parent
                        ).as_posix(),
                        "source_freeze_sha256": sha256_file(freeze_path),
                        "source_proof": proof,
                    },
                },
            }
        )
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return profile


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--package-dir", type=Path, required=True)
    value.add_argument("--profile", type=Path, required=True)
    value.add_argument("--family", choices=("D", "E"), required=True)
    value.add_argument("--bounded-sequential-v10", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    profile = build_profile(
        package_dir=args.package_dir.resolve(),
        profile_path=args.profile.resolve(),
        family=args.family,
        bounded_sequential_v10=args.bounded_sequential_v10,
    )
    print(json.dumps({
        "status": "PROFILE_BUILT",
        "profile": str(args.profile.resolve()),
        "artifact_sha256": profile["kernel"]["artifact_sha256"],
        "package_inventory_sha256": profile["kernel"]["package_inventory_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
