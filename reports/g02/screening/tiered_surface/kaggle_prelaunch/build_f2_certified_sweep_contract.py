#!/usr/bin/env python3
"""Build an F2 D/E wrapper contract only after every inherited gate passes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import kaggle_prelaunch_guard as guard


def _reference(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "sha256": guard.sha256(path),
    }


def _child_source(code: Path) -> str:
    tree = ast.parse(code.read_text(encoding="utf-8"), filename=str(code))
    value = guard._literal_assignment(tree, "CHILD_SOURCE", "F2 sweep CHILD_SOURCE")
    if not isinstance(value, str) or not value:
        raise guard.GuardError("F2 sweep CHILD_SOURCE is empty")
    return value


def build_contract(
    *,
    family: str,
    kernel_dir: Path,
    certificate: Path,
    profile: Path,
    child_kernel_dir: Path | None,
    child_visual_contract: Path | None,
    project_root: Path,
    native_t4_visual_contract: Path | None = None,
    downloaded_visual_zip: Path | None = None,
) -> dict[str, Any]:
    family = family.upper()
    native_route_requested = (
        native_t4_visual_contract is not None or downloaded_visual_zip is not None
    )
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata = guard.load_json(metadata_path)
    native_profile: str | None = None
    if native_route_requested:
        if (
            family != "D"
            or native_t4_visual_contract is None
            or downloaded_visual_zip is None
            or child_kernel_dir is not None
            or child_visual_contract is not None
        ):
            raise guard.GuardError(
                "native route requires exact D, production visual contract, downloaded ZIP, and no v5 child evidence"
            )
        route_matches = [
            (key, route)
            for key, route in guard.F2_D_NATIVE_VISUAL_ROUTES.items()
            if metadata.get("id") == route["bindings"]["kernel_id"]
        ]
        if len(route_matches) != 1:
            raise guard.GuardError(
                "native route package must match exactly one registered D route"
            )
        route_key, route = route_matches[0]
        expected = dict(guard.F2_SWEEP_V6_KERNELS["D"])
        expected["kernel_id"] = route["bindings"]["kernel_id"]
        native_profile = route["execution_profile"]
        if route_key != "v7":
            guard._require_f2_d_native_route_enabled(
                route_key, project_root=project_root
            )
    else:
        if child_kernel_dir is None or child_visual_contract is None:
            raise guard.GuardError(
                "v5 requires the exact child package and child visual contract"
            )
        expected = guard.F2_SWEEP_KERNELS.get(family)
    if expected is None:
        raise guard.GuardError(
            "family/profile must be v5 D/E or one registered native D route"
        )
    code_path = kernel_dir / str(metadata.get("code_file", ""))
    if not code_path.is_file():
        raise guard.GuardError("declared F2 sweep code is missing")
    freeze_path = kernel_dir / "F2_SWEEP_SOURCE_FREEZE.json"
    if not freeze_path.is_file():
        raise guard.GuardError("F2 sweep source freeze is missing")
    child_source = _child_source(code_path)
    candidates = [dict(item) for item in guard.F2_SWEEP_CANDIDATES[family]]
    package = guard._package_inventory(kernel_dir, project_root)
    contract = {
        "schema": guard.F2_CERTIFIED_SWEEP_CONTRACT_SCHEMA,
        "execution_profile": guard.CERTIFIED_SWEEP_WRAPPER_EXECUTION_PROFILE,
        "family": family,
        "kernel_id": expected["kernel_id"],
        "remote_preflight_kernel_id": expected["kernel_id"],
        "lane_owner": expected["lane_owner"],
        "model_execution": True,
        "code_sha256": guard.sha256(code_path),
        "metadata_sha256": guard.sha256(metadata_path),
        "package_tree_sha256": package["tree_sha256"],
        "package_inventory_sha256": guard._ship_package_inventory_sha256(kernel_dir),
        "embedded_child_sha256": hashlib.sha256(
            child_source.encode("utf-8")
        ).hexdigest(),
        "source_freeze": _reference(freeze_path, project_root),
        "frozen_candidates_sha256": guard._canonical_sha(candidates),
        "preregistered_visual_identities": list(guard.F2_SWEEP_IDENTITIES),
        "same_run_control_id": "CONTROL_V32",
        "submission_output_permitted": False,
        "final_graph_rows_required": True,
        "arbitrary_candidate_or_config_surface_permitted": False,
        "ship_kaggle_certificate": _reference(certificate, project_root),
        "ship_kaggle_profile": _reference(profile, project_root),
    }
    if native_route_requested:
        contract.update(
            {
                "visual_evidence_profile": native_profile,
                "native_t4_visual_production_contract": _reference(
                    native_t4_visual_contract, project_root
                ),
                "downloaded_native_visual_zip": _reference(
                    downloaded_visual_zip, project_root
                ),
            }
        )
    else:
        contract["child_visual_evidence"] = {
            "kernel_dir": str(
                child_kernel_dir.resolve().relative_to(project_root.resolve())
            ),
            "contract": _reference(child_visual_contract, project_root),
        }
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("D", "E", "d", "e"), required=True)
    parser.add_argument("--kernel-dir", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--child-kernel-dir", type=Path)
    parser.add_argument("--child-visual-contract", type=Path)
    parser.add_argument("--native-t4-visual-contract", type=Path)
    parser.add_argument("--downloaded-visual-zip", type=Path)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=guard.PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    try:
        contract = build_contract(
            family=args.family,
            kernel_dir=args.kernel_dir.resolve(),
            certificate=args.certificate.resolve(),
            profile=args.profile.resolve(),
            child_kernel_dir=(
                args.child_kernel_dir.resolve() if args.child_kernel_dir else None
            ),
            child_visual_contract=(
                args.child_visual_contract.resolve()
                if args.child_visual_contract
                else None
            ),
            project_root=root,
            native_t4_visual_contract=(
                args.native_t4_visual_contract.resolve()
                if args.native_t4_visual_contract
                else None
            ),
            downloaded_visual_zip=(
                args.downloaded_visual_zip.resolve()
                if args.downloaded_visual_zip
                else None
            ),
        )
        result = guard.check_kernel(
            args.kernel_dir.resolve(),
            contract,
            guard.load_json(args.registry.resolve()),
            "inline",
            expected_initial_kernel_version=1,
            project_root=root,
        )
    except (guard.GuardError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2
    if result.get("status") != "PASS":
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    # The output is deliberately written only after the exact ELIGIBLE ship
    # certificate and exact trusted child visual evidence have both passed.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
